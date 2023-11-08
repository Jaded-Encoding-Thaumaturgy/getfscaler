import argparse
import logging
import operator
import runpy
import time
from math import ceil
from random import randint
from typing import Any, cast

from rich.logging import RichHandler
from vskernels import (AdobeBicubic, Bicubic, BicubicSharp, Bilinear, BSpline,
                       Catrom, Descaler, FFmpegBicubic, Hermite, Kernel,
                       KernelT, Lanczos, Mitchell, Point, Robidoux,
                       RobidouxSharp, RobidouxSoft, Spline16, Spline36,
                       Spline64)
from vsmasktools import Sobel, replace_squaremask
from vsscale import fdescale_args
from vssource import source
from vstools import (CustomIndexError, FieldBased, FieldBasedT,
                     FileWasNotFoundError, SPath, core, get_prop, get_w, plane,
                     set_output, vs)

# Logging stolen from vsmuxtools
FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True, omit_repeated_times=False, show_path=False)]
)

logger = logging.getLogger("getfscaler")


def _format_msg(msg: str, caller: Any) -> str:
    if caller and not isinstance(caller, str):
        caller = caller.__class__.__qualname__ if hasattr(caller, "__class__") and \
            caller.__class__.__name__ not in ["function", "method"] else caller

        caller = caller.__name__ if not isinstance(caller, str) else caller

    return msg if caller is None else f"[bold]{caller}:[/] {msg}"


def debug(msg: str, caller: Any = None) -> None:
    if not logger.level == logging.DEBUG:
        return

    message = _format_msg(msg, caller)
    logger.debug(message)


def warn(msg: str, caller: Any = None, sleep: int = 0) -> None:
    message = _format_msg(msg, caller)
    logger.warning(message)

    if sleep:
        time.sleep(sleep)


caller_name = SPath(__file__).stem


def get_kernel_name(kernel: KernelT) -> tuple[str, str]:
    kernel = Kernel.ensure_obj(kernel)

    kernel_name = kernel.__class__.__name__
    extended_name = kernel_name

    if isinstance(kernel, Bicubic):
        extended_name += f" (Bicubic b={kernel.b:.2f}, c={kernel.c:.2f})"
    elif isinstance(kernel, Lanczos):
        extended_name += f" (taps={kernel.taps})"

    debug(f"Checking error for {kernel.__class__.__name__} ({extended_name})...", get_kernel_name)

    return kernel_name, extended_name


def get_error(
    clip: vs.VideoNode,
    width: float = 1280.0, height: float = 720.0,
    line_mask: vs.VideoNode | None = None, crop: int = 8,
    kernel: KernelT | None = None,
) -> dict[str, float]:
    """Get the descale error."""
    debug(str(kernel), get_error)

    if not issubclass(kernel if isinstance(kernel, type) else type(kernel), Descaler):
        if args.debug:
            warn(f"Kernel \"{kernel}\" is not a subclass of Descaler! Skipping...", get_error)

        return {}

    kernel = Kernel.ensure_obj(kernel)
    kernel_name, kernel_class = get_kernel_name(kernel)
    kernel_out = kernel_name if args.swap else kernel_class

    ceil_bh = ceil(height) & ~1
    ceil_bw = ceil(width) & ~1

    de_args, up_args = fdescale_args(clip, height, ceil_bh, ceil_bw, up_rate=1.0, src_width=width)

    debug(f"Descaling using the following parameters: {de_args}", get_error)
    debug(f"Upscaling using the following parameters: {up_args}", get_error)

    if not args.fields:
        descaled = kernel.scale(kernel.descale(clip, **de_args), clip.width, clip.height, **up_args)

        if args.out:
            set_output(descaled, name=f"{kernel_name} (rescaled)")

        descaled = post_descale(clip, descaled, line_mask, crop)

        err = get_prop(descaled.std.PlaneStats(clip), "PlaneStatsDiff", float)

        return {kernel_out: err}

    descaled_reg, shifts_reg = descale_fields(clip, de_args.get("height", 720), kernel, args.fields, False)
    descaled_reg = post_descale(clip, descaled_reg, line_mask, crop)
    err_reg = get_prop(descaled_reg.std.PlaneStats(clip), "PlaneStatsDiff", float)

    debug(f"Error for {kernel_class} [{kernel_name}, {shifts_reg[-1]}]: {err_reg:.13f}", get_error)

    descaled_neg, shifts_neg = descale_fields(clip, de_args.get("height", 720), kernel, args.fields, True)
    descaled_neg = post_descale(clip, descaled_neg, line_mask, crop)
    err_neg = get_prop(descaled_neg.std.PlaneStats(clip), "PlaneStatsDiff", float)

    debug(f"Error for {kernel_class} [{kernel_name}, {shifts_neg[-1]}]: {err_neg:.13f}", get_error)

    return {
        f"{kernel_out} [{shifts_reg[0]:.3f}, {shifts_reg[1]:.3f}]": err_reg,
        f"{kernel_out} [{shifts_neg[0]:.3f}, {shifts_neg[1]:.3f}]": err_neg,
    }


def post_descale(
    og_clip: vs.VideoNode, descaled_clip: vs.VideoNode,
    line_mask: vs.VideoNode | None = None, crop: int = 8
) -> vs.VideoNode:
    # Reduce error by applying the descale to only the lineart and removing edges from the equation.
    if line_mask:
        descaled_clip = og_clip.std.MaskedMerge(descaled_clip, line_mask)

    if crop:
        descaled_clip = replace_squaremask(
            descaled_clip, og_clip, (og_clip.width - crop * 2, og_clip.height - crop * 2, crop, crop), invert=True
        )

    return descaled_clip


def descale_fields(
    clip: vs.VideoNode,
    height: int,
    kernel: Kernel,
    tff: FieldBasedT = FieldBased.TFF,
    shift_negative: bool = False
) -> tuple[vs.VideoNode, tuple[float, float]]:
    """
    Descale the frame per-field.
    This is used to descale cross conversions.
    """
    neg = -1 if shift_negative else 1
    target_shift = (height / clip.height) * 0.25

    clip_y = plane(clip, 0)

    to_descale = FieldBased.ensure_presence(clip_y, tff)
    to_descale = to_descale.std.SeparateFields()

    descaled_1 = kernel.descale(
        to_descale[0::2], get_w(height, clip),
        height // 2, (target_shift, 0.0)
    )
    descaled_2 = kernel.descale(
        to_descale[1::2], get_w(height, clip),
        height // 2, (target_shift * neg, 0.0)
    )

    upscaled = FieldBased.PROGRESSIVE.apply(core.std.Interleave([
        kernel.scale(descaled_1, clip.width, to_descale.height, (target_shift, 0)),
        kernel.scale(descaled_2, clip.width, to_descale.height, (target_shift * neg, 0))
    ]).std.DoubleWeave(tff=False))[::2]

    if args.out:
        set_output(upscaled, name=f"{kernel.__class__.__name__} [{target_shift, target_shift * neg}] (rescaled)")

    return upscaled, (target_shift, target_shift * neg)


def get_kernels() -> list[KernelT]:
    kernels: list[KernelT] = [
        # Bicubic-based
        Hermite,  # Bicubic b=0.0, c=0.0
        Catrom,  # Bicubic b=0.0, c=0.5
        Mitchell,  # Bicubic b=0.333, c=0.333
        BicubicSharp,  # Bicubic b=0.0, c=1.0

        # Bicubic-based but from specific applications
        FFmpegBicubic,  # Bicubic b=0.0, c=0.6. FFmpeg's swscale
        AdobeBicubic,  # Bicubic b=0.0, c=0.75. Adobe's "Bicubic" interpolation

        # Bilinear-based
        Bilinear,

        # Lanczos-based
        Lanczos(taps=3),
        Lanczos(taps=4),

        # Point-based
        Point,
    ]

    if not args.extensive:
        return kernels

    warn(
        "Extensive kernel checking enabled. Note that the original set of kernels were chosen because they are used "
        "in professional software, and as such are astronomically more likely to be used in real productions. Please "
        "be extra careful when using non-default kernels!"
    )

    kernels += [
        # Bicubic-based
        BSpline,
        Robidoux,
        RobidouxSoft,
        RobidouxSharp,

        # Lanczos-based
        Lanczos(taps=2),
        Lanczos(taps=5),

        # Spline-based
        Spline16,
        Spline36,
        Spline64,
    ]

    return kernels


def print_results(clip: vs.VideoNode, errors: dict[str, float], framenum: int = 0) -> None:
    errors_sorted: list[tuple[str, float]] = sorted(errors.items(), key=operator.itemgetter(1))

    if not errors_sorted:
        warn("Could not get any values!", print_results)
        return

    best = errors_sorted[0]

    clip = FieldBased.ensure_presence(clip, args.fields)

    height = f"{args.native_height:.3f}" if not float(args.native_height).is_integer() else int(args.native_height)
    width = f"{args.native_width:.3f}" if not float(args.native_width).is_integer() else int(args.native_width)

    header = f"\nResults for frame {framenum} (resolution: {width}/{height}, " \
        f"AR: {args.native_width / args.native_height:.3f}, " \
        f"field-based: {FieldBased.from_video(clip).pretty_string}):"

    print(header)
    print("-" * max(80, len(header)))
    print(f'{"Scaler":<44}\t{"Error%":>7}\t{"Abs. Error":>18}')

    for name, abserr in errors_sorted:
        relerr = abserr / best[1] if best[1] != 0 else 0
        print(f"{name:<44}\t{relerr:>8.1%}\t{abserr:.13f}")

    print("-" * max(80, len(header)))
    print(f"Smallest error achieved by \"{best[0]}\" ({best[1]:.10f})\n")

    if best[1] > 0.008:
        warn(
            "The error rates for this frame are on the low end of acceptable errors. "
            "This can be happen if you have the wrong native resolution or there are FHD elements in the image. "
            "Be extra careful when trying to descale this frame using these results!"
        )

    if any(x in best[0].lower() for x in ("mitchell", "0.33")):
        warn(
            "Note that Mitchell is a common false-positive. "
            "Carefully compare your descaling results with Catrom or Lanczos!"
        )

    if "spline" in best[0].lower():
        warn(
            "Note that Spline is an EXTREMELY uncommon custom kernel. "
            "Carefully compare your descaling results with Catrom or Lanczos!"
        )

    warn("getscaler is not perfect! Please don't blindly trust these results "
         "and carefully verify them for yourself!")


def get_vnode_from_script(script: SPath) -> vs.VideoNode:
    """Get a videonode from a python script. This will always be output 0."""
    runpy.run_path(script.to_str(), {}, '__vapoursynth__')

    out_vnode = vs.get_output(0)

    if not isinstance(out_vnode, vs.VideoNode):
        try:
            out_vnode = out_vnode[0]  # type:ignore[assignment]
        except IndexError:
            raise CustomIndexError("Cannot find an output node! Please set one in the script!", get_vnode_from_script)
        except Exception:
            raise

    return cast(vs.VideoNode, out_vnode)


def main() -> None:
    if not (p := SPath(args.input_file)).exists():
        raise FileWasNotFoundError(f"Could not find the file, \"{p}\"!", main)

    if p.suffix in (".py", ".vpy"):
        clip = get_vnode_from_script(p)
    else:
        clip = source(p)

    if args.native_height == -1 and args.native_width == -1:
        warn("You cannot set both \"--native-height\" and \"--native-width\" to \"-1\"!", main)

        return

    if args.native_height == -1:
        args.native_height = clip.height

    if args.native_width == -1:
        args.native_width = clip.width

    if args.fields and not float(args.native_height).is_integer():
        warn("Float values are not currently supported when scaling per-field! Rounding...")

        if args.native_width is not None:
            args.native_width = int(args.native_width)

        args.native_height = int(args.native_height)

    if args.native_width is None:
        args.native_width = args.native_height * clip.width / clip.height

    framenum = args.frame or randint(0, clip.num_frames - 1)

    try:
        frame = clip[framenum]
    except IndexError:
        raise CustomIndexError(
            f"Custom frame number ({framenum}) exceeds number of frames in the clip ({clip.num_frames})!", main
        )

    if args.frame is None:
        debug(f"No frame number given. Grabbing random frame ({framenum}/{clip.num_frames-1})...")

    frame_y = plane(frame, 0)
    mask = Sobel.edgemask(frame_y)

    if args.out:
        set_output(frame_y, name="original frame (luma)")

    kernels = get_kernels()
    errors: dict[str, float] = dict()

    for kernel in kernels:
        err = get_error(frame_y, args.native_width, args.native_height, mask, args.crop, kernel)

        if err:
            errors |= err

    print_results(clip, errors, framenum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best inverse scaler for a given frame")

    parser.add_argument(
        dest="input_file",
        type=str,
        help="Absolute or relative path to the input file (video/script/image)",
    )
    parser.add_argument(
        "--native-height",
        "-nh",
        dest="native_height",
        type=float,
        default=720.0,
        help="Approximated native height. Passing \"-1\" will use the input's height. "
             "Default is 720.0. Accepts both int and float",
    )
    parser.add_argument(
        "--native_width",
        "-nw",
        dest="native_width",
        type=float,
        default=None,
        help="Approximated native width. Passing \"-1\" will use the input's width. "
             "Default is None (auto-calculate from input and height)",
    )
    parser.add_argument(
        "--crop",
        "-c",
        dest="crop",
        type=int,
        default=8,
        help="Number of pixels to crop the edges of the result by, reducing error caused by dirty edges",
    )
    parser.add_argument(
        "--frame",
        "-f",
        dest="frame",
        type=int,
        default=None,
        help="Specify a frame for the analysis. Random if unspecified",
    )
    parser.add_argument(
        "--field-based",
        "-fb",
        dest="fields",
        type=int,
        default=0,
        help="How to treat the field properties of the frame. "
             "0 = Progressive, 1 = Bottom-Field-First, 2 = Top-Field-First. "
             "The shifts that were applied will be added to the Scaler in square brackets. "
             "Defaults to 0 (Progressive)"
    )
    parser.add_argument(
        "--swap",
        "-s",
        default=False,
        action="store_true",
        help="Swap the kernel names to use more descriptive names, i.e. Catrom => Bicubic (b=0.00, c=0.50)",
    )
    parser.add_argument(
        "--extensive",
        "-e",
        action="store_true",
        help="Perform a more extensive check using headcrafted kernels and parameters",
    )
    parser.add_argument(
        "--out",
        "-o",
        action="store_true",
        help="Set an output node for the clips",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debugger logging",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    debug("Debug logging enabled")

    main()
