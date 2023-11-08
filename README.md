# getfscaler

This is an unofficial "companion" script to getfnative <https://github.com/YomikoR/GetFnative>.

Note that the results of this are **NOT** conclusive on their own!
When using this script, be extra mindful that results may be inaccurate,
and that you should always double-check yourself!

TRUST YOUR EYES OVER THIS SCRIPT!

# Example usage

Regular usage, using an image file:

```terminal
python getfscaler "image.png" -nh 810
```

Fractional native resolution, using a video file:

```terminal
python getfscaler "input.mkv" -nh 719.8 -bh 720
```

Fractional native resolution, using a Vapoursynth Python file:

```terminal
python getfscaler "input.vpy" -nh 719.8 -bh 720
```

Cross-conversion, using a vpy file:

```terminal
python getfscaler "input.vpy" -nh 720 -fb 2
```

HDCAM master, but you're stubborn and want to try to descale it vertically anyway:

```terminal
python getfscaler "input.png" -nh 720 -nw 1920
```

## Changes

This is a rewrite of the original getscaler <https://gist.github.com/cN3rd/51077b6abf45b684bf9a3c657d859b43>
and features the following changes:

-   Use more robust IEW tooling
-   Only use kernels used in professional software + Point by default
-   Add additional post-filtering methods to reduce error caused by dithering and dirty edges
-   Add fractional support (see: getfnative)
-   Add support for descaling cross-converted video
-   Support many more types of images and don't rely on ffms2 (known to cause issues)
-   Print additional information and warnings
-   (Optional) One-dimensional scaling (horizontal/vertical only, or both)
-   (Optional) Set output nodes for every single image (UNTESTED!)
-   (Optional) Check a bunch of additional kernels (NOT RECOMMENDED!)
-   (Optional) More verbose output to give the user a better idea of what's going on internally (--debug)

Note that the errors may appear higher on average than with the original getscaler.

This script will warn you if the error is likely too high to be reliable, but again, use your eyes.
