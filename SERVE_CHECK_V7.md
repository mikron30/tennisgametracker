# V7 receiver recovery motion evidence

V6 found the actual ball beside the receiver but committed zero motion
metadata. Later arbitration restored the old marker. V7 measures the mean
and maximum grayscale difference in a radius-8 patch, using the actual image
pair associated with temporal confirmation. It supplies these measured values
to the existing recovery commit. No arbitration guard is bypassed.

71 focused tests pass. The new integration test sends a synthetic moving
candidate through detection, real recovery commit and real weak-player
arbitration (unrelated AI callbacks stubbed). It checks the selected position
survives and evidence equals the image difference. On supplied step-two JPGs,
mean/max motion at 4134 is 31.8/182, at 4136 29.8/210, at 4138 39.8/176,
at 4142 20.8/112 and at 4144 18.7/99. These are helper-only checks.

Full video remains unavailable locally. Run frames 4000-4499 into
 tmp/serve_check_v7. Compare RECEIVER TEMPORAL RECOVER with the following
TRACK position, preserve serve 4074/bounce 4117 and absent false serve 4375.
The 4300 endpoint and rally count are still unverified. Follow
REGRESSION_GUARDS.md before promotion.
