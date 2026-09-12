# V5 experimental tracking evidence

V4 recovered the real ball at 4285, but switched again and falsely ended at
4300. A weak player detection (confidence about 0.46) falsely placed the feet
behind the baseline, enabling the walking-player serve at 4375.

V5 requires central-body image motion for night player detections below 0.60
confidence before certifying legal stance. Missing motion evidence holds the
candidate and invalidates the legal-stance cache. Strong detections can still
certify stationary servers. This does not repair the person detector itself.

Night HSV overrides cannot replace a moving, closer predicted candidate with
an off-path fragment below 10 pixels area and below 20% of current area.
This targets candidate switches before point-end logic. It does not guarantee
recovery when all candidates are wrong.

66 focused tests pass. Tests include weak stationary player rejection and
negative cases retaining small on-path balls and recovery from static blobs.
Full source video is unavailable locally. Thresholds require video validation;
low-confidence stationary real players may be held. Do not promote before
focused 4000-4499 validation and the full REGRESSION_GUARDS.md checks.

Audit output directory: tmp/serve_check_v5. Check real serve 4074, bounce 4117,
tracking after 4285, and walking-player false serve 4375.
