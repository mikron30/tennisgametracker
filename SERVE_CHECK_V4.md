# V4 experimental pre-score verification

V3 video validation failed: the marker switched to the player at f4285 and
awarded a false OUT. The f4375 false serve also remained.

Before either point-end path accepts a sideline OUT, V4 checks a sudden
motion jump against an image-supported moving candidate. It only continues
when the replacement is substantially closer to the incoming prediction and
has not reversed the incoming vertical motion. Motion state/history are
rebuilt from the previous frame anchor, not the rejected artifact. Static
candidates, normal rebounds and unavailable images retain the existing result.

Quiet audit logs now include all SERVE_STANCE decisions and V3 gate messages.
Serve detection rules are unchanged in this revision; the false start is
still unresolved pending these diagnostics.

60 focused tests pass, including the logged f4285 coordinates with a supplied
recovery candidate (not end-to-end image detection), static/rebound negatives,
and quiet output. Full video remains unavailable locally. Run 4000–4499 into
tmp/serve_check_v4, inspect real serve/bounce 4074/4117, false endpoint 4285,
and false start 4375, then follow REGRESSION_GUARDS.md before promotion.
