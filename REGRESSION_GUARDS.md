# Tracking regression guards

Any tracker logic change must retain every previously visually validated point
before it can be kept.  The current regression target is frames `0-27920` of
`04_Left_Court_tennis_20260623220627_20260623222559_1274169779.mp4`; the
working ledger is `TRACKING_VALIDATION.json`.

Rules for each experiment:

1. Test the smallest affected serve/point window first.
2. Compare both the new target and all earlier validated starts/endpoints.
3. Run the full `0-27920` regression only after the focused check passes.
4. Reject and restore any candidate that changes a protected point or removes
   a known point. Never leave a failed experiment as the runtime default.

## Serve-start experiments

| Date | Change | Focused result | Decision |
| --- | --- | --- | --- |
| 2026-08-03 | Buffered far-toss pre-point tracker | Correctly omitted f7917/f8013 tosses and produced f8044-8053 and f8527-8536, but removed the protected f8619-8722 point. | Rejected and restored. |
| 2026-08-03 | Raw physical-continuation far-toss gate | Rejected f7917 but still falsely created f8012-8026. | Rejected and restored. |
| 2026-08-03 | v4 patch model trained with reviewed f8023-f8025 hard negatives | Suppressed the false f8026 recovery *and* the true f8536 recovery. | Experimental only; not enabled. |
| 2026-08-03 | v5 patch model trained with the same negatives plus reviewed f8533-f8536 true flight patches | Also suppressed both the false and true recovery; generic validation was weak. | Rejected; not enabled. |
| 2026-08-04 | Immediately discard a provisional far serve when local AI rejects its first 55+ px HSV jump | Removed f7917/f7918 as intended, but then accepted another false start at f8013 and failed to recover the known real f8044 serve. | Rejected and restored; focused run `tmp/regression/f7918_discard_caught_toss_20260804`. |
| 2026-08-04 | Reject a distant, zero-motion first provisional recovery | Prevented the exact f7918 `(2065,197)` blob, but selected a player/shoe fragment at f7919 instead of a coherent path. | Rejected and restored; focused run `tmp/regression/f7918_motion_start_gate_20260804`. |
| 2026-08-04 | Activate existing arm-then-confirm far-toss path | Correctly rejected f7917 and f8013, but produced no follow-on serve through f8170. | Rejected and restored; focused run `tmp/regression/f7917_arm_confirm_v1_20260804`. |
| 2026-08-04 | Deferred three-frame static-artifact guard for any local-AI-rejected, 55-600px zero-motion jump | In f7918-f7919, rejected the exact persistent `(2065,197)` zero-motion blob and re-ran f7919 to select the visible `(1960,92)` ball with clean velocity history `[26.0]`. The same guarded path is now available during ordinary rallies; it still requires AI rejection plus next-frame persistence. | Focused pass only; full regression comparison is required before acceptance. |

The next serve-start model must be temporal: classify an entire candidate
launch (ball trajectory plus racket/player motion), not independent patches.
It must win the f8012-vs-f8527 comparison before touching runtime logic.
