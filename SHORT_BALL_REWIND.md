# Automatic short ball rewind

Enabled by default when Local AI is enabled. Use `--no-short-rewind` for an
otherwise identical comparison run. It adds no new model or dependency.

The normal tracker remains the fast path. Recent independently trusted ball
positions, motion and semantic tracking checkpoints are retained within the
current point. A ball miss, unsupported jump, player lock or contradictory
consecutive hits can request recovery. Intentional off-screen waits and the
existing verified contact recoveries are respected.

Recovery normally uses a 12-frame window and can expand to 24 frames, with
at most two attempts per failure episode. It reads the original video through
a separate decoder; the live capture position and GUI do not seek backwards.
It does not keep an additional full-point video in RAM.

The existing model scores fresh motion candidates beyond the configured HSV
ranges. Independently trusted history before the suspected mistake is reused
on the first attempt; the expanded attempt can reconsider that history.
A bounded beam search requires coherent motion, appearance support, clearance
from the player, and fresh consecutive observations at the end. Missing images
remain unobserved: up to two normally, with additional bounded gaps during
measured player occlusion. Positions are never interpolated into ball evidence.

An accepted path restores the checkpoint and replays observations through the
existing ball, bounce and contact committers. It replaces the affected trace
and repairs the original short AI buffer too. Replay cannot commit a point or
CSV row. A failed replay restores counters, player events/profile, tracking
state and buffers. Learning files are not written during a replay transaction.
If the ball position is recovered but a required contact remains unverified,
the corrected flight can continue while its event history and score remain
provisional; the older checkpoints are retained for the wider retry.
Manual HSV saves/reloads invalidate older checkpoints.

For the calibrated daytime near serve, fresh, coherent observations of a
large unstruck toss can also establish checkpoints before any bounce or
racket contact. A sudden tiny fragment above racket reach requests recovery
even if the ordinary detector previously accepted it. This condition uses
serve phase, measured motion and relative contour size, without frame targets.
If the broad AI motion mask misses the toss against net texture, replay may
instead use the configured HSV detector over at least four consecutive real
images. Every observation must retain a large moving contour in the narrow
toss lane, with plausible vertical motion and no player overlap. These are
reported as HSV observations, never as AI confidence. The same transaction
and scoring restrictions apply. Actual far-side small-ball net drops retain
their existing detector; the near toss cannot enter that mode.

Rewind uses a separate allowance of at most two extra batched inferences per
25 video frames, at least eight frames apart. A rejected replay therefore does
not consume the fast tracker's inference slots and change its normal decisions.
Checkpoint overhead and attempts are printed in `SHORT_REWIND_SUMMARY`.

The logs report `SHORT_REWIND_START`, `SHORT_REWIND_ACCEPTED` and
`SHORT_REWIND_REJECTED`; recovery transitions are also included in the existing
`tracking_losses` column of point history. If contradictory hit ownership
remains unresolved, scoring reports `tracking_uncertain` and does not award a
guessed point. Unresolved timeout/loss decisions after exhausted recovery are
also left unawarded. Such a run is not a verified complete match score.

The model can still reject a real ball or lack sufficient evidence during an
occlusion. Rewind is automatic recovery and explicit uncertainty handling;
it does not guarantee that every rally can be recovered by the current model.

Tests are in `tests/test_short_ball_rewind.py`. They cover path validation,
occlusion, transactional rollback, duplicate-event prevention, uncertainty
scoring, and isolation from the normal inference budget. Native validation
artifacts and the real-model controlled recovery check are kept under `tmp/`.

For court1, replay 33500–43200 with `hsv_config_court1_trial.json`, initial
score `0:0 0:0` and initial server end `--far`. Check all 20 reviewed events:

```powershell
py.exe validate_court1_history.py <history.csv> --reference regression_court1_through_double_fault.csv
```

The fixture preserves the verified first 17 events and checks the near-side
game-three net point, first fault and second-serve double fault, ending at
`2:0 15:15` with the far receiver winning. The validator's default remains
the original 17-event first-two-game fixture for shorter regressions.
