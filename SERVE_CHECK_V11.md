# V11 prevents a recovery jump from masquerading as the next OUT bounce

The V10 audit confirms the intended fix at frame 4306: the visible-ball recovery at `(2979,593)` is now accepted and survives the outer jump guard. The previous `[JUMP_REJECTED] ... limit=400px` is gone.

That exposes the next terminal error one frame later. At 4307 the sideline detector reports an OUT using `prev_motion=(-833,-1043)` and `curr_motion=(-73,167)`. The 1334.8 px `prev_motion` is not a physical incoming-ball vector: it is the correction from the poisoned tracker anchor `(3812,1636)` to the verified recovery `(2979,593)`. Treating that re-anchor vector as incoming flight creates an artificial 117.8 degree turn and a false bounce.

V11 adds a narrow guard in `out_bounce_verification.recover_continuing_ball()`. An OUT is suppressed for the first frame after a verified motion reacquisition only when all of these are true:

- `_last_motion_reacq_frame` is exactly the previous frame;
- `_last_motion_reacq_pos` matches the previous-frame anchor reconstructed from the current motion; and
- the incoming vector exceeds a conservative synthetic-recovery threshold `max(400, frame_width * 0.10)`.

This does not suppress ordinary rebounds, ordinary reacquisitions, or later frames. It does not use a frame-number or coordinate exception. It only prevents a large re-anchor correction from being interpreted as tennis-ball flight geometry.

Focused tests cover the observed recovery-grace case and verify that a stale/mismatched recovery marker does not suppress a normal OUT proposal.

Run the focused video audit:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v11 --provider local --max-events 12 --endpoint-audit
```

Expected comparison with V10: preserve the serve around 4074, service bounce around 4117, receiver recovery around 4133-4147, V9 suppression at 4298, and accepted recovery at 4306. At 4307 expect `[OUT VERIFY RECOVERY GRACE]` instead of `POINT ENDED`. Inspect the next endpoint after 4307 before any broader tracker change.
