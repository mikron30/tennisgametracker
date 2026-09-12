# V10 preserves the verified night recovery through the outer jump guard

The V9 audit confirms that frame 4298 no longer ends the point. That exposed the
next failure cleanly: after the tracker drifts onto tiny right-side fragments, it
reaches `(3812,1636)` by frame 4305. At frame 4306 the night visible-ball recovery
finds a strongly moving candidate at `(2979,593)` (`area=37.5`, motion
`23.7/142.0`), but the outer main-loop jump guard rejects the 1335 px correction
with its ordinary 400 px limit and restores the poisoned anchor.

The jump guard already has a dedicated 1500 px allowance for a same-frame motion
reacquisition, but it deliberately requires both markers to match: the recovery
frame and the exact recovered position. Other motion-reacquisition paths set both
`_last_motion_reacq_frame` and `_last_motion_reacq_pos`. The night visible-ball
commit path set only the frame, so its own valid recovery could never satisfy
that existing guard.

V10 fixes that state invariant only. `_commit_night_visible_ball_recovery()` now
pairs `_last_motion_reacq_frame` with `_last_motion_reacq_pos = tuple(new_pos)`.
There is no new frame number, coordinate exception, or wider global jump limit.
The existing outer guard still requires the current tracked position to be the
same verified recovery before granting its pre-existing 1500 px allowance.

The focused regression test now asserts that a committed visible-ball recovery
carries both identity markers and demonstrates a synthetic poisoned-anchor jump
that is larger than the normal 400 px limit but below the existing 1500 px
motion-reacquisition limit.

Run the focused video audit:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v10 --provider local --max-events 12 --endpoint-audit
```

Expected comparison with V9: preserve the real serve around 4074 and service
bounce at 4117; keep the false serve around 4375 absent; keep the receiver
recovery sequence around 4133-4147; keep the V9 suppression at 4298; and at 4306
accept the verified `(2979,593)` night recovery instead of logging
`[JUMP_REJECTED] ... limit=400px`. Inspect the next endpoint after that recovery
before making any broader tracking change.
