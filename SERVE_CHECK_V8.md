# V8 receiver recovery survives same-frame Contact Local AI miss

V7 correctly measured real image motion for the temporally confirmed receiver
ball, but the full main loop has an earlier arbitration layer that V7's test did
not exercise. After normal tracking returns the recovered receiver ball,
`_contact_local_ai_trigger()` can arm Contact Local AI because the ball is beside
the receiver. If Contact Local AI then finds no acceptable candidate, the
fallback restores the pre-track snapshot and replaces the recovered position
with `prev_ball_center`.

V8 leaves that general fallback unchanged and makes the smallest targeted
change: a `receiver_temporal` candidate that has already passed the unique
two-frame temporal match receives same-frame Contact Local AI protection only
when its measured grayscale motion is independently supported
(`motion_mean >= 4` or `motion_max >= 25`). The protection uses the existing
`_contact_local_ai_cooldown_until_frame` mechanism and ends on the current
frame, so Contact Local AI may arm again immediately on the next frame if the
trajectory becomes suspicious. Normal buffered Local AI arbitration remains
enabled.

There are no frame-number or court-coordinate exceptions. Static, ambiguous,
missing-pair, and weak-motion receiver candidates do not receive the protection.

The receiver recovery regression now explicitly checks the previously missed
main-loop layer: after temporal recovery it calls the real
`_contact_local_ai_trigger()` and verifies that it does not re-arm on that same
frame. The existing commit/arbitration assertions remain in place.

Focused video validation is still required before promotion. Run:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v8 --provider local --max-events 12 --endpoint-audit
```

Expected comparison with V7: preserve the real serve at 4074 and serve bounce
at 4117; keep the reviewed false serve around 4375 absent; after each
`RECEIVER TEMPORAL RECOVER`, the following `[TRACK]` should keep the recovered
moving position instead of reverting to the old receiver/player anchor. Recheck
the endpoint around 4300 and the rally count. Follow `REGRESSION_GUARDS.md`
before promotion.
