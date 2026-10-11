# V12 ignores synthetic terminal-recovery vectors in OUT detection

The V11 audit confirms that the 4307 false OUT is suppressed correctly. The next false endpoint appears at frame 4335.

At frame 4331 the timeout verifier rejects a stale static marker at `(3766,402)` and recovers a separately moving ball at `(2983,733)`. That 850 px correction is intentionally recorded as motion so tracking can continue, but it is an identity repair rather than a physical one-frame ball flight.

By frame 4335 that synthetic `(-783,+331)` vector is still the detector's `prev_motion`. The current local step `(144,-143)` therefore appears to be a 158-degree rebound and incorrectly creates a right-sideline OUT.

V12 does not add a frame or coordinate exception. `out_bounce_verification.py` now recognizes this case only when two independent runtime records agree:

- the point tracking trace explicitly tags the recent recovery as `terminal_motion_recovery`; and
- the motion-history entry from that exact recovery has old/new positions whose vector still matches the current `prev_motion`.

The correction must also exceed the existing large synthetic-motion threshold (`max(400px, 10% of frame width)`). The protection lasts only while that exact synthetic vector remains in `prev_motion`; once a normal local ball step replaces it, ordinary OUT/bounce logic resumes.

Focused regression tests reproduce the observed 4331/4335 geometry and verify that the OUT is suppressed only with matching terminal-recovery provenance. A similarly large vector without that provenance remains eligible for the normal detector.

Run:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v12 --provider local --max-events 12 --endpoint-audit
```

Expected: preserve the real serve around 4074, serve bounce at 4117, receiver recovery around 4133-4147, V9 suppression at 4298, V11 recovery grace at 4307, and suppress the false 4335 endpoint with `[OUT VERIFY TERMINAL RECOVERY GRACE]`. Then inspect the next endpoint before broader regression.
