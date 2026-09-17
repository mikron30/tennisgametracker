# V14 confirms a suppressed night OUT from delayed rebound evidence

The V13 focused audit exposed a real OUT that the night static-artifact guard was suppressing.

At frame 4284 the tracked ball is outside the right singles sideline at about `(2922,729)` after a coherent descending sequence. The current contour has weak local motion, so `deep_static_out` suppresses it as a possible night hotspot. Frames 4285 and 4286 intentionally hold the same point while opposite-direction candidates are waiting for confirmation. At frame 4287 the tracker accepts `(2894,590)` with a strong upward step `(-28,-139)` and about a 178.9 degree reversal. That later accepted rebound is strong evidence that the original frame-4284 outside point was a real bounce, not a static artifact.

V14 keeps only this narrow `deep_static_out` case as `_pending_night_static_out` when it is backed by coherent descending flight. The existing two-frame static-artifact wait is preserved. Starting at age 3, a pending OUT can be confirmed only by a bounded upward rebound with a large direction reversal and a reasonable displacement from the original candidate. Implausibly large reacquisitions do not confirm it, and an unconfirmed pending candidate expires after four frames.

When confirmation succeeds, the endpoint is reported from the original pending bounce position and side rather than from the later rebound position. There are no frame-number or coordinate exceptions in production logic.

Focused regression tests cover:

- a descending deep-static candidate arming pending OUT state;
- the third-frame coherent rebound confirming the original OUT position;
- an implausibly large reacquisition failing to confirm the pending OUT;
- all previously protected serve, recovery, and OUT-regression tests.

Run the focused video audit:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v14 --provider local --max-events 12 --endpoint-audit
```

Expected V14 sequence around the target:

- frame 4284: `[OUT-BOUNCE PENDING]` followed by the existing static-artifact suppression;
- frames 4285-4286: existing `[OUT-BOUNCE WAIT]` behavior;
- frame 4287: `[OUT-BOUNCE PENDING CONFIRMED]` with `source_f=4284`, original point near `(2922,729)`, a rebound near `(2894,590)`, and a turn near 179 degrees;
- the point should end as a right-sideline OUT using the original bounce location instead of drifting to the later STUCK_TIMEOUT.

Also verify all protected behavior from V13 remains unchanged before running a broader regression.
