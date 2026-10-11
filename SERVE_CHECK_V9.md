# V9 suppresses tiny impossible night OUT jumps

The V8 audit confirms that the receiver-side temporal recovery now survives the
same-frame Contact Local AI arbitration. The recovered positions at 4133-4147
are retained by the following `[TRACK]` lines, so that layer is no longer the
cause of the endpoint error.

The remaining bad endpoint is frame 4298. The tracker has already drifted onto
tiny night-scene fragments near the right side. At 4298 the accepted contour is
only 2 px, yet it moves 190.1 px in one frame from `(2944,542)` to `(2972,730)`.
That artificial jump creates a 118.7 degree turn and the generic sideline logic
reports an OUT. For the 3840-pixel video, the existing physical sideline-step
bound is 134.4 px (`max(125, width * 0.035)`), so the 190.1 px step is already
outside the detector's own physical range.

V9 changes only the night terminal verifier in `out_bounce_verification.py`.
After the ordinary detector has proposed an OUT, a current contour is not
allowed to establish that OUT by itself when both conditions hold:

- its tracked size is at most 3 px; and
- its one-frame displacement exceeds `max(125, frame_width * 0.035)`.

The verifier records the existing `_last_out_bounce_suppressed_frame` state, so
the normal two-frame `[OUT-BOUNCE WAIT]` protection remains in effect. It does
not change the tracker position, manufacture a replacement ball, alter serve
logic, or use a frame-number/coordinate exception. Larger candidates and normal
rebound speeds keep the previous behaviour.

Focused tests cover both sides of the gate: the observed 2 px / 190 px pattern
is suppressed, while an 8 px candidate with the same motion is not automatically
accepted as continuing play. Existing moving-ball recovery tests remain intact.

Run the focused video audit:

```powershell
python tracking_audit.py --court night --start-frame 4000 --max-frames 500 --output-dir tmp\serve_check_v9 --provider local --max-events 12 --endpoint-audit
```

Expected comparison with V8: preserve the real serve at 4074 and serve bounce at
4117; keep the false serve around 4375 absent; keep the receiver recovery sequence
at 4133-4147; and remove the false `POINT ENDED - Ball bounced out of court` at
4298. The next endpoint after 4298 must then be inspected before making any
broader tracking change. Follow `REGRESSION_GUARDS.md` before promotion.
