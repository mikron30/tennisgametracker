# V6 experimental far receiver recovery

V5 preserved serve 4074/4117 and removed the false 4375 serve in the focused
video, but still lost the ball near 4131 and falsely ended at 4300.

V6 probes a small region near the last ball only during night player
reacquisition, in the upper 30% of the image, after the serve bounce phase.
Color/shape candidates must form exactly one moving pair across adjacent
samples (one or two frames). Static, ambiguous and stale pairs are rejected.
This permits small bright balls beside a receiver without categorically
rejecting everything inside an inaccurate player box. Existing recovery
commit/arbiter logic is retained. No hit or score is assigned by this helper.

70 focused tests pass. A helper-only replay on supplied original JPGs
(step two, source frame numbers) recovered the visible ball at 4134, 4136,
4138, 4142, 4144; 4132 is initialization and 4140 remains unconfirmed.
This is not an end-to-end replay of the tracker. No claim that the 4300 false
end or rally counts are fixed. Moving racket highlights remain a risk.

Run the 4000-4499 audit into tmp/serve_check_v6. Verify serve 4074/4117,
receiver recovery logs and actual trajectory, endpoint and false-start
absence. Full REGRESSION_GUARDS.md validation is required before promotion.
