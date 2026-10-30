# Missing serve recovery

For the calibrated `hsv_config_court1_trial.json` camera, a clear near-side
serving stance that contradicts the point score's service half requests a
replay. A partly visible toss can request the search; service-side parity
never supplies a winner or an extra point.

The tracker retains a semantic checkpoint after the last committed history
row, including a first-serve fault. If no later row or score change has been
committed, it can seek back and rescan up to 2000 frames once per checkpoint.
The retry adds fresh yellow-ball pixels to the near-toss search while keeping
the stance, motion and shape checks. Existing CSV rows are retained; recovered
serves and outcomes pass through the ordinary scoring/history writer.

Logs report `MISSED_SERVE_REPLAY` and `MISSED_SERVE_REPLAY_REJECTED`. Future
image buffers and inference cooldowns are cleared on replay. A failed seek
or semantic restore leaves the live score and capture position intact.

Near-camera tosses retain their measured colour/size through projection over
the net. A falling toss and a short contact-frame turn permit a bounded fresh
departure search. Distant service impacts require two rebound observations,
and faults are evaluated against freshly detected service/court lines.

Tests: `test_missed_serve_replay.py`, `test_near_serve_scan_recovery.py`, and
`test_near_serve_rebound.py` under `tests/`. Native review artifacts are under
`tmp/missed_left_recovered_*` and `tmp/missed_serve_*`. The recovered left
sequence contains a long first fault followed by a second fault outside the
right singles sideline, advancing the seeded score from `2:0 15:0` to
`2:0 15:15`. Full night regression comparison is still a promotion gate.
