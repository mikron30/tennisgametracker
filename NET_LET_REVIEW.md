# Disputed game boundary

The final native court-one replay **passed** from 33500 through 42130,
starting at 0:0 with `--far` and `hsv_config_court1_trial.json`. All 17
reference events match on score, winner, category, serve attempt, point
index, next server/attempt, and serving/receiving ends. The first 15
endpoint frames are unchanged. The sole changed endpoint is the disputed
double fault: its actual net landing is now recognized at 41712,
confirmed at 41714, rather than waiting until 41751.

| Serve starts | Verified result | Score after event |
|---|---|---|
| 41244 | First fault | 1:0 30:30 |
| 41698 | Double fault, P1 wins | 1:0 40:30 |
| 42020 | Net error, P1 wins; game complete | 2:0 0:0 |

The final point ends at 42100; its confirmation at 42101 selects game
three's near serving end. The run source/config hashes remained unchanged.
Evidence: `tmp/net_drop_full_results.json`,
`tmp/net_drop_full_verified_20261026_190829.csv`, and
`tmp/net_drop_full_verified.log`. Final unit suite: **764 passed, 52
subtests passed**. The separate full night gate must be reported on its
own baseline comparison; this clip validation is not a claim about all
possible recordings.

The compact regression fixture is `regression_court1_first_two_games.csv`.
It contains the independently checked reference outcomes with the corrected
physical net-drop endpoint. To check another bounded replay:

```powershell
py.exe validate_court1_history.py path\to\bounded_point_history.csv
```

The validator compares the CSV after tracking; its fixture is not used by
the live scorer.

## Initial diagnosis and guard-only experiment

The latest `point_history_20261026_153343.csv` and the supplied earlier
history first diverge in scoring at the serve starting at frame 41698.
The earlier history records a double fault ending at 41751 (40:30).
The latest records a let at 41743 (30:30). Consequently, the point ending
at 42100 reaches 40:30 instead of ending the game at 2:0.

The latest row's score arithmetic is internally consistent. Its preceding
let classification is the suspect event; changing the winner or forcing a
game based only on the later server would conceal that error.

The alleged legal landing at (1864,499) is above the calibrated net's
local bottom edge (about y=529). The candidate sequence contains 1–2 pixel
contours with weak motion. Such an observation cannot establish a legal
receiving-side ground bounce.

The calibrated camera now rejects these projected-net service landings
before committing a bounce or let. It leaves the landing unresolved,
without assigning a fault or winner. This check applies only to downward
far serves, only within the calibrated net's horizontal coverage, and
uses its sloping bottom edge rather than a global height. Near serves and
night configurations retain their existing behavior.

The initial native replay revealed a second alleged landing at (1905,587)
one frame later. Terminal repair then replaced it with (3241,1864) while
retaining the let label. The scoring writer now marks calibrated lets as
`tracking_uncertain` when terminal repair relocates their endpoint by 70px
or more. It preserves the incoming serve attempt and awards no point;
neither a fault nor a legal let is established by that repair.

Targeted tests cover the recorded coordinate, sloping geometry, a genuine
receiving-side landing, other serving directions, and rejection without
mutating the recorded bounce or awarding a point. Video replay remains
necessary to establish the actual outcome of the disputed serve.

Earlier guard-only unit suite: 759 passed, 52 subtests passed. The initial native replay
predates the terminal-repair safeguard and is diagnostic, not final
acceptance evidence. A short replay starting at 41670 did not acquire the
serve, so it could not validate its outcome. That experiment did not
establish a corrected history; the final continuous replay above does.

## Root-cause correction

The actual far serve contacts the net, then drops almost vertically at
x=2143–2148. At frame 41708 the original tracker jumps from (2144,487)
to an unrelated highlight at (2245,367), rather than following the ball
at (2143,503). A narrow local net-drop search already existed, but the
early recovery dispatch did not call it while awaiting a service landing
inside the net projection. Confirmed slow net drops now use that dispatch
before generic HSV selection.

The raw-video ball subsequently moves through y=519,526,533,531,528.
Its first upward step is only two pixels. The net-drop rebound validator
now accepts two upward steps of at least two pixels, provided their total
rise is at least five pixels, and retains the existing consecutive-frame,
bounded-motion, prior-net-contact, and local bottom-edge checks. This
establishes a net fault from measured observations; no score or winner is
inferred from the expected history.

Unit suite for this correction: 764 passed, 52 subtests passed. The native
full replay and night gate have separate result artifacts under `tmp/`;
their completion and scoring comparisons must be checked before claiming
acceptance. Expected endpoints may occur earlier when the physical net
fault is recognized earlier; reference serve starts, attempts, categories,
winners and scores remain the acceptance criteria.

The native focused replay from 41000 through 42130 passed. It records the
first fault at 41251, the second-serve net landing at 41712 (confirmed at
41714) as a double fault to P1, and the next point ending at 42099
(confirmed at 42100) as a P1 net-error win. Scores are 30:30, 40:30, then
2:0 with game three's serving end switched to near. The replay starts
with explicit score 1:0 30:30 and observes both serve attempts; it does
not inject a second-serve state or a winner. Assertions and output are in
`tmp/check_net_drop_focus.py` and `tmp/net_drop_focus_results.json`.

## Night compatibility check and remaining limitation

The complete night run reached 27920 with unchanged source/config hashes.
All seven reviewed-ledger assessments were unchanged: one preserved guard,
six pre-existing mismatches, and **zero new reviewed-guard regressions**.
Its first 19 event outcomes matched the older full-night run, but a late
unreviewed event diverged. The old run had loaded 2868 persistent false
points; this run loaded 2887, so that full-history comparison did not use
identical learned inputs and remains `requires_review`.

A paired control replayed 24210–25210 with explicit initial score
0:0 AD:40, previous endpoint 24209, and the same frozen 2887-entry profile.
The before-fix and after-fix versions produced identical outcomes across
15 compared fields and identical 50-sample accepted tracking traces.
Both acquired the serves starting at 24556 and 24766. These controls use
fresh partial-replay state and do not certify the full night's outcomes.
The court-one recovery change is gated off for the night configuration.

Evidence: `tmp/net_drop_night_gate_result.json`,
`tmp/net_drop_night_guards.json`, and
`tmp/net_drop_night_controls_result.json`. The night run's newly learned
false-point entries were restored to the exact pre-test profile bytes.
Existing night-video tracking/review issues remain outside this verified
court-one sequence; universal tracking accuracy is not established.
