# Fast parallel point regression

Use reviewed point history to replay points concurrently, skipping the gaps
between them. First faults and their second serves stay in the same case so
the tracker itself must preserve the score and advance the serve attempt.
Only the score **before** a case, initial match court end, initial attempt and
history index are seeded. Expected endpoints, winners and positions are never
passed to the detector.

```powershell
py.exe parallel_point_regression.py regression_court1_through_double_fault.csv --court court1 --config hsv_config_court1_trial.json --initial-server-end far --baseline-start-frame 33500 --workers 2
```

The default 240-frame warm-up and 16-frame confirmation padding produce 13
cases covering all 20 reviewed events, with 6,441 total replay frames without
an initialization anchor (6,437 with the known court1 boundary above). Start
with two workers on this machine: decoding, player detection and AI inference
share CPU/GPU resources. More workers may reduce throughput.

To test only the missed double-fault point, use the same full reference and
add `--cases 13`. Its pre-point score is automatically `2:0 15:0`. Case 13
includes both the first fault and the second-serve double fault. Add `--dry-run`
to inspect all frame windows and initial scores without running the tracker.

For other histories, provide the corresponding court/config and the match's
**initial** server end. If the first reference row is already partway through
the match, supply its pre-row score with `--initial-score "2:0 0:0"` (using the
actual score for that history). Prefer keeping the complete reference and
selecting cases so their pre-point score can be taken from the preceding row.

When available, supply the sequential run's original start using
`--baseline-start-frame`. This preserves the first case's initialization
boundary, including when it becomes the prefix of an expanded context retry.
It does not reconstruct later player/tracker state. The validation trial
showed that even a four-frame shift at initialization can change a winner in
a later isolated case; matching the original boundary restored the result.

Every run gets a unique directory under `tmp/regression/parallel_points_*`:

- `plan.json`: frame windows, score seeds, worker count and input/source hashes.
- `point_NNN/manifest.json`: exact worker specification.
- `point_NNN/replay.log` and `history_*.csv`: native tracker output.
- `results.json`: per-case field differences, completion, timings and run status.

Workers use separate AI directories and private false-point learning files.
Frame-zero hotspot writes are redirected to the worker directory, and player
learning is disabled. Original profiles/configs stay unchanged. A source or
input edit during the run invalidates the combined result. Do not edit Python
sources while a run is processing.

The comparator checks event counts, start/end frames, scoring, winners, serve
attempts, next server and physical court ends. A missing event or native worker
failure cannot pass. Unexpected output is marked `needs_context_review`;
inspect its log and retry only the failed cases with preceding points:

```powershell
py.exe retry_parallel_point_context.py <completed-run-directory> --workers 2
```

This retries a mismatch with one preceding point, then two if necessary. It
checks all replayed context events as well as the target and writes
`context_results.json`, retaining the original cold-start differences. Already
passing cases are reused only if their source/input hashes still match. The
report records combined cold-batch and retry time. The plan retains all
reference cases even when `--cases` selects only a subset, allowing a failed
selected case to expand into its preceding context. If context still does not match,
use a wider sequential range. A cold start does not reconstruct the previous
player tracks, inference budget or event history.

Use this as the fast test after an edit. Keep the sequential court1 and full
night regression as the release check for cross-point state, court-end changes,
missed points between known windows, and cumulative scoring. Matching an
unreviewed history only demonstrates baseline equivalence, not correct winners.

The reviewed point after `2:0 15:15` has a focused fixture containing its
first fault and complete second-serve rally. Replay its original focused
initialization boundary with:

```powershell
py.exe parallel_point_regression.py regression_court1_point21.csv --court court1 --config hsv_config_court1_trial.json --initial-server-end far --initial-score "2:0 15:15" --baseline-start-frame 43340 --workers 1
```

Its reviewed outcome is P2, `2:0 15:30`, after the near player's final return
bounces outside the far right singles line. The fixture's exact serve onset
frames belong to this focused initialization; continuous replays also need
their original accumulated player/tracker state.


The reviewed court1 fixture through the third game is `regression_court1_through_game3.csv`: 26 serve-attempt records, 17 awarded-point cases, final score 3:0. It was verified by the continuous 33500?45780 replay. Record 18 now ends at confirmed net frame 42635 (previous fixture 42625), with the same winner/score/start; see `tmp/point23_fix_20261028/REPORT.md` for evidence and the raw isolated-context warnings. A cold isolated replay resets player/tracking memory, so `needs_context_review` must remain a review result rather than be presented as a pass.
