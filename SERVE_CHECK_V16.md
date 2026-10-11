# V16 candidate: roll back bounce evidence from rejected tracking candidates

## Recovered continuation state

The latest available run is `serve_check_long.zip` uploaded September 13,
after `serve_check_v15.zip`. The remote baseline is `e0a41b8`; its only change
after V15 (`148cbcd`) is an inspection workflow, not a runtime V16 fix.

The long run retains V15's first endpoint: start 4074, end 4287, winner P1,
right-sideline OUT, original bounce position `(2922,729)`.

The next point, starting 4566, exposes a separate failure. Its final trace
holds `(1783,815)` while rejected HSV proposals continue to register events.
At 4663 and 4667, `predicted launch` registers the same origin and incoming
motion `(-35,34,48.8)` twice. The queued double-bounce terminal bypasses the
player-reacquisition grace and scores P2 at 4667. The endpoint sheet shows
the held marker on the walking near player; these are not two validated
ground bounces. The correct earlier end and winner remain unverified.

## Change

Extend the existing pre-candidate tracking snapshot/restore operation to
include bounce count and time, pending terminal, bounce debug evidence,
impact/review markers, and serve-bounce flags/statistics. A rejected candidate
must not retain these side effects after its position and motion are restored.
Nested values are copied on both save and restore so retries cannot mutate
the saved statistics or review events. Lazily created bounce fields are
removed when they did not exist before the rejected candidate.

This does not add coordinate/frame exceptions, change candidate acceptance,
train a model, or claim to recover the actual ball through player occlusion.
Independent recovery grace and V14/V15 OUT provenance fields are retained.
The outer maximum-jump gate remains a separate existing path; this change
addresses the established snapshot rollback paths only.

## Validation

Five new tests replay real bounce registration followed by rejection, preserve
a prior accepted bounce, allow accepted double bounces, and verify retry-safe
serve statistics and review-event restoration. Four fail on `e0a41b8` and pass
with V16; the accepted-double-bounce control passes on both.

The focused compatibility selection (11 test files) has 130 passing tests,
11 pre-existing failures and 2 passing subtests. Running the identical tests
on `e0a41b8`, including the new regression tests, gives 126 passes and 15
failures, with the same 2 passing subtests. The four fixed failures are the
new rollback regressions; the old failures concern disabled reference
overrides and an incomplete shallow-serve fixture. They were not changed.

No source-video replay was performed here: the supplied archives contain
logs and annotated contact sheets, not the original video or AI checkpoint.
This remains an unpromoted candidate pending focused video comparison.

## Next run

```powershell
git switch codex/shallow-serve-turn-candidate
git pull --ff-only
python tracking_audit.py --court night --start-frame 4000 --max-frames 900 --output-dir tmp\serve_check_v16 --provider local --max-events 20 --endpoint-audit
Compress-Archive -Path tmp\serve_check_v16\* -DestinationPath tmp\serve_check_v16.zip -Force
```

Check that the first point and its original endpoint remain unchanged, then
inspect `[REJECTED_CANDIDATE_BOUNCE_ROLLBACK]` and the actual continuation of
the point starting 4566. Suppressing the false 4667 endpoint alone does not
validate the replacement end or winner. Only after that comparison passes
should the full 0-27920 protected regression be run, as in REGRESSION_GUARDS.md.

The long run's later endpoints at 5123, 5578, 6051 and 6466 remain to be
reviewed. Several audit jump/gap warnings also span separate points and are
not evidence of an in-rally teleport; do not use the raw severity count as
a correctness score.
