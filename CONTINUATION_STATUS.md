# Continuation status — 2026-09-10

## Baseline and current candidate

The latest work was on `codex-ai-recovery-wip` at `ae11ebb`, not `main`.
The last shallow-serve workflow run (34480918881) failed with a NameError:
its patch referenced `shallow_serve_turn` in the IN classifier, although that
variable existed only in the OUT classifier. The patch was never committed.

This candidate defines the same shallow-turn evidence in the IN classifier,
uses the existing service-line slack for that evidence, and retains the
recent-static-artifact restriction. A confirmed post-start racket contact
blocks serve-only IN inference, matching the OUT path. The temporary workflow
now tests the checked-out code; it no longer edits or pushes runtime code.

The preserved f4118 regression fixture matches motion recorded in `all.txt`
(uploaded September 10): last center (1733,381), incoming (-12,4), outgoing
(-13,-56), angle 95.4 degrees, speed 57.5. Its service geometry is inherited
from the previous workflow fixture; pixel geometry and the true IN verdict
still require source-video verification. Added negative cases cover a rally
contact, wrong target half, expired serve, missing turn, day camera, static
artifact and excess service-line distance.

## Validation

- Focused suite: 40 passed (11 new cases plus 29 existing scoring/AI cases).
- Baseline full suite in a separate checkout: 14 failed, 137 passed,
  2 subtests passed. The candidate's first full run had the identical failure
  list and 147 passes, before adding the final passing static-artifact test.
- The 14 existing reported failures concern receiver selection, player-overlap
  endpoints, reference/false-start overrides, and a shallow-bounce fixture
  lacking active serve state. Do not assume that stale expected overrides
  should be restored to production merely to make tests pass.
- These are unit/replay tests, not full video validation or AI model training.

## Remaining work and promotion gate

1. Obtain `04_Left_Court_tennis_20260623220627_20260623222559_1274169779.mp4`.
   It is absent from the repository and was not found among accessible files.
   `all.txt` and `2000.txt` are available, but do not provide source pixels.
2. Run baseline/candidate video comparison around serve f4077 and turn
   f4116–4118 first. Check service-box geometry and final point ownership.
3. Run frames 0–27920 and compare all protected starts, endpoints and winners
   per REGRESSION_GUARDS.md, before promoting this candidate or the WIP AI
   trajectory guards. In particular review the reported 0:40 outcome,
   last-hit ownership and the false serve during the f2853+ rally.
4. Triage baseline test/ledger drift against those source images.
5. Continue LOCAL_AI_REVIEW_WORKFLOW.md using the actual reviewed database,
   incumbent checkpoint and held-out footage. None of these model/data assets
   is committed here; no new model has been trained or activated.

Candidate only: do not merge into an active runtime until video comparison
passes. Production branches and model selection remain unchanged.


## Clean-frame follow-up

Sequentially extracted source images support the service bounce near f4116
and show the far player's return around f4132. The runtime trace subsequently
holds stale coordinates despite continued play. At f4290–4292 the selected
marker overlaps the near player's head while the ball remains lower down.
These observations do not validate the complete point winner or full match.

The f4292 log also contains inconsistent motion records: displacement
(26,16) paired with direction -129.7 degrees, followed by (24,8) paired with
-12.5 degrees. Replaying these values reproduces the baseline's false OUT
classification. A candidate guard rejects sideline-bounce inference whenever
a stored motion direction differs from its displacement direction by more
than 5 degrees. It leaves trajectory and score state untouched and logs
OUT-BOUNCE MOTION CONFLICT. Real consistent outward rebounds remain eligible.
This is a containment fix, not a repair of candidate selection or a proof of
the correct winner. Full-video regression and tracking recovery remain open.

Focused tests: 50 passed, including the replay that fails on the old method,
a consistent real out bounce and equivalent angles wrapped by 360 degrees.
Audit sheets now decode sequentially with corrected frame mapping as well.


## V2 source-video result — September 11: FAILED promotion gate

The user replayed frames 4000–4499 and supplied serve_check_v2.zip. The process
exited 0 and sequential event images are now visually clean. This does not
mean tracking is correct. The f4292 mixed-motion OUT was suppressed, but the
same rally ended at f4314 instead, with the marker on the racket while the
near player visibly carries a ball in his other hand. Its stored winner also
changed from P1 to P2; neither result is validated.

A false subsequent serve is accepted at f4375 (history start f4370) while the
near player is walking, producing another false OUT at f4422. A further
provisional start occurs at f4439. Event counts fell from 9/7 high-severity
to 6/3, but that is not a correctness metric. The candidate remains a draft
and must not be promoted as a successful tracking/scoring fix.

Next work must address candidate identity across racket/head occlusion and
require serve-launch evidence independent of a rising player fragment. More
endpoint-specific suppression alone merely shifts the error. Existing clean
images and both logs are sufficient evidence for these failures; the user
does not need to repeat the same audit unchanged.
