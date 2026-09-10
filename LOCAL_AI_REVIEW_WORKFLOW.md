# Reviewed ball data

The SQLite database indexes tracker output; tracker output is not ground truth.
`training_frames` now includes only explicit `accepted` or `corrected` reviews.
Corrections preserve the original coordinates, and can rescue held/lost frames.
A false serve or wrong point outcome does **not** make a visible ball a negative
appearance example. Negatives must lack a ball at the candidate center.

1. Back up the live database using SQLite's backup API before a review batch.
2. Use `ball_ai_log_review.py --log all.txt --video VIDEO --output DIRECTORY`
   to extract unlabelled review samples across logged rallies. Logs nominate
   samples; even `likely_correct` point verdicts never approve ball labels.
   The tool uses tracker frame minus one as the zero-based source frame and
   stops before the next video resets frame numbering. Verify that convention
   for logs from other tracker versions.
3. Inspect source images and temporal neighbours. Leave ambiguous examples
   unreviewed. Review only the individual frames inspected, not whole rallies.
4. Import explicit patch reviews using the AI environment:
   `.tools\ball-ai-venv310\Scripts\python.exe ball_ai_review.py --database DATABASE --reviews REVIEWS.json`.
   Each JSON row requires `image_path`, `video_path`, zero-based `source_frame`,
   `label` (`positive`/`negative`), `x`, `y`, `reviewer`, and an evidence `note`.
   Use `ball_ai_database.py review` for corrections/acceptance of indexed frames.
5. `ball_local_ai.py train` uses reviewed examples only. Entire source videos
   stay together across positive and negative tables; changing the sample limit
   does not change evaluation membership. Conflicting labels or missing source
   provenance stop training. At least 100 reviewed training positives and a
   held-out set with both classes are required. This is a minimum, not a promise
   of adequate diversity or model quality.
6. Train to a **new** candidate filename. Existing models cannot be overwritten.
   Evaluate old and candidate checkpoints on the same frozen reviewed set with
   the same seed. Report patch accuracy, true-center top-1 accuracy, hard-negative
   false-positive rate, and top-1 performance on difficult reviewed positives.
   The current top-1 benchmark compares the true center with four offset crops;
   also run actual recovery-candidate/video tests before promotion.
   The `compare` command accepts `--database`, `--old-model`,
   `--candidate-model`, and `--report` and writes both evaluations together.
   Checkpoints created by this workflow record their held-out source videos;
   evaluation rejects a different split or seed.
7. Old checkpoints lack training provenance. A split reserved now is not proof
   that the old model never saw those frames. A defensible promotion comparison
   needs additional reviewed footage excluded from **both** models' training.
   Require improved center ranking/accuracy without worse hard-negative or
   difficult-frame performance, and complete the tracker/scoring regression.
   Keep the incumbent and its hash/backup. There is no automatic model promotion.

Training and patch evaluation use the dedicated Python 3.10 AI environment.
Video extraction and tracker tests use the main Python environment with OpenCV.

The draft runtime guard is in `proposals/local_ai_physics.patch`. It is not
enabled in the tracker. `git apply --ignore-space-change --check` validates its
applicability despite Windows line endings; full video regression is still
required before enabling it.
