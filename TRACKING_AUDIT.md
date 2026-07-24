# Tracking audit workflow

`tracking_audit.py` runs a bounded headless regression and reviews only frames
where tracking looks risky. It does not modify the tracker or learned player
profile (`--disable-player-learning` is used for the run).

Run the current night-session segment:

```powershell
python tracking_audit.py --court night --start-frame 0 --max-frames 360 `
  --output-dir tmp/tracking_audit_f50 --provider local
```

The report is written to `tmp/tracking_audit_f50/index.html`. It also contains:

- `tracking_audit.json`: suspicious windows, evidence, point association, and optional vision verdicts;
- `tracking_trace.jsonl`: parsed per-frame tracker positions;
- `events/*.jpg`: contact sheets with the selected tracker position in green;
- `tracker.log`: the bounded regression log;
- timestamped point history CSV from that run.

## Live progress

Every run writes two small live files in the output directory:

- `progress.txt` is a human-readable progress bar that can stay open in VS Code;
- `progress.json` contains the same state for scripts and tooling.

For example, while this command is running:

```powershell
python tracking_audit.py --court night --start-frame 0 --max-frames 360 `
  --output-dir tmp\tracking_audit_resume
```

open `tmp\tracking_audit_resume\progress.txt`. It updates during the headless
regression and again while suspicious visual windows are rendered. Use
`--progress-file <path>` to choose a different JSON path; the companion text
file uses the same name with a `.txt` suffix.

To rerun the audit without rerunning the tracker:

```powershell
python tracking_audit.py --court night --skip-run `
  --log tmp/tracking_audit_f50/tracker.log `
  --history tmp/tracking_audit_f50/point_history_YYYYMMDD_HHMMSS.csv `
  --output-dir tmp/tracking_audit_f50_review --provider local
```

With the optional vision dependencies and `OPENAI_API_KEY`, use
`--provider auto` or `--provider openai` to send only flagged contact sheets
for structured visual review. Local mode remains deterministic and does not
make network calls.

## Reliable point-end sheets

HEVC files can show corrupted frames when OpenCV seeks directly to many small
windows. For endpoint decisions, use one sequential forward decode instead:

```powershell
python tracking_audit.py --court night --skip-run `
  --log tmp\tracking_audit_full\tracker.log `
  --history tmp\tracking_audit_full\point_history_YYYYMMDD_HHMMSS.csv `
  --output-dir tmp\tracking_audit_full_endpoint_review `
  --endpoint-only --endpoint-before 4 --endpoint-after 5
```

Open `endpoint_index.html`, or inspect `endpoint_audit.json`. Each sheet labels
both the tracker frame and zero-based raw video frame. Green is the live ball
tracker marker; red is the stored point endpoint. The source is decoded once
from the beginning, so no random HEVC seek is used for these images. Progress
is visible in `progress.txt` throughout the decode.
