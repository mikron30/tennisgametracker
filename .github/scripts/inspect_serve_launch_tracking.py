from pathlib import Path

path = Path("interactive_ball_analyzer.py")
lines = path.read_text(encoding="utf-8").splitlines()

needles = [
    "def _prefer_serve_contact_launch_candidate",
    "SERVE-CONTACT LAUNCH",
    "def _prefer_night_contact_near_continuation_candidate",
    "Ball likely exited through upper side",
    "TOP-RETURN WAIT] activated",
    "def _top_return_wait_active",
    "top_return_wait",
    "serve_contact_grace",
]

out = []
for needle in needles:
    out.append("\n" + "=" * 100)
    out.append(f"NEEDLE: {needle}")
    out.append("=" * 100)
    matches = [i for i, line in enumerate(lines) if needle in line]
    out.append(f"matches={len(matches)}")
    for match_index, idx in enumerate(matches, start=1):
        start = max(0, idx - 140)
        end = min(len(lines), idx + 180)
        out.append(f"\n--- match {match_index} at line {idx + 1}, context {start + 1}:{end} ---")
        for line_no in range(start, end):
            out.append(f"{line_no + 1:6d}: {lines[line_no]}")

text = "\n".join(out) + "\n"
print(text)
Path(".github/scripts/serve_launch_context.txt").write_text(text, encoding="utf-8")
