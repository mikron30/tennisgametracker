from pathlib import Path

path = Path("interactive_ball_analyzer.py")
lines = path.read_text(encoding="utf-8").splitlines()

needles = [
    "def _night_startup",
    "NIGHT STARTUP REGULAR RECOVER",
    "reference_target_hold =",
    "reference_target_frame =",
    "_reference_point",
    "_find_night_lower_contact_launch_candidate(",
    "REFERENCE_POINT_HOLD",
]

out = []
for needle in needles:
    out.append("\n" + "=" * 100)
    out.append(f"NEEDLE: {needle}")
    out.append("=" * 100)
    matches = [i for i, line in enumerate(lines) if needle in line]
    out.append(f"matches={len(matches)}")
    for match_index, idx in enumerate(matches, start=1):
        start = max(0, idx - 120)
        end = min(len(lines), idx + 130)
        out.append(f"\n--- match {match_index} at line {idx + 1}, context {start + 1}:{end} ---")
        for line_no in range(start, end):
            out.append(f"{line_no + 1:6d}: {lines[line_no]}")

text = "\n".join(out) + "\n"
print(text)
Path(".github/scripts/serve_launch_context.txt").write_text(text, encoding="utf-8")
