from pathlib import Path

# V38 source-only inspection helper. It does not modify runtime code.
SOURCE = Path("interactive_ball_analyzer.py")
OUT = Path("V38_NET_TERMINATION_CONTEXT.md")

TARGETS = [
    "Ball hit the net",
    "Ball bounced before passing the net",
    "SAME-SIDE PRE-NET BOUNCE",
    "POST-CONTACT TIMEOUT HOLD",
    "net_error",
    "out_error",
]

lines = SOURCE.read_text(encoding="utf-8").splitlines()
sections = []
found_essential = {TARGETS[0]: 0, TARGETS[1]: 0}

for target in TARGETS:
    hits = [i for i, line in enumerate(lines) if target in line]
    if target in found_essential:
        found_essential[target] = len(hits)
    sections.append(f"## `{target}` — {len(hits)} occurrence(s)\n")
    if not hits:
        sections.append("_No occurrence found._\n")
        continue

    for n, idx in enumerate(hits, 1):
        start = max(0, idx - 120)
        end = min(len(lines), idx + 121)
        sections.append(f"### occurrence {n}: line {idx + 1}\n")
        sections.append("```python")
        for j in range(start, end):
            marker = ">>>" if j == idx else "   "
            sections.append(f"{marker} {j + 1:6d}: {lines[j]}")
        sections.append("```\n")

missing = [name for name, count in found_essential.items() if count == 0]
if missing:
    raise SystemExit(f"Missing essential target(s): {missing}")

header = [
    "# V38 net termination source context",
    "",
    f"Source: `{SOURCE}` ({len(lines)} lines)",
    "",
    "Generated automatically to inspect the exact runtime branches before applying V38.",
    "",
]
OUT.write_text("\n".join(header + sections) + "\n", encoding="utf-8")
print(f"Wrote {OUT} with contexts for {len(TARGETS)} targets")
