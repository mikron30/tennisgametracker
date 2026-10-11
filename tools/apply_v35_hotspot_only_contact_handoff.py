from pathlib import Path

PATH = Path("interactive_ball_analyzer.py")
text = PATH.read_text(encoding="utf-8")

if "V35_HOTSPOT_ONLY_CONTACT_HANDOFF" in text:
    print("V35 already applied")
    raise SystemExit(0)

start_marker = "                                    # V34: this exact frame is a detector handoff, not a\n"
end_marker = (
    "                                    print(\n"
    "                                        f\"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: \"\n"
)

start = text.find(start_marker)
if start < 0:
    raise SystemExit("V34 handoff block start not found")
end = text.find(end_marker, start)
if end < 0:
    raise SystemExit("CONTACT_LOCAL_AI_REPLACE print after V34 block not found")

block = text[start:end]
indented_lines = []
for line in block.splitlines(keepends=True):
    if line.strip():
        indented_lines.append("    " + line)
    else:
        indented_lines.append(line)
indented_block = "".join(indented_lines)

guard = (
    "                                    # V35_HOTSPOT_ONLY_CONTACT_HANDOFF\n"
    "                                    # Preserve the pre-handoff motion only for the\n"
    "                                    # trajectory-hotspot takeover that caused the\n"
    "                                    # false f4303 OUT reversal. Normal contact-stall\n"
    "                                    # and contact-jump replacements must retain the\n"
    "                                    # proven V32 behavior and commit the AI step.\n"
    "                                    if str(contact_reason or '').startswith(\n"
    "                                        'trajectory-hotspot-turn:'\n"
    "                                    ):\n"
)

text = text[:start] + guard + indented_block + text[end:]
PATH.write_text(text, encoding="utf-8")
print("Applied V35 hotspot-only Contact Local-AI handoff guard")
