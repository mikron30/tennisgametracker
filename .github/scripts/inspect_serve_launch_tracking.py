from pathlib import Path

path = Path("interactive_ball_analyzer.py")
lines = path.read_text(encoding="utf-8").splitlines()

def extract_function(name):
    needle = f"    def {name}("
    starts = [i for i, line in enumerate(lines) if line.startswith(needle)]
    if not starts:
        return [f"MISSING {name}"]
    start = starts[0]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("    def "):
            end = i
            break
    out = [f"FUNCTION {name} lines {start + 1}:{end}"]
    out.extend(f"{i + 1:6d}: {lines[i]}" for i in range(start, end))
    return out

sections = []
for name in [
    "_prefer_serve_contact_launch_candidate",
    "_prefer_night_contact_near_continuation_candidate",
    "_should_start_top_return_wait",
    "_top_return_wait_active",
]:
    sections.append("\n" + "=" * 100)
    sections.extend(extract_function(name))

# Also include the call site where top-return wait is armed.
for needle in ["top_return_trigger_mode = self._should_start_top_return_wait", "[TOP-RETURN WAIT] activated"]:
    matches = [i for i, line in enumerate(lines) if needle in line]
    sections.append("\n" + "=" * 100)
    sections.append(f"CALLSITE {needle} matches={len(matches)}")
    for idx in matches:
        start = max(0, idx - 35)
        end = min(len(lines), idx + 55)
        sections.extend(f"{i + 1:6d}: {lines[i]}" for i in range(start, end))

text = "\n".join(sections) + "\n"
print(text)
Path(".github/scripts/serve_launch_context.txt").write_text(text, encoding="utf-8")
