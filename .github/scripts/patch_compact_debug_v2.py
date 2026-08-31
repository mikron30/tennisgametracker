from pathlib import Path
import ast

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

# These messages are useful only during deep frame-by-frame diagnosis.  Compact
# mode keeps one canonical [TRACK] line plus state transitions/recovery events.
noisy_markers = (
    'DEBUG: retrack_using_alt2 skipping',
    'DEBUG: [TOP-RETURN WAIT] holding near top edge',
    '[TOP-RETURN WAIT] ignoring non-reentry blob',
    'DEBUG: [SERVE-CONTACT] wide search radius=',
    'DEBUG: [GROUND-BOUNCE PREP]',
    'DEBUG: [SELECTED] contour at',
    'DEBUG: Singles sideline model built:',
    'DEBUG: Service box model built:',
    'Ball tracking continued',
    'Ball tracked at (',
)

tree = ast.parse(text)
lines = text.splitlines(keepends=True)
line_starts = []
pos = 0
for line in lines:
    line_starts.append(pos)
    pos += len(line)

replacements = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if not isinstance(node.func, ast.Name) or node.func.id != 'print':
        continue
    if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
        continue
    segment = ''.join(lines[node.lineno - 1:node.end_lineno])
    if not any(marker in segment for marker in noisy_markers):
        continue
    start = line_starts[node.lineno - 1] + node.col_offset
    end = line_starts[node.end_lineno - 1] + node.end_col_offset
    call_text = text[start:end]
    if not call_text.startswith('print('):
        continue
    replacements.append((start, end, '_verbose_debug_print(' + call_text[len('print('):]))

for start, end, replacement in sorted(replacements, reverse=True):
    text = text[:start] + replacement + text[end:]

path.write_text(text, encoding='utf-8')
print(f'Gated {len(replacements)} additional repetitive debug print calls')
