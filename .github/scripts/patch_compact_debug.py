from pathlib import Path
import ast

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

# Add one module-level switch/helper. Use globals() later so this also works if
# argument parsing lives inside a function or __main__ block.
if '_verbose_debug_enabled = False' not in text:
    anchor = 'import argparse\n'
    if anchor not in text:
        raise SystemExit('Could not find argparse import')
    helper = '''import argparse\n\n# Compact console output is the default.  Detailed contour/search diagnostics\n# can be restored with --verbose-debug when a difficult frame needs inspection.\n_verbose_debug_enabled = False\n\ndef _verbose_debug_print(*args, **kwargs):\n    if _verbose_debug_enabled:\n        print(*args, **kwargs)\n'''
    text = text.replace(anchor, helper, 1)

# Add CLI flag next to --quiet.
if '"--verbose-debug"' not in text:
    quiet = '''    parser.add_argument("--quiet", action="store_true",\n                        help="Suppress tracker debug/status console output")\n'''
    if quiet not in text:
        raise SystemExit('Could not find --quiet parser block')
    replacement = quiet + '''    parser.add_argument("--verbose-debug", action="store_true",\n                        help="Show detailed contour/search rejection diagnostics")\n'''
    text = text.replace(quiet, replacement, 1)

# Activate the module-level flag immediately after parsing.
if "globals()['_verbose_debug_enabled']" not in text:
    anchor = '    args = parser.parse_args()\n'
    if anchor not in text:
        raise SystemExit('Could not find parse_args line')
    text = text.replace(
        anchor,
        anchor + "    globals()['_verbose_debug_enabled'] = bool(args.verbose_debug)\n",
        1,
    )

# Convert only noisy diagnostic print calls to the gated helper.  High-value
# events such as SELECTED, TRACK, recovery, point-end, OUT/NET, Local AI, etc.
# stay visible in compact mode.
noisy_markers = (
    'DEBUG: Contour',
    'DEBUG: Player-context penalty',
    'DEBUG: Serve-scan hint search around',
    'DEBUG: No previous ball position, searching entire frame',
    'DEBUG: [SERVE-DIR] Searching',
    'DEBUG: Searching at (',
    'DEBUG: Found {len(contours)} total contours',
    'DEBUG: Contact recovery active - checking BOTH HSV filters',
    'DEBUG: [LOWER-COURT BALL MASKS]',
    'DEBUG: [PROBLEM]',
    'DEBUG: HSV filter range',
    'DEBUG: HSV filter mode',
    'DEBUG: Search region:',
    'DEBUG: Previous ball position:',
    'DEBUG: KEEPING marker',
    'DEBUG: REASON:',
    'DEBUG:   - ',
    'DEBUG: Total contours:',
    'DEBUG: All {len(contours)} contours were rejected',
    'DEBUG: Rejected sizes',
    'DEBUG: Predicted point was',
)

tree = ast.parse(text)
lines = text.splitlines(keepends=True)
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
    start = sum(len(s) for s in lines[:node.lineno - 1]) + node.col_offset
    end = sum(len(s) for s in lines[:node.end_lineno - 1]) + node.end_col_offset
    call_text = text[start:end]
    if not call_text.startswith('print('):
        continue
    replacements.append((start, end, '_verbose_debug_print(' + call_text[len('print('):]))

for start, end, replacement in sorted(replacements, reverse=True):
    text = text[:start] + replacement + text[end:]

path.write_text(text, encoding='utf-8')
print(f'Gated {len(replacements)} noisy debug print calls')
