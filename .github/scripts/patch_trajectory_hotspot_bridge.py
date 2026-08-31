from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')
lines = text.splitlines()

# 1) A candidate returned by the strict predicted-continuation selector has
# already passed direction, step-size, prediction-error, motion and player-zone
# gates.  The generic frame-0 hotspot override intentionally only covers lower
# speeds, so mark high-speed selected continuations explicitly instead of
# globally relaxing the hotspot detector.
start = next(i for i, line in enumerate(lines) if line.startswith('    def _prefer_predicted_continuation_candidate'))
end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith('    def '))
return_indices = [i for i in range(start, end) if lines[i].strip() == 'return chosen']
if not return_indices:
    raise RuntimeError('Could not locate final return chosen in predicted continuation helper')
return_idx = return_indices[-1]
marker = "        if lm_dist > 120.0:"
if marker not in lines[start:end]:
    insertion = [
        '        # The normal hotspot-path override is intentionally capped at lower speeds.',
        '        # A high-speed candidate reaching this point has already passed the stricter',
        '        # trajectory/direction/motion gates above, so preserve that evidence for the',
        '        # later false-positive guard instead of globally weakening hotspot protection.',
        '        if lm_dist > 120.0:',
        "            chosen['trajectory_continuation_selected'] = True",
    ]
    lines[return_idx:return_idx] = insertion

text = '\n'.join(lines) + ('\n' if text.endswith('\n') else '')

# 2) Let the selected-candidate false-positive guard honor the strict trajectory
# selection in addition to the ordinary low-speed predicted-path override.
old = """                selected_predicted_path_override = (\n                    selected_meta_for_guard is not None and\n                    bool(selected_meta_for_guard.get('predicted_path_hotspot_override'))\n                )\n"""
new = """                selected_trajectory_continuation = (\n                    selected_meta_for_guard is not None and\n                    bool(selected_meta_for_guard.get('trajectory_continuation_selected'))\n                )\n                selected_predicted_path_override = (\n                    selected_meta_for_guard is not None and\n                    bool(\n                        selected_meta_for_guard.get('predicted_path_hotspot_override') or\n                        selected_trajectory_continuation\n                    )\n                )\n"""
if old in text:
    text = text.replace(old, new, 1)
elif "selected_trajectory_continuation = (" not in text:
    raise RuntimeError('Could not locate selected predicted-path hotspot guard')

path.write_text(text, encoding='utf-8')

# Static regression checks: the strict selector carries the marker and the
# later hotspot guard consumes it.  These are deliberately behavior-shape
# checks, not clip/frame-specific tests.
updated = path.read_text(encoding='utf-8')
assert "chosen['trajectory_continuation_selected'] = True" in updated
assert "selected_meta_for_guard.get('trajectory_continuation_selected')" in updated
assert "selected_predicted_path_override = (" in updated
print('trajectory hotspot bridge patch applied')
