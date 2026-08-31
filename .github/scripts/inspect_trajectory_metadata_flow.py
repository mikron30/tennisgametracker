from pathlib import Path

path = Path('interactive_ball_analyzer.py')
lines = path.read_text(encoding='utf-8').splitlines()
patterns = [
    'selected_meta_for_guard',
    '_prefer_predicted_continuation_candidate',
    'trusted_trajectory_continuation',
    'predicted_path_hotspot_override',
    'selected_contour',
    'candidate_meta',
]

hits = []
for i, line in enumerate(lines, start=1):
    if any(p in line for p in patterns):
        hits.append(i)

# Merge generous windows around relevant hits so we can trace metadata flow.
windows = []
for i in hits:
    start = max(1, i - 25)
    end = min(len(lines), i + 35)
    if windows and start <= windows[-1][1] + 1:
        windows[-1] = (windows[-1][0], max(windows[-1][1], end))
    else:
        windows.append((start, end))

out = []
for start, end in windows:
    out.append('=' * 100)
    out.append(f'SOURCE lines {start}:{end}')
    out.append('=' * 100)
    for n in range(start, end + 1):
        out.append(f'{n:6d}: {lines[n-1]}')

Path('.github/scripts/trajectory_metadata_flow.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
print(f'wrote {len(windows)} windows from {len(hits)} hits')
