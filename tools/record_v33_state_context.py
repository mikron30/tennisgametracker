from pathlib import Path

src = Path('interactive_ball_analyzer.py').read_text(encoding='utf-8').splitlines()
needles = [
    'def _try_local_ai_recovery',
    'repaired_position = (int(recovered["x"]), int(recovered["y"]))',
    'def _snapshot_tracking_state_for_provisional_guard',
    'def _restore_tracking_state_for_provisional_guard',
    'tracked_position = self._try_local_ai_recovery(',
]

out = ['# V33 post-track state context', '']
seen = set()
for needle in needles:
    hits = [i for i, line in enumerate(src) if needle in line]
    out.append(f'## {needle!r} — {len(hits)} hit(s)')
    out.append('```python')
    for i in hits:
        if needle == 'def _try_local_ai_recovery':
            before, after = 35, 520
        elif needle.startswith('def '):
            before, after = 35, 260
        elif needle.startswith('repaired_position'):
            before, after = 35, 180
        else:
            before, after = 70, 180
        start = max(0, i - before)
        end = min(len(src), i + after)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        for n in range(start, end):
            out.append(f'{n+1:06d}: {src[n]}')
    out.append('```')
    out.append('')

Path('V33_POST_TRACK_CONTEXT.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('Wrote V33_POST_TRACK_CONTEXT.md')
