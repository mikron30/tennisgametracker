from pathlib import Path

src = Path('interactive_ball_analyzer.py').read_text(encoding='utf-8').splitlines()
needles = [
    'contact_reason is not None',
    'CONTACT_LOCAL_AI_REPLACE',
    'contact_local_ai_position',
    '_contact_local_ai_trigger(',
]

out = ['# V34 Contact Local-AI replacement context', '']
seen = set()
for needle in needles:
    hits = [i for i, line in enumerate(src) if needle in line]
    out.append(f'## {needle!r} — {len(hits)} hit(s)')
    out.append('```python')
    for i in hits:
        start = max(0, i - 80)
        end = min(len(src), i + 180)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        for n in range(start, end):
            out.append(f'{n+1:06d}: {src[n]}')
    out.append('```')
    out.append('')

Path('V34_CONTACT_REPLACE_CONTEXT.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('Wrote V34_CONTACT_REPLACE_CONTEXT.md')
