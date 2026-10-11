from pathlib import Path

path = Path('interactive_ball_analyzer.py')
lines = path.read_text(encoding='utf-8').splitlines()
needles = [
    'tracked_position = self.track_ball_in_frame(frame)',
    'Ball tracked at',
    '_contact_normal_fallback_path',
    '_contact_local_ai_trigger(',
    'prev_ball_center',
]

seen = set()
for needle in needles:
    hits = [i for i, line in enumerate(lines) if needle in line]
    print(f'\n=== NEEDLE {needle!r}: {len(hits)} hits ===')
    for i in hits:
        start = max(0, i - 45)
        end = min(len(lines), i + 105)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        print(f'\n--- context lines {start+1}-{end} around hit {i+1} ---')
        for n in range(start, end):
            print(f'{n+1:06d}: {lines[n]}')
