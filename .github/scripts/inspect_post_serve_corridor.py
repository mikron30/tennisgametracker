from pathlib import Path

src = Path('interactive_ball_analyzer.py').read_text(encoding='utf-8').splitlines()
tests = Path('tests/test_serve_launch_handoff.py').read_text(encoding='utf-8').splitlines()

needles = [
    'def _try_local_ai_trajectory_rescue',
    'def _try_local_ai_recovery',
    'def _post_serve_rejected_update_reason',
    '_post_serve_launch_lock_until_frame',
    '_serve_launch_direction_x',
    'rank_trajectory_candidates',
]

hits = []
for i, line in enumerate(src, 1):
    if any(n in line for n in needles):
        hits.append(i)

windows = []
for i in hits:
    a = max(1, i - 70)
    b = min(len(src), i + 130)
    if windows and a <= windows[-1][1] + 1:
        windows[-1] = (windows[-1][0], max(windows[-1][1], b))
    else:
        windows.append((a, b))

out = []
for a, b in windows:
    out.append('=' * 100)
    out.append(f'SOURCE {a}:{b}')
    out.append('=' * 100)
    for n in range(a, b + 1):
        out.append(f'{n:6d}: {src[n-1]}')

out.append('\n' + '=' * 100)
out.append('TEST FILE')
out.append('=' * 100)
for n, line in enumerate(tests, 1):
    if ('PostServe' in line or 'Trajectory' in line or 'post_serve' in line or
            'trajectory' in line or 'rollback' in line):
        a = max(1, n - 20)
        b = min(len(tests), n + 55)
        out.append(f'--- TESTS {a}:{b} ---')
        for j in range(a, b + 1):
            out.append(f'{j:6d}: {tests[j-1]}')

Path('.github/scripts/post_serve_corridor_context.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')

start = next(i for i, line in enumerate(src) if line.startswith('    def _try_local_ai_trajectory_rescue('))
end = next(i for i in range(start + 1, len(src)) if src[i].startswith('    def ') or src[i].startswith('    @'))
exact = []
for i in range(start, end):
    exact.append(f'{i+1:6d}: {src[i]}')
Path('.github/scripts/post_serve_trajectory_function.txt').write_text('\n'.join(exact) + '\n', encoding='utf-8')

print(f'wrote {len(windows)} source windows, {len(hits)} hits, exact function {start+1}:{end}')
