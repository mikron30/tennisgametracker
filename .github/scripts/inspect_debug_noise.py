from pathlib import Path

p = Path('interactive_ball_analyzer.py')
lines = p.read_text(encoding='utf-8').splitlines()
needles = [
    'REJECTED -',
    'CANDIDATE - pos=',
    'Searching at (',
    'Serve-scan hint search around',
    'Found {len(contours)} total contours',
    '[PROBLEM] No valid candidate found',
    'Rejected sizes (first 5)',
    'REASON: Ball size changed',
    'argparse.ArgumentParser',
    'parse_args()',
]
for needle in needles:
    print('\n' + '='*100)
    print('NEEDLE:', needle)
    matches = [i for i,s in enumerate(lines) if needle in s]
    print('matches=', len(matches))
    for i in matches:
        a=max(0,i-12); b=min(len(lines),i+20)
        print(f'--- line {i+1}, context {a+1}:{b} ---')
        for j in range(a,b):
            print(f'{j+1:6d}: {lines[j]}')
