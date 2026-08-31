from pathlib import Path

path = Path('interactive_ball_analyzer.py')
lines = path.read_text(encoding='utf-8').splitlines()
needles = [
    'TRACKING_FP_LEARN',
    'Predicted continuation -> preferring',
    'def _prefer_predicted_continuation_candidate',
    'frame0 hotspot',
    'PLAYER_REACQ_WEAK_JUMP_REJECTED',
    'NIGHT STARTUP REGULAR RECOVER',
    'REFERENCE_POINT_HOLD',
    'def _prefer_serve_contact_launch_candidate',
]
out=[]
for needle in needles:
    out.append('\n' + '='*100)
    out.append(f'NEEDLE: {needle}')
    out.append('='*100)
    matches=[i for i,line in enumerate(lines) if needle in line]
    out.append(f'matches={len(matches)}')
    for j,idx in enumerate(matches,1):
        start=max(0,idx-120)
        end=min(len(lines),idx+160)
        out.append(f'\n--- match {j} at line {idx+1}, context {start+1}:{end} ---')
        for n in range(start,end):
            out.append(f'{n+1:6d}: {lines[n]}')
text='\n'.join(out)+'\n'
print(text)
Path('.github/scripts/fp_trajectory_context.txt').write_text(text,encoding='utf-8')
