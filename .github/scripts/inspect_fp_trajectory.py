from pathlib import Path

path = Path('interactive_ball_analyzer.py')
lines = path.read_text(encoding='utf-8').splitlines()
out=[]

def add_region(start, end, title):
    out.append('\n' + '='*100)
    out.append(title)
    out.append('='*100)
    for n in range(max(1,start), min(len(lines),end)+1):
        out.append(f'{n:6d}: {lines[n-1]}')

def add_function(name):
    needle = f'    def {name}'
    matches=[i for i,line in enumerate(lines) if line.startswith(needle)]
    for idx in matches:
        end=idx+1
        while end < len(lines):
            line=lines[end]
            if line.startswith('    def ') and end>idx:
                break
            end+=1
        add_region(idx+1,end,f'FUNCTION {name} lines {idx+1}:{end}')

for name in [
    '_prefer_predicted_continuation_candidate',
    '_predicted_path_hotspot_override',
    '_tracking_false_positive_metrics',
    '_learn_ignored_tracking_position',
    '_prefer_serve_contact_launch_candidate',
    '_should_start_top_return_wait',
]:
    add_function(name)

add_region(18260,18480,'SELECTED CANDIDATE FALSE-POSITIVE GUARD')
add_region(12070,12490,'RETRACK HOTSPOT OVERRIDE PATH')
add_region(16980,17180,'MAIN CONTOUR CANDIDATE METADATA')

text='\n'.join(out)+'\n'
print(text)
Path('.github/scripts/fp_trajectory_context.txt').write_text(text,encoding='utf-8')
