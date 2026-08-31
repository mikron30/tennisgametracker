from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')
old = "            selected_meta_for_guard = None\n"
new = """            # Preserve the metadata attached to the contour that ultimately won all\n            # selection/arbitration stages.  Helpers such as predicted-continuation\n            # annotate their candidate metadata with evidence that the downstream\n            # false-positive guard needs.  Reconstructing the selected position while\n            # dropping that metadata makes the guard reject a trajectory-approved\n            # moving ball as a static/frame-0 hotspot.\n            selected_meta_for_guard = next(\n                (meta for meta in candidate_meta if meta.get('contour') is best_contour),\n                None,\n            )\n"""
count = text.count(old)
if count != 1:
    raise SystemExit(f'Expected exactly one selected_meta_for_guard initializer, found {count}')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
