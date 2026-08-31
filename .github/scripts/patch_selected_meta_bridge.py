from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')
old = """            selected_meta_for_guard = None\n            should_guard_selected = self.ball_center and not allow_inactive and (\n"""
new = """            # Preserve the metadata of the contour that actually won all candidate\n            # arbitration stages.  Several trajectory selectors attach evidence to the\n            # candidate metadata; the false-positive guard must inspect that same entry\n            # instead of starting from None and losing the selection evidence.\n            selected_meta_for_guard = next(\n                (meta for meta in candidate_meta if meta.get('contour') is best_contour),\n                None,\n            )\n            should_guard_selected = self.ball_center and not allow_inactive and (\n"""
count = text.count(old)
if count != 1:
    raise SystemExit(f'expected exactly one selected_meta_for_guard initialization, found {count}')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
print('patched selected contour metadata bridge')
