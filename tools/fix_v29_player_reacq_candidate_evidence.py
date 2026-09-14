from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")

old = '''                    candidate_sat = candidate_val = None
                    try:
                        if hsv_values is not None and len(hsv_values) >= 2:
                            candidate_sat = int(hsv_values[1])
                            candidate_val = int(hsv_values[2]) if len(hsv_values) >= 3 else None
                    except (TypeError, ValueError, IndexError):
                        candidate_sat = candidate_val = None

                    low_sat_large_fragment = (
                        candidate_sat is not None and
                        candidate_sat <= 80 and
                        float(current_ball_size or 0.0) >= 18.0
                    )
'''

new = '''                    # V29: this branch can run before hsv_values is bound in
                    # track_ball_in_frame.  Read already-bound candidate HSV when
                    # available; otherwise sample a tiny patch at the candidate.
                    # Keep candidate size tied to the selected contour rather than
                    # the previous committed ball size whenever possible.
                    candidate_hsv = locals().get('hsv_values')
                    candidate_sat = candidate_val = None
                    try:
                        if candidate_hsv is not None and len(candidate_hsv) >= 2:
                            candidate_sat = int(candidate_hsv[1])
                            candidate_val = (
                                int(candidate_hsv[2])
                                if len(candidate_hsv) >= 3 else None
                            )
                        else:
                            x0 = max(0, int(cx) - 2)
                            x1 = min(frame.shape[1], int(cx) + 3)
                            y0 = max(0, int(cy) - 2)
                            y1 = min(frame.shape[0], int(cy) + 3)
                            candidate_patch = frame[y0:y1, x0:x1]
                            if candidate_patch.size:
                                candidate_patch_hsv = cv2.cvtColor(
                                    candidate_patch, cv2.COLOR_BGR2HSV
                                )
                                candidate_sat = int(
                                    np.median(candidate_patch_hsv[:, :, 1])
                                )
                                candidate_val = int(
                                    np.median(candidate_patch_hsv[:, :, 2])
                                )
                    except (TypeError, ValueError, IndexError, cv2.error):
                        candidate_sat = candidate_val = None

                    try:
                        candidate_contour = locals().get('best_contour')
                        candidate_area = (
                            float(cv2.contourArea(candidate_contour))
                            if candidate_contour is not None
                            else float(current_ball_size or 0.0)
                        )
                    except (TypeError, ValueError, cv2.error):
                        candidate_area = float(current_ball_size or 0.0)

                    low_sat_large_fragment = (
                        candidate_sat is not None and
                        candidate_sat <= 80 and
                        candidate_area >= 18.0
                    )
'''

if text.count(old) != 1:
    raise SystemExit(f"expected exactly one V28 evidence block, found {text.count(old)}")

text = text.replace(old, new, 1)

old_log = '''                                    f"sat={candidate_sat} size={float(current_ball_size or 0.0):.1f}px "'''
new_log = '''                                    f"sat={candidate_sat} size={candidate_area:.1f}px "'''
if text.count(old_log) != 1:
    raise SystemExit(f"expected exactly one V28 size log, found {text.count(old_log)}")
text = text.replace(old_log, new_log, 1)

path.write_text(text, encoding="utf-8")
print("Applied V29 safe player-reacquisition candidate evidence")
