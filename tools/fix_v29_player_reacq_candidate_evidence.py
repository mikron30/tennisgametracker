from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")

old = '''                try:
                    player_reacq_selected_h = int(hsv_values[0])
                    player_reacq_selected_s = int(hsv_values[1])
                    player_reacq_selected_v = int(hsv_values[2])
                    player_reacq_selected_area = float(bulb_size or 0.0)
                except (TypeError, ValueError, IndexError):
                    player_reacq_selected_h = -1
                    player_reacq_selected_s = -1
                    player_reacq_selected_v = -1
                    player_reacq_selected_area = 0.0
'''

new = '''                # V29: this V28 probe executes before hsv_values/bulb_size are
                # guaranteed to be bound on every tracking path.  Prefer those
                # already-bound values when present, otherwise derive evidence
                # directly from the selected candidate location/contour.
                player_reacq_candidate_hsv = locals().get('hsv_values')
                player_reacq_candidate_area = locals().get('bulb_size')
                try:
                    if (
                            player_reacq_candidate_hsv is not None and
                            len(player_reacq_candidate_hsv) >= 3
                    ):
                        player_reacq_selected_h = int(player_reacq_candidate_hsv[0])
                        player_reacq_selected_s = int(player_reacq_candidate_hsv[1])
                        player_reacq_selected_v = int(player_reacq_candidate_hsv[2])
                    else:
                        x0 = max(0, int(cx) - 2)
                        x1 = min(frame.shape[1], int(cx) + 3)
                        y0 = max(0, int(cy) - 2)
                        y1 = min(frame.shape[0], int(cy) + 3)
                        candidate_patch = frame[y0:y1, x0:x1]
                        if not candidate_patch.size:
                            raise ValueError('empty candidate patch')
                        candidate_patch_hsv = cv2.cvtColor(
                            candidate_patch, cv2.COLOR_BGR2HSV
                        )
                        player_reacq_selected_h = int(
                            np.median(candidate_patch_hsv[:, :, 0])
                        )
                        player_reacq_selected_s = int(
                            np.median(candidate_patch_hsv[:, :, 1])
                        )
                        player_reacq_selected_v = int(
                            np.median(candidate_patch_hsv[:, :, 2])
                        )

                    if player_reacq_candidate_area is not None:
                        player_reacq_selected_area = float(
                            player_reacq_candidate_area or 0.0
                        )
                    else:
                        candidate_contour = locals().get('best_contour')
                        player_reacq_selected_area = (
                            float(cv2.contourArea(candidate_contour))
                            if candidate_contour is not None else 0.0
                        )
                except (TypeError, ValueError, IndexError, cv2.error):
                    player_reacq_selected_h = -1
                    player_reacq_selected_s = -1
                    player_reacq_selected_v = -1
                    player_reacq_selected_area = 0.0
'''

count = text.count(old)
if count != 1:
    raise SystemExit(f"expected exactly one V28 unsafe evidence block, found {count}")

text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("Applied V29 safe player-reacquisition candidate evidence")
