from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")

old = '''                    else:
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
'''

new = '''                    else:
                        # V30: the final per-candidate HSV is not bound yet on the
                        # strong-motion reacquisition path.  Use the exact candidate
                        # center pixel instead of a 5x5 median, which can be dominated
                        # by the surrounding court/player pixels.  This matches the
                        # same selected location that is about to be committed.
                        px = int(cx)
                        py = int(cy)
                        if not (
                                0 <= px < frame.shape[1] and
                                0 <= py < frame.shape[0]
                        ):
                            raise ValueError('candidate center outside frame')
                        candidate_pixel = frame[py:py + 1, px:px + 1]
                        if not candidate_pixel.size:
                            raise ValueError('empty candidate center pixel')
                        candidate_pixel_hsv = cv2.cvtColor(
                            candidate_pixel, cv2.COLOR_BGR2HSV
                        )[0, 0]
                        player_reacq_selected_h = int(candidate_pixel_hsv[0])
                        player_reacq_selected_s = int(candidate_pixel_hsv[1])
                        player_reacq_selected_v = int(candidate_pixel_hsv[2])

                    if player_reacq_candidate_area is not None:
                        player_reacq_selected_area = float(
                            player_reacq_candidate_area or 0.0
                        )
                    else:
                        candidate_contour = locals().get('best_contour')
                        if candidate_contour is not None:
                            player_reacq_selected_area = float(
                                cv2.contourArea(candidate_contour)
                            )
                        else:
                            # On this early reacquisition path the selected contour
                            # area may not be exposed until later.  Keep the guard
                            # conservative by using the last trusted ball size only
                            # as a large-object proxy; reanchoring still requires an
                            # independent visible-ball candidate far away.
                            player_reacq_selected_area = float(
                                current_ball_size or 0.0
                            )
'''

if text.count(old) != 1:
    raise SystemExit(f"expected exactly one V29 evidence fallback block, found {text.count(old)}")
text = text.replace(old, new, 1)

needle = '''                if player_reacq_visible_conflict_probe:
                    visible_ball = None
'''
replacement = '''                if player_reacq_visible_conflict_probe:
                    print(
                        f"Frame {self.frame_count}: "
                        f"[PLAYER-REACQ VISIBLE-CONFLICT PROBE] "
                        f"candidate=({cx},{cy}) "
                        f"hsv=({player_reacq_selected_h},"
                        f"{player_reacq_selected_s},"
                        f"{player_reacq_selected_v}) "
                        f"area={player_reacq_selected_area:.1f}px "
                        f"jump={actual_distance:.1f}px "
                        f"pred_dist={selected_predicted_distance:.1f}px"
                    )
                    visible_ball = None
'''
if text.count(needle) != 1:
    raise SystemExit(f"expected exactly one visible-conflict probe block, found {text.count(needle)}")
text = text.replace(needle, replacement, 1)

path.write_text(text, encoding="utf-8")
print("Applied V30 player-reacquisition precommit evidence fix")
