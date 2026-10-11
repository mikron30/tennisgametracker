from pathlib import Path


TARGET = Path("interactive_ball_analyzer.py")

OLD = '''                    try:
                        rebound_x = int(round(float(ball_position[0])))
                        rebound_y = int(round(float(ball_position[1])))
                        frame_h, frame_w = frame.shape[:2]
                        rebound_x = max(0, min(frame_w - 1, rebound_x))
                        rebound_y = max(0, min(frame_h - 1, rebound_y))
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        rebound_hsv = hsv_frame[rebound_y, rebound_x]
                        rebound_h = int(rebound_hsv[0])
                        rebound_s = int(rebound_hsv[1])
                        rebound_v = int(rebound_hsv[2])
                        dark_alt4_like = (
                            rebound_h >= 88 and
                            rebound_s >= 85 and
                            rebound_v <= 130
                        )
                    except Exception:
                        rebound_h = rebound_s = rebound_v = -1
                        dark_alt4_like = False
'''

NEW = '''                    try:
                        tracked_hsv = getattr(self, 'ball_hsv', None)
                        current_rebound_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
                        if tracked_hsv is not None and len(tracked_hsv) >= 3:
                            rebound_h = int(tracked_hsv[0])
                            rebound_s = int(tracked_hsv[1])
                            rebound_v = int(tracked_hsv[2])
                        else:
                            rebound_x = int(round(float(ball_position[0])))
                            rebound_y = int(round(float(ball_position[1])))
                            frame_h, frame_w = frame.shape[:2]
                            rebound_x = max(0, min(frame_w - 1, rebound_x))
                            rebound_y = max(0, min(frame_h - 1, rebound_y))
                            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            rebound_hsv = hsv_frame[rebound_y, rebound_x]
                            rebound_h = int(rebound_hsv[0])
                            rebound_s = int(rebound_hsv[1])
                            rebound_v = int(rebound_hsv[2])
                        dark_alt4_like = (
                            current_rebound_size <= 30.0 and
                            rebound_h >= 88 and
                            rebound_s >= 85 and
                            rebound_v <= 130
                        )
                    except Exception:
                        rebound_h = rebound_s = rebound_v = -1
                        current_rebound_size = 0.0
                        dark_alt4_like = False
'''

OLD_LOG = '''                                f"hsv=({rebound_h},{rebound_s},{rebound_v}) "
                                f"visible={visible_pos} "
'''
NEW_LOG = '''                                f"hsv=({rebound_h},{rebound_s},{rebound_v}) "
                                f"size={current_rebound_size:.1f}px "
                                f"visible={visible_pos} "
'''


def main():
    text = TARGET.read_text(encoding="utf-8")
    if "size={current_rebound_size:.1f}px" in text:
        raise SystemExit("V26 tracked-evidence refinement already present")
    if text.count(OLD) != 1:
        raise SystemExit(f"expected one V26 HSV block, found {text.count(OLD)}")
    if text.count(OLD_LOG) != 1:
        raise SystemExit(f"expected one V26 log block, found {text.count(OLD_LOG)}")
    updated = text.replace(OLD, NEW, 1).replace(OLD_LOG, NEW_LOG, 1)
    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")
    print("Refined V26 to use tracked HSV/size evidence")


if __name__ == "__main__":
    main()
