from pathlib import Path


TARGET = Path("interactive_ball_analyzer.py")

ANCHOR = '''                    coherent_rebound = False

                if coherent_rebound:
                    original_position = (
'''

INSERT = '''                    coherent_rebound = False

                # V26: the V20 player-zone guard can miss when player tracks are
                # stale/invisible.  Do not resurrect a suppressed static night
                # endpoint from a dark ALT4-like fragment when the existing
                # visible-ball recovery independently sees a clear ball elsewhere
                # in the same frame.  This is deliberately a two-signal guard:
                # a dark candidate alone remains legal, and a second visible
                # candidate alone does not veto a normal rebound.
                if coherent_rebound and frame is not None:
                    try:
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

                    visible_rebound = None
                    if dark_alt4_like:
                        try:
                            rebound_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            visible_rebound = self._find_night_visible_ball_candidate(
                                frame, rebound_gray
                            )
                        except Exception:
                            visible_rebound = None

                    if visible_rebound is not None:
                        try:
                            visible_pos = tuple(visible_rebound.get('pos', ()))
                            visible_conflict_distance = math.hypot(
                                float(visible_pos[0]) - float(ball_position[0]),
                                float(visible_pos[1]) - float(ball_position[1]),
                            ) if len(visible_pos) >= 2 else 0.0
                        except (TypeError, ValueError, IndexError):
                            visible_pos = ()
                            visible_conflict_distance = 0.0

                        visible_conflict_limit = max(80.0, float(width) * 0.02)
                        if visible_conflict_distance >= visible_conflict_limit:
                            print(
                                f"Frame {self.frame_count}: "
                                f"[OUT-BOUNCE PENDING VISIBLE-CONFLICT REJECT] "
                                f"source_f={pending_frame} point={pending_pos} "
                                f"rebound={tuple(ball_position)} "
                                f"hsv=({rebound_h},{rebound_s},{rebound_v}) "
                                f"visible={visible_pos} "
                                f"conflict={visible_conflict_distance:.1f}px"
                            )
                            coherent_rebound = False

                if coherent_rebound:
                    original_position = (
'''


def main():
    text = TARGET.read_text(encoding="utf-8")
    if "OUT-BOUNCE PENDING VISIBLE-CONFLICT REJECT" in text:
        raise SystemExit("V26 guard already present")
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(f"expected exactly one V26 anchor, found {count}")
    updated = text.replace(ANCHOR, INSERT, 1)
    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")
    print("Applied V26 pending-OUT visible-conflict guard")


if __name__ == "__main__":
    main()
