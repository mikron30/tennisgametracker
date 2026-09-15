from pathlib import Path


TARGET = Path("interactive_ball_analyzer.py")

OLD = '''                            print(f"Frame {self.frame_count}: [ALT4 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            override_applied = True
                            hsv_override_applied = True
'''

NEW = '''                            print(f"Frame {self.frame_count}: [ALT4 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            override_applied = True
                            hsv_override_applied = True

                            # V27: resolve the V26 pending-OUT conflict before the
                            # dark ALT4 fragment is committed into ball motion and
                            # direction-change state.  At f4287 the pending static
                            # side candidate is still anchored at the last trusted
                            # position while ALT4 selects a tiny dark player/court
                            # fragment.  The existing night visible-ball finder sees
                            # the real ball elsewhere in the same frame.  Reanchor
                            # here, before Movement/direction-change processing, so
                            # the false fragment never poisons bounce or trajectory
                            # history.  Keep this deliberately two-signal and only
                            # inside the short pending-static-OUT confirmation window.
                            pending_static_out = getattr(self, '_pending_night_static_out', None)
                            if isinstance(pending_static_out, dict):
                                try:
                                    pending_frame = int(
                                        pending_static_out.get('frame', -1000000)
                                    )
                                    pending_age = int(self.frame_count) - pending_frame
                                    alt4_hsv = retrack4.get('hsv')
                                    alt4_area = float(
                                        retrack4.get('area', bulb_size) or 0.0
                                    )
                                    alt4_dark_tiny = (
                                        alt4_hsv is not None and
                                        len(alt4_hsv) >= 3 and
                                        alt4_area <= 30.0 and
                                        int(alt4_hsv[0]) >= 88 and
                                        int(alt4_hsv[1]) >= 85 and
                                        int(alt4_hsv[2]) <= 130
                                    )
                                except (TypeError, ValueError, IndexError):
                                    pending_frame = -1000000
                                    pending_age = -1
                                    alt4_area = 0.0
                                    alt4_dark_tiny = False

                                if 3 <= pending_age <= 4 and alt4_dark_tiny:
                                    visible_rebound = self._find_night_visible_ball_candidate(
                                        frame, frame_gray
                                    )
                                    if visible_rebound is not None:
                                        try:
                                            visible_pos = tuple(
                                                visible_rebound.get('pos', ())
                                            )
                                            visible_conflict_distance = (
                                                math.hypot(
                                                    float(visible_pos[0]) - float(cx),
                                                    float(visible_pos[1]) - float(cy),
                                                )
                                                if len(visible_pos) >= 2 else 0.0
                                            )
                                        except (TypeError, ValueError, IndexError):
                                            visible_pos = ()
                                            visible_conflict_distance = 0.0

                                        visible_conflict_limit = max(
                                            80.0, float(frame.shape[1]) * 0.02
                                        )
                                        if visible_conflict_distance >= visible_conflict_limit:
                                            print(
                                                f"Frame {self.frame_count}: "
                                                f"[OUT-PENDING ALT4 VISIBLE-CONFLICT REJECT] "
                                                f"source_f={pending_frame} "
                                                f"rejected=({cx},{cy}) "
                                                f"hsv=({int(alt4_hsv[0])},"
                                                f"{int(alt4_hsv[1])},"
                                                f"{int(alt4_hsv[2])}) "
                                                f"size={alt4_area:.1f}px "
                                                f"visible={visible_pos} "
                                                f"conflict={visible_conflict_distance:.1f}px"
                                            )
                                            self._pending_night_static_out = None
                                            reanchor_candidate = dict(visible_rebound)
                                            reanchor_candidate['recovery_label'] = (
                                                'OUT-PENDING ALT4 VISIBLE REANCHOR'
                                            )
                                            return self._commit_night_visible_ball_recovery(
                                                reanchor_candidate, frame
                                            )
'''


def main():
    text = TARGET.read_text(encoding="utf-8")
    marker = "OUT-PENDING ALT4 VISIBLE-CONFLICT REJECT"
    if marker in text:
        raise SystemExit("V27 precommit guard already present")
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"expected exactly one V27 ALT4 target block, found {count}")
    updated = text.replace(OLD, NEW, 1)
    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")
    print("Applied V27 pending-OUT ALT4 visible-ball precommit reanchor")


if __name__ == "__main__":
    main()
