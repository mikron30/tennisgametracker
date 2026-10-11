from pathlib import Path


TRACKER = Path("interactive_ball_analyzer.py")
MARKER = "[PLAYER-REACQ EXTREME-PREDICTION REJECT]"

ANCHOR = '''                    return self.ball_center

                # Clearing the protection now needs either strong image motion
                # or geometric continuity. This preserves slow nearby tracks
'''

INSERT = '''                    return self.ball_center

                # V25: strong image motion by itself must not clear the
                # player-contact reacquisition guard for an extreme jump that
                # strongly contradicts the predicted ball trajectory. The
                # reviewed V20 f175 failure jumped 388px from the recovered
                # ball while landing about 560px from prediction; moving-player
                # image motion made that false fragment look valid. Hold the
                # last trusted ball for one more frame instead. Specialized
                # top/back-return paths keep their own long-jump validation.
                player_reacq_extreme_jump_limit = max(
                    240.0, float(frame_width) * 0.065
                )
                player_reacq_extreme_pred_limit = max(
                    150.0, float(frame_width) * 0.045
                )
                player_reacq_extreme_prediction_conflict = (
                    player_reacq_guard_active and
                    actual_distance >= player_reacq_extreme_jump_limit and
                    selected_predicted_distance is not None and
                    selected_predicted_distance >= player_reacq_extreme_pred_limit and
                    not top_return_search_context and
                    not back_return_search_context
                )
                if player_reacq_extreme_prediction_conflict:
                    self._record_rejected_contour_debug(
                        best_contour,
                        x1,
                        y1,
                        cx,
                        cy,
                        selected_area_for_guard,
                        (
                            f"player-reacq extreme prediction conflict "
                            f"jump={actual_distance:.1f}px "
                            f"pred_dist={selected_predicted_distance:.1f}px"
                        ),
                        source=best_source,
                    )
                    self.stuck_frame_count = max(
                        int(getattr(self, 'stuck_frame_count', 0)) + 1,
                        1,
                    )
                    print(
                        f"Frame {self.frame_count}: "
                        f"[PLAYER-REACQ EXTREME-PREDICTION REJECT] "
                        f"holding {self.ball_center} instead of ({cx},{cy}) "
                        f"jump={actual_distance:.1f}px "
                        f"limit={player_reacq_extreme_jump_limit:.1f}px "
                        f"pred_dist={selected_predicted_distance:.1f}px "
                        f"pred_limit={player_reacq_extreme_pred_limit:.1f}px "
                        f"motion={motion_mean:.1f}/{motion_max:.1f}"
                    )
                    return self.ball_center

                # Clearing the protection now needs either strong image motion
                # or geometric continuity. This preserves slow nearby tracks
'''


def main() -> None:
    text = TRACKER.read_text(encoding="utf-8")
    if MARKER in text:
        print("V25 guard already present")
        return
    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(
            f"Expected exactly one V25 insertion anchor, found {count}; refusing unsafe patch"
        )
    TRACKER.write_text(text.replace(ANCHOR, INSERT, 1), encoding="utf-8")
    print("V25 extreme reacquisition prediction guard inserted")


if __name__ == "__main__":
    main()
