from pathlib import Path


TARGET = Path("interactive_ball_analyzer.py")

OLD = '''        requested_history_end_frame = history_end_frame
        reason_lower = (reason or "").lower()
        confirmed_pending_endpoint = self._confirmed_pending_out_endpoint(reason)
        if confirmed_pending_endpoint is not None:
            # V15: the rebound frame proves the OUT but is not the bounce location.
            # Preserve V14's original source point instead of letting generic
            # terminal endpoint repair replace it with a current-frame blob.
            end_position = confirmed_pending_endpoint
            print(
                f"[POINT_END POSITION OVERRIDE] f{self.frame_count}: "
                f"using confirmed pending OUT source={end_position}"
            )
        else:
            end_position = self._terminal_player_overlap_position(reason, end_position, frame=frame)
        if "video_read_failure" in reason_lower:
'''

NEW = '''        requested_history_end_frame = history_end_frame
        reason_lower = (reason or "").lower()
        confirmed_pending_endpoint = self._confirmed_pending_out_endpoint(reason)
        if confirmed_pending_endpoint is not None:
            # V15: the rebound frame proves the OUT but is not the bounce location.
            # Preserve V14's original source point instead of letting generic
            # terminal endpoint repair replace it with a current-frame blob.
            end_position = confirmed_pending_endpoint
            print(
                f"[POINT_END POSITION OVERRIDE] f{self.frame_count}: "
                f"using confirmed pending OUT source={end_position}"
            )
        else:
            original_end_position = end_position
            end_position = self._terminal_player_overlap_position(reason, end_position, frame=frame)

            # V19: generic terminal repair can prove that a stale player-side
            # timeout marker was not the ball.  In V18 the repaired endpoint was
            # persisted, but the old \"Ball stopped on player side\" reason stayed
            # attached to it and was then copied verbatim into point history.
            # Re-label only the reviewed night-session failure mode: the original
            # reason must be exact, the repair must be a substantial relocation,
            # and the repaired point must no longer support the in-court stopped
            # interpretation.  Confirmed bounce/out endpoints above are untouched.
            terminal_repair_distance = 0.0
            if original_end_position is not None and end_position is not None:
                terminal_repair_distance = math.hypot(
                    float(end_position[0]) - float(original_end_position[0]),
                    float(end_position[1]) - float(original_end_position[1]),
                )
            if (
                self._is_night_session_config() and
                reason_lower == "ball stopped on player side" and
                terminal_repair_distance >= 70.0 and
                self._in_court_timeout_landing_outcome(end_position, frame) is None
            ):
                previous_reason = reason
                reason = "Ball lost (likely out of court)"
                reason_lower = reason.lower()
                print(
                    f"[TERMINAL REASON REPAIR] f{self.frame_count}: "
                    f"{previous_reason} at {original_end_position} -> "
                    f"{reason} at {end_position} "
                    f"distance={terminal_repair_distance:.1f}px"
                )
        if "video_read_failure" in reason_lower:
'''


def main():
    text = TARGET.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"expected exactly one V19 target block, found {count}")
    updated = text.replace(OLD, NEW, 1)
    compile(updated, str(TARGET), "exec")
    TARGET.write_text(updated, encoding="utf-8")
    print("Applied V19 terminal reason repair")


if __name__ == "__main__":
    main()
