from pathlib import Path

p = Path('interactive_ball_analyzer.py')
s = p.read_text(encoding='utf-8')

old = '''        if self.frame_count <= int(getattr(self, "_local_ai_follow_until_frame", -1)):
            return None
        if not self.local_ai_recovery.ready(self.frame_count):
            return None
'''
new = '''        if self.frame_count <= int(getattr(self, "_local_ai_follow_until_frame", -1)):
            # A verified trajectory recovery normally hands control straight
            # back to the HSV tracker.  If that tracker immediately *holds*
            # the repaired point instead of advancing it, however, the short
            # handoff must not suppress the very trajectory rescue that can
            # bridge the remaining player occlusion.  This is based only on
            # tracker state (same point / rising stuck count), never on a
            # particular frame or court coordinate.
            handoff_step = None
            if previous_position is not None and tracked_position is not None:
                handoff_step = math.hypot(
                    float(tracked_position[0]) - float(previous_position[0]),
                    float(tracked_position[1]) - float(previous_position[1]),
                )
            handoff_held = (
                tracked_position is None or
                (handoff_step is not None and handoff_step <= 2.5) or
                int(getattr(self, "stuck_frame_count", 0) or 0) > int(previous_stuck or 0)
            )
            if handoff_held and previous_position is not None:
                print(
                    f"[LOCAL_AI_HANDOFF_STALL] f{self.frame_count}: "
                    f"prev={previous_position} tracked={tracked_position} "
                    f"step={handoff_step if handoff_step is not None else 'missing'} "
                    f"stuck={int(getattr(self, 'stuck_frame_count', 0) or 0)}"
                )
                return "player-region:handoff-stall"
            return None
        if not self.local_ai_recovery.ready(self.frame_count):
            return None
'''
if old not in s:
    raise SystemExit('recovery follow-window marker not found')
s = s.replace(old, new, 1)

old2 = '''                        if local_ai_handoff_active:
                            # A post-contact HSV contour can mimic a same-side
                            # bounce during the very frames the local model has
                            # verified as a continuous outgoing flight.  Do not
                            # score a point until that short handoff finishes.
                            point_ended, reason = False, None
                            print(
                                f"[LOCAL_AI_POINT_END_HOLD] f{self.frame_count}: "
                                f"verified recovery path active through "
                                f"f{self._local_ai_follow_until_frame}"
                            )
                        else:
                            point_ended, reason = self.detect_point_end(tracked_position, frame)
'''
new2 = '''                        if local_ai_handoff_active:
                            # The handoff protects against player/racket blobs
                            # masquerading as a bounce, but it must never hide
                            # a geometry-confirmed boundary bounce.  Give the
                            # existing generic boundary-reversal detector first
                            # refusal; only ambiguous endpoint evidence is held.
                            boundary_out, boundary_reason = (
                                self._confirmed_boundary_reversal_out_candidate(
                                    tracked_position, frame
                                )
                            )
                            if boundary_out:
                                point_ended, reason = True, boundary_reason
                                print(
                                    f"[LOCAL_AI_POINT_END_BOUNDARY_OVERRIDE] "
                                    f"f{self.frame_count}: {boundary_reason}"
                                )
                            else:
                                point_ended, reason = False, None
                                print(
                                    f"[LOCAL_AI_POINT_END_HOLD] f{self.frame_count}: "
                                    f"verified recovery path active through "
                                    f"f{self._local_ai_follow_until_frame}"
                                )
                        else:
                            point_ended, reason = self.detect_point_end(tracked_position, frame)
'''
if old2 not in s:
    raise SystemExit('point-end handoff marker not found')
s = s.replace(old2, new2, 1)

p.write_text(s, encoding='utf-8')
print('patched Local AI handoff stall + boundary override')
