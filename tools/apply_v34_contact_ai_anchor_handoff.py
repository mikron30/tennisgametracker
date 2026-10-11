from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old = '''                                if tracked_position is not None:
                                    contact_local_ai = True
                                    print(
                                        f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "
                                        f"HSV={rejected_hsv} -> AI={tracked_position}"
                                    )
'''

new = '''                                if tracked_position is not None:
                                    contact_local_ai = True

                                    # V34: this exact frame is a detector handoff, not a
                                    # second physical ball observation.  The normal HSV
                                    # candidate was already rolled back above; Local AI
                                    # may legitimately choose a different anchor around
                                    # racket contact.  Keep that AI anchor, but do not let
                                    # the anchor displacement become last_motion and fake
                                    # a ground-bounce reversal on the same frame.
                                    ai_anchor = tuple(tracked_position)
                                    ai_ball_size = self.ball_size
                                    ai_ball_hsv = self.ball_hsv
                                    if pre_track_snapshot is not None:
                                        if 'prev_motion' in pre_track_snapshot:
                                            prior_prev_motion = pre_track_snapshot.get('prev_motion')
                                            self.prev_motion = (
                                                dict(prior_prev_motion)
                                                if isinstance(prior_prev_motion, dict)
                                                else prior_prev_motion
                                            )
                                        if 'last_motion' in pre_track_snapshot:
                                            prior_last_motion = pre_track_snapshot.get('last_motion')
                                            self.last_motion = (
                                                dict(prior_last_motion)
                                                if isinstance(prior_last_motion, dict)
                                                else prior_last_motion
                                            )
                                        if 'last_nonzero_motion' in pre_track_snapshot:
                                            prior_nonzero = pre_track_snapshot.get('last_nonzero_motion')
                                            self.last_nonzero_motion = (
                                                dict(prior_nonzero)
                                                if isinstance(prior_nonzero, dict)
                                                else prior_nonzero
                                            )
                                        if 'last_delta' in pre_track_snapshot:
                                            self.last_delta = pre_track_snapshot.get('last_delta')
                                        if 'last_direction' in pre_track_snapshot:
                                            self.last_direction = pre_track_snapshot.get('last_direction')
                                        if 'ball_velocity_history' in pre_track_snapshot:
                                            self.ball_velocity_history = list(
                                                pre_track_snapshot.get('ball_velocity_history') or []
                                            )[-5:]
                                        if 'direction_change_streak' in pre_track_snapshot:
                                            self.direction_change_streak = int(
                                                pre_track_snapshot.get('direction_change_streak') or 0
                                            )

                                    # Re-assert the accepted AI observation after the
                                    # motion-state repair.  Contact Local-AI keeps its own
                                    # validated history, so the next frame can still follow
                                    # the outgoing trajectory from this anchor.
                                    self.ball_center = ai_anchor
                                    self.ball_size = ai_ball_size
                                    self.ball_hsv = ai_ball_hsv
                                    self.last_seen_frame = self.frame_count
                                    self.stuck_frame_count = 0
                                    self._pending_rally_end_reason = None
                                    self._pending_rally_end_frame = -1
                                    if frame is not None:
                                        try:
                                            self._prev_frame_gray = cv2.cvtColor(
                                                frame, cv2.COLOR_BGR2GRAY
                                            )
                                        except cv2.error:
                                            pass
                                    self._contact_local_ai_anchor_handoff_frame = int(
                                        self.frame_count
                                    )
                                    prior_motion = self.last_motion or {}
                                    print(
                                        f"[CONTACT_LOCAL_AI_ANCHOR_HANDOFF] f{self.frame_count}: "
                                        f"anchor={ai_anchor} preserving prior motion="
                                        f"({float(prior_motion.get('dx', 0.0) or 0.0):.1f},"
                                        f"{float(prior_motion.get('dy', 0.0) or 0.0):.1f}) "
                                        f"vel_hist={[round(float(v), 1) for v in self.ball_velocity_history[-5:]]}"
                                    )
                                    print(
                                        f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "
                                        f"HSV={rejected_hsv} -> AI={tracked_position}"
                                    )
'''

if old not in text:
    if '[CONTACT_LOCAL_AI_ANCHOR_HANDOFF]' in text:
        print('V34 contact Local-AI anchor handoff already applied')
        raise SystemExit(0)
    raise SystemExit('V34 target block not found')

text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
compile(text, str(path), 'exec')
print('Applied V34 contact Local-AI anchor handoff')
