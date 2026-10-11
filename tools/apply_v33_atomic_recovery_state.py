from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old = '''        self._maybe_clear_post_serve_pre_net_recovery(
            previous_position, repaired_position, source='local-ai'
        )
        self.ball_center = repaired_position
        self.ball_size = float(recovered.get("area", self.ball_size or 0.0))
        if recovered.get("trajectory_rescue"):
            trajectory_motion = dict(recovered.get("trajectory_motion") or {})
            prior_motion = (pre_track_snapshot or {}).get("last_motion")
            self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
            self.last_motion = trajectory_motion
            if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
                self.last_nonzero_motion = dict(trajectory_motion)
            prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
            prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
            self.ball_velocity_history = prior_history[-5:]
            self._held_direction_candidate = None
'''

new = '''        # V33: Local-AI recovery runs after the normal tracker has already
        # committed its candidate.  If AI replaces that candidate, repair the
        # *entire* live tracking state atomically.  Keeping only the repaired
        # coordinate leaves last_motion / velocity history from the rejected
        # contour and poisons the next-frame prediction.
        committed_position = (
            tuple(tracked_position) if tracked_position is not None else None
        )
        recovery_replaced_committed_candidate = (
            pre_track_snapshot is not None and
            previous_position is not None and
            committed_position is not None and
            committed_position != tuple(repaired_position)
        )
        recovered_area = float(recovered.get("area", self.ball_size or 0.0))

        if recovery_replaced_committed_candidate:
            self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
            self._maybe_clear_post_serve_pre_net_recovery(
                previous_position, repaired_position, source='local-ai'
            )

            repaired_dx = float(repaired_position[0]) - float(previous_position[0])
            repaired_dy = float(repaired_position[1]) - float(previous_position[1])
            repaired_distance = math.hypot(repaired_dx, repaired_dy)
            repaired_direction = (
                math.degrees(math.atan2(repaired_dy, repaired_dx))
                if repaired_distance > 0.0 else None
            )
            repaired_motion = {
                'distance': repaired_distance,
                'dx': repaired_dx,
                'dy': repaired_dy,
                'direction_deg': repaired_direction,
            }

            self.ball_center = repaired_position
            self.ball_size = recovered_area
            prior_motion = pre_track_snapshot.get("last_motion")
            self.prev_motion = (
                dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
            )
            self.last_motion = repaired_motion
            if repaired_distance >= 3.0:
                self.last_nonzero_motion = dict(repaired_motion)
            prior_history = list(pre_track_snapshot.get("ball_velocity_history") or [])
            prior_history.append(repaired_distance)
            self.ball_velocity_history = prior_history[-5:]
            self._held_direction_candidate = None

            # The rollback restores the prior gray frame as well.  For an
            # accepted current-frame repair, the next frame must compare against
            # this decoded frame, not against frame N-1 twice.
            if frame is not None:
                try:
                    self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except cv2.error:
                    pass

            if (
                getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
                getattr(self, '_last_motion_reacq_pos', None) == committed_position
            ):
                self._last_motion_reacq_frame = -1000000
                self._last_motion_reacq_pos = None

            # Replace a same-frame motion-history sample from the rejected
            # contour rather than leaving two contradictory versions of frame N.
            if self.motion_history and int(self.motion_history[-1].get('frame', -1)) == int(self.frame_count):
                self.motion_history[-1] = {
                    'frame': self.frame_count,
                    'distance': repaired_distance,
                    'direction_deg': repaired_direction,
                    'pos': tuple(repaired_position),
                    'prev_pos': tuple(previous_position),
                }

            print(
                f"[POST_TRACK_STATE_REPAIR] f{self.frame_count}: "
                f"committed={committed_position} -> repaired={repaired_position} "
                f"motion={repaired_distance:.1f}px "
                f"velocity_history={self.ball_velocity_history}"
            )
        else:
            self._maybe_clear_post_serve_pre_net_recovery(
                previous_position, repaired_position, source='local-ai'
            )
            self.ball_center = repaired_position
            self.ball_size = recovered_area
            if recovered.get("trajectory_rescue"):
                trajectory_motion = dict(recovered.get("trajectory_motion") or {})
                prior_motion = (pre_track_snapshot or {}).get("last_motion")
                self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
                self.last_motion = trajectory_motion
                if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
                    self.last_nonzero_motion = dict(trajectory_motion)
                prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
                prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
                self.ball_velocity_history = prior_history[-5:]
                self._held_direction_candidate = None
'''

if old not in text:
    if '[POST_TRACK_STATE_REPAIR]' in text:
        print('V33 atomic recovery state repair already applied')
        raise SystemExit(0)
    raise SystemExit('V33 target block not found')

text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
compile(text, str(path), 'exec')
print('Applied V33 atomic post-track recovery state repair')
