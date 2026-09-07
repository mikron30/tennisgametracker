from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')
start = text.index('    def _upper_racket_contact_turn_candidate_ok(')
end = text.index('    def _upper_slow_arc_active(', start)
new = '''    def _upper_racket_contact_turn_candidate_ok(
        self,
        pos,
        area,
        frame_shape,
        dx,
        dy,
        angle_jump,
        speed_ratio,
        motion_mean=None,
        motion_max=None,
    ):
        """Recognize a far-player racket return after an upper-court stall.

        The original path handles a stalled apex that accelerates downward
        after contact. A second, stricter path handles a lateral racket return:
        on the opponent's upper-court side, a nearly stalled ball reverses its
        horizontal motion and accelerates sharply. That pattern can be a volley,
        so it must not depend on an earlier ground bounce.
        """
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False
        if getattr(self, '_awaiting_serve_bounce', False):
            return False
        if getattr(self, 'direction_change_streak', 0) < 2:
            return False

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        prev_size = float(self.ball_size)
        prev_dx = float(self.last_motion.get('dx', 0.0))
        prev_dy = float(self.last_motion.get('dy', 0.0))
        prev_speed = float(self.last_motion.get('distance', 0.0))
        cx, cy = pos
        turn_dist = math.hypot(dx, dy)

        if origin_y < max(150, int(frame_height * 0.07)):
            return False
        if origin_y > max(360, int(frame_height * 0.18)):
            return False
        if cy > max(430, int(frame_height * 0.22)):
            return False
        if prev_size > 45.0 or area > 60.0:
            return False
        if prev_dy > 4.0 or prev_speed > 10.0:
            return False

        point_start = getattr(self, 'point_start_frame_internal', None)
        last_contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        last_contact_player = getattr(self, '_last_racket_contact_player', None)
        candidate_player = self._player_index_at_point(self.ball_center)
        active_previous_hit = (
            point_start is not None and
            last_contact_frame >= int(point_start) and
            last_contact_player in (0, 1)
        )
        opponent_contact_side = (
            active_previous_hit and
            candidate_player in (0, 1) and
            candidate_player != last_contact_player
        )

        # Gravity can reverse vertical motion at an apex, but cannot reverse
        # meaningful horizontal motion and multiply speed at the same instant.
        # Restrict that signature to the opponent's court end so it represents
        # the only player who could have returned the previous hitter's ball.
        lateral_upper_racket_reversal = (
            opponent_contact_side and
            abs(prev_dx) >= 2.0 and
            abs(dx) >= 12.0 and
            (prev_dx * float(dx)) <= -24.0 and
            prev_speed <= 8.0 and
            abs(prev_dy) <= 4.0 and
            speed_ratio >= 3.0 and
            angle_jump >= 50.0 and
            15.0 <= turn_dist <= max(75.0, prev_speed * 7.0) and
            abs(float(dy)) <= max(65.0, frame_height * 0.035)
        )
        if lateral_upper_racket_reversal:
            print(
                f"Frame {self.frame_count}: [UPPER LATERAL RACKET RETURN] "
                f"previous_hitter={self.player_names[last_contact_player]} "
                f"contact_player={self.player_names[candidate_player]} "
                f"origin={self.ball_center} new_pos={pos} "
                f"motion=({prev_dx:.1f},{prev_dy:.1f})->({float(dx):.1f},{float(dy):.1f}) "
                f"speed={prev_speed:.1f}x{speed_ratio:.1f} angle={angle_jump:.1f}"
            )
            return True

        # Preserve the older, softer apex/downward-return signature. Because it
        # lacks the strong horizontal reversal above, retain its bounce gate.
        if self.ground_bounce_count <= 0:
            return False
        if dy < max(6.0, frame_height * 0.0035):
            return False
        recent_serve_return_window = (
            self.ground_bounce_count == 1 and
            (self.frame_count - getattr(self, '_last_serve_bounce_frame', -1000000)) <= 45
        )
        min_turn_angle = 65.0 if recent_serve_return_window else 70.0
        if angle_jump < min_turn_angle:
            return False
        if speed_ratio < 1.55:
            return False
        if turn_dist < 8.0 or turn_dist > max(60.0, prev_speed * 5.0):
            return False
        if abs(dx) > max(90.0, frame_width * 0.03):
            return False
        if motion_mean is not None and motion_max is not None:
            if motion_mean < 12.0 and motion_max < 70.0:
                return False
        return True

'''
path.write_text(text[:start] + new + text[end:], encoding='utf-8')
print('far-player lateral hit patch applied')
