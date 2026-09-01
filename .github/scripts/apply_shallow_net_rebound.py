from pathlib import Path

p = Path('interactive_ball_analyzer.py')
s = p.read_text(encoding='utf-8')


def one(old, new, label):
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 match, found {count}')
    s = s.replace(old, new, 1)

# Preserve the existing high-energy bounce path, but do not return before a
# low-energy net-deflected rebound gets a chance to be classified.
one(
'''        curr_speed = float(self.last_motion.get('distance', 0.0) or 0.0)\n        prev_speed = float(self.prev_motion.get('distance', 0.0) or 0.0)\n        if prev_speed < 18.0 or curr_speed < 10.0:\n            return False, None\n\n        x, y = ball_position\n''',
'''        curr_speed = float(self.last_motion.get('distance', 0.0) or 0.0)\n        prev_speed = float(self.prev_motion.get('distance', 0.0) or 0.0)\n        strong_bounce_motion = prev_speed >= 18.0 and curr_speed >= 10.0\n\n        x, y = ball_position\n''',
'same-side speed gate')

one(
'''        vertical_reversal = prev_dy >= 18.0 and curr_dy <= -10.0\n        sharp_turn = angle_diff >= 120.0 and prev_dy >= 12.0 and curr_dy <= -6.0\n        if not (vertical_reversal or sharp_turn):\n            return False, None\n\n        outside, _, left_x, right_x = self._point_outside_singles_sidelines(bounce_point, frame)\n''',
'''        vertical_reversal = prev_dy >= 18.0 and curr_dy <= -10.0\n        sharp_turn = angle_diff >= 120.0 and prev_dy >= 12.0 and curr_dy <= -6.0\n\n        # A ball that clips the net can lose most of its speed before bouncing\n        # back on the hitter side.  Keep this softer signature tightly scoped\n        # to the hitter-side strip immediately next to the net so ordinary\n        # low-speed direction changes elsewhere on court are unaffected.\n        if contact_side == "near":\n            net_rebound_depth = max(0.0, bounce_y - net_max)\n        else:\n            net_rebound_depth = max(0.0, net_min - bounce_y)\n        near_net_rebound = net_rebound_depth <= max(150.0, height * 0.085)\n        shallow_net_rebound = (\n            near_net_rebound and\n            prev_speed >= 5.0 and curr_speed >= 6.0 and\n            prev_dy >= 4.0 and curr_dy <= -4.0 and\n            angle_diff >= 80.0\n        )\n        strong_reversal = strong_bounce_motion and (vertical_reversal or sharp_turn)\n        if not (strong_reversal or shallow_net_rebound):\n            return False, None\n        bounce_mode = "shallow-net-rebound" if shallow_net_rebound else "strong-reversal"\n\n        outside, _, left_x, right_x = self._point_outside_singles_sidelines(bounce_point, frame)\n''',
'shallow net rebound classifier')

one(
'''            f"Frame {self.frame_count}: [SAME-SIDE PRE-NET BOUNCE] "\n            f"bounce_point={bounce_point} contact={contact_point} side={contact_side} "\n            f"frames_since_contact={frames_since_contact} prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "\n            f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) angle_diff={angle_diff:.1f}"\n            f"{crossing_text}"\n''',
'''            f"Frame {self.frame_count}: [SAME-SIDE PRE-NET BOUNCE] "\n            f"bounce_point={bounce_point} contact={contact_point} side={contact_side} "\n            f"mode={bounce_mode} net_depth={net_rebound_depth:.1f}px "\n            f"frames_since_contact={frames_since_contact} prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "\n            f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) angle_diff={angle_diff:.1f}"\n            f"{crossing_text}"\n''',
'same-side bounce log mode')

p.write_text(s, encoding='utf-8')
print('SHALLOW_NET_REBOUND_PATCH_APPLIED')
