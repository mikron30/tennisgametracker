from pathlib import Path

p = Path('interactive_ball_analyzer.py')
s = p.read_text(encoding='utf-8')

marker = '    def _ground_bounce_has_racket_below_contact(self, ground_bounce_context, frame):\n'
if marker not in s:
    raise SystemExit('helper insertion marker missing')

helper = '''    def _boundary_reversal_should_stay_ground_bounce(self, ground_bounce_context, new_pos, frame):
        """Return True when court geometry and vertical motion prove a boundary bounce.

        This runs before a visual racket/player hint is allowed to reinterpret a
        ground-bounce candidate as contact. It is deliberately geometry-based:
        no frame numbers, player coordinates, or clip-specific positions are used.
        """
        if ground_bounce_context is None or new_pos is None or frame is None:
            return False, None
        origin = ground_bounce_context.get('origin')
        if origin is None:
            return False, None

        incoming_dy = float(ground_bounce_context.get('incoming_dy', 0.0) or 0.0)
        incoming_dist = float(ground_bounce_context.get('incoming_dist', 0.0) or 0.0)
        outgoing_dx = float(new_pos[0]) - float(origin[0])
        outgoing_dy = float(new_pos[1]) - float(origin[1])
        outgoing_dist = math.hypot(outgoing_dx, outgoing_dy)
        frame_height = float(frame.shape[0])
        vertical_floor = max(4.0, frame_height * 0.0025)
        if incoming_dy < vertical_floor or outgoing_dy > -vertical_floor:
            return False, None
        if incoming_dist < 6.0 or outgoing_dist < 6.0:
            return False, None

        outside = False
        boundary = None
        try:
            outside, boundary, _, _, _, _ = self._point_outside_singles_court(origin, frame)
        except Exception:
            outside = False
        if not outside:
            try:
                outside_far, _ = self._point_outside_top_singles_baseline(origin, frame)
            except Exception:
                outside_far = False
            if outside_far:
                outside = True
                boundary = 'far_baseline'

        if not outside:
            return False, None

        debug = {
            'origin': tuple(origin),
            'new_pos': (int(new_pos[0]), int(new_pos[1])),
            'boundary': boundary,
            'incoming_dy': incoming_dy,
            'incoming_dist': incoming_dist,
            'outgoing_dy': outgoing_dy,
            'outgoing_dist': outgoing_dist,
        }
        return True, debug

'''

s = s.replace(marker, helper + marker, 1)

old = '''                        if racket_below_contact:
                            self._handle_racket_below_contact_instead_of_bounce(
                                ground_bounce_context,
                                frame,
                                (cx, cy),
                                bulb_size,
                                racket_below_debug,
                            )
                            print(f"Frame {self.frame_count}: Allowing racket-contact continuation")
                        else:
'''
new = '''                        boundary_reversal_override, boundary_reversal_debug = (
                            self._boundary_reversal_should_stay_ground_bounce(
                                ground_bounce_context, (cx, cy), frame
                            )
                        )
                        if racket_below_contact and boundary_reversal_override:
                            print(
                                f"Frame {self.frame_count}: [BOUNDARY-BEATS-RACKET] "
                                f"origin={boundary_reversal_debug.get('origin')} "
                                f"new_pos={boundary_reversal_debug.get('new_pos')} "
                                f"boundary={boundary_reversal_debug.get('boundary')} "
                                f"incoming_dy={boundary_reversal_debug.get('incoming_dy'):.1f} "
                                f"outgoing_dy={boundary_reversal_debug.get('outgoing_dy'):.1f}; "
                                f"keeping ground-bounce classification"
                            )
                            racket_below_contact = False
                        if racket_below_contact:
                            self._handle_racket_below_contact_instead_of_bounce(
                                ground_bounce_context,
                                frame,
                                (cx, cy),
                                bulb_size,
                                racket_below_debug,
                            )
                            print(f"Frame {self.frame_count}: Allowing racket-contact continuation")
                        else:
'''
if old not in s:
    raise SystemExit('contact branch marker missing')
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf-8')
print('patched boundary reversal precedence over racket inference')
