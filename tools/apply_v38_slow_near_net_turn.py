from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old = '''                        direct_contact_zone = core_net_zone or net_geometry['near_top_tape']
                        last_change_point = getattr(self, '_last_direction_change_point', None)
                        last_change_geometry = (
                            self._net_contact_geometry(last_change_point)
                            if last_change_point is not None else None
                        )
                        recent_net_turn = (
                            getattr(self, '_last_direction_change_frame', -1000000) == self.frame_count and
                            last_change_geometry is not None and
                            (last_change_geometry['inside'] or last_change_geometry['near_top_tape']) and
                            curr_dy >= 4.0 and
                            direct_contact_zone
                        )
                        if recent_net_turn:
                            return True, "Ball hit the net"
'''

new = '''                        direct_contact_zone = core_net_zone or net_geometry['near_top_tape']
                        last_change_point = getattr(self, '_last_direction_change_point', None)
                        last_change_geometry = (
                            self._net_contact_geometry(last_change_point)
                            if last_change_point is not None else None
                        )
                        recent_net_turn = (
                            getattr(self, '_last_direction_change_frame', -1000000) == self.frame_count and
                            last_change_geometry is not None and
                            (last_change_geometry['inside'] or last_change_geometry['near_top_tape']) and
                            curr_dy >= 4.0 and
                            direct_contact_zone
                        )

                        # V38: V37's repaired trajectory reaches the net correctly, but
                        # its low-speed f322 turn can sit just outside the strict net
                        # polygon/top-tape gate.  Accept only this very narrow physical
                        # signature: the ball was crawling immediately before a sharp
                        # downward turn, both the live point and the direction-change
                        # point are still within the calibrated near-net polygon, and
                        # the current step remains slow.  The surrounding branch already
                        # excludes recent ground/serve bounces, so ordinary near-net
                        # court bounces keep their existing classification.
                        v38_slow_near_net_turn = (
                            getattr(self, '_last_direction_change_frame', -1000000) == self.frame_count and
                            last_change_geometry is not None and
                            net_geometry.get('near_polygon', False) and
                            last_change_geometry.get('near_polygon', False) and
                            not direct_contact_zone and
                            prev_dist <= 5.0 and
                            curr_dist <= 15.0 and
                            curr_dy >= 4.0 and
                            angle_diff >= 100.0
                        )
                        if recent_net_turn or v38_slow_near_net_turn:
                            if v38_slow_near_net_turn and not recent_net_turn:
                                print(
                                    f"Frame {self.frame_count}: [V38 SLOW NEAR-NET TURN] "
                                    f"prev_speed={prev_dist:.1f}px curr_speed={curr_dist:.1f}px "
                                    f"curr_dy={curr_dy:.1f}px angle_diff={angle_diff:.1f} "
                                    f"signed_dist={net_geometry.get('signed_dist', 0.0):.1f}px "
                                    f"turn_signed_dist={last_change_geometry.get('signed_dist', 0.0):.1f}px"
                                )
                            return True, "Ball hit the net"
'''

if old not in text:
    if '[V38 SLOW NEAR-NET TURN]' in text:
        print('V38 already applied')
        compile(text, str(path), 'exec')
        raise SystemExit(0)
    raise SystemExit('V38 target block not found')

text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
compile(text, str(path), 'exec')
print('Applied V38 slow near-net turn guard')
