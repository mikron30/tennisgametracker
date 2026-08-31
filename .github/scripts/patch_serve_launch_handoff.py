from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old_call = '''                if serve_launch_meta is None:\n                    serve_launch_meta = self._prefer_serve_contact_launch_candidate(\n                        candidate_meta, frame.shape\n                    )\n'''
new_call = '''                # Once a serve launch has already produced a verified forward/upward\n                # motion, hand control back to ordinary trajectory continuation.  Keeping\n                # the broad contact-launch selector active for every grace frame lets a\n                # distant moving player/background blob repeatedly replace an established\n                # airborne path.  The broad selector remains available when the launch has\n                # not been established yet, or when the last motion no longer proves the\n                # forward/upward flight.\n                launch_already_airborne = (\n                    int(getattr(self, '_serve_launch_direction_x', 0) or 0) != 0 and\n                    self._is_forward_serve_launch_motion(self.last_motion)\n                )\n                if serve_launch_meta is None and not launch_already_airborne:\n                    serve_launch_meta = self._prefer_serve_contact_launch_candidate(\n                        candidate_meta, frame.shape\n                    )\n                elif serve_launch_meta is None and launch_already_airborne:\n                    _verbose_debug_print(\n                        f"  DEBUG: [SERVE-LAUNCH HANDOFF] established flight; "\n                        f"using predicted continuation from {self.ball_center}"\n                    )\n'''
if old_call not in text:
    raise SystemExit('serve launch callsite not found')
text = text.replace(old_call, new_call, 1)

old_top = '''        if strong_upper_exit and (\n                self.ground_bounce_count > 0 or\n                ball_size >= 80.0 or\n                recent_vel >= 120.0):\n            return "upper_side"\n'''
new_top = '''        if strong_upper_exit and (\n                self.ground_bounce_count > 0 or\n                ball_size >= 80.0 or\n                recent_vel >= 120.0):\n            # During the initial serve flight, the far-court/receiver band overlaps\n            # the broad "upper side" heuristic.  A single fast tracked step there is\n            # not proof that the ball has left the image.  Keep ordinary tracking for\n            # one more frame; if the ball is genuinely gone, the resulting miss/stuck\n            # evidence can arm the delayed top-return search on the next frame.  True\n            # literal top-edge exits (handled by the stricter branches above) remain\n            # immediate.\n            pre_bounce_serve_flight = (\n                getattr(self, '_serve_contact_grace_frames', 0) > 0 and\n                self.ground_bounce_count == 0\n            )\n            if pre_bounce_serve_flight and getattr(self, 'stuck_frame_count', 0) == 0:\n                _verbose_debug_print(\n                    f"  DEBUG: [TOP-RETURN DEFER] pre-bounce serve flight at "\n                    f"{self.ball_center}; waiting for miss evidence"\n                )\n                return None\n            return "upper_side"\n'''
if old_top not in text:
    raise SystemExit('strong upper exit block not found')
text = text.replace(old_top, new_top, 1)

path.write_text(text, encoding='utf-8')
print('Applied serve launch handoff + pre-bounce top-return defer patch')
