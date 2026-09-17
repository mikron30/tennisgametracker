from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old_init = '''                    forced_local_ai = False
                    contact_local_ai = False
                    tracked_position = self._force_local_ai_frame(frame, prev_ball_center)
'''
new_init = '''                    forced_local_ai = False
                    contact_local_ai = False
                    # V36: protect an explicitly verified NIGHT LOWER CONTACT LAUNCH
                    # from later same-frame post-track recovery arbitration.
                    verified_lower_contact_launch = False
                    tracked_position = self._force_local_ai_frame(frame, prev_ball_center)
'''

old_recovery = '''                    if not forced_local_ai and not contact_local_ai:
                        tracked_position = self._try_local_ai_recovery(
                            prev_ball_center, tracked_position, prev_stuck,
                            pre_track_snapshot=pre_track_snapshot,
                            frame=frame,
                        )
'''
new_recovery = '''                    if verified_lower_contact_launch:
                        print(
                            f"[VERIFIED-LOWER-LAUNCH POST-RECOVERY SKIP] "
                            f"f{self.frame_count}: keeping launch={tuple(tracked_position)}"
                        )
                    if (
                        not forced_local_ai and
                        not contact_local_ai and
                        not verified_lower_contact_launch
                    ):
                        tracked_position = self._try_local_ai_recovery(
                            prev_ball_center, tracked_position, prev_stuck,
                            pre_track_snapshot=pre_track_snapshot,
                            frame=frame,
                        )
'''

if old_init not in text:
    if 'verified_lower_contact_launch = False' not in text:
        raise SystemExit('V36 init target not found')
else:
    text = text.replace(old_init, new_init, 1)

if old_recovery not in text:
    if '[VERIFIED-LOWER-LAUNCH POST-RECOVERY SKIP]' in text:
        print('V36 already applied')
        path.write_text(text, encoding='utf-8')
        compile(text, str(path), 'exec')
        raise SystemExit(0)
    raise SystemExit('V36 recovery target not found')

text = text.replace(old_recovery, new_recovery, 1)
path.write_text(text, encoding='utf-8')
compile(text, str(path), 'exec')
print('Applied V36 verified lower launch post-recovery skip')
