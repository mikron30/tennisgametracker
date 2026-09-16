from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old = '''                    if verified_lower_contact_launch:
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

new = '''                    # V37: once NIGHT LOWER CONTACT LAUNCH has committed a real
                    # outbound ball, do not let the generic post-track Local-AI hold
                    # roll that anchor back on the launch frame or on a physically
                    # coherent immediate continuation.  This is deliberately narrow:
                    # only the next two frames are eligible, the motion must still be
                    # upward/netward, and it must agree with the previously committed
                    # motion vector.  A player/body jump or reversal therefore remains
                    # eligible for normal Local-AI arbitration.
                    lower_launch_frame = int(getattr(
                        self, '_last_verified_lower_contact_launch_frame', -1000000
                    ))
                    lower_launch_age = int(self.frame_count) - lower_launch_frame
                    verified_lower_contact_continuation = False
                    continuation_step = 0.0
                    continuation_cos = -1.0
                    if (
                        tracked_position is not None and
                        prev_ball_center is not None and
                        pre_track_snapshot is not None and
                        1 <= lower_launch_age <= 2
                    ):
                        continuation_dx = (
                            float(tracked_position[0]) - float(prev_ball_center[0])
                        )
                        continuation_dy = (
                            float(tracked_position[1]) - float(prev_ball_center[1])
                        )
                        continuation_step = math.hypot(
                            continuation_dx, continuation_dy
                        )
                        prior_motion = pre_track_snapshot.get('last_motion') or {}
                        prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
                        prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
                        prior_step = float(
                            prior_motion.get('distance', 0.0) or
                            math.hypot(prior_dx, prior_dy)
                        )
                        if continuation_step > 0.0 and prior_step > 0.0:
                            continuation_cos = (
                                continuation_dx * prior_dx +
                                continuation_dy * prior_dy
                            ) / (continuation_step * prior_step)
                        continuation_limit = max(
                            180.0,
                            min(320.0, prior_step * 2.2 + 60.0),
                        )
                        verified_lower_contact_continuation = (
                            12.0 <= continuation_step <= continuation_limit and
                            continuation_dy <= -12.0 and
                            prior_dy <= -8.0 and
                            continuation_cos >= 0.25
                        )

                    verified_lower_contact_post_guard = (
                        tracked_position is not None and
                        (
                            lower_launch_age == 0 or
                            verified_lower_contact_launch or
                            verified_lower_contact_continuation
                        )
                    )
                    if verified_lower_contact_post_guard:
                        guard_kind = (
                            'launch'
                            if lower_launch_age == 0 or verified_lower_contact_launch
                            else 'continuation'
                        )
                        print(
                            f"[VERIFIED-LOWER-CONTACT POST-RECOVERY SKIP] "
                            f"f{self.frame_count}: kind={guard_kind} age={lower_launch_age} "
                            f"keeping={tuple(tracked_position)} step={continuation_step:.1f}px "
                            f"cos={continuation_cos:.3f}"
                        )
                    if (
                        not forced_local_ai and
                        not contact_local_ai and
                        not verified_lower_contact_post_guard
                    ):
                        tracked_position = self._try_local_ai_recovery(
                            prev_ball_center, tracked_position, prev_stuck,
                            pre_track_snapshot=pre_track_snapshot,
                            frame=frame,
                        )
'''

if old not in text:
    if '[VERIFIED-LOWER-CONTACT POST-RECOVERY SKIP]' in text:
        print('V37 already applied')
        compile(text, str(path), 'exec')
        raise SystemExit(0)
    raise SystemExit('V37 target block not found')

text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
compile(text, str(path), 'exec')
print('Applied V37 verified lower-contact continuation guard')
