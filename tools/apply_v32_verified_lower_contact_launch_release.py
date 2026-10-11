from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")

release_log = "[CONTACT_LOCAL_AI VERIFIED-LOWER-LAUNCH RELEASE]"
if release_log in text:
    print("V32 already applied")
    raise SystemExit(0)

old_commit = """        recovery_label = candidate.get('recovery_label', 'NIGHT VISIBLE BALL RECOVER')
        print(
            f\"Frame {self.frame_count}: [{recovery_label}] Ball at {new_pos} \"
"""
new_commit = """        recovery_label = candidate.get('recovery_label', 'NIGHT VISIBLE BALL RECOVER')
        # V32: remember an explicit normal-path lower-contact launch for this
        # exact frame. Contact Local-AI may still inspect ordinary suspicious
        # HSV proposals, but it must not roll back a launch already verified by
        # the dedicated lower-contact continuation geometry.
        if (
                recovery_label == 'NIGHT LOWER CONTACT LAUNCH' and
                candidate.get('source') != 'local_ai_tight_roi'
        ):
            self._last_verified_lower_contact_launch_frame = int(self.frame_count)
        print(
            f\"Frame {self.frame_count}: [{recovery_label}] Ball at {new_pos} \"
"""
if text.count(old_commit) != 1:
    raise SystemExit(
        f"expected exactly one recovery-label commit block, found {text.count(old_commit)}"
    )
text = text.replace(old_commit, new_commit, 1)

old_arb = """                            tracked_position = self.track_ball_in_frame(frame)
                            contact_reason = self._contact_local_ai_trigger(
                                prev_ball_center, tracked_position, pre_track_snapshot
                            )
                            if contact_reason is not None:
"""
new_arb = """                            tracked_position = self.track_ball_in_frame(frame)
                            verified_lower_contact_launch = (
                                tracked_position is not None and
                                int(getattr(
                                    self,
                                    '_last_verified_lower_contact_launch_frame',
                                    -1000000,
                                )) == int(self.frame_count)
                            )
                            if verified_lower_contact_launch:
                                # A dedicated NIGHT LOWER CONTACT LAUNCH has already
                                # committed a geometrically verified outbound ball.
                                # Do not let a Contact Local-AI miss restore the stale
                                # pre-contact anchor on this exact frame.
                                self._contact_local_ai_state = None
                                self._contact_local_ai_cooldown_until_frame = max(
                                    int(getattr(
                                        self,
                                        '_contact_local_ai_cooldown_until_frame',
                                        -1000000,
                                    )),
                                    int(self.frame_count) + 2,
                                )
                                self._contact_normal_fallback_path = []
                                contact_reason = None
                                print(
                                    f\"[CONTACT_LOCAL_AI VERIFIED-LOWER-LAUNCH RELEASE] \"
                                    f\"f{self.frame_count}: preserving normal launch=\"
                                    f\"{tuple(tracked_position)}\"
                                )
                            else:
                                contact_reason = self._contact_local_ai_trigger(
                                    prev_ball_center, tracked_position, pre_track_snapshot
                                )
                            if contact_reason is not None:
"""
if text.count(old_arb) != 1:
    raise SystemExit(
        f"expected exactly one Contact Local-AI arbitration block, found {text.count(old_arb)}"
    )
text = text.replace(old_arb, new_arb, 1)

path.write_text(text, encoding="utf-8")
print("Applied V32 verified lower-contact launch release")
