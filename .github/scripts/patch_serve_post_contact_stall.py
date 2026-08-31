from pathlib import Path

path = Path('interactive_ball_analyzer.py')
text = path.read_text(encoding='utf-8')

old_motion = """                if (\n                        motion is not None and motion_mean < 5.0 and motion_max < 25.0 and\n                        # A genuine ball may have weak frame-difference energy\n                        # when it is only a few pixels wide, but it should\n                        # still be a short step from the last ball position.\n                        # Do not admit a distant static court highlight.\n                        (source != 'regular' or distance > 65.0)\n                ):\n                    continue\n"""
new_motion = """                weak_static_motion = (\n                    motion is not None and motion_mean < 5.0 and motion_max < 25.0\n                )\n                if weak_static_motion and (\n                        # During an established serve flight a real ball must\n                        # produce local frame-to-frame change.  Previously a\n                        # nearby regular-mask court highlight was allowed to\n                        # survive solely because it was within 65 px, which\n                        # could hand the post-contact flight to a static blob.\n                        upward_flight_continuation or\n                        lower_contact_flight or\n                        source != 'regular' or\n                        distance > 65.0\n                ):\n                    continue\n"""

old_hold = """                    elif reference_target_hold:\n                        self.stuck_frame_count = 14\n                        print(\n                            f\"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] \"\n                            f\"suppressing stuck timeout until f{reference_target_frame}\"\n                        )\n"""
new_hold = """                    elif reference_target_hold and self.stuck_frame_count >= 15:\n                        # A reviewed/reference endpoint may keep the point alive,\n                        # but it must not erase evidence that tracking is stuck.\n                        # Preserve the stuck count so the next frame can invoke\n                        # held-position Local AI / reacquisition instead of\n                        # repeatedly pinning a stale marker at count 14.\n                        print(\n                            f\"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] \"\n                            f\"suppressing point-end timeout until f{reference_target_frame}; \"\n                            f\"preserving stuck={self.stuck_frame_count} for recovery\"\n                        )\n"""

for label, old, new in [
    ('startup weak-motion gate', old_motion, new_motion),
    ('reference hold stuck preservation', old_hold, new_hold),
]:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected exactly 1 match, found {count}')
    text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
print('Patched post-serve static recovery and reference hold behavior')
