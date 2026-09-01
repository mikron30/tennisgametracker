from pathlib import Path

p = Path('interactive_ball_analyzer.py')
s = p.read_text(encoding='utf-8')

out_marker = '        if "bounce outside singles court" in reason_lower or "bounced out of court" in reason_lower:\n'
start = s.find(out_marker)
if start < 0:
    raise SystemExit('out scoring block not found')

# Stay inside _infer_point_outcome: the next same-level reason branch closes
# this OUT branch.  We only replace the legacy precedence tail.
end_marker = '        if "bounced twice" in reason_lower'
end = s.find(end_marker, start)
if end < 0:
    raise SystemExit('end of out scoring block not found')
block = s[start:end]

legacy_landing = '            if landing_player is not None:\n                return self._point_outcome(landing_player, "ball out on player court; opponent fault", "out_error", 1 - landing_player)\n'
legacy_early = '''            if (
                self.point_start_frame_internal is not None and
                (self.frame_count - self.point_start_frame_internal) <= 45
            ):
                return self._point_outcome(receiver_idx, "early serve/rally out by server", "out_error", server_idx)
'''
legacy_contact = '            if recent_contact:\n                return self._point_outcome(1 - contact_player, "last hitter missed court", "out_error", contact_player)\n'

for label, needle in (
    ('landing fallback', legacy_landing),
    ('early fallback', legacy_early),
    ('recent-contact fallback', legacy_contact),
):
    if block.count(needle) != 1:
        raise SystemExit(f'{label}: expected 1 match in OUT block, found {block.count(needle)}')

# Preserve the special lower-right "unreturned" exception above this tail.
# For a normal OUT, recent verified hitter ownership is authoritative.
legacy_tail_start = block.index(legacy_landing)
legacy_tail_end = block.index(legacy_contact) + len(legacy_contact)
prefix = block[:legacy_tail_start]
suffix = block[legacy_tail_end:]

new_tail = '''            # For an OUT, the landing side tells us where the miss landed; it
            # does not tell us who made the error. A recent verified racket
            # contact is direct ownership evidence, so the last hitter loses.
            if recent_contact:
                winner_idx = 1 - contact_player
                landing_text = (
                    self.player_names[landing_player]
                    if landing_player in (0, 1) else "unknown"
                )
                print(
                    f"[OUT_SCORE_OWNER] f{self.frame_count}: "
                    f"last_hitter={self.player_names[contact_player]} "
                    f"contact=f{contact_frame} landing_side={landing_text} "
                    f"-> winner={self.player_names[winner_idx]}"
                )
                return self._point_outcome(
                    winner_idx,
                    "last hitter missed court",
                    "out_error",
                    contact_player,
                )
            if (
                self.point_start_frame_internal is not None and
                (self.frame_count - self.point_start_frame_internal) <= 45
            ):
                return self._point_outcome(
                    receiver_idx,
                    "early serve/rally out by server",
                    "out_error",
                    server_idx,
                )
            if landing_player is not None:
                return self._point_outcome(
                    landing_player,
                    "ball out; hitter unknown, landing-side fallback",
                    "out_error",
                    1 - landing_player,
                )
'''

new_block = prefix + new_tail + suffix
s = s[:start] + new_block + s[end:]
p.write_text(s, encoding='utf-8')
print('OUT_SCORE_LAST_HITTER_PATCH_APPLIED')
