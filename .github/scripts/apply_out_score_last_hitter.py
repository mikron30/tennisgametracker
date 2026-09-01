from pathlib import Path

p = Path('interactive_ball_analyzer.py')
s = p.read_text(encoding='utf-8')

old = '''        if landing_player is not None:\n            return self._point_outcome(landing_player, "ball out on player court; opponent fault", "out_error", 1 - landing_player)\n        if (\n            self.point_start_frame_internal is not None and\n            (self.frame_count - self.point_start_frame_internal) <= 45\n        ):\n            return self._point_outcome(receiver_idx, "early serve/rally out by server", "out_error", server_idx)\n        if recent_contact:\n            return self._point_outcome(1 - contact_player, "last hitter missed court", "out_error", contact_player)\n'''

new = '''        # For an OUT, the landing side tells us *where* the miss landed; it does\n        # not tell us who made the error.  A recent verified racket contact is\n        # direct ownership evidence, so the last hitter must lose the point.\n        # Keep landing-side geometry only as a fallback when hitter ownership is\n        # genuinely unavailable.\n        if recent_contact:\n            winner_idx = 1 - contact_player\n            landing_text = (\n                self.player_names[landing_player]\n                if landing_player in (0, 1) else "unknown"\n            )\n            print(\n                f"[OUT_SCORE_OWNER] f{self.frame_count}: "\n                f"last_hitter={self.player_names[contact_player]} "\n                f"contact=f{contact_frame} landing_side={landing_text} "\n                f"-> winner={self.player_names[winner_idx]}"\n            )\n            return self._point_outcome(winner_idx, "last hitter missed court", "out_error", contact_player)\n        if (\n            self.point_start_frame_internal is not None and\n            (self.frame_count - self.point_start_frame_internal) <= 45\n        ):\n            return self._point_outcome(receiver_idx, "early serve/rally out by server", "out_error", server_idx)\n        if landing_player is not None:\n            return self._point_outcome(\n                landing_player,\n                "ball out; hitter unknown, landing-side fallback",\n                "out_error",\n                1 - landing_player,\n            )\n'''

count = s.count(old)
if count != 1:
    raise SystemExit(f'expected exactly one out-score block, found {count}')

s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')
print('OUT_SCORE_LAST_HITTER_PATCH_APPLIED')
