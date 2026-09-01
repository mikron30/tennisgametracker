from pathlib import Path

path = Path('interactive_ball_analyzer.py')
s = path.read_text(encoding='utf-8')


def replace_once(old, new, label):
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 match, found {count}')
    s = s.replace(old, new, 1)

# 1) A near-player zone opens a short watch window. It is NOT by itself a takeover.
replace_once(
'''        if (\n                getattr(self, 'local_ai_recovery', None) is None or\n                previous_position is None or tracked_position is None or\n                int(self.frame_count) <= int(getattr(\n                    self, '_contact_local_ai_cooldown_until_frame', -1000000\n                ))):\n            return None\n\n        previous = (int(previous_position[0]), int(previous_position[1]))\n        tracked = (int(tracked_position[0]), int(tracked_position[1]))\n''',
'''        current = int(self.frame_count)\n        if (\n                getattr(self, 'local_ai_recovery', None) is None or\n                previous_position is None or\n                current <= int(getattr(\n                    self, '_contact_local_ai_cooldown_until_frame', -1000000\n                ))):\n            return None\n\n        previous = (int(previous_position[0]), int(previous_position[1]))\n        watch_until = int(getattr(\n            self, '_contact_local_ai_watch_until_frame', -1000000\n        ))\n        if tracked_position is None:\n            if current <= watch_until:\n                return 'contact-loss-after-player-proximity'\n            return None\n        tracked = (int(tracked_position[0]), int(tracked_position[1]))\n''',
'contact trigger header')

replace_once(
'''        if zone is None and not contact_near:\n            return None\n\n        recent_speeds = [\n''',
'''        watch_active = current <= watch_until\n        if zone is not None or contact_near:\n            self._contact_local_ai_watch_until_frame = max(watch_until, current + 8)\n            watch_active = True\n        if zone is None and not contact_near and not watch_active:\n            return None\n\n        recent_speeds = [\n''',
'contact watch window')

replace_once(
'''        if zone in player_zones and max(prior_speed, proposed_speed) >= 12.0:\n            return (\n                f'player-zone:{zone}/step={proposed_speed:.0f}px/'\n                f'angle={angle_delta:.0f}deg'\n            )\n''',
'''        if (\n                zone in player_zones and max(prior_speed, proposed_speed) >= 12.0 and\n                prediction_error >= 30.0 and\n                (angle_delta >= 30.0 or size_ratio >= 2.0)):\n            return (\n                f'contact-player-artifact:{zone}/step={proposed_speed:.0f}px/'\n                f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'\n            )\n''',
'player zone no longer unconditional')

# The watch remains valid briefly after the ball leaves the player box.
s = s.replace('                contact_near and proposed_speed >= jump_floor and\n',
              '                (contact_near or watch_active) and proposed_speed >= jump_floor and\n', 1)
s = s.replace('                contact_near and prior_speed >= 18.0 and proposed_speed >= 18.0 and\n',
              '                (contact_near or watch_active) and prior_speed >= 18.0 and proposed_speed >= 18.0 and\n', 1)
s = s.replace('                contact_near and proposed_speed >= 35.0 and size_ratio >= 2.6 and\n',
              '                (contact_near or watch_active) and proposed_speed >= 35.0 and size_ratio >= 2.6 and\n', 1)

replace_once(
'''        if (\n                (contact_near or watch_active) and proposed_speed >= 35.0 and size_ratio >= 2.6 and\n                prediction_error >= 35.0):\n''',
'''        if (\n                (contact_near or watch_active) and prior_speed >= 18.0 and\n                proposed_speed <= max(6.0, prior_speed * 0.35) and\n                prediction_error >= 18.0):\n            return (\n                f'contact-stall:{prior_speed:.0f}->{proposed_speed:.0f}px/'\n                f'pred={prediction_error:.0f}px'\n            )\n\n        if (\n                (contact_near or watch_active) and proposed_speed >= 35.0 and size_ratio >= 2.6 and\n                prediction_error >= 35.0):\n''',
'contact stall trigger')

# 2) During contact assist, score more than the nearest 128 candidates.
replace_once(
'''        subset = clear_candidates[:128]\n        scored = recovery._score(frame, current, subset) if subset else []\n''',
'''        max_scored = max(32, min(512, int(getattr(\n            self, "_force_local_ai_max_score_candidates", 128\n        ))))\n        subset = clear_candidates[:max_scored]\n        scored = recovery._score(frame, current, subset) if subset else []\n''',
'force AI candidate pool')

# 3) Automatic contact mode must return None on a miss instead of freezing at anchor.
replace_once(
'''        if not eligible:\n            self.ball_center = tuple(anchor)\n            self.stuck_frame_count = max(1, int(getattr(self, "stuck_frame_count", 0)) + 1)\n            print(\n                f"[FORCE_LOCAL_AI_HOLD] f{current}: no clear candidate >= {min_score:.3f}; "\n                f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "\n                f"clear={len(clear_candidates)} scored={len(scored)}"\n            )\n            return tuple(anchor)\n''',
'''        if not eligible:\n            if bool(getattr(self, "_force_local_ai_return_none_on_miss", False)):\n                print(\n                    f"[FORCE_LOCAL_AI_MISS] f{current}: no clear candidate >= {min_score:.3f}; "\n                    f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "\n                    f"clear={len(clear_candidates)} scored={len(scored)}"\n                )\n                return None\n            self.ball_center = tuple(anchor)\n            self.stuck_frame_count = max(1, int(getattr(self, "stuck_frame_count", 0)) + 1)\n            print(\n                f"[FORCE_LOCAL_AI_HOLD] f{current}: no clear candidate >= {min_score:.3f}; "\n                f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "\n                f"clear={len(clear_candidates)} scored={len(scored)}"\n            )\n            return tuple(anchor)\n''',
'force AI no-freeze miss')

# 4) Contact owner sets non-blocking mode, wider pool, and expands radius after misses.
replace_once(
'''                'outside_count': 0,\n                'reason': str(arm_reason),\n''',
'''                'outside_count': 0,\n                'miss_count': 0,\n                'reason': str(arm_reason),\n''',
'contact state miss counter')

replace_once(
'''        saved_min_score = float(getattr(self, '_force_local_ai_min_score', 0.985))\n        try:\n''',
'''        saved_min_score = float(getattr(self, '_force_local_ai_min_score', 0.985))\n        saved_return_none = bool(getattr(self, '_force_local_ai_return_none_on_miss', False))\n        saved_max_scored = int(getattr(self, '_force_local_ai_max_score_candidates', 128))\n        try:\n''',
'contact save assist settings')

replace_once(
'''            self._force_local_ai_radius = float(\n                getattr(self, '_contact_local_ai_radius', 450.0)\n            )\n            self._force_local_ai_min_score = float(\n                getattr(self, '_contact_local_ai_min_score', 0.985)\n            )\n            selected = self._force_local_ai_frame(frame, previous_position)\n''',
'''            miss_count = int(state.get('miss_count', 0))\n            base_radius = float(getattr(self, '_contact_local_ai_radius', 450.0))\n            self._force_local_ai_radius = min(900.0, base_radius + 90.0 * miss_count)\n            self._force_local_ai_min_score = float(\n                getattr(self, '_contact_local_ai_min_score', 0.985)\n            )\n            self._force_local_ai_return_none_on_miss = True\n            self._force_local_ai_max_score_candidates = 384\n            selected = self._force_local_ai_frame(frame, previous_position)\n''',
'contact dynamic search')

replace_once(
'''            self._force_local_ai_radius = saved_radius\n            self._force_local_ai_min_score = saved_min_score\n\n        if selected is None:\n            return None\n\n        selected = (int(selected[0]), int(selected[1]))\n''',
'''            self._force_local_ai_radius = saved_radius\n            self._force_local_ai_min_score = saved_min_score\n            self._force_local_ai_return_none_on_miss = saved_return_none\n            self._force_local_ai_max_score_candidates = saved_max_scored\n\n        if selected is None:\n            state['miss_count'] = int(state.get('miss_count', 0)) + 1\n            self._contact_local_ai_state = state\n            print(\n                f"[CONTACT_LOCAL_AI_MISS] f{current}: normal tracker remains active; "\n                f"misses={state['miss_count']}"\n            )\n            return None\n\n        state['miss_count'] = 0\n        selected = (int(selected[0]), int(selected[1]))\n''',
'contact miss fallback state')

# 5) If the first AI attempt misses, restore the normal frame result instead of holding.
replace_once(
'''                                rejected_hsv = (\n                                    tuple(tracked_position)\n                                    if tracked_position is not None else None\n                                )\n                                self._restore_tracking_state_for_provisional_guard(\n                                    pre_track_snapshot\n                                )\n''',
'''                                rejected_hsv = (\n                                    tuple(tracked_position)\n                                    if tracked_position is not None else None\n                                )\n                                post_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()\n                                self._restore_tracking_state_for_provisional_guard(\n                                    pre_track_snapshot\n                                )\n''',
'post HSV snapshot')

replace_once(
'''                                else:\n                                    # A failed AI arm must not leave state partially\n                                    # rolled back.  Hold the last trusted ball and retry\n                                    # on the next frame rather than accepting player HSV.\n                                    self.ball_center = tuple(prev_ball_center)\n                                    self.stuck_frame_count = max(\n                                        1, int(prev_stuck) + 1\n                                    )\n                                    tracked_position = tuple(prev_ball_center)\n                                    contact_local_ai = True\n''',
'''                                else:\n                                    # AI is an assist: if it has no accepted candidate,\n                                    # keep the normal tracker result for this frame.\n                                    self._restore_tracking_state_for_provisional_guard(\n                                        post_track_snapshot\n                                    )\n                                    tracked_position = rejected_hsv\n                                    contact_local_ai = False\n                                    print(\n                                        f"[CONTACT_LOCAL_AI_FALLBACK] f{self.frame_count}: "\n                                        f"AI miss; keeping normal={tracked_position}"\n                                    )\n''',
'failed contact arm no freeze')

path.write_text(s, encoding='utf-8')
print('PATCH_APPLIED')
