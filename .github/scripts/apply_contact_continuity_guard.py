from pathlib import Path

p = Path('interactive_ball_analyzer.py')
s = p.read_text(encoding='utf-8')


def one(old, new, label):
    global s
    c = s.count(old)
    if c != 1:
        raise SystemExit(f'{label}: expected 1 match, found {c}')
    s = s.replace(old, new, 1)

# Contact assist keeps trajectory continuity even when an AI frame misses and
# the normal tracker handles that frame.
one(
'''        saved_return_none = bool(getattr(self, '_force_local_ai_return_none_on_miss', False))
        saved_max_scored = int(getattr(self, '_force_local_ai_max_score_candidates', 128))
        try:
''',
'''        saved_return_none = bool(getattr(self, '_force_local_ai_return_none_on_miss', False))
        saved_max_scored = int(getattr(self, '_force_local_ai_max_score_candidates', 128))
        saved_allow_gap = bool(getattr(self, '_force_local_ai_allow_history_gap', False))
        saved_continuity_guard = bool(getattr(self, '_force_local_ai_continuity_guard', False))
        try:
''',
'contact saved settings')

one(
'''            self._force_local_ai_return_none_on_miss = True
            self._force_local_ai_max_score_candidates = 384
            selected = self._force_local_ai_frame(frame, previous_position)
''',
'''            self._force_local_ai_return_none_on_miss = True
            self._force_local_ai_max_score_candidates = 384
            self._force_local_ai_allow_history_gap = True
            self._force_local_ai_continuity_guard = True
            selected = self._force_local_ai_frame(frame, previous_position)
''',
'contact enable continuity')

one(
'''            self._force_local_ai_return_none_on_miss = saved_return_none
            self._force_local_ai_max_score_candidates = saved_max_scored
''',
'''            self._force_local_ai_return_none_on_miss = saved_return_none
            self._force_local_ai_max_score_candidates = saved_max_scored
            self._force_local_ai_allow_history_gap = saved_allow_gap
            self._force_local_ai_continuity_guard = saved_continuity_guard
''',
'contact restore continuity')

# A normal-tracker frame between AI frames is a trusted bridge; do not throw
# away the whole trajectory just because Local AI missed one frame.
one(
'''        history = list(getattr(self, "_force_local_ai_history", []) or [])
        if history and int(history[-1].get("frame", -1000000)) != current - 1:
            history = []

        anchor = tuple(history[-1]["pos"]) if history else (
''',
'''        history = list(getattr(self, "_force_local_ai_history", []) or [])
        if history and int(history[-1].get("frame", -1000000)) != current - 1:
            if bool(getattr(self, "_force_local_ai_allow_history_gap", False)) and previous_position is not None:
                # The previous frame was handled by the normal tracker.  Use that
                # trusted result to bridge AI history so prediction survives misses.
                bridge_frame = current - 1
                bridge_pos = tuple(previous_position)
                history.append({
                    "frame": bridge_frame,
                    "pos": bridge_pos,
                    "score": 1.0,
                    "normal_bridge": True,
                })
                history = history[-4:]
                print(
                    f"[CONTACT_AI_HISTORY_BRIDGE] f{current}: "
                    f"using normal f{bridge_frame} pos={bridge_pos}"
                )
            else:
                history = []

        anchor = tuple(history[-1]["pos"]) if history else (
''',
'AI history bridge')

# When there are not yet two trajectory points, do not allow a high AI score to
# teleport the ball hundreds of pixels.  This guard is enabled only by automatic
# contact assist, not by the user's explicit forced-AI debug range.
one(
'''        tolerance = None
        if predicted is not None:
            tolerance = max(120.0, min(300.0, previous_speed * 1.8 + 60.0))

        eligible = []
''',
'''        tolerance = None
        if predicted is not None:
            tolerance = max(120.0, min(300.0, previous_speed * 1.8 + 60.0))

        continuity_cap = None
        if predicted is None and bool(getattr(self, "_force_local_ai_continuity_guard", False)):
            prior_motion = dict(getattr(self, "last_motion", {}) or {})
            tracker_speed = float(prior_motion.get("distance", 0.0) or 0.0)
            recent = [
                float(v) for v in list(getattr(self, "ball_velocity_history", []) or [])[-5:]
                if v is not None
            ]
            recent_median = float(np.median(recent)) if recent else tracker_speed
            continuity_cap = min(
                float(radius),
                max(180.0, tracker_speed * 3.0 + 60.0, recent_median * 3.5 + 40.0),
            )

        eligible = []
        continuity_rejected = 0
''',
'continuity cap setup')

one(
'''            point = (int(candidate["x"]), int(candidate["y"]))
            if self._player_point_zone(point) is not None:
                continue

            pred_dist = None
''',
'''            point = (int(candidate["x"]), int(candidate["y"]))
            if self._player_point_zone(point) is not None:
                continue

            anchor_dist = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
            if continuity_cap is not None:
                # No established AI trajectory yet: stay local to the trusted
                # ball.  Large first-step jumps require near-certain AI confidence.
                if anchor_dist > float(continuity_cap):
                    continuity_rejected += 1
                    continue
                if anchor_dist > 140.0 and score < 0.995:
                    continuity_rejected += 1
                    continue

            pred_dist = None
''',
'continuity candidate filter')

one(
'''                    f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
                    f"clear={len(clear_candidates)} scored={len(scored)}"
''',
'''                    f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
                    f"clear={len(clear_candidates)} scored={len(scored)} "
                    f"continuity_rejected={continuity_rejected}"
''',
'miss continuity log')

p.write_text(s, encoding='utf-8')
print('CONTACT_CONTINUITY_PATCH_APPLIED')
