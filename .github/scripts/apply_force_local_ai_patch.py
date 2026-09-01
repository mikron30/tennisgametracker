from pathlib import Path

path = Path("interactive_ball_analyzer.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    text = text.replace(old, new, 1)


if "--force-local-ai-range" in text:
    print("forced Local AI range already installed")
    raise SystemExit(0)

# 1) Runtime-only state. Existing runs remain unchanged.
replace_once(
    '''        self._debug_local_ai_range = debug_local_ai_range\n        self._debug_local_ai_radius = max(20.0, float(debug_local_ai_radius))\n        self._local_ai_frame_buffer = frame_buffer(12)\n''',
    '''        self._debug_local_ai_range = debug_local_ai_range\n        self._debug_local_ai_radius = max(20.0, float(debug_local_ai_radius))\n        self._force_local_ai_range = None\n        self._force_local_ai_radius = 450.0\n        self._force_local_ai_min_score = 0.985\n        self._force_local_ai_history = []\n        self._local_ai_frame_buffer = frame_buffer(12)\n''',
    "force state",
)

# 2) Forced-AI tracker. It never accepts a candidate inside a player zone.
method_anchor = '''    def _debug_local_ai_shadow_frame(self, frame, previous_position, normal_position):\n'''
if text.count(method_anchor) != 1:
    raise RuntimeError(f"force method anchor: expected 1 match, found {text.count(method_anchor)}")

force_method = '''    def _force_local_ai_frame(self, frame, previous_position):
        """TEST ONLY: bypass HSV and let Local AI own an exact frame range.

        The previous AI-selected point becomes the next search anchor. Player,
        head, shoe, and racket regions are excluded before scoring. Once two AI
        points exist, high-confidence candidates must also fit the extrapolated
        AI trajectory. Outside --force-local-ai-range this method returns None
        and the normal tracker is unchanged.
        """
        frame_range = getattr(self, "_force_local_ai_range", None)
        recovery = getattr(self, "local_ai_recovery", None)
        if frame_range is None or frame is None or recovery is None:
            return None

        start_frame, end_frame = frame_range
        current = int(self.frame_count)
        if not (int(start_frame) <= current <= int(end_frame)):
            if current > int(end_frame):
                self._force_local_ai_history = []
            return None

        from ball_ai_recovery_probe import collect_candidates

        history = list(getattr(self, "_force_local_ai_history", []) or [])
        if history and int(history[-1].get("frame", -1000000)) != current - 1:
            history = []

        anchor = tuple(history[-1]["pos"]) if history else (
            tuple(previous_position) if previous_position is not None else None
        )
        if anchor is None:
            print(f"[FORCE_LOCAL_AI_HOLD] f{current}: no anchor available")
            return previous_position

        radius = max(40.0, float(getattr(self, "_force_local_ai_radius", 450.0)))
        min_score = float(getattr(self, "_force_local_ai_min_score", 0.985))

        config = dict(recovery._config)
        config["force_low_sat"] = {
            "h_min": 80, "h_max": 135,
            "s_min": 10, "s_max": 60,
            "v_min": 110, "v_max": 255,
        }
        modes = (
            "regular_court", "alt1", "alt2", "alt3", "s_30", "h_10",
            "force_low_sat",
        )

        predicted = None
        previous_speed = 0.0
        previous_vector = None
        if len(history) >= 2:
            p0 = tuple(history[-2]["pos"])
            p1 = tuple(history[-1]["pos"])
            vx = float(p1[0] - p0[0])
            vy = float(p1[1] - p0[1])
            previous_speed = math.hypot(vx, vy)
            if previous_speed >= 3.0:
                previous_vector = (vx, vy)
                predicted = (int(round(p1[0] + vx)), int(round(p1[1] + vy)))

        candidates = collect_candidates(
            frame, config, modes=modes,
            min_area=3.0, max_area=1200.0,
            around=anchor, radius=radius,
            dedup_distance=4.0,
        )

        clear_candidates = []
        for candidate in candidates:
            point = (int(candidate["x"]), int(candidate["y"]))
            if self._player_point_zone(point) is None:
                clear_candidates.append(candidate)

        sort_anchor = predicted if predicted is not None else anchor
        clear_candidates.sort(
            key=lambda item: math.hypot(
                float(item["x"]) - float(sort_anchor[0]),
                float(item["y"]) - float(sort_anchor[1]),
            )
        )
        subset = clear_candidates[:128]
        scored = recovery._score(frame, current, subset) if subset else []

        tolerance = None
        if predicted is not None:
            tolerance = max(120.0, min(300.0, previous_speed * 1.8 + 60.0))

        eligible = []
        for candidate in scored:
            score = float(candidate.get("ai_score", 0.0) or 0.0)
            if score < min_score:
                continue
            point = (int(candidate["x"]), int(candidate["y"]))
            if self._player_point_zone(point) is not None:
                continue

            pred_dist = None
            cosine = None
            if predicted is not None:
                pred_dist = math.hypot(point[0] - predicted[0], point[1] - predicted[1])
                if pred_dist > float(tolerance):
                    continue
                if previous_vector is not None:
                    sx = float(point[0] - anchor[0])
                    sy = float(point[1] - anchor[1])
                    step = math.hypot(sx, sy)
                    if step >= 3.0 and previous_speed >= 3.0:
                        cosine = (
                            previous_vector[0] * sx + previous_vector[1] * sy
                        ) / (previous_speed * step)
                        if cosine < -0.20:
                            continue

            eligible.append({
                "candidate": candidate,
                "point": point,
                "score": score,
                "pred_dist": pred_dist,
                "cosine": cosine,
            })

        if predicted is None:
            eligible.sort(
                key=lambda item: (
                    -item["score"],
                    math.hypot(item["point"][0] - anchor[0], item["point"][1] - anchor[1]),
                )
            )
        else:
            eligible.sort(key=lambda item: (float(item["pred_dist"]), -item["score"]))

        for rank, item in enumerate(eligible[:5], 1):
            pd = "n/a" if item["pred_dist"] is None else f"{item['pred_dist']:.1f}"
            cs = "n/a" if item["cosine"] is None else f"{item['cosine']:.3f}"
            print(
                f"[FORCE_LOCAL_AI_CAND] f{current} #{rank} pos={item['point']} "
                f"ai={item['score']:.6f} pred_dist={pd} cos={cs}"
            )

        if not eligible:
            self.ball_center = tuple(anchor)
            self.stuck_frame_count = max(1, int(getattr(self, "stuck_frame_count", 0)) + 1)
            print(
                f"[FORCE_LOCAL_AI_HOLD] f{current}: no clear candidate >= {min_score:.3f}; "
                f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
                f"clear={len(clear_candidates)} scored={len(scored)}"
            )
            return tuple(anchor)

        selected = eligible[0]
        candidate = selected["candidate"]
        point = selected["point"]
        area = float(candidate.get("area", 0.0) or 0.0)

        dx = int(point[0] - anchor[0])
        dy = int(point[1] - anchor[1])
        distance = math.hypot(dx, dy)
        direction_deg = math.degrees(math.atan2(dy, dx)) if distance > 0.0 else None

        self.ball_center = point
        self.ball_size = area
        height, width = frame.shape[:2]
        px = max(0, min(width - 1, point[0]))
        py = max(0, min(height - 1, point[1]))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.ball_hsv = hsv_frame[py, px]

        self.prev_motion = self.last_motion
        self.last_motion = {
            "distance": float(distance), "dx": dx, "dy": dy,
            "direction_deg": direction_deg,
        }
        if distance >= 3.0:
            self.last_nonzero_motion = dict(self.last_motion)
        if direction_deg is not None:
            self.last_direction = direction_deg
        self.last_delta = (dx, dy)
        self.stuck_frame_count = 0
        self.ball_stopped = False
        self.last_seen_frame = current

        self.ball_velocity_history.append(float(distance))
        if len(self.ball_velocity_history) > 10:
            self.ball_velocity_history = self.ball_velocity_history[-10:]
        self.motion_history.append({
            "frame": current,
            "distance": float(distance),
            "direction_deg": direction_deg,
            "pos": tuple(point),
            "prev_pos": tuple(anchor),
            "forced_local_ai": True,
        })
        if len(self.motion_history) > 200:
            self.motion_history = self.motion_history[-200:]
        self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        history.append({"frame": current, "pos": tuple(point), "score": selected["score"]})
        self._force_local_ai_history = history[-4:]

        print(
            f"[FORCE_LOCAL_AI] f{current}: selected={point} ai={selected['score']:.6f} "
            f"area={area:.1f}px anchor={anchor} predicted={predicted} "
            f"normal_hsv_bypassed=True"
        )
        return tuple(point)

'''
text = text.replace(method_anchor, force_method + method_anchor, 1)

# 3) Bypass HSV only inside the force range; otherwise preserve the old flow.
replace_once(
    '''                    tracked_position = self.track_ball_in_frame(frame)\n                    if self.local_ai_recovery is not None and self._local_ai_frame_buffer:\n                        self._local_ai_frame_buffer[-1]["normal_position"] = (\n                            tuple(tracked_position) if tracked_position is not None else None\n                        )\n                    self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)\n                    tracked_position = self._try_local_ai_recovery(\n                        prev_ball_center, tracked_position, prev_stuck,\n                        pre_track_snapshot=pre_track_snapshot,\n                        frame=frame,\n                    )\n''',
    '''                    forced_local_ai = False\n                    tracked_position = self._force_local_ai_frame(frame, prev_ball_center)\n                    if tracked_position is not None:\n                        forced_local_ai = True\n                    else:\n                        tracked_position = self.track_ball_in_frame(frame)\n                    if self.local_ai_recovery is not None and self._local_ai_frame_buffer:\n                        self._local_ai_frame_buffer[-1]["normal_position"] = (\n                            tuple(tracked_position) if tracked_position is not None else None\n                        )\n                    self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)\n                    if not forced_local_ai:\n                        tracked_position = self._try_local_ai_recovery(\n                            prev_ball_center, tracked_position, prev_stuck,\n                            pre_track_snapshot=pre_track_snapshot,\n                            frame=frame,\n                        )\n''',
    "process force bypass",
)

# 4) CLI.
replace_once(
    '''    parser.add_argument("--debug-local-ai-radius", type=float, default=140.0,\n                        help="ROI radius for --debug-local-ai-range (default 140 px)")\n    parser.add_argument("--debug-local-ai-top-n", type=int, default=10,\n''',
    '''    parser.add_argument("--debug-local-ai-radius", type=float, default=140.0,\n                        help="ROI radius for --debug-local-ai-range (default 140 px)")\n    parser.add_argument("--force-local-ai-range", type=_parse_frame_range, metavar="START:END", default=None,\n                        help="TEST ONLY: bypass HSV and let Local AI own an inclusive frame range")\n    parser.add_argument("--force-local-ai-radius", type=float, default=450.0,\n                        help="Search radius for --force-local-ai-range (default 450 px)")\n    parser.add_argument("--force-local-ai-min-score", type=float, default=0.985,\n                        help="Minimum Local-AI score in forced range (default 0.985)")\n    parser.add_argument("--debug-local-ai-top-n", type=int, default=10,\n''',
    "force CLI",
)

# 5) Wire parsed values into the analyzer without changing its constructor API.
replace_once(
    '''                analyzer._debug_local_ai_top_n = args.debug_local_ai_top_n\n                analyzer._debug_hsv_sweep = args.debug_hsv_sweep\n                result = analyzer.process_video(auto_play=args.auto_play, max_frames=max_frames_for_run)\n''',
    '''                analyzer._debug_local_ai_top_n = args.debug_local_ai_top_n\n                analyzer._debug_hsv_sweep = args.debug_hsv_sweep\n                analyzer._force_local_ai_range = args.force_local_ai_range\n                analyzer._force_local_ai_radius = max(40.0, float(args.force_local_ai_radius))\n                analyzer._force_local_ai_min_score = max(0.0, min(1.0, float(args.force_local_ai_min_score)))\n                if args.force_local_ai_range is not None:\n                    print(\n                        f"[FORCE_LOCAL_AI_MODE] range={args.force_local_ai_range[0]}:"\n                        f"{args.force_local_ai_range[1]} radius={analyzer._force_local_ai_radius:.0f}px "\n                        f"min_score={analyzer._force_local_ai_min_score:.3f}"\n                    )\n                result = analyzer.process_video(auto_play=args.auto_play, max_frames=max_frames_for_run)\n''',
    "force args assignment",
)

required = (
    "--force-local-ai-range",
    "[FORCE_LOCAL_AI_MODE]",
    "[FORCE_LOCAL_AI]",
    "[FORCE_LOCAL_AI_HOLD]",
    "normal_hsv_bypassed=True",
    "force_low_sat",
)
for marker in required:
    if marker not in text:
        raise RuntimeError(f"missing forced Local AI marker: {marker}")

path.write_text(text, encoding="utf-8")
print("forced Local AI range test patch applied")
