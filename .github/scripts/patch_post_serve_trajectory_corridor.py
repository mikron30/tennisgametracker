from pathlib import Path

ANALYZER = Path('interactive_ball_analyzer.py')
RECOVERY = Path('ball_local_ai_recovery.py')
TESTS = Path('tests/test_serve_launch_handoff.py')

analyzer_text = ANALYZER.read_text(encoding='utf-8')
recovery_text = RECOVERY.read_text(encoding='utf-8')
test_text = TESTS.read_text(encoding='utf-8')

new_trajectory_function = '''    def _try_local_ai_trajectory_rescue(
            self, frame, previous_position, tracked_position, previous_stuck,
            pre_track_snapshot, reason):
        """Rescue a held player-occlusion track using physics plus Local AI.

        Ordinary tracking uses one constant-velocity prediction.  Immediately
        after a rejected post-serve handoff, however, the first accepted
        racket-to-ball displacement is not a trustworthy one-frame velocity:
        part of that displacement can be the visual re-acquisition jump itself.
        In that bounded recovery state, treat the vector as a direction with an
        along-track uncertainty corridor.  Candidates still need independent
        AI, motion, size, direction, and player-zone support before acceptance.
        """
        recovery = getattr(self, "local_ai_recovery", None)
        ranker = getattr(recovery, "rank_local_roi_candidates", None)
        corridor_ranker = getattr(recovery, "rank_local_corridor_candidates", None)
        if (
                recovery is None or not callable(ranker) or frame is None or
                previous_position is None or
                not str(reason or "").startswith("player-region:")
        ):
            return None

        snapshot = pre_track_snapshot or {}
        last_motion = snapshot.get("last_motion") or {}
        last_nonzero = snapshot.get("last_nonzero_motion") or {}
        prior_distance = float(last_motion.get("distance", 0.0) or 0.0)
        prior = last_motion if prior_distance >= 5.0 else (last_nonzero or last_motion)
        prior_dx = float(prior.get("dx", 0.0) or 0.0)
        prior_dy = float(prior.get("dy", 0.0) or 0.0)
        prior_speed = math.hypot(prior_dx, prior_dy)
        if prior_speed < 6.0:
            return None

        if tracked_position is None:
            held_distance = 0.0
        else:
            held_distance = math.hypot(
                float(tracked_position[0]) - float(previous_position[0]),
                float(tracked_position[1]) - float(previous_position[1]),
            )
        held_limit = max(3.0, min(10.0, prior_speed * 0.25))
        if held_distance > held_limit and int(previous_stuck or 0) <= 0:
            return None

        predicted = (
            int(round(float(previous_position[0]) + prior_dx)),
            int(round(float(previous_position[1]) + prior_dy)),
        )
        post_serve_corridor_active = (
            int(self.frame_count) <= int(
                getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)
            )
        )
        launch_direction_x = int(getattr(self, "_serve_launch_direction_x", 0) or 0)

        corridor_min_progress = None
        corridor_max_progress = None
        corridor_cross_cap = None
        corridor_anchors = [predicted]
        roi_radius = max(18.0, min(42.0, prior_speed * 0.80 + 7.0))

        if post_serve_corridor_active:
            # The first visual handoff may greatly overstate one-frame speed.
            # Keep the trusted direction, but search a bounded range of progress
            # along that direction.  All values scale from observed motion; no
            # frame, player location, or court coordinate is encoded here.
            unit_dx = prior_dx / prior_speed
            unit_dy = prior_dy / prior_speed
            corridor_min_progress = max(12.0, min(90.0, prior_speed * 0.15))
            corridor_max_progress = max(
                corridor_min_progress + 32.0,
                min(520.0, prior_speed * 1.10 + 18.0),
            )
            corridor_cross_cap = max(26.0, min(84.0, prior_speed * 0.20 + 12.0))
            corridor_span = corridor_max_progress - corridor_min_progress
            anchor_spacing = max(45.0, corridor_cross_cap * 0.85)
            anchor_count = max(
                3,
                min(7, int(math.ceil(corridor_span / anchor_spacing)) + 1),
            )
            corridor_anchors = []
            for index in range(anchor_count):
                fraction = index / max(1, anchor_count - 1)
                progress = corridor_min_progress + corridor_span * fraction
                corridor_anchors.append((
                    int(round(float(previous_position[0]) + unit_dx * progress)),
                    int(round(float(previous_position[1]) + unit_dy * progress)),
                ))
            roi_radius = corridor_cross_cap
            print(
                f"[LOCAL_AI_POST_SERVE_CORRIDOR] f{self.frame_count}: "
                f"prev={previous_position} dir=({unit_dx:.3f},{unit_dy:.3f}) "
                f"progress={corridor_min_progress:.1f}-{corridor_max_progress:.1f}px "
                f"cross={corridor_cross_cap:.1f}px anchors={len(corridor_anchors)}"
            )

        try:
            if post_serve_corridor_active and callable(corridor_ranker):
                ranked = corridor_ranker(
                    int(self.frame_count),
                    frame,
                    anchors=corridor_anchors,
                    radius=roi_radius,
                    maximum_candidates=48,
                )
            elif post_serve_corridor_active:
                # Backward-compatible fallback if an older recovery object is
                # injected by a test or external caller.  De-duplicate repeated
                # ROI results before trajectory arbitration.
                merged = {}
                for anchor in corridor_anchors:
                    for candidate in ranker(
                            int(self.frame_count),
                            frame,
                            anchor=anchor,
                            radius=roi_radius,
                            maximum_candidates=10):
                        key = (
                            int(candidate.get("x", 0)),
                            int(candidate.get("y", 0)),
                            str(candidate.get("mode") or ""),
                        )
                        current = merged.get(key)
                        if (
                                current is None or
                                float(candidate.get("ai_score", 0.0) or 0.0) >
                                float(current.get("ai_score", 0.0) or 0.0)
                        ):
                            merged[key] = dict(candidate)
                ranked = sorted(
                    merged.values(),
                    key=lambda item: float(item.get("ai_score", 0.0) or 0.0),
                    reverse=True,
                )
            else:
                ranked = ranker(
                    int(self.frame_count),
                    frame,
                    anchor=predicted,
                    radius=roi_radius,
                    maximum_candidates=10,
                )
        except Exception as exc:
            print(
                f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: scorer failed "
                f"({type(exc).__name__}: {exc})"
            )
            return None

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = snapshot.get("_prev_frame_gray")
        previous_size = float(snapshot.get("ball_size", 0.0) or 0.0)
        prediction_cap = (
            float(corridor_cross_cap)
            if post_serve_corridor_active else
            max(9.0, min(22.0, prior_speed * 0.65))
        )
        min_step = (
            max(4.0, float(corridor_min_progress) * 0.70)
            if post_serve_corridor_active else
            max(4.0, prior_speed * 0.35)
        )
        max_step = (
            float(corridor_max_progress) + float(corridor_cross_cap) * 0.50
            if post_serve_corridor_active else
            max(28.0, prior_speed * 1.80 + 8.0)
        )
        diagnostics = []
        valid = []
        candidate_limit = 48 if post_serve_corridor_active else 10

        unit_dx = prior_dx / prior_speed
        unit_dy = prior_dy / prior_speed
        for candidate in ranked[:candidate_limit]:
            point = (int(candidate["x"]), int(candidate["y"]))
            ai_score = float(candidate.get("ai_score", 0.0) or 0.0)
            area = float(candidate.get("area", 0.0) or 0.0)
            zone = self._player_point_zone(point)
            step_dx = float(point[0]) - float(previous_position[0])
            step_dy = float(point[1]) - float(previous_position[1])
            step = math.hypot(step_dx, step_dy)
            direction_cos = (
                (step_dx * prior_dx + step_dy * prior_dy) / (step * prior_speed)
                if step > 1e-6 and prior_speed > 1e-6 else -1.0
            )

            corridor_along = None
            corridor_cross = None
            if post_serve_corridor_active:
                corridor_along = step_dx * unit_dx + step_dy * unit_dy
                corridor_cross = abs(step_dx * unit_dy - step_dy * unit_dx)
                along_miss = max(
                    float(corridor_min_progress) - corridor_along,
                    corridor_along - float(corridor_max_progress),
                    0.0,
                )
                prediction_error = math.hypot(corridor_cross, along_miss)
            else:
                prediction_error = math.hypot(
                    float(point[0]) - float(predicted[0]),
                    float(point[1]) - float(predicted[1]),
                )

            motion = self._candidate_motion_metrics(
                current_gray,
                point[0],
                point[1],
                radius=7,
                previous_gray=previous_gray,
            ) or {}
            motion_mean = float(motion.get("mean", 0.0) or 0.0)
            motion_max = float(motion.get("max", 0.0) or 0.0)
            area_ratio = area / previous_size if previous_size > 0.0 else 1.0

            lateral_ok = True
            if post_serve_corridor_active and launch_direction_x != 0:
                signed_lateral = step_dx * float(launch_direction_x)
                lateral_ok = signed_lateral >= -max(8.0, float(corridor_cross_cap) * 0.25)

            ai_floor = (
                (0.35 if zone is not None else 0.25)
                if post_serve_corridor_active else
                (0.30 if zone is not None else 0.25)
            )
            checks = {
                "prediction": prediction_error <= prediction_cap,
                "direction": direction_cos >= (0.50 if post_serve_corridor_active else 0.45),
                "step": min_step <= step <= max_step,
                "motion": motion_mean >= 4.0 or motion_max >= 25.0,
                "area": previous_size <= 0.0 or 0.12 <= area_ratio <= 5.0,
                "lateral": lateral_ok,
                "ai": ai_score >= ai_floor,
            }
            reject_reason = next((name for name, ok in checks.items() if not ok), None)
            source_penalty = 0.0 if candidate.get("mode") == "regular_court" else 0.75
            zone_penalty = 1.25 if (post_serve_corridor_active and zone is not None) else (0.75 if zone is not None else 0.0)
            trajectory_cost = (
                prediction_error +
                max(0.0, 1.0 - direction_cos) * (14.0 if post_serve_corridor_active else 10.0) +
                max(0.0, 1.0 - ai_score) * (10.0 if post_serve_corridor_active else 8.0) +
                source_penalty + zone_penalty
            )
            row = {
                "rank": int(candidate.get("roi_rank", len(diagnostics) + 1)),
                "position": point,
                "ai_score": ai_score,
                "mode": candidate.get("mode"),
                "zone": zone,
                "area": area,
                "area_ratio": area_ratio,
                "prediction_error": prediction_error,
                "direction_cos": direction_cos,
                "step": step,
                "motion_mean": motion_mean,
                "motion_max": motion_max,
                "corridor_along": corridor_along,
                "corridor_cross": corridor_cross,
                "trajectory_cost": trajectory_cost,
                "valid": reject_reason is None,
                "reject_reason": reject_reason,
            }
            diagnostics.append(row)
            if reject_reason is None:
                valid.append((trajectory_cost, prediction_error, -ai_score, candidate, row))

        for row in diagnostics[:8 if post_serve_corridor_active else 5]:
            verdict = "OK" if row["valid"] else f"reject:{row['reject_reason']}"
            corridor_detail = ""
            if post_serve_corridor_active:
                corridor_detail = (
                    f" lane={row['corridor_along']:.1f}/"
                    f"{row['corridor_cross']:.1f}px"
                )
            print(
                f"[LOCAL_AI_TRAJECTORY_CAND] f{self.frame_count} "
                f"#{row['rank']} pos={row['position']} ai={row['ai_score']:.6f} "
                f"mode={row['mode'] or 'unknown'} "
                f"err={row['prediction_error']:.1f}/{prediction_cap:.1f}px "
                f"cos={row['direction_cos']:.2f} step={row['step']:.1f}px"
                f"{corridor_detail} "
                f"motion={row['motion_mean']:.1f}/{row['motion_max']:.1f} "
                f"area_ratio={row['area_ratio']:.2f} zone={row['zone'] or 'clear'} "
                f"cost={row['trajectory_cost']:.2f} {verdict}"
            )

        selected = min(valid, key=lambda item: (item[0], item[1], item[2])) if valid else None
        selected_candidate = dict(selected[3]) if selected is not None else None
        selected_row = selected[4] if selected is not None else None

        payload = {
            "frame": int(self.frame_count),
            "stage": "trajectory-rescue",
            "reason": str(reason),
            "previous_position": tuple(previous_position),
            "normal_position": tuple(tracked_position) if tracked_position is not None else None,
            "previous_stuck": int(previous_stuck or 0),
            "prior_motion": {"dx": prior_dx, "dy": prior_dy, "speed": prior_speed},
            "predicted_position": predicted,
            "roi_radius": roi_radius,
            "prediction_cap": prediction_cap,
            "post_serve_corridor": post_serve_corridor_active,
            "corridor_anchors": corridor_anchors if post_serve_corridor_active else None,
            "corridor_min_progress": corridor_min_progress,
            "corridor_max_progress": corridor_max_progress,
            "corridor_cross_cap": corridor_cross_cap,
            "candidates": diagnostics,
            "accepted": selected_candidate is not None,
            "selected": selected_row,
        }
        writer = getattr(recovery, "_write_event", None)
        if callable(writer):
            try:
                writer(payload)
            except Exception as exc:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: "
                    f"debug event write failed ({type(exc).__name__}: {exc})"
                )

        work_dir = getattr(recovery, "work_dir", None)
        if work_dir is not None:
            try:
                debug_dir = os.path.join(str(work_dir), "trajectory_frames")
                os.makedirs(debug_dir, exist_ok=True)
                debug_image = frame.copy()
                cv2.circle(debug_image, tuple(map(int, previous_position)), 13, (0, 255, 255), 2)
                if post_serve_corridor_active:
                    cv2.line(
                        debug_image,
                        tuple(map(int, corridor_anchors[0])),
                        tuple(map(int, corridor_anchors[-1])),
                        (255, 255, 0),
                        2,
                    )
                    for anchor in corridor_anchors:
                        cv2.circle(debug_image, tuple(map(int, anchor)), 7, (255, 255, 0), 1)
                else:
                    cv2.circle(debug_image, predicted, int(round(roi_radius)), (255, 255, 0), 2)
                if tracked_position is not None:
                    cv2.circle(debug_image, tuple(map(int, tracked_position)), 11, (0, 0, 255), 2)
                for row in diagnostics[:8 if post_serve_corridor_active else 5]:
                    point = tuple(map(int, row["position"]))
                    cv2.circle(debug_image, point, 12, (255, 0, 255), 2)
                    label_error = row["corridor_cross"] if post_serve_corridor_active else row["prediction_error"]
                    cv2.putText(
                        debug_image,
                        f"#{row['rank']} e{label_error:.0f} a{row['ai_score']:.2f}",
                        (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                    )
                output_path = os.path.join(
                    debug_dir, f"trajectory_f{int(self.frame_count):08d}.jpg"
                )
                cv2.imwrite(output_path, debug_image)
                print(f"[LOCAL_AI_TRAJECTORY_IMAGE] f{self.frame_count} saved={output_path}")
            except Exception as exc:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: "
                    f"debug image failed ({type(exc).__name__}: {exc})"
                )

        if selected_candidate is None:
            if post_serve_corridor_active:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: no safe candidate "
                    f"prev={previous_position} corridor="
                    f"{corridor_min_progress:.1f}-{corridor_max_progress:.1f}px "
                    f"cross={corridor_cross_cap:.1f}px"
                )
            else:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: no safe candidate "
                    f"prev={previous_position} predicted={predicted} radius={roi_radius:.1f}px"
                )
            return None

        point = (int(selected_candidate["x"]), int(selected_candidate["y"]))
        step_dx = float(point[0]) - float(previous_position[0])
        step_dy = float(point[1]) - float(previous_position[1])
        step = math.hypot(step_dx, step_dy)
        selected_candidate["trajectory_rescue"] = True
        selected_candidate["trajectory_prediction"] = predicted
        selected_candidate["trajectory_prediction_error"] = float(selected_row["prediction_error"])
        selected_candidate["trajectory_cost"] = float(selected_row["trajectory_cost"])
        if post_serve_corridor_active:
            selected_candidate["trajectory_post_serve_corridor"] = True
            selected_candidate["trajectory_corridor_along"] = float(selected_row["corridor_along"])
            selected_candidate["trajectory_corridor_cross"] = float(selected_row["corridor_cross"])
        selected_candidate["trajectory_motion"] = {
            "dx": step_dx,
            "dy": step_dy,
            "distance": step,
            "direction_deg": math.degrees(math.atan2(step_dy, step_dx)) if step > 0.0 else None,
        }
        mode_label = " corridor" if post_serve_corridor_active else ""
        print(
            f"[LOCAL_AI_TRAJECTORY_ACCEPT] f{self.frame_count}:{mode_label} "
            f"{point} ai={float(selected_candidate.get('ai_score', 0.0)):.6f} "
            f"pred={predicted} err={selected_row['prediction_error']:.1f}px "
            f"motion={selected_row['motion_mean']:.1f}/{selected_row['motion_max']:.1f} "
            f"zone={selected_row['zone'] or 'clear'} cost={selected_row['trajectory_cost']:.2f}"
        )
        return selected_candidate

'''

start = analyzer_text.index('    def _try_local_ai_trajectory_rescue(')
end = analyzer_text.index('    def _try_local_ai_recovery(', start)
analyzer_text = analyzer_text[:start] + new_trajectory_function + analyzer_text[end:]
ANALYZER.write_text(analyzer_text, encoding='utf-8')

corridor_method = '''    def rank_local_corridor_candidates(
        self,
        frame_index: int,
        image: np.ndarray,
        *,
        anchors: Iterable[tuple[int, int]],
        radius: float = 48.0,
        maximum_candidates: int = 48,
    ) -> list[dict]:
        """Score HSV candidates from a bounded directional corridor in one AI call.

        ``anchors`` describe uncertainty along a motion direction.  Candidate
        generation is local around each anchor, then de-duplicated and scored as
        one batch so widening a short recovery corridor does not multiply model
        process startup cost.
        """
        if image is None:
            return []
        normalized_anchors = [
            (int(anchor[0]), int(anchor[1]))
            for anchor in anchors
            if isinstance(anchor, (tuple, list)) and len(anchor) >= 2
        ]
        if not normalized_anchors:
            return []

        corridor_radius = max(8.0, float(radius))
        unique: dict[tuple[int, int, str], dict] = {}
        for anchor in normalized_anchors:
            candidates = collect_candidates(
                image,
                self._config,
                min_area=3.0,
                max_area=2000.0,
                around=anchor,
                radius=corridor_radius,
                dedup_distance=4.0,
            )
            for candidate in candidates:
                item = dict(candidate)
                point = (int(item["x"]), int(item["y"]))
                closest_anchor = min(
                    normalized_anchors,
                    key=lambda value: self._distance(point, value),
                )
                anchor_distance = self._distance(point, closest_anchor)
                item["corridor_anchor"] = closest_anchor
                item["corridor_anchor_distance"] = float(anchor_distance)
                key = (point[0], point[1], str(item.get("mode") or ""))
                previous = unique.get(key)
                if (
                        previous is None or
                        anchor_distance < float(previous.get("corridor_anchor_distance", float("inf")))
                ):
                    unique[key] = item

        if not unique:
            return []
        candidates = sorted(
            unique.values(),
            key=lambda item: (
                float(item.get("corridor_anchor_distance", float("inf"))),
                -float(item.get("area", 0.0) or 0.0),
            ),
        )[: max(1, int(maximum_candidates))]

        scored = self._score(image, int(frame_index), candidates)
        ranked = sorted(
            scored,
            key=lambda item: float(item.get("ai_score", 0.0)),
            reverse=True,
        )
        result: list[dict] = []
        for index, candidate in enumerate(ranked):
            item = dict(candidate)
            item["roi_anchor"] = None
            item["roi_radius"] = corridor_radius
            item["roi_candidates"] = len(ranked)
            item["roi_rank"] = index + 1
            item["corridor_anchor_count"] = len(normalized_anchors)
            result.append(item)
        return result

'''

if '    def rank_local_corridor_candidates(' not in recovery_text:
    marker = '    def rank_local_roi_candidate(\n'
    pos = recovery_text.index(marker)
    recovery_text = recovery_text[:pos] + corridor_method + recovery_text[pos:]
RECOVERY.write_text(recovery_text, encoding='utf-8')

test_block = '''\n\nclass PostServeTrajectoryCorridorTests(unittest.TestCase):
    class FakeRecovery:
        def __init__(self, *, corridor_candidates=None, roi_candidates=None):
            self.corridor_candidates = list(corridor_candidates or [])
            self.roi_candidates = list(roi_candidates or [])
            self.work_dir = None
            self.events = []
            self.corridor_calls = []
            self.roi_calls = []

        def rank_local_corridor_candidates(self, frame_index, image, *, anchors, radius, maximum_candidates):
            self.corridor_calls.append((list(anchors), float(radius), int(maximum_candidates)))
            return list(self.corridor_candidates)

        def rank_local_roi_candidates(self, frame_index, image, *, anchor, radius, maximum_candidates):
            self.roi_calls.append((tuple(anchor), float(radius), int(maximum_candidates)))
            return list(self.roi_candidates)

        def _write_event(self, payload):
            self.events.append(payload)

    @staticmethod
    def _candidate(x, y, *, score=0.55, area=90.0):
        return {
            "x": int(x),
            "y": int(y),
            "ai_score": float(score),
            "area": float(area),
            "mode": "regular_court",
            "roi_rank": 1,
        }

    def _analyzer(self, recovery, *, corridor_active=True):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.local_ai_recovery = recovery
        analyzer.frame_count = 20
        analyzer._post_serve_recovery_hold_until_frame = 24 if corridor_active else -1000000
        analyzer._post_serve_launch_lock_until_frame = 22 if corridor_active else -1
        analyzer._serve_launch_direction_x = -1
        analyzer._player_point_zone = lambda point: None
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: {"mean": 14.0, "max": 80.0}
        return analyzer

    @staticmethod
    def _snapshot(frame_shape):
        return {
            "last_motion": {"dx": -80.0, "dy": -160.0, "distance": 178.9},
            "last_nonzero_motion": {"dx": -80.0, "dy": -160.0, "distance": 178.9},
            "ball_size": 100.0,
            "_prev_frame_gray": np.zeros(frame_shape[:2], dtype=np.uint8),
        }

    def test_post_serve_recovery_accepts_on_direction_candidate_before_full_velocity_prediction(self):
        candidate = self._candidate(755, 610)
        recovery = self.FakeRecovery(corridor_candidates=[candidate])
        analyzer = self._analyzer(recovery, corridor_active=True)
        frame = np.zeros((1000, 1200, 3), dtype=np.uint8)

        selected = analyzer._try_local_ai_trajectory_rescue(
            frame,
            previous_position=(800, 700),
            tracked_position=(800, 700),
            previous_stuck=1,
            pre_track_snapshot=self._snapshot(frame.shape),
            reason="player-region:racket_fragment",
        )

        self.assertIsNotNone(selected)
        self.assertEqual((selected["x"], selected["y"]), (755, 610))
        self.assertTrue(selected.get("trajectory_post_serve_corridor"))
        self.assertEqual(len(recovery.corridor_calls), 1)
        self.assertEqual(len(recovery.roi_calls), 0)

    def test_post_serve_corridor_rejects_candidate_against_outgoing_direction(self):
        candidate = self._candidate(855, 790, score=0.90)
        recovery = self.FakeRecovery(corridor_candidates=[candidate])
        analyzer = self._analyzer(recovery, corridor_active=True)
        frame = np.zeros((1000, 1200, 3), dtype=np.uint8)

        selected = analyzer._try_local_ai_trajectory_rescue(
            frame,
            previous_position=(800, 700),
            tracked_position=(800, 700),
            previous_stuck=1,
            pre_track_snapshot=self._snapshot(frame.shape),
            reason="player-region:racket_fragment",
        )

        self.assertIsNone(selected)
        self.assertEqual(len(recovery.corridor_calls), 1)

    def test_normal_trajectory_rescue_keeps_single_constant_velocity_roi(self):
        candidate = self._candidate(720, 540)
        recovery = self.FakeRecovery(roi_candidates=[candidate])
        analyzer = self._analyzer(recovery, corridor_active=False)
        frame = np.zeros((1000, 1200, 3), dtype=np.uint8)

        selected = analyzer._try_local_ai_trajectory_rescue(
            frame,
            previous_position=(800, 700),
            tracked_position=(800, 700),
            previous_stuck=1,
            pre_track_snapshot=self._snapshot(frame.shape),
            reason="player-region:racket_fragment",
        )

        self.assertIsNotNone(selected)
        self.assertEqual((selected["x"], selected["y"]), (720, 540))
        self.assertFalse(selected.get("trajectory_post_serve_corridor", False))
        self.assertEqual(len(recovery.roi_calls), 1)
        self.assertEqual(len(recovery.corridor_calls), 0)
'''

if 'class PostServeTrajectoryCorridorTests' not in test_text:
    test_text = test_text.rstrip() + test_block + '\n'
TESTS.write_text(test_text, encoding='utf-8')

print('Patched post-serve trajectory uncertainty corridor and tests')
