from pathlib import Path
import textwrap

analyzer_path = Path('interactive_ball_analyzer.py')
recovery_path = Path('ball_local_ai_recovery.py')
test_path = Path('tests/test_ball_local_ai_recovery.py')

analyzer = analyzer_path.read_text(encoding='utf-8')
recovery = recovery_path.read_text(encoding='utf-8')
tests = test_path.read_text(encoding='utf-8')

# 1) Expose ranked Top-N candidates from the bounded Local-AI ROI.
rank_start = recovery.find('    def rank_local_roi_candidate(\n')
rank_end = recovery.find('    def _best_candidate(\n', rank_start)
if rank_start < 0 or rank_end < 0:
    raise RuntimeError('rank_local_roi_candidate block not found')
rank_block = '''    def rank_local_roi_candidates(
        self,
        frame_index: int,
        image: np.ndarray,
        *,
        anchor: tuple[int, int],
        radius: float = 25.0,
        maximum_candidates: int = 12,
    ) -> list[dict]:
        """Return model-ranked real HSV contours inside one bounded ROI.

        This function intentionally does not decide whether a candidate is safe
        enough to rewrite the tracker.  The caller can combine the relative AI
        ranking with trajectory, motion, size, and player-context evidence.
        """
        if image is None or anchor is None:
            return []
        roi_radius = max(8.0, float(radius))
        roi_anchor = (int(anchor[0]), int(anchor[1]))
        candidates = collect_candidates(
            image,
            self._config,
            min_area=3.0,
            max_area=2000.0,
            around=roi_anchor,
            radius=roi_radius,
        )
        candidates = self._candidate_subset(candidates, roi_anchor)[
            : max(1, int(maximum_candidates))
        ]
        if not candidates:
            return []

        scored = self._score(image, int(frame_index), candidates)
        ranked = sorted(
            scored,
            key=lambda item: float(item.get("ai_score", 0.0)),
            reverse=True,
        )
        result: list[dict] = []
        for index, candidate in enumerate(ranked):
            item = dict(candidate)
            item["roi_anchor"] = roi_anchor
            item["roi_radius"] = roi_radius
            item["roi_candidates"] = len(ranked)
            item["roi_rank"] = index + 1
            result.append(item)
        return result

    def rank_local_roi_candidate(
        self,
        frame_index: int,
        image: np.ndarray,
        *,
        anchor: tuple[int, int],
        radius: float = 25.0,
        maximum_candidates: int = 12,
    ) -> Optional[dict]:
        """Return the model's top contour inside one tightly bounded ROI."""
        ranked = self.rank_local_roi_candidates(
            frame_index,
            image,
            anchor=anchor,
            radius=radius,
            maximum_candidates=maximum_candidates,
        )
        if not ranked:
            return None
        selected = dict(ranked[0])
        selected["roi_score_margin"] = (
            float(selected.get("ai_score", 0.0)) -
            float(ranked[1].get("ai_score", 0.0))
            if len(ranked) >= 2 else None
        )
        return selected

'''
recovery = recovery[:rank_start] + rank_block + recovery[rank_end:]

# 2) Add a conservative trajectory-aware Local-AI rescue to the analyzer.
helper_marker = '    def _try_local_ai_recovery(\n'
helper_pos = analyzer.find(helper_marker)
if helper_pos < 0:
    raise RuntimeError('_try_local_ai_recovery marker not found')
if '    def _try_local_ai_trajectory_rescue(' not in analyzer:
    helper = '''    def _try_local_ai_trajectory_rescue(
            self, frame, previous_position, tracked_position, previous_stuck,
            pre_track_snapshot, reason):
        """Rescue a held player-occlusion track using physics plus Local AI.

        This is deliberately general: no frame numbers or court coordinates are
        encoded here.  It only activates when the normal tracker is effectively
        held at the previous point inside a player-region recovery, then asks AI
        to rank candidates in a small ROI around the constant-velocity prediction.
        A lower-AI candidate can beat a player fragment only when trajectory,
        direction, motion, and size all independently support it.
        """
        recovery = getattr(self, "local_ai_recovery", None)
        ranker = getattr(recovery, "rank_local_roi_candidates", None)
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
        roi_radius = max(18.0, min(42.0, prior_speed * 0.80 + 7.0))
        try:
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
        prediction_cap = max(9.0, min(22.0, prior_speed * 0.65))
        min_step = max(4.0, prior_speed * 0.35)
        max_step = max(28.0, prior_speed * 1.80 + 8.0)
        diagnostics = []
        valid = []

        for candidate in ranked[:10]:
            point = (int(candidate["x"]), int(candidate["y"]))
            ai_score = float(candidate.get("ai_score", 0.0) or 0.0)
            area = float(candidate.get("area", 0.0) or 0.0)
            zone = self._player_point_zone(point)
            prediction_error = math.hypot(
                float(point[0]) - float(predicted[0]),
                float(point[1]) - float(predicted[1]),
            )
            step_dx = float(point[0]) - float(previous_position[0])
            step_dy = float(point[1]) - float(previous_position[1])
            step = math.hypot(step_dx, step_dy)
            direction_cos = (
                (step_dx * prior_dx + step_dy * prior_dy) / (step * prior_speed)
                if step > 1e-6 and prior_speed > 1e-6 else -1.0
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

            checks = {
                "prediction": prediction_error <= prediction_cap,
                "direction": direction_cos >= 0.45,
                "step": min_step <= step <= max_step,
                "motion": motion_mean >= 4.0 or motion_max >= 25.0,
                "area": previous_size <= 0.0 or 0.12 <= area_ratio <= 5.0,
                "ai": ai_score >= (0.30 if zone is not None else 0.25),
            }
            reject_reason = next((name for name, ok in checks.items() if not ok), None)
            source_penalty = 0.0 if candidate.get("mode") == "regular_court" else 0.75
            zone_penalty = 0.75 if zone is not None else 0.0
            trajectory_cost = (
                prediction_error +
                max(0.0, 1.0 - direction_cos) * 10.0 +
                max(0.0, 1.0 - ai_score) * 8.0 +
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
                "trajectory_cost": trajectory_cost,
                "valid": reject_reason is None,
                "reject_reason": reject_reason,
            }
            diagnostics.append(row)
            if reject_reason is None:
                valid.append((trajectory_cost, prediction_error, -ai_score, candidate, row))

        for row in diagnostics[:5]:
            verdict = "OK" if row["valid"] else f"reject:{row['reject_reason']}"
            print(
                f"[LOCAL_AI_TRAJECTORY_CAND] f{self.frame_count} "
                f"#{row['rank']} pos={row['position']} ai={row['ai_score']:.6f} "
                f"err={row['prediction_error']:.1f}/{prediction_cap:.1f}px "
                f"cos={row['direction_cos']:.2f} step={row['step']:.1f}px "
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
                cv2.circle(debug_image, predicted, int(round(roi_radius)), (255, 255, 0), 2)
                if tracked_position is not None:
                    cv2.circle(debug_image, tuple(map(int, tracked_position)), 11, (0, 0, 255), 2)
                for row in diagnostics[:5]:
                    point = tuple(map(int, row["position"]))
                    cv2.circle(debug_image, point, 12, (255, 0, 255), 2)
                    cv2.putText(
                        debug_image,
                        f"#{row['rank']} e{row['prediction_error']:.0f} a{row['ai_score']:.2f}",
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
        selected_candidate["trajectory_motion"] = {
            "dx": step_dx,
            "dy": step_dy,
            "distance": step,
            "direction_deg": math.degrees(math.atan2(step_dy, step_dx)) if step > 0.0 else None,
        }
        print(
            f"[LOCAL_AI_TRAJECTORY_ACCEPT] f{self.frame_count}: "
            f"{point} ai={float(selected_candidate.get('ai_score', 0.0)):.6f} "
            f"pred={predicted} err={selected_row['prediction_error']:.1f}px "
            f"motion={selected_row['motion_mean']:.1f}/{selected_row['motion_max']:.1f} "
            f"zone={selected_row['zone'] or 'clear'} cost={selected_row['trajectory_cost']:.2f}"
        )
        return selected_candidate

'''
    analyzer = analyzer[:helper_pos] + helper + analyzer[helper_pos:]

# 3) Try the trajectory rescue before broad multi-frame recovery.
old_recover = '''        recovered = self.local_ai_recovery.recover(
            self.frame_count,
            self._local_ai_frame_buffer,
            predicted_position=previous_position,
            player_zone=self._player_point_zone,
            reason=reason,
            force=False,
        )
'''
if old_recover not in analyzer:
    raise RuntimeError('normal recovery call block not found')
new_recover = '''        trajectory_recovered = self._try_local_ai_trajectory_rescue(
            frame,
            previous_position,
            tracked_position,
            previous_stuck,
            pre_track_snapshot,
            reason,
        )
        if trajectory_recovered is not None:
            recovered = trajectory_recovered
            reason = f"{reason}+trajectory"
        else:
            recovered = self.local_ai_recovery.recover(
                self.frame_count,
                self._local_ai_frame_buffer,
                predicted_position=previous_position,
                player_zone=self._player_point_zone,
                reason=reason,
                force=False,
            )
'''
analyzer = analyzer.replace(old_recover, new_recover, 1)

# 4) When trajectory rescue wins, make its physical motion the live motion state.
old_state = '''        self.ball_size = float(recovered.get("area", self.ball_size or 0.0))
        self.last_seen_frame = self.frame_count
'''
if old_state not in analyzer:
    raise RuntimeError('accepted recovery state marker not found')
new_state = '''        self.ball_size = float(recovered.get("area", self.ball_size or 0.0))
        if recovered.get("trajectory_rescue"):
            trajectory_motion = dict(recovered.get("trajectory_motion") or {})
            prior_motion = (pre_track_snapshot or {}).get("last_motion")
            self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
            self.last_motion = trajectory_motion
            if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
                self.last_nonzero_motion = dict(trajectory_motion)
            prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
            prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
            self.ball_velocity_history = prior_history[-5:]
            self._held_direction_candidate = None
        self.last_seen_frame = self.frame_count
'''
analyzer = analyzer.replace(old_state, new_state, 1)

# 5) Add focused unit tests for the generic behaviour.
if 'class TrajectoryAwareLocalAITests' not in tests:
    insert_at = tests.find('\n\nif __name__ == "__main__":')
    if insert_at < 0:
        raise RuntimeError('test main marker not found')
    test_block = r'''

class TrajectoryAwareLocalAITests(unittest.TestCase):
    @staticmethod
    def _analyzer(candidate, *, zone="player_body", motion=(12.0, 90.0)):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 100
        analyzer._player_point_zone = lambda point: zone
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: {
            "mean": motion[0], "max": motion[1]
        }

        class Recovery:
            def __init__(self):
                self.events = []

            def rank_local_roi_candidates(self, *args, **kwargs):
                return [dict(candidate)]

            def _write_event(self, payload):
                self.events.append(payload)

        analyzer.local_ai_recovery = Recovery()
        return analyzer

    def test_trajectory_rescue_accepts_lower_ai_candidate_on_physical_prediction(self):
        candidate = {
            "x": 87, "y": 115, "area": 10.0, "ai_score": 0.52,
            "mode": "regular_court", "roi_rank": 2,
        }
        analyzer = self._analyzer(candidate)
        snapshot = {
            "last_motion": {"dx": -13.0, "dy": 15.0, "distance": 19.85},
            "last_nonzero_motion": {"dx": -13.0, "dy": 15.0, "distance": 19.85},
            "ball_size": 24.0,
            "ball_velocity_history": [18.0, 19.0, 20.0],
            "_prev_frame_gray": np.zeros((180, 180), dtype=np.uint8),
        }
        result = analyzer._try_local_ai_trajectory_rescue(
            np.zeros((180, 180, 3), dtype=np.uint8),
            (100, 100),
            (100, 100),
            0,
            snapshot,
            "player-region:player_body",
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["trajectory_rescue"])
        self.assertEqual((result["x"], result["y"]), (87, 115))
        self.assertLess(result["ai_score"], 0.985)
        self.assertTrue(analyzer.local_ai_recovery.events[-1]["accepted"])

    def test_trajectory_rescue_does_not_override_a_moving_normal_track(self):
        candidate = {
            "x": 87, "y": 115, "area": 10.0, "ai_score": 0.90,
            "mode": "regular_court", "roi_rank": 1,
        }
        analyzer = self._analyzer(candidate)
        snapshot = {
            "last_motion": {"dx": -13.0, "dy": 15.0, "distance": 19.85},
            "last_nonzero_motion": {"dx": -13.0, "dy": 15.0, "distance": 19.85},
            "ball_size": 24.0,
            "_prev_frame_gray": np.zeros((180, 180), dtype=np.uint8),
        }
        result = analyzer._try_local_ai_trajectory_rescue(
            np.zeros((180, 180, 3), dtype=np.uint8),
            (100, 100),
            (118, 118),
            0,
            snapshot,
            "player-region:player_body",
        )
        self.assertIsNone(result)
        self.assertEqual(analyzer.local_ai_recovery.events, [])

    def test_trajectory_rescue_rejects_static_prediction_artifact(self):
        candidate = {
            "x": 87, "y": 115, "area": 10.0, "ai_score": 0.99,
            "mode": "regular_court", "roi_rank": 1,
        }
        analyzer = self._analyzer(candidate, motion=(0.5, 4.0))
        snapshot = {
            "last_motion": {"dx": -13.0, "dy": 15.0, "distance": 19.85},
            "last_nonzero_motion": {"dx": -13.0, "dy": 15.0, "distance": 19.85},
            "ball_size": 24.0,
            "_prev_frame_gray": np.zeros((180, 180), dtype=np.uint8),
        }
        result = analyzer._try_local_ai_trajectory_rescue(
            np.zeros((180, 180, 3), dtype=np.uint8),
            (100, 100),
            (100, 100),
            1,
            snapshot,
            "player-region:player_body",
        )
        self.assertIsNone(result)
        self.assertFalse(analyzer.local_ai_recovery.events[-1]["accepted"])
        self.assertEqual(
            analyzer.local_ai_recovery.events[-1]["candidates"][0]["reject_reason"],
            "motion",
        )
'''
    tests = tests[:insert_at] + test_block + tests[insert_at:]

analyzer_path.write_text(analyzer, encoding='utf-8')
recovery_path.write_text(recovery, encoding='utf-8')
test_path.write_text(tests, encoding='utf-8')
print('patched trajectory-aware Local AI and tests')
