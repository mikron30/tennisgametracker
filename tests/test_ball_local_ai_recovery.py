import unittest
from unittest.mock import patch

import numpy as np

from ball_local_ai_recovery import LocalBallAIRecovery
from interactive_ball_analyzer import InteractiveBallAnalyzer


class _DeterministicRecovery(LocalBallAIRecovery):
    """No Torch/runtime dependency: emulate already-scored AI candidates."""

    def __init__(self):
        # collect_candidates is mocked in these unit tests, but recover still
        # passes the configured HSV map through to that helper.
        self._config = {}
        self.lookback_frames = 4
        self.cooldown_frames = 1
        self.minimum_score = 0.985
        self.maximum_candidates = 8
        self.last_attempt_frame = -1000
        self.events = []

    def _score(self, image, source_frame, candidates):
        # The selected ball follows a continuous upward trajectory.  A body
        # fragment has a lower score and must not win the decision.
        y = {10: 1400, 11: 1240, 12: 1080, 13: 900}[source_frame]
        return [
            {"x": 1500, "y": y, "area": 500.0, "ai_score": 0.999999},
            {"x": 1700, "y": 1370, "area": 600.0, "ai_score": 0.990000},
        ]

    def _write_event(self, payload):
        self.events.append(payload)


class LocalBallAIRecoveryTests(unittest.TestCase):
    @staticmethod
    def _tight_roi_analyzer(frame_index=4126):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = frame_index
        analyzer.ball_center = (1680, 212)
        analyzer.ball_size = 31.0
        analyzer.last_motion = {
            "dx": -5.0,
            "dy": -8.0,
            "distance": 9.4,
            "direction_deg": -122.0,
        }
        analyzer.ball_velocity_history = [19.0, 17.1, 14.9, 12.5, 9.4]
        analyzer._local_ai_tight_roi_follow = None
        analyzer._local_ai_tight_roi_accept_frame = -1000000
        analyzer._local_ai_tight_roi_attempt_frame = -1000000
        analyzer._player_point_zone = lambda point: "player_body"
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: {
            "mean": 27.5,
            "max": 186.0,
        }
        return analyzer

    def test_tight_roi_prefers_model_at_causal_anchor_for_player_turn(self):
        analyzer = self._tight_roi_analyzer()

        class Recovery:
            calls = []

            def rank_local_roi_candidate(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {
                    "x": 1673,
                    "y": 213,
                    "area": 275.5,
                    "bbox": [1664, 198, 20, 28],
                    "ai_score": 0.922004,
                }

        analyzer.local_ai_recovery = Recovery()
        frame = np.zeros((260, 1800, 3), dtype=np.uint8)
        proposed = {
            "pos": (1676, 225),
            "area": 12.0,
            "hsv": np.zeros(3, dtype=np.uint8),
            "motion_mean": 7.4,
            "motion_max": 31.0,
            "source": "regular",
        }

        preferred = analyzer._prefer_night_tight_local_ai_candidate(
            frame, np.zeros(frame.shape[:2], dtype=np.uint8), proposed
        )

        self.assertEqual(preferred["source"], "local_ai_tight_roi")
        self.assertEqual(preferred["pos"], (1675, 204))
        self.assertEqual(preferred["area"], 31.0)
        self.assertAlmostEqual(preferred["local_ai_score"], 0.922004)
        self.assertEqual(
            analyzer.local_ai_recovery.calls[0][1]["anchor"], (1675, 204)
        )
        self.assertEqual(analyzer.local_ai_recovery.calls[0][1]["radius"], 25.0)
        self.assertEqual(analyzer._local_ai_tight_roi_follow["deadline"], 4150)
        self.assertEqual(
            analyzer._local_ai_tight_roi_follow["velocity_history"],
            preferred["local_ai_velocity_history"],
        )

    def test_tight_roi_active_lock_accepts_relative_low_score_candidate(self):
        analyzer = self._tight_roi_analyzer(frame_index=4131)
        analyzer.ball_center = (1655, 192)
        analyzer.ball_size = 7.0
        analyzer.last_motion = {"dx": -2.0, "dy": 3.0, "distance": 3.6}
        analyzer._local_ai_tight_roi_follow = {
            "start_frame": 4126,
            "deadline": 4150,
            "last_frame": 4130,
            "last_pos": (1655, 192),
            "velocity": (-2.0, 3.0),
            "velocity_history": [9.4, 10.0, 3.6],
            "outside_count": 0,
        }

        class Recovery:
            def rank_local_roi_candidate(self, *args, **kwargs):
                return {
                    "x": 1648,
                    "y": 192,
                    "area": 3.5,
                    "ai_score": 0.565648,
                }

        analyzer.local_ai_recovery = Recovery()
        frame = np.zeros((260, 1900, 3), dtype=np.uint8)
        proposed = {
            "pos": (1829, 822),
            "area": 34.0,
            "hsv": np.zeros(3, dtype=np.uint8),
            "motion_mean": 0.0,
            "motion_max": 0.0,
            "source": "player_reacq",
        }

        preferred = analyzer._prefer_night_tight_local_ai_candidate(
            frame, np.zeros(frame.shape[:2], dtype=np.uint8), proposed
        )

        self.assertEqual(preferred["source"], "local_ai_tight_roi")
        self.assertEqual(preferred["pos"], (1648, 192))
        self.assertAlmostEqual(preferred["local_ai_score"], 0.565648)
        self.assertEqual(analyzer._local_ai_tight_roi_follow["last_pos"], (1648, 192))
        self.assertEqual(
            analyzer._local_ai_tight_roi_follow["velocity_history"],
            preferred["local_ai_velocity_history"],
        )

    def test_tight_roi_low_score_requires_catastrophic_locked_jump(self):
        analyzer = self._tight_roi_analyzer(frame_index=4131)
        analyzer.ball_center = (1655, 192)
        analyzer.ball_size = 7.0
        analyzer._local_ai_tight_roi_follow = {
            "start_frame": 4126,
            "deadline": 4150,
            "last_frame": 4130,
            "last_pos": (1655, 192),
            "velocity": (-2.0, 3.0),
            "velocity_history": [9.4, 10.0, 3.6],
            "outside_count": 0,
        }

        class Recovery:
            def rank_local_roi_candidate(self, *args, **kwargs):
                return {
                    "x": 1648,
                    "y": 192,
                    "area": 3.5,
                    "ai_score": 0.565648,
                }

        analyzer.local_ai_recovery = Recovery()
        frame = np.zeros((260, 1800, 3), dtype=np.uint8)
        proposed = {
            "pos": (1655, 192),
            "area": 7.0,
            "motion_mean": 0.0,
            "motion_max": 0.0,
            "source": "tight_roi_held_marker",
            "tight_roi_force_rank": True,
        }

        result = analyzer._prefer_night_tight_local_ai_candidate(
            frame, np.zeros(frame.shape[:2], dtype=np.uint8), proposed
        )

        self.assertIs(result, proposed)
        self.assertEqual(analyzer._local_ai_tight_roi_follow["last_pos"], (1655, 192))

    def test_tight_roi_does_not_probe_a_strong_normal_candidate(self):
        analyzer = self._tight_roi_analyzer()

        class Recovery:
            calls = 0

            def rank_local_roi_candidate(self, *args, **kwargs):
                self.calls += 1
                return None

        analyzer.local_ai_recovery = Recovery()
        frame = np.zeros((260, 1800, 3), dtype=np.uint8)
        proposed = {
            "pos": (1676, 225),
            "area": 12.0,
            "motion_mean": 10.0,
            "motion_max": 50.0,
            "source": "regular",
        }

        result = analyzer._prefer_night_tight_local_ai_candidate(
            frame, np.zeros(frame.shape[:2], dtype=np.uint8), proposed
        )

        self.assertIs(result, proposed)
        self.assertEqual(analyzer.local_ai_recovery.calls, 0)

    def test_tight_roi_initial_probe_respects_spatial_guards(self):
        frame = np.zeros((260, 1800, 3), dtype=np.uint8)
        cases = (
            ("outside player", (1676, 225), lambda point: None),
            ("prediction error too large", (1737, 225), lambda point: "player_body"),
        )
        for label, position, zone in cases:
            with self.subTest(label=label):
                analyzer = self._tight_roi_analyzer()

                class Recovery:
                    calls = 0

                    def rank_local_roi_candidate(self, *args, **kwargs):
                        self.calls += 1
                        return None

                analyzer.local_ai_recovery = Recovery()
                analyzer._player_point_zone = zone
                proposed = {
                    "pos": position,
                    "area": 12.0,
                    "motion_mean": 7.4,
                    "motion_max": 31.0,
                    "source": "regular",
                }

                result = analyzer._prefer_night_tight_local_ai_candidate(
                    frame, np.zeros(frame.shape[:2], dtype=np.uint8), proposed
                )

                self.assertIs(result, proposed)
                self.assertEqual(analyzer.local_ai_recovery.calls, 0)

    def test_try_local_ai_recovery_forwards_causal_snapshot_to_hold(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4132
        analyzer._finalize_pending_provisional_static_candidate = lambda: False
        prior_gray = np.zeros((8, 8), dtype=np.uint8)
        captured = {}

        def tight_hold(frame, tracked_position, previous_gray=None):
            captured["previous_gray"] = previous_gray
            return (1643, 196)

        analyzer._try_active_tight_local_ai_hold = tight_hold
        result = analyzer._try_local_ai_recovery(
            (1648, 192),
            (1648, 192),
            15,
            pre_track_snapshot={"_prev_frame_gray": prior_gray},
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
        )

        self.assertEqual(result, (1643, 196))
        self.assertIs(captured["previous_gray"], prior_gray)

    def test_motion_metrics_can_use_causal_pre_track_frame(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4132
        current = np.zeros((21, 21), dtype=np.uint8)
        current[10, 10] = 255
        analyzer._prev_frame_gray = current.copy()

        cached_state_result = analyzer._candidate_motion_metrics(current, 10, 10, radius=2)
        causal_result = analyzer._candidate_motion_metrics(
            current,
            10,
            10,
            radius=2,
            previous_gray=np.zeros_like(current),
        )

        self.assertEqual(cached_state_result["max"], 0.0)
        self.assertEqual(causal_result["max"], 255.0)
        self.assertGreater(causal_result["mean"], 0.0)

    def test_contact_stall_arms_prediction_aware_fast_roi(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4156
        analyzer.ball_center = (1648, 771)
        analyzer.ball_size = 66.0
        analyzer.last_motion = {
            "dx": 1.0,
            "dy": 26.0,
            "distance": 26.0,
            "direction_deg": 87.8,
        }
        analyzer.ball_velocity_history = [54.0, 62.1, 68.1, 76.2, 26.0]
        analyzer._local_ai_tight_roi_follow = None
        analyzer._local_ai_tight_roi_accept_frame = -1000000
        analyzer._local_ai_tight_roi_attempt_frame = -1000000
        analyzer._player_point_zone = lambda point: None
        analyzer._point_in_player_contact_corridor = lambda point: True

        selections = [
            {
                "x": 1647,
                "y": 770,
                "area": 111.5,
                "bbox": [1638, 764, 18, 14],
                "ai_score": 0.999986,
                "roi_candidates": 1,
            },
            {
                "x": 1653,
                "y": 748,
                "area": 270.0,
                "bbox": [1646, 734, 14, 28],
                "ai_score": 1.0,
                "roi_candidates": 1,
            },
            {
                "x": 1672,
                "y": 658,
                "area": 248.5,
                "bbox": [1666, 646, 14, 26],
                "ai_score": 1.0,
                "roi_candidates": 4,
            },
        ]

        class Recovery:
            calls = []

            def rank_local_roi_candidate(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return selections[len(self.calls) - 1]

        analyzer.local_ai_recovery = Recovery()
        motion = iter(((16.945, 85.0), (52.422, 110.0), (57.225, 114.0)))
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: dict(
            zip(("mean", "max"), next(motion))
        )
        frame = np.zeros((850, 1800, 3), dtype=np.uint8)

        first = analyzer._prefer_night_tight_local_ai_candidate(
            frame,
            np.zeros(frame.shape[:2], dtype=np.uint8),
            {
                "pos": (1649, 773),
                "area": 77.0,
                "motion_mean": 12.3,
                "motion_max": 86.0,
                "source": "regular",
            },
        )
        self.assertEqual(first["pos"], (1647, 770))
        self.assertEqual(analyzer._local_ai_tight_roi_follow["mode"], "fast_contact")
        self.assertEqual(analyzer.local_ai_recovery.calls[0][1]["radius"], 25.0)

        analyzer.ball_center = first["pos"]
        analyzer.ball_size = first["area"]
        analyzer.last_motion = first["local_ai_motion_override"]
        analyzer.ball_velocity_history = first["local_ai_velocity_history"]
        analyzer.frame_count = 4157
        second = analyzer._prefer_night_tight_local_ai_candidate(
            frame,
            np.zeros(frame.shape[:2], dtype=np.uint8),
            {
                "pos": (1688, 787),
                "area": 37.5,
                "motion_mean": 22.9,
                "motion_max": 108.0,
                "source": "regular",
            },
        )
        self.assertEqual(second["pos"], (1653, 748))
        self.assertEqual(analyzer.local_ai_recovery.calls[1][1]["anchor"], (1647, 770))
        self.assertEqual(analyzer.local_ai_recovery.calls[1][1]["radius"], 32.0)

        analyzer.ball_center = second["pos"]
        analyzer.ball_size = second["area"]
        analyzer.last_motion = second["local_ai_motion_override"]
        analyzer.ball_velocity_history = second["local_ai_velocity_history"]
        analyzer.frame_count = 4158
        third = analyzer._prefer_night_tight_local_ai_candidate(
            frame,
            np.zeros(frame.shape[:2], dtype=np.uint8),
            {
                "pos": (1693, 766),
                "area": 38.0,
                "motion_mean": 27.1,
                "motion_max": 169.0,
                "source": "regular",
            },
        )
        self.assertEqual(third["pos"], (1672, 658))
        self.assertEqual(analyzer.local_ai_recovery.calls[2][1]["anchor"], (1659, 726))
        self.assertEqual(analyzer.local_ai_recovery.calls[2][1]["radius"], 75.0)
        self.assertEqual(analyzer._local_ai_tight_roi_follow["velocity"], (19.0, -90.0))

    def test_fast_roi_accepts_single_prediction_aligned_top_edge_ball(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4167
        analyzer.ball_center = (1841, 84)
        analyzer.ball_size = 223.5
        analyzer.last_motion = {"dx": 20.0, "dy": -58.0, "distance": 61.4}
        analyzer.ball_velocity_history = [73.0, 72.0, 68.0, 67.0, 61.4]
        analyzer._local_ai_tight_roi_follow = {
            "start_frame": 4156,
            "deadline": 4176,
            "last_frame": 4166,
            "last_pos": (1841, 84),
            "velocity": (20.0, -58.0),
            "velocity_history": [73.0, 72.0, 68.0, 67.0, 61.4],
            "outside_count": 0,
            "mode": "fast_contact",
            "ai_accept_count": 9,
        }
        analyzer._local_ai_tight_roi_accept_frame = -1000000
        analyzer._local_ai_tight_roi_attempt_frame = -1000000
        analyzer._player_point_zone = lambda point: None
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: {
            "mean": 122.92,
            "max": 239.0,
        }

        class Recovery:
            calls = []

            def rank_local_roi_candidate(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {
                    "x": 1861,
                    "y": 30,
                    "area": 253.5,
                    "bbox": [1854, 19, 17, 24],
                    "ai_score": 0.499844,
                    "roi_candidates": 1,
                }

        analyzer.local_ai_recovery = Recovery()
        frame = np.zeros((240, 1900, 3), dtype=np.uint8)
        preferred = analyzer._prefer_night_tight_local_ai_candidate(
            frame,
            np.zeros(frame.shape[:2], dtype=np.uint8),
            {
                "pos": (1730, 653),
                "area": 38.0,
                "motion_mean": 72.8,
                "motion_max": 241.0,
                "source": "player_reacq",
            },
        )

        self.assertEqual(preferred["pos"], (1861, 30))
        self.assertAlmostEqual(preferred["local_ai_score"], 0.499844)
        self.assertEqual(analyzer.local_ai_recovery.calls[0][1]["anchor"], (1861, 26))
        self.assertEqual(analyzer.local_ai_recovery.calls[0][1]["radius"], 35.0)
        self.assertEqual(preferred["local_ai_motion_override"]["dy"], -54.0)
        self.assertEqual(analyzer._local_ai_tight_roi_follow["deadline"], 4167)

    def test_fast_roi_catastrophic_normal_jump_keeps_standard_score_floor(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4164
        analyzer.ball_center = (1777, 271)
        analyzer.ball_size = 171.5
        analyzer.last_motion = {"dx": 23.0, "dy": -65.0, "distance": 68.9}
        analyzer.ball_velocity_history = [74.0, 73.0, 70.0, 68.9]
        analyzer._local_ai_tight_roi_follow = {
            "start_frame": 4156,
            "deadline": 4176,
            "last_frame": 4163,
            "last_pos": (1777, 271),
            "velocity": (23.0, -65.0),
            "velocity_history": [74.0, 73.0, 70.0, 68.9],
            "outside_count": 0,
            "mode": "fast_contact",
            "ai_accept_count": 7,
        }
        analyzer._local_ai_tight_roi_accept_frame = -1000000
        analyzer._local_ai_tight_roi_attempt_frame = -1000000
        analyzer._player_point_zone = lambda point: None
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: {
            "mean": 55.0,
            "max": 180.0,
        }

        class Recovery:
            def rank_local_roi_candidate(self, *args, **kwargs):
                return {
                    "x": 1800,
                    "y": 206,
                    "area": 160.0,
                    "ai_score": 0.60,
                    "roi_candidates": 1,
                }

        analyzer.local_ai_recovery = Recovery()
        proposed = {
            "pos": (1680, 800),
            "area": 35.0,
            "motion_mean": 25.0,
            "motion_max": 120.0,
            "source": "player_reacq",
        }
        result = analyzer._prefer_night_tight_local_ai_candidate(
            np.zeros((900, 2000, 3), dtype=np.uint8),
            np.zeros((900, 2000), dtype=np.uint8),
            proposed,
        )

        self.assertIs(result, proposed)
        self.assertEqual(analyzer._local_ai_tight_roi_follow["last_pos"], (1777, 271))

    def test_tight_roi_ranker_returns_relative_winner_below_recovery_threshold(self):
        recovery = _DeterministicRecovery()
        recovery._score = lambda image, source_frame, candidates: [
            {**candidates[0], "ai_score": 0.922004},
            {**candidates[1], "ai_score": 0.878805},
        ]
        candidates = [
            {"x": 1673, "y": 213, "area": 275.5, "mode": "alt2"},
            {"x": 1667, "y": 196, "area": 27.0, "mode": "h_10"},
        ]

        with patch(
            "ball_local_ai_recovery.collect_candidates", return_value=candidates
        ) as collect:
            selected = recovery.rank_local_roi_candidate(
                4126,
                np.zeros((32, 32, 3), dtype=np.uint8),
                anchor=(1675, 204),
                radius=25.0,
            )

        self.assertIsNotNone(selected)
        self.assertEqual((selected["x"], selected["y"]), (1673, 213))
        self.assertLess(selected["ai_score"], recovery.minimum_score)
        self.assertEqual(selected["roi_anchor"], (1675, 204))
        self.assertEqual(selected["roi_radius"], 25.0)
        self.assertEqual(selected["roi_candidates"], 2)
        self.assertAlmostEqual(selected["roi_score_margin"], 0.043199, places=6)
        self.assertEqual(collect.call_args.kwargs["around"], (1675, 204))
        self.assertEqual(collect.call_args.kwargs["radius"], 25.0)

    def test_tight_roi_ranker_returns_none_when_roi_has_no_contour(self):
        recovery = _DeterministicRecovery()
        with patch(
            "ball_local_ai_recovery.collect_candidates", return_value=[]
        ), patch.object(recovery, "_score") as score:
            selected = recovery.rank_local_roi_candidate(
                4127,
                np.zeros((32, 32, 3), dtype=np.uint8),
                anchor=(1670, 196),
                radius=25.0,
            )

        self.assertIsNone(selected)
        score.assert_not_called()

    def test_recover_requires_and_returns_a_continuous_path(self):
        recovery = _DeterministicRecovery()
        samples = [
            {"frame": frame, "image": np.zeros((32, 32, 3), dtype=np.uint8), "normal_position": None}
            for frame in range(10, 14)
        ]
        with patch("ball_local_ai_recovery.collect_candidates", return_value=[{"x": 1500, "y": 1000}]):
            repaired = recovery.recover(
                13,
                samples,
                predicted_position=(1495, 1450),
                player_zone=lambda point: None,
                reason="player-region:player_body",
            )
        self.assertIsNotNone(repaired)
        self.assertEqual((repaired["x"], repaired["y"]), (1500, 900))
        self.assertTrue(recovery.events[-1]["accepted"])

    def test_body_candidate_needs_near_conclusive_score(self):
        recovery = _DeterministicRecovery()
        selected = recovery._best_candidate(
            [{"x": 1500, "y": 1000, "ai_score": 0.990000}],
            anchor=(1500, 1100),
            player_zone=lambda point: "player_body",
        )
        self.assertIsNone(selected)

    def test_recover_rejects_path_made_entirely_of_player_body(self):
        recovery = _DeterministicRecovery()
        samples = [
            {"frame": frame, "image": np.zeros((32, 32, 3), dtype=np.uint8), "normal_position": None}
            for frame in range(10, 14)
        ]
        with patch("ball_local_ai_recovery.collect_candidates", return_value=[{"x": 1500, "y": 1000}]):
            repaired = recovery.recover(
                13,
                samples,
                predicted_position=(1495, 1450),
                player_zone=lambda point: "player_body",
                reason="player-region:player_body",
            )
        self.assertIsNone(repaired)
        self.assertEqual(recovery.events[-1]["rejection"], "all-player-body-path")

    def test_all_body_recovery_discards_unconfirmed_serve(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 3044
        analyzer.point_start_frame_internal = 2961
        analyzer._awaiting_serve_bounce = True
        analyzer._point_hit_count = 0
        analyzer._local_ai_all_body_rejections = 0
        analyzer._local_ai_recovery_count = 0
        analyzer._discard_provisional_serve_from_ai = False
        analyzer._local_ai_frame_buffer = []
        analyzer._local_ai_recovery_reason = lambda *args: "player-region:player_body"

        class RejectionRecovery:
            last_rejection = "all-player-body-path"

            def recover(self, *args, **kwargs):
                return None

        analyzer.local_ai_recovery = RejectionRecovery()
        result = analyzer._try_local_ai_recovery((1684, 910), (1679, 896), 0)

        self.assertEqual(result, (1679, 896))
        self.assertTrue(analyzer._discard_provisional_serve_from_ai)

    def test_all_body_recovery_does_not_discard_a_previously_repaired_serve(self):
        """A contact/body occlusion cannot erase a real recovered flight."""
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 2876
        analyzer.point_start_frame_internal = 2780
        analyzer._awaiting_serve_bounce = True
        analyzer._point_hit_count = 0
        analyzer._local_ai_all_body_rejections = 0
        analyzer._local_ai_recovery_count = 1
        analyzer._discard_provisional_serve_from_ai = False
        analyzer._local_ai_frame_buffer = []
        analyzer._local_ai_recovery_reason = lambda *args: "player-region:player_body"

        class RejectionRecovery:
            last_rejection = "all-player-body-path"

            def recover(self, *args, **kwargs):
                return None

        analyzer.local_ai_recovery = RejectionRecovery()
        analyzer._try_local_ai_recovery((1700, 400), (2006, 1336), 0)

        self.assertFalse(analyzer._discard_provisional_serve_from_ai)

    def test_recover_rejects_a_static_false_path(self):
        recovery = _DeterministicRecovery()
        recovery._score = lambda image, source_frame, candidates: [
            {"x": 1409, "y": 156, "area": 55.5, "ai_score": 0.9999}
        ]
        samples = [
            {"frame": frame, "image": np.zeros((32, 32, 3), dtype=np.uint8), "normal_position": None}
            for frame in range(10, 14)
        ]
        with patch("ball_local_ai_recovery.collect_candidates", return_value=[{"x": 1409, "y": 156}]):
            repaired = recovery.recover(
                13,
                samples,
                predicted_position=(1410, 157),
                player_zone=lambda point: None,
                reason="player-region:player_body",
            )
        self.assertIsNone(repaired)
        self.assertEqual(recovery.events[-1]["rejection"], "static-recovery-path")

    def test_verified_handoff_protects_normal_tracking_without_rescoring(self):
        """A verified recovery holds endpoints but does not force AI per frame."""
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 2309
        analyzer.local_ai_recovery = object()
        analyzer._local_ai_frame_buffer = []
        analyzer._local_ai_follow_until_frame = -1
        analyzer._local_ai_handoff_deadline_frame = -1
        analyzer._local_ai_recovery_count = 0
        analyzer.ball_size = 100.0
        analyzer._pending_rally_end_reason = "stale"
        analyzer._pending_rally_end_frame = 2308
        analyzer._player_point_zone = lambda point: None

        class Recovery:
            lookback_frames = 4

            def recover(self, *args, **kwargs):
                return {"x": 1604, "y": 1154, "area": 401.5, "ai_score": 1.0}

        analyzer.local_ai_recovery = Recovery()
        analyzer._local_ai_recovery_reason = lambda *args: "player-region:player_body"
        repaired = analyzer._try_local_ai_recovery((1625, 1097), (1604, 1154), 1)
        self.assertEqual(repaired, (1604, 1154))
        self.assertEqual(analyzer._local_ai_handoff_deadline_frame, 2323)
        self.assertEqual(analyzer._local_ai_follow_until_frame, 2313)

        analyzer.frame_count = 2313
        analyzer._local_ai_recovery_reason = InteractiveBallAnalyzer._local_ai_recovery_reason.__get__(
            analyzer, InteractiveBallAnalyzer
        )
        self.assertIsNone(analyzer._local_ai_recovery_reason((1776, 825), (1830, 727), 0))
        self.assertEqual(analyzer._local_ai_follow_until_frame, 2313)

    @staticmethod
    def _active_upper_side_return_wait(frame_index):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = frame_index
        analyzer.ball_center = (1861, 30)
        analyzer._top_return_wait_frames = 80
        analyzer._top_return_anchor = (1861, 30)
        analyzer._top_return_origin_frame = 4168
        analyzer._top_return_mode = "upper_side"
        analyzer._top_return_exit_dx = 20.0
        return analyzer

    def test_top_return_wait_rejects_weak_top_line_blob(self):
        analyzer = self._active_upper_side_return_wait(4201)
        shape = (2160, 3840, 3)

        valid, reason = analyzer._top_return_reentry_ok(
            (1300, 41), 34.0, 6.7, 36.0, shape
        )
        override, override_reason = (
            analyzer._top_return_player_reacq_jump_override_ok(
                (1300, 41), 34.0, 6.7, 36.0, "alt6", shape
            )
        )

        self.assertFalse(valid)
        self.assertIsNotNone(reason)
        self.assertFalse(override)
        self.assertIn("source alt6", override_reason)

        trusted_weak, trusted_weak_reason = (
            analyzer._top_return_player_reacq_jump_override_ok(
                (1300, 41), 34.0, 6.7, 36.0, "primary", shape
            )
        )
        self.assertFalse(trusted_weak)
        self.assertIn("lacks strong ball evidence", trusted_weak_reason)

    def test_top_return_wait_accepts_strong_ball_after_offscreen_flight(self):
        analyzer = self._active_upper_side_return_wait(4203)
        shape = (2160, 3840, 3)

        valid, reason = analyzer._top_return_reentry_ok(
            (2383, 27), 127.0, 93.8, 224.0, shape
        )
        override, override_reason = (
            analyzer._top_return_player_reacq_jump_override_ok(
                (2383, 27), 127.0, 93.8, 224.0, "primary", shape
            )
        )

        self.assertTrue(valid, reason)
        self.assertTrue(override, override_reason)

        analyzer._top_return_wait_frames = 0
        inactive, inactive_reason = (
            analyzer._top_return_player_reacq_jump_override_ok(
                (2383, 27), 127.0, 93.8, 224.0, "primary", shape
            )
        )
        self.assertFalse(inactive)
        self.assertEqual(inactive_reason, "top-return wait inactive")

    def test_top_return_strong_motion_still_obeys_exit_direction(self):
        analyzer = self._active_upper_side_return_wait(4203)
        valid, reason = analyzer._top_return_reentry_ok(
            (1300, 27), 127.0, 93.8, 224.0, (2160, 3840, 3)
        )

        self.assertFalse(valid)
        self.assertIn("rightward reentry progress", reason)


if __name__ == "__main__":
    unittest.main()
