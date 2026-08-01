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


if __name__ == "__main__":
    unittest.main()
