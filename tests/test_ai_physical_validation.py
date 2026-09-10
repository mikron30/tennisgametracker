import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np

from ball_local_ai_recovery import validate_ai_step
from interactive_ball_analyzer import InteractiveBallAnalyzer


class PhysicalValidationTests(unittest.TestCase):
    def setUp(self):
        self.history = [{'frame': 10, 'pos': (80, 80)}, {'frame': 11, 'pos': (90, 80)}]
        self.motion = {'mean': 8., 'max': 70.}

    def test_jump_then_static_blob_never_changes_anchor(self):
        original = deepcopy(self.history)
        for frame in range(12, 16):
            self.assertIsNotNone(validate_ai_step((400, 50), frame, self.history,
                                                  motion={'mean': .1, 'max': 3.}))
        self.assertEqual(original, self.history)

    def test_repeated_coordinate_rejected_even_with_image_change(self):
        self.history.append({'frame': 12, 'pos': (90, 80)})
        self.assertEqual(validate_ai_step((90, 80), 13, self.history, motion=self.motion),
                         'stationary-ai-path')

    def test_moving_path_accepted(self):
        for frame in range(12, 17):
            point = (90 + 10 * (frame - 11), 80)
            self.assertIsNone(validate_ai_step(point, frame, self.history, motion=self.motion))
            self.history.append({'frame': frame, 'pos': point})

    def test_contact_reversal_and_bounce_accepted(self):
        for point in [(75, 80), (95, 65)]:
            self.assertIsNone(validate_ai_step(point, 12, self.history, motion=self.motion))

    def test_motion_cannot_excuse_teleport(self):
        self.assertEqual(validate_ai_step((800, 80), 12, self.history, motion=self.motion),
                         'unsupported-jump')

    def test_forced_high_score_rejection_preserves_motion_and_history(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer._force_local_ai_range = (12, 20)
        analyzer._force_local_ai_history = deepcopy(self.history)
        analyzer.ball_center = (90, 80)
        analyzer.ball_velocity_history = [10.] * 5
        analyzer.last_motion = {'dx': 10, 'dy': 0, 'distance': 10}
        analyzer._player_point_zone = lambda point: None
        analyzer._candidate_motion_metrics = lambda *a, **k: {'mean': .1, 'max': 3.}
        class Scorer:
            _config = {}
            def _score(self, *args):
                return [{'x': 350, 'y': 80, 'area': 5., 'ai_score': .999999}]
        analyzer.local_ai_recovery = Scorer()
        with patch('ball_ai_recovery_probe.collect_candidates', return_value=[{'x': 350, 'y': 80}]):
            for frame in range(12, 16):
                analyzer.frame_count = frame
                self.assertEqual(analyzer._force_local_ai_frame(np.zeros((200, 400, 3), np.uint8),
                                                                (90, 80)), (90, 80))
        self.assertEqual(analyzer._force_local_ai_history, self.history)
        self.assertEqual(analyzer.ball_velocity_history, [10.] * 5)
        self.assertEqual(analyzer.last_motion['distance'], 10)

    def test_first_contact_frame_prefers_ai_candidate_confirming_normal_tracker(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer._force_local_ai_range = (4100, 4110)
        analyzer._force_local_ai_history = []
        analyzer.ball_center = (2214, 860)
        analyzer.ball_size = 401.0
        analyzer.ball_velocity_history = [31.3, 38.2, 39.2, 46.3, 35.8]
        analyzer.last_motion = {'dx': -16, 'dy': 32, 'distance': 35.8}
        analyzer.motion_history = []
        analyzer._contact_local_ai_debug_normal_candidate = (2141, 743)
        analyzer._player_point_zone = lambda point: None
        analyzer._candidate_motion_metrics = lambda *a, **k: {'mean': 20., 'max': 100.}

        class Scorer:
            _config = {}
            def _score(self, *args):
                return [
                    {'x': 2221, 'y': 924, 'area': 225., 'ai_score': 1.0},
                    {'x': 2139, 'y': 740, 'area': 521., 'ai_score': 1.0},
                    {'x': 2217, 'y': 924, 'area': 220., 'ai_score': 0.999996},
                ]

        analyzer.local_ai_recovery = Scorer()
        analyzer.frame_count = 4100
        candidates = [
            {'x': 2221, 'y': 924},
            {'x': 2139, 'y': 740},
            {'x': 2217, 'y': 924},
        ]
        with patch('ball_ai_recovery_probe.collect_candidates', return_value=candidates):
            selected = analyzer._force_local_ai_frame(
                np.zeros((1200, 2400, 3), np.uint8), (2214, 860)
            )
        self.assertEqual(selected, (2139, 740))
        self.assertEqual(analyzer._force_local_ai_history[-1]['pos'], (2139, 740))


if __name__ == '__main__':
    unittest.main()
