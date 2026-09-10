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


if __name__ == '__main__':
    unittest.main()
