import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from night_serve_geometry import near_baseline
from serve_stance_guard import evaluate_serve_stance


class NightServeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        # Night-camera baseline differs from the unrelated legacy text file.
        cv2.line(self.frame, (500, 1600), (3300, 1600), (255, 255, 255), 12)

    def analyzer(self, feet_y):
        track = SimpleNamespace(bbox=(2100, feet_y-600, 300, 600),
                                center=(2250, feet_y-300), shoes=(2250, feet_y),
                                visible=True, confidence=.9)
        return SimpleNamespace(config_file='hsv_config_04_left_night.json',
                               frame_count=4375, player_tracking_enabled=True,
                               player_tracker=SimpleNamespace(tracks={'near': track}),
                               net_area_y_min=400, net_area_y_max=585)

    def test_detects_physical_baseline_without_roi_border(self):
        model = near_baseline(self.frame)
        self.assertIsNotNone(model)
        self.assertLess(abs(model['a']*2250+model['b']-1600), 20)

    def test_inside_walking_player_is_rejected(self):
        result = evaluate_serve_stance(self.analyzer(1400), (2200, 850), self.frame)
        self.assertEqual(result['decision'], 'reject')

    def test_real_behind_baseline_stance_is_allowed(self):
        result = evaluate_serve_stance(self.analyzer(1700), (2200, 850), self.frame)
        self.assertEqual(result['decision'], 'allow')

    def test_missing_night_geometry_holds_instead_of_using_other_camera(self):
        self.frame[:] = 0
        result = evaluate_serve_stance(self.analyzer(1400), (2200, 850), self.frame)
        self.assertEqual(result['decision'], 'hold')
