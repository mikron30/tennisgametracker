"""Replay the f4292 false turn from serve_check/tracker.log, without video."""
import math
import unittest
from copy import deepcopy
import numpy as np
from interactive_ball_analyzer import InteractiveBallAnalyzer


class OutBounceMotionConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.a = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        self.a.frame_count = 4292
        self.a.point_start_frame_internal = 4077
        self.a.net_area_y_min = 400
        self.a.net_area_y_max = 585
        self.a.serve_direction_dy = -1
        self.a.ball_size = 5.0
        self.a.config_file = 'hsv_config_04_left_night.json'
        self.a._serve_bounce_frame_limit = lambda: 45
        self.a._point_outside_singles_sidelines = lambda p, f: (True, 'right', 1369.2, 2449.0)
        self.a._build_singles_sideline_model = lambda f: {'margin': 12.0}
        self.a._sideline_line_contact_override = lambda *args, **kwargs: False
        self.frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        self.a.prev_motion = self.motion(26, 16, -129.7)
        self.a.last_motion = self.motion(24, 8, -12.5)

    @staticmethod
    def motion(dx, dy, direction=None):
        return dict(dx=dx, dy=dy, distance=math.hypot(dx, dy),
                    direction_deg=math.degrees(math.atan2(dy, dx)) if direction is None else direction)

    def test_logged_f4292_mixed_motion_cannot_end_point(self):
        before = deepcopy((self.a.prev_motion, self.a.last_motion))
        self.assertEqual(self.a._detect_out_of_court_bounce((2912, 568), self.frame), (False, None))
        self.assertEqual(before, (self.a.prev_motion, self.a.last_motion))

    def test_consistent_real_outward_bounce_remains_out(self):
        self.a.prev_motion = self.motion(20, 30)
        self.a.last_motion = self.motion(20, -30)
        self.assertEqual(self.a._detect_out_of_court_bounce((2912, 700), self.frame),
                         (True, 'Ball bounced out of court (right sideline)'))

    def test_wrapped_equivalent_direction_remains_out(self):
        self.a.prev_motion = self.motion(20, 30)
        self.a.last_motion = self.motion(20, -30)
        self.a.last_motion['direction_deg'] += 360
        self.assertTrue(self.a._detect_out_of_court_bounce((2912, 700), self.frame)[0])
