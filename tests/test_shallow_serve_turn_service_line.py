import unittest

import numpy as np

from interactive_ball_analyzer import InteractiveBallAnalyzer


class ShallowServeTurnServiceLineTests(unittest.TestCase):
    def setUp(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 4118
        analyzer.point_start_frame_internal = 4077
        analyzer._serve_phase_active = True
        analyzer._awaiting_serve_bounce = True
        analyzer.ball_center = (1733, 381)
        analyzer.last_motion = {"dx": -12, "dy": 4, "distance": 12.6}
        analyzer.prev_motion = {"dx": -10, "dy": 4, "distance": 10.8}
        analyzer.serve_direction_dy = -1
        analyzer._point_target_service_side = "left"
        analyzer._serve_bounce_frame_limit = lambda: 45
        analyzer._evaluate_target_service_box_fresh = lambda point, frame, side: (
            False,
            "Serve bounce outside left service box",
            {
                "center_x": 1930.0,
                "service_y": 452.0,
                "net_y": 492.5,
                "net_margin": 38.0,
            },
        )
        analyzer._net_contact_geometry = lambda point: None
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        self.analyzer = analyzer
        self.frame = frame

    def event(self, **motion):
        values = dict(dx=-13, dy=-56, angle_jump=95.4, velocity=57.5)
        values.update(motion)
        return self.analyzer._serve_bounce_in_from_turn_candidate(
            (1720, 381 + values['dy']), self.frame, **values
        )

    def test_faster_shallow_turn_gets_service_line_slack(self):
        event = self.event()
        self.assertIsNotNone(event)
        self.assertEqual(event['point'], (1733, 381))
        self.assertTrue(event['soft_vertical_reversal'])
        self.assertTrue(event['sharp_turn'])
        self.assertTrue(event['shallow_serve_turn'])
        self.assertFalse(event['shallow_perspective_bounce'])
        self.assertTrue(event['shallow_service_box_slack'])

    def test_two_pixel_shallow_turn_is_accepted(self):
        event = self.event(dy=-2, angle_jump=65, velocity=13.2)
        self.assertIsNotNone(event)
        self.assertFalse(event['soft_vertical_reversal'])
        self.assertFalse(event['sharp_turn'])
        self.assertTrue(event['shallow_serve_turn'])

    def test_no_serve_in_after_rally_contact(self):
        self.analyzer._last_racket_contact_frame = 4100
        self.assertIsNone(self.event())

    def test_old_contact_does_not_block_new_serve(self):
        self.analyzer._last_racket_contact_frame = 4000
        self.assertIsNotNone(self.event())

    def test_wrong_target_half_is_rejected(self):
        self.analyzer._point_target_service_side = 'right'
        self.assertIsNone(self.event())

    def test_beyond_service_line_slack_is_rejected(self):
        self.analyzer.ball_center = (1733, 300)
        self.assertIsNone(self.event())

    def test_ordinary_rally_is_rejected(self):
        self.analyzer._serve_phase_active = False
        self.assertIsNone(self.event())

    def test_expired_serve_is_rejected(self):
        self.analyzer.frame_count = 4200
        self.assertIsNone(self.event())

    def test_no_turn_is_rejected_without_undefined_variable(self):
        self.assertIsNone(self.event(dy=5, angle_jump=10))

    def test_recent_static_artifact_does_not_get_extra_slack(self):
        self.analyzer._last_out_bounce_suppressed_frame = 4116
        self.analyzer._last_out_bounce_suppressed_point = (1733, 381)
        self.assertIsNone(self.event())

    def test_day_camera_does_not_get_extra_slack(self):
        self.analyzer.config_file = 'hsv_config_court2.json'
        self.assertIsNone(self.event())


if __name__ == '__main__':
    unittest.main()
