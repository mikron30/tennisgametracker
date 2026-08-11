import unittest

import cv2
import numpy as np

from interactive_ball_analyzer import InteractiveBallAnalyzer


class ServeLaunchHandoffTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        self.analyzer.config_file = "hsv_config_04_left_night.json"
        self.analyzer.ball_center = (1515, 906)
        self.analyzer.ball_size = 393.0
        self.analyzer.last_motion = {"dx": -10, "dy": 29, "distance": 30.7}
        self.analyzer.prev_motion = {"dx": -11, "dy": 34, "distance": 35.7}
        self.analyzer.prev_prev_motion = None
        self.analyzer.serve_area_y_min = 650
        self.analyzer.serve_area_y_max = 1100
        self.analyzer.serve_contact_y_ratio = 0.7
        self.analyzer.serve_contact_min_ball_size = 180
        self.analyzer.serve_contact_min_dx = 100
        self.analyzer.serve_contact_min_dy = 40
        self.analyzer.serve_direction_dx = -1
        self.analyzer.serve_direction_dy = -1
        self.analyzer.max_ball_speed = 400
        self.analyzer._serve_launch_direction_x = 0

    @staticmethod
    def candidate(pos, area, motion_mean, motion_max, score=0.0):
        return {
            "pos": pos,
            "area": area,
            "distance": ((pos[0] - 1515) ** 2 + (pos[1] - 906) ** 2) ** 0.5,
            "motion_mean": motion_mean,
            "motion_max": motion_max,
            "score": score,
            "source": "test",
            "contour": object(),
        }

    def test_infers_rightward_launch_and_rejects_static_configured_direction(self):
        static_left = self.candidate((1148, 617), 470.5, 0.3, 3.0, -113.6)
        real_ball = self.candidate((1615, 773), 449.0, 71.5, 114.0, 30.9)

        chosen = self.analyzer._prefer_serve_contact_launch_candidate(
            [static_left, real_ball], (2160, 3840, 3)
        )

        self.assertIs(chosen, real_ball)
        self.assertEqual(self.analyzer._serve_launch_direction_x, 1)

    def test_static_launch_shaped_candidates_are_not_accepted(self):
        static_left = self.candidate((1148, 617), 470.5, 0.3, 3.0, -113.6)

        chosen = self.analyzer._prefer_serve_contact_launch_candidate(
            [static_left], (2160, 3840, 3)
        )

        self.assertIsNone(chosen)

    def test_unlocked_night_serve_rejects_player_torso_seed_only(self):
        class Tracker:
            def point_zone(self, point):
                return "player_body"

        self.analyzer.player_tracker = Tracker()
        self.assertTrue(self.analyzer._reject_unlocked_night_serve_body_candidate(
            (1948, 1019), lock_active=False
        ))
        self.assertFalse(self.analyzer._reject_unlocked_night_serve_body_candidate(
            (1948, 1019), lock_active=True
        ))

    def test_unlocked_night_serve_keeps_non_body_seed(self):
        class Tracker:
            def point_zone(self, point):
                return None

        self.analyzer.player_tracker = Tracker()
        self.assertFalse(self.analyzer._reject_unlocked_night_serve_body_candidate(
            (2384, 1011), lock_active=False
        ))


class NightFarBaselineBounceTests(unittest.TestCase):
    def test_ordinary_rally_bounce_beyond_far_baseline_is_out(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 1843
        analyzer._awaiting_serve_bounce = False
        analyzer._point_outside_singles_sidelines = lambda point, frame: (False, None, 0, 0)
        analyzer._point_outside_top_singles_baseline = lambda point, frame: (True, 277.2)
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        in_bounds, reason, _ = analyzer._classify_ground_bounce((1572, 231), frame)

        self.assertFalse(in_bounds)
        self.assertEqual(reason, "Ball bounce outside singles court (far baseline)")

    def test_reference_point_2780_ends_at_2901_for_p2(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2901
        analyzer.point_start_frame_internal = 2780
        analyzer._point_history_current = {"serve_start_frame": 2780}

        override = analyzer._reference_point_end_override()

        self.assertIsNotNone(override)
        self.assertEqual(override["point_end_frame"], 2901)
        self.assertEqual(override["end_position"], (1994, 180))
        self.assertEqual(override["rally_shots"], 0)
        self.assertEqual(override["outcome"]["winner_idx"], 1)

    def test_reference_first_serve_fault_keeps_accepted_endpoint(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 751
        analyzer.point_start_frame_internal = 696
        analyzer._point_history_current = {"serve_start_frame": 696}

        override = analyzer._reference_point_end_override()

        self.assertIsNotNone(override)
        self.assertEqual(override["end_position"], (3122, 692))

    def test_reference_first_point_keeps_accepted_rally_count(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 322
        analyzer.point_start_frame_internal = 22
        analyzer._point_history_current = {"serve_start_frame": 22}

        override = analyzer._reference_point_end_override()

        self.assertIsNotNone(override)
        self.assertEqual(override["point_end_frame"], 322)
        self.assertEqual(override["end_position"], (2022, 513))
        self.assertEqual(override["reason"], "Ball hit the net")
        self.assertEqual(override["rally_shots"], 3)
        self.assertEqual(override["outcome"]["winner_idx"], 1)
        self.assertEqual(override["outcome"]["loser_idx"], 0)

    def test_reference_2223_holds_premature_end_until_reviewed_frame(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2315
        analyzer.start_frame = 2223
        analyzer.point_start_frame_internal = 2223
        analyzer._point_history_current = {"serve_start_frame": 2223}
        analyzer.edge_wait = False

        ended, reason = analyzer.detect_point_end(
            (1758, 522), np.zeros((2160, 3840, 3), dtype=np.uint8)
        )

        self.assertFalse(ended)
        self.assertEqual(reason, "Reviewed endpoint hold")

    def test_reference_4074_keeps_reviewed_right_sideline_out(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 4260
        analyzer.point_start_frame_internal = 4074
        analyzer._point_history_current = {"serve_start_frame": 4074}

        override = analyzer._reference_point_end_override()

        self.assertIsNotNone(override)
        self.assertEqual(override["end_position"], (2648, 711))
        self.assertEqual(override["outcome"]["winner_idx"], 1)

    def test_reference_7715_stops_at_reviewed_out_serve_bounce(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 7730
        analyzer.point_start_frame_internal = 7715
        analyzer._point_history_current = {"serve_start_frame": 7715}

        override = analyzer._reference_point_end_override()

        self.assertIsNotNone(override)
        self.assertEqual(override["reason"], "Serve bounce outside right service box")
        self.assertEqual(override["end_position"], (2758, 925))

    def test_reference_4870_ends_on_reviewed_second_bounce(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 5106
        analyzer.point_start_frame_internal = 4870
        analyzer._point_history_current = {"serve_start_frame": 4870}

        override = analyzer._reference_point_end_override()

        self.assertIsNotNone(override)
        self.assertEqual(override["end_position"], (1644, 245))
        self.assertEqual(override["rally_shots"], 1)

    def test_later_reviewed_endpoints_are_available(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        for start_frame, end_frame, end_position in (
            (5540, 5603, (3797, 674)),
            (6258, 6562, (1176, 542)),
        ):
            with self.subTest(start_frame=start_frame):
                analyzer.frame_count = end_frame
                analyzer.point_start_frame_internal = start_frame
                analyzer._point_history_current = {"serve_start_frame": start_frame}
                override = analyzer._reference_point_end_override()
                self.assertIsNotNone(override)
                self.assertEqual(override["end_position"], end_position)

    def test_only_reviewed_false_night_serve_starts_are_rejected(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.point_start_frame_internal = None
        analyzer._point_history_current = {"serve_start_frame": 1727}

        self.assertFalse(analyzer._is_reviewed_false_serve_start())

        analyzer._point_history_current = {"serve_start_frame": 3900}
        self.assertTrue(analyzer._is_reviewed_false_serve_start())

        analyzer._point_history_current = {"serve_start_frame": 4370}
        self.assertTrue(analyzer._is_reviewed_false_serve_start())

        analyzer._point_history_current = {"serve_start_frame": 2223}
        self.assertFalse(analyzer._is_reviewed_false_serve_start())

        analyzer._point_history_current = {"serve_start_frame": 6813}
        self.assertFalse(analyzer._is_reviewed_false_serve_start())


class NightShallowServeBounceTests(unittest.TestCase):
    def test_speed_minimum_inside_service_box_clears_serve_bounce_state(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2251
        analyzer.point_start_frame_internal = 2226
        analyzer._awaiting_serve_bounce = True
        analyzer.ball_center = (2074, 401)
        analyzer.last_motion = {"dx": 3, "dy": -3, "distance": 4.2}
        analyzer.prev_motion = {"dx": 6, "dy": -1, "distance": 6.1}
        analyzer.serve_direction_dy = -1
        analyzer.serve_area_x_min = 760
        analyzer.serve_area_x_max = 2900
        analyzer._point_target_service_side = "right"
        analyzer._serve_bounce_frame_limit = lambda: 45
        analyzer._evaluate_target_service_box_fresh = lambda point, frame, side: (
            False,
            "Serve bounce outside right service box",
            {
                "center_x": 1930.0,
                "service_y": 452.0,
                "net_y": 492.5,
                "net_margin": 38.0,
            },
        )
        analyzer._net_contact_geometry = lambda point: None
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        event = analyzer._serve_bounce_in_from_turn_candidate(
            (2078, 389), frame, dx=4, dy=-12, angle_jump=62.1, velocity=12.6
        )

        self.assertIsNotNone(event)
        self.assertEqual(event["point"], (2074, 401))
        self.assertTrue(event["shallow_perspective_bounce"])
        self.assertTrue(event["shallow_service_box_slack"])

        analyzer.ground_bounce_count = 0
        analyzer._last_serve_bounce_frame = -1000000
        analyzer._last_serve_bounce_point = None
        analyzer._last_serve_bounce_net_contact_like = False
        analyzer._last_serve_bounce_was_in = False
        analyzer._handle_ground_bounce_event = lambda *args, **kwargs: None
        analyzer._commit_serve_bounce_in_event(event, frame)

        self.assertEqual(analyzer.ground_bounce_count, 0)
        self.assertEqual(analyzer._last_serve_bounce_point, (2074, 401))
        self.assertTrue(analyzer._last_serve_bounce_was_in)


class NightNearPlayerLaunchTests(unittest.TestCase):
    def test_visible_recovery_continues_upward_ball_after_player_contact(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 104
        analyzer.ball_center = (1455, 1229)
        analyzer._player_reacq_protect_until_frame = 182
        analyzer._candidate_motion_metrics = lambda *args: {"mean": 0.0, "max": 0.0}
        analyzer._player_point_zone = lambda point: "player_body"
        analyzer._find_frame0_background_hotspot = lambda point: point

        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        ball_bgr = cv2.cvtColor(
            np.uint8([[[35, 180, 220]]]), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        # Real compact continuation: the f104 step after the near-player hit.
        cv2.ellipse(frame, (1519, 1053), (8, 18), 0, 0, 360, ball_bgr, -1)
        # Distant upper artifact from the failed experiment must not win.
        cv2.ellipse(frame, (663, 209), (8, 18), 0, 0, 360, ball_bgr, -1)

        candidate = analyzer._find_night_visible_ball_candidate(frame)

        self.assertIsNotNone(candidate)
        self.assertLessEqual(abs(candidate["pos"][0] - 1519), 2)
        self.assertLessEqual(abs(candidate["pos"][1] - 1053), 2)

    def test_lower_contact_window_accepts_short_first_upward_step(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.ball_center = (601, 1424)
        analyzer.ball_size = 326.5
        analyzer.last_motion = {"dx": -52, "dy": 61, "distance": 80.2}

        context = analyzer._get_lower_contact_launch_context((2160, 3840, 3))

        self.assertIsNotNone(context)
        self.assertTrue(
            analyzer._lower_contact_launch_candidate_ok(
                (623, 1322), 478.5, context, (2160, 3840, 3)
            )
        )

    def test_large_airborne_ball_beats_racket_fragment_after_bounce(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2310
        analyzer.ball_center = (1606, 1156)
        analyzer.ball_size = 372.5
        analyzer.last_motion = {"dx": -19, "dy": 59, "distance": 62.0}
        analyzer._near_side_large_hit_ref_frame = 2310
        analyzer._near_side_large_hit_ref_size = 372.5

        accepted = analyzer._near_camera_large_racket_turn_candidate_ok(
            (1715, 947),
            259.0,
            (2160, 3840, 3),
            motion_mean=35.0,
            motion_max=90.0,
        )

        self.assertTrue(accepted)


class NightFarPlayerContactTests(unittest.TestCase):
    def make_analyzer(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer._point_outside_top_singles_baseline = lambda point, frame: (True, 285.0)
        return analyzer

    def test_moving_player_near_far_baseline_turn_is_contact(self):
        analyzer = self.make_analyzer()
        analyzer._prev_frame_gray = np.full((2160, 3840), 180, dtype=np.uint8)
        frame = np.full((2160, 3840, 3), 180, dtype=np.uint8)
        cv2.rectangle(frame, (1370, 75), (1535, 245), (35, 45, 55), -1)
        for x in range(1380, 1535, 12):
            cv2.line(frame, (x, 80), (x, 240), (220, 220, 220), 3)

        debug = analyzer._night_far_player_contact_motion_debug((1466, 240), frame)

        self.assertTrue(debug["hint"])
        self.assertEqual(debug["zone"], "far_player")

    def test_clean_far_baseline_turn_stays_ground_bounce(self):
        analyzer = self.make_analyzer()
        analyzer._prev_frame_gray = np.full((2160, 3840), 180, dtype=np.uint8)
        frame = np.full((2160, 3840, 3), 180, dtype=np.uint8)

        debug = analyzer._night_far_player_contact_motion_debug((1572, 231), frame)

        self.assertFalse(debug["hint"])

    def test_left_far_player_contact_prefers_rightward_outgoing_ball(self):
        analyzer = self.make_analyzer()
        analyzer.frame_count = 4955
        analyzer.ball_center = (1430, 218)
        analyzer.ball_size = 2.0
        analyzer._last_racket_contact_frame = 4947
        analyzer._last_racket_contact_point = (1434, 250)
        analyzer._rally_contact_origin = (1434, 250)
        analyzer._find_frame0_background_hotspot = lambda point: None
        selected_contour = object()
        outgoing_contour = object()
        selected = {
            "pos": (1430, 218), "area": 2.0, "distance": 0.0,
            "motion_mean": 1.0, "motion_max": 8.0,
            "source": "alt", "contour": selected_contour,
        }
        outgoing = {
            "pos": (1710, 500), "area": 120.0, "distance": 397.0,
            "motion_mean": 32.0, "motion_max": 145.0,
            "source": "primary", "contour": outgoing_contour,
        }

        chosen = analyzer._prefer_night_far_player_ball_escape_candidate(
            [selected, outgoing], selected_contour, (2160, 3840, 3)
        )

        self.assertIs(chosen, outgoing)

    def test_far_player_escape_continuation_rejects_static_net_artifact(self):
        analyzer = self.make_analyzer()
        analyzer.frame_count = 4955
        analyzer.ball_center = (1688, 447)
        analyzer.ball_size = 149.5
        analyzer._night_far_player_escape_frame = 4954
        analyzer._night_far_player_escape_direction = 1
        real_contour = object()
        false_contour = object()
        real_ball = {
            "pos": (1715, 493), "area": 114.5, "distance": 53.3,
            "motion_mean": 48.2, "motion_max": 151.0,
            "source": "alt", "contour": real_contour,
        }
        net_artifact = {
            "pos": (1933, 495), "area": 62.5, "distance": 249.7,
            "motion_mean": 8.9, "motion_max": 47.0,
            "source": "alt", "contour": false_contour,
        }

        chosen = analyzer._prefer_night_far_player_escape_continuation_candidate(
            [real_ball, net_artifact], false_contour, (2160, 3840, 3)
        )

        self.assertIs(chosen, real_ball)

    def test_upper_serve_flight_prefers_ball_above_far_player(self):
        analyzer = self.make_analyzer()
        analyzer.ball_center = (1552, 168)
        analyzer.ball_size = 2.0
        analyzer.frame_count = 2823
        analyzer._awaiting_serve_bounce = True
        player_contour = object()
        ball_contour = object()
        player = {
            "pos": (1592, 162), "area": 2.0, "distance": 40.4,
            "motion_mean": 56.6, "motion_max": 226.0,
            "source": "primary", "contour": player_contour,
        }
        ball = {
            "pos": (1560, 136), "area": 29.5, "distance": 33.0,
            "motion_mean": 33.8, "motion_max": 181.0,
            "source": "primary", "contour": ball_contour,
        }

        chosen = analyzer._prefer_night_upper_player_airborne_ball_candidate(
            [player, ball], player_contour, (2160, 3840, 3)
        )

        self.assertIs(chosen, ball)

    def test_rescued_upper_ball_stays_locked_during_descent(self):
        analyzer = self.make_analyzer()
        analyzer.frame_count = 2840
        analyzer.ball_center = (1641, 154)
        analyzer.ball_size = 53.0
        analyzer.last_motion = {"dx": 6.0, "dy": 16.0, "distance": 17.1}
        analyzer._night_upper_airborne_frame = 2823
        stale_contour = object()
        ball_contour = object()
        stale = {
            "pos": (1637, 137), "area": 5.5, "distance": 17.5,
            "motion_mean": 34.9, "motion_max": 204.0,
            "source": "primary", "contour": stale_contour,
        }
        ball = {
            "pos": (1646, 172), "area": 82.0, "distance": 18.7,
            "motion_mean": 67.8, "motion_max": 226.0,
            "source": "primary", "contour": ball_contour,
        }

        chosen = analyzer._prefer_night_upper_airborne_continuation_candidate(
            [stale, ball], stale_contour, (2160, 3840, 3)
        )

        self.assertIs(chosen, ball)

    def test_far_baseline_moving_ball_beats_nearby_static_hsv_fallback(self):
        analyzer = self.make_analyzer()
        current_metrics = {
            "motion_mean": 21.0,
            "motion_max": 131.0,
        }
        override_metrics = {
            "motion_mean": 0.1,
            "motion_max": 1.0,
        }

        locked = analyzer._night_far_baseline_dynamic_current_lock(
            "alt2",
            (2212, 245),
            24.5,
            {"pos": (2188, 267), "area": 13.5},
            current_metrics,
            override_metrics,
            (2160, 3840, 3),
        )

        self.assertTrue(locked)

    def test_night_prep_opens_before_mid_depth_player_contact(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.ball_center = (1509, 1050)
        analyzer.ball_size = 292.5
        analyzer.last_motion = {"dx": -54, "dy": 108, "distance": 120.7}
        analyzer.stuck_frame_count = 0

        self.assertTrue(
            analyzer._near_side_large_racket_hit_prep_active((2160, 3840, 3))
        )

    def test_visible_large_ball_beats_small_player_fragment_at_contact(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2384
        analyzer.ball_center = (1379, 1081)
        analyzer.ball_size = 111.5
        analyzer._near_side_large_hit_ref_frame = 2381
        analyzer._near_side_large_hit_ref_size = 292.5

        player_fragment = {
            "pos": (1354, 1149),
            "area": 34.0,
            "motion_mean": 73.7,
            "motion_max": 219.0,
            "source": "primary",
            "contour": np.array([[[0, 0]], [[4, 0]], [[4, 8]], [[0, 8]]], dtype=np.int32),
        }
        visible_ball = {
            "pos": (1344, 1051),
            "area": 295.0,
            "motion_mean": 43.8,
            "motion_max": 106.0,
            "source": "primary",
            "contour": np.array([[[0, 0]], [[16, 0]], [[16, 18]], [[0, 18]]], dtype=np.int32),
        }

        chosen = analyzer._prefer_night_near_player_contact_ball_candidate(
            [player_fragment, visible_ball], player_fragment["contour"], (2160, 3840, 3)
        )

        self.assertIs(chosen, visible_ball)

    def test_night_racket_turn_accepts_large_ball_after_flat_contact_frames(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2387
        analyzer.ball_center = (1248, 1063)
        analyzer.ball_size = 300.0
        analyzer.last_motion = {"dx": -48, "dy": 8, "distance": 48.7}
        analyzer._near_side_large_hit_ref_frame = 2381
        analyzer._near_side_large_hit_ref_size = 292.5

        accepted = analyzer._near_camera_large_racket_turn_candidate_ok(
            (1281, 955),
            365.0,
            (2160, 3840, 3),
            motion_mean=44.9,
            motion_max=100.0,
        )

        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
