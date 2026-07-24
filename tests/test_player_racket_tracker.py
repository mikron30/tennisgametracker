import tempfile
import unittest
import os
import math

import cv2
import numpy as np

from player_racket_tracker import PlayerRacketTracker, PlayerTrack
from interactive_ball_analyzer import InteractiveBallAnalyzer


class PlayerRacketTrackerTests(unittest.TestCase):
    def test_accessory_penalty_does_not_reject_strong_round_ball(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.tracks["far"] = PlayerTrack(
            "far",
            bbox=(180, 80, 120, 300),
            center=(240, 230),
            confidence=0.9,
            visible=True,
            head=(240, 120),
            shoes=(240, 368),
        )
        penalty, reason = tracker.candidate_penalty(
            (240, 120), 18.0, 42.0, 160.0,
            contour=cv2.ellipse2Poly((240, 120), (5, 5), 0, 0, 360, 10),
            predicted_distance=5.0,
        )
        self.assertEqual(penalty, 0.0)
        self.assertIsNone(reason)

    def test_small_head_fragment_is_penalized(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.tracks["far"] = PlayerTrack(
            "far", bbox=(180, 80, 120, 300), center=(240, 230), confidence=0.9, visible=True
        )
        penalty, reason = tracker.candidate_penalty(
            (240, 120), 3.0, 1.0, 8.0, predicted_distance=60.0
        )
        self.assertGreaterEqual(penalty, 600.0)
        self.assertEqual(reason, "player_head_hat")

    def test_head_guard_does_not_penalize_ball_below_head(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.tracks["far"] = PlayerTrack(
            "far", bbox=(1840, 87, 244, 492), center=(1962, 333),
            confidence=0.7, visible=True, head=(1962, 156),
        )
        head_penalty, head_reason = tracker.candidate_penalty(
            (2018, 153), 14.5, 70.9, 232.0, head_only=True,
        )
        ball_penalty, ball_reason = tracker.candidate_penalty(
            (1942, 195), 12.5, 51.9, 166.0, head_only=True,
        )
        self.assertGreaterEqual(head_penalty, 1500.0)
        self.assertEqual(head_reason, "player_head_hat")
        self.assertEqual(ball_penalty, 0.0)
        self.assertIsNone(ball_reason)

    def test_point_zone_identifies_body_and_racket_regions(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.tracks["near"] = PlayerTrack(
            "near", bbox=(1500, 1000, 600, 600), center=(1800, 1300),
            confidence=0.9, visible=True, head=(1800, 1090),
            shoes=(1800, 1570),
            racket={"line": [1720, 1100, 1745, 1345]},
        )
        self.assertEqual(tracker.point_zone((1605, 1155)), "player_body")
        self.assertEqual(tracker.point_zone((1735, 1210)), "racket_fragment")

    def test_body_penalty_is_active_when_not_strong_ball(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.tracks["near"] = PlayerTrack(
            "near", bbox=(1500, 1000, 600, 600), center=(1800, 1300),
            confidence=0.9, visible=True, head=(1800, 1090),
        )
        penalty, reason = tracker.candidate_penalty(
            (1605, 1155), 30.0, 5.0, 10.0, predicted_distance=180.0,
        )
        self.assertGreater(penalty, 0.0)
        self.assertEqual(reason, "player_body")

    def test_player_reacq_guard_rejects_distant_static_blob(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 2353
        analyzer.ball_center = (2666, 574)
        analyzer._player_reacq_protect_until_frame = 2363
        static_blob = {
            "pos": (2631, 740), "area": 78.0,
            "motion_mean": 0.0, "motion_max": 0.0,
        }
        self.assertTrue(analyzer._player_reacq_static_candidate(
            static_blob, predicted_point=(2747, 667)
        ))

    def test_player_reacq_guard_allows_moving_candidate(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 2353
        analyzer.ball_center = (2666, 574)
        analyzer._player_reacq_protect_until_frame = 2363
        moving_ball = {
            "pos": (2747, 667), "area": 33.0,
            "motion_mean": 22.0, "motion_max": 110.0,
        }
        self.assertFalse(analyzer._player_reacq_static_candidate(
            moving_ball, predicted_point=(2747, 667)
        ))

    def test_stuck_reacq_rejects_distant_static_player_body(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer._is_night_session_config = lambda: True
        analyzer._player_point_zone = lambda position: "player_body"
        self.assertTrue(analyzer._night_stuck_player_artifact_candidate(
            (1829, 1555), motion_mean=0.0, motion_max=0.0, distance=505.9
        ))
        self.assertFalse(analyzer._night_stuck_player_artifact_candidate(
            (1599, 1651), motion_mean=18.0, motion_max=90.0, distance=249.2
        ))

    def test_upper_fence_end_ignores_held_player_fragment(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 140
        analyzer.ball_center = (1942, 195)
        analyzer.ball_size = 12.5
        analyzer.last_motion = {"distance": 2.0, "dy": -2.0}
        analyzer._upper_slow_arc_until_frame = 220
        analyzer._upper_fence_fall_frames = 0
        analyzer._player_reacq_protect_until_frame = 220
        analyzer.player_tracker = PlayerRacketTracker(frame_shape=(1080, 3840), learning_enabled=False)
        analyzer.player_tracker.tracks["far"] = PlayerTrack(
            "far", bbox=(1840, 87, 244, 492), center=(1962, 333),
            confidence=0.7, visible=True, head=(1962, 156),
        )
        frame = np.zeros((1080, 3840, 3), dtype=np.uint8)
        self.assertFalse(analyzer._upper_fence_fall_end_candidate((1942, 195), frame))

    def test_terminal_endpoint_backtracks_player_overlap(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 20
        analyzer.motion_history = [
            {'frame': 18, 'pos': (80, 80), 'distance': 20.0},
            {'frame': 19, 'pos': (130, 130), 'distance': 1.0},
        ]
        analyzer.player_tracker = PlayerRacketTracker(
            frame_shape=(1080, 3840), learning_enabled=False
        )
        analyzer.player_tracker.tracks['near'] = PlayerTrack(
            'near', bbox=(90, 90, 80, 120), center=(130, 150),
            confidence=0.9, visible=True, head=(130, 102),
        )
        corrected = analyzer._terminal_player_overlap_position(
            'Ball stopped on player side', (130, 130), frame=20
        )
        self.assertEqual(corrected, (80, 80))

    def test_terminal_endpoint_accepts_image_frame_argument(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 20
        analyzer.motion_history = [
            {'frame': 18, 'pos': (80, 80), 'distance': 20.0},
            {'frame': 19, 'pos': (130, 130), 'distance': 1.0},
        ]
        analyzer.player_tracker = PlayerRacketTracker(
            frame_shape=(1080, 3840), learning_enabled=False
        )
        analyzer.player_tracker.tracks['near'] = PlayerTrack(
            'near', bbox=(90, 90, 80, 120), center=(130, 150),
            confidence=0.9, visible=True, head=(130, 102),
        )
        image_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        corrected = analyzer._terminal_player_overlap_position(
            'Ball stopped on player side', (130, 130), frame=image_frame
        )
        self.assertEqual(corrected, (80, 80))

    def test_terminal_endpoint_prefers_moving_ball_inside_player_box(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 20
        analyzer.motion_history = []
        analyzer.player_tracker = PlayerRacketTracker(
            frame_shape=(300, 400), learning_enabled=False
        )
        analyzer.player_tracker.tracks['near'] = PlayerTrack(
            'near', bbox=(90, 40, 130, 210), center=(155, 145),
            confidence=0.9, visible=True, head=(155, 0), shoes=(155, 245),
        )
        previous = np.zeros((300, 400, 3), dtype=np.uint8)
        current = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(previous, (118, 175), 10, (0, 220, 220), -1)
        cv2.circle(current, (130, 185), 10, (0, 220, 220), -1)
        analyzer._terminal_previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_frame = current
        corrected = analyzer._terminal_player_overlap_position(
            'Ball bounced before crossing net on hitter side', (155, 75), frame=current
        )
        # Frame differencing returns the centroid of the two-frame motion
        # streak (between the old and new centers), not necessarily the latest
        # circle center.
        self.assertLess(abs(corrected[0] - 124), 8)
        self.assertLess(abs(corrected[1] - 180), 8)

    def test_terminal_endpoint_repairs_empty_court_marker(self):
        """A stale marker away from the player is replaced by moving ball HSV."""
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4660
        analyzer.motion_history = []
        analyzer.player_tracker = None
        previous = np.zeros((300, 400, 3), dtype=np.uint8)
        current = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(previous, (70, 210), 10, (0, 220, 220), -1)
        cv2.circle(current, (95, 225), 10, (0, 220, 220), -1)
        analyzer._terminal_previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_frame = current
        corrected = analyzer._terminal_player_overlap_position(
            'Ball stopped on player side', (250, 40), frame=current
        )
        self.assertLess(abs(corrected[0] - 93), 8)
        self.assertLess(abs(corrected[1] - 223), 8)

    def test_terminal_motion_verifier_resumes_ambiguous_timeout(self):
        """A real moving ball cancels an inferred timeout before it is scored."""
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 4660
        analyzer.ball_center = (250, 40)
        analyzer.ball_size = 30.0
        analyzer.last_motion = None
        analyzer.last_nonzero_motion = None
        analyzer.last_delta = None
        analyzer.ball_velocity_history = []
        analyzer.motion_history = []
        analyzer.last_seen_frame = 4650
        analyzer.stuck_frame_count = 15
        analyzer.edge_wait = True
        analyzer._player_reacq_protect_until_frame = 4670
        analyzer._player_reacq_motion_failed_until_frame = 4670
        analyzer._point_history_current = {"tracking_trace": []}
        analyzer.player_tracker = None
        previous = np.zeros((300, 400, 3), dtype=np.uint8)
        current = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(previous, (70, 210), 10, (0, 220, 220), -1)
        cv2.circle(current, (95, 225), 10, (0, 220, 220), -1)
        analyzer._terminal_previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_frame = current

        self.assertTrue(analyzer._resume_from_terminal_motion_candidate(
            "Ball stopped on player side", (250, 40), frame=current
        ))
        self.assertLess(abs(analyzer.ball_center[0] - 93), 8)
        self.assertLess(abs(analyzer.ball_center[1] - 223), 8)
        self.assertEqual(analyzer.stuck_frame_count, 0)
        self.assertEqual(
            analyzer._point_history_current["tracking_trace"][-1]["source"],
            "terminal_motion_recovery",
        )

    def test_terminal_motion_verifier_does_not_cancel_verified_net(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 20
        analyzer.ball_center = (250, 40)
        analyzer.ball_size = 30.0
        analyzer.player_tracker = None
        self.assertFalse(analyzer._resume_from_terminal_motion_candidate(
            "Ball hit the net", (250, 40)
        ))

    def test_terminal_endpoint_repairs_stopped_edge_ball(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 5157
        analyzer.motion_history = []
        analyzer.player_tracker = None
        previous = np.zeros((300, 400, 3), dtype=np.uint8)
        current = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(previous, (90, 220), 9, (0, 220, 220), -1)
        cv2.circle(current, (90, 220), 9, (0, 220, 220), -1)
        analyzer._terminal_previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_frame = current
        corrected = analyzer._terminal_player_overlap_position(
            'Ball bounced out of court', (250, 40), frame=current
        )
        self.assertLess(abs(corrected[0] - 90), 5)
        self.assertLess(abs(corrected[1] - 220), 5)

    def test_terminal_endpoint_rejects_learned_static_false_point(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.player_tracker = None
        analyzer._persistent_false_points = [{
            "pos": [300, 25], "radius": 25,
            "filter_keys": ["regular_court"], "reason": "fixed fixture",
        }]
        analyzer._frame0_hotspot_combined_mask = None
        previous = np.zeros((300, 400, 3), dtype=np.uint8)
        current = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(previous, (300, 25), 8, (0, 220, 220), -1)
        cv2.circle(current, (300, 25), 8, (0, 220, 220), -1)
        candidate = analyzer._terminal_moving_ball_candidate(
            (250, 240), current,
            cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
            allow_static_anywhere=True, allow_small_static=True,
        )
        self.assertIsNone(candidate)

    def test_terminal_endpoint_repairs_stopped_ball_outside_court(self):
        """A stopped ball near the camera-side boundary beats racket motion."""
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 27911
        analyzer.motion_history = []
        analyzer.player_tracker = None
        previous = np.zeros((300, 1200, 3), dtype=np.uint8)
        current = np.zeros((300, 1200, 3), dtype=np.uint8)
        # Stationary ball near the right edge of the camera image.
        cv2.circle(previous, (1080, 230), 9, (0, 220, 220), -1)
        cv2.circle(current, (1080, 230), 9, (0, 220, 220), -1)
        # A larger moving racket-like blob elsewhere should not win.
        cv2.rectangle(previous, (760, 70), (800, 130), (0, 220, 220), -1)
        cv2.rectangle(current, (790, 75), (835, 135), (0, 220, 220), -1)
        analyzer._terminal_previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_frame = current
        corrected = analyzer._terminal_player_overlap_position(
            'Ball bounced out of court (right sideline)', (700, 180), frame=current
        )
        self.assertLess(abs(corrected[0] - 1080), 8)
        self.assertLess(abs(corrected[1] - 230), 8)

    def test_terminal_endpoint_repairs_small_stopped_ball_near_net(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 27149
        analyzer.motion_history = []
        analyzer.player_tracker = None
        previous = np.zeros((300, 1200, 3), dtype=np.uint8)
        current = np.zeros((300, 1200, 3), dtype=np.uint8)
        cv2.circle(previous, (620, 120), 4, (0, 220, 220), -1)
        cv2.circle(current, (620, 120), 4, (0, 220, 220), -1)
        cv2.rectangle(previous, (760, 70), (800, 130), (0, 220, 220), -1)
        cv2.rectangle(current, (790, 75), (835, 135), (0, 220, 220), -1)
        analyzer._terminal_previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        analyzer._terminal_current_frame = current
        corrected = analyzer._terminal_player_overlap_position(
            'Ball stopped on player side', (700, 180), frame=current
        )
        self.assertLess(abs(corrected[0] - 620), 8)
        self.assertLess(abs(corrected[1] - 120), 8)

    def test_serve_context_chooses_player_below_ball(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.tracks["far"] = PlayerTrack(
            "far", bbox=(1900, 180, 180, 260), center=(1990, 310),
            confidence=0.9, visible=True,
        )
        tracker.tracks["near"] = PlayerTrack(
            "near", bbox=(1840, 800, 360, 700), center=(2020, 1150),
            confidence=0.9, visible=True,
        )
        context = tracker.serve_context((2000, 500), player_name="P1")
        self.assertEqual(context["player"], "P1")
        self.assertEqual(context["player_side"], "near")
        self.assertEqual(context["player_position"], [2020, 1150])

    def test_serve_context_prefers_raw_detection_below_ball(self):
        tracker = PlayerRacketTracker(
            frame_shape=(2160, 3840, 3), net_y=492,
            learning_enabled=False, court_x_range=(760, 2900),
        )
        tracker.last_detections = [{
            "bbox": (2081, 1033, 464, 929),
            "score": 0.227,
            "source": "hog",
        }]
        context = tracker.serve_context((2350, 1023), player_name="P1")
        self.assertEqual(context["server_source"], "serve_detection_below_ball")
        self.assertEqual(context["player_side"], "near")
        self.assertEqual(context["player_position"], [2313, 1498])

    def test_stroke_learning_is_recorded_in_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/profile.json"
            tracker = PlayerRacketTracker(profile_path=path, learning_enabled=True)
            tracker.tracks["near"] = PlayerTrack(
                "near",
                bbox=(100, 100, 100, 300),
                center=(150, 250),
                confidence=1.0,
                visible=True,
                racket={"line": [190, 200, 240, 200], "center": [215, 200], "side": "right", "confidence": 0.8},
                racket_side="right",
            )
            event = tracker.record_shot(12, (220, 200), player_name="P1", label="racket contact")
            self.assertEqual(event["stroke"], "forehand")
            self.assertEqual(tracker.profile["shots"]["P1"]["forehand"], 1)
            tracker.save_profile()
            self.assertTrue(os.path.exists(path))

    def test_update_handles_empty_motion_and_keeps_api_stable(self):
        tracker = PlayerRacketTracker(detection_interval=2, learning_enabled=False)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        snapshot = tracker.update(frame, 0, net_y=120)
        self.assertIn("near", snapshot)
        self.assertIn("far", snapshot)

    def test_player_assignment_rejects_single_frame_teleport(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.set_geometry((720, 1280, 3), net_y=360)
        tracker.tracks["near"] = PlayerTrack(
            "near", bbox=(100, 400, 120, 240), center=(160, 520),
            confidence=0.9, last_frame=10, visible=True,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracker._assign(
            [{"bbox": (1000, 400, 120, 240), "score": 8.0, "source": "hog"}],
            frame,
            15,
        )
        self.assertEqual(tracker.tracks["near"].center, (160, 520))
        self.assertFalse(tracker.tracks["near"].visible)

    def test_4k_reacquires_full_near_player_after_stale_track(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.set_geometry((2160, 3840, 3), net_y=1080)
        tracker.tracks["near"] = PlayerTrack(
            "near", bbox=(2820, 1200, 240, 480), center=(2940, 1440),
            confidence=0.45, last_frame=165, visible=True,
        )
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        tracker._assign(
            [{"bbox": (677, 1410, 244, 489), "score": 0.03, "source": "hog"}],
            frame,
            170,
        )
        self.assertEqual(tracker.tracks["near"].center, (799, 1654))
        self.assertTrue(tracker.tracks["near"].visible)

    def test_player_candidates_stay_inside_main_court_sidelines(self):
        tracker = PlayerRacketTracker(learning_enabled=False)
        tracker.set_geometry((720, 1280, 3), net_y=360)
        tracker.set_court_region({
            "left": {"a": 0.0, "b": 100.0},
            "right": {"a": 0.0, "b": 900.0},
            "y_min": 20.0,
            "y_max": 700.0,
        })
        self.assertTrue(tracker._center_is_in_court((500, 400)))
        self.assertFalse(tracker._center_is_in_court((1100, 400)))

    def test_point_history_schema_contains_player_context_fields(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        headers = analyzer._point_history_headers()
        for field in (
            "serve_player_side", "serve_player_position", "serve_ball_position", "serve_player_source", "serve_racket_side",
            "receive_player_side", "receive_player_position", "receive_area", "receive_player_source",
            "net_player", "shot_strokes", "tracking_losses", "player_tracking_summary",
        ):
            self.assertIn(field, headers)

    def test_ball_loss_event_is_kept_in_current_point_context(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 140
        analyzer.ball_center = (1942, 195)
        analyzer.last_motion = {"dx": 0.0, "dy": -2.0, "distance": 2.0}
        analyzer.stuck_frame_count = 5
        analyzer.last_seen_frame = 139
        analyzer._point_history_current = {}
        analyzer._record_ball_loss_event(
            "player-occlusion artifact", recovery="motion reacquisition pending"
        )
        self.assertEqual(len(analyzer._point_history_current["tracking_losses"]), 1)
        event = analyzer._point_history_current["tracking_losses"][0]
        self.assertEqual(event["frame"], 140)
        self.assertEqual(event["reason"], "player-occlusion artifact")

    def test_night_continuity_recovery_survives_first_return_after_serve_window(self):
        """The post-hit f120 flight must not be cut off at video-start + 96."""
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 120
        analyzer.start_frame = 0
        analyzer.point_start_frame_internal = 22
        analyzer._last_racket_contact_frame = 103
        analyzer.ball_center = (1791, 426)
        analyzer.ball_size = 91.5
        analyzer.last_motion = {"distance": 35.0}
        analyzer.hsv_regular = {
            "lower": np.array([20, 50, 130], dtype=np.uint8),
            "upper": np.array([85, 255, 255], dtype=np.uint8),
        }
        analyzer.h10_hsv_lower = None
        analyzer.h10_hsv_upper = None
        analyzer.s30_hsv_lower = None
        analyzer.s30_hsv_upper = None
        analyzer._is_night_session_config = lambda: True
        analyzer._player_point_zone = lambda pos: None
        analyzer._candidate_motion_metrics = lambda frame_gray, x, y: {
            "mean": 25.0, "max": 120.0,
        }
        hsv = np.zeros((2160, 3840, 3), dtype=np.uint8)
        cv2.circle(hsv, (1886, 318), 6, (30, 255, 255), -1)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        candidate = analyzer._find_night_startup_regular_candidate(
            frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        self.assertIsNotNone(candidate)
        self.assertLess(math.hypot(
            candidate["pos"][0] - 1886, candidate["pos"][1] - 318
        ), 4.0)

    def test_stuck_timeout_history_frame_is_backdated_to_start_of_stuck_run(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.frame_count = 14088
        analyzer.stuck_frame_count = 15
        self.assertEqual(analyzer._stuck_timeout_end_frame(13931), 14074)
        analyzer.stuck_frame_count = 0
        self.assertEqual(analyzer._stuck_timeout_end_frame(13931), 14088)

    def test_serve_context_finds_receiver_on_opposite_side(self):
        tracker = PlayerRacketTracker(
            frame_shape=(2160, 3840, 3), net_y=492,
            learning_enabled=False, court_x_range=(760, 2900),
        )
        tracker.last_detections = [
            {"bbox": (2081, 1033, 464, 929), "score": 0.227, "source": "hog"},
            {"bbox": (1194, 0, 321, 635), "score": -0.035, "source": "hog"},
        ]
        context = tracker.serve_context((2350, 1023), player_name="P1")
        self.assertEqual(context["player_side"], "near")
        self.assertEqual(context["receiver_side"], "far")
        self.assertEqual(context["receiver_position"], [1354, 318])
        self.assertEqual(context["receiver_area"], "left_far")


if __name__ == "__main__":
    unittest.main()
