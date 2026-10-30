import unittest
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from interactive_ball_analyzer import InteractiveBallAnalyzer


class FarServeOcclusionTests(unittest.TestCase):
    def test_near_toss_roi_uses_only_fresh_baseline_player(self):
        for age,feet,side,camera,expected_width in (
                (2,900,'near',True,320),(13,900,'near',True,1000),
                (2,600,'near',True,1000),(2,900,'far',True,1000),
                (2,900,'near',False,1000)):
            with self.subTest(age=age,feet=feet,side=side,camera=camera):
                owner=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
                owner.frame_count=100
                owner.serve_area_x_min,owner.serve_area_x_max=0,1000
                owner.serve_area_y_min,owner.serve_area_y_max=0,500
                owner._active_serve_area_end=side
                owner.far_serve_image_baseline=camera
                owner.tracking_hsv_ranges=[dict(h_min=20,h_max=40,s_min=0,s_max=255,v_min=0,v_max=255)]
                owner.player_tracker=SimpleNamespace(tracks={'near':SimpleNamespace(
                    bbox=(400,feet-384,200,400),confidence=.8,last_frame=100-age)})
                owner.hsv_lower=np.array([20,0,0],np.uint8)
                owner.hsv_upper=np.array([40,255,255],np.uint8)
                owner._prune_ignored_serve_positions=Mock()
                widths=[]
                def capture(mask,*args):
                    widths.append(mask.shape[1])
                    return [],None
                with patch('interactive_ball_analyzer.cv2.findContours',side_effect=capture):
                    owner.detect_serve_position(np.zeros((1000,1000,3),np.uint8))
                self.assertEqual(widths,[expected_width])

    def test_configured_near_serve_uses_ball_colours_without_changing_legacy_or_far_mask(self):
        for side,camera,far_ranges,invalid,expected in (
                ('near',True,False,False,255),('near',False,False,False,0),
                ('far',True,True,False,255),('far',False,True,False,255),
                ('near',True,False,True,0),('far',True,False,False,0)):
            with self.subTest(side=side,camera=camera,far_ranges=far_ranges,invalid=invalid):
                owner=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
                owner.frame_count=100
                owner.serve_area_x_min,owner.serve_area_x_max=0,40
                owner.serve_area_y_min,owner.serve_area_y_max=0,40
                owner._active_serve_area_end=side
                owner.far_serve_image_baseline=camera
                owner.hsv_lower=np.array([90,0,0],np.uint8)
                owner.hsv_upper=np.array([110,255,255],np.uint8)
                yellow=dict(h_min=20,h_max=39,s_min=5,s_max=100,v_min=150,v_max=255)
                owner.far_serve_hsv_ranges=[yellow] if far_ranges else []
                owner.tracking_hsv_ranges=[dict(yellow,h_min=101,h_max=4)] if invalid else [yellow]
                owner._prune_ignored_serve_positions=Mock()
                frame=cv2.cvtColor(np.full((40,40,3),(35,78,255),np.uint8),cv2.COLOR_HSV2BGR)
                masks=[]
                def capture(mask,*args):
                    masks.append(mask.copy())
                    return [],None
                with patch('interactive_ball_analyzer.cv2.findContours',side_effect=capture):
                    self.assertIsNone(owner.detect_serve_position(frame))
                self.assertEqual(int(masks[0][5,5]),expected)

    def owner(self):
        owner=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        owner.frame_count=13
        owner.point_start_frame_internal=10
        owner.tracking_hsv_ranges=[dict(h_min=20,h_max=85,s_min=20,s_max=255,v_min=130,v_max=255)]
        owner.far_serve_image_baseline=True
        owner._active_serve_area_end='far'
        owner._awaiting_serve_bounce=True
        owner.ball_center=(200,100)
        owner.ball_size=35.5
        owner.last_motion=dict(dx=-10,dy=20,distance=math.hypot(10,20),direction_deg=116.6)
        owner.motion_history=[dict(frame=10,pos=(220,60)),dict(frame=11,pos=(210,80)),dict(frame=12,pos=(200,100))]
        owner.player_tracker=SimpleNamespace(tracks={'far':SimpleNamespace(
            bbox=(170,40,80,120),last_frame=13,confidence=.7)})
        owner.net_area_y_min=150
        owner.net_area_y_max=200
        owner.stuck_frame_count=0
        owner._terminal_previous_gray=np.zeros((300,400),np.uint8)
        owner._get_ground_bounce_context=lambda *a,**kw:None
        owner._serve_bounce_context_live=lambda:True
        owner._visible_far_return_active=lambda:False
        owner._record_racket_contact=Mock()
        def commit(candidate,frame):
            dx,dy=candidate['pos'][0]-owner.ball_center[0],candidate['pos'][1]-owner.ball_center[1]
            owner.last_motion=dict(dx=dx,dy=dy,distance=math.hypot(dx,dy),direction_deg=math.degrees(math.atan2(dy,dx)))
            owner.ball_velocity_history=[owner.last_motion['distance']]
            owner.ball_center=candidate['pos']
            owner.ball_size=candidate['area']
            return owner.ball_center
        owner._commit_night_visible_ball_recovery=commit
        return owner

    def test_occluded_serve_holds_then_uses_real_two_frame_elapsed_time(self):
        owner=self.owner()
        image=np.zeros((300,400,3),np.uint8)
        wrong=dict(pos=(220,120),area=12.5,motion_mean=57,motion_max=232)
        actual=dict(pos=(180,145),area=44.5,motion_mean=20,motion_max=133)
        with patch('causal_ball_recovery.find_local_ball',side_effect=[wrong,actual]):
            self.assertEqual(owner._recover_far_serve_occlusion(image),(200,100))
            self.assertEqual(len(owner.motion_history),3)
            self.assertIsNone(getattr(owner,'_configured_causal_evidence',None))
            owner.motion_history.append(dict(frame=13,pos=(200,100),distance=0))
            owner.frame_count=14
            self.assertEqual(owner._recover_far_serve_occlusion(image),(180,145))
        self.assertEqual(owner.last_motion['dx'],-10)
        self.assertEqual(owner.last_motion['dy'],22.5)
        self.assertEqual([e['frame'] for e in owner.motion_history],[10,11,12,14])
        self.assertEqual(owner._configured_causal_evidence,dict(frame=14,pos=(180,145),previous=(200,100)))
        self.assertIsNone(owner._contact_local_ai_trigger((200,100),(180,145),{},frame=image))
        owner._record_racket_contact.assert_not_called()

    def test_missing_ball_hold_expires_after_two_images(self):
        owner=self.owner()
        image=np.zeros((300,400,3),np.uint8)
        with patch('causal_ball_recovery.find_local_ball',return_value=None):
            self.assertEqual(owner._recover_far_serve_occlusion(image),(200,100))
            owner.frame_count=14
            self.assertEqual(owner._recover_far_serve_occlusion(image),(200,100))
            owner.frame_count=15
            self.assertIsNone(owner._recover_far_serve_occlusion(image))
        self.assertIsNone(owner._far_serve_occlusion_state)

    def test_legacy_profile_never_enters_or_mutates_configured_camera_recovery(self):
        owner=self.owner()
        owner.tracking_hsv_ranges=[]
        before=dict(owner.__dict__)
        with patch('causal_ball_recovery.find_local_ball') as find:
            self.assertIsNone(owner._recover_far_serve_occlusion(np.zeros((300,400,3),np.uint8)))
        self.assertEqual(owner.__dict__,before)
        find.assert_not_called()

    def test_initial_launch_anchor_is_a_real_observation_in_first_measured_step(self):
        owner=self.owner()
        owner.motion_history=[dict(frame=11,pos=(210,80),prev_pos=(220,60)),
                              dict(frame=12,pos=(200,100),prev_pos=(210,80))]
        with patch('causal_ball_recovery.find_local_ball',return_value=None):
            self.assertEqual(owner._recover_far_serve_occlusion(np.zeros((300,400,3),np.uint8)),(200,100))
        self.assertEqual(owner._far_serve_occlusion_state['anchor_frame'],12)
        self.assertEqual(len(owner.motion_history),2)

    def test_bounded_serve_occlusion_is_not_reinterpreted_as_racket_stall(self):
        owner=self.owner()
        image=np.zeros((300,400,3),np.uint8)
        with patch('causal_ball_recovery.find_local_ball',return_value=None):
            owner._recover_far_serve_occlusion(image)
        self.assertTrue(owner._far_serve_occlusion_hold_active())
        self.assertIsNone(owner._contact_local_ai_trigger((200,100),(200,100),{},frame=image))
        self.assertIsNone(owner._run_contact_local_ai_frame(image,(200,100),arm_reason='contact-stall'))
        owner._short_rewind_controller=Mock(return_value=Mock())
        self.assertIsNone(owner._try_short_ball_rewind(image,(200,100),(200,100),1))
        owner._short_rewind_controller.return_value.problem.assert_not_called()
        self.assertEqual(owner._try_local_ai_recovery((200,100),(200,100),1,{},image),(200,100))
        owner.frame_count=16
        self.assertFalse(owner._far_serve_occlusion_hold_active())


class ServeLaunchHandoffTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        self.analyzer.frame_count = 100
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

    def test_reviewed_frame_numbers_do_not_override_live_tracking_or_scoring(self):
        # Reviewed video labels are offline regression guards. Reintroducing
        # frame-specific answers would conceal tracking failures in replays.
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        for start, end in ((22,322),(696,749),(2223,2459),(2780,2901),
                           (4074,4256),(4870,5123),(5540,5578),(6258,6562),(7715,7730)):
            for current in (end-1,end,end+1):
                with self.subTest(start=start,current=current):
                    analyzer.frame_count=current
                    analyzer.start_frame=start
                    analyzer.point_start_frame_internal=start
                    analyzer._point_history_current={"serve_start_frame":start}
                    self.assertEqual(analyzer._reference_point_end_targets(),())
                    self.assertIsNone(analyzer._reference_point_end_target())
                    self.assertIsNone(analyzer._reviewed_point_continuation_frame())
                    self.assertIsNone(analyzer._reference_point_end_override())
                    self.assertIsNone(analyzer._point_timeout_result_override("Ball lost"))

    def test_serve_starts_are_not_rejected_by_reviewed_frame_number(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.point_start_frame_internal = None
        for start in (1727,2223,3900,4370,6813):
            with self.subTest(start=start):
                analyzer._point_history_current={"serve_start_frame":start}
                self.assertFalse(analyzer._is_reviewed_false_serve_start())


class NightShallowServeBounceTests(unittest.TestCase):
    def test_speed_minimum_inside_service_box_clears_serve_bounce_state(self):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.config_file = "hsv_config_04_left_night.json"
        analyzer.frame_count = 2251
        analyzer.point_start_frame_internal = 2226
        analyzer._serve_phase_active = True
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

class PostServeTrajectoryCorridorTests(unittest.TestCase):
    class FakeRecovery:
        def __init__(self, *, corridor_candidates=None, roi_candidates=None):
            self.corridor_candidates = list(corridor_candidates or [])
            self.roi_candidates = list(roi_candidates or [])
            self.work_dir = None
            self.events = []
            self.corridor_calls = []
            self.roi_calls = []

        def rank_local_corridor_candidates(self, frame_index, image, *, anchors, radius, maximum_candidates):
            self.corridor_calls.append((list(anchors), float(radius), int(maximum_candidates)))
            return list(self.corridor_candidates)

        def rank_local_roi_candidates(self, frame_index, image, *, anchor, radius, maximum_candidates):
            self.roi_calls.append((tuple(anchor), float(radius), int(maximum_candidates)))
            return list(self.roi_candidates)

        def _write_event(self, payload):
            self.events.append(payload)

    @staticmethod
    def _candidate(x, y, *, score=0.55, area=90.0):
        return {
            "x": int(x),
            "y": int(y),
            "ai_score": float(score),
            "area": float(area),
            "mode": "regular_court",
            "roi_rank": 1,
        }

    def _analyzer(self, recovery, *, corridor_active=True):
        analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
        analyzer.local_ai_recovery = recovery
        analyzer.frame_count = 20
        analyzer._post_serve_recovery_hold_until_frame = 24 if corridor_active else -1000000
        analyzer._post_serve_launch_lock_until_frame = 22 if corridor_active else -1
        analyzer._serve_launch_direction_x = -1
        analyzer._player_point_zone = lambda point: None
        analyzer._candidate_motion_metrics = lambda *args, **kwargs: {"mean": 14.0, "max": 80.0}
        return analyzer

    @staticmethod
    def _snapshot(frame_shape):
        return {
            "last_motion": {"dx": -80.0, "dy": -160.0, "distance": 178.9},
            "last_nonzero_motion": {"dx": -80.0, "dy": -160.0, "distance": 178.9},
            "ball_size": 100.0,
            "_prev_frame_gray": np.zeros(frame_shape[:2], dtype=np.uint8),
        }

    def test_post_serve_recovery_accepts_on_direction_candidate_before_full_velocity_prediction(self):
        candidate = self._candidate(755, 610)
        recovery = self.FakeRecovery(corridor_candidates=[candidate])
        analyzer = self._analyzer(recovery, corridor_active=True)
        frame = np.zeros((1000, 1200, 3), dtype=np.uint8)

        selected = analyzer._try_local_ai_trajectory_rescue(
            frame,
            previous_position=(800, 700),
            tracked_position=(800, 700),
            previous_stuck=1,
            pre_track_snapshot=self._snapshot(frame.shape),
            reason="player-region:racket_fragment",
        )

        self.assertIsNotNone(selected)
        self.assertEqual((selected["x"], selected["y"]), (755, 610))
        self.assertTrue(selected.get("trajectory_post_serve_corridor"))
        self.assertEqual(len(recovery.corridor_calls), 1)
        self.assertEqual(len(recovery.roi_calls), 0)

    def test_post_serve_corridor_rejects_candidate_against_outgoing_direction(self):
        candidate = self._candidate(855, 790, score=0.90)
        recovery = self.FakeRecovery(corridor_candidates=[candidate])
        analyzer = self._analyzer(recovery, corridor_active=True)
        frame = np.zeros((1000, 1200, 3), dtype=np.uint8)

        selected = analyzer._try_local_ai_trajectory_rescue(
            frame,
            previous_position=(800, 700),
            tracked_position=(800, 700),
            previous_stuck=1,
            pre_track_snapshot=self._snapshot(frame.shape),
            reason="player-region:racket_fragment",
        )

        self.assertIsNone(selected)
        self.assertEqual(len(recovery.corridor_calls), 1)

    def test_normal_trajectory_rescue_keeps_single_constant_velocity_roi(self):
        candidate = self._candidate(720, 540)
        recovery = self.FakeRecovery(roi_candidates=[candidate])
        analyzer = self._analyzer(recovery, corridor_active=False)
        frame = np.zeros((1000, 1200, 3), dtype=np.uint8)

        selected = analyzer._try_local_ai_trajectory_rescue(
            frame,
            previous_position=(800, 700),
            tracked_position=(800, 700),
            previous_stuck=1,
            pre_track_snapshot=self._snapshot(frame.shape),
            reason="player-region:racket_fragment",
        )

        self.assertIsNotNone(selected)
        self.assertEqual((selected["x"], selected["y"]), (720, 540))
        self.assertFalse(selected.get("trajectory_post_serve_corridor", False))
        self.assertEqual(len(recovery.roi_calls), 1)
        self.assertEqual(len(recovery.corridor_calls), 0)
