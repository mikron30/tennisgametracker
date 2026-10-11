from interactive_ball_analyzer import InteractiveBallAnalyzer
import numpy as np


def _analyzer(motion_mean=77.7, motion_max=149.0):
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 2883
    analyzer.ball_size = 297.0
    analyzer._last_motion_reacq_frame = 2883
    analyzer._last_motion_reacq_pos = (2160, 349)
    analyzer._last_tracked_candidate_motion_mean = motion_mean
    analyzer._last_tracked_candidate_motion_max = motion_max
    analyzer._contact_local_ai_state = {"miss_count": 8}
    analyzer._contact_local_ai_cooldown_until_frame = -1
    analyzer._contact_normal_fallback_path = [(1, 2)]
    analyzer._player_point_zone = lambda point: None
    analyzer._player_index_on_court_end = lambda court_end: 0
    analyzer._last_racket_contact_player = 1
    analyzer._last_racket_contact_frame = 2850
    analyzer.recorded_contacts = []
    analyzer._record_racket_contact = lambda point, label=None, player_idx_override=None: (
        analyzer.recorded_contacts.append((point, label, player_idx_override))
    )
    return analyzer


def test_strong_visible_ball_recovery_releases_stale_contact_owner():
    analyzer = _analyzer()

    released = analyzer._strong_visible_recovery_releases_contact_ai(
        (2057, 1510),
        (2160, 349),
        {"ball_size": 1053.0, "stuck_frame_count": 3},
    )

    assert released is True
    assert analyzer._contact_local_ai_state is None
    assert analyzer._contact_local_ai_cooldown_until_frame == 2886
    assert analyzer._contact_normal_fallback_path == []
    assert analyzer.recorded_contacts == [
        (
            (2057, 1510),
            "motion-confirmed lower contact launch",
            0,
        )
    ]


def test_static_far_court_candidate_does_not_release_contact_owner():
    analyzer = _analyzer(motion_mean=1.5, motion_max=8.0)

    released = analyzer._strong_visible_recovery_releases_contact_ai(
        (2057, 1510),
        (2160, 349),
        {"ball_size": 1053.0, "stuck_frame_count": 3},
    )

    assert released is False
    assert analyzer._contact_local_ai_state == {"miss_count": 8}


def test_three_moving_ai_candidates_outgrow_stale_contact_anchor():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer._force_local_ai_continuity_guard = True
    analyzer._force_local_ai_min_score = 0.985
    analyzer._contact_local_ai_state = {"miss_count": 3}

    points = ((1547, 247), (1563, 266), (1579, 287))
    promoted = None
    for frame_number, point in zip((4947, 4948, 4949), points):
        analyzer.frame_count = frame_number
        promoted = analyzer._promote_contact_visible_rejected_path([{
            "candidate": {"area": 68.0, "mode": "regular_court"},
            "point": point,
            "score": 0.993,
            "motion": {"mean": 22.0, "max": 110.0},
            "rejection": "unsupported-jump",
        }])

    assert promoted is not None
    assert promoted["point"] == (1579, 287)
    assert promoted["motion_anchor"] == (1563, 266)
    assert [entry["point"] for entry in promoted["visible_path"]] == list(points)


def test_static_or_incoherent_rejected_candidates_are_not_promoted():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer._force_local_ai_continuity_guard = True
    analyzer._force_local_ai_min_score = 0.985
    analyzer._contact_local_ai_state = {"miss_count": 3}

    for frame_number, point, motion in (
        (10, (100, 100), {"mean": 0.5, "max": 5.0}),
        (11, (130, 130), {"mean": 20.0, "max": 100.0}),
        (12, (80, 170), {"mean": 20.0, "max": 100.0}),
        (13, (160, 110), {"mean": 20.0, "max": 100.0}),
    ):
        analyzer.frame_count = frame_number
        promoted = analyzer._promote_contact_visible_rejected_path([{
            "candidate": {"area": 40.0},
            "point": point,
            "score": 0.999,
            "motion": motion,
            "rejection": "unsupported-jump",
        }])
        assert promoted is None


def test_recovered_upward_path_ends_after_bounded_top_absence():
    analyzer = _analyzer()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    analyzer._point_outside_top_singles_baseline = lambda point, image: (
        point[1] < 300,
        300.0,
    )
    analyzer._contact_visible_recovery_exit = {
        "release_frame": 2883,
        "origin": (2160, 349),
        "last_pos": (2160, 349),
        "last_move_frame": 2883,
        "coherent_up_steps": 0,
        "outside_far_baseline": False,
    }
    for frame_number, position in (
        (2884, (2169, 255)),
        (2885, (2176, 177)),
        (2886, (2183, 114)),
        (2887, (2188, 63)),
    ):
        analyzer.frame_count = frame_number
        assert analyzer._contact_visible_recovery_far_exit_end_candidate(
            position, frame
        ) == (False, None)

    analyzer.frame_count = 2900
    assert analyzer._contact_visible_recovery_far_exit_end_candidate(
        (2188, 63), frame
    ) == (False, None)
    analyzer.frame_count = 2901
    assert analyzer._contact_visible_recovery_far_exit_end_candidate(
        (2188, 63), frame
    ) == (True, "Ball bounce outside singles court (far baseline)")
    assert analyzer._contact_visible_recovery_confirmed_end_frame == 2901
    assert analyzer._contact_visible_recovery_confirmed_end_position == (2188, 63)


def test_recovered_upward_path_cancels_when_ball_reenters():
    analyzer = _analyzer()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    analyzer._point_outside_top_singles_baseline = lambda point, image: (
        point[1] < 300,
        300.0,
    )
    analyzer._contact_visible_recovery_exit = {
        "release_frame": 2883,
        "origin": (2160, 349),
        "last_pos": (2188, 63),
        "last_move_frame": 2887,
        "coherent_up_steps": 4,
        "outside_far_baseline": True,
    }
    analyzer.frame_count = 2894

    assert analyzer._contact_visible_recovery_far_exit_end_candidate(
        (2192, 110), frame
    ) == (False, None)
    assert analyzer._contact_visible_recovery_exit is None


def test_contact_ai_immediate_downward_reversal_retries_normal_tracker():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 4158
    analyzer.local_ai_recovery = object()
    analyzer._contact_local_ai_state = {
        "start_frame": 4156,
        "deadline": 4178,
        "history": [
            {"frame": 4155, "pos": (1680, 830)},
            {"frame": 4156, "pos": (1647, 770)},
            {"frame": 4157, "pos": (1653, 748)},
        ],
        "outside_count": 0,
        "miss_count": 0,
        "reason": "contact-turn:174deg/pred=156px",
    }
    analyzer.ball_center = (1653, 748)
    analyzer.ball_size = 20.0
    analyzer.ball_hsv = np.array([30, 120, 220], dtype=np.uint8)
    analyzer.last_motion = {"dx": 6, "dy": -22, "distance": 22.8}
    analyzer.prev_motion = {"dx": -33, "dy": -60, "distance": 68.5}
    analyzer.ball_velocity_history = [68.5, 22.8]
    analyzer.stuck_frame_count = 0
    analyzer._force_local_ai_history = []
    analyzer._player_point_zone = lambda point: None
    analyzer._point_in_player_contact_corridor = lambda point: True

    def select_bad_reversal(frame, previous_position):
        analyzer.ball_center = (1674, 789)
        analyzer.last_motion = {"dx": 21, "dy": 41, "distance": 46.1}
        analyzer._force_local_ai_history.append(
            {"frame": 4158, "pos": (1674, 789)}
        )
        return (1674, 789)

    analyzer._force_local_ai_frame = select_bad_reversal
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    assert analyzer._run_contact_local_ai_frame(
        frame, (1653, 748)
    ) is None
    assert analyzer.ball_center == (1653, 748)
    assert analyzer._contact_local_ai_state is None
    assert analyzer._contact_local_ai_cooldown_until_frame == 4160


def test_contact_ai_records_general_near_player_launch():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 300
    analyzer._awaiting_serve_bounce = False
    analyzer.net_area_y_max = 510
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_player = 1
    analyzer.ground_bounce_count = 1
    analyzer.last_ground_bounce_frame = 250
    analyzer._player_point_zone = lambda point: (
        'player_body' if point == (1000, 800) else None
    )
    analyzer._player_index_on_court_end = lambda side: 0 if side == 'near' else 1
    recorded = []
    analyzer._record_racket_contact = lambda point, label=None, player_idx_override=None: (
        recorded.append((point, label, player_idx_override))
    )

    analyzer._record_contact_ai_launch(
        (1000, 800), (1025, 725),
        {'last_motion': {'dx': 4, 'dy': 55, 'distance': 55.2}},
        np.zeros((1080, 1920, 3), dtype=np.uint8),
    )

    assert recorded == [((1000, 800), 'AI-confirmed racket contact', 0)]
    assert analyzer.ground_bounce_count == 0
    assert analyzer.last_ground_bounce_frame == -1000000


def test_post_contact_handoff_rejects_large_jump_back_to_player():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 4164
    analyzer._contact_local_ai_handoff_frame = 4162
    analyzer._player_point_zone = lambda point: "player_head_hat"

    assert analyzer._post_contact_handoff_player_jump_invalid(
        (1780, 272),
        (1766, 650),
        {"last_motion": {"dx": 23, "dy": -68, "distance": 71.8}},
    ) is True

    analyzer._player_point_zone = lambda point: None
    assert analyzer._post_contact_handoff_player_jump_invalid(
        (1780, 272),
        (1766, 650),
        {"last_motion": {"dx": 23, "dy": -68, "distance": 71.8}},
    ) is True

    analyzer.frame_count = 4166
    assert analyzer._post_contact_handoff_player_jump_invalid(
        (1780, 272),
        (1766, 650),
        {"last_motion": {"dx": 23, "dy": -68, "distance": 71.8}},
    ) is False


def test_contact_ai_confirms_visible_right_sideline_rebound():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 4256
    analyzer._point_outside_singles_court = lambda point, frame: (
        point == (2754, 684), 'right sideline', 0.0, 0.0, 0.0, 0.0
    )
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    candidate = analyzer._contact_ai_out_bounce_candidate({
        'history': [
            {'frame': 4254, 'pos': (2750, 670)},
            {'frame': 4255, 'pos': (2754, 684)},
            {'frame': 4256, 'pos': (2758, 667)},
        ],
    }, frame)

    assert candidate == {
        'position': (2754, 684),
        'reason': 'Ball bounced out of court (right sideline)',
        'angle': candidate['angle'],
    }
    assert candidate['angle'] >= 120.0


def test_current_out_bounce_is_not_replaced_by_contact_recovery():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 260
    analyzer._pending_rally_end_frame = 260
    analyzer.last_ground_bounce_frame = 260
    analyzer._pending_rally_end_reason = (
        'Ball bounce outside singles court (far baseline)'
    )

    assert analyzer._current_terminal_ground_bounce_pending() is True

    analyzer._pending_rally_end_frame = 259
    assert analyzer._current_terminal_ground_bounce_pending() is False
