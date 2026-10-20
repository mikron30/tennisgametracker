"""Regression evidence from 2000.txt; source video is not needed for scoring.

Motion/line measurements are replayed into the real boundary classifier and
scorer. This does not substitute for rerunning pixel detection on the video.
"""
import csv
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import retro_let_scoring as retro
from interactive_ball_analyzer import InteractiveBallAnalyzer
from serve_side_consistency import _serve_in_evidence


@pytest.fixture
def analyzer(monkeypatch, tmp_path):
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture', lambda _: MagicMock())
    monkeypatch.setattr(retro, '_PATCHED', False)

    class Tracker(InteractiveBallAnalyzer):
        pass

    retro._patch_tracker_class(Tracker)
    obj = Tracker('unused', enable_player_tracking=False, enable_player_learning=False,
                  point_history_file=str(tmp_path / 'points.csv'))
    obj.net_area_y_min, obj.net_area_y_max = 400, 585
    obj.serve_area_x_min, obj.serve_area_x_max = 760, 2900
    obj.serve_direction_dy = -1
    obj._player_serve_context = lambda point: {
        'player_position': [1568, 1500], 'player_side': 'near',
        'player_source': 'server', 'serve_x': point[0], 'serve_y': point[1],
    }
    return obj


def start(obj, frame, position=(1806, 1009)):
    obj.frame_count = obj.point_start_frame_internal = frame
    obj._start_point_context(position, serve_start_frame=frame)


def recovered_reversal(obj, offset=0):
    start(obj, 699 + offset)
    obj.frame_count = 749 + offset
    obj.ball_center, obj.ball_size = (2575, 478), 16.0
    obj.prev_motion = {'dx': 13, 'dy': 24, 'distance': math.hypot(13, 24)}
    obj.last_motion = {'dx': 10, 'dy': -21, 'distance': math.hypot(10, 21)}
    obj._last_direction_change_frame = obj.frame_count
    obj._last_direction_change_vertical_reversal = True
    obj._last_direction_change_point = (2565, 499)
    obj._local_ai_follow_until_frame = 751 + offset
    obj.motion_history = [
        {'frame': 734 + offset, 'pos': (2381, 322)},
        {'frame': 747 + offset, 'pos': (2552, 475)},
        {'frame': 748 + offset, 'pos': (2565, 499)},
    ]
    # Right line derived from the log's boundary measurements at y=402/685.
    slope = (2523.5 - 2388.9) / (685 - 402)
    obj._singles_sideline_model = {
        'left': {'a': 0.0, 'b': 1269.2},
        'right': {'a': slope, 'b': 2523.5 - slope * 685}, 'margin': 20.0,
    }
    obj._singles_sideline_frame_shape = (2160, 3840)
    return np.zeros((2160, 3840, 3), dtype=np.uint8)


@pytest.mark.parametrize('offset', [0, 9000])
def test_recovered_first_bounce_is_fault_during_ai_hold(analyzer, offset):
    frame = recovered_reversal(analyzer, offset)
    ended, reason = analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame)
    assert ended
    assert reason == 'Serve bounce outside singles court (right sideline)'
    assert analyzer.frame_count <= analyzer._local_ai_follow_until_frame


@pytest.mark.parametrize('case', ['no_crossing', 'stale_crossing', 'large_blob',
                                  'no_reversal', 'line_contact'])
def test_recovery_exception_keeps_existing_safety_gates(analyzer, case):
    frame = recovered_reversal(analyzer)
    if case == 'no_crossing':
        analyzer.motion_history = []
    elif case == 'stale_crossing':
        analyzer.motion_history = [{'frame': 300, 'pos': (2381, 322)}]
    elif case == 'rally_contact':
        analyzer._last_racket_contact_frame = 730
    elif case == 'shot_event':
        analyzer._point_history_current['shot_events'] = [{'player': 'P2', 'frame': 730}]
    elif case == 'serve_in':
        analyzer._serve_landed_in_current_attempt = True
    elif case == 'expired':
        analyzer.point_start_frame_internal = 500
    elif case == 'large_blob':
        analyzer.ball_size = 500.0
    elif case == 'no_reversal':
        analyzer._last_direction_change_vertical_reversal = False
    elif case == 'line_contact':
        analyzer._sideline_line_contact_override = lambda *a, **k: True
    assert analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame) == (False, None)



@pytest.mark.parametrize('case', ['rally_contact', 'shot_event', 'serve_in', 'expired'])
def test_stale_serve_state_yields_to_regular_boundary_out(analyzer, case):
    frame = recovered_reversal(analyzer)
    if case == 'rally_contact':
        analyzer._last_racket_contact_frame = 730
    elif case == 'shot_event':
        analyzer._point_history_current['shot_events'] = [{'player': 'P2', 'frame': 730}]
    elif case == 'serve_in':
        analyzer._serve_landed_in_current_attempt = True
    elif case == 'expired':
        analyzer.point_start_frame_internal = 500
    ended, reason = analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame)
    assert ended
    assert reason == 'Ball bounce outside singles court (right sideline)'


def test_expired_serve_flag_does_not_hide_far_baseline_reversal(analyzer):
    start(analyzer, 2783, (2384, 1011))
    analyzer.frame_count = 2930
    analyzer.ball_center = (2174, 222)
    analyzer.ball_size = 1.0
    analyzer.prev_motion = {'dx': -3, 'dy': 38, 'distance': math.hypot(3, 38)}
    analyzer.last_motion = {'dx': 21, 'dy': -40, 'distance': math.hypot(21, 40)}
    analyzer._last_direction_change_frame = 2930
    analyzer._last_direction_change_vertical_reversal = True
    analyzer._last_direction_change_point = (2153, 262)
    analyzer._player_reacq_protect_until_frame = 3002
    analyzer._point_outside_singles_court = lambda point, frame: (True, 'far baseline', None, None, None, None)
    analyzer._classify_ground_bounce = lambda point, frame: (False, 'Ball bounce outside singles court (far baseline)', (0, 0, 255))
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    ended, reason = analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame)
    assert ended
    assert reason == 'Ball bounce outside singles court (far baseline)'

def test_rally_boundary_reversal_still_awards_regular_out(analyzer):
    frame = recovered_reversal(analyzer)
    analyzer._awaiting_serve_bounce = analyzer._serve_phase_active = False
    analyzer._point_hit_count = 3
    ended, reason = analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame)
    assert ended and reason == 'Ball bounce outside singles court (right sideline)'


def test_first_fault_second_fault_then_rally_reaches_zero_forty(analyzer):
    # Point 1 already awarded to P2 in the user's log.
    analyzer.score_points = [0, 1]
    frame = recovered_reversal(analyzer)
    ended, reason = analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame)
    assert ended
    analyzer._record_point_result(reason, analyzer.ball_center, frame=frame)
    assert analyzer.score_points == [0, 1]
    assert analyzer.current_serve_attempt == 2
    assert getattr(analyzer, '_retro_let_pending', None) is None

    start(analyzer, 1245, (1665, 977))
    analyzer.frame_count = 1283
    analyzer._record_point_result('Serve net then bounce outside right service box', (2048, 530))
    assert analyzer.score_points == [0, 2]
    assert analyzer.current_serve_attempt == 1
    assert analyzer._last_point_outcome_category == 'double_fault'

    start(analyzer, 1730, (2342, 1022))
    analyzer.frame_count = 1843
    analyzer._last_racket_contact_frame = 1813
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_point = (1489, 1473)
    analyzer._point_hit_count = 1
    analyzer._point_history_current['shot_events'] = [{'player': 'P1', 'frame': 1813}]
    analyzer._record_point_result('Ball bounce outside singles court (far baseline)', (1572, 231))
    assert analyzer._score_summary() == '0:0 0:40'
    assert analyzer.serve_stats[0]['first_faults'] == 1
    assert analyzer.serve_stats[0]['double_faults'] == 1
    assert getattr(analyzer, '_retro_let_pending', None) is None
    with open(analyzer.point_history_file, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [r['current_score'] for r in rows] == ['0:0 0:15', '0:0 0:30', '0:0 0:40']
    assert [r['serve_attempt'] for r in rows] == ['1st', '2nd', '1st']
    assert [r['point_awarded'] for r in rows] == ['no', 'yes', 'yes']


@pytest.mark.parametrize('known_in', [False, True])
@pytest.mark.parametrize('net_touch', [False, True])
def test_same_side_replay_requires_in_serve_and_net_touch_evidence(analyzer, known_in, net_touch):
    analyzer.score_points = [0, 1]
    start(analyzer, 699)
    analyzer.frame_count = 795
    if known_in:
        analyzer._record_serve_in()
    if net_touch:
        analyzer._serve_net_contact_frame = 719
        analyzer._serve_net_contact_point = (1900,480)
    analyzer._record_point_result('Ball bounced out of court (right sideline)', (3563, 505))
    assert (getattr(analyzer, '_retro_let_pending', None) is not None) == (known_in and net_touch)
    start(analyzer, 1245)
    analyzer.frame_count = 1283
    # A measured net touch plus legal landing still supports a real let replay.
    confirmed = known_in and net_touch
    assert retro._evaluate_pending(analyzer, final=True) == ('confirmed' if confirmed else 'none')
    assert analyzer.score_points == ([0, 1] if confirmed else [1, 1])


def test_offline_audit_does_not_treat_unknown_serve_as_in():
    row = {'point_awarded': 'yes', 'rally_shots': '0', 'category': 'out_error'}
    assert _serve_in_evidence(row)[0] is False
    assert _serve_in_evidence(dict(row, serve_in='yes'))[0] is True
    assert _serve_in_evidence(dict(row, category='serve_let'))[0] is True


def test_pre_reversal_contact_point_is_checked_for_boundary_out(analyzer):
    start(analyzer, 100, (1800, 1000))
    analyzer._serve_phase_active = False
    analyzer._awaiting_serve_bounce = False
    analyzer.frame_count = 140
    analyzer.ball_center = (220, 180)
    analyzer.ball_size = 12.0
    analyzer.prev_motion = {'dx': 2, 'dy': 34, 'distance': math.hypot(2, 34)}
    analyzer.last_motion = {'dx': 20, 'dy': -40, 'distance': math.hypot(20, 40)}
    analyzer._last_direction_change_frame = analyzer.frame_count
    analyzer._last_direction_change_vertical_reversal = True
    analyzer._last_direction_change_point = (220, 180)
    analyzer._point_outside_singles_court = lambda point, frame: (
        (True, 'far baseline', None, None, None, None)
        if tuple(point) == (200, 220)
        else (False, None, None, None, None, None)
    )
    analyzer._classify_ground_bounce = lambda point, frame: (
        (False, 'Ball bounce outside singles court (far baseline)', (0, 0, 255))
        if tuple(point) == (200, 220)
        else (True, 'Bounce in singles court', (255, 0, 0))
    )
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)
    ended, reason = analyzer._confirmed_boundary_reversal_out_candidate(analyzer.ball_center, frame)
    assert ended
    assert reason == 'Ball bounce outside singles court (far baseline)'


def test_serve_seed_recovers_missed_returns_and_deciding_game(analyzer):
    analyzer.score_games = [0, 0]
    analyzer.score_points = [1, 3]
    analyzer._current_server_index = lambda: 0
    start(analyzer, 100, (1800, 1000))
    analyzer._serve_phase_active = False
    analyzer._awaiting_serve_bounce = False
    analyzer._player_index_at_point = lambda point, frame=None: 0 if point[1] > 500 else 1
    frame = np.zeros((1000, 1600, 3), dtype=np.uint8)

    for f, point in ((110, (700, 350)), (111, (700, 390)), (112, (700, 430))):
        analyzer.frame_count = f
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point, frame)
    analyzer.frame_count = 113
    assert not analyzer._maybe_infer_return_contact_from_side_recross((700, 510), frame)
    analyzer.frame_count = 114
    assert analyzer._maybe_infer_return_contact_from_side_recross((700, 540), frame)
    assert analyzer._last_racket_contact_player == 1
    assert analyzer._point_hit_count == 1

    for f, point in ((120, (720, 650)), (121, (720, 620)), (122, (720, 590))):
        analyzer.frame_count = f
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point, frame)
    analyzer.frame_count = 123
    assert not analyzer._maybe_infer_return_contact_from_side_recross((730, 480), frame)
    analyzer.frame_count = 124
    assert analyzer._maybe_infer_return_contact_from_side_recross((740, 450), frame)
    assert analyzer._last_racket_contact_player == 0
    assert analyzer._point_hit_count == 2

    analyzer.frame_count = 125
    analyzer._record_point_result(
        'Ball bounce outside singles court (far baseline)',
        (740, 220),
        frame=frame,
    )
    assert analyzer.score_games == [0, 1]
    assert analyzer.score_points == [0, 0]
    assert analyzer._score_summary() == '0:1 0:0'


def test_discontinuous_side_reacquisition_does_not_infer_return(analyzer):
    start(analyzer, 100, (1800, 900))
    analyzer.max_ball_speed = 400
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (1800, 800)
    analyzer._point_hit_count = 0
    analyzer._point_history_current['shot_events'] = []
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    for f in (110, 111, 112):
        analyzer.frame_count = f
        assert not analyzer._maybe_infer_return_contact_from_side_recross((2156, 262), frame)

    analyzer.frame_count = 113
    assert not analyzer._maybe_infer_return_contact_from_side_recross((1756, 823), frame)
    analyzer.frame_count = 114
    assert not analyzer._maybe_infer_return_contact_from_side_recross((1754, 821), frame)
    assert analyzer._last_racket_contact_player == 0
    assert analyzer._point_hit_count == 0


def test_continuous_side_recross_still_infers_return(analyzer):
    start(analyzer, 100, (1800, 900))
    analyzer.max_ball_speed = 400
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (1800, 800)
    analyzer._point_hit_count = 0
    analyzer._point_history_current['shot_events'] = []
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    for f, point in ((110, (1800, 350)), (111, (1805, 390)), (112, (1810, 430))):
        analyzer.frame_count = f
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point, frame)
    analyzer.frame_count = 113
    assert not analyzer._maybe_infer_return_contact_from_side_recross((1815, 510), frame)
    analyzer.frame_count = 114
    assert analyzer._maybe_infer_return_contact_from_side_recross((1820, 540), frame)
    assert analyzer._last_racket_contact_player == 1
    assert analyzer._point_hit_count == 1


@pytest.mark.parametrize('contact_opportunity', [False, True])
def test_side_recross_requires_opponent_contact_opportunity(analyzer, contact_opportunity):
    start(analyzer, 100, (1800, 900))
    analyzer.max_ball_speed = 400
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (1800, 800)
    analyzer._point_hit_count = 0
    analyzer._point_history_current['shot_events'] = []
    analyzer._player_index_on_court_end = lambda side: 0 if side == 'near' else 1
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    opponent_track = SimpleNamespace(
        side='far', bbox=(1500, 100, 300, 300), center=(1650, 250),
        confidence=0.8, last_frame=114,
    )

    class PlayerTracker:
        tracks = {'far': opponent_track}
        detection_interval = 5
        last_frame = 114

        def _track_for_point(self, point):
            return opponent_track if contact_opportunity else None

    analyzer.player_tracker = PlayerTracker()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    result = False
    for number, point in ((110, (1800, 350)), (111, (1805, 390)),
                          (112, (1810, 430)), (113, (1815, 510)),
                          (114, (1820, 540))):
        analyzer.frame_count = number
        opponent_track.last_frame = number
        analyzer.player_tracker.last_frame = number
        result = analyzer._maybe_infer_return_contact_from_side_recross(point, frame)

    assert result is contact_opportunity


def test_held_exit_marker_does_not_create_opponent_observations(analyzer):
    start(analyzer, 100, (1800, 900))
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer._player_index_at_point = lambda point, frame=None: 1
    analyzer._reset_side_recross_watch(0)
    frame = np.zeros((2160, 3840, 3), np.uint8)
    for number in range(110, 150):
        analyzer.frame_count = number
        assert not analyzer._maybe_infer_return_contact_from_side_recross(
            (1800, 30), frame, previous_position=(1800, 30))
    assert analyzer._side_recross_opponent_frames == 0
    assert analyzer._side_recross_opponent_last_frame == -1


def test_measured_incoming_lane_can_prove_return_during_reentry_grace(analyzer):
    start(analyzer, 100, (1800, 900))
    analyzer.max_ball_speed = 400
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (1800, 800)
    analyzer._recent_offscreen_return_frame = 108
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    analyzer._player_point_zone = lambda point: None
    analyzer.ball_size = 36
    analyzer.ball_velocity_history = [40, 40, 40]
    analyzer._terminal_previous_gray = np.zeros((2160, 3840), np.uint8)
    previous = (1800, 310)
    answers = []
    for number, point in ((110, (1800,350)), (111, (1805,390)),
                          (112, (1810,430)), (113, (1815,510)), (114, (1820,540))):
        frame = np.zeros((2160, 3840, 3), np.uint8)
        x, y = point
        frame[y-3:y+4, x-3:x+4] = 255
        analyzer.frame_count = number
        answers.append(analyzer._maybe_infer_return_contact_from_side_recross(
            point, frame, previous_position=previous))
        analyzer._terminal_previous_gray = frame[:,:,0].copy()
        previous = point
    assert answers == [False, False, False, False, True]
    assert analyzer._last_racket_contact_player == 1
    assert analyzer._point_hit_count == 1


def test_shallow_net_recross_does_not_infer_return(analyzer):
    start(analyzer, 290, (1150, 825))
    analyzer.max_ball_speed = 400
    analyzer._serve_phase_active = False
    analyzer._awaiting_serve_bounce = False
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 290
    analyzer._last_racket_contact_point = (1150, 825)
    analyzer._point_hit_count = 6
    analyzer._point_history_current['shot_events'] = []
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    # Replay the user's f298-f312 path: it crosses the net center smoothly,
    # but never leaves the configured 400..585 net band on the far side.
    path = [
        (298, (1663, 479)), (299, (1703, 470)), (300, (1739, 460)),
        (301, (1772, 454)), (302, (1805, 450)), (303, (1835, 445)),
        (304, (1866, 446)), (305, (1894, 449)), (306, (1920, 453)),
        (307, (1946, 458)), (308, (1972, 465)), (309, (1983, 475)),
        (310, (1989, 489)), (311, (1992, 504)), (312, (1996, 520)),
    ]
    for f, point in path:
        analyzer.frame_count = f
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point, frame)

    assert analyzer._last_racket_contact_player == 0
    assert analyzer._last_racket_contact_frame == 290
    assert analyzer._point_hit_count == 6


def test_continuously_recovered_ground_rebound_is_not_a_racket_return(analyzer):
    start(analyzer, 100)
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer.last_ground_bounce_frame = 109
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    for number, point in ((110, (1800,350)), (111, (1800,390)),
                          (112, (1800,430)), (113, (1800,510)), (114, (1800,540))):
        analyzer.frame_count = analyzer._causal_flight_anchor_frame = number
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point)
    assert analyzer._last_racket_contact_player == 0
    assert analyzer._side_recross_opponent_frames == 0


def test_ground_bounce_on_last_hitter_side_blocks_later_side_recross_return(analyzer):
    start(analyzer, 100)
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer.last_ground_bounce_frame = 109
    analyzer._ground_bounce_origin = (1800, 540)
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    for number, point in ((110, (1800, 350)), (111, (1805, 390)),
                          (112, (1810, 430)), (113, (1815, 510)),
                          (114, (1820, 540))):
        analyzer.frame_count = number
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point, frame)

    assert analyzer._last_racket_contact_player == 0
    assert analyzer._point_hit_count == 0


def test_ground_bounce_on_opponent_side_still_allows_hidden_return_inference(analyzer):
    start(analyzer, 100)
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 100
    analyzer.last_ground_bounce_frame = 109
    analyzer._ground_bounce_origin = (1800, 350)
    analyzer._player_index_at_point = lambda point, frame=None: 1 if point[1] < 500 else 0
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    for number, point in ((110, (1800, 350)), (111, (1805, 390)),
                          (112, (1810, 430)), (113, (1815, 510))):
        analyzer.frame_count = number
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point, frame)
    analyzer.frame_count = 114
    assert analyzer._maybe_infer_return_contact_from_side_recross((1820, 540), frame)
    assert analyzer._last_racket_contact_player == 1


def test_second_bounce_outside_is_unreturned_not_hitter_error(analyzer):
    start(analyzer, 100)
    analyzer.frame_count = 180
    analyzer._serve_phase_active = analyzer._awaiting_serve_bounce = False
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 110
    analyzer._last_racket_contact_point = (1800, 350)
    analyzer.ground_bounce_count = 2
    analyzer._classify_ground_bounce = lambda *a, **k: (False, 'Ball bounced out of court (right sideline)', (0,0,255))
    analyzer._handle_ground_bounce_event((2800, 700), np.zeros((2160,3840,3), np.uint8))
    assert analyzer._pending_rally_end_reason == 'Ball bounced twice on court'
    result = analyzer._infer_point_outcome(analyzer._pending_rally_end_reason, (2800,700))
    assert result['winner_idx'] == 0
    assert result['category'] == 'unreturned'


def test_late_inferred_contact_keeps_measured_next_receiver_samples(analyzer):
    start(analyzer,100)
    analyzer._last_racket_contact_player=0
    analyzer._last_racket_contact_frame=100
    analyzer._player_index_at_point=lambda point,frame=None: 1 if point[1]<500 else 0
    frame=np.zeros((2160,3840,3),np.uint8)
    for number,point in ((110,(1800,350)),(111,(1800,390)),(112,(1800,430)),
                         (113,(1800,580)),(114,(1800,620))):
        analyzer.frame_count=number
        analyzer._maybe_infer_return_contact_from_side_recross(point,frame)
    assert analyzer._last_racket_contact_player==1
    # The confirmation sample and two following visible samples belong to
    # the next receiver, including during the contact cooldown.
    for number,point in ((115,(1800,660)),(116,(1800,700))):
        analyzer.frame_count=number
        assert not analyzer._maybe_infer_return_contact_from_side_recross(point,frame)
    analyzer.frame_count=120
    assert not analyzer._maybe_infer_return_contact_from_side_recross((1800,440),frame)
    analyzer.frame_count=121
    assert analyzer._maybe_infer_return_contact_from_side_recross((1800,400),frame)
    assert analyzer._last_racket_contact_player==0
    assert analyzer._point_hit_count==2


@pytest.mark.parametrize('previous,current,net_hit', [
    ((-16, -317), (22, -75), False),
    ((16, -317), (-22, -75), False),
    ((-30, -10), (15, -5), True),
    ((0, -30), (0, 15), True),
])
def test_net_impact_requires_flight_reversal_not_minor_lateral_sign_change(analyzer, previous, current, net_hit):
    start(analyzer, 10)
    analyzer.frame_count = 100
    analyzer.ball_center = (1737, 417)
    analyzer.ball_size = 140
    analyzer._serve_phase_active = analyzer._awaiting_serve_bounce = False
    analyzer._contact_visible_recovery_far_exit_end_candidate = lambda *args: (False, None)
    analyzer._confirmed_outbound_sideline_exit = lambda *args: (False, None)
    analyzer._top_far_baseline_fall_out_candidate = lambda *args: (False, None)
    analyzer._upper_fence_fall_end_candidate = lambda *args: False
    analyzer._detect_out_of_court_bounce = lambda *args: (False, None)
    analyzer._mark_serve_net_contact_candidate = lambda *args: False
    analyzer._same_side_pre_net_bounce_candidate = lambda *args: (False, None)
    analyzer._net_contact_geometry = lambda point: {
        'inside': True, 'near_polygon': True, 'near_top_tape': False,
        'near_hull': True, 'signed_dist': 0,
    }
    def motion(delta):
        dx, dy = delta
        return {'dx': dx, 'dy': dy, 'distance': math.hypot(dx, dy),
                'direction_deg': math.degrees(math.atan2(dy, dx))}
    analyzer.prev_motion = motion(previous)
    analyzer.last_motion = motion(current)
    result, reason = analyzer.detect_point_end(analyzer.ball_center, np.zeros((2160, 3840, 3), np.uint8))
    assert (result and reason == 'Ball hit the net') is net_hit


def test_verified_receiver_second_bounce_keeps_physical_endpoint(analyzer):
    start(analyzer, 10)
    analyzer.frame_count = 100
    analyzer._serve_phase_active = analyzer._awaiting_serve_bounce = False
    analyzer._receiver_bounce_terminal = {
        'confirmed_frame':100, 'frame':99, 'position':(50,70), 'point_start':10}
    analyzer._terminal_player_overlap_position = MagicMock(
        side_effect=AssertionError('Verified bounce must not search for another ball'))
    analyzer._record_point_result('Ball bounced twice on court', end_position=(48,60))
    analyzer._terminal_player_overlap_position.assert_not_called()
    rows = list(csv.DictReader(open(analyzer.point_history_file, encoding='utf-8-sig')))
    assert rows[-1]['point_end_frame'] == '99'
    assert rows[-1]['end_position'] == '(50,70)'


@pytest.mark.parametrize('provisional', [True, False])
def test_service_landing_requires_confirmed_touch_for_let(analyzer, provisional):
    start(analyzer,10)
    analyzer.frame_count = 30
    analyzer._serve_net_contact_provisional = provisional
    analyzer._serve_net_touch_active = lambda **kwargs: True
    analyzer._night_serve_net_bounce_wrong_side_reason = lambda *args: None
    analyzer._classify_ground_bounce = lambda *args, **kwargs: (True,'IN',(255,0,0))
    analyzer._record_serve_in = MagicMock()
    analyzer._handle_ground_bounce_event((1800,350),np.zeros((2160,3840,3),np.uint8))
    assert (analyzer._pending_rally_end_reason == 'Serve let') is (not provisional)
    assert analyzer._record_serve_in.call_count == int(provisional)


@pytest.mark.parametrize("server", [0, 1])
@pytest.mark.parametrize("reason", ["Ball bounced out of court (right sideline)", "Ball bounced twice on court"])
def test_unreturned_in_serve_wins_even_when_second_bounce_is_outside(analyzer, server, reason):
    start(analyzer, 100)
    analyzer.frame_count = 130
    analyzer._current_server_index = lambda: server
    analyzer._serve_landed_in_current_attempt = True
    analyzer._last_serve_bounce_was_in = True
    analyzer._last_serve_bounce_frame = 119
    analyzer._point_contact_info = lambda **k: (False, None, None, None)
    analyzer._recent_contact_info = lambda **k: (False, None, None, None)
    analyzer._player_index_at_point = lambda *args: server
    outcome = analyzer._infer_point_outcome(reason, (3000, 1200))
    assert outcome['winner_idx'] == server
    assert outcome['category'] == 'unreturned'


@pytest.mark.parametrize("missing", ["serve_in", "confirmed_bounce", "later_bounce", "untouched"])
def test_unreturned_serve_award_requires_prior_confirmed_in_and_no_return(analyzer, missing):
    start(analyzer, 100)
    analyzer.frame_count = 130
    analyzer._current_server_index = lambda: 0
    analyzer._serve_landed_in_current_attempt = missing != "serve_in"
    analyzer._last_serve_bounce_was_in = missing != "confirmed_bounce"
    analyzer._last_serve_bounce_frame = 129 if missing == "later_bounce" else 119
    analyzer._point_contact_info = lambda **k: (missing == "untouched", 0, 125, None)
    analyzer._recent_contact_info = lambda **k: (False, None, None, None)
    analyzer._player_index_at_point = lambda *args: 0
    analyzer._is_lower_right_exit_score_context = lambda *args: False
    outcome = analyzer._infer_point_outcome("Ball bounced out of court (right sideline)", (3000, 1200))
    assert outcome['winner_idx'] == 1
    assert outcome['category'] == 'out_error'


def test_measured_boundary_rebound_keeps_physical_landing_in_history(analyzer):
    start(analyzer, 10)
    analyzer.frame_count = 100
    analyzer._serve_phase_active = analyzer._awaiting_serve_bounce = False
    reason = 'Ball bounce outside singles court (left sideline)'
    analyzer._causal_boundary_terminal = dict(confirmed_frame=100, frame=99,
        position=(50,70), point_start=10, reason=reason)
    analyzer._terminal_player_overlap_position = MagicMock(
        side_effect=AssertionError('Measured bounce cannot be replaced with a player blob'))
    analyzer._record_point_result(reason, end_position=(48,60))
    analyzer._terminal_player_overlap_position.assert_not_called()
    rows = list(csv.DictReader(open(analyzer.point_history_file, encoding='utf-8-sig')))
    assert rows[-1]['point_end_frame'] == '99'
    assert rows[-1]['end_position'] == '(50,70)'


@pytest.mark.parametrize('outside', [True, False])
@pytest.mark.parametrize('offset', [0, 10000])
def test_measured_boundary_check_precedes_predicted_bounce_context(analyzer, outside, offset):
    analyzer.frame_count = 250 + offset
    analyzer.point_start_frame_internal = 100 + offset
    analyzer.ball_center = (100, 100)
    analyzer.ball_size = 24
    analyzer.last_motion = {'dx': -5, 'dy': 5, 'distance': 7.1, 'direction_deg': 135}
    analyzer._get_ground_bounce_context = lambda *a, **k: {'origin': (100,100)}
    analyzer._serve_bounce_context_live = lambda: False
    analyzer._player_point_zone = lambda p: None
    analyzer._last_racket_contact_frame = 200 + offset
    reason = 'Ball bounce outside singles court (left sideline)'
    analyzer._confirmed_boundary_reversal_out_candidate = MagicMock(return_value=(outside, reason if outside else None))
    analyzer._register_ground_bounce_from_context = MagicMock()
    candidate = dict(pos=(98,83), area=17, hsv=np.array([30,100,200],np.uint8),
        motion_mean=12, motion_max=80)
    analyzer._commit_local_ball_flight(candidate, np.zeros((240,320,3),np.uint8))
    analyzer._confirmed_boundary_reversal_out_candidate.assert_called_once()
    if outside:
        analyzer._register_ground_bounce_from_context.assert_not_called()
        assert analyzer._pending_rally_end_reason == reason
        assert analyzer._causal_boundary_terminal['position'] == (100,100)
        assert analyzer._causal_boundary_terminal['frame'] == 249 + offset
    else:
        analyzer._register_ground_bounce_from_context.assert_called_once()
        assert analyzer._pending_rally_end_reason is None


@pytest.mark.parametrize('paint',[True,False])
def test_measured_rebound_can_use_local_paint_when_cached_line_misses_curvature(analyzer,paint):
    analyzer.frame_count=150
    analyzer.point_start_frame_internal=100
    analyzer.far_serve_image_baseline=True
    analyzer._serve_phase_active=analyzer._awaiting_serve_bounce=False
    analyzer._last_direction_change_frame=150
    analyzer._last_direction_change_vertical_reversal=True
    analyzer._last_direction_change_point=(494,359)
    analyzer.ball_size=16
    analyzer.prev_motion={'dx':-5,'dy':5,'distance':7.1}
    analyzer.last_motion={'dx':-2,'dy':-17,'distance':17.1}
    analyzer._last_racket_contact_frame=125
    analyzer._point_outside_singles_court=lambda *a:(False,None,500,2500,None,None)
    analyzer._point_outside_singles_sidelines=lambda *a:(False,None,500,2500)
    frame=np.zeros((2160,3840,3),np.uint8)
    if paint:
        import cv2
        cv2.line(frame,(600,250),(390,550),(255,255,255),5)
    ended,reason=analyzer._confirmed_boundary_reversal_out_candidate((492,342),frame)
    assert ended is paint
    if paint:
        assert reason=='Ball bounce outside singles court (left sideline)'


def test_serve_history_keeps_initial_server_while_receiver_refreshes(analyzer):
    analyzer.player_tracker=object()
    initial={'player_side':'far','player_position':[1900,220], 'server_source':'verified_serve_stance',
        'serve_x':1850,'serve_y':120,'receiver_position':[2200,1400]}
    analyzer._point_history_current={'serve_context':initial.copy()}
    analyzer._player_serve_context=lambda p: {'player_side':'near','player_position':[2000,1000],
        'server_source':'serve_detection_below_ball','receiver_position':[1950,250]}
    analyzer._refresh_player_serve_context()
    result=analyzer._point_history_current['serve_context']
    assert result['player_side']=='far'
    assert result['player_position']==[1900,220]
    assert result['server_source']=='verified_serve_stance'
    assert result['receiver_position']==[1950,250]


@pytest.mark.parametrize('side,age,stuck,gap,scale', [
    ('far',3,1,2,2),('far',12,5,10,3),('near',3,1,2,1),
    ('far',20,1,2,1),('far',3,0,2,1)])
def test_missed_serve_handoff_projects_only_bounded_recent_far_flight(analyzer,monkeypatch,side,age,stuck,gap,scale):
    analyzer.frame_count=200
    analyzer.point_start_frame_internal=200-age
    analyzer._active_serve_area_end=side
    analyzer.ball_center=(100,100)
    analyzer.ball_size=24
    analyzer.last_motion={'dx':2,'dy':20,'distance':20.1}
    analyzer.stuck_frame_count=stuck
    analyzer.motion_history=[{'frame':200-gap,'pos':(100,100)}]
    analyzer._player_point_zone=lambda point: 'player_body'
    analyzer.tracking_hsv_ranges=[dict(h_min=20,h_max=110,s_min=5,s_max=100,v_min=150,v_max=255)]
    seen=[]
    def search(frame,previous,position,motion,area,filters,**kwargs):
        seen.append(motion)
    monkeypatch.setattr('causal_ball_recovery.find_local_ball',search)
    assert analyzer._recover_local_ball_flight(np.zeros((600,800,3),np.uint8)) is None
    assert seen[0]['dx']==2*scale
    assert seen[0]['dy']==20*scale


@pytest.mark.parametrize('case,expected', [('contact_step',True),('gap',False),('no_incoming',False),('player',False)])
def test_boundary_rebound_accepts_slow_contact_step_only_after_measured_arrival(analyzer,case,expected):
    analyzer.frame_count=150
    analyzer.point_start_frame_internal=100
    analyzer._serve_phase_active=analyzer._awaiting_serve_bounce=False
    analyzer._last_direction_change_frame=150
    analyzer._last_direction_change_vertical_reversal=True
    analyzer._last_direction_change_point=(600,353)
    analyzer._last_racket_contact_frame=100
    analyzer.ball_size=22
    analyzer.prev_motion={'dx':0,'dy':3,'distance':3}
    analyzer.last_motion={'dx':0,'dy':-23,'distance':23}
    analyzer.motion_history=[{'frame':147 if case=='gap' else 148,'pos':(600,350),
        'prev_pos':(600,355 if case=='no_incoming' else 325),'distance':25}]
    analyzer._player_point_zone=lambda p:'player_body' if case=='player' else None
    analyzer._point_outside_singles_court=lambda *a:(True,'right sideline',100,500,None,None)
    analyzer._classify_ground_bounce=lambda *a:(False,'Ball bounce outside singles court (right sideline)',None)
    result,_=analyzer._confirmed_boundary_reversal_out_candidate((600,330),np.zeros((1000,1200,3),np.uint8))
    assert result is expected

def test_measured_mesh_collision_assigns_net_error_to_verified_hitter():
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=150
    obj.point_start_frame_internal=100
    obj.ball_center=(230,160)
    obj._current_server_index=lambda:1
    obj._recent_contact_info=lambda **kw:(True,0,130,(100,500))
    obj._point_contact_info=lambda **kw:(True,0,130,(100,500))
    # The projected landing is closer to the far player. Contact ownership
    # must still decide who hit the net when a collision was measured.
    obj._player_index_at_point=lambda *a:1
    obj._causal_boundary_terminal=dict(point_start=100,confirmed_frame=150,reason='Ball hit the net')
    result=obj._infer_point_outcome('Ball hit the net')
    assert result['winner_idx']==1 and result['category']=='net_error'
    assert result['loser_idx']==0
