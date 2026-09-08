"""Regression evidence from 2000.txt; source video is not needed for scoring.

Motion/line measurements are replayed into the real boundary classifier and
scorer. This does not substitute for rerunning pixel detection on the video.
"""
import csv
import math
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


@pytest.mark.parametrize('case', ['no_crossing', 'stale_crossing', 'rally_contact',
                                  'shot_event', 'serve_in', 'expired', 'large_blob',
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
def test_same_side_replay_requires_in_serve_evidence(analyzer, known_in):
    analyzer.score_points = [0, 1]
    start(analyzer, 699)
    analyzer.frame_count = 795
    if known_in:
        analyzer._record_serve_in()
    analyzer._record_point_result('Ball bounced out of court (right sideline)', (3563, 505))
    assert (getattr(analyzer, '_retro_let_pending', None) is not None) == known_in
    start(analyzer, 1245)
    analyzer.frame_count = 1283
    # A legitimate known-IN short serve still supports the existing let rule.
    assert retro._evaluate_pending(analyzer, final=True) == ('confirmed' if known_in else 'none')
    assert analyzer.score_points == ([0, 1] if known_in else [1, 1])


def test_offline_audit_does_not_treat_unknown_serve_as_in():
    row = {'point_awarded': 'yes', 'rally_shots': '0', 'category': 'out_error'}
    assert _serve_in_evidence(row)[0] is False
    assert _serve_in_evidence(dict(row, serve_in='yes'))[0] is True
    assert _serve_in_evidence(dict(row, category='serve_let'))[0] is True
