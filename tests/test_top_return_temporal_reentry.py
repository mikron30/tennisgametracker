from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer
from causal_ball_recovery import find_local_ball


def tracker():
    obj = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj._top_return_wait_active = lambda: True
    obj._is_night_session_config = lambda: True
    obj._top_return_mode = 'edge_clip'
    obj._top_return_anchor = (900, 2)
    obj._top_return_origin_frame = 100
    obj.primary_hsv_lower = np.array([20, 50, 130], np.uint8)
    obj.primary_hsv_upper = np.array([85, 255, 255], np.uint8)
    obj._player_point_zone = lambda pos: None
    obj._commit_night_visible_ball_recovery = Mock(side_effect=lambda c, f: c['pos'])
    return obj


def observe(obj, number, pos, before=None):
    image = np.zeros((1000, 1200, 3), np.uint8)
    cv2.ellipse(image, pos, (6, 12), 0, 0, 360, (20, 230, 230), -1)
    obj.frame_count = number
    obj._terminal_previous_gray = np.zeros(image.shape[:2], np.uint8) if before is None else before
    return obj._recover_wide_top_return_flight(image), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def test_far_lane_return_requires_three_visible_samples_and_measured_velocity():
    obj = tracker()
    result, gray = observe(obj, 130, (250, 30))
    assert result is None
    result, gray = observe(obj, 131, (210, 100), gray)
    assert result is None
    result, gray = observe(obj, 132, (170, 180), gray)
    assert np.linalg.norm(np.array(result)-[170, 180]) < 2
    candidate = obj._commit_night_visible_ball_recovery.call_args.args[0]
    assert candidate['source'] == 'temporal_top_return'
    assert candidate['local_ai_motion_override']['dx'] == pytest.approx(-40, abs=1)
    assert candidate['local_ai_motion_override']['dy'] == pytest.approx(80, abs=1)
    assert obj._top_return_wait_frames == 0
    assert obj._temporal_top_reentry_pos == result


@pytest.mark.parametrize('case', ['static', 'upward', 'jump', 'gap', 'player', 'wrong_mode', 'early', 'below_top'])
def test_unproven_return_does_not_release_anchor(case):
    obj = tracker()
    if case == 'player': obj._player_point_zone = lambda pos: 'player_body'
    if case == 'wrong_mode': obj._top_return_mode = 'upper_side'
    first = (250, 250) if case == 'below_top' else (250, 30)
    number = 110 if case == 'early' else 130
    _, gray = observe(obj, number, first)
    second = {'static': first, 'upward': (210, 10), 'jump': (900, 100)}.get(case, (210, 100))
    _, gray = observe(obj, number + (2 if case == 'gap' else 1), second, gray)
    result, _ = observe(obj, number+3, (170, 180), gray)
    assert result is None
    obj._commit_night_visible_ball_recovery.assert_not_called()


@pytest.mark.parametrize('elongated', [False, True])
def test_fast_flight_requires_explicit_temporal_context(elongated):
    before = np.zeros((400, 500, 3), np.uint8)
    after = before.copy()
    cv2.circle(before, (250, 50), 9, (20, 230, 230), -1)
    cv2.ellipse(after, (200,210), (4,14) if elongated else (9,9),
                0,0,360,(20,230,230),-1)
    filters = [(np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8))]
    args = (after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (250,50),
            {'dx':-50,'dy':160}, 220, filters)
    assert find_local_ball(*args) is None
    found = find_local_ball(*args, maximum_speed=400)
    assert found is not None
    assert np.linalg.norm(np.array(found['pos'])-[200,210]) < 2


def test_generic_visible_recovery_cannot_bypass_top_exit_wait():
    obj = tracker()
    obj.frame_count = 130
    obj.ball_center = (900,2)
    result = InteractiveBallAnalyzer._commit_night_visible_ball_recovery(
        obj, {'source':'regular','pos':(300,700)}, None)
    assert result == (900,2)
    assert obj.ball_center == (900,2)


def test_temporal_confirmation_does_not_request_patch_ai_again():
    obj = tracker()
    obj.frame_count = obj._temporal_top_reentry_frame = 132
    obj._temporal_top_reentry_pos = (170,180)
    obj._automatic_local_ai_scoring_ready = Mock()
    assert obj._try_local_ai_recovery((900,2),(170,180),0) == (170,180)
    obj._automatic_local_ai_scoring_ready.assert_not_called()


def test_confirmed_return_releases_stale_contact_ai_without_restoring_old_anchor():
    obj = tracker()
    obj.frame_count = obj._temporal_top_reentry_frame = 132
    obj._temporal_top_reentry_pos = (170,180)
    obj._contact_local_ai_state = {'anchor':(900,2), 'miss_count':0}
    assert obj._strong_visible_recovery_releases_contact_ai((900,2),(170,180),
        {'ball_size':1, 'last_motion':{'distance':1900}})
    assert obj._contact_local_ai_state is None
    assert obj._contact_normal_fallback_path == []


@pytest.mark.parametrize('mode,anchor,age', [
    ('upper_side', (900,2), 30),
    ('edge_clip', (900,35), 30),
    ('edge_clip', (900,2), 10),
])
def test_wide_return_guards_do_not_change_other_exit_contexts(mode, anchor, age):
    obj = tracker()
    obj._top_return_mode = mode
    obj._top_return_anchor = anchor
    obj.frame_count = obj._top_return_origin_frame + age
    assert not obj._delayed_literal_top_return_context()
