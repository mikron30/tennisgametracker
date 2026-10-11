from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from interactive_ball_analyzer import InteractiveBallAnalyzer


def tracker():
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj._top_return_wait_active=lambda: True
    obj.ball_size=5
    obj._top_return_origin_frame=100
    obj.net_area_y_max=585
    obj.local_ai_recovery=SimpleNamespace(scoring_ready=Mock(return_value=True),
        _score=Mock(return_value=[{'ai_score':.999}]))
    obj._commit_night_visible_ball_recovery=Mock(side_effect=lambda c,f:c['pos'])
    return obj


def observe(obj,number,pos,**changes):
    obj.frame_count=number
    obj._find_night_visible_ball_candidate=lambda f,**kwargs: dict(
        pos=pos,area=80,hsv=np.array([32,150,240]),motion_mean=30,motion_max=150,**changes)
    return obj._recover_live_ball_beyond_weak_top_anchor(np.zeros((2160,3840,3),np.uint8))


def test_two_coherent_observations_and_ai_release_weak_anchor():
    obj=tracker()
    assert observe(obj,120,(1800,1000)) is None
    obj.local_ai_recovery._score.assert_not_called()
    assert observe(obj,124,(1840,1080))==(1840,1080)
    obj.local_ai_recovery._score.assert_called_once()
    candidate=obj._commit_night_visible_ball_recovery.call_args.args[0]
    assert candidate['local_ai_motion_override']['dx']==10
    assert candidate['local_ai_motion_override']['dy']==20
    assert obj._top_return_wait_frames==0


@pytest.mark.parametrize('case',['static','jump','budget','low_ai','upper'])
def test_unproven_ball_cannot_release_exit_wait(case):
    obj=tracker()
    observe(obj,120,(1800,1000))
    if case=='budget': obj.local_ai_recovery.scoring_ready.return_value=False
    if case=='low_ai': obj.local_ai_recovery._score.return_value=[{'ai_score':.5}]
    pos={'static':(1800,1000),'jump':(3000,1800),'upper':(1840,200)}.get(case,(1840,1080))
    assert observe(obj,124,pos) is None
    obj._commit_night_visible_ball_recovery.assert_not_called()


def test_clear_existing_anchor_keeps_normal_top_wait():
    obj=tracker()
    obj.ball_size=100
    obj.frame_count=124
    obj._find_night_visible_ball_candidate=Mock()
    assert obj._recover_live_ball_beyond_weak_top_anchor(np.zeros((20,20,3),np.uint8)) is None
    obj._find_night_visible_ball_candidate.assert_not_called()


def test_static_reacquisition_after_expiry_can_recover_ball_in_upper_court():
    obj=tracker()
    obj._top_return_wait_active=lambda:False
    obj._top_return_expired_frame=118
    obj.ball_size=80
    obj._last_tracked_candidate_motion_mean=0
    obj._last_tracked_candidate_motion_max=1
    assert observe(obj,120,(2200,200)) is None
    assert observe(obj,124,(2180,300))==(2180,300)


def test_motion_proven_anchor_after_expiry_does_not_start_global_ai_search():
    obj=tracker()
    obj._top_return_wait_active=lambda:False
    obj._top_return_expired_frame=118
    obj.ball_size=80
    obj._last_tracked_candidate_motion_mean=40
    obj._last_tracked_candidate_motion_max=100
    obj.frame_count=124
    obj._find_night_visible_ball_candidate=Mock()
    assert obj._recover_live_ball_beyond_weak_top_anchor(np.zeros((20,20,3),np.uint8)) is None
    obj._find_night_visible_ball_candidate.assert_not_called()
