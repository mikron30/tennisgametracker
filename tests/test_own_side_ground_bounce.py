from unittest.mock import Mock
import numpy as np
import pytest
from interactive_ball_analyzer import InteractiveBallAnalyzer


def tracker(owner=1):
    a=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    a.frame_count=100
    a.point_start_frame_internal=20
    a.net_area_y_min,a.net_area_y_max=400,585
    a._serve_phase_active=a._awaiting_serve_bounce=False
    a._player_point_zone=lambda point:None
    a._point_contact_info=lambda *args,**kw:(True,owner,40,(1700,830))
    a._recent_contact_info=lambda **kw:(True,owner,40,(1700,830))
    a._player_index_on_court_end=lambda side:1 if side=='near' else 0
    a._player_index_at_point=lambda *args:1
    a._current_server_index=lambda:1
    a.ball_center=(2461,581)
    a.ball_size=52
    a.ground_bounce_count=1
    a.motion_history=[{'frame':99,'pos':(2446,622)}]
    a._classify_ground_bounce=lambda *args,**kw:(True,'Bounce in singles court',(255,0,0))
    a._add_impact_marker=Mock()
    a._append_direction_change_review_event=Mock()
    return a


def test_near_hitter_loses_on_first_own_side_landing():
    a=tracker()
    a._handle_ground_bounce_event((2446,622),np.zeros((2160,3840,3),np.uint8))
    assert a._pending_rally_end_reason=="Ball bounced on hitter's own side"
    assert a._own_side_bounce_terminal['frame']==99
    assert a._own_side_bounce_terminal['position']==(2446,622)
    result=a._infer_point_outcome(a._pending_rally_end_reason,(2446,622))
    assert result['winner_idx']==0
    assert result['category']=='own_side_bounce'


def test_far_hitter_also_loses_on_own_side_landing():
    a=tracker(owner=0)
    assert a._confirmed_own_side_ground_bounce((1800,320),np.zeros((2160,3840,3),np.uint8))
    result=a._infer_point_outcome("Ball bounced on hitter's own side",(1800,320))
    assert result['winner_idx']==1


@pytest.mark.parametrize('case',['opponent_return','player_fragment','net_band','no_contact','fresh_contact'])
def test_ambiguous_or_opponent_side_landing_does_not_end_rally(case):
    a=tracker(owner=0 if case=='opponent_return' else 1)
    point=(2446,622)
    if case=='player_fragment': a._player_point_zone=lambda point:'racket_fragment'
    if case=='net_band': point=(2446,580)
    if case=='no_contact': a._point_contact_info=lambda *args:(False,None,-1,None)
    if case=='fresh_contact': a._point_contact_info=lambda *args:(True,1,100,(1700,830))
    assert not a._confirmed_own_side_ground_bounce(point,np.zeros((2160,3840,3),np.uint8))


def test_first_serve_landing_keeps_service_fault_and_let_rules():
    a=tracker()
    a._serve_phase_active=a._awaiting_serve_bounce=True
    a._serve_net_touch_active=lambda **kw:False
    a._serve_landed_in_current_attempt=False
    a._record_serve_in=Mock()
    a._handle_ground_bounce_event((2446,622),np.zeros((2160,3840,3),np.uint8))
    assert not hasattr(a,'_pending_rally_end_reason')
    a._record_serve_in.assert_called_once()
