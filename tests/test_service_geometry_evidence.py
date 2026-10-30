from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from night_serve_geometry import far_service_box


@pytest.mark.parametrize('length',[110,119,150])
def test_far_service_line_accepts_supported_hough_segments(length,monkeypatch):
    frame=np.full((540,960,3),30,np.uint8)
    cv2.line(frame,(430,86),(430+length,86),(240,240,240),2)
    cv2.line(frame,(480,89),(480,110),(240,240,240),2)
    frame=cv2.resize(frame,(3840,2160))
    monkeypatch.setattr(cv2,'HoughLinesP',lambda *args,**kwargs:
        np.array([[[430,86,430+length,92]]],np.int32))
    model=far_service_box(frame,492)
    assert model is not None
    line=model['service_line']
    assert abs(line['a']*480+line['b']/4-88.5)<=3


def test_missing_calibrated_service_geometry_cannot_fall_back_to_net_tape(monkeypatch):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    owner=SimpleNamespace(_service_box_model=None,_service_box_frame_shape=None,
        net_area_y_min=100,net_area_y_max=150,serve_direction_dy=-1,
        far_serve_image_baseline=True,_is_night_session_config=lambda:False)
    monkeypatch.setattr('night_serve_geometry.far_service_box',lambda *args:None)
    monkeypatch.setattr(cv2,'HoughLinesP',lambda *args,**kwargs:pytest.fail('unsafe fallback'))
    result=InteractiveBallAnalyzer._build_service_box_model(owner,np.zeros((540,960,3),np.uint8))
    assert result is None


@pytest.mark.parametrize('racket,reverse,clear',[(True,True,True),(False,True,True),(True,False,True),(True,True,False)])
def test_measured_lateral_racket_turn_requires_the_verified_contact_context(racket,reverse,clear):
    from far_contact_recovery import changes_trajectory
    path=[dict(frame=n,pos=(100+(6 if reverse else -2)*n,100-18*n)) for n in range(4)]
    if not clear:path=path[:2]
    assert changes_trajectory(path,dict(dx=-2,dy=-14),racket_contact=racket)==(racket and reverse and clear)


@pytest.mark.parametrize('invalid',[None,'uncalibrated','stale','weak','player'])
def test_large_near_boundary_ball_needs_fresh_clear_measured_return(monkeypatch,invalid):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    owner=SimpleNamespace(frame_count=100,ball_size=347,far_serve_image_baseline=True,
        _last_direction_change_frame=100,_last_direction_change_vertical_reversal=True,
        _last_direction_change_point=(2535,1915),_last_racket_contact_frame=50,
        prev_motion=dict(dx=6,dy=155,distance=155),last_motion=dict(dx=22,dy=-27,distance=35),
        _configured_causal_evidence=dict(frame=100,pos=(2557,1888),previous=(2535,1915)),
        _last_tracked_candidate_motion_frame=100,_last_tracked_candidate_motion_mean=100,
        _last_tracked_candidate_motion_max=145,
        _visible_far_return_active=lambda:True,_serve_bounce_context_live=lambda:False,
        _player_point_zone=lambda p:'player_body' if invalid=='player' else None,
        _point_outside_singles_court=lambda *args:(True,'near baseline',None,None,200,1650),
        _point_outside_singles_sidelines=lambda *args:(False,None,None,None),
        _classify_ground_bounce=lambda *args:(False,'Ball bounce outside singles court (near baseline)',None))
    if invalid=='uncalibrated':owner.far_serve_image_baseline=False
    if invalid=='stale':owner._configured_causal_evidence['frame']=99
    if invalid=='weak':owner._last_tracked_candidate_motion_max=20
    result=InteractiveBallAnalyzer._confirmed_boundary_reversal_out_candidate(
        owner,(2557,1888),np.zeros((2160,3840,3),np.uint8))
    assert result[0]==(invalid is None)


def test_calibrated_near_baseline_uses_the_visible_curve(monkeypatch):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    monkeypatch.setattr('night_serve_geometry.near_baseline',lambda frame:dict(curve=[.00005,-.2,1800]))
    owner=SimpleNamespace(far_serve_image_baseline=True,_awaiting_serve_bounce=False,
        _point_outside_singles_sidelines=lambda *args:(False,None,500,3300))
    inside,reason,_=InteractiveBallAnalyzer._classify_ground_bounce(
        owner,(2500,1900),np.zeros((2160,3840,3),np.uint8))
    assert inside is False and 'near baseline' in reason


@pytest.mark.parametrize('x,target,expected',[(1952,'left',True),(1970,'left',False),(1925,'right',True),(1900,'right',False)])
def test_service_half_uses_only_the_calibrated_center_line_uncertainty(x,target,expected):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    model=dict(center_line=dict(a=0,b=1938),service_line=dict(a=0,b=350),
        service_margin=12,net_margin=30,net_y=492,center_margin=15.36)
    owner=SimpleNamespace(_service_box_model=None,_service_box_frame_shape=None,
        _singles_sideline_model=None,_singles_sideline_frame_shape=None,serve_direction_dy=-1,
        _point_outside_singles_sidelines=lambda *args:(False,None,1400,2400),
        _build_service_box_model=lambda *args:model)
    actual,_,_=InteractiveBallAnalyzer._evaluate_target_service_box_fresh(
        owner,(x,397),np.zeros((2160,3840,3),np.uint8),target)
    assert actual==expected



def test_late_player_box_compares_departure_to_the_precontact_leg():
    from far_contact_recovery import changes_trajectory
    outgoing=[dict(frame=n,pos=(100+5*n,100+6*n)) for n in range(4)]
    assert not changes_trajectory(outgoing,dict(dx=5,dy=6))
    assert changes_trajectory(outgoing,dict(dx=-3,dy=-4))


@pytest.mark.parametrize('paint',[True,False])
def test_shadowed_service_paint_requires_local_contrast(monkeypatch,paint):
    frame=np.full((540,960,3),30 if paint else 130,np.uint8)
    if paint:
        cv2.line(frame,(430,86),(549,92),(130,130,130),2)
        cv2.line(frame,(480,91),(480,110),(130,130,130),2)
    frame=cv2.resize(frame,(3840,2160))
    def detected(mask,*args,**kwargs):
        return np.array([[[430,86,549,92]]],np.int32) if np.count_nonzero(mask) else None
    monkeypatch.setattr(cv2,'HoughLinesP',detected)
    model=far_service_box(frame,492)
    assert (model is not None)==paint


@pytest.mark.parametrize('invalid',[None,'short','steep','no_center'])
def test_occluded_far_service_stripe_keeps_only_supported_short_geometry(monkeypatch,invalid):
    frame=np.full((540,960,3),30,np.uint8)
    cv2.line(frame,(430,86),(500,90),(240,240,240),2)
    if invalid!='no_center':cv2.line(frame,(480,91),(480,110),(240,240,240),2)
    frame=cv2.resize(frame,(3840,2160))
    def detected(mask,*args,**kwargs):
        if kwargs['minLineLength']>=100:return None
        length=40 if invalid=='short' else 70
        rise=15 if invalid=='steep' else 4
        return np.array([[[430,86,430+length,86+rise]]],np.int32)
    monkeypatch.setattr(cv2,'HoughLinesP',detected)
    model=far_service_box(frame,492)
    assert (model is not None)==(invalid is None)
