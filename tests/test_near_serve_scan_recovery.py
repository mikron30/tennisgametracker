from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from near_serve_scan_recovery import (hold_near_toss_gap, recent_near_stance,
                                     near_toss_entry_y_min, partial_near_serve_contact)
from near_serve_scan_recovery import near_serve_impact, far_side_incoming_serve
from near_serve_scan_recovery import falling_near_serve_contact
from near_serve_scan_recovery import near_partial_strike_area
from near_serve_scan_recovery import measured_near_serve_in
from near_serve_scan_recovery import verified_near_serve_net_drop
from near_serve_scan_recovery import above_local_net_top


@pytest.mark.parametrize('side,point,expected',[
    ('near',(150,135),True),('near',(150,145),False),
    ('near',(100,119),True),('near',(100,120),False),
    ('far',(150,135),False),('far',(150,99),True)])
def test_far_flight_uses_sloped_tape_only_for_calibrated_near_side(side,point,expected):
    polygon=np.array([[0,100],[100,120],[200,160],[200,240],[100,220],[0,200]],np.float32)
    obj=SimpleNamespace(far_serve_image_baseline=True,_active_serve_area_end=side,
        net_area_y_min=100,_get_net_polygon=lambda:polygon)
    assert above_local_net_top(obj,point)==expected


@pytest.mark.parametrize('invalid',[None,'toss','receiver','landed','bounce','old_net','before_hit','far','inferred'])
def test_counted_serve_stroke_can_keep_a_confirmed_net_drop_pending(invalid):
    obj=SimpleNamespace(frame_count=100,point_start_frame_internal=50,
        far_serve_image_baseline=True,_active_serve_area_end='far' if invalid=='far' else 'near',
        _serve_phase_active=True,_serve_landed_in_current_attempt=invalid=='landed',
        _point_hit_count=0 if invalid=='toss' else 1,ground_bounce_count=1 if invalid=='bounce' else 0,
        _last_racket_contact_frame=80,_serve_net_contact_frame=60 if invalid=='old_net' else 75 if invalid=='before_hit' else 90,
        _last_racket_contact_label='inferred return' if invalid=='inferred' else 'verified outgoing contact flight',
        _last_racket_contact_player=1 if invalid=='receiver' else 0,
        _current_server_index=lambda:0,_last_racket_contact_point=(200,500),net_area_y_max=400)
    assert verified_near_serve_net_drop(obj)==(invalid is None)


@pytest.mark.parametrize('invalid',[None,'out','unknown','body','far','receiver_hit',
                                  'second_bounce','already_landed','old_contact','other_player','inferred'])
def test_measured_first_in_landing_repairs_only_verified_near_serve(invalid):
    owner=SimpleNamespace(frame_count=100,point_start_frame_internal=50,
        far_serve_image_baseline=True,_active_serve_area_end='far' if invalid=='far' else 'near',
        _last_racket_contact_frame=40 if invalid=='old_contact' else 80,
        _last_racket_contact_point=(200,500),net_area_y_max=400,
        _last_racket_contact_label='inferred return' if invalid=='inferred' else 'verified outgoing contact flight',
        _last_racket_contact_player=1 if invalid=='other_player' else 0,
        _point_hit_count=2 if invalid=='receiver_hit' else 1,
        ground_bounce_count=2 if invalid=='second_bounce' else 1,
        _serve_landed_in_current_attempt=invalid=='already_landed',
        _current_server_index=lambda:0,
        _player_point_zone=lambda p:'player_body' if invalid=='body' else None,
        _evaluate_target_service_box_fresh=lambda *a:(None if invalid=='unknown' else invalid!='out','reason',{}))
    assert measured_near_serve_in(owner,(180,300),np.zeros((600,800,3),np.uint8))==(invalid is None)


def test_perspective_serve_landing_can_flatten_without_positive_image_y():
    samples=[dict(frame=i,pos=p,area=40) for i,p in enumerate(
        [(2017,391),(2007,384),(1998,379),(1990,377),(1982,377),(1976,363),(1972,329)],1)]
    landing=near_serve_impact(samples)
    assert landing['frame']==5 and landing['pos']==(1982,377)
    assert landing['perspective_flattening']
    assert near_serve_impact(samples[:-1]) is None


@pytest.mark.parametrize('kind',['constant_flight','no_rebound','side_jump','gap','size_jump'])
def test_perspective_flattening_does_not_invent_a_landing(kind):
    samples=[dict(frame=i,pos=p,area=40) for i,p in enumerate(
        [(2017,391),(2007,384),(1998,379),(1990,377),(1982,377),(1976,363),(1972,329)],1)]
    if kind=='constant_flight':
        for i,s in enumerate(samples):s['pos']=(2017-i*8,391-i*8)
    if kind=='no_rebound':samples[-1]['pos']=(1972,363)
    if kind=='side_jump':samples[-1]['pos']=(1930,329)
    if kind=='gap':samples[3]['frame']+=1
    if kind=='size_jump':samples[-1]['area']=2
    assert near_serve_impact(samples) is None


def partial_contact_owner():
    obj=owner();obj.ball_center=(2240,848);obj.ball_size=609
    obj._awaiting_serve_bounce=True;obj._point_hit_count=0;obj.net_area_y_max=585
    obj._serve_stance_last_valid=dict(frame=90,side='near',bbox=(2000,1200,400,650))
    obj.serve_area_y_min=650;obj.serve_area_y_max=1100;obj.point_start_frame_internal=1
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(96,100),
        [(2255,740),(2252,787),(2248,838),(2240,848)])]
    obj._configured_flight_area_reference=dict(point_start=1,frame=99,area=380)
    return obj


def test_partial_contact_uses_measured_ball_size_before_racket_overlap():
    obj=partial_contact_owner()
    assert near_partial_strike_area(obj,609)==380


@pytest.mark.parametrize('kind',['far','other_camera','rally','stale','other_point','ordinary_size'])
def test_partial_contact_area_cannot_change_unrelated_flights(kind):
    obj=partial_contact_owner()
    if kind=='far':obj._active_serve_area_end='far'
    if kind=='other_camera':obj.far_serve_image_baseline=False
    if kind=='rally':obj._point_hit_count=1
    if kind=='stale':obj._configured_flight_area_reference['frame']=98
    if kind=='other_point':obj._configured_flight_area_reference['point_start']=2
    if kind=='ordinary_size':obj._configured_flight_area_reference['area']=500
    assert near_partial_strike_area(obj,609)==609


def owner():
    return SimpleNamespace(frame_count=100, far_serve_image_baseline=True,
        _active_serve_area_end='near', _serve_stance_guard_decision='allow',
        _serve_stance_last_valid=dict(frame=99, side='near', confidence=.8))


def test_recent_legal_stance_retains_compact_rising_toss_across_missing_image():
    obj=owner()
    detail=dict(frame=99,motion_delta=30,aspect_ratio=1.1,circularity=.5)
    assert hold_near_toss_gap(obj, [(1600,1000),(1605,935)], [detail])
    assert hold_near_toss_gap(obj, [(1605,935)], [detail])
    assert not hold_near_toss_gap(obj, [(1600,900),(1605,935)], [detail])
    assert not hold_near_toss_gap(obj, [(1600,935),(1605,935)], [detail])
    obj.frame_count=102
    assert not hold_near_toss_gap(obj, [(1605,935)], [detail])


def test_delayed_stance_uses_server_scale_without_lowering_the_rise_requirements():
    obj=owner();obj.serve_area_y_min=650;obj.serve_area_y_max=1100
    obj._serve_stance_last_valid['bbox']=(1400,1100,300,700)
    assert near_toss_entry_y_min(obj,965) == 762.5
    obj._serve_stance_guard_decision='reject'
    assert near_toss_entry_y_min(obj,965) == 965


def test_partial_serve_contact_requires_a_continuous_toss_and_baseline_player():
    obj=owner();obj.frame_count=104;obj.ball_size=200;obj.ball_center=(1780,826)
    obj._point_hit_count=0;obj._awaiting_serve_bounce=True;obj.net_area_y_max=585
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(100,104),
        [(1747,732),(1751,780),(1755,831),(1780,826)])]
    obj._serve_stance_last_valid['bbox']=(1400,1100,300,700)
    assert partial_near_serve_contact(obj)
    obj._point_hit_count=1
    assert not partial_near_serve_contact(obj)


def test_contact_search_includes_racket_reach_outside_a_narrow_body_box():
    obj=owner();obj.frame_count=104;obj.ball_size=570;obj.ball_center=(1774,828)
    obj._point_hit_count=0;obj._awaiting_serve_bounce=True;obj.net_area_y_max=585
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(100,104),
        [(1747,732),(1751,780),(1755,831),(1774,828)])]
    obj._serve_stance_last_valid['bbox']=(1352,1100,280,650)
    assert partial_near_serve_contact(obj)
    obj.ball_center=(3000,828);obj.motion_history[-1]['pos']=obj.ball_center
    assert not partial_near_serve_contact(obj)


def test_cropped_player_box_does_not_hide_a_toss_in_the_calibrated_serve_band():
    obj=owner();obj.frame_count=104;obj.ball_size=296;obj.ball_center=(1694,880)
    obj._point_hit_count=0;obj._awaiting_serve_bounce=True;obj.net_area_y_max=585
    obj.serve_area_y_min=650;obj.serve_area_y_max=1100
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(101,104),
        [(1690,766),(1693,822),(1694,880)])]
    obj._serve_stance_last_valid['bbox']=(1580,1320,180,360)
    assert falling_near_serve_contact(obj)
    obj._point_hit_count=0;obj.motion_history[-2]['pos']=(1755,700)
    assert not partial_near_serve_contact(obj)


def test_fresh_full_ball_partial_strike_is_followed_by_measured_departure():
    from causal_ball_recovery import find_local_ball
    previous=np.full((300,300,3),40,np.uint8)
    cv2.circle(previous,(130,160),8,(0,220,220),-1)
    partial=np.full_like(previous,40)
    cv2.circle(partial,(150,159),8,(0,220,220),-1)
    filters=[(np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8))]
    kwargs=dict(minimum_speed=2,maximum_speed=400,minimum_saturation=5,
                contact_window=True,flexible_contact=True)
    args=(partial,cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),(130,160),
          dict(dx=3,dy=30),160,filters)
    assert find_local_ball(*args,**kwargs) is None
    first=find_local_ball(*args,serve_contact_search=True,**kwargs)
    assert first is not None and first['partial_serve_strike']
    assert np.linalg.norm(np.subtract(first['pos'],(150,159)))<2
    outgoing=np.full_like(previous,40)
    cv2.circle(outgoing,(195,100),8,(0,220,220),-1)
    second=find_local_ball(outgoing,cv2.cvtColor(partial,cv2.COLOR_BGR2GRAY),
        first['pos'],dict(dx=20,dy=-1),first['area'],filters,rising_contact=True,**kwargs)
    assert second is not None and second['verified_contact_departure']
    assert np.linalg.norm(np.subtract(second['pos'],(195,100)))<2


def test_departing_serve_beats_large_body_fragment_outside_a_cropped_player_box():
    from causal_ball_recovery import find_local_ball
    previous=np.full((500,500,3),40,np.uint8)
    cv2.circle(previous,(200,200),10,(0,220,220),-1)
    current=np.full_like(previous,40)
    cv2.rectangle(current,(168,235),(190,259),(170,180,180),-1)
    cv2.circle(current,(287,114),10,(0,220,220),-1)
    filters=[(np.array([20,0,130],np.uint8),np.array([85,255,255],np.uint8))]
    result=find_local_ball(current,cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),
        (200,200),dict(dx=1,dy=58),300,filters,minimum_speed=2,maximum_speed=400,
        minimum_saturation=5,contact_window=True,flexible_contact=True,serve_contact_search=True)
    assert result is not None and result['verified_contact_departure']
    assert np.linalg.norm(np.subtract(result['pos'],(287,114)))<2


def test_shallow_far_landing_needs_two_fresh_rebound_observations():
    samples=[dict(frame=n,pos=p,area=25) for n,p in enumerate(
        [(2200,312),(2204,314),(2208,317),(2211,320),(2215,294),(2219,271)],1)]
    assert near_serve_impact(samples)['frame']==4
    assert near_serve_impact(samples[:-1]) is None
    samples[-1]['pos']=(2419,271)
    assert near_serve_impact(samples) is None
    plateau=[dict(frame=n,pos=p,area=25) for n,p in enumerate(
        [(2384,333),(2390,341),(2396,350),(2401,360),(2406,359),(2408,329),(2416,315)],1)]
    assert near_serve_impact(plateau)['frame']==4


@pytest.mark.parametrize('area',[3,6,25,80])
def test_small_incoming_serve_continues_only_while_fresh_motion_is_observed(area):
    obj=owner();obj.ball_center=(200,100);obj.ball_size=area;obj.net_area_y_min=150
    obj._awaiting_serve_bounce=True
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(97,100),[(190,110),(195,104),(200,100)])]
    obj._last_tracked_candidate_motion_frame=99
    obj._last_tracked_candidate_motion_mean=10;obj._last_tracked_candidate_motion_max=80
    assert far_side_incoming_serve(obj)
    obj.motion_history[-1]['pos']=(195,104)
    assert not far_side_incoming_serve(obj)


def test_recent_strong_server_survives_a_weak_empty_court_detection(monkeypatch):
    from serve_stance_guard import evaluate_serve_stance
    monkeypatch.setattr('serve_stance_guard._load_full_frame_points',lambda _: {
        1:(1000,250),2:(2500,250),3:(500,1600),4:(3300,1600)})
    track=SimpleNamespace(bbox=(1450,1100,300,700),center=(1600,1450),
        shoes=(1600,1780),confidence=.8,visible=False,last_frame=98)
    obj=owner();obj.net_area_y_min=400;obj.net_area_y_max=585
    obj.config_file='hsv_config_court1_trial.json';obj.player_tracking_enabled=True
    obj.player_tracker=SimpleNamespace(tracks={'near':track})
    obj._serve_stance_last_valid=None
    image=np.zeros((2160,3840,3),np.uint8)
    result=evaluate_serve_stance(obj,(1650,900),image)
    assert result['decision']=='allow'
    legal=obj._serve_stance_last_valid.copy()
    obj.frame_count=101
    obj.player_tracker.tracks['near']=SimpleNamespace(bbox=(1900,1800,160,300),
        center=(1980,1950),shoes=(1980,2090),confidence=.39,visible=True,last_frame=101)
    assert evaluate_serve_stance(obj,(1650,900),image)['decision']=='allow'
    assert obj._serve_stance_last_valid==legal
    # A clear fresh player inside the baseline must still invalidate the cache.
    obj.player_tracker.tracks['near']=SimpleNamespace(bbox=(1450,1000,300,400),
        center=(1600,1200),shoes=(1600,1390),confidence=.8,visible=True,last_frame=101)
    assert evaluate_serve_stance(obj,(1650,900),image)['decision']=='reject'


def test_measured_outgoing_serve_does_not_inherit_inflated_contact_blob_size(monkeypatch):
    from unittest.mock import MagicMock
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture',lambda _:MagicMock())
    obj=InteractiveBallAnalyzer('unused',enable_player_tracking=False,
        enable_player_learning=False,point_history_file=None)
    obj.frame_count=100;obj.point_start_frame_internal=1
    obj.far_serve_image_baseline=True;obj._active_serve_area_end='near'
    obj._awaiting_serve_bounce=True;obj.net_area_y_min=400;obj.net_area_y_max=585
    obj.ball_center=(2060,461);obj.ball_size=97.5
    obj.last_motion=dict(dx=42,dy=-52,distance=66.8)
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(97,100),
        [(1963,585),(2018,513),(2060,461)])]
    obj._last_tracked_candidate_motion_frame=99
    obj._last_tracked_candidate_motion_mean=53;obj._last_tracked_candidate_motion_max=202
    obj.tracking_hsv_ranges=[dict(h_min=20,h_max=85,s_min=5,s_max=255,v_min=100,v_max=255)]
    obj._configured_flight_area_reference=dict(point_start=1,frame=99,area=570)
    obj._player_point_zone=lambda _:None
    captured=[]
    def capture(*args,**kwargs): captured.append(args[4]);return None
    monkeypatch.setattr('causal_ball_recovery.find_local_ball',capture)
    obj._recover_local_ball_flight(np.zeros((2160,3840,3),np.uint8))
    assert captured==[97.5]


@pytest.mark.parametrize('field,value',[
    ('far_serve_image_baseline',False),('_active_serve_area_end','far'),
    ('_serve_stance_guard_decision','reject'),('_serve_stance_guard_decision','hold'),
    ('_serve_stance_last_valid',dict(frame=80,side='near',confidence=.8)),
    ('_serve_stance_last_valid',dict(frame=99,side='near',confidence=.3))])
def test_recovery_does_not_override_rejected_stale_or_other_camera_stances(field,value):
    obj=owner();setattr(obj,field,value)
    assert not recent_near_stance(obj)


@pytest.mark.parametrize('invalid',[None,'no_rebound','two_fragments','gap','lateral_reversal'])
def test_perspective_serve_bounce_survives_one_partial_paint_crossing(invalid):
    points=[(136,60),(125,53),(117,51),(110,50),(100,49),(93,45),(90,12),(87,-17)]
    areas=[49,37,30.5,6,18,25.5,30.5,29]
    if invalid=='no_rebound':points[-1]=(87,11)
    if invalid=='two_fragments':areas[4]=4
    if invalid=='lateral_reversal':points[-1]=(95,-17)
    samples=[dict(frame=n+(1 if invalid=='gap' and n>=5 else 0),pos=p,area=a)
             for n,(p,a) in enumerate(zip(points,areas),1)]
    result=near_serve_impact(samples)
    assert (result is not None)==(invalid is None)
    if result:assert result['frame']==5


@pytest.mark.parametrize('area,mean,peak,expected',[(6,5,65,True),(6,4,65,False),(25,5,65,False),(6,5,45,False)])
def test_partial_far_serve_keeps_only_a_fresh_compact_fragment(area,mean,peak,expected):
    obj=owner();obj.ball_center=(200,100);obj.ball_size=area;obj.net_area_y_min=150
    obj._awaiting_serve_bounce=True
    obj.motion_history=[dict(frame=n,pos=p) for n,p in zip(range(97,100),[(190,110),(195,104),(200,100)])]
    obj._last_tracked_candidate_motion_frame=99
    obj._last_tracked_candidate_motion_mean=mean;obj._last_tracked_candidate_motion_max=peak
    assert far_side_incoming_serve(obj)==expected


def test_shallow_distant_landing_keeps_one_pixel_quantization_noise():
    points=[(283,411),(297,406),(310,404),(322,401),(331,402),(341,383),(348,354)]
    areas=[21,12,7,47,35.5,5.5,37]
    samples=[dict(frame=n,pos=p,area=a) for n,(p,a) in enumerate(zip(points,areas),1)]
    result=near_serve_impact(samples)
    assert result and result['frame']==5


@pytest.mark.parametrize('invalid',[None,'rally','receiver','landed','stale','far','inferred','two_bounces'])
def test_verified_server_stroke_keeps_first_service_landing_live(invalid):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=SimpleNamespace(frame_count=100,point_start_frame_internal=60,
        _serve_phase_active=True,_awaiting_serve_bounce=True,
        _serve_landed_in_current_attempt=False,_point_hit_count=1,
        _point_history_current=dict(shot_events=[dict(player='P1')]),
        _last_racket_contact_frame=85,_last_racket_contact_point=(1600,900),
        _last_racket_contact_label='verified outgoing contact flight',
        _last_racket_contact_player=0,_current_server_index=lambda:0,
        far_serve_image_baseline=True,_active_serve_area_end='near',
        ground_bounce_count=0,net_area_y_max=585,_serve_bounce_frame_limit=lambda:100)
    if invalid=='rally':obj._point_hit_count=2
    if invalid=='receiver':obj._last_racket_contact_player=1
    if invalid=='landed':obj._serve_landed_in_current_attempt=True
    if invalid=='stale':obj._last_racket_contact_frame=40
    if invalid=='far':obj._active_serve_area_end='far'
    if invalid=='inferred':obj._last_racket_contact_label='inferred contact'
    if invalid=='two_bounces':obj.ground_bounce_count=2
    assert InteractiveBallAnalyzer._serve_bounce_context_live(obj)==(invalid is None)


@pytest.mark.parametrize('invalid',[None,'far','receiver','landed','inferred','before_start'])
def test_service_flight_window_uses_the_verified_strike_after_the_toss(invalid):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=SimpleNamespace(frame_count=100,point_start_frame_internal=40,
        far_serve_image_baseline=True,_active_serve_area_end='near',
        _serve_landed_in_current_attempt=False,_point_hit_count=1,
        _last_racket_contact_frame=70,_last_racket_contact_point=(1600,900),
        _last_racket_contact_label='verified outgoing contact flight',
        _last_racket_contact_player=0,_current_server_index=lambda:0,net_area_y_max=585)
    if invalid=='far':obj._active_serve_area_end='far'
    if invalid=='receiver':obj._last_racket_contact_player=1
    if invalid=='landed':obj._serve_landed_in_current_attempt=True
    if invalid=='inferred':obj._last_racket_contact_label='inferred contact'
    if invalid=='before_start':obj._last_racket_contact_frame=20
    assert InteractiveBallAnalyzer._serve_bounce_frame_limit(obj)==(75 if invalid is None else 45)


@pytest.mark.parametrize('kind',['serve','far','player','no_reversal','already_bounced','rally'])
def test_measured_first_landing_is_not_blocked_by_live_service_context(monkeypatch,kind):
    from unittest.mock import MagicMock
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture',lambda _:MagicMock())
    obj=InteractiveBallAnalyzer('unused',enable_player_tracking=False,
        enable_player_learning=False,point_history_file=None)
    obj.frame_count=100;obj.point_start_frame_internal=50
    obj.far_serve_image_baseline=True;obj._active_serve_area_end='far' if kind=='far' else 'near'
    obj._awaiting_serve_bounce=True;obj.ball_center=(187,300);obj.ball_size=30
    obj.net_area_y_min=400;obj.net_area_y_max=585
    obj.last_motion=dict(dx=-10,dy=6,distance=11.66,direction_deg=149)
    obj.motion_history=[dict(frame=n,pos=p,prev_pos=(p[0]+10,p[1]-6),distance=11.66)
        for n,p in zip(range(97,100),[(207,288),(197,294),(187,300)])]
    obj._last_tracked_candidate_motion_frame=99
    obj._last_tracked_candidate_motion_mean=10;obj._last_tracked_candidate_motion_max=90
    obj._causal_flight_anchor_frame=99;obj._last_racket_contact_frame=80
    obj.ground_bounce_count=1 if kind=='already_bounced' else 0
    obj.tracking_hsv_ranges=[dict(h_min=20,h_max=85,s_min=5,s_max=255,v_min=100,v_max=255)]
    obj._player_point_zone=lambda p:'player_body' if kind=='player' else None
    obj._serve_bounce_context_live=lambda:kind!='rally'
    obj._get_ground_bounce_context=lambda *args,**kwargs:dict(origin=(187,300),expected=(187,300))
    obj._confirmed_boundary_reversal_out_candidate=lambda *args:(False,None)
    registered=[]
    obj._register_ground_bounce_from_context=lambda *args:registered.append(args) or True
    pos=(177,306) if kind=='no_reversal' else (180,284)
    candidate=dict(pos=pos,area=18,hsv=np.array([40,80,200]),motion_mean=10,motion_max=90)
    def commit(c,frame):
        dx,dy=c['pos'][0]-obj.ball_center[0],c['pos'][1]-obj.ball_center[1]
        obj.ball_center=c['pos'];obj.ball_size=c['area']
        obj.last_motion=dict(dx=dx,dy=dy,distance=float(np.hypot(dx,dy)),direction_deg=0)
        return obj.ball_center
    obj._commit_night_visible_ball_recovery=commit
    obj._commit_local_ball_flight(candidate,np.zeros((600,800,3),np.uint8))
    assert bool(registered)==(kind in ('serve','rally'))


@pytest.mark.parametrize('kind',['left_reach','inside','too_far','rally','unlanded','stale'])
def test_verified_serve_receiver_reach_includes_the_racket_outside_a_narrow_box(monkeypatch,kind):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    track=SimpleNamespace(bbox=(200,120,100,140),last_frame=70 if kind=='stale' else 99,confidence=.6)
    point=(220,364) if kind=='inside' else (170,364) if kind=='too_far' else (196,364)
    obj=SimpleNamespace(frame_count=100,point_start_frame_internal=50,
        far_serve_image_baseline=True,_active_serve_area_end='near',_awaiting_serve_bounce=False,
        _serve_landed_in_current_attempt=kind!='unlanded',_last_serve_bounce_was_in=True,
        _point_hit_count=2 if kind=='rally' else 1,ground_bounce_count=1,last_ground_bounce_frame=90,
        _last_racket_contact_frame=80,_far_contact_pending=None,
        player_tracker=SimpleNamespace(tracks=dict(far=track)),ball_center=point,ball_size=18,
        last_motion=dict(dx=-7,dy=-16,distance=17.5),motion_history=[],net_area_y_min=400,net_area_y_max=585,
        tracking_hsv_ranges=[{}],_visible_near_return_active=lambda:True,_player_point_zone=lambda p:None,
        _causal_flight_anchor_frame=99,_terminal_previous_gray=None,stuck_frame_count=0)
    monkeypatch.setattr('far_contact_recovery.candidates',lambda *args,**kwargs:[])
    result=InteractiveBallAnalyzer._recover_far_player_contact(obj,np.zeros((600,800,3),np.uint8))
    assert (result is not None)==(kind in ('left_reach','inside'))
