from types import SimpleNamespace
import math
import cv2
import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.mark.parametrize('net_drop,stationary,expected', [
    (True,False,True),(False,False,False),(True,True,False)])
def test_small_net_drop_retains_fresh_slow_ball_but_rejects_static_pixels(net_drop,stationary,expected):
    from causal_ball_recovery import find_local_ball
    before=np.full((200,200,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(102,100),3,(220,220,220),-1)
    if stationary:
        before=after.copy()
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),
        (100,100),dict(dx=0,dy=10),30,
        [(np.array([0,0,140]),np.array([179,45,255]))],
        minimum_speed=2,maximum_speed=35,minimum_saturation=0,small_net_ball=net_drop)
    assert (found is not None)==expected


def test_measured_yellow_departure_on_racket_beats_clear_orange_shirt():
    from causal_ball_recovery import find_local_ball
    before=np.full((600,800,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(390,360),8,(0,240,240),-1)
    orange=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[4,100,180]]]),cv2.COLOR_HSV2BGR)[0,0])
    cv2.circle(after,(290,340),11,orange,-1)
    result=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(400,400),
        dict(dx=-11,dy=45),156,[(np.array([0,0,130]),np.array([179,255,255]))],
        maximum_speed=400,contact_window=True,minimum_saturation=5,
        flexible_contact=True,contact_radius_limit=180,
        player_zone=lambda p:'racket_fragment' if p[0]>350 else None)
    assert result and math.dist(result['pos'],(390,360))<2


@pytest.mark.parametrize('outgoing,expected', [
    ((382,552), True),   # dissipative rebound in the incoming lateral lane
    ((420,552), False),  # racket changes lateral direction
    ((350,480), False),  # accelerated return
    ((398,552), False),  # almost stationary racket fragment
    ((382,610), False),  # ball continues descending
])
def test_ground_rebound_requires_lane_and_energy_continuity(outgoing, expected):
    from causal_ball_recovery import matches_ground_rebound
    assert matches_ground_rebound((400,600), dict(dx=-37,dy=49),
                                  dict(pos=outgoing), (367,544)) == expected


@pytest.mark.parametrize('prior_bounce,contact_after_bounce,accelerated', [
    (True, False, False), (False, False, False),
    (True, True, False), (True, False, True),
])
def test_used_bounce_is_not_erased_by_a_bounce_like_contact_window(
        prior_bounce, contact_after_bounce, accelerated):
    from unittest.mock import Mock
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count=40
    owner.point_start_frame_internal=1
    owner.ball_center=(400,600)
    owner.ball_size=40
    owner.last_motion=dict(dx=-37,dy=49,distance=61.4)
    owner.motion_history=[dict(frame=37,pos=(468,506),prev_pos=(500,464)),
                          dict(frame=38,pos=(437,551),prev_pos=(468,506)),
                          dict(frame=39,pos=(400,600),prev_pos=(437,551))]
    owner._causal_flight_anchor_frame=39
    owner.tracking_hsv_ranges=[{}]
    owner.net_area_y_max=200
    owner.ground_bounce_count=int(prior_bounce)
    owner.last_ground_bounce_frame=20
    owner._last_racket_contact_frame=30 if contact_after_bounce else 10
    owner._get_ground_bounce_context=lambda *a,**kw:dict(origin=(400,600),expected=(367,544))
    owner._serve_bounce_context_live=lambda:False
    owner._player_point_zone=lambda p:'racket_fragment'
    owner._confirmed_boundary_reversal_out_candidate=lambda *a:(False,None)
    owner._record_racket_contact=Mock()
    owner._player_index_at_point=lambda *a:0
    owner._add_impact_marker=Mock()
    def register(*a):
        owner.ground_bounce_count+=1
        return True
    owner._register_ground_bounce_from_context=register
    def commit(c,frame):
        owner.last_motion=dict(dx=c['pos'][0]-400,dy=c['pos'][1]-600,distance=51.3,direction_deg=-110)
        owner.ball_velocity_history=[51.3]
        owner.ball_center=c['pos']
        return c['pos']
    owner._commit_night_visible_ball_recovery=commit
    candidate=dict(pos=(350,480) if accelerated else (382,552), area=80,
                   motion_mean=44,motion_max=128,verified_contact_departure=True)
    owner._commit_local_ball_flight(candidate,np.zeros((1000,800,3),np.uint8))
    terminal=prior_bounce and not contact_after_bounce and not accelerated
    assert owner._record_racket_contact.call_count == (0 if terminal else 1)
    if terminal:
        assert owner.ground_bounce_count==2
        assert owner._pending_rally_end_reason=='Ball bounced twice on court'
        assert owner._receiver_bounce_terminal['frame']==39


@pytest.mark.parametrize('fresh_person', [True, False])
def test_fresh_person_can_arm_volley_search_without_reassigning_player(fresh_person):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 20
    owner.point_start_frame_internal = 1
    owner.ball_center = (400,400)
    owner.ball_size = 139
    owner.last_motion = dict(dx=12,dy=40,distance=41.8)
    owner.motion_history = []
    owner.net_area_y_min = 80
    owner.net_area_y_max = 100
    owner.tracking_hsv_ranges = [dict(h_min=0,h_max=179,s_min=0,s_max=255,v_min=130,v_max=255)]
    owner._visible_near_return_active = lambda: False
    owner._visible_far_return_active = lambda: True
    owner._visible_extended_near_return_active = lambda: False
    owner._slow_serve_net_drop_active = lambda: False
    owner._point_in_player_contact_corridor = lambda p: False
    owner._player_point_zone = lambda p: None
    owner._last_player_detection_refresh_frame = 20 if fresh_person else 5
    owner.player_tracker = SimpleNamespace(tracks={}, detection_interval=5,
        last_detections=[dict(source='hog',score=.03,bbox=(360,350,80,200))],
        _center_is_in_court=lambda p: True)
    before = np.full((600,800,3),30,np.uint8)
    after = before.copy()
    cv2.circle(after,(412,440),5,(180,180,180),-1)
    cv2.circle(after,(390,300),8,(0,240,240),-1)
    owner._terminal_previous_gray = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    found = owner._recover_local_ball_flight(after)
    assert found is not None
    assert found['verified_contact_departure'] == fresh_person
    expected = (390,300) if fresh_person else (412,440)
    assert np.linalg.norm(np.subtract(found['pos'],expected)) <= 2
    assert owner.player_tracker.tracks == {}


@pytest.mark.parametrize('fresh,person_sized,in_court,linked', [
    (True, True, True, False), (False, True, True, False), (True, False, True, False),
    (True, True, False, True), (True, True, False, False)])
def test_stretched_serve_receiver_motion_only_arms_measured_departure(fresh, person_sized, in_court, linked):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 20
    owner.point_start_frame_internal = 1
    owner.ball_center = (600,400)
    owner.ball_size = 139
    owner.last_motion = dict(dx=90,dy=104,distance=137.5)
    owner.motion_history = []
    owner.net_area_y_min, owner.net_area_y_max = 80,100
    owner.tracking_hsv_ranges = [dict(h_min=0,h_max=179,s_min=0,s_max=255,v_min=130,v_max=255)]
    owner._serve_landed_in_current_attempt = owner._last_serve_bounce_was_in = True
    owner._visible_near_return_active = lambda: False
    owner._visible_far_return_active = lambda: False
    owner._visible_extended_near_return_active = lambda: False
    owner._slow_serve_net_drop_active = lambda: False
    owner._point_in_player_contact_corridor = lambda p: False
    owner._player_point_zone = lambda p: None
    owner._last_player_detection_refresh_frame = 20 if fresh else 5
    tracks = {'near': SimpleNamespace(bbox=(450,360,100,190),last_frame=19)} if linked else {}
    owner.player_tracker = SimpleNamespace(tracks=tracks, detection_interval=5,
        last_detections=[dict(source='motion',score=3,bbox=(490,350,160,180 if person_sized else 30))],
        _center_is_in_court=lambda p: in_court)
    before = np.full((600,800,3),30,np.uint8)
    after = before.copy()
    cv2.circle(after,(690,504),7,(180,180,180),-1)
    cv2.circle(after,(598,332),8,(0,240,240),-1)
    owner._terminal_previous_gray = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    found = owner._recover_local_ball_flight(after)
    assert found is not None
    verified = fresh and person_sized and (in_court or linked)
    assert found['verified_contact_departure'] == verified
    if verified:
        assert np.linalg.norm(np.subtract(found['pos'],(598,332))) <= 2
    assert owner.player_tracker.tracks == tracks


@pytest.mark.parametrize('label,player,age,expected', [
    ('inferred side-recross return',0,18,True),
    ('inferred side-recross return',1,18,False),
    ('inferred side-recross return',0,46,False),
    ('verified outgoing contact flight',0,18,False)])
def test_near_return_ownership_inference_does_not_use_netside_sample_as_racket_position(label, player, age, expected):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 100
    owner.point_start_frame_internal = 1
    owner.net_area_y_max = 560
    owner._last_racket_contact_frame = 100-age
    owner._last_racket_contact_point = (1892,539)
    owner._last_racket_contact_label = label
    owner._last_racket_contact_player = player
    owner._player_index_on_court_end = lambda side: 0
    assert owner._visible_near_return_active() == expected


@pytest.mark.parametrize('contact', [False, True])
def test_contact_ball_remains_separate_from_touching_white_racket_pixels(contact):
    from causal_ball_recovery import find_local_ball
    previous = np.full((300,300,3),20,np.uint8)
    current = previous.copy()
    cv2.line(current,(125,102),(155,135),(220,220,220),4)
    cv2.circle(current,(132,110),6,(0,240,240),-1)
    found = find_local_ball(current,cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),
        (150,150),dict(dx=-37,dy=49),100,
        [(np.array([0,0,130],np.uint8),np.array([179,255,255],np.uint8))],
        maximum_speed=400,contact_window=contact,flexible_contact=True,
        minimum_saturation=0,player_zone=lambda p:None)
    assert found and found['verified_contact_departure'] == contact
    assert np.linalg.norm(np.subtract(found['pos'],(132,110))) <= 2
    assert int(found['hsv'][1]) >= 50


def test_rally_return_is_not_restricted_to_stretched_serve_receiver_sideways_stop():
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 20
    owner.point_start_frame_internal = 1
    owner._point_hit_count = 2
    owner.ball_center = (400,400)
    owner.ball_size = 35
    owner.last_motion = dict(dx=-37,dy=49,distance=61.4)
    owner.motion_history = []
    owner.net_area_y_min, owner.net_area_y_max = 80,100
    owner.tracking_hsv_ranges = [dict(h_min=0,h_max=179,s_min=0,s_max=255,v_min=130,v_max=255)]
    owner._serve_landed_in_current_attempt = owner._last_serve_bounce_was_in = True
    owner._visible_near_return_active = lambda: False
    owner._visible_far_return_active = lambda: True
    owner._visible_extended_near_return_active = lambda: False
    owner._slow_serve_net_drop_active = lambda: False
    owner._point_in_player_contact_corridor = lambda p: False
    owner._player_point_zone = lambda p: 'player_body'
    owner.player_tracker = SimpleNamespace(tracks={'near':SimpleNamespace(bbox=(360,350,80,200),last_frame=19)},
        detection_interval=5,last_detections=[])
    previous = np.full((600,800,3),20,np.uint8)
    current = previous.copy()
    cv2.circle(current,(382,352),5,(0,240,240),-1)
    owner._terminal_previous_gray = cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    found = owner._recover_local_ball_flight(current)
    assert found and found['verified_contact_departure']
    assert np.linalg.norm(np.subtract(found['pos'],(382,352))) <= 2


@pytest.mark.parametrize('fresh,error', [(True,0),(False,0),(True,25)])
def test_small_ball_crossing_white_paint_can_continue_a_fresh_measured_return(fresh,error):
    owner=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count=20
    owner.point_start_frame_internal=1
    owner.ball_center=(400,400)
    owner.ball_size=100
    owner.last_motion=dict(dx=-25,dy=-35,distance=43)
    owner.motion_history=[]
    owner.net_area_y_min,owner.net_area_y_max=80,100
    owner.tracking_hsv_ranges=[dict(h_min=0,h_max=179,s_min=0,s_max=255,v_min=130,v_max=255)]
    owner._visible_near_return_active=lambda:True
    owner._visible_far_return_active=lambda:False
    owner._visible_extended_near_return_active=lambda:False
    owner._slow_serve_net_drop_active=lambda:False
    owner._point_in_player_contact_corridor=lambda p:True
    owner._player_point_zone=lambda p:'player_body'
    owner._causal_flight_anchor_frame=19 if fresh else 17
    owner._last_racket_contact_frame=18
    owner.player_tracker=SimpleNamespace(tracks={},detection_interval=5,last_detections=[])
    previous=np.full((600,800,3),250,np.uint8)
    current=previous.copy()
    cv2.circle(current,(375-error,365),3,(0,180,180),-1)
    owner._terminal_previous_gray=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    found=owner._recover_local_ball_flight(current)
    if fresh and error==0:
        assert found and 6 <= found['motion_mean'] < 12
        assert np.linalg.norm(np.subtract(found['pos'],(375,365)))<=2
    else:
        assert found is None


@pytest.mark.parametrize('reference,point,expected', [
    (0,(225,177),True),(0,(200,130),False),
    (140,(200,130),True),(140,(50,50),False)])
def test_full_frame_after_partial_contact_uses_bounded_incoming_speed(reference,point,expected):
    from causal_ball_recovery import find_local_ball
    before=np.full((500,500,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,point,8,(0,240,240),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(250,250),
        dict(dx=-3,dy=-13),160,
        [(np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8))],
        minimum_speed=2,maximum_speed=400,outgoing_handoff=True,
        handoff_reference_speed=reference,player_zone=lambda p:None)
    assert (found is not None)==expected
    if expected:
        assert np.linalg.norm(np.subtract(found['pos'],point))<=2


@pytest.mark.parametrize('point,radius,expected', [
    ((187,185),8,True),((230,185),8,False),((187,185),3,False)])
def test_slow_ground_rebound_in_front_of_player_remains_ball_flight(point,radius,expected):
    from causal_ball_recovery import find_local_ball
    before=np.full((400,400,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,point,radius,(0,240,240),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(200,200),
        dict(dx=-17,dy=8),189,
        [(np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8))],
        minimum_speed=2,maximum_speed=400,contact_window=True,flexible_contact=True,
        player_zone=lambda p:None)
    assert (found is not None)==expected
    if expected:
        assert not found['verified_contact_departure']
        assert np.linalg.norm(np.subtract(found['pos'],point))<=2


@pytest.mark.parametrize('zone,moving,expected', [
    ('player_body',True,True),('player_body',False,False),
    ('player_head_hat',True,False),('player_shoes',True,False),('racket_fragment',True,False)])
def test_measured_first_return_can_clear_a_broad_body_box(zone,moving,expected):
    from causal_ball_recovery import find_local_ball
    before=np.full((500,500,3),30,np.uint8)
    if not moving:
        cv2.circle(before,(225,177),8,(0,240,240),-1)
    after=before.copy()
    cv2.circle(after,(225,177),8,(0,240,240),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(250,250),
        dict(dx=-3,dy=-13),160,
        [(np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8))],
        minimum_speed=2,maximum_speed=400,outgoing_handoff=True,
        player_zone=lambda p:zone)
    assert (found is not None)==expected


@pytest.mark.parametrize('point,hit', [((280,285),False),((270,210),True),((315,285),True)])
def test_rising_ball_needs_a_real_flight_change_before_counting_another_hit(point,hit):
    from causal_ball_recovery import find_local_ball
    before=np.full((600,600,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,point,7,(0,240,240),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(300,300),
        dict(dx=-10,dy=-10),140,
        [(np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8))],
        minimum_speed=2,maximum_speed=400,contact_window=True,flexible_contact=True,
        rising_contact=True,player_zone=lambda p:None)
    assert found and found['verified_contact_departure']==hit
    assert np.linalg.norm(np.subtract(found['pos'],point))<=2


@pytest.mark.parametrize('radius,ball_selected', [(300,True),(120,False)])
def test_slow_incoming_ball_can_leave_contact_fast_within_explicit_search_limit(radius,ball_selected):
    from causal_ball_recovery import find_local_ball
    before=np.full((600,600,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(215,269),7,(180,180,180),-1)
    cv2.circle(after,(352,155),10,(0,240,240),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(300,300),
        dict(dx=-19,dy=16),226,
        [(np.array([0,0,130],np.uint8),np.array([179,255,255],np.uint8))],
        minimum_speed=2,maximum_speed=400,minimum_saturation=0,contact_window=True,flexible_contact=True,
        contact_radius_limit=radius,player_zone=lambda p:None)
    assert found and (np.linalg.norm(np.subtract(found['pos'],(352,155)))<=2)==ball_selected
