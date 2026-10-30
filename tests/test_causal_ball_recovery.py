import cv2
import numpy as np
import pytest

import math
from causal_ball_recovery import find_local_ball


@pytest.mark.parametrize('invalid',[None,'disabled','tiny','pale','static','head'])
def test_partial_serve_departure_needs_fresh_ball_colour_and_full_next_flight(invalid):
    before=np.full((900,900,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(334,480),5 if invalid=='tiny' else 8,
               (220,230,235) if invalid=='pale' else (0,220,240),-1)
    previous=cv2.cvtColor(after if invalid=='static' else before,cv2.COLOR_BGR2GRAY)
    filters=[(np.array([15,0,130],np.uint8),np.array([100,255,255],np.uint8))]
    candidate=find_local_ball(after,previous,(300,500),dict(dx=1,dy=60),480,filters,
        maximum_speed=400,contact_window=True,flexible_contact=True,
        serve_contact_search=invalid!='disabled',
        player_zone=lambda p:'player_head_hat' if invalid=='head' else None)
    if invalid is not None:
        assert candidate is None
        return
    assert candidate and candidate['verified_contact_departure']
    assert math.dist(candidate['pos'],(334,480))<=2
    outgoing=before.copy();cv2.circle(outgoing,(415,355),10,(0,220,240),-1)
    result=find_local_ball(outgoing,cv2.cvtColor(after,cv2.COLOR_BGR2GRAY),candidate['pos'],
        dict(dx=34,dy=-20),candidate['area'],filters,maximum_speed=400,
        outgoing_handoff=True,handoff_reference_speed=60)
    assert result and math.dist(result['pos'],(415,355))<=2


def test_partial_rally_strike_keeps_precontact_size_for_next_clear_departure():
    before=np.full((900,900,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(325,605),8,(0,240,240),-1)
    kwargs=dict(maximum_speed=400,contact_window=True,flexible_contact=True,contact_radius_limit=300)
    partial=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(300,600),
        dict(dx=20,dy=60),420,FILTERS,partial_contact_search=True,**kwargs)
    assert partial and partial['partial_return_contact'] and not partial['verified_contact_departure']
    outgoing=before.copy()
    cv2.circle(outgoing,(315,450),10,(0,240,240),-1)
    result=find_local_ball(outgoing,cv2.cvtColor(after,cv2.COLOR_BGR2GRAY),partial['pos'],
        dict(dx=20,dy=60),420,FILTERS,rising_contact=True,**kwargs)
    assert result and math.dist(result['pos'],(315,450))<=2 and result['verified_contact_departure']


@pytest.mark.parametrize('kind',['ball','slow_ball','disabled','body','static','side_jump','tiny'])
def test_measured_perspective_rebound_retains_yellow_ball_in_approach_lane(kind):
    before=np.full((900,900,3),30,np.uint8)
    cv2.circle(before,(300,500),8,(0,240,240),-1)
    after=np.full_like(before,30)
    target=(390,515) if kind=='side_jump' else (320,513) if kind=='slow_ball' else (330,515)
    cv2.circle(after,target,3 if kind=='tiny' else 8,(0,240,240),-1)
    previous=cv2.cvtColor(after if kind=='static' else before,cv2.COLOR_BGR2GRAY)
    found=find_local_ball(after,previous,(300,500),dict(dx=30,dy=120),180,FILTERS,
        maximum_speed=400,contact_window=True,flexible_contact=True,contact_radius_limit=300,
        perspective_rebound_search=kind!='disabled',player_zone=lambda p:'player_body' if kind=='body' else None)
    if kind in ('ball','slow_ball'):
        assert found is not None and math.dist(found['pos'],target)<=2
        assert not found['verified_contact_departure']
    else:
        assert found is None


def test_verified_contact_window_finds_outgoing_ball_over_tiny_racket_fragment():
    before = np.zeros((600, 1000, 3), np.uint8)
    after = before.copy()
    cv2.circle(after, (500, 270), 8, (0, 240, 240), -1)
    cv2.circle(after, (500, 430), 3, (0, 240, 240), -1)
    args = (after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (500, 400),
            {'dx': 0, 'dy': 70}, 90, [((20, 50, 130), (85, 255, 255))])
    result = find_local_ball(*args, contact_window=True)
    assert result is not None and math.dist(result['pos'], (500, 270)) <= 2
    assert result['verified_contact_departure']
    normal = find_local_ball(*args)
    assert normal is None or math.dist(normal['pos'], (500, 270)) > 2


def test_contact_window_does_not_recover_stationary_background():
    image = np.zeros((600, 1000, 3), np.uint8)
    cv2.circle(image, (500, 270), 8, (0, 240, 240), -1)
    assert find_local_ball(image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        (500, 400), {'dx': 0, 'dy': 70}, 90,
        [((20, 50, 130), (85, 255, 255))], contact_window=True) is None


def test_fast_contact_departure_can_use_bounded_speed_scaled_search():
    before = np.full((800, 1000, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(after, (430, 335), 8, (0, 240, 240), -1)
    args = (after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (300, 500),
            {'dx': -70, 'dy': 90}, 150, [((20, 50, 130), (85, 255, 255))])
    assert find_local_ball(*args, maximum_speed=400, contact_window=True) is None
    found = find_local_ball(*args, maximum_speed=400, contact_window=True,
                            contact_radius_limit=300)
    assert found is not None and found['verified_contact_departure']
    assert math.dist(found['pos'], (430, 335)) <= 2
    assert find_local_ball(after, cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
                          *args[2:], maximum_speed=400, contact_window=True,
                          contact_radius_limit=300) is None


def test_flexible_contact_keeps_smaller_incoming_ball_in_measured_lane():
    before = np.full((600, 1000, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(after, (512, 440), 5, (0, 240, 240), -1)
    cv2.circle(after, (570, 430), 8, (0, 240, 240), -1)
    result = find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (500, 400), {'dx':12,'dy':40}, 200, [((20,50,130),(85,255,255))],
        contact_window=True, flexible_contact=True)
    assert result is not None and math.dist(result['pos'], (512,440)) <= 2
    assert not result['verified_contact_departure']


FILTERS=[(np.array([15,40,100],np.uint8),np.array([85,255,255],np.uint8))]


def test_configured_desaturated_ball_uses_motion_without_accepting_static_pixels():
    before = np.full((240, 320, 3), 40, np.uint8)
    hsv = np.uint8([[[76, 13, 219]]])
    colour = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    after = before.copy()
    cv2.circle(after, (112, 116), 5, colour, -1)
    filters = [(np.array([40, 5, 150]), np.array([110, 60, 255]))]
    args = (after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (100, 100),
            {'dx': 12, 'dy': 16}, 66, filters)
    assert find_local_ball(*args) is None
    found = find_local_ball(*args, minimum_saturation=5)
    assert found is not None and math.dist(found['pos'], (112, 116)) <= 2
    assert find_local_ball(after, cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
                          *args[2:], minimum_saturation=5) is None


def test_desaturated_ball_crossing_white_line_can_be_darker_than_background():
    before = np.full((240, 320, 3), 40, np.uint8)
    before[40:200, 100:120] = 255
    after = before.copy()
    colour = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[76, 20, 219]]]),
                                               cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(after, (110, 116), 5, colour, -1)
    filters = [(np.array([40, 5, 150]), np.array([110, 60, 255]))]
    result = find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (100, 100), {'dx':10,'dy':16}, 66, filters, minimum_saturation=5)
    assert result is not None and math.dist(result['pos'], (110, 116)) <= 2
    assert find_local_ball(before, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (100,100), {'dx':10,'dy':16}, 66, filters, minimum_saturation=5) is None


def scene(offset=0):
    previous=np.full((240,320,3),30,np.uint8)
    current=previous.copy()
    cv2.circle(previous,(100+offset,100),5,(20,210,210),-1)
    cv2.circle(current,(112+offset,116),5,(20,210,210),-1)
    # A more distant moving yellow object must not steal the short flight.
    cv2.circle(current,(180+offset,170),4,(20,240,240),-1)
    return previous,current


@pytest.mark.parametrize('offset',[0,70])
def test_recovers_current_ball_in_measured_lane(offset):
    before,after=scene(offset)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),
                          (100+offset,100),{'dx':12,'dy':16},66,FILTERS)
    assert found is not None
    assert np.linalg.norm(np.array(found['pos'])-np.array([112+offset,116])) <= 2


def test_departing_ball_footprint_cannot_be_recovered():
    before,after=scene()
    after[:]=30
    assert find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),
                           (100,100),{'dx':12,'dy':16},66,FILTERS) is None


def test_static_yellow_hotspot_is_not_motion_evidence():
    before,after=scene()
    before[:]=after
    assert find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),
                           (100,100),{'dx':12,'dy':16},66,FILTERS) is None


def test_visible_ball_slowing_into_bounce_stays_in_local_flight():
    before=np.full((240,320,3),30,np.uint8)
    after=before.copy()
    cv2.circle(before,(100,100),5,(20,210,210),-1)
    cv2.circle(after,(102,114),5,(20,210,210),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),
                          (100,100),{'dx':1,'dy':39},66,FILTERS)
    assert found is not None
    assert np.linalg.norm(np.array(found['pos'])-np.array([102,114]))<=2


def test_full_rebound_beats_smaller_fragment_near_old_position():
    before=np.full((240,320,3),30,np.uint8)
    after=before.copy()
    cv2.circle(before,(100,140),6,(20,210,210),-1)
    cv2.circle(after,(110,115),6,(20,210,210),-1)
    cv2.circle(after,(93,136),3,(20,210,210),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),
                          (100,140),{'dx':5,'dy':14},96,FILTERS)
    assert found is not None
    assert np.linalg.norm(np.array(found['pos'])-np.array([110,115]))<=2


def test_slow_distant_rebound_requires_explicit_recovery_context():
    before=np.full((240,320,3),30,np.uint8)
    after=before.copy()
    cv2.circle(before,(100,100),3,(20,210,210),-1)
    cv2.circle(after,(99,94),3,(20,210,210),-1)
    gray=cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    args=(after,gray,(100,100),{'dx':-2,'dy':4},20,FILTERS)
    assert find_local_ball(*args) is None
    found=find_local_ball(*args,minimum_speed=2)
    assert found is not None
    assert np.linalg.norm(np.array(found['pos'])-np.array([99,94]))<=2


@pytest.mark.parametrize('case',['no_reference','wrong_shape','no_size','fast_contact'])
def test_unsupported_recovery_context_is_left_to_existing_tracker(case):
    before,after=scene()
    gray=cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    area=66
    motion={'dx':12,'dy':16}
    if case=='no_reference': gray=None
    if case=='wrong_shape': gray=gray[:100]
    if case=='no_size': area=None
    if case=='fast_contact': motion={'dx':150,'dy':160}
    assert find_local_ball(after,gray,(100,100),motion,area,FILTERS) is None


@pytest.mark.parametrize("zone, expected", [("player_body", True), ("player_head_hat", False), ("player_shoes", False)])
def test_smaller_departing_ball_can_cross_body_but_not_head_or_shoes(zone, expected):
    before = np.full((240, 320, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(before, (150, 150), 8, (20, 210, 210), -1)
    cv2.circle(after, (132, 120), 6, (20, 210, 210), -1)
    result = find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (150, 150), {"dx": -10, "dy": 40}, 170, FILTERS,
        contact_window=True, flexible_contact=True, player_zone=lambda p: zone)
    assert (result is not None) == expected
    if result:
        assert result["verified_contact_departure"]


def test_lateral_body_fragment_cannot_replace_netward_return():
    before = np.full((240, 320, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(before, (100, 150), 6, (20, 210, 210), -1)
    cv2.circle(after, (140, 135), 6, (20, 210, 210), -1)
    assert find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (100, 150), {"dx": -30, "dy": 30}, 96, FILTERS,
        contact_window=True, flexible_contact=True,
        player_zone=lambda p: "player_body") is None

def test_small_net_ball_can_shrink_at_impact_then_reappear_without_accepting_background():
    before = np.zeros((180, 200, 3), np.uint8)
    after = before.copy()
    cv2.rectangle(after, (99, 110), (101, 112), (0, 240, 240), -1)
    args = (after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (100, 100),
            {'dx': 0, 'dy': 12}, 15, FILTERS)
    assert find_local_ball(*args) is None
    found = find_local_ball(*args, small_net_ball=True, maximum_speed=35)
    assert found is not None and math.dist(found['pos'], (100, 111)) <= 2
    assert find_local_ball(after, cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
                           *args[2:], small_net_ball=True, maximum_speed=35) is None
    rebound = before.copy()
    cv2.circle(rebound, (99, 97), 4, (0, 240, 240), -1)
    recovered = find_local_ball(rebound, cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
        found['pos'], {'dx': 0, 'dy': 12}, 8, FILTERS,
        small_net_ball=True, maximum_speed=35)
    assert recovered is not None and math.dist(recovered['pos'], (99, 97)) <= 2
    # A brighter unrelated spot outside the bounded drop cannot replace it.
    far = before.copy()
    cv2.circle(far, (140, 110), 5, (0, 240, 240), -1)
    assert find_local_ball(far, args[1], *args[2:], small_net_ball=True,
                           maximum_speed=35) is None

@pytest.mark.parametrize("fragment_zone", ["player_head_hat", "racket_fragment"])
def test_fast_clear_contact_return_can_leave_incoming_lane_without_taking_head_fragment(fragment_zone):
    before = np.full((900, 900, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(after, (455, 450), 8, (0, 240, 240), -1)
    cv2.circle(after, (340, 560), 6, (0, 240, 240), -1)
    args = (after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (300, 600),
            {'dx': -45, 'dy': 65}, 200, FILTERS)
    zone = lambda pos: fragment_zone if pos[0] < 380 and pos[1] > 500 else None
    found = find_local_ball(*args, maximum_speed=400, contact_window=True,
        flexible_contact=True, contact_radius_limit=300, player_zone=zone)
    assert found is not None and math.dist(found['pos'], (455, 450)) <= 2
    assert found['verified_contact_departure']
    assert find_local_ball(after, cv2.cvtColor(after, cv2.COLOR_BGR2GRAY),
        *args[2:], maximum_speed=400, contact_window=True, flexible_contact=True,
        contact_radius_limit=300, player_zone=zone) is None

def test_return_colour_context_expires_on_opponent_contact_or_offscreen_return():
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 120
    analyzer.point_start_frame_internal = 80
    analyzer.net_area_y_max = 300
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (200, 500)
    assert analyzer._visible_near_return_active()
    for field, invalid in [('_top_return_origin_frame', 110),
                           ('_recent_offscreen_return_frame', 115),
                           ('_last_racket_contact_point', (200, 200)),
                           ('frame_count', 146)]:
        original = getattr(analyzer, field, None)
        setattr(analyzer, field, invalid)
        assert not analyzer._visible_near_return_active()
        if original is None:
            delattr(analyzer, field)
        else:
            setattr(analyzer, field, original)


def test_nearly_grey_ball_hue_can_change_while_static_white_mark_stays_rejected():
    before = np.full((200, 300, 3), 50, np.uint8)
    cv2.circle(before, (190, 80), 4, (230, 230, 230), -1)
    after = before.copy()
    colour = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[165, 5, 200]]]), cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(after, (120, 85), 3, colour, -1)
    filters = [(np.array([0, 0, 140], np.uint8), np.array([179, 25, 255], np.uint8))]
    found = find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (100, 80), {'dx': 20, 'dy': 4}, 20, filters, minimum_saturation=0)
    assert found is not None and math.dist(found['pos'], (120, 85)) <= 2
    assert find_local_ball(before, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (100, 80), {'dx': 20, 'dy': 4}, 20, filters, minimum_saturation=0) is None

def test_first_return_sample_can_be_shorter_than_half_incoming_speed():
    before = np.full((700, 700, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(after, (388, 345), 8, (0, 240, 240), -1)
    cv2.circle(after, (220, 490), 6, (0, 240, 240), -1)
    result = find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (300, 400), {'dx': -80, 'dy': 160}, 140, FILTERS,
        maximum_speed=400, contact_window=True, flexible_contact=True,
        contact_radius_limit=300, player_zone=lambda p: None)
    assert result is not None and math.dist(result['pos'], (388,345)) <= 2
    assert result['verified_contact_departure']


def test_minor_upward_racket_movement_is_not_a_fast_contact_departure():
    before = np.full((700, 700, 3), 30, np.uint8)
    after = before.copy()
    cv2.circle(after, (279, 380), 6, (0, 240, 240), -1)
    result = find_local_ball(after, cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
        (300,400), {'dx': -80, 'dy': 79}, 146, FILTERS,
        maximum_speed=400, contact_window=True, flexible_contact=True,
        contact_radius_limit=300, player_zone=lambda p: 'racket_fragment')
    assert result is None

def test_outgoing_handoff_recovers_full_step_after_partial_contact_sample():
    before = np.full((900,900,3),30,np.uint8)
    after = before.copy()
    cv2.circle(after,(488,407),8,(0,240,240),-1)
    args=(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(300,600),
          {'dx':88,'dy':-55},290,FILTERS)
    assert find_local_ball(*args,maximum_speed=400) is None
    cv2.circle(after,(390,530),8,(0,240,240),-1)
    zone=lambda p:'player_body' if p[0]<420 and p[1]>500 else None
    found=find_local_ball(*args,maximum_speed=400,outgoing_handoff=True,player_zone=zone)
    assert found is not None and math.dist(found['pos'],(488,407))<=2
    assert find_local_ball(after,cv2.cvtColor(after,cv2.COLOR_BGR2GRAY),
        *args[2:],maximum_speed=400,outgoing_handoff=True,player_zone=zone) is None


def test_outgoing_handoff_rejects_backward_and_player_fragments():
    before=np.full((900,900,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(100,700),8,(0,240,240),-1)
    cv2.circle(after,(390,530),8,(0,240,240),-1)
    args=(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(300,600),
          {'dx':88,'dy':-55},290,FILTERS)
    zone=lambda p:'player_body' if p[0]>300 else None
    assert find_local_ball(*args,maximum_speed=400,outgoing_handoff=True,player_zone=zone) is None
    assert find_local_ball(after,args[1],args[2],{'dx':88,'dy':55},290,FILTERS,
                          maximum_speed=400,outgoing_handoff=True) is None


@pytest.mark.parametrize('enabled,static',[(True,False),(False,False),(True,True)])
def test_distant_serve_joins_only_fresh_small_paint_fragments(enabled,static):
    before=np.full((120,120,3),30,np.uint8)
    after=before.copy()
    cv2.rectangle(after,(48,47),(50,48),(240,240,240),-1)
    cv2.rectangle(after,(48,51),(50,52),(240,240,240),-1)
    previous=cv2.cvtColor(after if static else before,cv2.COLOR_BGR2GRAY)
    filters=[(np.array([0,0,140],np.uint8),np.array([179,45,255],np.uint8))]
    result=find_local_ball(after,previous,(60,50),dict(dx=-7,dy=-1),8,filters,
        minimum_speed=2,minimum_saturation=0,serve_rebound_search=enabled)
    if enabled and not static:
        assert result and math.dist(result['pos'],(49,49))<=2
    else:
        assert result is None


@pytest.mark.parametrize('invalid',[None,'disabled','player','static','pale'])
def test_fast_measured_return_keeps_a_short_clear_ground_rebound(invalid):
    before=np.full((500,500,3),30,np.uint8)
    after=before.copy()
    cv2.circle(after,(270,270),11,(220,230,235) if invalid=='pale' else (0,220,240),-1)
    filters=[(np.array([0,0,100],np.uint8),np.array([179,255,255],np.uint8))]
    result=find_local_ball(after,cv2.cvtColor(after if invalid=='static' else before,cv2.COLOR_BGR2GRAY),
        (250,300),dict(dx=6,dy=155),364,filters,maximum_speed=400,
        perspective_rebound_search=invalid!='disabled',minimum_saturation=0,
        player_zone=lambda p:'player_body' if invalid=='player' else None)
    assert (result is not None)==(invalid is None)
    if result:assert not result['verified_contact_departure']
