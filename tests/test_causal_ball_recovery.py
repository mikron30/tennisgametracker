import cv2
import numpy as np
import pytest

import math
from causal_ball_recovery import find_local_ball


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


FILTERS=[(np.array([15,40,100],np.uint8),np.array([85,255,255],np.uint8))]


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
