import cv2
import numpy as np
import pytest

from causal_ball_recovery import find_local_ball


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
