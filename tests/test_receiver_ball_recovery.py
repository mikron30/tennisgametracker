from types import SimpleNamespace
import cv2
import numpy as np
from receiver_ball_recovery import find_receiver_ball


def frame(points):
    f=np.zeros((600,1000,3),dtype=np.uint8)
    for p in points:cv2.circle(f,p,4,(0,255,255),-1)
    return f


def test_two_images_required():
    a=SimpleNamespace(ball_center=(450,100),frame_count=1)
    assert find_receiver_ball(a,frame([(440,100)])) is None
    a.frame_count=2
    r=find_receiver_ball(a,frame([(437,104)]))
    assert r['pos']==(437,104)


def test_static_patch_not_recovered():
    a=SimpleNamespace(ball_center=(450,100),frame_count=1)
    f=frame([(440,100)])
    find_receiver_ball(a,f);a.frame_count=2
    assert find_receiver_ball(a,f) is None


def test_gap_restarts_confirmation():
    a=SimpleNamespace(ball_center=(450,100),frame_count=1)
    find_receiver_ball(a,frame([(440,100)]));a.frame_count=10
    assert find_receiver_ball(a,frame([(437,104)])) is None


def test_multiple_moving_blobs_are_ambiguous():
    a=SimpleNamespace(ball_center=(450,100),frame_count=1)
    find_receiver_ball(a,frame([(440,100),(490,120)]));a.frame_count=2
    assert find_receiver_ball(a,frame([(437,104),(487,124)])) is None
