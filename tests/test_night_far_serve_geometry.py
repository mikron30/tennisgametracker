from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from night_serve_geometry import far_baseline
from serve_stance_guard import evaluate_serve_stance


def image():
    frame=np.zeros((2160,3840,3),np.uint8)
    cv2.line(frame,(1400,250),(2700,275),(255,255,255),8)
    cv2.line(frame,(1200,365),(2900,385),(255,255,255),8)
    return frame


def test_detects_far_baseline_before_service_line():
    model=far_baseline(image())
    assert model is not None
    assert abs(model['a']*2100+model['b']-263.5)<12


@pytest.mark.parametrize('feet_y,decision',[(240,'allow'),(360,'reject')])
def test_far_stance_uses_camera_line(feet_y,decision):
    track=SimpleNamespace(bbox=(2050,feet_y-180,130,180),center=(2115,feet_y-90),
                          shoes=(2115,feet_y),visible=True,confidence=.9)
    a=SimpleNamespace(config_file='hsv_config_04_left_night.json',frame_count=100,
                      player_tracking_enabled=True,player_tracker=SimpleNamespace(tracks={'far':track}),
                      net_area_y_min=400,net_area_y_max=585)
    result=evaluate_serve_stance(a,(2100,80),image())
    assert result['decision']==decision
    assert 250<result['baseline_y']<280


def test_missing_far_line_holds_instead_of_using_other_camera():
    assert far_baseline(np.zeros((2160,3840,3),np.uint8)) is None


def test_custom_camera_can_opt_into_image_baseline():
    track = SimpleNamespace(bbox=(2050, 60, 130, 180), center=(2115, 150),
                            shoes=(2115, 240), visible=True, confidence=.9)
    analyzer = SimpleNamespace(config_file='custom_camera.json', frame_count=100,
        far_serve_image_baseline=True, player_tracking_enabled=True,
        player_tracker=SimpleNamespace(tracks={'far': track}),
        net_area_y_min=400, net_area_y_max=585)
    result = evaluate_serve_stance(analyzer, (2100, 80), image())
    assert result['decision'] == 'allow'
    assert 250 < result['baseline_y'] < 280


def test_far_box_scale_glitch_keeps_recent_verified_legal_stance():
    track = SimpleNamespace(bbox=(2050, 60, 130, 180), center=(2115, 150),
                            shoes=(2115, 240), visible=True, confidence=.9)
    analyzer = SimpleNamespace(config_file='custom_camera.json', frame_count=100,
        far_serve_image_baseline=True, player_tracking_enabled=True,
        player_tracker=SimpleNamespace(tracks={'far': track}),
        net_area_y_min=400, net_area_y_max=585)
    assert evaluate_serve_stance(analyzer, (2100, 80), image())['decision'] == 'allow'
    analyzer.frame_count = 105
    track.bbox, track.center, track.shoes = (1980, 40, 270, 540), (2115, 310), (2115, 580)
    result = evaluate_serve_stance(analyzer, (2100, 80), image())
    assert result['decision'] == 'allow'
    assert result['feet'] == (2115, 240)
    # A normal-sized player moving inside remains a real rejection.
    track.bbox, track.center, track.shoes = (2050, 180, 130, 180), (2115, 270), (2115, 360)
    assert evaluate_serve_stance(analyzer, (2100, 80), image())['decision'] == 'reject'
    # An old legal stance cannot excuse a later oversized detection.
    analyzer.frame_count = 120
    track.bbox, track.center, track.shoes = (1980, 40, 270, 540), (2115, 310), (2115, 580)
    assert evaluate_serve_stance(analyzer, (2100, 80), image())['decision'] == 'reject'


def test_segmented_dim_baseline_survives_server_occlusion():
    frame=np.full((2160,3840,3),110,np.uint8)
    for first,last in ((1450,1700),(1800,2050),(2200,2500)):
        cv2.line(frame,(first,250+int((first-1450)*.025)),
                 (last,250+int((last-1450)*.025)),(190,190,190),6)
    # A brighter service line must not replace the back boundary.
    cv2.line(frame,(1200,365),(2900,385),(255,255,255),8)
    model=far_baseline(frame)
    assert model is not None
    assert abs(model['a']*2000+model['b']-264)<12


def test_short_highlight_cannot_certify_a_baseline():
    frame=np.zeros((2160,3840,3),np.uint8)
    cv2.line(frame,(1950,260),(2200,265),(255,255,255),8)
    assert far_baseline(frame) is None


@pytest.mark.parametrize("verified, age, inside, expected", [
    (True, 5, 28, 'allow'), (False, 5, 28, 'reject'),
    (True, 11, 28, 'reject'), (True, 5, 50, 'reject')])
def test_verified_toss_survives_only_small_recent_baseline_jitter(verified, age, inside, expected):
    track = SimpleNamespace(bbox=(2050,60,130,180), center=(2115,150),
        shoes=(2115,240), visible=True, confidence=.9)
    analyzer = SimpleNamespace(config_file='custom_camera.json', frame_count=100,
        far_serve_image_baseline=True, player_tracking_enabled=True,
        player_tracker=SimpleNamespace(tracks={'far': track}),
        net_area_y_min=400, net_area_y_max=585)
    legal = evaluate_serve_stance(analyzer, (2100,80), image())
    assert legal['decision'] == 'allow'
    analyzer.frame_count += age
    bottom = legal['baseline_y'] + inside
    track.bbox, track.center, track.shoes = (2050,bottom-180,130,180), (2115,bottom-90), (2115,bottom)
    result = evaluate_serve_stance(analyzer, (2100,80), image(), verified_toss=verified)
    assert result['decision'] == expected
    if expected == 'allow':
        assert analyzer._serve_stance_last_valid['frame'] == 100
        assert result['feet'] == legal['feet']


def test_local_sideline_uses_inner_visible_paint_and_requires_spanning_segment():
    from night_serve_geometry import local_sideline_x
    frame=np.zeros((1000,1000,3),np.uint8)
    cv2.line(frame,(370,250),(230,450),(255,255,255),4)
    cv2.line(frame,(320,250),(180,450),(255,255,255),4)
    assert abs(local_sideline_x(frame,350,280,'left')-300)<6
    frame[:]=0
    cv2.line(frame,(230,350),(370,350),(255,255,255),4)
    assert local_sideline_x(frame,350,280,'left') is None
