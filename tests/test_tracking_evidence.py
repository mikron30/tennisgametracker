from types import SimpleNamespace
import numpy as np
from tracking_evidence import player_box_motion, unsupported_size_collapse
from serve_stance_guard import evaluate_serve_stance


def test_empty_player_box_has_no_support():
    z=np.zeros((100,100),dtype=np.uint8)
    a=SimpleNamespace(_terminal_previous_gray=z,_terminal_current_gray=z.copy())
    assert player_box_motion(a,(10,10,70,70))==0
    a._terminal_current_gray[40:50,40:50]=100
    assert player_box_motion(a,(10,10,70,70))>.01


def test_no_image_is_not_motion_evidence():
    assert player_box_motion(SimpleNamespace(),(0,0,10,10)) is None


def test_tiny_off_path_override_rejected():
    assert unsupported_size_collapse(5,62.5,dict(predicted_distance=10,motion_max=40),dict(predicted_distance=90))


def test_small_on_path_ball_retained():
    assert not unsupported_size_collapse(5,62.5,dict(predicted_distance=90,motion_max=40),dict(predicted_distance=10))


def test_static_current_does_not_lock_out_recovery():
    assert not unsupported_size_collapse(5,62.5,dict(predicted_distance=10,motion_max=0),dict(predicted_distance=90))


def test_weak_stationary_false_player_cannot_certify_stance():
    z=np.zeros((2160,3840),dtype=np.uint8)
    track=SimpleNamespace(bbox=(2800,1240,300,450),center=(2950,1465),confidence=.46,visible=True,shoes=(2950,1672))
    a=SimpleNamespace(config_file='hsv_config_04_left_night.json',frame_count=4374,
                      player_tracker=SimpleNamespace(tracks={'near':track}),
                      player_tracking_enabled=True,net_area_y_min=400,net_area_y_max=585,
                      _terminal_previous_gray=z,_terminal_current_gray=z)
    r=evaluate_serve_stance(a,(2870,760),np.zeros((2160,3840,3),dtype=np.uint8))
    assert r['decision']=='hold'
    assert 'motion support' in r['reason']
    assert a._serve_stance_last_valid is None
