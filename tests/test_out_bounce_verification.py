import io
from types import SimpleNamespace
import numpy as np
from out_bounce_verification import recover_continuing_ball
from interactive_ball_analyzer import _QuietTrackerOutput


def fixture(candidate=(2933,691), moving=True):
    after = np.zeros((900,3200), dtype=np.uint8)
    if moving:
        x,y = candidate
        after[y-10:y+11,x-10:x+11] = 100
    return SimpleNamespace(
        _is_night_session_config=lambda: True, frame_count=4285,
        prev_motion=dict(dx=1,dy=25,distance=25),
        last_motion=dict(dx=-32,dy=-122,distance=126.1),
        ball_center=(2889,567), ball_size=12.0, _terminal_current_frame=after,
        _terminal_previous_gray=np.zeros_like(after), _terminal_current_gray=after,
        _terminal_moving_ball_candidate=lambda *a,**kw: candidate,
        ball_velocity_history=[25,126.1], motion_history=[dict(frame=4285,pos=(2889,567))],
        _point_history_current={'tracking_trace':[dict(frame=4285,pos=[2889,567])]})


def test_v3_switch_uses_previous_frame_anchor():
    a=fixture()
    assert recover_continuing_ball(a,a.ball_center)
    assert a.ball_center == (2933,691)
    assert a.last_delta == (12,2)
    assert a.prev_motion['dy'] == 25
    assert a.motion_history[-1]['prev_pos'] == (2921,689)
    assert len(a.motion_history) == 1
    assert a._point_history_current['tracking_trace'][0]['pos'] == [2933,691]


def test_static_candidate_cannot_cancel_out():
    a=fixture(moving=False)
    assert not recover_continuing_ball(a,a.ball_center)
    assert a.ball_center == (2889,567)


def test_rebounding_candidate_cannot_cancel_out():
    a=fixture(candidate=(2933,660))
    assert not recover_continuing_ball(a,a.ball_center)


def test_missing_images_preserve_out():
    a=fixture();a._terminal_previous_gray=None
    assert not recover_continuing_ball(a,a.ball_center)


def test_normal_rebound_is_not_replaced():
    a=fixture();a.last_motion=dict(dx=1,dy=-25)
    assert not recover_continuing_ball(a,a.ball_center)


def test_tiny_high_speed_night_fragment_cannot_end_out():
    a=fixture(moving=False)
    a.frame_count=4298
    a.ball_center=(2972,730)
    a.ball_size=2.0
    a.prev_motion=dict(dx=29,dy=-22,distance=36.4)
    a.last_motion=dict(dx=28,dy=188,distance=190.1)
    a._terminal_moving_ball_candidate=lambda *args,**kwargs: None
    assert recover_continuing_ball(a,a.ball_center)
    assert a._last_out_bounce_suppressed_frame == 4298
    assert a._last_out_bounce_suppressed_point == (2972,730)


def test_large_high_speed_candidate_still_requires_real_recovery():
    a=fixture(moving=False)
    a.ball_center=(2972,730)
    a.ball_size=8.0
    a.prev_motion=dict(dx=29,dy=-22,distance=36.4)
    a.last_motion=dict(dx=28,dy=188,distance=190.1)
    a._terminal_moving_ball_candidate=lambda *args,**kwargs: None
    assert not recover_continuing_ball(a,a.ball_center)


def test_large_verified_recovery_vector_cannot_create_next_frame_out():
    a=fixture(moving=False)
    a.frame_count=4307
    a.ball_center=(2906,760)
    a.prev_motion=dict(dx=-833,dy=-1043,distance=1334.8)
    a.last_motion=dict(dx=-73,dy=167,distance=182.3)
    a._last_motion_reacq_frame=4306
    a._last_motion_reacq_pos=(2979,593)
    a._terminal_moving_ball_candidate=lambda *args,**kwargs: None
    assert recover_continuing_ball(a,a.ball_center)
    assert a._last_out_bounce_suppressed_frame == 4307
    assert a._last_out_bounce_suppressed_point == (2906,760)


def test_mismatched_recovery_marker_does_not_suppress_out():
    a=fixture(moving=False)
    a.frame_count=4307
    a.ball_center=(2906,760)
    a.prev_motion=dict(dx=-833,dy=-1043,distance=1334.8)
    a.last_motion=dict(dx=-73,dy=167,distance=182.3)
    a._last_motion_reacq_frame=4306
    a._last_motion_reacq_pos=(2500,500)
    a._terminal_moving_ball_candidate=lambda *args,**kwargs: None
    assert not recover_continuing_ball(a,a.ball_center)


def test_quiet_keeps_stance_diagnostics():
    target=io.StringIO();stream=_QuietTrackerOutput(target)
    stream.write('[SERVE_STANCE_ALLOW] legal\n[SERVE STANCE V3] inside court\ncontour noise\n')
    assert target.getvalue() == '[SERVE_STANCE_ALLOW] legal\n[SERVE STANCE V3] inside court\n'
