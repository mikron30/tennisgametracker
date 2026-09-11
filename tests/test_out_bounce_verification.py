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
        ball_center=(2889,567), _terminal_current_frame=after,
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


def test_quiet_keeps_stance_diagnostics():
    target=io.StringIO();stream=_QuietTrackerOutput(target)
    stream.write('[SERVE_STANCE_ALLOW] legal\n[SERVE STANCE V3] inside court\ncontour noise\n')
    assert target.getvalue() == '[SERVE_STANCE_ALLOW] legal\n[SERVE STANCE V3] inside court\n'
