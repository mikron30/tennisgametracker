from types import SimpleNamespace
import math
import cv2
import numpy as np
from receiver_ball_recovery import confirm_visible_reentry, find_receiver_ball


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


def test_near_player_reacquires_unique_moving_ball_beside_net():
    a=SimpleNamespace(
        ball_center=(700,500), frame_count=20,
        _player_reacq_protect_until_frame=80,
        _last_racket_contact_frame=12, _last_racket_contact_point=(700,500),
        net_area_points=[(300,260),(700,270),(900,280),(300,330),(700,340),(900,350)],
    )
    assert find_receiver_ball(a, frame([(720,360)]), allow_near_net=True) is None
    a.frame_count=21
    recovered=find_receiver_ball(a, frame([(745,390)]), allow_near_net=True)
    assert recovered['pos']==(745,390)
    assert recovered['source']=='near_net_temporal'


def test_near_player_reentry_requires_recent_contact():
    a=SimpleNamespace(
        ball_center=(700,500), frame_count=20,
        _player_reacq_protect_until_frame=80,
        _last_racket_contact_frame=-100, _last_racket_contact_point=(700,500),
        net_area_points=[(300,260),(700,270),(900,280),(300,330),(700,340),(900,350)],
    )
    assert find_receiver_ball(a,frame([(720,360)])) is None
    a.frame_count=21
    assert find_receiver_ball(a,frame([(745,390)])) is None


def test_visible_reentry_accepts_one_ranked_moving_path():
    a=SimpleNamespace(frame_count=1)
    first={'pos':(400,260),'area':38.0,'hsv':(30,255,255)}
    second={'pos':(425,240),'area':40.0,'hsv':(30,255,255)}
    assert confirm_visible_reentry(a, frame([first['pos']]), first) is None
    a.frame_count=2
    recovered=confirm_visible_reentry(a, frame([second['pos']]), second)
    assert recovered['pos']==(425,240)
    assert recovered['local_ai_motion_override']['dx']==25.0
    assert recovered['local_ai_motion_override']['dy']==-20.0


def test_visible_reentry_rejects_position_change_without_image_motion():
    a=SimpleNamespace(frame_count=1)
    first={'pos':(400,260),'area':38.0}
    second={'pos':(425,240),'area':40.0}
    confirm_visible_reentry(a,frame([]),first)
    a.frame_count=2
    assert confirm_visible_reentry(a,frame([]),second) is None


def test_visible_reentry_velocity_is_per_frame_after_gap():
    a=SimpleNamespace(frame_count=1)
    first={'pos':(400,260),'area':38.0}
    second={'pos':(425,240),'area':40.0}
    confirm_visible_reentry(a,frame([first['pos']]),first)
    a.frame_count=3
    recovered=confirm_visible_reentry(a,frame([second['pos']]),second)
    assert recovered['local_ai_motion_override']['dx']==12.5
    assert recovered['local_ai_motion_override']['dy']==-10.0


def test_visible_reentry_commit_preserves_measured_trajectory_sample():
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    a=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    a.frame_count=1
    first={'pos':(400,260),'area':38.0,'hsv':(30,255,255)}
    second={'pos':(425,240),'area':40.0,'hsv':(30,255,255)}
    confirm_visible_reentry(a,frame([first['pos']]),first)
    a.frame_count=2
    candidate=confirm_visible_reentry(a,frame([second['pos']]),second)
    a.ball_center=(800,500)  # The stale player anchor is not the ball's origin.
    a.motion_history=[]
    a._local_ai_tight_roi_attempt_frame=2
    a._activate_regular_hsv=lambda: None
    a._commit_night_visible_ball_recovery(candidate,frame([second['pos']]))
    assert a.last_delta==(25.,-20.)
    assert a.motion_history[-1]['pos']==second['pos']
    assert a.motion_history[-1]['prev_pos']==first['pos']
    assert a.motion_history[-1]['frame']==2




def test_recovery_carries_measured_motion_to_commit_and_arbitration():
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    a=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    a.ball_center=(450,100);a.frame_count=1
    first=frame([(440,100)]);second=frame([(437,104)])
    find_receiver_ball(a,first);a.frame_count=2
    candidate=find_receiver_ball(a,second)
    expected=cv2.absdiff(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)[96:113,429:446],
                         cv2.cvtColor(second,cv2.COLOR_BGR2GRAY)[96:113,429:446])
    assert candidate['motion_max']==float(expected.max())
    assert candidate['motion_mean']==float(expected.mean())
    assert candidate['motion_max']>=50
    assert candidate['contact_ai_same_frame_protected'] is True
    assert a._contact_local_ai_cooldown_until_frame==2

    # This is the main-loop arbitration layer V7 did not exercise. A verified
    # temporal receiver recovery must not arm Contact Local AI again on the
    # same frame; its "AI miss" fallback would restore the old anchor.
    a.local_ai_recovery=object()
    assert a._contact_local_ai_trigger((450,100),candidate['pos'],{}) is None

    a._local_ai_tight_roi_attempt_frame=2
    a._update_recovered_motion=lambda old,new: None
    a._activate_regular_hsv=lambda: None
    a._commit_night_visible_ball_recovery(candidate,second)
    assert a._last_tracked_candidate_motion_max==candidate['motion_max']
    assert a._last_motion_reacq_frame==a.frame_count
    assert a._last_motion_reacq_pos==candidate['pos']

    # V10 repairs the pair consumed by the outer jump guard. A large recovery
    # from a poisoned anchor is therefore eligible for the existing 1500 px
    # motion-reacquisition limit instead of being rejected by the normal 400 px
    # one-frame gate.
    poisoned_anchor=(1500,900)
    jump=math.hypot(candidate['pos'][0]-poisoned_anchor[0],
                    candidate['pos'][1]-poisoned_anchor[1])
    assert 400 < jump < 1500
    assert (
        a._last_motion_reacq_frame==a.frame_count and
        a._last_motion_reacq_pos==candidate['pos']
    )

    a._finalize_pending_provisional_static_candidate=lambda: False
    a._player_point_zone=lambda p:'racket_fragment'
    a._try_active_tight_local_ai_hold=lambda *args,**kwargs: None
    a._local_ai_recovery_reason=lambda *args: None
    a._maybe_clear_post_serve_pre_net_recovery=lambda *args,**kwargs: None
    def unexpected_restore(snapshot):
        raise AssertionError('measured moving candidate was rolled back')
    a._restore_tracking_state_for_provisional_guard=unexpected_restore
    selected=a._try_local_ai_recovery((450,100),candidate['pos'],5,
                                     pre_track_snapshot={'ball_center':(450,100)},frame=second)
    assert selected==candidate['pos']==a.ball_center
