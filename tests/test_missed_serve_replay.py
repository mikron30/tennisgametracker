from types import SimpleNamespace

import pytest

from missed_serve_replay import MissedServeReplay


class Capture:
    def __init__(self): self.position=100;self.seeks=[];self.fail=False
    def get(self, key): return self.position
    def set(self, key, value):
        self.seeks.append(value)
        if self.fail: return False
        self.position=value
        return True


def scenario():
    owner=SimpleNamespace(frame_count=100,cap=Capture(),far_serve_image_baseline=True,
        _active_serve_area_end='near',_last_point_outcome_category='net_error',
        _point_history_point_index=1,score_games=[2,0],score_points=[1,0],
        current_serve_attempt=1,player_tracker=None,_serve_stance_guard_decision='allow',
        _serve_stance_last_valid=dict(frame=100,side='near',confidence=.8),
        _current_server_index=lambda:0)
    replay=MissedServeReplay();replay.remember(owner,99)
    owner.frame_count=200;owner.cap.position=200
    owner._serve_stance_last_valid['frame']=200
    history=[(2200,1000),(2205,950),(2210,900),(2215,850)]
    details=[dict(frame=n,aspect_ratio=1.1,circularity=.5,motion_delta=30) for n in range(197,201)]
    model=dict(center_line=dict(a=0,b=2000))
    return owner,replay,history,details,model


def test_missing_left_point_rewinds_without_guessing_a_winner_or_duplicate_row():
    owner,replay,*args=scenario()
    assert replay.rewind_if_missing(owner,*args)==99
    assert owner.frame_count==100 and owner.cap.position==100
    assert owner.score_games==[2,0] and owner.score_points==[1,0]
    assert owner._point_history_point_index==1 and owner._missed_serve_retry_until==200
    owner.frame_count=200;owner._serve_stance_last_valid['frame']=200
    assert replay.rewind_if_missing(owner,*args) is None
    assert owner.cap.seeks==[100]


@pytest.mark.parametrize('kind',['stale_stance','falling','static','new_row','expected_side','seek_failure'])
def test_ambiguous_or_already_committed_intervals_are_not_restored(kind):
    owner,replay,history,details,model=scenario()
    if kind=='stale_stance': owner._serve_stance_last_valid['frame']=100
    if kind=='falling': history=[(2200,900)]*4
    if kind=='static': details[-1]['motion_delta']=0
    if kind=='new_row': owner._point_history_point_index=2
    if kind=='expected_side': history=[(x-500,y) for x,y in history]
    if kind=='seek_failure': owner.cap.fail=True
    assert replay.rewind_if_missing(owner,history,details,model) is None
    assert owner.frame_count==200 and owner.cap.position==200
    assert owner.score_points==[1,0]


def test_wrong_side_after_first_fault_recovers_the_missing_second_serve():
    owner,replay,history,details,model=scenario()
    owner.frame_count=100;owner._last_point_outcome_category='first_serve_fault'
    owner.current_serve_attempt=2;owner._point_history_point_index=2
    replay.remember(owner,99)
    owner.frame_count=200;owner._serve_stance_last_valid['frame']=200
    assert replay.rewind_if_missing(owner,history,details,model)==99
    assert owner.current_serve_attempt==2 and owner._point_history_point_index==2
    assert owner.score_points==[1,0]


def test_partial_falling_toss_on_wrong_server_half_requests_evidence_not_a_point():
    owner,replay,history,details,model=scenario()
    owner._serve_stance_last_valid['center']=(2250,1500)
    history=[(2050,990),(2045,995),(2040,1005),(2035,1015)]
    assert replay.rewind_if_missing(owner,history,details,model)==99
    assert owner.score_points==[1,0] and owner._point_history_point_index==1


def test_replay_cannot_use_future_frame_buffers():
    owner,replay,*args=scenario()
    owner._local_ai_frame_buffer=[dict(frame=200)]
    owner._terminal_current_gray=object()
    assert replay.rewind_if_missing(owner,*args)==99
    assert owner._local_ai_frame_buffer==[] and owner._terminal_current_gray is None


def test_failed_restore_keeps_live_score_and_capture(monkeypatch):
    import missed_serve_replay as module
    owner,replay,*args=scenario()
    original=module.restore
    calls=[]
    def restore_once(obj,state):
        calls.append(state)
        if len(calls)==1:
            obj.score_points=[99,99]
            raise RuntimeError('injected restore failure')
        original(obj,state)
    monkeypatch.setattr(module,'restore',restore_once)
    assert replay.rewind_if_missing(owner,*args) is None
    assert owner.score_points==[1,0] and owner.cap.position==200 and owner.frame_count==200


def test_missing_service_lines_use_calibrated_midpoint_only_to_request_replay():
    owner,replay,history,details,_=scenario()
    owner.serve_area_x_min=760;owner.serve_area_x_max=2900
    assert replay.rewind_if_missing(owner,history,details,None)==99
    assert owner.score_points==[1,0]


def test_wrong_side_after_fault_can_request_replay_from_a_clear_partial_toss():
    owner,replay,history,details,model=scenario()
    owner.frame_count=100;owner.current_serve_attempt=2
    replay=MissedServeReplay();replay.remember(owner,99)
    owner.frame_count=200;owner._serve_stance_last_valid['frame']=200
    owner._serve_stance_last_valid['center']=(2250,1500)
    last=dict(details[-1],area=150)
    assert replay.rewind_if_missing(owner,history[-1:],[last],model)==99
    assert owner.score_points==[1,0] and owner.current_serve_attempt==2
