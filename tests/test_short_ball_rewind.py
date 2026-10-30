from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from short_ball_rewind import (Checkpoint, ReplayRejected, ShortBallRewind,
                               best_path, fresh_candidates, restore, snapshot,
                               prestrike_toss_checkpoint, prestrike_toss_path)


@pytest.mark.parametrize('failure',[None,'tiny','jump','body','static','launched','gap'])
def test_prestrike_hsv_replay_requires_contiguous_physical_toss(monkeypatch,failure):
    state=dict(_near_serve_toss_seed=dict(frame=0,position=(100,250),area=300),
               tracking_hsv_ranges=[dict(h_min=20,h_max=85,s_min=50,s_max=255,v_min=130,v_max=255)],
               far_serve_image_baseline=True,_active_serve_area_end='near',
               net_area_y_max=180)
    checkpoint=Checkpoint(frame=10,source=0,point=1,position=(100,120),
                          velocity=(0,-10),area=300,state=(state,{}),prestrike_toss=True)
    labels=list(range(11,15))
    frames=[np.zeros((300,300,3),np.uint8) for _ in labels]
    ledger={n:dict(tracks={}) for n in labels}
    samples=[dict(pos=(100,120-10*i),area=300,motion_mean=60,motion_max=150)
             for i in range(1,5)]
    if failure=='tiny':samples[1]['area']=4
    if failure=='jump':samples[1]['pos']=(180,100)
    if failure=='body':ledger[12]['tracks']={'near':SimpleNamespace(bbox=(90,90,20,20))}
    if failure=='static':samples[1]['motion_mean']=0
    if failure=='launched':state['_serve_launch_direction_x']=1
    if failure=='gap':labels[-1]+=1
    stream=iter(samples)
    monkeypatch.setattr('causal_ball_recovery.find_local_ball',lambda *a,**k:next(stream))
    result=prestrike_toss_path(checkpoint,frames,labels,ledger,frames[0])
    if failure is not None:
        assert result is None
    else:
        assert [s['pos'] for s in result]==[s['pos'] for s in samples]
        assert all('ai_score' not in s for s in result)


def candidate(frame, x, y=80, score=.999, zone=None):
    return dict(frame=frame,x=x,y=y,pos=(x,y),area=20,ai_score=score,
                motion_mean=25,motion_max=120,player_zone=zone)


def test_rewind_beam_recovers_flight_instead_of_high_scoring_stationary_blob():
    groups=[(n,[candidate(n,10*n),candidate(n,30,score=1)]) for n in range(1,7)]
    result=best_path(groups,0,(0,80),(10,0))
    assert [c['x'] for c in result]==[10,20,30,40,50,60]


@pytest.mark.parametrize('kind',['static','teleport','low_score','body','short','missing_end'])
def test_rewind_rejects_unsupported_paths(kind):
    groups=[(n,[candidate(n,10*n)]) for n in range(1,7)]
    if kind=='static':
        groups=[(n,[candidate(n,20)]) for n in range(1,7)]
    if kind=='teleport':
        for _, cs in groups[2:]: cs[0]['x']+=2000
    if kind=='low_score':
        for _, cs in groups: cs[0]['ai_score']=.5
    if kind=='body':
        for _, cs in groups: cs[0]['player_zone']='near'
    if kind=='short':groups=groups[:3]
    if kind=='missing_end':groups[-1]=(6,[])
    assert best_path(groups,0,(0,80),(10,0)) is None


def test_rewind_bridges_occlusion_without_inventing_missing_observations():
    groups=[(n,[] if n in (3,4) else [candidate(n,10*n)]) for n in range(1,9)]
    result=best_path(groups,0,(0,80),(10,0))
    assert [c['frame'] for c in result]==[1,2,5,6,7,8]
    groups[4]=(5,[])
    assert best_path(groups,0,(0,80),(10,0)) is None


def test_fresh_search_sees_ball_outside_configured_hsv_and_excludes_static_ball():
    previous=np.full((400,500,3),30,np.uint8)
    cv2.circle(previous,(200,150),5,(230,230,230),-1)
    image=previous.copy()
    cv2.circle(image,(240,160),5,(0,230,230),-1)
    found=fresh_candidates(image,cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),
                           (210,160),(20,0),1,50)
    assert len(found)==1 and np.linalg.norm(np.array([found[0]['x'],found[0]['y']])-(240,160))<2


def owner():
    obj=SimpleNamespace(frame_count=1,point_start_frame_internal=1,ball_center=(10,80),
        ball_size=20,last_motion=dict(dx=10,dy=0,distance=10),stuck_frame_count=0,
        score=[0,0],ground_bounce_count=1,_point_hit_count=1,net_area_y_max=50,
        _point_history_current=dict(tracking_trace=[dict(frame=1,pos=[10,80])],shot_events=[]),
        motion_history=[dict(frame=1,pos=(10,80),prev_pos=(0,80))],
        player_tracker=SimpleNamespace(tracks={},point_events=[],profile=dict(hits=1),profile_path='profile.json'),
        external_handle=object(),_player_point_zone=lambda p:None,
        _last_tracked_candidate_motion_frame=1,_last_tracked_candidate_motion_mean=25,
        _last_tracked_candidate_motion_max=100,_pending_rally_end_reason=None)
    obj._top_return_wait_active=lambda:False
    obj._back_return_wait_active=lambda:False
    def commit(c, image):
        old=obj.ball_center
        obj.ball_center=c['pos']
        obj.ball_size=c['area']
        obj.motion_history.append(dict(frame=obj.frame_count,pos=c['pos'],prev_pos=old))
        if obj.frame_count==4:
            obj._point_hit_count+=1
            obj.player_tracker.profile['hits']+=1
        if obj.frame_count==5: obj.ground_bounce_count+=1
    obj._commit_local_ball_flight=commit
    obj._maybe_infer_return_contact_from_side_recross=lambda *a,**kw:None
    return obj


def toss_owner(n=103):
    obj=owner()
    obj.frame_count=n
    obj.far_serve_image_baseline=True
    obj._active_serve_area_end='near'
    obj._serve_phase_active=obj._awaiting_serve_bounce=True
    obj._point_hit_count=obj._serve_launch_direction_x=0
    obj._serve_landed_in_current_attempt=False
    obj._near_serve_toss_seed=dict(frame=99,area=300,position=(100,1000))
    obj.serve_area_x_min,obj.serve_area_x_max=0,1000
    obj.net_area_y_max=585
    obj.ball_size=300
    obj.ball_center=(104,465-(n-103)*12)
    obj.last_motion=dict(dx=2,dy=-12,distance=12.2)
    obj._last_tracked_candidate_motion_frame=n
    obj.motion_history=[dict(frame=k,pos=(100+(k-(n-2))*2,obj.ball_center[1]+12*(n-k)))
                        for k in range(n-2,n+1)]
    return obj


def test_fresh_unstruck_toss_can_checkpoint_before_first_bounce():
    obj=toss_owner();rewind=ShortBallRewind()
    rewind.remember(obj,102)
    obj=toss_owner(104)
    rewind.remember(obj,103)
    assert len(rewind.checkpoints)==1 and rewind.checkpoints[0].frame==104
    assert obj._point_hit_count==0 and obj.score==[0,0]


@pytest.mark.parametrize('change',['far','night','launched','tiny','gap','lateral_jump','rally'])
def test_toss_checkpoint_does_not_trust_other_phases_or_fragments(change):
    obj=toss_owner()
    if change=='far':obj._active_serve_area_end='far'
    elif change=='night':obj.far_serve_image_baseline=False
    elif change=='launched':obj._serve_launch_direction_x=1
    elif change=='tiny':obj.ball_size=4
    elif change=='gap':obj.motion_history[0]['frame']-=1
    elif change=='lateral_jump':obj.motion_history[0]['pos']=(0,489)
    else:obj._point_hit_count=1
    assert not prestrike_toss_checkpoint(obj)


def test_tiny_fragment_above_racket_reach_requests_rewind_despite_ai_label():
    obj=toss_owner();rewind=ShortBallRewind()
    rewind.remember(obj,102)
    obj=toss_owner(104);rewind.remember(obj,103)
    old=obj.ball_center
    obj.frame_count=105;obj.ball_center=(80,420);obj.ball_size=4
    obj._configured_causal_evidence=dict(frame=105)
    assert rewind.problem(obj,old,obj.ball_center,0)=='prestrike-toss-fragment'
    obj.local_ai_recovery=SimpleNamespace()
    rewind.try_recover(obj,None,'prestrike-toss-fragment',104)
    assert rewind.hard_uncertainty and rewind.prestrike_uncertainty
    rewind.remember(obj,104)
    assert rewind.hard_uncertainty  # No silent clearance from an empty hit list.
    assert obj.score==[0,0]


def test_measured_departure_below_net_does_not_trigger_toss_fragment_replay():
    obj=toss_owner();rewind=ShortBallRewind()
    rewind.remember(obj,102)
    obj=toss_owner(104);obj.ball_center=(104,880)
    for s in obj.motion_history:s['pos']=(s['pos'][0],s['pos'][1]+427)
    rewind.remember(obj,103)
    obj.frame_count=105;obj._serve_launch_direction_x=1;obj.ball_size=35
    obj.ball_center=(155,640);obj._configured_causal_evidence=dict(frame=105)
    assert rewind.problem(obj,(104,880),obj.ball_center,0) is None


def replay_data(obj):
    checkpoint=Checkpoint(1,0,1,(10,80),(10,0),20,snapshot(obj))
    # Existing faulty events must be replaced, not incremented again.
    obj._point_hit_count=7
    obj.ground_bounce_count=8
    obj.player_tracker.profile['hits']=7
    obj._point_history_current['tracking_trace'].append(dict(frame=5,pos=[999,999]))
    frames=list(range(2,7))
    images=[np.zeros((120,200,3),np.uint8) for _ in frames]
    ledger={n:dict(tracks={}) for n in frames}
    path=[candidate(n,n*10) for n in frames]
    return checkpoint,path,images,frames,ledger,np.zeros((120,200,3),np.uint8)


def test_accepted_rewind_rebuilds_counts_and_trace_without_duplicating_events():
    obj=owner(); data=replay_data(obj)
    rewind=ShortBallRewind()
    assert rewind.apply(obj,*data)==(60,80)
    assert obj._point_hit_count==2 and obj.ground_bounce_count==2
    assert obj.player_tracker.profile==dict(hits=2)
    assert obj.player_tracker.profile_path=='profile.json'
    assert [s['frame'] for s in obj._point_history_current['tracking_trace']]==[1,2,3,4,5]
    assert obj.score==[0,0] and not obj._short_rewind_replaying


def test_verified_replay_clears_prestrike_uncertainty_without_awarding_points():
    obj=owner();data=replay_data(obj);rewind=ShortBallRewind()
    rewind.hard_uncertainty=rewind.prestrike_uncertainty=rewind.unresolved=True
    rewind.hard_since=rewind.unresolved_since=2
    assert rewind.apply(obj,*data)==(60,80)
    assert obj.score==[0,0]
    assert not rewind.hard_uncertainty and not rewind.prestrike_uncertainty
    assert not rewind.unresolved


def test_failed_replay_restores_every_event_counter_and_player_profile():
    obj=owner();data=replay_data(obj); old_trace=list(obj._point_history_current['tracking_trace'])
    old_commit=obj._commit_local_ball_flight
    def failing(c,image):
        old_commit(c,image)
        if obj.frame_count==5:
            obj.score[0]=99
            obj.new_bad_field=True
            obj.player_tracker.new_bad_event=True
            raise RuntimeError('injected failure')
    obj._commit_local_ball_flight=failing
    with pytest.raises(RuntimeError,match='injected failure'):
        ShortBallRewind().apply(obj,*data)
    assert obj._point_hit_count==7 and obj.ground_bounce_count==8
    assert obj.player_tracker.profile==dict(hits=7)
    assert obj.player_tracker.profile_path=='profile.json'
    assert obj._point_history_current['tracking_trace']==old_trace
    assert obj.score==[0,0] and not hasattr(obj,'new_bad_field')
    assert not hasattr(obj.player_tracker,'new_bad_event')
    assert not obj._short_rewind_replaying


def test_scoring_and_csv_commit_are_forbidden_during_replay():
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj._short_rewind_replaying=True
    with pytest.raises(ReplayRejected,match='scoring-during-replay'):
        obj._record_point_result('Ball hit the net')


def test_checkpoint_copies_mutable_state_but_preserves_external_handles():
    obj=owner(); handle=obj.external_handle; saved=snapshot(obj)
    obj.score[0]=4
    obj._point_history_current['tracking_trace'][0]['pos'][0]=999
    restore(obj,saved)
    assert obj.score==[0,0]
    assert obj._point_history_current['tracking_trace'][0]['pos']==[10,80]
    assert obj.external_handle is handle


def test_checkpoints_and_retries_do_not_cross_point_boundaries():
    obj=owner();rewind=ShortBallRewind()
    for n in range(1,15):
        obj.frame_count=n;obj._last_tracked_candidate_motion_frame=n
        rewind.remember(obj,n+100)
    assert len(rewind.checkpoints)==13
    assert rewind.choose(15).frame==3
    rewind.attempts=1
    assert rewind.choose(15).frame==2
    obj.point_start_frame_internal=15
    obj.frame_count=15
    obj._last_tracked_candidate_motion_frame=15
    rewind.remember(obj,115)
    assert rewind.point==15 and len(rewind.checkpoints)==0
    assert rewind.attempts==0 and rewind.choose(16) is None


def test_no_checkpoint_does_not_freeze_future_trusted_observations():
    obj=owner();obj.local_ai_recovery=SimpleNamespace()
    rewind=ShortBallRewind();rewind.point=1
    assert rewind.try_recover(obj,None,'ball-miss',0) is None
    assert rewind.first_problem is None and rewind.attempts==0


def test_intentional_occlusion_wait_does_not_trigger_rewind():
    obj=owner();obj.stuck_frame_count=4;obj._far_contact_pending=dict(frame=1)
    rewind=ShortBallRewind();rewind.point=1
    assert rewind.problem(obj,(10,80),(10,80),4) is None


def test_recent_verified_contact_does_not_rewind_its_successful_recovery():
    obj=owner();obj.stuck_frame_count=3;obj._temporal_near_contact_frame=obj.frame_count
    rewind=ShortBallRewind();rewind.point=1
    assert rewind.problem(obj,(10,80),(10,80),3) is None


def test_body_occlusion_can_span_more_images_without_inventing_positions():
    groups=[(n,[] if 3<=n<=6 else [candidate(n,10*n)]) for n in range(1,11)]
    assert best_path(groups,0,(0,80),(10,0)) is None
    path=best_path(groups,0,(0,80),(10,0),occlusion_frames={3,4,5,6})
    assert [c['frame'] for c in path]==[1,2,7,8,9,10]


def test_existing_trusted_history_cannot_substitute_for_fresh_ai_confirmation():
    groups=[(n,[dict(candidate(n,10*n),trusted_prefix=True)]) for n in range(1,7)]
    assert best_path(groups,0,(0,80),(10,0)) is None
    for _,cs in groups[-3:]:cs[0]['trusted_prefix']=False
    assert best_path(groups,0,(0,80),(10,0)) is not None


def test_repeated_same_player_hit_is_suspicious_even_if_hsv_claims_strong_evidence():
    obj=owner();obj.frame_count=100;obj._configured_causal_evidence=dict(frame=100)
    obj._point_history_current['shot_events']=[dict(frame=40,player='P1'),dict(frame=100,player='P1')]
    rewind=ShortBallRewind();rewind.point=1
    assert rewind.problem(obj,(10,80),(20,80),0)=='repeated-same-player-hit'


def test_hard_unresolved_failure_cannot_be_hidden_by_lack_of_checkpoint():
    obj=owner();obj.local_ai_recovery=SimpleNamespace()
    rewind=ShortBallRewind();rewind.point=1
    assert rewind.try_recover(obj,None,'repeated-same-player-hit',0) is None
    assert rewind.unresolved and rewind.hard_uncertainty


def test_accepted_replay_repairs_the_original_short_ai_buffer_too():
    obj=owner();data=replay_data(obj)
    obj._local_ai_frame_buffer=[dict(frame=n,normal_position=(999,999)) for n in range(2,7)]
    ShortBallRewind().apply(obj,*data)
    assert [s['normal_position'] for s in obj._local_ai_frame_buffer]==[(20,80),(30,80),(40,80),(50,80),(60,80)]


def test_replay_inference_does_not_steal_normal_recovery_budget(monkeypatch,tmp_path):
    from ball_local_ai_recovery import LocalBallAIRecovery
    from local_ai_budget import InferenceBudget
    recovery=LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    recovery.work_dir=tmp_path
    recovery.python_executable='python'
    recovery.model_path=tmp_path/'model.pt'
    recovery._sequence=0
    recovery._record_inference=lambda *a:None
    assert recovery._begin_inference(100)
    normal=list(recovery._inference_budget.frames)
    monkeypatch.setattr('ball_local_ai_recovery.subprocess.run',
        lambda *a,**kw:SimpleNamespace(returncode=0,stdout='[[]]'))
    separate=InferenceBudget()
    assert recovery._score_batch([(np.zeros((20,20,3),np.uint8),101,[])],inference_budget=separate)==[[]]
    assert list(recovery._inference_budget.frames)==normal
    assert list(separate.frames)==[101]


def test_unresolved_replay_records_review_without_awarding_or_duplicating_point(monkeypatch,tmp_path):
    import csv
    from unittest.mock import MagicMock
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture',lambda _:MagicMock())
    obj=InteractiveBallAnalyzer('unused',enable_player_tracking=False,enable_player_learning=False,
                               point_history_file=str(tmp_path/'points.csv'))
    obj.frame_count=obj.point_start_frame_internal=10
    obj._start_point_context((1500,1000),serve_start_frame=10)
    obj.frame_count=100
    obj._short_rewind=SimpleNamespace(point=10,unresolved=True,hard_uncertainty=True,review_points=[])
    before=obj._score_summary()
    obj._record_point_result('Ball bounced out of court (right sideline)',end_position=(2500,800))
    obj._record_point_result('Ball bounced out of court (right sideline)',end_position=(2500,800))
    assert obj._score_summary()==before
    assert obj._last_point_outcome_category=='tracking_uncertain'
    rows=list(csv.DictReader(open(obj.point_history_file,encoding='utf-8-sig')))
    assert len(rows)==1 and rows[0]['point_awarded']=='no'
    assert len(obj._short_rewind.review_points)==1


def test_late_replay_failure_also_restores_short_buffer_and_acceptance_markers():
    obj=owner();data=replay_data(obj)
    obj._local_ai_frame_buffer=[dict(frame=6,normal_position=(999,999))]
    obj._short_rewind_accepted_frame=0
    rewind=ShortBallRewind()
    def fail(*a,**kw):raise RuntimeError('late failure')
    obj._record_ball_loss_event=fail
    with pytest.raises(RuntimeError,match='late failure'):
        rewind.apply(obj,*data)
    assert obj._local_ai_frame_buffer[0]['normal_position']==(999,999)
    assert obj._short_rewind_accepted_frame==0 and rewind.accepted==0
    assert obj._point_hit_count==7 and obj.ground_bounce_count==8


def test_position_recovery_keeps_score_provisional_when_contact_history_is_incomplete():
    obj=owner()
    obj._last_racket_contact_player=0
    obj._last_racket_contact_frame=0
    obj._side_recross_opponent_frames=3
    obj._player_index_at_point=lambda p,image:0
    def rejected(*a,**kw):
        obj._side_recross_opponent_frames=0
        return False
    obj._maybe_infer_return_contact_from_side_recross=rejected
    data=replay_data(obj)
    rewind=ShortBallRewind();rewind.attempts=1;rewind.first_problem=6
    rewind.checkpoints.append(data[0])
    assert rewind.apply(obj,*data)==(60,80)
    assert rewind.hard_uncertainty and rewind.contact_uncertainty and rewind.unresolved
    assert rewind.attempts==1 and rewind.first_problem==6
    assert len(rewind.checkpoints)==1  # Retain the older state for the wider retry.


def test_retry_window_expands_and_stops_after_two_failed_attempts(monkeypatch):
    class Decoder:
        def set(self,key,value):self.pos=value
        def get(self,key):return self.pos
        def read(self):
            self.pos+=1
            return True,np.zeros((120,200,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr('short_ball_rewind.cv2.VideoCapture',lambda path:Decoder())
    obj=owner();obj.video_path='unused'
    calls=[]
    def score(samples,**kw):
        calls.append(len(samples))
        return [[] for _ in samples]
    obj.local_ai_recovery=SimpleNamespace(minimum_score=.985,_score_batch=score)
    rewind=ShortBallRewind();rewind.point=1
    rewind.checkpoints.append(Checkpoint(1,0,1,(10,80),(10,0),20,snapshot(obj)))
    rewind.observations.extend(dict(frame=n,source=n-1,pos=(10,80),tracks={}) for n in range(2,24))
    obj.frame_count=15
    assert rewind.try_recover(obj,None,'ball-miss',14) is None
    obj.frame_count=16
    assert rewind.try_recover(obj,None,'ball-miss',15) is None
    assert rewind.attempts==1
    obj.frame_count=23
    assert rewind.try_recover(obj,None,'ball-miss',22) is None
    obj.frame_count=40
    assert rewind.try_recover(obj,None,'ball-miss',39) is None
    assert rewind.attempts==2 and rewind.unresolved
    assert calls==[14,22]
    rewind.close()


def test_ball_replay_preserves_the_live_player_detectors_clock_and_positions():
    obj=owner();obj.player_tracker.last_frame=1;obj.player_tracker._last_detection_frame=1
    data=replay_data(obj)
    obj.player_tracker.last_frame=6
    obj.player_tracker._last_detection_frame=5
    obj.player_tracker.point_positions=[dict(frame=6)]
    ShortBallRewind().apply(obj,*data)
    assert obj.player_tracker.last_frame==6 and obj.player_tracker._last_detection_frame==5
    assert obj.player_tracker.point_positions==[dict(frame=6)]
    assert obj.player_tracker.profile['hits']==2


def test_manual_configuration_change_invalidates_old_checkpoints():
    obj=owner();rewind=ShortBallRewind()
    rewind.point=1
    rewind.checkpoints.append(Checkpoint(1,0,1,(10,80),(10,0),20,snapshot(obj)))
    rewind.first_problem=4
    rewind.attempts=1
    rewind.invalidate()
    assert not rewind.checkpoints and not rewind.observations
    assert rewind.first_problem is None and rewind.attempts==0


@pytest.mark.parametrize('saturation',[26,32,44])
@pytest.mark.parametrize('zone',[None,'player_body'])
@pytest.mark.parametrize('fresh',[True,False])
def test_distant_return_keeps_weak_yellow_ball_between_grey_and_yellow_filters(monkeypatch,tmp_path,saturation,zone,fresh):
    from unittest.mock import MagicMock
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture',lambda _:MagicMock())
    obj=InteractiveBallAnalyzer('unused',enable_player_tracking=False,enable_player_learning=False,
                               point_history_file=str(tmp_path/'points.csv'))
    before=np.full((300,400,3),40,np.uint8)
    after=before.copy()
    colour=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[32,saturation,220]]]),cv2.COLOR_HSV2BGR)[0,0])
    cv2.circle(after,(188,110),3,colour,-1)
    obj.frame_count=20;obj.point_start_frame_internal=1
    obj.ball_center=(200,100);obj.ball_size=7.5 if zone else 20
    obj.last_motion=dict(dx=-12,dy=10,distance=15.6)
    obj.net_area_y_min=150;obj.net_area_y_max=180
    obj.tracking_hsv_ranges=[dict(h_min=79,h_max=92,s_min=24,s_max=54,v_min=77,v_max=135)]
    obj._terminal_previous_gray=cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    obj._visible_near_return_active=lambda:True
    obj._visible_extended_near_return_active=lambda:True
    obj._visible_far_return_active=lambda:False
    obj._player_point_zone=lambda p:zone
    obj._slow_serve_net_drop_active=lambda:False
    obj._causal_flight_anchor_frame=19
    obj._last_tracked_candidate_motion_frame=19 if fresh else 18
    obj._last_tracked_candidate_motion_mean=10
    obj._last_tracked_candidate_motion_max=100
    obj._configured_flight_area_reference=dict(point_start=1,frame=19,area=20)
    found=obj._recover_local_ball_flight(after)
    if zone and not fresh:
        assert found is None
    else:
        assert found is not None and np.linalg.norm(np.array(found['pos'])-(188,110))<=2
    obj._terminal_previous_gray=cv2.cvtColor(after,cv2.COLOR_BGR2GRAY)
    assert obj._recover_local_ball_flight(after) is None


def test_slow_far_anchor_keeps_waiting_for_verified_departure_but_still_expires():
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.point_start_frame_internal=80;obj.frame_count=145
    obj.ball_center=(50,50);obj.stuck_frame_count=4
    obj._terminal_previous_gray=np.zeros((200,200),np.uint8)
    obj._player_point_zone=lambda p:None
    obj._visible_near_return_active=lambda:False
    obj._far_contact_pending=dict(frame=100,point_start=80,anchor=(50,50),center=(50,50),
                                  bounds=(0,0,200,200),samples=[],weak_anchor=True)
    frame=np.zeros((200,200,3),np.uint8)
    assert obj._recover_far_player_contact(frame)==(50,50)
    assert obj._far_contact_pending is not None
    obj.frame_count=161
    assert obj._recover_far_player_contact(frame) is None
    assert obj._far_contact_pending is None


@pytest.mark.parametrize('score,source,expected',[(.7,'hog',True),(-.3,'hog',False),(2.,'motion',False)])
def test_near_player_can_reacquire_toward_net_only_with_strong_person_evidence(score,source,expected):
    from player_racket_tracker import PlayerRacketTracker, PlayerTrack
    tracker=PlayerRacketTracker(frame_shape=(2160,3840),net_y=550,learning_enabled=False)
    track=PlayerTrack('near',bbox=(2400,1400,250,500),center=(2525,1650),last_frame=10)
    candidate=dict(bbox=(2100,580,250,500),score=score,source=source)
    frame=np.zeros((2160,3840,3),np.uint8)
    assert tracker._candidate_is_full_player_reacquisition(candidate,track,frame,'near',100)==expected
    candidate['bbox']=(2100,0,250,500)
    assert not tracker._candidate_is_full_player_reacquisition(candidate,track,frame,'near',100)


@pytest.mark.parametrize('fresh',[True,False])
def test_fresh_upper_player_reach_can_verify_a_volley_without_prior_ground_bounce(monkeypatch,tmp_path,fresh):
    from unittest.mock import MagicMock
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture',lambda _:MagicMock())
    obj=InteractiveBallAnalyzer('unused',enable_player_tracking=False,enable_player_learning=False,
                               point_history_file=str(tmp_path/'points.csv'))
    before=np.full((1200,3000,3),40,np.uint8);after=before.copy()
    cv2.circle(after,(2370,600),7,(0,240,240),-1)
    obj.frame_count=100;obj.point_start_frame_internal=1
    obj.ball_center=(2440,700);obj.ball_size=140
    obj.last_motion=dict(dx=20,dy=40,distance=44.7)
    obj.net_area_y_min=450;obj.net_area_y_max=550
    obj.tracking_hsv_ranges=[dict(h_min=79,h_max=92,s_min=24,s_max=54,v_min=77,v_max=135)]
    obj._terminal_previous_gray=cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    obj._visible_near_return_active=lambda:False
    obj._visible_extended_near_return_active=lambda:False
    obj._visible_far_return_active=lambda:True
    obj._player_point_zone=lambda p:None
    obj._slow_serve_net_drop_active=lambda:False
    obj.player_tracker=SimpleNamespace(tracks={'near':SimpleNamespace(
        bbox=(2100,500,250,500),last_frame=99 if fresh else 80)})
    found=obj._recover_local_ball_flight(after)
    assert (found is not None and found['verified_contact_departure'])==fresh


@pytest.mark.parametrize('gap,eligible',[(1,True),(3,False),(4,False)])
def test_far_contact_search_requires_consecutive_rebound_evidence_without_awarding_hit(gap,eligible):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=100;obj.point_start_frame_internal=50
    obj.ball_center=(200,150);obj.ball_size=20
    obj.last_motion=dict(dx=-10,dy=-20)
    obj.ground_bounce_count=0;obj.last_ground_bounce_frame=-1000000
    obj._last_racket_contact_frame=90
    obj.motion_history=[dict(frame=99-2*gap,pos=(200,152)),
                        dict(frame=99-gap,pos=(200,158)),dict(frame=99,pos=(200,150))]
    obj.net_area_y_min=240;obj.net_area_y_max=260
    obj.tracking_hsv_ranges=[{}];obj.stuck_frame_count=0
    obj._causal_flight_anchor_frame=99
    obj._visible_near_return_active=lambda:True
    obj._player_point_zone=lambda p:'player_body'
    obj._terminal_previous_gray=np.zeros((300,400),np.uint8)
    obj.player_tracker=SimpleNamespace(tracks={'far':SimpleNamespace(
        bbox=(180,100,50,100),last_frame=99,confidence=.8)})
    assert (obj._recover_far_player_contact(np.zeros((300,400,3),np.uint8)) is not None)==eligible
    assert obj.ground_bounce_count==0 and obj._last_racket_contact_frame==90


@pytest.mark.parametrize('invalid',[None,'out_serve','no_landing','far_serve','stale','later_hit'])
def test_verified_in_serve_can_reach_receiver_with_short_warm_player_box(invalid):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=100;obj.point_start_frame_internal=50
    obj.far_serve_image_baseline=True
    obj._active_serve_area_end='far' if invalid=='far_serve' else 'near'
    obj._awaiting_serve_bounce=False
    obj._serve_landed_in_current_attempt=invalid!='no_landing'
    obj._last_serve_bounce_was_in=invalid!='out_serve'
    obj._point_hit_count=2 if invalid=='later_hit' else 1
    obj.ball_center=(200,260);obj.ball_size=18
    obj.last_motion=dict(dx=-7,dy=-16)
    obj.ground_bounce_count=1;obj.last_ground_bounce_frame=98
    obj._last_racket_contact_frame=90
    obj.motion_history=[]
    obj.net_area_y_min=300;obj.net_area_y_max=320
    obj.tracking_hsv_ranges=[{}];obj.stuck_frame_count=0
    obj._causal_flight_anchor_frame=98 if invalid=='stale' else 99
    obj._visible_near_return_active=lambda:True
    obj._player_point_zone=lambda p:None
    obj._terminal_previous_gray=np.zeros((400,500),np.uint8)
    obj.player_tracker=SimpleNamespace(tracks={'far':SimpleNamespace(
        bbox=(180,100,50,100),last_frame=99,confidence=.6)})
    result=obj._recover_far_player_contact(np.zeros((400,500,3),np.uint8))
    assert (result is not None)==(invalid is None)
    assert obj._point_hit_count==(2 if invalid=='later_hit' else 1)
    assert obj.last_ground_bounce_frame==98


@pytest.mark.parametrize('invalid',[None,'gap','far_serve','uncalibrated','stale','wide','not_slowing'])
def test_slow_far_approach_holds_for_measured_departure_without_inventing_hit(invalid):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=100;obj.point_start_frame_internal=50
    obj.far_serve_image_baseline=invalid!='uncalibrated'
    obj._active_serve_area_end='far' if invalid=='far_serve' else 'near'
    obj._awaiting_serve_bounce=False
    obj.ball_center=(210,226);obj.ball_size=42
    obj.last_motion=dict(dx=-2,dy=-4)
    obj.ground_bounce_count=0;obj.last_ground_bounce_frame=-1000000
    obj._last_racket_contact_frame=90
    obj.motion_history=[dict(frame=96,pos=(218,245)),dict(frame=97,pos=(215,237)),
                        dict(frame=98,pos=(212,230)),dict(frame=99,pos=(210,226))]
    if invalid=='gap': obj.motion_history[1]['frame']=95
    if invalid=='not_slowing': obj.motion_history[-1]['pos']=(210,223)
    obj.net_area_y_min=240;obj.net_area_y_max=260
    obj.tracking_hsv_ranges=[{}];obj.stuck_frame_count=0
    obj._causal_flight_anchor_frame=98 if invalid=='stale' else 99
    obj._visible_near_return_active=lambda:True
    obj._player_point_zone=lambda p:None
    obj._terminal_previous_gray=np.zeros((300,400),np.uint8)
    obj.player_tracker=SimpleNamespace(tracks={'far':SimpleNamespace(
        bbox=(0 if invalid=='wide' else 130,100,50,100),last_frame=99,confidence=.8)})
    result=obj._recover_far_player_contact(np.zeros((300,400,3),np.uint8))
    assert (result is not None)==(invalid is None)
    assert obj.ground_bounce_count==0 and obj._last_racket_contact_frame==90
    if invalid is None:
        assert obj._far_contact_pending['approach_motion']==dict(dx=-3,dy=-8)
        assert obj._far_contact_pending['bounds']==(0,0,400,290)


@pytest.mark.parametrize('invalid',[None,'player','gap','apex','reverse_x','no_descent','new_hit'])
def test_shallow_rebound_needs_descent_two_flat_images_and_clear_continuous_motion(invalid):
    from causal_ball_recovery import shallow_rebound
    history=[dict(frame=10,prev_pos=(200,100),pos=(150,162)),
             dict(frame=11,prev_pos=(150,162),pos=(113,159)),
             dict(frame=12,prev_pos=(113,159),pos=(75,158))]
    corridor=lambda p:invalid=='player'
    if invalid=='gap':history[0]['frame']=8
    if invalid=='apex':history[0]['prev_pos']=(200,220)
    if invalid=='reverse_x':history[2]['pos']=(160,158)
    if invalid=='no_descent':history[0]['prev_pos']=(200,160)
    result=shallow_rebound(history,11 if invalid=='new_hit' else 1,corridor)
    assert (result is not None)==(invalid is None)
    if result:assert result['origin']==(150,162) and result['frame']==10


@pytest.mark.parametrize('stale',['none','point','contact'])
def test_sideline_exit_after_verified_shallow_legal_landing_is_unreturned_only_for_same_shot(stale):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=100;obj.point_start_frame_internal=10
    obj.ball_center=(50,500);obj.player_names=['P1','P2']
    obj._current_server_index=lambda:0
    obj._recent_contact_info=lambda **kw:(True,0,50,(200,100))
    obj._point_contact_info=lambda **kw:(True,0,50,(200,100))
    obj._player_index_at_point=lambda *a:1
    obj._is_lower_right_exit_score_context=lambda *a:False
    obj._measured_shallow_legal_bounce=dict(point_start=9 if stale=='point' else 10,
        contact_frame=49 if stale=='contact' else 50,frame=70,position=(200,400))
    result=obj._infer_point_outcome('Ball bounced out of court (left sideline)')
    assert result['winner_idx']==(0 if stale=='none' else 1)
    assert result['category']==('unreturned' if stale=='none' else 'out_error')


@pytest.mark.parametrize('near_player',[False,True])
def test_causal_committer_records_shallow_legal_bounce_once_away_from_player(near_player):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=12;obj.point_start_frame_internal=1
    obj.ball_center=(113,159);obj.ball_size=95;obj.net_area_y_max=100
    obj.last_motion=dict(dx=-37,dy=-3,distance=37.1,direction_deg=-175)
    obj.motion_history=[dict(frame=10,prev_pos=(200,100),pos=(150,162)),
                        dict(frame=11,prev_pos=(150,162),pos=(113,159))]
    obj.tracking_hsv_ranges=[{}]
    obj._last_racket_contact_frame=5;obj.last_ground_bounce_frame=-1000000
    obj._get_ground_bounce_context=lambda *a,**kw:None
    obj._serve_bounce_context_live=lambda:False
    obj._player_point_zone=lambda p:None
    obj._point_in_player_contact_corridor=lambda p:near_player
    obj._classify_ground_bounce=lambda *a:(True,'in',(0,255,0))
    records=[]
    def register(context,*args):
        records.append(context);obj.last_ground_bounce_frame=obj.frame_count;return True
    obj._register_ground_bounce_from_context=register
    def commit(candidate,frame):
        obj.ball_center=candidate['pos']
        obj.last_motion=dict(dx=-38,dy=-1,distance=38.,direction_deg=-178)
        return obj.ball_center
    obj._commit_night_visible_ball_recovery=commit
    candidate=dict(pos=(75,158),area=95,motion_mean=35,motion_max=124)
    image=np.zeros((300,400,3),np.uint8)
    obj._commit_local_ball_flight(candidate,image)
    assert len(records)==(0 if near_player else 1)
    if not near_player:
        assert obj._measured_shallow_legal_bounce==dict(point_start=1,contact_frame=5,frame=10,position=(150,162))
        obj._commit_local_ball_flight(candidate,image)
        assert len(records)==1


@pytest.mark.parametrize('incoming',['fresh','static','absent'])
def test_visible_incoming_ball_beats_speculative_player_departure(incoming):
    from causal_ball_recovery import find_local_ball
    before=np.full((600,800,3),30,np.uint8);after=before.copy()
    if incoming!='absent':cv2.circle(after,(412,440),5,(0,240,240),-1)
    if incoming=='static':cv2.circle(before,(412,440),5,(0,240,240),-1)
    cv2.circle(after,(390,300),8,(0,240,240),-1)
    found=find_local_ball(after,cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),(400,400),
        dict(dx=12,dy=40),90,[(np.array([20,50,130]),np.array([85,255,255]))],
        contact_window=True,flexible_contact=True,maximum_speed=400,player_zone=lambda p:None)
    assert found is not None
    assert found['verified_contact_departure']==(incoming!='fresh')
    assert np.linalg.norm(np.array(found['pos'])-((412,440) if incoming=='fresh' else (390,300)))<=2


@pytest.mark.parametrize('kind',['spike','sustained_hit','bounce'])
def test_rewind_rejects_isolated_acceleration_but_keeps_sustained_hits_and_rebounds(kind):
    ys=([40,80,120,160,300,310,250] if kind=='spike' else
        [40,80,120,160,300,430,550] if kind=='sustained_hit' else
        [140,280,420,560,700,650,605])
    groups=[(i,[candidate(i,80,y)]) for i,y in enumerate(ys,1)]
    found=best_path(groups,0,(80,0),(0,140 if kind=='bounce' else 40))
    assert (found is not None)==(kind!='spike')
