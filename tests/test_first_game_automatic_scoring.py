import csv
from unittest.mock import MagicMock

import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.fixture
def tracker(monkeypatch,tmp_path):
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture',lambda _:MagicMock())
    obj=InteractiveBallAnalyzer('unused',enable_player_tracking=False,
        enable_player_learning=False,point_history_file=str(tmp_path/'points.csv'))
    obj.net_area_y_min,obj.net_area_y_max=400,585
    obj._player_serve_context=lambda point: {
        'player_position':[1500,1500],'player_side':'near','player_source':'server',
        'serve_x':point[0],'serve_y':point[1]}
    return obj


@pytest.mark.parametrize('offset',[0,11000])
def test_fault_sequence_then_rallies_complete_first_game(tracker,offset):
    # Physical outcomes, rather than stored frame endpoints, drive the scorer.
    outcomes=[('Ball stopped on player side',1),
              ('Serve bounce outside singles court (right sideline)',None),
              ('Ball hit the net',None),
              ('Ball bounce outside singles court (far baseline)',1),
              ('Ball bounce outside singles court (left sideline)',0),
              ('Ball bounce outside singles court (far baseline)',1)]
    observed=[]
    for index,(reason,winner) in enumerate(outcomes):
        tracker.frame_count=offset+100+index*300
        tracker.point_start_frame_internal=tracker.frame_count
        tracker._start_point_context((1800,1000),serve_start_frame=tracker.frame_count)
        tracker.frame_count+=40
        if winner is not None:
            tracker._point_hit_count=3
            tracker._last_racket_contact_frame=tracker.frame_count-5
            tracker._last_racket_contact_player=winner if index==0 else 1-winner
            tracker._last_racket_contact_point=(1800,900 if winner==1 else 200)
        tracker._record_point_result(reason,(1800,900 if index==0 else 200))
        observed.append(tracker._score_summary())
    assert observed==['0:0 0:15','0:0 0:15','0:0 0:30',
                      '0:0 0:40','0:0 15:40','0:1 0:0']
    assert tracker.score_game_index==1
    assert tracker._current_server_index()==1
    with open(tracker.point_history_file,newline='') as handle:
        rows=list(csv.DictReader(handle))
    assert [row['serve_attempt'] for row in rows]==['1st','1st','2nd','1st','1st','1st']
    assert rows[2]['category']=='double_fault'


def test_local_recovery_runs_in_tracking_before_alternate_search(tracker):
    frame=np.zeros((240,320,3),np.uint8)
    tracker.tracking=True
    tracker.hsv_lower=np.array([20,50,130],np.uint8)
    tracker.frame_count=150
    tracker._causal_flight_until_frame=152
    tracker._recover_local_ball_flight=lambda image:{'pos':(112,116)}
    calls=[]
    tracker._commit_local_ball_flight=lambda candidate,image:(calls.append(candidate) or candidate['pos'])
    assert tracker.track_ball_in_frame(frame)==(112,116)
    assert calls==[{'pos':(112,116)}]


def test_expired_serve_flag_cannot_hide_far_baseline_bounce(tracker):
    tracker.frame_count=250
    tracker.point_start_frame_internal=100
    tracker._awaiting_serve_bounce=True
    tracker._serve_phase_active=True
    tracker._point_outside_singles_sidelines=lambda *args:(False,None,100,300)
    tracker._is_night_session_config=lambda:True
    tracker._point_outside_top_singles_baseline=lambda *args:(True,200)
    frame=np.zeros((2160,3840,3),np.uint8)
    inside,reason,_=tracker._classify_ground_bounce((180,160),frame)
    assert not inside
    assert reason=='Ball bounce outside singles court (far baseline)'
    assert not tracker._awaiting_serve_bounce


def test_recovery_history_remains_compatible_with_normal_motion_logging(tracker):
    tracker.frame_count=150
    tracker.ball_center=(100,100)
    tracker.ball_size=66
    tracker.last_motion={'dx':12,'dy':16,'distance':20,'direction_deg':53.13}
    tracker._causal_flight_anchor_frame=148
    frame=np.zeros((240,320,3),np.uint8)
    candidate={'pos':(124,132),'area':66,'hsv':np.array([30,180,210],np.uint8),
               'motion_mean':20,'motion_max':120}
    assert tracker._commit_local_ball_flight(candidate,frame)==(124,132)
    assert tracker.last_motion['distance']==20
    assert tracker.motion_history[-1]['distance']==20
    assert tracker.motion_history[-1]['prev_pos']==(100,100)
    tracker.frame_count+=1
    tracker.ball_center=(136,148)
    tracker.log_motion_metrics((124,132),12,16,20,53.13)
    assert tracker.motion_history[-1]['pos']==(136,148)


def test_normal_tracking_between_recoveries_preserves_measured_speed(tracker):
    tracker.frame_count=152
    tracker.ball_center=(124,132)
    tracker.ball_size=66
    tracker.last_motion={'dx':12,'dy':16,'distance':20,'direction_deg':53.13}
    tracker._causal_flight_anchor_frame=150
    tracker.motion_history=[{'frame':151,'pos':(124,132),'prev_pos':(112,116),
                            'distance':20,'direction_deg':53.13}]
    candidate={'pos':(136,148),'area':66,'hsv':np.array([30,180,210],np.uint8),
               'motion_mean':20,'motion_max':120}
    tracker._commit_local_ball_flight(candidate,np.zeros((240,320,3),np.uint8))
    assert tracker.last_motion['distance']==20


@pytest.mark.parametrize('strong_motion',[False,True])
def test_visible_lower_contact_flight_arms_far_exit_without_ai(tracker,strong_motion):
    tracker.frame_count=250
    tracker.ball_center=(1500,1500)
    tracker._player_reacq_protect_until_frame=300
    tracker._player_point_zone=lambda point:'player_body' if point==(1500,1500) else None
    tracker._last_racket_contact_frame=100
    tracker._last_racket_contact_player=1
    candidate={'pos':(1600,350),'area':300,'hsv':np.array([30,180,210],np.uint8),
               'motion_mean':60 if strong_motion else 2,'motion_max':150 if strong_motion else 8}
    tracker._commit_night_visible_ball_recovery(candidate,np.zeros((2160,3840,3),np.uint8))
    assert (tracker._contact_visible_recovery_exit is not None)==strong_motion
