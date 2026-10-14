from types import SimpleNamespace
from interactive_ball_analyzer import InteractiveBallAnalyzer


def test_impossible_ai_handoff_preserves_already_committed_hsv_state():
    analyzer=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count=100
    analyzer.ball_center=(450,500)
    analyzer.ball_size=240
    analyzer.stuck_frame_count=0
    analyzer.max_ball_speed=400
    analyzer.last_motion={'dx':50,'dy':0,'distance':50}
    analyzer.ball_velocity_history=[40,50]
    analyzer.motion_history=[{'frame':100,'pos':(450,500)}]
    analyzer._local_ai_frame_buffer=[]
    analyzer._finalize_pending_provisional_static_candidate=lambda:False
    analyzer._normal_ball_evidence_is_strong=lambda *args:False
    analyzer._automatic_local_ai_scoring_ready=lambda:True
    analyzer._try_active_tight_local_ai_hold=lambda *args,**kw:None
    analyzer._local_ai_recovery_reason=lambda *args:'player-region:player_body'
    analyzer._player_point_zone=lambda point:None
    analyzer.local_ai_recovery=SimpleNamespace(
        recover=lambda *args,**kw:{'x':900,'y':500,'area':9},
        ready=lambda frame:True,
    )
    result=analyzer._try_local_ai_recovery((400,500),(450,500),0)
    assert result==(450,500)
    assert analyzer.ball_center==(450,500)
    assert analyzer.ball_size==240
    assert analyzer.last_motion=={'dx':50,'dy':0,'distance':50}
    assert analyzer.ball_velocity_history==[40,50]
    assert analyzer.motion_history==[{'frame':100,'pos':(450,500)}]
    assert analyzer.stuck_frame_count==0
