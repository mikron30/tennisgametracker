import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from local_ai_budget import InferenceBudget
from ball_local_ai_recovery import LocalBallAIRecovery
from interactive_ball_analyzer import InteractiveBallAnalyzer


def test_retries_are_bounded_across_all_recovery_paths():
    budget = InferenceBudget()
    allowed = [frame for frame in range(100) if budget.reserve(frame)]
    assert allowed == [0, 8, 25, 33, 50, 58, 75, 83]


def test_same_frame_cannot_start_two_scorers_and_seek_resets_budget():
    budget = InferenceBudget()
    assert budget.reserve(100)
    assert not budget.reserve(100)
    assert budget.reserve(10)
    assert not budget.reserve(11)


def test_player_loss_retry_can_run_after_four_frames_without_raising_window_limit():
    recovery = LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    assert recovery._begin_inference(100)
    assert not recovery.scoring_ready(103, contact_loss=True)
    assert not recovery.scoring_ready(104)
    assert recovery.scoring_ready(104, contact_loss=True)
    assert recovery._begin_inference(104)
    assert not recovery._begin_inference(104)
    assert not recovery.scoring_ready(108, contact_loss=True)
    assert recovery.scoring_ready(125)


def test_automatic_player_loss_requests_bounded_urgent_retry():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 104
    analyzer.ball_center = (50,50)
    analyzer.stuck_frame_count = 2
    analyzer._player_point_zone = lambda pos:'player_body'
    recovery = LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    recovery._begin_inference(100)
    analyzer.local_ai_recovery = recovery
    assert analyzer._automatic_local_ai_scoring_ready()
    assert recovery._begin_inference(104)


def test_healthy_or_clear_held_position_keeps_normal_cooldown():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 104
    analyzer.ball_center = (50,50)
    recovery = LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    recovery._begin_inference(100)
    analyzer.local_ai_recovery = recovery
    analyzer.stuck_frame_count = 0
    analyzer._player_point_zone = lambda pos:'player_body'
    assert not analyzer._automatic_local_ai_scoring_ready()
    analyzer.stuck_frame_count = 3
    analyzer._player_point_zone = lambda pos:None
    assert not analyzer._automatic_local_ai_scoring_ready()


def test_pre_track_player_loss_survives_a_reacquisition_stuck_reset():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count=104
    analyzer.ball_center=(50,50)
    analyzer.ball_size=36
    analyzer.stuck_frame_count=0
    analyzer._player_point_zone=lambda point:'player_body'
    analyzer._normal_ball_evidence_is_strong=lambda *args:False
    analyzer._point_in_player_contact_corridor=lambda point:True
    recovery=LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    recovery._begin_inference(100)
    analyzer.local_ai_recovery=recovery
    snapshot={'stuck_frame_count':3,'last_motion':{'dx':0,'dy':20,'distance':20},'ball_size':36}
    assert analyzer._contact_local_ai_trigger((50,50),(50,50),snapshot) is not None
    assert recovery._begin_inference(104)


def test_throttled_scorers_do_not_start_processes_or_serialize_images():
    recovery = LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    assert recovery._begin_inference(100)
    with patch('ball_local_ai_recovery.subprocess.run') as run, patch('ball_local_ai_recovery.cv2.imwrite') as write:
        image = np.zeros((20, 20, 3), np.uint8)
        assert recovery._score(image, 101, [dict(x=10,y=10)]) == []
        assert recovery._score_batch([(image, 100, []), (image, 102, [dict(x=10,y=10)])]) == [[], []]
        run.assert_not_called()
        write.assert_not_called()


def test_healthy_normal_flight_skips_ai_even_with_contact_owner():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 40
    analyzer.ball_size = 36
    analyzer._player_point_zone = lambda pos: None
    analyzer._contact_local_ai_state = dict(miss_count=5)
    analyzer._terminal_previous_gray = np.zeros((100, 100), np.uint8)
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[47:54, 57:64] = 255
    snapshot = dict(ball_velocity_history=[10, 10, 10])
    assert analyzer._contact_local_ai_trigger((50,50), (60,50), snapshot, frame=frame) is None
    assert analyzer._contact_local_ai_state is None


def test_player_occlusion_static_patch_and_teleport_are_not_healthy():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.ball_size = 36
    analyzer._player_point_zone = lambda pos: 'player_body'
    frame = np.full((100,100,3),255,np.uint8)
    analyzer._terminal_previous_gray = np.zeros((100,100),np.uint8)
    assert not analyzer._normal_ball_evidence_is_strong(frame, (50,50), (60,50), {})
    analyzer._player_point_zone = lambda pos: None
    analyzer._terminal_previous_gray[:] = 255
    assert not analyzer._normal_ball_evidence_is_strong(frame, (50,50), (60,50), {})
    assert not analyzer._normal_ball_evidence_is_strong(frame, (50,50), (50,50), {})
    assert not analyzer._normal_ball_evidence_is_strong(frame, (0,0), (500,500), {})


def test_budget_skip_does_not_restore_or_change_a_normal_observation():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 101
    analyzer.ball_center = (60,50)
    recovery = LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    recovery._begin_inference(100)
    analyzer.local_ai_recovery = recovery
    analyzer._normal_ball_evidence_is_strong = lambda *args: False
    assert analyzer._contact_local_ai_trigger((50,50), (60,50), {}) is None
    assert analyzer.ball_center == (60,50)


def test_genuine_tracking_loss_is_still_eligible_for_ai():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.local_ai_recovery = LocalBallAIRecovery.__new__(LocalBallAIRecovery)
    analyzer._contact_local_ai_watch_until_frame = 105
    assert analyzer._contact_local_ai_trigger((50,50), None, {}) == 'contact-loss-after-player-proximity'


def test_expected_edge_absence_does_not_trigger_contact_or_buffer_ai():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.ball_center = (50, 5)
    analyzer.local_ai_recovery = SimpleNamespace(scoring_ready=lambda frame: pytest.fail('edge wait must not request inference'))
    analyzer._contact_local_ai_watch_until_frame = 105
    analyzer._contact_local_ai_state = dict(miss_count=0)
    analyzer._top_return_wait_frames = 15
    analyzer._top_return_anchor = (50, 5)
    assert not analyzer._automatic_local_ai_scoring_ready()
    assert analyzer._contact_local_ai_trigger((50,5), None, {}) is None
    assert analyzer._contact_local_ai_state is None
    assert analyzer.ball_center == (50, 5)


def test_sparse_ai_history_predicts_per_frame_speed_not_eightfold_velocity():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 110
    analyzer._force_local_ai_range = (100, 200)
    analyzer._force_local_ai_history = [dict(frame=100,pos=(100,100)), dict(frame=108,pos=(180,100))]
    analyzer._player_point_zone = lambda pos: None
    captured = []
    def score(image, frame, candidates):
        captured.extend(candidates)
        raise LookupError('stop after observing candidate priority')
    analyzer.local_ai_recovery = SimpleNamespace(_config={}, _score=score)
    with patch('ball_ai_recovery_probe.collect_candidates', return_value=[dict(x=260,y=100),dict(x=200,y=100)]):
        with pytest.raises(LookupError):
            analyzer._force_local_ai_frame(np.zeros((300,300,3),np.uint8), (180,100))
    assert captured[0]['x'] == 200


def test_ai_miss_preserves_normal_loss_clock_and_rejects_bad_new_anchor():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    def restore(state):
        analyzer.stuck_frame_count = state['stuck_frame_count']
    analyzer._restore_tracking_state_for_provisional_guard = restore
    before = dict(stuck_frame_count=1)
    after = dict(stuck_frame_count=5)
    assert analyzer._normal_fallback_after_ai_miss((50,50), (50,50), before, after) == (50,50)
    assert analyzer.stuck_frame_count == 5
    assert analyzer._normal_fallback_after_ai_miss((50,50), None, before, after) is None
    assert analyzer.stuck_frame_count == 5
    assert analyzer._normal_fallback_after_ai_miss((50,50), (500,500), before, dict(stuck_frame_count=0)) == (50,50)
    assert analyzer.stuck_frame_count == 2


@pytest.mark.parametrize("variant", ["current", "stale", "other_position", "other_previous", "held"])
def test_measured_causal_observation_only_protects_current_exact_step(variant):
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.ball_size = 10
    analyzer.stuck_frame_count = int(variant == "held")
    analyzer._configured_causal_evidence = dict(
        frame=99 if variant == "stale" else 100,
        pos=(61, 50) if variant == "other_position" else (60, 50),
        previous=(49, 50) if variant == "other_previous" else (50, 50))
    # A size collapse ordinarily requires model confirmation. Only the exact
    # measured step may bypass that fallback; stale evidence cannot do so.
    assert analyzer._normal_ball_evidence_is_strong(
        np.zeros((100, 100, 3), np.uint8), (50, 50), (60, 50),
        {"ball_size": 100}) == (variant == "current")


@pytest.mark.parametrize('case,expected', [('measured',True),('stale',False),('fragment',False),
    ('jump',False),('static',False),('head',False),('lost',False)])
def test_coherent_configured_serve_crossing_racket_needs_no_ai(case,expected):
    import cv2
    obj=InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    obj.frame_count=1001
    obj.point_start_frame_internal=980 if case=='stale' else 1000
    obj._active_serve_area_end='far'
    obj.tracking_hsv_ranges=[{}]
    obj.stuck_frame_count=int(case=='lost')
    obj.ball_size=4 if case=='fragment' else 16.5
    obj._player_point_zone=lambda p:'player_head_hat' if case=='head' else 'racket_fragment'
    previous=(106,58)
    current=(20,160) if case=='jump' else (80,92)
    image=np.zeros((200,200,3),np.uint8)
    cv2.circle(image,current,3,(20,240,240),-1)
    obj._terminal_previous_gray=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if case=='static'
        else np.zeros((200,200),np.uint8))
    snapshot=dict(ball_size=26,stuck_frame_count=0,ball_velocity_history=[37.6],
        last_motion={'dx':-23,'dy':30,'distance':37.6})
    assert obj._normal_ball_evidence_is_strong(image,previous,current,snapshot) is expected
