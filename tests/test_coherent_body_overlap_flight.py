import numpy as np
import pytest
import cv2

from interactive_ball_analyzer import InteractiveBallAnalyzer


def flight():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.ball_size = 40
    analyzer.stuck_frame_count = 0
    analyzer._player_point_zone = lambda point: 'player_body'
    analyzer.motion_history = [
        {'frame': 97, 'pos': (20, 20)},
        {'frame': 98, 'pos': (30, 23)},
        {'frame': 99, 'pos': (40, 27)},
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[28:38, 45:55] = 240
    snapshot = {'stuck_frame_count': 0,
                '_prev_frame_gray': np.zeros((100, 100), dtype=np.uint8)}
    return analyzer, frame, snapshot


def test_visible_coherent_body_overlap_keeps_hsv_without_ai():
    analyzer, frame, snapshot = flight()
    analyzer._finalize_pending_provisional_static_candidate = lambda: False
    def unexpected_ai():
        pytest.fail('Healthy visible flight must not request inference')
    analyzer._automatic_local_ai_scoring_ready = unexpected_ai
    assert analyzer._try_local_ai_recovery(
        (40, 27), (50, 32), 0, snapshot, frame) == (50, 32)


@pytest.mark.parametrize('case', ['fresh', 'expired', 'different_position', 'held'])
def test_verified_receiver_pair_protects_only_its_measured_current_observation(case):
    analyzer, frame, snapshot = flight()
    analyzer.motion_history = []
    analyzer._visible_receiver_pair_evidence = {
        'frame': 99 if case == 'expired' else 100,
        'pos': (55, 32) if case == 'different_position' else (50, 32),
    }
    if case == 'held':
        analyzer.stuck_frame_count = 1
    assert analyzer._normal_ball_evidence_is_strong(frame, (40, 27), (50, 32), snapshot) == (case == 'fresh')


def test_small_direction_noise_does_not_expose_flight_to_ai_takeover():
    analyzer, _, snapshot = flight()
    analyzer.motion_history = [
        {'frame': 97, 'pos': (67, 72)},
        {'frame': 98, 'pos': (61, 56)},
        {'frame': 99, 'pos': (59, 41)},
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[28:38, 45:55] = 240
    assert analyzer._normal_ball_evidence_is_strong(frame, (59, 41), (50, 32), snapshot)


def test_moving_fragment_with_abrupt_size_collapse_is_not_strong_ball_evidence():
    analyzer, frame, snapshot = flight()
    analyzer._player_point_zone = lambda point: None
    analyzer.ball_size = 16
    snapshot['ball_size'] = 144
    assert not analyzer._normal_ball_evidence_is_strong(frame, (40, 27), (50, 32), snapshot)


@pytest.mark.parametrize('case', ['bounce', 'stale', 'jump', 'no_history', 'static'])
def test_measured_ground_bounce_can_overlap_shoe_box_without_ai_rollback(case):
    analyzer, frame, snapshot = flight()
    analyzer._player_point_zone = lambda point: 'player_shoes'
    analyzer.ball_size = 34
    analyzer.last_ground_bounce_frame = 99 if case == 'stale' else 100
    snapshot.update(ball_size=21, motion_history=[
        {'frame':97, 'pos':(60,20)}, {'frame':98, 'pos':(55,38)},
        {'frame':99, 'pos':(50,55)}])
    if case == 'no_history':
        snapshot['motion_history'] = []
    frame[:] = 0
    if case != 'static':
        frame[35:45, 43:53] = 240
    target = (75,40) if case == 'jump' else (48,40)
    assert analyzer._normal_ball_evidence_is_strong(frame,(50,55),target,snapshot) == (case == 'bounce')


def test_size_collapse_near_contact_triggers_ai_without_a_large_turn():
    analyzer, _, snapshot = flight()
    analyzer.ball_size = 16
    analyzer.local_ai_recovery = object()
    analyzer._automatic_local_ai_scoring_ready = lambda: True
    analyzer._point_in_player_contact_corridor = lambda point: True
    snapshot.update(ball_size=144, last_motion={'dx': 0, 'dy': 30, 'distance': 30})
    reason = analyzer._contact_local_ai_trigger((40, 27), (40, 57), snapshot)
    assert reason.startswith('trajectory-size-collapse:')


def test_fast_tiny_fragment_triggers_verification_when_player_detection_misses():
    analyzer, _, snapshot = flight()
    analyzer.ball_size = 16
    analyzer.local_ai_recovery = object()
    analyzer._player_point_zone = lambda point: None
    analyzer._automatic_local_ai_scoring_ready = lambda: True
    analyzer._point_in_player_contact_corridor = lambda point: False
    snapshot.update(ball_size=144, last_motion={'dx': 0, 'dy': 60, 'distance': 60})
    assert analyzer._contact_local_ai_trigger((40, 27), (40, 87), snapshot).startswith('trajectory-size-collapse:')


def test_fresh_visible_ball_repairs_fragment_before_ai():
    analyzer, frame, snapshot = flight()
    analyzer.ball_size = 16
    analyzer.primary_hsv_lower = (20, 50, 130)
    analyzer.primary_hsv_upper = (85, 255, 255)
    frame[:] = 0
    cv2.circle(frame, (60, 60), 8, (0, 240, 240), -1)
    frame[77:80, 69:72] = (0, 240, 240)  # tiny racket-colour fragment
    snapshot.update(ball_size=144, last_motion={'dx': 20, 'dy': 40}, ball_center=(50, 30))
    restored = []
    analyzer._restore_tracking_state_for_provisional_guard = lambda state: restored.append(state)
    analyzer._commit_local_ball_flight = lambda candidate, image: candidate['pos']
    result = analyzer._repair_tiny_fragment_from_visible_flight(frame, (50, 30), (70, 78), snapshot)
    assert result is not None and np.linalg.norm(np.array(result)-(60, 60)) <= 2
    assert restored == [snapshot]


def test_static_coloured_patch_cannot_repair_fragment():
    analyzer, frame, snapshot = flight()
    analyzer.ball_size = 16
    analyzer.primary_hsv_lower = (20, 50, 130)
    analyzer.primary_hsv_upper = (85, 255, 255)
    frame[:] = 0
    cv2.circle(frame, (60, 60), 8, (0, 240, 240), -1)
    snapshot.update(ball_size=144, last_motion={'dx': 20, 'dy': 40},
                    _prev_frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert analyzer._repair_tiny_fragment_from_visible_flight(frame, (50, 30), (70, 78), snapshot) is None


@pytest.mark.parametrize('evidence', ['complete', 'moving_outline', 'no_racket', 'stale', 'no_reversal'])
def test_visible_volley_opportunity_requires_fresh_repair_reversal_and_racket(evidence):
    analyzer, frame, _ = flight()
    analyzer.point_start_frame_internal = 10
    analyzer._last_racket_contact_player = 0
    analyzer._last_racket_contact_frame = 50
    analyzer._side_recross_hitter = 0
    analyzer._recent_offscreen_return_hold_active = lambda **kwargs: False
    analyzer._player_index_at_point = lambda point, image=None: 1
    analyzer._side_recross_has_opponent_contact_opportunity = lambda *args: False
    analyzer._visible_fragment_repair_frame = 98 if evidence == 'stale' else 99
    analyzer._visible_fragment_repair_position = (50, 80)
    analyzer._last_direction_change_frame = 100
    analyzer._last_direction_change_vertical_reversal = evidence != 'no_reversal'
    analyzer._racket_below_ball_motion_debug = lambda *args, **kwargs: (
        {'hint': False, 'motion_hint': True, 'edge_ratio': .072, 'high_sat_ratio': .071}
        if evidence == 'moving_outline' else {'hint': evidence != 'no_racket'})
    assert not analyzer._maybe_infer_return_contact_from_side_recross((50, 40), frame, (50, 80))
    assert (getattr(analyzer, '_side_recross_opponent_contact_frame', -1) == 100) == (evidence in ('complete', 'moving_outline'))


@pytest.mark.parametrize('problem', ['gap', 'held', 'previous_loss', 'large', 'shoes', 'reversal', 'static'])
def test_ambiguous_player_overlap_still_needs_recovery(problem):
    analyzer, frame, snapshot = flight()
    target = (50, 32)
    if problem == 'gap':
        analyzer.motion_history[0]['frame'] = 96
    elif problem == 'held':
        analyzer.motion_history[1]['pos'] = (40, 27)
    elif problem == 'previous_loss':
        snapshot['stuck_frame_count'] = 2
    elif problem == 'large':
        analyzer.ball_size = 300
    elif problem == 'shoes':
        analyzer._player_point_zone = lambda point: 'player_shoes'
    elif problem == 'reversal':
        target = (30, 22)
    elif problem == 'static':
        snapshot['_prev_frame_gray'] = frame[:, :, 0].copy()
    assert not analyzer._normal_ball_evidence_is_strong(frame, (40, 27), target, snapshot)
