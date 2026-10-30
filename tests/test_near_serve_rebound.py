from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from causal_ball_recovery import find_local_ball
from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.mark.parametrize('age,zone,stuck,arrival,aligned,expected', [
    (1,None,0,True,True,True),
    (2,None,0,True,True,True),
    (3,None,0,True,True,False),
    (1,'player_head_hat',0,True,True,False),
    (1,None,1,True,True,False),
    (1,None,0,False,True,False),
    (1,None,0,True,False,False),
])
def test_fresh_launch_can_accelerate_beyond_toss_median(age,zone,stuck,arrival,aligned,expected):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 100
    owner._last_racket_contact_frame = 100-age
    owner.stuck_frame_count = stuck
    owner.ball_size = 276
    owner._player_point_zone = Mock(return_value=zone)
    owner._terminal_previous_gray = np.zeros((500,500),np.uint8)
    owner._configured_serve_flight_is_coherent = Mock(return_value=False)
    owner._compact_body_overlap_flight_is_coherent = Mock(return_value=False)
    frame = np.zeros((500,500,3),np.uint8)
    previous = (300,300)
    current = (189,117) if aligned else (411,117)
    if arrival:
        cv2.circle(frame,current,10,(240,240,240),-1)
    snapshot = dict(ball_size=308,stuck_frame_count=stuck,
        last_motion=dict(dx=-38,dy=-27),ball_velocity_history=[17,22,26,33,47])
    assert owner._normal_ball_evidence_is_strong(frame,previous,current,snapshot) == expected


def test_fragmented_unstruck_near_toss_cannot_become_net_contact():
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.far_serve_image_baseline = True
    owner._active_serve_area_end = 'near'
    owner._serve_phase_active = owner._awaiting_serve_bounce = True
    owner._point_hit_count = owner._serve_launch_direction_x = 0
    owner.frame_count = 110
    owner._near_serve_toss_seed = dict(frame=100, area=300, position=(1500,1000))
    owner.ball_center = (1520,450)
    owner.ball_size = 80
    owner.serve_area_x_min, owner.serve_area_x_max = 700,2900
    assert owner._near_serve_toss_in_progress(owner.ball_center)
    assert not owner._mark_serve_net_contact_candidate(owner.ball_center, None)
    owner._serve_launch_direction_x = 1
    assert not owner._near_serve_toss_in_progress(owner.ball_center)
    owner._serve_launch_direction_x = 0
    owner.far_serve_image_baseline = False
    assert not owner._near_serve_toss_in_progress(owner.ball_center)


@pytest.mark.parametrize('end,direction,expected', [('far',1,True),('near',-1,False),('far',-1,False)])
def test_small_net_drop_recovery_requires_a_struck_far_serve(end,direction,expected):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.far_serve_image_baseline = True
    owner._active_serve_area_end = end
    owner.serve_direction_dy = direction
    owner._awaiting_serve_bounce = True
    owner._serve_net_contact_provisional = False
    owner.frame_count = 110
    owner._serve_net_contact_frame = 109
    owner._serve_net_contact_point = (200,470)
    owner.ball_center = (201,487)
    owner.last_motion = dict(distance=17)
    owner._player_point_zone = lambda p: None
    assert owner._slow_serve_net_drop_active() is expected


@pytest.mark.parametrize('ys,expected', [
    ([519, 526, 533, 531, 528], (2147, 533)),
    ([519, 526, 533, 531, 529], None),  # Jitter is not a rebound.
    ([519, 526, 533, 535, 537], None),  # Still descending.
    ([519, 526, 533, 531, 535], None),  # One upward observation.
])
def test_small_net_rebound_needs_two_rising_samples_and_total_rise(ys, expected):
    from serve_net_drop import net_drop_bounce
    polygon = np.array([[1164,400],[1934,420],[2709,468],
                        [1168,518],[1936,530],[2697,585]], dtype=np.float32)
    samples = [dict(frame=100+i, pos=(2147, y)) for i, y in enumerate(ys)]
    assert net_drop_bounce(samples, polygon, (2145,471), 2160) == expected


@pytest.mark.parametrize('point,blocked', [
    ((1864, 499), True),  # Reported false let: inside the net mesh.
    ((1864, 550), False),  # Genuine near-side landing at the same x.
    ((2600, 550), True),  # Sloping net: use the local ground edge.
    ((1200, 550), False),
    ((3000, 499), False),  # No calibrated net coverage.
])
def test_far_serve_requires_receiving_ground_landing(point, blocked):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.far_serve_image_baseline = True
    owner._serve_phase_active = owner._awaiting_serve_bounce = True
    owner.serve_direction_dy = 1
    owner.net_area_points = [[1164, 400], [1934, 420], [2709, 468],
                             [1168, 518], [1936, 530], [2697, 585]]
    assert owner._serve_landing_obscured_by_net(point) is blocked
    owner.serve_direction_dy = -1
    assert not owner._serve_landing_obscured_by_net(point)
    owner.serve_direction_dy = 1
    owner._awaiting_serve_bounce = False
    assert not owner._serve_landing_obscured_by_net(point)


def test_unconfirmed_net_landing_cannot_award_let_or_fault():
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner._serve_landing_obscured_by_net = lambda point: True
    owner.ground_bounce_count = 1
    owner.frame_count = 41743
    owner._pending_rally_end_reason = None
    owner._classify_ground_bounce = Mock()
    owner._handle_ground_bounce_event((1864, 499), None)
    assert owner.ground_bounce_count == 0
    assert owner._awaiting_serve_bounce
    assert owner._pending_rally_end_reason is None
    owner._classify_ground_bounce.assert_not_called()
    owner._last_serve_bounce_point = None
    owner._commit_serve_bounce_in_event({'point': (1864, 499)}, None)
    assert owner._last_serve_bounce_point is None
    assert owner.ground_bounce_count == 0
    assert owner._awaiting_serve_bounce


def test_bounded_serve_search_finds_rising_ball_outside_slow_projection():
    previous = np.full((200, 300, 3), 40, np.uint8)
    image = previous.copy()
    cv2.circle(previous, (150, 110), 4, (0, 220, 220), -1)
    cv2.circle(image, (144, 73), 4, (0, 220, 220), -1)
    filters = [(np.array([20, 50, 130], np.uint8), np.array([85, 255, 255], np.uint8))]
    args = (image, cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
            (150, 110), dict(dx=-7, dy=0), 38, filters)
    assert find_local_ball(*args) is None
    result = find_local_ball(*args, serve_rebound_search=True)
    assert result is not None and np.linalg.norm(np.subtract(result['pos'], (144, 73))) < 2
    # Enlarging the crop cannot turn a stationary highlight into ball evidence.
    assert find_local_ball(image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                           *args[2:], serve_rebound_search=True) is None


@pytest.mark.parametrize('moving', [False, True])
def test_static_exclusion_does_not_bypass_replacement_validation(moving):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 100
    owner._far_serve_occlusion_hold_active = lambda: False
    owner._last_static_guard_position = (120, 120)
    owner._finalize_pending_provisional_static_candidate = Mock(side_effect=[True, False])
    owner._last_tracked_candidate_motion_frame = 100
    owner._last_tracked_candidate_motion_mean = 20 if moving else 0
    owner._last_tracked_candidate_motion_max = 120 if moving else 0
    owner.track_ball_in_frame = Mock(return_value=(300, 140))
    owner._static_blob_near = Mock(return_value=None)
    owner._restore_tracking_state_for_provisional_guard = Mock()
    owner._normal_ball_evidence_is_strong = Mock(return_value=True)
    snapshot = dict(ball_velocity_history=[7, 9, 11])
    result = owner._try_local_ai_recovery((100, 100), (120, 120), 1,
                                         snapshot, np.zeros((400, 400, 3), np.uint8))
    if moving:
        assert result == (300, 140)
        owner._normal_ball_evidence_is_strong.assert_called_once()
    else:
        assert result == (100, 100) and owner.stuck_frame_count == 2
        assert owner._pending_provisional_static_candidate['pos'] == (300, 140)
        owner._normal_ball_evidence_is_strong.assert_not_called()
