import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer


def test_same_side_pre_net_bounce_ignores_player_overlap():
    """A player-associated rebound cannot end a point after racket contact."""
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 120
    analyzer._awaiting_serve_bounce = False
    analyzer._recent_racket_rebound_bounce_frame = -1000000
    analyzer._last_racket_contact_frame = 114
    analyzer._last_racket_contact_point = (150, 700)
    analyzer.net_area_y_min = 430
    analyzer.net_area_y_max = 510
    analyzer.prev_motion = {'dx': 4.0, 'dy': 68.0, 'distance': 68.1, 'direction_deg': 87.0}
    analyzer.last_motion = {'dx': -3.0, 'dy': -56.0, 'distance': 56.1, 'direction_deg': 273.0}
    analyzer._player_point_zone = lambda point: 'player_shoes' if point == (160, 930) else None

    ended, reason = analyzer._same_side_pre_net_bounce_candidate(
        (157, 874), np.zeros((1080, 1920, 3), dtype=np.uint8)
    )

    assert not ended
    assert reason is None


def test_repaired_terminal_on_hitter_side_is_pre_net_failure():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.net_area_y_min = 430
    analyzer.net_area_y_max = 510
    analyzer._last_racket_contact_frame = 114
    analyzer._last_racket_contact_point = (150, 700)
    analyzer.motion_history = []

    reason = analyzer._terminal_repaired_pre_net_reason(
        (1600, 740), (1080, 1920, 3)
    )

    assert reason == 'Ball bounced before crossing net on hitter side'


@pytest.mark.parametrize('gap, player_overlap, expected', [(False, False, True), (True, False, False), (False, True, False)])
def test_falling_ball_bounces_on_near_hitter_side(gap, player_overlap, expected):
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 160
    analyzer._awaiting_serve_bounce = False
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (900, 800)
    analyzer.net_area_y_min = 430
    analyzer.net_area_y_max = 510
    analyzer.motion_history = [
        {'frame': f, 'pos': p} for f, p in [
            (157, (1100, 510)),
            (158 if not gap else 157, (1104, 545)), (159, (1108, 573)), (160, (1120, 545)),
        ]
    ]
    analyzer.prev_motion = {'dx': 4., 'dy': 28., 'distance': 28.3, 'direction_deg': 82.}
    analyzer.last_motion = {'dx': 12., 'dy': -28., 'distance': 30.5, 'direction_deg': -67.}
    analyzer._player_point_zone = lambda point: 'player_body' if player_overlap else None
    analyzer._point_outside_singles_sidelines = lambda point, frame: (False, None, 200, 1700)
    ended, reason = analyzer._same_side_pre_net_bounce_candidate(
        (1120, 545), np.zeros((1080, 1920, 3), dtype=np.uint8)
    )
    assert ended is expected
    assert reason == ('Ball bounced before crossing net on hitter side' if expected else None)


def test_current_visible_reversal_can_end_before_direction_hold():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 160
    analyzer._awaiting_serve_bounce = False
    analyzer._recent_racket_rebound_bounce_frame = -1000000
    analyzer._recent_offscreen_return_frame = 150
    analyzer._last_racket_contact_frame = 100
    analyzer._last_racket_contact_point = (900, 800)
    analyzer.net_area_y_min = 430
    analyzer.net_area_y_max = 510
    analyzer.motion_history = [
        {'frame': 158, 'pos': (1104, 545)},
        {'frame': 159, 'pos': (1108, 573)},
    ]
    analyzer.prev_motion = {'dx': 4., 'dy': 35., 'distance': 35.2, 'direction_deg': 83.}
    analyzer.last_motion = {'dx': 4., 'dy': 28., 'distance': 28.3, 'direction_deg': 82.}
    analyzer._player_point_zone = lambda point: None
    analyzer._point_outside_singles_sidelines = lambda point, frame: (False, None, 200, 1700)

    ended, reason = analyzer._same_side_pre_net_bounce_candidate(
        (1120, 545), np.zeros((1080, 1920, 3), dtype=np.uint8),
        require_observed_reversal=True,
        proposed_motion={'dx': 12., 'dy': -28., 'distance': 30.5, 'direction_deg': -67.},
    )

    assert ended
    assert reason == 'Ball bounced before crossing net on hitter side'


def test_offscreen_reentry_does_not_infer_opponent_racket_contact():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 120
    analyzer.point_start_frame_internal = 20
    analyzer._last_racket_contact_player = 1
    analyzer._last_racket_contact_frame = 80
    analyzer._recent_offscreen_return_frame = 110
    analyzer.player_names = ['P1', 'P2']
    analyzer._reset_side_recross_watch(1)
    analyzer._player_index_at_point = lambda point, frame=None: 1

    inferred = analyzer._maybe_infer_return_contact_from_side_recross(
        (1100, 700), np.zeros((1080, 1920, 3), dtype=np.uint8)
    )

    assert not inferred
    assert analyzer._last_racket_contact_player == 1


def test_near_hitter_bounce_uses_symmetric_netward_direction():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 120
    analyzer._awaiting_serve_bounce = False
    analyzer._recent_racket_rebound_bounce_frame = -1000000
    analyzer._last_racket_contact_frame = 114
    analyzer._last_racket_contact_point = (900, 760)
    analyzer.net_area_y_min = 430
    analyzer.net_area_y_max = 510
    analyzer.motion_history = [
        {'frame': frame, 'pos': (900, y)}
        for frame, y in [(115, 730), (116, 680), (117, 630), (118, 580), (119, 545), (120, 575)]
    ]
    analyzer.prev_motion = {'dx': 2.0, 'dy': -35.0, 'distance': 35.1, 'direction_deg': 274.0}
    analyzer.last_motion = {'dx': 1.0, 'dy': 30.0, 'distance': 30.0, 'direction_deg': 88.0}
    analyzer._player_point_zone = lambda point: None
    analyzer._point_outside_singles_sidelines = lambda point, frame: (False, None, 200, 1700)

    ended, reason = analyzer._same_side_pre_net_bounce_candidate(
        (902, 575), np.zeros((1080, 1920, 3), dtype=np.uint8)
    )

    assert ended
    assert reason == 'Ball bounced before crossing net on hitter side'
