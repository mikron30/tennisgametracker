from unittest.mock import Mock

import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.mark.parametrize('candidate,zone,expected', [
    ((2987, 1902), None, (2068, 425)),
    ((3151, 1590), 'player_body', (2068, 425)),
    ((2080, 430), 'racket_fragment', (2068, 425)),
    ((2080, 430), None, (2080, 430)),
    (None, None, (2068, 425)),
])
def test_net_endpoint_repair_preserves_impact_identity(candidate, zone, expected):
    tracker = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    tracker.frame_count = 37679
    tracker._terminal_current_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    tracker._terminal_moving_ball_candidate = Mock(return_value=candidate)
    tracker._player_point_zone = Mock(return_value=zone)
    assert tracker._terminal_player_overlap_position(
        'Ball hit the net', (2068, 425)) == expected
