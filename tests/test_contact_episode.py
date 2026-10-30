from unittest.mock import Mock
import pytest
from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.mark.parametrize('gap,second_player,new_point,expected', [
    (0,0,False,1), (1,0,False,1), (3,0,False,1),
    (4,0,False,2), (2,1,False,2), (3,0,True,2),
])
def test_contact_episode_counts_once_without_merging_opponents_or_later_hits(gap,second_player,new_point,expected):
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count = 100
    owner.point_start_frame_internal = 90
    owner.player_tracker = None
    owner.player_names = ['P1','P2']
    owner.point_stats = [dict(total_hits=0),dict(total_hits=0)]
    owner._point_hit_count = 0
    owner._last_counted_contact_frame = -1
    owner._reset_side_recross_watch = Mock()
    owner._record_racket_contact((100,100), player_idx_override=0)
    owner.frame_count = 100+gap
    if new_point:
        owner.point_start_frame_internal = 101
    owner._record_racket_contact((200,150), player_idx_override=second_player)
    assert owner._point_hit_count == expected
    assert sum(s['total_hits'] for s in owner.point_stats) == expected
    assert owner._reset_side_recross_watch.call_count == expected
    assert owner._last_racket_contact_frame == 100+gap
