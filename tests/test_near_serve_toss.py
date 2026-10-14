import numpy as np
import pytest
from interactive_ball_analyzer import InteractiveBallAnalyzer


def analyzer():
    a = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    a._near_serve_toss_seed = dict(frame=100, position=(1700,1000), area=350)
    a.frame_count = 120
    a.ball_size = 330
    a.serve_area_x_min, a.serve_area_x_max = 760,2900
    a._serve_phase_active = True
    a._awaiting_serve_bounce = True
    a._point_hit_count = 0
    a._serve_launch_direction_x = 0
    return a


def test_large_toss_projecting_over_net_is_not_a_fault():
    a = analyzer()
    point = (1750,520)
    frame = np.zeros((2160,3840,3),np.uint8)
    assert a._near_serve_toss_in_progress(point)
    assert a.detect_point_end(point,frame) == (False,'Serve toss before racket contact')
    assert a._mark_serve_net_contact_candidate(point,frame) is False
    assert a._serve_bounce_in_from_turn_candidate(point,frame,2,5,120,8) is None
    assert a._serve_bounce_out_from_turn_candidate(point,frame,2,5,120,8) is None


@pytest.mark.parametrize('field,value',[
    ('ball_size',40),('_point_hit_count',1),('_serve_launch_direction_x',1),
    ('frame_count',146),('_awaiting_serve_bounce',False),('_serve_phase_active',False)])
def test_launched_ball_or_expired_toss_is_not_protected(field,value):
    a = analyzer()
    setattr(a,field,value)
    assert not a._near_serve_toss_in_progress((1750,520))


def test_diagonal_launch_leaves_toss_lane():
    assert not analyzer()._near_serve_toss_in_progress((1940,670))
