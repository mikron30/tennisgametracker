import math
from interactive_ball_analyzer import InteractiveBallAnalyzer


def analyzer():
    a = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    a.config_file = 'hsv_config_04_left_night.json'
    a.frame_count = 100
    a.ball_center, a.ball_size = (2213,880),327
    a.serve_area_y_min, a.serve_area_y_max = 650,1100
    a.serve_contact_y_ratio = .7
    a.serve_contact_min_ball_size = 180
    a.serve_contact_min_dx, a.serve_contact_min_dy = 100,40
    a.serve_direction_dx, a.serve_direction_dy = -1,-1
    a._serve_contact_grace_frames = 10
    a._serve_launch_direction_x = 0
    a._awaiting_serve_bounce = True
    a._near_serve_toss_seed = dict(frame=80,position=(2170,1000),area=327)
    a._point_hit_count = 0
    a.max_ball_speed = 400
    a.last_motion = dict(dx=-4,dy=43,distance=43.2)
    a.prev_motion = dict(dx=-2,dy=40,distance=40.1)
    a.prev_prev_motion = None
    return a


def candidate(a,point,motion):
    return dict(pos=point,area=300,distance=math.dist(a.ball_center,point),
                motion_mean=motion,motion_max=motion*2,score=10,
                source='primary',contour=object())


def test_high_contact_window_prefers_airborne_ball_over_player():
    a = analyzer()
    assert a._night_serve_contact_launch_scan_active()
    ball = candidate(a,(2060,700),50)
    player = candidate(a,(2209,1052),60)
    static = candidate(a,(2000,650),1)
    assert a._prefer_serve_contact_launch_candidate([player,static,ball],(2160,3840)) is ball
    assert a._serve_launch_direction_x == -1


def test_contact_window_stays_bounded_to_descent_and_grace():
    a = analyzer()
    a.ball_center = (2213,830)
    assert not a._night_serve_contact_launch_scan_active()
    a.ball_center = (2213,880)
    a.last_motion = dict(dx=0,dy=-40,distance=40)
    assert not a._night_serve_contact_launch_scan_active()
    a = analyzer()
    assert not a._night_serve_contact_launch_scan_active(False)


def test_day_camera_keeps_existing_contact_margin():
    a = analyzer()
    a.config_file = 'hsv_config_court2.json'
    assert a._serve_contact_search_margin() == 81


def test_ambiguous_racket_fragment_cannot_replace_contact_reference():
    a = analyzer()
    fragment = candidate(a,(2225,946),30)
    fragment['area'] = 34
    assert a._serve_contact_fragment_should_hold(fragment)
    assert a._serve_contact_fragment_should_hold(candidate(a,(2209,1052),50))
    # A genuine normal toss continuation and a departing ball remain eligible.
    assert not a._serve_contact_fragment_should_hold(candidate(a,(2210,923),40))
    assert not a._serve_contact_fragment_should_hold(candidate(a,(2060,700),50))
    a._serve_launch_direction_x = -1
    assert not a._serve_contact_fragment_should_hold(fragment)
    a = analyzer()
    a.frame_count = 126
    assert not a._serve_contact_fragment_should_hold(fragment)
    a = analyzer()
    a._point_hit_count = 1
    assert not a._serve_contact_fragment_should_hold(fragment)
