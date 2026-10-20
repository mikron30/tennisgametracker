from far_serve_launch import (outgoing_flight, rising_toss, top_reentry_limit,
                             prestance_toss, flight_stance_anchor)


def test_body_fragment_after_toss_does_not_replace_verified_apex():
    history = [(2000, 114), (1999, 100), (2000, 84), (1998, 70),
               (1997, 58), (1998, 46), (1998, 38), (1997, 29),
               (2016, 126), (2020, 142), (2017, 140), (1985, 24)]
    frames = [100, 102, 103, 104, 105, 106, 107, 108, 113, 115, 116, 121]
    result = rising_toss(history, frames, 145, 121)
    assert result['apex_pos'] == (1997, 29)
    assert result['low_frame'] == 100


def test_expired_or_jumping_highlights_cannot_extend_scan():
    history = [(2000, 114), (2000, 95), (2000, 70), (2000, 40)]
    assert rising_toss(history, [1, 2, 3, 4], 145, 50) is None
    assert rising_toss(history, [1, 5, 9, 13], 145, 13) is None
    assert rising_toss([(1700, 114), (1800, 95), (1900, 70), (2000, 40)],
                       [1, 2, 3, 4], 145, 4) is None


def test_outgoing_flight_requires_consecutive_fast_steps():
    assert outgoing_flight([(2000, 110), (1990, 135), (1980, 165), (1970, 205)],
                           [1, 2, 3, 4])
    assert not outgoing_flight([(2000, 110), (1998, 122), (1996, 136), (1994, 151)],
                               [1, 2, 3, 4])
    assert not outgoing_flight([(2000, 110), (1990, 135), (1980, 165), (1970, 205)],
                               [1, 2, 4, 5])
    assert not outgoing_flight([(2000, 110), (1990, 135), (1980, 165), (1970, 400)],
                               [1, 2, 3, 4])


def test_diagonal_departure_with_less_vertical_speed_is_a_strike():
    assert outgoing_flight([(1900, 120), (1920, 133), (1942, 150), (1965, 172)],
                           [1, 2, 3, 4], allow_diagonal=True)
    assert not outgoing_flight([(1900, 120), (1920, 133), (1900, 150), (1923, 172)],
                               [1, 2, 3, 4], allow_diagonal=True)
    assert not outgoing_flight([(1900, 120), (1920, 133), (1942, 150), (1965, 172)],
                               [1, 2, 3, 4])


def test_top_exit_wait_accepts_only_top_strip_until_reentry():
    points = [(2000, 114), (2000, 90), (2000, 65), (1998, 29)]
    frames = [10, 11, 12, 13]
    assert top_reentry_limit(points, frames, 145, 20) < 60
    assert top_reentry_limit(points, frames, 145, 59) is None
    # Continuous descending flight after re-entry can leave the strip.
    assert top_reentry_limit(points + [(1997, 35)], frames + [20], 145, 21) is None


def test_missed_falling_toss_is_not_an_offscreen_exit():
    points = [(1900,126), (1904,100), (1908,75), (1912,50),
              (1916,30), (1920,36)]
    assert top_reentry_limit(points, list(range(10,16)), 145, 18) is None


def test_slow_rising_toss_near_top_keeps_body_fragment_guard():
    points = [(1900,126), (1904,100), (1908,75), (1912,50), (1916,30)]
    assert top_reentry_limit(points, list(range(10,15)), 145, 17) < 60


def test_delayed_stance_preserves_live_toss_with_brief_visibility_gap():
    positions = [(2000,141), (2000,118), (2001,92), (2000,66),
                 (2001,40), (2000,24), (2001,9)]
    frames = [100,101,102,103,104,105,107]
    assert prestance_toss(positions, frames, 145, 107) is not None
    assert prestance_toss(positions, frames[:-1]+[110], 145, 110) is None
    assert prestance_toss(positions+[(2000,80)], frames+[108],145,108) is None
    assert prestance_toss(positions,frames,145,112) is None


def test_diagonal_outgoing_serve_validates_original_toss_alignment():
    points = [(2010,141), (2010,110), (2009,80), (2009,45), (2008,9),
              (1990,30), (1970,70), (1945,110), (1920,150), (1895,180)]
    frames = list(range(100,110))
    assert flight_stance_anchor((1870,210), points, frames,145,110) == points[0]
    assert flight_stance_anchor((2100,30), points, frames,145,110) == (2100,30)
    assert flight_stance_anchor((1600,210), points, frames,145,110) == (1600,210)
    assert flight_stance_anchor((1870,210), points, frames,145,160) == (1870,210)


def test_outgoing_candidates_keep_diagonal_ball_over_slow_shirt_fragment():
    from far_serve_launch import outgoing_candidate_path
    samples = []
    for i, pos in enumerate([(250, 100), (228, 110), (207, 125), (185, 140)]):
        samples.append(dict(frame=100+i, candidates=[
            dict(pos=(290-i, 170+i), area=18, player_zone='player_body'),
            dict(pos=pos, area=24, player_zone=None)]))
    path = outgoing_candidate_path(samples)
    assert [c['pos'] for c in path] == [(250, 100), (228, 110), (207, 125), (185, 140)]
    samples[-1]['frame'] += 1
    assert outgoing_candidate_path(samples) is None


def test_outgoing_candidates_reject_player_motion_jumps_and_slow_falling_toss():
    from far_serve_launch import outgoing_candidate_path
    def samples(points, zone=None):
        return [dict(frame=20+i, candidates=[dict(pos=p, area=24, player_zone=zone)])
                for i, p in enumerate(points)]
    flight = [(250, 100), (228, 110), (207, 125), (185, 140)]
    assert outgoing_candidate_path(samples(flight, 'player_body')) is None
    assert outgoing_candidate_path(samples([(250,100),(248,110),(247,125),(245,140)])) is None
    assert outgoing_candidate_path(samples([(250,100),(228,110),(280,150),(185,170)])) is None
    assert outgoing_candidate_path(samples(flight)[:3]) is None
    interrupted = samples(flight)
    interrupted[1]['candidates'] = []
    assert outgoing_candidate_path(interrupted) is None

def test_outgoing_path_must_start_near_the_verified_toss():
    from far_serve_launch import outgoing_candidate_path
    samples = [dict(frame=20+i, candidates=[dict(pos=p, area=24, player_zone=None)])
        for i, p in enumerate([(250,100),(228,110),(207,125),(185,140)])]
    assert outgoing_candidate_path(samples, origin=(260,120)) is not None
    assert outgoing_candidate_path(samples, origin=(600,120)) is None
