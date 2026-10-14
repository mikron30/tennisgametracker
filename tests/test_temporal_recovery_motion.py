from interactive_ball_analyzer import InteractiveBallAnalyzer


def recovering_tracker():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.ball_center = (2104, 1124)
    analyzer.last_motion = {'dx': 41, 'dy': 151, 'distance': 156.5}
    analyzer.motion_history = [{'frame': 98, 'pos': analyzer.ball_center}]
    analyzer._delayed_literal_top_return_context = lambda: False
    analyzer._activate_regular_hsv = lambda: None
    def old_anchor_motion(previous, current):
        analyzer.prev_motion = dict(analyzer.last_motion)
    analyzer._update_recovered_motion = old_anchor_motion
    return analyzer


def candidate():
    return {'pos': (2287, 446), 'area': 129, 'hsv': (35, 160, 220),
            'motion_mean': 57, 'motion_max': 221,
            'local_ai_motion_override': {'dx': -17, 'dy': -48,
                'distance': 51.2, 'direction_deg': -109.5},
            'local_ai_velocity_history': [51.2],
            'temporal_history_entry': {'frame': 100, 'pos': (2287, 446),
                'prev_pos': (2304, 494), 'distance': 51.2}}


def test_new_temporal_flight_does_not_compare_against_occluding_fragment():
    analyzer = recovering_tracker()
    analyzer._commit_night_visible_ball_recovery(candidate(), None)
    assert analyzer.prev_motion is None
    assert analyzer.last_motion['dy'] == -48
    assert analyzer.ball_velocity_history == [51.2]
    assert analyzer.motion_history[-1]['prev_pos'] == (2304, 494)
    assert analyzer.ball_center == (2287, 446)


def test_continuing_verified_flight_keeps_prior_motion_for_real_impacts():
    analyzer = recovering_tracker()
    analyzer.last_motion = {'dx': -19, 'dy': -51, 'distance': 54.4}
    observation = candidate()
    observation['local_ai_velocity_history'] = [54.4, 51.2]
    analyzer._commit_night_visible_ball_recovery(observation, None)
    assert analyzer.prev_motion['dy'] == -51


def test_ai_motion_override_without_temporal_pair_keeps_contact_motion():
    analyzer = recovering_tracker()
    observation = candidate()
    observation.pop('temporal_history_entry')
    analyzer._commit_night_visible_ball_recovery(observation, None)
    assert analyzer.prev_motion['dy'] == 151
