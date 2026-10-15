import numpy as np
import pytest
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


def test_single_image_recovery_cannot_teleport_to_other_player():
    analyzer = recovering_tracker()
    analyzer.motion_history = [{'frame': 99, 'pos': analyzer.ball_center}]
    analyzer.stuck_frame_count = 1
    observation = dict(pos=(2062, 200), area=24, hsv=(35, 160, 220))
    assert analyzer._commit_night_visible_ball_recovery(observation, None) == (2104, 1124)
    assert analyzer.ball_center == (2104, 1124)
    assert analyzer.stuck_frame_count == 2
    assert analyzer.last_motion['dy'] == 151


def test_recovery_speed_limit_accounts_for_frames_with_missing_ball():
    analyzer = recovering_tracker()
    analyzer.motion_history = [{'frame':99, 'pos':analyzer.ball_center}]
    analyzer.stuck_frame_count = 5
    observation = {'pos':(500,300), 'area':24, 'hsv':(35,160,220)}
    assert analyzer._commit_night_visible_ball_recovery(observation, None) == (500,300)


@pytest.mark.parametrize('case', ['bounce', 'racket', 'fast_return', 'no_first_bounce',
                                 'stale_pair', 'weak_image', 'gap'])
def test_verified_receiver_recovery_preserves_second_bounce(case):
    a = recovering_tracker()
    a.ball_center = (50,70)
    a.motion_history = [{'frame':97,'pos':(60,40)}, {'frame':98,'pos':(55,55)},
                        {'frame':99,'pos':(50,70)}]
    a.ground_bounce_count = 0 if case == 'no_first_bounce' else 1
    a.last_ground_bounce_frame = 75
    a._last_racket_contact_frame = 90 if case == 'racket' else 60
    a._get_ground_bounce_context = lambda *args, **kwargs: {'origin':(50,70)}
    candidate = {'source':'receiver_temporal','pos':(48,60),'area':30,
                 'motion_frames':(99,100),'motion_mean':24,'motion_max':120}
    if case == 'fast_return': candidate['pos'] = (48,20)
    if case == 'stale_pair': candidate['motion_frames'] = (98,100)
    if case == 'weak_image': candidate['motion_mean'] = 1
    if case == 'gap': a.motion_history[1]['frame'] = 97
    result = a._receiver_second_bounce_context(candidate, np.zeros((500,500,3),dtype=np.uint8))
    assert (result is not None) == (case == 'bounce')
