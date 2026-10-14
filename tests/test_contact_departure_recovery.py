import math
import cv2
import numpy as np
import pytest
from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.mark.parametrize('evidence', ['fresh', 'expired', 'different_anchor'])
def test_wider_contact_search_requires_exact_previous_ai_observation(evidence):
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.ball_center = (500, 400)
    analyzer.ball_size = 90
    analyzer.last_motion = {'dx': 0, 'dy': 70}
    analyzer.net_area_y_max = 300
    analyzer.primary_hsv_lower = (20, 50, 130)
    analyzer.primary_hsv_upper = (85, 255, 255)
    analyzer.alt2_hsv_lower = (15, 40, 100)
    analyzer.alt2_hsv_upper = (85, 255, 255)
    analyzer._point_in_player_contact_corridor = lambda point: False
    analyzer._player_point_zone = lambda point: None
    analyzer._terminal_previous_gray = np.zeros((600, 1000), np.uint8)
    analyzer._verified_ai_contact_sample = {
        'frame': 98 if evidence == 'expired' else 99,
        'pos': (490, 400) if evidence == 'different_anchor' else (500, 400),
        'reason': 'contact-size-collapse:300->90px',
    }
    frame = np.zeros((600, 1000, 3), np.uint8)
    cv2.circle(frame, (500, 270), 8, (0, 240, 240), -1)
    # A closer moving fragment admitted by ALT2 must not win the wider search.
    colour = cv2.cvtColor(np.array([[[17, 150, 220]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    cv2.circle(frame, (430, 420), 8, tuple(map(int, colour)), -1)
    result = analyzer._recover_local_ball_flight(frame)
    correct = result is not None and math.dist(result['pos'], (500, 270)) <= 2
    assert correct == (evidence == 'fresh')
    if correct:
        assert result['verified_contact_departure']
