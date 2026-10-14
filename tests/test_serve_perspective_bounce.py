import cv2
import numpy as np
from serve_perspective_bounce import impact
from night_serve_geometry import near_service_box
from interactive_ball_analyzer import InteractiveBallAnalyzer


def samples():
    positions = [(1934, 650), (1931, 738), (1930, 836),
                 (1934, 858), (1936, 876), (1941, 901)]
    return [dict(frame=100+i, pos=point, area=150) for i, point in enumerate(positions)]


def test_deceleration_and_rebound_identify_landing_without_y_reversal():
    result = impact(samples())
    assert result['frame'] == 103
    assert result['pos'] == (1934, 858)


def test_impact_requires_continuity_and_measured_rebound():
    values = samples()
    values[4]['frame'] += 1
    assert impact(values) is None
    values = samples()
    values[-1]['pos'] = (1937, 878)
    assert impact(values) is None
    values = samples()
    values[-1]['area'] = 1000
    assert impact(values) is None
    values = samples()
    values[3]['pos'] = (2100, 858)
    assert impact(values) is None


def test_constant_flight_does_not_become_a_bounce():
    values = [dict(frame=i, pos=(1900, 600+i*40), area=100) for i in range(6)]
    assert impact(values) is None


def test_near_box_uses_physical_centre_line():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    cv2.line(frame, (1000, 858), (2700, 916), (255,255,255), 8)
    cv2.line(frame, (1920, 550), (1900, 900), (255,255,255), 8)
    model = near_service_box(frame, 492)
    assert model is not None
    assert abs(model['center_line']['a'] * 858 + model['center_line']['b'] - 1902) < 12
    assert abs(model['service_line']['a'] * 1934 + model['service_line']['b'] - 890) < 12


def test_missing_centre_line_cannot_certify_a_service_half():
    frame = np.zeros((2160, 3840, 3), np.uint8)
    cv2.line(frame, (1000, 858), (2700, 916), (255,255,255), 8)
    assert near_service_box(frame, 492) is None


def test_pending_fault_preserves_landing_through_confirmation_delay():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 106
    analyzer.point_start_frame_internal = 90
    reason = 'Serve bounce outside left service box'
    analyzer._serve_perspective_bounce_terminal = dict(
        point_start=90, confirmed_frame=105, frame=103,
        position=(1934,858), reason=reason)
    assert analyzer._confirmed_pending_out_endpoint(reason) == (1934,858)
    assert analyzer._confirmed_pending_out_endpoint('different reason') is None
    analyzer.point_start_frame_internal = 100
    assert analyzer._confirmed_pending_out_endpoint(reason) is None
