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


def test_configured_projected_bounces_keep_lateral_motion_and_confirm_rebound():
    paths = [
        [(100,100),(90,198),(79,296),(71,322),(61,355),(51,395),(40,444)],
        [(100,100),(91,220),(80,358),(71,432),(63,484),(54,547),(44,625)],
        [(100,100),(149,176),(201,257),(242,276),(283,273),(325,274),(368,278),(412,286)],
    ]
    for path in paths:
        values = [dict(frame=100+i, pos=p, area=100) for i,p in enumerate(path)]
        assert impact(values, projected=True)['pos'] == path[3]
        assert impact(values[:5], projected=True) is None
        values[4]['frame'] += 1
        assert impact(values, projected=True) is None


def test_projected_mode_does_not_accept_smooth_flight_or_lateral_substitution():
    values = [dict(frame=i, pos=(100+i*8, 100+i*40), area=100) for i in range(8)]
    assert impact(values, projected=True) is None
    values = [dict(frame=i, pos=p, area=100) for i,p in enumerate(
        [(100,100),(90,198),(79,296),(200,322),(61,355),(51,395),(40,444)])]
    assert impact(values, projected=True) is None


def test_configured_camera_proximity_does_not_label_fault_as_confirmed_net_touch():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.far_serve_image_baseline = True
    analyzer._serve_net_contact_provisional = True
    reason = 'Serve bounce outside right service box'
    assert analyzer._serve_net_fault_reason(reason) == reason
    analyzer._serve_net_contact_provisional = False
    assert analyzer._serve_net_fault_reason(reason).startswith('Serve net then')


def test_straight_flight_keeps_existing_rebound_confirmation_threshold():
    positions = [(100,100),(119,215),(141,320),(162,356),(188,400),(216,456)]
    values = [dict(frame=100+i,pos=point,area=180) for i,point in enumerate(positions)]
    assert impact(values) is None


def test_diagonal_rebound_can_have_a_brief_horizontal_plateau():
    positions = [(100,100),(137,171),(173,249),(205,250),
                 (238,252),(270,258),(306,267)]
    values = [dict(frame=100+i, pos=point, area=130) for i,point in enumerate(positions)]
    assert impact(values[:6]) is None
    assert impact(values)['pos'] == (205,250)
    values[4]['pos'] = (400,252)
    assert impact(values) is None


def test_accelerating_net_crossing_is_not_a_net_touch():
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 100
    analyzer.point_start_frame_internal = 80
    analyzer._serve_phase_active = analyzer._awaiting_serve_bounce = True
    analyzer._near_serve_toss_in_progress = lambda point: False
    analyzer._is_night_session_config = lambda: True
    analyzer._serve_bounce_frame_limit = lambda: 45
    analyzer._serve_net_touch_active = lambda **kwargs: False
    analyzer._serve_net_zone_frames = 1
    analyzer._net_contact_geometry = lambda point: {
        'inside':True,'near_polygon':True,'near_top_tape':False,
        'signed_dist':20,'top_dist':31}
    analyzer._add_impact_marker = lambda *args, **kwargs: None
    analyzer.prev_motion = {'distance':53.3,'dy':46,'direction_deg':59.6}
    analyzer.last_motion = {'distance':60.9,'dy':53,'direction_deg':60.5}
    frame = np.zeros((1000,1000,3), np.uint8)
    assert not analyzer._mark_serve_net_contact_candidate((500,500), frame)
    assert analyzer._serve_net_contact_provisional
    # A real change from an upward approach to a downward net drop remains eligible.
    analyzer.prev_motion = {'distance':60,'dy':-55,'direction_deg':-80}
    analyzer.last_motion = {'distance':40,'dy':38,'direction_deg':80}
    assert analyzer._mark_serve_net_contact_candidate((500,500), frame)


def test_diagonal_service_uses_toss_half_before_outgoing_ball_crosses_centre():
    a = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    a._is_night_session_config = lambda: True
    a.serve_direction_dy = 1
    a._serve_phase_active = a._awaiting_serve_bounce = True
    a.point_start_frame_internal = 80
    a._last_racket_contact_frame = -1000000
    a._serve_bounce_frame_limit = lambda: 45
    a._player_point_zone = lambda point: None
    a.stuck_frame_count = 0
    a.net_area_y_max = 585
    a.initial_ball_position = (260,50)
    a._point_target_service_side = 'left'
    a._build_service_box_model = lambda frame: {'center_line':{'a':0,'b':300}}
    evaluated = []
    def evaluate(point, frame, side):
        evaluated.append(side)
        return True, None, {}
    a._evaluate_target_service_box_fresh = evaluate
    a._handle_ground_bounce_event = lambda *args: None
    a._pending_rally_end_reason = None
    a.ground_bounce_count = 0
    frame = np.zeros((1000,1000,3),np.uint8)
    for i, point in enumerate([(100,700),(137,771),(173,849),(205,850),
                               (238,852),(270,858),(306,867)]):
        a.frame_count = 100+i
        a.ball_size = 130
        a._observe_far_serve_perspective_bounce(point,frame)
    assert evaluated == ['right']
    assert a.last_ground_bounce_frame == 103


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


def test_collinear_service_line_fragments_can_bridge_a_shadow():
    frame = np.zeros((2160,3840,3), np.uint8)
    cv2.line(frame,(1300,870),(1750,885),(255,255,255),8)
    cv2.line(frame,(1850,889),(2350,906),(255,255,255),8)
    cv2.line(frame,(1920,550),(1900,900),(255,255,255),8)
    model = near_service_box(frame,492)
    assert model is not None
    assert abs(model['service_line']['a']*1934+model['service_line']['b']-892) < 12
    # Separate parallel markings are not evidence of the same stripe.
    frame[:] = 0
    cv2.line(frame,(1300,870),(1750,885),(255,255,255),8)
    cv2.line(frame,(1850,969),(2350,986),(255,255,255),8)
    cv2.line(frame,(1920,550),(1900,900),(255,255,255),8)
    assert near_service_box(frame,492) is None


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
