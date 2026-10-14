import cv2
import numpy as np
from night_serve_geometry import far_service_box


def court():
    frame = np.full((2160, 3840, 3), 100, np.uint8)
    cv2.line(frame, (1450, 345), (2650, 367), (240, 240, 240), 6)
    cv2.line(frame, (1930, 355), (1933, 412), (240, 240, 240), 6)
    # A long net reflection in the search band must not replace the service line.
    cv2.line(frame, (1400, 386), (2650, 377), (250, 250, 250), 5)
    cv2.line(frame, (1150, 445), (2900, 470), (255, 255, 255), 8)
    return frame


def test_service_line_is_not_the_net_tape_or_its_reflection():
    model = far_service_box(court(), 492.5)
    assert model is not None
    line = model['service_line']
    assert abs(line['a']*1600+line['b']-347.75) < 8
    center = model['center_line']
    assert abs(center['a']*360+center['b']-1930) < 20


def test_missing_court_markings_are_not_invented():
    assert far_service_box(np.zeros((2160, 3840, 3), np.uint8), 492.5) is None
