"""Near baseline estimate for the fixed night camera, including lens curvature."""
import cv2
import numpy as np


def near_baseline(frame):
    h, w = frame.shape[:2]
    scale = 960.0 / w
    small = cv2.resize(frame, (960, int(round(h * scale))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 210]), np.array([179, 65, 255]))
    sh, sw = mask.shape
    # The near baseline is the lowest bright court marking in this band.
    top, bottom = int(sh * .65), int(sh * .86)
    points = []
    for x in range(int(sw * .18), int(sw * .82), 4):
        ys = np.flatnonzero(mask[top:bottom, x])
        if len(ys):
            points.append((x / scale, (top + ys[-1]) / scale))
    if len(points) < 40:
        return None
    points = np.asarray(points)
    for _ in range(3):
        coefficients = np.polyfit(points[:, 0], points[:, 1], 2)
        error = np.abs(np.polyval(coefficients, points[:, 0]) - points[:, 1])
        points = points[error <= max(12.0 / scale, 3 * np.median(error))]
        if len(points) < 40:
            return None
    if np.ptp(points[:, 0]) < w * .35:
        return None
    coefficients = np.polyfit(points[:, 0], points[:, 1], 2)
    return {'curve': coefficients.tolist(), 'a': 0.0,
            'b': float(np.polyval(coefficients, w / 2)),
            'source': 'night image near baseline'}
