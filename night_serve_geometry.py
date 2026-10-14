"""Image-based baseline estimates for the fixed night camera."""
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


def far_baseline(frame):
    """Locate the bright back boundary before the far service line."""
    height, width = frame.shape[:2]
    scale = 960.0 / width
    small = cv2.resize(frame, (960, int(round(height * scale))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    contrast = cv2.subtract(gray, cv2.medianBlur(gray, 15))
    white = ((contrast >= 12) & (hsv[:, :, 1] <= 90) &
             (hsv[:, :, 2] >= 100)).astype(np.uint8) * 255
    sh, sw = white.shape
    top, bottom = int(sh * .105), int(sh * .14)
    left, right = int(sw * .30), int(sw * .75)
    lines = cv2.HoughLinesP(white[top:bottom, left:right], 1, np.pi / 180,
                           threshold=20, minLineLength=int(sw * .06), maxLineGap=20)
    candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if abs(x2 - x1) < sw * .06 or abs(y2 - y1) > abs(x2 - x1) * .10:
                continue
            candidates.extend([(x1 + left, y1 + top), (x2 + left, y2 + top)])
    if not candidates:
        return None
    points = np.asarray(candidates, dtype=float)
    for _ in range(3):
        if len(points) < 4 or np.ptp(points[:, 0]) < sw * .18:
            return None
        a, b = np.polyfit(points[:, 0], points[:, 1], 1)
        errors = np.abs(points[:, 1] - (a * points[:, 0] + b))
        points = points[errors <= max(2.5, 3 * np.median(errors))]
    if len(points) < 4 or np.ptp(points[:, 0]) < sw * .18:
        return None
    a, b = np.polyfit(points[:, 0], points[:, 1], 1)
    return {'a': a, 'b': b / scale, 'source': 'night image far baseline'}


def far_service_box(frame, net_y):
    """Read the far service line without mistaking the net tape for it."""
    height, width = frame.shape[:2]
    scale = 960.0 / width
    small = cv2.resize(frame, (960, int(round(height * scale))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([179, 90, 255]))
    mask = white.copy()
    mask[:int(height*.14*scale)] = 0
    mask[int(height*.18*scale):] = 0
    mask[:, :int(960*.25)] = 0
    mask[:, int(960*.75):] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, 20, minLineLength=100, maxLineGap=12)
    horizontal = []
    if lines is None:
        return None
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2-x1), float(y2-y1)
        if abs(dx) >= 120 and abs(dy) <= abs(dx)*.08:
            a = dy/dx
            horizontal.append({'a': a, 'b': (y1-a*x1)/scale, 'length': abs(dx)/scale})
    if not horizontal:
        return None
    service = min(horizontal, key=lambda line: line['a']*(width*.5)+line['b'])
    service_y = service['a']*(width*.5)+service['b']
    points = []
    left, right = int(960*.45), int(960*.55)
    for y in range(int((service_y+12)*scale), int(min(net_y-height*.04, height*.19)*scale)):
        xs = np.flatnonzero(white[y, left:right])
        if len(xs):
            x = left+xs[np.argmin(abs(left+xs-480))]
            points.append((y/scale, x/scale))
    if len(points) < 6:
        return None
    a, b = np.polyfit(*np.asarray(points).T, 1)
    if abs(a) > .15:
        a, b = 0.0, float(np.median(np.asarray(points)[:, 1]))
    return {'net_y': float(net_y), 'service_line': service,
            'center_line': {'a': float(a), 'b': float(b)},
            'service_margin': max(8, width*.003), 'net_margin': max(24, width*.008)}


def near_service_box(frame, net_y):
    """Read the receiver's service line and actual centre marking."""
    height, width = frame.shape[:2]
    scale = 960.0 / width
    small = cv2.resize(frame, (960, int(round(height * scale))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 70, 255]))
    top = int((net_y + height * .05) * scale)
    bottom = int(height * .52 * scale)
    mask[:top] = 0
    mask[bottom:] = 0
    mask[:, :int(960 * .2)] = 0
    mask[:, int(960 * .8):] = 0
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 35,
                           minLineLength=60, maxLineGap=8)
    horizontal, central = [], []
    if lines is None:
        return None
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) >= 180 and abs(dy) <= abs(dx) * .08:
            a = float(dy) / dx
            horizontal.append({'a': a, 'b': (y1 - a * x1) / scale,
                               'length': abs(dx) / scale})
        if (abs(dy) >= 50 and abs(dx) <= abs(dy) * .12 and
                960 * .43 <= (x1 + x2) / 2 <= 960 * .57):
            a = float(dx) / dy
            central.append({'a': a, 'b': (x1 - a * y1) / scale,
                            'length': abs(dy) / scale})
    if not horizontal or not central:
        return None
    service = max(horizontal, key=lambda line: line['length'])
    center = max(central, key=lambda line: line['length'])
    return {'net_y': float(net_y), 'service_line': service,
            'center_line': center, 'service_margin': max(8, width * .003),
            'net_margin': max(24, width * .008)}
