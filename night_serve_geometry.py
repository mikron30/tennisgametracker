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


def far_service_box(frame, net_y, *, shadow=False):
    """Read the far service line without mistaking the net tape for it."""
    height, width = frame.shape[:2]
    scale = 960.0 / width
    small = cv2.resize(frame, (960, int(round(height * scale))))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    if shadow:
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        contrast=cv2.subtract(gray,cv2.medianBlur(gray,15))
        white=((contrast>=12)&(hsv[:,:,1]<=90)&(hsv[:,:,2]>=100)).astype(np.uint8)*255
    else:
        white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([179, 90, 255]))
    def missing():
        # Read dim paint by local contrast in the same calibrated court band;
        # never fall back to an unrelated horizontal net-tape line.
        return None if shadow else far_service_box(frame,net_y,shadow=True)
    mask = white.copy()
    mask[:int(height*.14*scale)] = 0
    mask[int(height*.18*scale):] = 0
    mask[:, :int(960*.25)] = 0
    mask[:, int(960*.75):] = 0
    horizontal = []
    for minimum in (100,60):
        # Prefer the original long support. Under occlusion, a shorter clear
        # stripe is sufficient only together with the centre marking below.
        lines=cv2.HoughLinesP(mask,1,np.pi/180,20,minLineLength=minimum,maxLineGap=12)
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0,:]:
                dx,dy=float(x2-x1),float(y2-y1)
                if abs(dx)>=minimum and abs(dy)<=abs(dx)*.08:
                    slope=dy/dx
                    horizontal.append({'a':slope,'b':(y1-slope*x1)/scale,
                                       'length':abs(dx)/scale})
        if horizontal:
            break
    if not horizontal:
        return missing()
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
        return missing()
    a, b = np.polyfit(*np.asarray(points).T, 1)
    if abs(a) > .15:
        a, b = 0.0, float(np.median(np.asarray(points)[:, 1]))
    return {'net_y': float(net_y), 'service_line': service,
            'center_line': {'a': float(a), 'b': float(b)},
            'center_margin': max(4, width*.004),
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
    if not horizontal:
        # Shadows can split the same service stripe into shorter segments.
        # Join only collinear pieces spanning a substantial part of the court.
        fragments = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = float(x2-x1), float(y2-y1)
            if abs(dx) >= 60 and abs(dy) <= abs(dx)*.08:
                slope = dy/dx
                fragments.append((slope, y1+slope*(480-x1), (x1,y1), (x2,y2)))
        for slope, middle, _, _ in fragments:
            group = [item for item in fragments
                     if abs(item[0]-slope) <= .025 and abs(item[1]-middle) <= 5]
            points = np.asarray([point for item in group for point in item[2:]])
            if len(group) >= 2 and np.ptp(points[:,0]) >= 180:
                a, b = np.polyfit(points[:,0], points[:,1], 1)
                horizontal.append({'a':float(a), 'b':float(b/scale),
                                   'length':float(np.ptp(points[:,0])/scale)})
    if not horizontal or not central:
        return None
    service = max(horizontal, key=lambda line: line['length'])
    center = max(central, key=lambda line: line['length'])
    return {'net_y': float(net_y), 'service_line': service,
            'center_line': center, 'service_margin': max(8, width * .003),
            'net_margin': max(24, width * .008)}


def local_sideline_x(frame, y, predicted_x, side):
    """Read nearby sideline paint instead of extrapolating a distant segment."""
    h,w = frame.shape[:2]
    rx,ry = max(70,int(w*.025)),max(80,int(h*.06))
    x1,x2=max(0,int(predicted_x)-rx),min(w,int(predicted_x)+rx+1)
    y1,y2=max(0,int(y)-ry),min(h,int(y)+ry+1)
    if x2<=x1 or y2<=y1:
        return None
    hsv=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,0,180]),np.array([179,95,255]))
    lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=30,
        minLineLength=max(45,int(h*.025)),maxLineGap=10)
    if lines is None:
        return None
    candidates=[]
    for xa,ya,xb,yb in lines[:,0,:]:
        dy=float(yb-ya)
        if abs(dy)<40:
            continue
        slope=float(xb-xa)/dy
        if not (.3<=abs(slope)<=1.5 and (slope<0)==(side=='left')):
            continue
        if not min(ya,yb)+y1-15<=y<=max(ya,yb)+y1+15:
            continue
        value=xa+x1+slope*(y-ya-y1)
        if abs(value-predicted_x)<=rx:
            candidates.append(value)
    if not candidates:
        return None
    # When both singles and doubles paint are visible, the inner stripe is
    # the singles boundary. Restricting the search keeps other courts out.
    return float(max(candidates) if side=='left' else min(candidates))
