"""Measured net impacts from ball motion and surrounding image evidence."""
import math
import cv2
import numpy as np


def net_mesh_reversal(samples, polygon, previous_gray, frame):
    """Confirm a low return turning down in the net while the mesh deflects."""
    if len(samples) < 5 or polygon is None or previous_gray is None:
        return False
    if previous_gray.shape != frame.shape[:2]:
        return False
    samples = samples[-5:]
    if any(b['frame'] - a['frame'] != 1 for a, b in zip(samples, samples[1:])):
        return False
    points = [sample['pos'] for sample in samples]
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:])]
    if not (all(dy <= -4 for _, dy in steps[:2]) and
            all(dy >= 1 for _, dy in steps[2:]) and
            points[-1][1] - points[2][1] >= 6 and
            all(math.hypot(dx,dy) <= 45 for dx,dy in steps) and
            math.hypot(*steps[-1]) <= math.hypot(*steps[0]) * .7):
        return False
    hull = cv2.convexHull(polygon.astype(np.float32))
    if any(cv2.pointPolygonTest(hull, tuple(map(float,p)), False) < 0 for p in points):
        return False
    apex = points[2]
    top = polygon.reshape(-1,2)[:3]
    top_y = None
    for a,b in zip(top,top[1:]):
        if min(a[0],b[0]) <= apex[0] <= max(a[0],b[0]) and b[0] != a[0]:
            top_y = a[1] + (apex[0]-a[0]) * (b[1]-a[1]) / (b[0]-a[0])
    if top_y is None or apex[1] < top_y + 20:
        return False
    return net_mesh_deflection(polygon, previous_gray, frame, points[-2:])


def net_mesh_deflection(polygon, previous_gray, frame, points):
    """Require surrounding mesh movement, excluding the ball footprints."""
    if polygon is None or previous_gray is None or not points or previous_gray.shape != frame.shape[:2]:
        return False
    hull = cv2.convexHull(polygon.astype(np.float32))
    if cv2.pointPolygonTest(hull,tuple(map(float,points[-1])),False) < 0:
        return False
    x,y = map(int,points[-1])
    x1,x2 = max(0,x-50),min(frame.shape[1],x+51)
    y1,y2 = max(0,y-45),min(frame.shape[0],y+46)
    mask = np.zeros((y2-y1,x2-x1),np.uint8)
    local_hull = (hull.reshape(-1,2) - np.array([x1,y1])).astype(np.int32)
    cv2.fillConvexPoly(mask,local_hull,255)
    # Exclude both ball footprints; require motion in the surrounding mesh.
    for p in points:
        cv2.circle(mask,(int(p[0])-x1,int(p[1])-y1),15,0,-1)
    gray = cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(gray,previous_gray[y1:y2,x1:x2])[mask != 0]
    return bool(motion.size and motion.mean() >= 8 and motion.max() >= 70 and
                np.count_nonzero(motion >= 30) >= 80)


def net_drop_bounce(samples, polygon, contact, frame_height):
    if len(samples) < 5 or polygon is None or contact is None:
        return None
    samples = samples[-5:]
    if any(b['frame'] - a['frame'] != 1 for a, b in zip(samples, samples[1:])):
        return None
    points = [sample['pos'] for sample in samples]
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    if not (all(dy >= 1 for _, dy in steps[:2]) and
            all(dy <= -2 for _, dy in steps[2:]) and
            points[2][1] - points[-1][1] >= 5 and
            points[2][1] - points[0][1] >= 10 and
            all(math.hypot(dx, dy) <= 35 for dx, dy in steps) and
            max(p[0] for p in points) - min(p[0] for p in points) <= 35):
        return None
    landing = points[2]
    if landing[1] - contact[1] < 30 or abs(landing[0] - contact[0]) > 80:
        return None
    # The calibrated polygon's final three vertices describe its bottom edge.
    bottom = polygon.reshape(-1, 2)[3:]
    for first, last in zip(bottom, bottom[1:]):
        if min(first[0], last[0]) <= landing[0] <= max(first[0], last[0]) and last[0] != first[0]:
            y = first[1] + (landing[0] - first[0]) * (last[1] - first[1]) / (last[0] - first[0])
            if abs(landing[1] - y) <= max(20, frame_height * .012):
                return tuple(landing)
    return None
