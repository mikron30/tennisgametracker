"""Recover a ball leaving a player from consecutive fresh image observations."""
import math

import cv2
import numpy as np


def changes_trajectory(path, incoming, *, racket_contact=False):
    """Clearing a player's image is not itself evidence of a racket hit."""
    if len(path) < 2 or not incoming:
        return False
    mx,my = incoming.get('dx',0),incoming.get('dy',0)
    elapsed = path[-1]['frame']-path[0]['frame']
    if elapsed <= 0:
        return False
    dx=(path[-1]['pos'][0]-path[0]['pos'][0])/elapsed
    dy=(path[-1]['pos'][1]-path[0]['pos'][1])/elapsed
    before,after=math.hypot(mx,my),math.hypot(dx,dy)
    lateral_racket_turn = bool(racket_contact and len(path)>=4 and
        abs(mx)>=2 and abs(dx)>=5 and mx*dx<0 and
        all(mx*(b['pos'][0]-a['pos'][0])<0 for a,b in zip(path,path[1:])))
    return bool(before >= 5 and after >= 5 and
                (mx*dx+my*dy < before*after*math.cos(math.radians(50)) or lateral_racket_turn or
                 after >= max(20,before*1.8)))


def candidates(frame, previous_gray, bounds, player_zone, *, near=False,
               include_yellow=False, minimum_area=None):
    if previous_gray is None or previous_gray.shape != frame.shape[:2]:
        return []
    x1, y1, x2, y2 = map(int, bounds)
    roi = frame[y1:y2, x1:x2]
    if not roi.size:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    previous = previous_gray[y1:y2, x1:x2]
    diff = cv2.absdiff(gray, previous)
    arrival = gray.astype(np.int16) - previous.astype(np.int16)
    colour = cv2.inRange(hsv, np.array([15, 20, 130]) if near else np.array([0, 0, 130]),
                         np.array([100, 255, 255]) if near else np.array([179, 100, 255]))
    if include_yellow:
        colour |= cv2.inRange(hsv,np.array([15,50,100]),np.array([100,255,255]))
    mask = cv2.bitwise_and(colour, np.where(arrival >= 18, 255, 0).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        _, _, w, h = cv2.boundingRect(contour)
        area_floor = float(
            minimum_area if minimum_area is not None else (20 if near else 3)
        )
        if not (area_floor <= area <= (600 if near else 160) and max(w, h) <= max(8, 4*min(w, h))):
            continue
        m = cv2.moments(contour)
        x, y = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
        patch = diff[max(0,y-6):y+7,max(0,x-6):x+7]
        if patch.mean() < 8 or patch.max() < 70:
            continue
        pos = (x+x1, y+y1)
        found.append(dict(pos=pos, area=area, hsv=hsv[y,x],
                          motion_mean=float(patch.mean()), motion_max=float(patch.max()),
                          player_zone=player_zone(pos)))
    return found


def departing_path(samples, anchor, player_center, *, near=False, falling=False,
                   start_radius=None, minimum_player_clearance=None, strict=False):
    require_clear_departure = near or strict
    required_samples = 4 if require_clear_departure else 3
    if len(samples) < required_samples:
        return None
    samples = samples[-required_samples:]
    if any(b['frame']-a['frame'] != 1 for a,b in zip(samples,samples[1:])):
        return None
    maximum_start_distance = float(
        start_radius if start_radius is not None else (500 if near else 240)
    )
    paths = [([c], math.dist(c['pos'], anchor) * (.05 if near else 0.0))
             for c in samples[0]['candidates']
             if math.dist(c['pos'],anchor) <= maximum_start_distance]
    for sample in samples[1:]:
        extended = []
        for path,cost in paths:
            for c in sample['candidates']:
                dx,dy = c['pos'][0]-path[-1]['pos'][0],c['pos'][1]-path[-1]['pos'][1]
                speed = math.hypot(dx,dy)
                if (not 5 <= speed <= (400 if near else 150) or
                        (near and dy > -8) or (falling and dy < 8)):
                    continue
                sizes = [p['area'] for p in path]+[c['area']]
                if max(sizes) > min(sizes)*5:
                    continue
                change = 0.
                if len(path) >= 2:
                    vx,vy = path[-1]['pos'][0]-path[-2]['pos'][0],path[-1]['pos'][1]-path[-2]['pos'][1]
                    change = math.hypot(dx-vx,dy-vy)
                    if change > max(12, math.hypot(vx,vy)*.55):
                        continue
                extended.append((path+[c],cost+change))
        beam_width = 512 if minimum_player_clearance is not None else 96
        paths = sorted(extended,key=lambda p:p[1])[:beam_width]
    paths = [(p,cost) for p,cost in paths if
             all(c['player_zone'] is None for c in p[-(2 if require_clear_departure else 1):]) and
             (not near or p[-1]['pos'][1] <= anchor[1]-180) and
             (not falling or p[-1]['pos'][1] >= anchor[1]+150) and
             math.dist(p[-1]['pos'],p[0]['pos']) >= (30 if require_clear_departure else 15) and
             math.dist(p[-1]['pos'],player_center) >=
             math.dist(p[0]['pos'],player_center)+(20 if require_clear_departure else 8) and
             (minimum_player_clearance is None or
              math.dist(p[-1]['pos'], player_center) >=
              float(minimum_player_clearance))]
    if not paths:
        return None
    path = min(paths,key=lambda p:p[1])[0]
    return [dict(c,frame=s['frame']) for c,s in zip(path,samples)]


def coherent_visible_path(samples, anchor, *, max_area=100, max_saturation=80):
    """Return a conservative three-frame ball path through a player ROI.

    Player boxes are deliberately generous, so a visible ball can remain
    labelled as body/feet while approaching a racket.  This path is only for
    preserving that measured flight; it is not evidence of racket contact.
    """
    if len(samples) < 3:
        return None
    samples = samples[-3:]
    if any(b['frame'] - a['frame'] != 1 for a, b in zip(samples, samples[1:])):
        return None
    paths = []
    for c in samples[0]['candidates']:
        h, s, v = map(int, c.get('hsv', (0, 0, 0)))
        if (math.dist(c['pos'], anchor) <= 130 and 3 <= c['area'] <= max_area and
                v >= 180 and s <= max_saturation):
            paths.append(([c], math.dist(c['pos'], anchor) * .08))
    for sample in samples[1:]:
        extended = []
        for path, cost in paths:
            for c in sample['candidates']:
                h, s, v = map(int, c.get('hsv', (0, 0, 0)))
                if not (3 <= c['area'] <= max_area and v >= 180 and s <= max_saturation):
                    continue
                dx = c['pos'][0] - path[-1]['pos'][0]
                dy = c['pos'][1] - path[-1]['pos'][1]
                speed = math.hypot(dx, dy)
                if not 3 <= speed <= 45:
                    continue
                sizes = [p['area'] for p in path] + [c['area']]
                if max(sizes) > min(sizes) * 5:
                    continue
                change = 0.0
                if len(path) >= 2:
                    vx = path[-1]['pos'][0] - path[-2]['pos'][0]
                    vy = path[-1]['pos'][1] - path[-2]['pos'][1]
                    change = math.hypot(dx - vx, dy - vy)
                    if change > max(7, math.hypot(vx, vy) * .4):
                        continue
                extended.append((path + [c], cost + change))
        paths = sorted(extended, key=lambda item: item[1])[:64]
    paths = [(path, cost) for path, cost in paths
             if math.dist(path[0]['pos'], path[-1]['pos']) >= 8]
    if not paths:
        return None
    path = min(paths, key=lambda item: item[1])[0]
    return [dict(c, frame=s['frame']) for c, s in zip(path, samples)]
