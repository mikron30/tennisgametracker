"""Local recovery from measured ball motion and fresh, colored image pixels."""
import math

import cv2
import numpy as np


def find_local_ball(frame, previous_gray, position, motion, previous_area, filters, *, minimum_speed=5, maximum_speed=90, contact_window=False, minimum_saturation=45, contact_radius_limit=180, flexible_contact=False, player_zone=None, small_net_ball=False, outgoing_handoff=False, rising_contact=False, lateral_contact=False):
    if previous_gray is None or position is None or not motion or previous_area is None:
        return None
    if previous_gray.shape != frame.shape[:2] or previous_area < (3 if small_net_ball else 8):
        return None
    dx, dy = float(motion.get('dx', 0)), float(motion.get('dy', 0))
    speed = math.hypot(dx, dy)
    if not max(2, minimum_speed) <= speed <= maximum_speed:
        return None
    if outgoing_handoff and dy >= -8:
        return None
    predicted = (position[0] + dx, position[1] + dy)
    radius = int(max(28, min(65, speed * 1.5)))
    if contact_window:
        radius = int(max(90, min(contact_radius_limit, speed*(3 if flexible_contact else 2))))
        if rising_contact:
            radius = max(radius, min(contact_radius_limit,180))
    if outgoing_handoff:
        radius = int(max(90, min(300, speed * 2.5)))
    height, width = frame.shape[:2]
    # Include the previous ball's neighborhood so a real bounce may reverse
    # direction without being forced onto an extrapolated downward lane.
    x1 = max(0, int(min(predicted[0],position[0]))-radius)
    x2 = min(width, int(max(predicted[0],position[0]))+radius+1)
    y1 = max(0, int(min(predicted[1],position[1]))-radius)
    y2 = min(height, int(max(predicted[1],position[1]))+radius+1)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, previous_gray[y1:y2,x1:x2])
    arrival = gray.astype(np.int16) - previous_gray[y1:y2,x1:x2].astype(np.int16)
    color = np.zeros(gray.shape, np.uint8)
    for lower, upper in filters:
        if lower is not None and upper is not None:
            color |= cv2.inRange(hsv, lower, upper)
    # Absolute differencing also exposes the old ball's departing footprint.
    # Only newly brighter colored pixels can seed the current ball contour.
    arrived = arrival >= 18
    if minimum_saturation < 45:
        previous_brightness = previous_gray[y1:y2, x1:x2]
        # A yellow ball crossing a white line darkens the image. Require fresh
        # colour as well as change there instead of demanding a brighter pixel.
        arrived |= ((diff >= 18) & (previous_brightness >= 170) &
                    (hsv[:, :, 1] >= max(10, minimum_saturation)))
    mask = cv2.bitwise_and(color, np.where(arrived, 255, 0).astype(np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not max(3 if small_net_ball else 6, previous_area * (.9 if contact_window and not flexible_contact else .12)) <= area <= min(1500 if maximum_speed > 90 else 450, previous_area * (10 if small_net_ball else 2.5)):
            continue
        _,_,w,h=cv2.boundingRect(contour)
        if max(w,h) > max(8, min(w,h)*(4 if maximum_speed > 90 or small_net_ball else 3)):
            continue
        m=cv2.moments(contour)
        if not m['m00']:
            continue
        cx,cy=int(m['m10']/m['m00'])+x1,int(m['m01']/m['m00'])+y1
        distance=math.hypot(cx-predicted[0],cy-predicted[1])
        previous_distance=math.hypot(cx-position[0],cy-position[1])
        if outgoing_handoff and (
                area < max(20, previous_area * .35) or
                previous_distance > min(400, speed * 3.5) or
                dx * (cx-position[0]) + dy * (cy-position[1]) < speed * previous_distance * .85 or
                (player_zone is not None and player_zone((cx, cy)) is not None)):
            continue
        if small_net_ball and (area > 80 or previous_distance > 35):
            continue
        # A strike partway through the frame produces a shorter first return
        # step; still exclude minor upward motion on a fast incoming racket.
        departure = bool(contact_window and (dy >= 12 or rising_contact) and
                         cy <= position[1] - (max(12, speed * .3) if flexible_contact else 12))
        if lateral_contact and departure:
            # An outside bounce keeps lateral travel. A stretched racket hit
            # can sharply stop it while sending the ball upward. The expanded
            # reach alone cannot justify a large sideways jump onto a player.
            departure = bool(abs(dx) >= 30 and
                abs(cx-position[0]) <= abs(dx)*.35)
        if contact_window and flexible_contact:
            if departure:
                if (area < max(20, previous_area * .4) or
                        (player_zone is not None and player_zone((cx, cy)) in
                         ('player_head_hat', 'player_shoes')) or
                        (player_zone is not None and player_zone((cx, cy)) == 'player_body' and
                         position[1] - cy < abs(cx-position[0]) * .65)):
                    continue
            elif (distance > max(28, speed * .75) or previous_distance > max(30, speed * 1.8) or
                  (dx * (cx-position[0]) + dy * (cy-position[1])) < speed * previous_distance * .5):
                continue
        if (min(distance,previous_distance) > radius or
                math.hypot(cx-position[0],cy-position[1]) < max(4, speed*.25) or
                int(hsv[cy-y1,cx-x1,1]) < minimum_saturation):
            continue
        patch=diff[max(0,cy-y1-8):cy-y1+9,max(0,cx-x1-8):cx-x1+9]
        mean, peak=float(patch.mean()),float(patch.max())
        if mean < (18 if contact_window or outgoing_handoff else 6) or peak < (70 if contact_window or outgoing_handoff else 35):
            continue
        score=min(distance,previous_distance+speed*.5) + abs(math.log(area/previous_area))*20
        candidates.append((score, {'pos':(cx,cy),'area':area,'hsv':hsv[cy-y1,cx-x1],
                                  'motion_mean':mean,'motion_max':peak,
                                  'verified_contact_departure': departure}))
    if flexible_contact:
        departures = [item for item in candidates if item[1]['verified_contact_departure']]
        if departures:
            if player_zone is not None:
                clear = [item for item in departures if player_zone(item[1]['pos']) is None]
                if clear:
                    departures = clear
            candidates = departures
    return min(candidates,key=lambda item:item[0])[1] if candidates else None
