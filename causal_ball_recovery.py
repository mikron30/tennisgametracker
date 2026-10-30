"""Local recovery from measured ball motion and fresh, colored image pixels."""
import math

import cv2
import numpy as np


def shallow_rebound(history, contact_frame, player_corridor):
    """A measured shallow rebound needs descent followed by two flat samples."""
    entries=history[-3:]
    if len(entries)!=3 or entries[0].get('prev_pos') is None:
        return None
    if any(b['frame']-a['frame']!=1 for a,b in zip(entries,entries[1:])):
        return None
    a,b,c=entries
    if a['frame']<=contact_frame or any(player_corridor(e['pos']) for e in entries):
        return None
    dx=a['pos'][0]-a['prev_pos'][0];dy=a['pos'][1]-a['prev_pos'][1]
    vx=b['pos'][0]-a['pos'][0];vy=b['pos'][1]-a['pos'][1]
    wx=c['pos'][0]-b['pos'][0];wy=c['pos'][1]-b['pos'][1]
    if not (dy>=18 and abs(dx)>=8 and dy>=abs(dx)*.6 and
            -4<vy<=0 and -4<wy<=4 and
            dx*vx>0 and vx*wx>0 and
            .45<=abs(vx/dx)<=1.1 and .7<=abs(wx/vx)<=1.3):
        return None
    return dict(frame=a['frame'],origin=tuple(a['pos']),incoming_dx=dx,incoming_dy=dy,
                incoming_dist=math.hypot(dx,dy),expected=tuple(c['pos']))


def matches_ground_rebound(position, motion, candidate, expected):
    """An upward sample alone cannot distinguish a bounce from a racket hit.

    Recognize a dissipative rebound in the measured incoming lateral lane.
    The caller must independently establish an earlier bounce and a continuous
    descending approach; this is not a standalone contact or point-end rule.
    """
    dx, dy = float(motion.get('dx', 0)), float(motion.get('dy', 0))
    ox, oy = candidate['pos'][0]-position[0], candidate['pos'][1]-position[1]
    return bool(abs(dx) >= 8 and dy >= 20 and dx*ox > 0 and
                .45 <= abs(ox/dx) <= 1.1 and .5*dy <= -oy <= 1.25*dy and
                math.hypot(ox, oy) <= math.hypot(dx, dy)*1.1 and
                math.dist(candidate['pos'], expected) <= max(20, math.hypot(dx, dy)*.4))


def find_local_ball(frame, previous_gray, position, motion, previous_area, filters, *, minimum_speed=5, maximum_speed=90, contact_window=False, minimum_saturation=45, contact_radius_limit=180, flexible_contact=False, player_zone=None, small_net_ball=False, outgoing_handoff=False, rising_contact=False, lateral_contact=False, handoff_reference_speed=None, serve_rebound_search=False, serve_contact_search=False, separate_profiles=False, perspective_rebound_search=False, partial_contact_search=False):
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
    handoff_speed = max(speed, min(400, float(handoff_reference_speed or 0)))
    predicted = (position[0] + dx, position[1] + dy)
    radius = int(max(28, min(65, speed * 1.5)))
    if serve_rebound_search:
        # Perspective can flatten the incoming serve immediately before its
        # bounce. The first rising observation can leave the slow projection's
        # 28px crop; retain fresh colour/shape/motion checks in a bounded ROI.
        radius = max(radius, 45)
    if contact_window:
        # A racket can accelerate a slow incoming ball beyond its old lane.
        # Keep the fresh configured-contact search bounded but wide enough
        # to see the actual departing ball instead of a closer body fragment.
        contact_floor = min(180, contact_radius_limit) if flexible_contact else 90
        radius = int(max(contact_floor, min(contact_radius_limit, speed*(3 if flexible_contact else 2))))
        if rising_contact:
            radius = max(radius, min(contact_radius_limit,180))
    if outgoing_handoff:
        radius = int(max(90, min(300, handoff_speed * 2.5)))
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
    if serve_rebound_search and previous_area <= 12:
        # A distant ball crossing white paint can leave two fresh slivers.
        # Join only their few-pixel gap; the caller requires a recent measured
        # incoming serve, and the usual motion/shape/area checks still apply.
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,3),np.uint8))
        fragments, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) + list(fragments)
    if flexible_contact:
        # The broad flight mask can join a yellow ball to white racket strings
        # or court paint. Measure its moving yellow pixels separately so the
        # merged contour's pale centroid cannot replace the ball at contact
        # or during its subsequent flight across a painted court line.
        yellow = cv2.inRange(hsv, np.array([20,50,130],np.uint8), np.array([85,255,255],np.uint8))
        yellow = cv2.bitwise_and(yellow, np.where(arrived,255,0).astype(np.uint8))
        yellow = cv2.morphologyEx(yellow,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8))
        yellow_contours,_ = cv2.findContours(yellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours) + list(yellow_contours)
    candidates=[]
    if serve_contact_search or separate_profiles or partial_contact_search:
        # Combining colour profiles can bridge the ball to racket strings.
        # Retain separately segmented, fresh blobs as well as the union mask.
        for lower, upper in filters:
            if lower is None or upper is None:
                continue
            isolated = cv2.bitwise_and(cv2.inRange(hsv, lower, upper),
                                      np.where(arrived,255,0).astype(np.uint8))
            extra,_ = cv2.findContours(isolated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours) + list(extra)
    for contour in contours:
        area=cv2.contourArea(contour)
        if not max(3 if small_net_ball or serve_rebound_search else 6, previous_area * (.9 if contact_window and not flexible_contact else .12)) <= area <= min(1500 if maximum_speed > 90 else 450, (80 if serve_rebound_search and previous_area<=12 else previous_area * (10 if small_net_ball else 4 if serve_contact_search else 2.5))):
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
                previous_distance > min(400, max(90, speed * 3.5, handoff_speed * 3.5 + 30)) or
                dx * (cx-position[0]) + dy * (cy-position[1]) < speed * previous_distance * .85 or
                        (player_zone is not None and player_zone((cx, cy)) in
                         ('player_head_hat', 'player_shoes', 'racket_fragment'))):
            continue
        if small_net_ball and (area > 80 or previous_distance > 35):
            continue
        # A strike partway through the frame produces a shorter first return
        # step; still exclude minor upward motion on a fast incoming racket.
        departure = bool(contact_window and (dy >= 12 or rising_contact) and
                         cy <= position[1] - (max(12, speed * .3) if flexible_contact else 12))
        if departure and rising_contact and dy < 12:
            # A ball already rising from a ground bounce is not another hit
            # merely because it clears the player. Require a changed flight.
            out_dx,out_dy=cx-position[0],cy-position[1]
            out_speed=math.hypot(out_dx,out_dy)
            departure = bool(dx*out_dx+dy*out_dy < speed*out_speed*math.cos(math.radians(50)) or
                             out_speed >= max(20,speed*1.8))
        if lateral_contact and departure:
            # An outside bounce keeps lateral travel. A stretched racket hit
            # can sharply stop it while sending the ball upward. The expanded
            # reach alone cannot justify a large sideways jump onto a player.
            departure = bool(abs(dx) >= 30 and
                abs(cx-position[0]) <= abs(dx)*.35)
        # A perspective bounce can flatten before rising, while retaining its
        # lateral lane and ball area. Preserve that flight without inventing a hit.
        gentle_rebound = (
            contact_window and flexible_contact and 0 <= dy < 12 and
            dx * (cx-position[0]) > 0 and .45 <= abs((cx-position[0])/dx) <= 1.5 and
            cy <= position[1]-4 and previous_distance <= max(30, speed*1.5) and
            area >= previous_area*.65 and 20 <= int(hsv[cy-y1,cx-x1,0]) <= 85 and
            int(hsv[cy-y1,cx-x1,1]) >= 50 and int(hsv[cy-y1,cx-x1,2]) >= 130)
        partial_strike = bool((serve_contact_search or partial_contact_search) and
            abs(cy-position[1]) <= 12 and 8 <= abs(cx-position[0]) <= 70 and
            previous_area*(.3 if partial_contact_search else .8) <= area <= previous_area*4 and
            20 <= int(hsv[cy-y1,cx-x1,0]) <= 85 and
            int(hsv[cy-y1,cx-x1,1]) >= 10 and int(hsv[cy-y1,cx-x1,2]) >= 170)
        arriving_slowdown = bool(perspective_rebound_search and dy >= 20 and abs(dx) >= 8 and
            dx*(cx-position[0]) > 0 and .6 <= abs((cx-position[0])/dx) <= 1.4 and
            0 <= cy-position[1] <= dy*.35 and area >= previous_area*.6 and
            20 <= int(hsv[cy-y1,cx-x1,0]) <= 85 and
            int(hsv[cy-y1,cx-x1,1]) >= 50 and int(hsv[cy-y1,cx-x1,2]) >= 170 and
            (player_zone is None or player_zone((cx,cy)) is None))
        perspective_ground_turn = bool(perspective_rebound_search and dy>=20 and
            -80<=cy-position[1]<=-4 and abs(cx-position[0])<=max(25,abs(dx)*1.4) and
            area>=previous_area*.6 and
            20<=int(hsv[cy-y1,cx-x1,0])<=85 and
            int(hsv[cy-y1,cx-x1,1])>=50 and int(hsv[cy-y1,cx-x1,2])>=170 and
            (player_zone is None or player_zone((cx,cy)) is None))
        if perspective_ground_turn:
            departure=False
        partial_serve_departure = bool(departure and serve_contact_search and
            area >= max(50,previous_area*.35) and
            20 <= int(hsv[cy-y1,cx-x1,0]) <= 85 and
            int(hsv[cy-y1,cx-x1,1]) >= 20 and int(hsv[cy-y1,cx-x1,2]) >= 170)
        if contact_window and flexible_contact:
            if departure:
                if (area < max(20, previous_area * (.35 if partial_serve_departure else .4)) or
                        (player_zone is not None and player_zone((cx, cy)) in
                         ('player_head_hat', 'player_shoes')) or
                        (player_zone is not None and player_zone((cx, cy)) == 'player_body' and
                         position[1] - cy < abs(cx-position[0]) * .65)):
                    continue
            elif not partial_strike and not arriving_slowdown and not perspective_ground_turn and (distance > max(28, speed * .75) or previous_distance > max(30, speed * 1.8) or
                  ((dx * (cx-position[0]) + dy * (cy-position[1])) < speed * previous_distance * .5 and
                   not gentle_rebound and not partial_strike)):
                continue
        if (min(distance,previous_distance) > radius or
                math.hypot(cx-position[0],cy-position[1]) < (1 if small_net_ball else max(4, speed*(0 if perspective_ground_turn else .1 if arriving_slowdown else .25))) or
                int(hsv[cy-y1,cx-x1,1]) < minimum_saturation):
            continue
        patch=diff[max(0,cy-y1-8):cy-y1+9,max(0,cx-x1-8):cx-x1+9]
        mean, peak=float(patch.mean()),float(patch.max())
        partial_far_rebound = bool(serve_rebound_search and area<=12 and
            not contact_window and not outgoing_handoff and mean>=5 and peak>=60)
        if (mean < (18 if contact_window or outgoing_handoff else 6) or
                peak < (70 if contact_window or outgoing_handoff else 35)) and not partial_far_rebound:
            continue
        if partial_strike and (mean < 30 or peak < 90):
            continue
        if partial_serve_departure and area < previous_area*.4 and (mean < 30 or peak < 90):
            continue
        if (outgoing_handoff and player_zone is not None and player_zone((cx, cy)) == 'player_body' and
                not (mean >= 40 and peak >= 100 and area >= previous_area*.65 and
                     20 <= int(hsv[cy-y1,cx-x1,0]) <= 85 and
                     int(hsv[cy-y1,cx-x1,1]) >= 20 and int(hsv[cy-y1,cx-x1,2]) >= 170)):
            continue
        score=min(distance,previous_distance+speed*.5) + abs(math.log(area/previous_area))*20
        candidates.append((score, {'pos':(cx,cy),'area':area,'hsv':hsv[cy-y1,cx-x1],
                                  'motion_mean':mean,'motion_max':peak,
                                  'partial_serve_strike': partial_strike,
                                  'partial_return_contact': partial_strike and partial_contact_search,
                                  'verified_contact_departure': departure}))
    partials = [item for item in candidates if item[1]['partial_serve_strike']]
    if partials:
        return min(partials,key=lambda item:item[0])[1]
    if flexible_contact:
        departures = [item for item in candidates if item[1]['verified_contact_departure']]
        if departures:
            # A player fragment reversing direction cannot displace a clearly
            # measured ball that is still arriving along its existing lane.
            incoming=[]
            clear_departure = any(
                20 <= int(item[1]['hsv'][0]) <= 85 and int(item[1]['hsv'][1]) >= 50 and
                int(item[1]['hsv'][2]) >= 170 and item[1]['area'] >= previous_area * .8
                for item in departures)
            for item in candidates:
                c=item[1];cx,cy=c['pos']
                ix,iy=cx-position[0],cy-position[1]
                if (c['verified_contact_departure'] or c['area']<previous_area*.35 or
                        c['motion_mean']<18 or c['motion_max']<70 or
                        (player_zone is not None and player_zone(c['pos']) is not None)):
                    continue
                if serve_contact_search and (
                        abs(ix-dx)>max(12,abs(dx)*.75) or c['area']>previous_area*1.6):
                    # A cropped person detector can leave moving body pixels
                    # outside its box. They cannot veto a measured ball leaving
                    # the racket merely by lying near the old downward lane.
                    continue
                if (clear_departure and c['area'] < previous_area * .65 and
                        (int(c['hsv'][1]) < 30 or not 20 <= int(c['hsv'][0]) <= 85)):
                    # A shrinking grey racket fragment cannot displace a
                    # full-sized, freshly moving yellow ball leaving contact.
                    continue
                if (clear_departure and c['area'] < previous_area * .8 and
                        item[0] > max(20, speed * .25)):
                    # A smaller fragment lagging well behind the incoming
                    # prediction is weaker evidence than a full-sized visible
                    # return. A ball actually continuing its measured lane
                    # still takes precedence, even beside a possible strike.
                    continue
                steady=item[0]<=max(20,speed*.6)
                slowing_descent=(dy>=12 and 0<=iy<=dy and dx*ix>0 and
                    .45<=abs(ix/dx)<=1.5 and item[0]<=max(30,speed*1.1))
                if steady or slowing_descent:incoming.append(item)
            if incoming:
                return min(incoming,key=lambda item:item[0])[1]
            vivid = [item for item in departures
                     if 20 <= int(item[1]['hsv'][0]) <= 85 and int(item[1]['hsv'][1]) >= 50 and
                     int(item[1]['hsv'][2]) >= 170 and item[1]['area'] >= previous_area * .8]
            # A real ball can still overlap the racket/body at departure.
            # Prefer measured ball-coloured pixels before using the player
            # zone as a tie-breaker; otherwise an orange shirt just outside
            # the box can displace the yellow ball on the racket.
            departures = vivid or departures
            if player_zone is not None:
                clear = [item for item in departures if player_zone(item[1]['pos']) is None]
                if clear:
                    departures = clear
            candidates = departures
    return min(candidates,key=lambda item:item[0])[1] if candidates else None
