"""Temporal recovery of a visible ball after a player occlusion."""
import math
import cv2
import numpy as np


def _moving_pair(analyzer, frame, choices, probe_name, max_step, preserve_local_motion=False):
    """Return one unambiguous, moving contour pair from consecutive frames."""
    prior = getattr(analyzer, probe_name, None)
    gray_name = f'{probe_name}_gray'
    previous_gray = getattr(analyzer, gray_name, None)
    now = int(analyzer.frame_count)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    setattr(analyzer, gray_name, current_gray)
    setattr(analyzer, probe_name, (now, choices))
    if prior is None or not 1 <= now - prior[0] <= 2:
        return None

    pairs = []
    for candidate in choices:
        for old in prior[1]:
            distance = math.dist(candidate['pos'], old['pos'])
            if 2 <= distance <= max_step and .35 <= candidate['area'] / old['area'] <= 2.8:
                pairs.append((candidate, old))
    if len(pairs) != 1 or previous_gray is None or previous_gray.shape != current_gray.shape:
        return None

    current, old = pairs[0]
    candidate = dict(current)
    cx, cy = candidate['pos']
    patch = np.s_[max(0, cy - 8):cy + 9, max(0, cx - 8):cx + 9]
    diff = cv2.absdiff(previous_gray[patch], current_gray[patch])
    candidate['motion_mean'] = float(np.mean(diff))
    candidate['motion_max'] = float(np.max(diff))
    candidate['motion_source'] = 'receiver_probe_frame_pair'
    candidate['motion_frames'] = (prior[0], now)
    if preserve_local_motion:
        elapsed = now - prior[0]
        dx = float(candidate['pos'][0] - old['pos'][0]) / elapsed
        dy = float(candidate['pos'][1] - old['pos'][1]) / elapsed
        distance = math.hypot(dx, dy)
        candidate['local_ai_motion_override'] = {
            'dx': dx, 'dy': dy, 'distance': distance,
            'direction_deg': math.degrees(math.atan2(dy, dx)) if distance else 0.0,
        }
        candidate['local_ai_velocity_history'] = [distance]
        candidate['temporal_history_entry'] = {
            'frame': now, 'pos': tuple(candidate['pos']), 'prev_pos': tuple(old['pos']),
            'distance': distance,
            'direction_deg': candidate['local_ai_motion_override']['direction_deg'],
        }
    return candidate


def _near_net_reentry_ball(analyzer, frame, origin):
    """Recover a compact moving ball near the net after a player-body lock.

    The normal receiver search is intentionally local to the far player.  A
    near player can instead hide the ball and leave that search anchored on a
    torso or racket.  This bounded net corridor is enabled only after a recent
    racket contact during the player-reacquisition guard, and still requires a
    unique moving contour in two consecutive frames.
    """
    h, w = frame.shape[:2]
    contact_frame = int(getattr(analyzer, '_last_racket_contact_frame', -1000000))
    elapsed = int(getattr(analyzer, 'frame_count', 0)) - contact_frame
    contact_point = getattr(analyzer, '_last_racket_contact_point', None)
    if not 3 <= elapsed <= 120 or contact_point is None:
        return None
    points = getattr(analyzer, 'net_area_points', None) or []
    if len(points) >= 3:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        left = max(0, int(min(xs) - max(80, w * .04)))
        right = min(w, int(max(xs) + max(80, w * .04)))
        top = max(0, int(min(ys) - max(130, h * .06)))
        bottom = min(h, int(max(ys) + max(70, h * .03)))
    else:
        net_min = float(getattr(analyzer, 'net_area_y_min', h * .38))
        net_max = float(getattr(analyzer, 'net_area_y_max', h * .48))
        left, right = int(w * .20), int(w * .92)
        top = max(0, int(net_min - max(130, h * .06)))
        bottom = min(h, int(net_max + max(70, h * .03)))
    if right <= left or bottom <= top:
        return None

    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 40, 120]), np.array([55, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 6 <= area <= 260:
            continue
        _, _, bw, bh = cv2.boundingRect(contour)
        if max(bw, bh) > max(3, min(bw, bh) * 2.8):
            continue
        moments = cv2.moments(contour)
        if not moments['m00']:
            continue
        cx, cy = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
        choices.append({
            'pos': (cx + left, cy + top), 'area': area,
            'hsv': tuple(int(value) for value in hsv[cy, cx]),
            'source': 'near_net_temporal', 'recovery_label': 'NEAR-NET TEMPORAL RECOVER',
        })
    candidate = _moving_pair(
        analyzer, frame, choices, '_near_net_ball_probe', max(70.0, h * .035)
    )
    if candidate is not None:
        candidate['contact_ai_same_frame_protected'] = True
    return candidate


def confirm_visible_reentry(analyzer, frame, candidate):
    """Confirm one ranked visible-ball candidate in two consecutive images."""
    if not candidate:
        return None
    probe = dict(candidate)
    probe['source'] = 'night_visible_temporal'
    probe['recovery_label'] = 'NIGHT VISIBLE TEMPORAL RECOVER'
    confirmed = _moving_pair(
        analyzer, frame, [probe], '_night_visible_reentry_probe',
        max(70.0, frame.shape[0] * .035), preserve_local_motion=True,
    )
    if confirmed is not None:
        if confirmed['motion_mean'] < 4.0 and confirmed['motion_max'] < 25.0:
            return None
        confirmed['contact_ai_same_frame_protected'] = True
    return confirmed


def find_receiver_ball(analyzer, frame, allow_far_receiver=True, allow_near_net=False):
    origin=getattr(analyzer,'ball_center',None)
    if origin is None or frame is None:
        return None
    h,w=frame.shape[:2]
    if origin[1] > h * .30:
        if not allow_near_net:
            return None
        candidate = _near_net_reentry_ball(analyzer, frame, origin)
        if candidate is not None:
            analyzer._contact_local_ai_cooldown_until_frame = max(
                int(getattr(analyzer, '_contact_local_ai_cooldown_until_frame', -1000000)),
                int(analyzer.frame_count),
            )
        return candidate
    if not allow_far_receiver:
        return None
    radius=max(100,int(w*.04))
    x1,x2=max(0,int(origin[0])-radius),min(w,int(origin[0])+radius)
    y1,y2=max(0,int(origin[1])-radius),min(h,int(origin[1])+radius)
    if x2<=x1 or y2<=y1:
        return None
    hsv=cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([25,40,130]),np.array([55,255,255]))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    choices=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not 6<=area<=200:
            continue
        _,_,bw,bh=cv2.boundingRect(contour)
        if max(bw,bh)>min(bw,bh)*2.5:
            continue
        m=cv2.moments(contour)
        if not m['m00']:
            continue
        cx,cy=int(m['m10']/m['m00']),int(m['m01']/m['m00'])
        choices.append(dict(pos=(cx+x1,cy+y1),area=area,hsv=tuple(int(v) for v in hsv[cy,cx]),
                            source='receiver_temporal',recovery_label='RECEIVER TEMPORAL RECOVER'))
    candidate = _moving_pair(analyzer, frame, choices, '_receiver_ball_probe', 65.0,
                             preserve_local_motion=True)
    if candidate is None:
        return None

    # The main tracking loop asks Contact Local AI to re-arbitrate suspicious
    # player-adjacent positions after normal tracking. On an AI miss that path
    # restores the pre-track anchor. A receiver recovery already has independent
    # two-frame temporal evidence, so a *moving* recovery must not be erased
    # merely because Contact Local AI has no alternative candidate. Suppress
    # only that same-frame trigger; all later frames and the regular buffered
    # Local-AI recovery remain active.
    moving_evidence = candidate['motion_mean'] >= 4.0 or candidate['motion_max'] >= 25.0
    candidate['contact_ai_same_frame_protected'] = bool(moving_evidence)
    if moving_evidence:
        analyzer._contact_local_ai_cooldown_until_frame = max(
            int(getattr(analyzer, '_contact_local_ai_cooldown_until_frame', -1000000)),
                int(analyzer.frame_count),
        )
    return candidate
