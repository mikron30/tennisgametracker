"""Local temporal recovery of a visible ball beside the far receiver."""
import math
import cv2
import numpy as np


def find_receiver_ball(analyzer, frame):
    origin=getattr(analyzer,'ball_center',None)
    if origin is None or frame is None:
        return None
    h,w=frame.shape[:2]
    if origin[1]>h*.30:
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
    prior=getattr(analyzer,'_receiver_ball_probe',None)
    now=int(analyzer.frame_count)
    previous_gray = getattr(analyzer, '_receiver_ball_probe_gray', None)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    analyzer._receiver_ball_probe_gray = current_gray
    analyzer._receiver_ball_probe=(now,choices)
    if prior is None or not 1<=now-prior[0]<=2:
        return None
    pairs=[]
    for c in choices:
        for old in prior[1]:
            distance=math.dist(c['pos'],old['pos'])
            if 2<=distance<=65 and .35<=c['area']/old['area']<=2.8:
                pairs.append(c)
    if len(pairs) != 1 or previous_gray is None or previous_gray.shape != current_gray.shape:
        return None
    candidate = dict(pairs[0])
    cx, cy = candidate['pos']
    # Use the same image pair that supplied temporal confirmation. Other
    # recovery paths may already have overwritten the tracker gray buffer.
    patch = np.s_[max(0,cy-8):cy+9, max(0,cx-8):cx+9]
    diff = cv2.absdiff(previous_gray[patch], current_gray[patch])
    candidate['motion_mean'] = float(np.mean(diff))
    candidate['motion_max'] = float(np.max(diff))
    candidate['motion_source'] = 'receiver_probe_frame_pair'
    candidate['motion_frames'] = (prior[0], now)
    return candidate
