"""Keep measured near-side toss evidence across brief contour/stance gaps."""
import math


def above_local_net_top(owner, point):
    """Use the sloped tape at this X for a calibrated near-side flight."""
    if point is None:
        return False
    fallback=float(getattr(owner,'net_area_y_min',0))
    if (getattr(owner,'far_serve_image_baseline',False) and
            getattr(owner,'_active_serve_area_end',None)=='near'):
        getter=getattr(owner,'_get_net_polygon',None)
        polygon=getter() if callable(getter) else None
        if polygon is not None:
            top=polygon.reshape(-1,2)[:3]
            for a,b in zip(top,top[1:]):
                if min(a[0],b[0])<=point[0]<=max(a[0],b[0]) and b[0]!=a[0]:
                    local=a[1]+(point[0]-a[0])*(b[1]-a[1])/(b[0]-a[0])
                    return bool(point[1]<local)
    return bool(point[1]<fallback)


def verified_near_serve_net_drop(owner):
    """A measured server strike remains a serve until its first landing."""
    start = getattr(owner,'point_start_frame_internal',None)
    strike = getattr(owner,'_last_racket_contact_frame',-1000000)
    contact = getattr(owner,'_last_racket_contact_point',None)
    net = getattr(owner,'_serve_net_contact_frame',-1000000)
    return bool(getattr(owner,'far_serve_image_baseline',False) and
        getattr(owner,'_active_serve_area_end',None)=='near' and
        getattr(owner,'_serve_phase_active',False) and
        not getattr(owner,'_serve_landed_in_current_attempt',False) and
        int(getattr(owner,'_point_hit_count',0))==1 and
        int(getattr(owner,'ground_bounce_count',0))==0 and
        start is not None and start <= strike < net <= owner.frame_count and
        owner.frame_count-strike <= 45 and owner.frame_count-net <= 30 and
        getattr(owner,'_last_racket_contact_label','')=='verified outgoing contact flight' and
        getattr(owner,'_last_racket_contact_player',None)==owner._current_server_index() and
        contact is not None and contact[1]>owner.net_area_y_max)


def measured_near_serve_in(owner, point, frame):
    """Restore a first IN landing after the serve stroke closed legacy context.

    Only the measured-bounce caller uses this predicate. A verified server
    strike and fresh service-box geometry are required; a rally bounce or an
    OUT/unknown landing cannot reopen serve bookkeeping.
    """
    start = getattr(owner, 'point_start_frame_internal', None)
    contact = getattr(owner, '_last_racket_contact_point', None)
    contact_frame = getattr(owner, '_last_racket_contact_frame', -1000000)
    if not (getattr(owner,'far_serve_image_baseline',False) and
            getattr(owner,'_active_serve_area_end',None)=='near' and
            start is not None and start <= contact_frame < owner.frame_count and
            owner.frame_count-contact_frame <= 45 and
            int(getattr(owner,'_point_hit_count',0))==1 and
            int(getattr(owner,'ground_bounce_count',0))==1 and
            not getattr(owner,'_serve_landed_in_current_attempt',False) and
            getattr(owner,'_last_racket_contact_label','')=='verified outgoing contact flight' and
            getattr(owner,'_last_racket_contact_player',None)==owner._current_server_index() and
            contact is not None and contact[1]>owner.net_area_y_max and
            owner._player_point_zone(point) is None):
        return False
    service_ok,_,_ = owner._evaluate_target_service_box_fresh(
        point,frame,getattr(owner,'_point_target_service_side',None))
    return service_ok is True


def recent_near_stance(owner):
    stance = getattr(owner, '_serve_stance_last_valid', None)
    if (not getattr(owner, 'far_serve_image_baseline', False) or
            getattr(owner, '_active_serve_area_end', None) != 'near' or
            getattr(owner, '_serve_stance_guard_decision', None) != 'allow' or
            not isinstance(stance, dict) or stance.get('side') != 'near'):
        return False
    return (0 <= owner.frame_count - stance.get('frame', -1000000) <= 12 and
            float(stance.get('confidence') or 0) >= .45)


def hold_near_toss_gap(owner, history, details):
    if not recent_near_stance(owner) or not history or not details:
        return False
    last = details[-1]
    if not 1 <= owner.frame_count - last.get('frame', -1000000) <= 2:
        return False
    if (float(last.get('motion_delta') or 0) < 12 or
            float(last.get('aspect_ratio') or 100) > 3.2 or
            float(last.get('circularity') or 0) < .12):
        return False
    # A single fresh compact seed may bridge one missed image. Longer history
    # must actually be rising; stationary/falling candidates cannot be retained.
    return len(history) == 1 or history[-2][1] - history[-1][1] >= 6


def near_toss_entry_y_min(owner, default):
    """Use the certified server's scale when the lower toss was occluded."""
    if not recent_near_stance(owner):
        return default
    bbox = owner._serve_stance_last_valid.get('bbox')
    if bbox is None or len(bbox) != 4:
        return default
    _, top, _, height = bbox
    if height <= 0:
        return default
    area_height = owner.serve_area_y_max - owner.serve_area_y_min
    return min(default, max(owner.serve_area_y_min + area_height * .25,
                            top - height * .5))


def partial_near_serve_contact(owner):
    """A falling toss followed by a short turning step may span the strike."""
    if (not getattr(owner, 'far_serve_image_baseline', False) or
            getattr(owner, '_active_serve_area_end', None) != 'near' or
            not getattr(owner, '_awaiting_serve_bounce', False) or
            getattr(owner, '_point_hit_count', 0) != 0 or
            float(getattr(owner, 'ball_size', 0) or 0) < 80):
        return False
    samples = getattr(owner, 'motion_history', [])[-4:]
    if (len(samples) != 4 or
            [s['frame'] for s in samples] != list(range(owner.frame_count-4, owner.frame_count)) or
            tuple(samples[-1]['pos']) != tuple(owner.ball_center)):
        return False
    a,b,c,d = [s['pos'] for s in samples]
    if not (b[1]-a[1] >= 12 and c[1]-b[1] >= 12 and
            -12 <= d[1]-c[1] <= 12 and 8 <= math.dist(c,d) <= 70 and
            d[1] > owner.net_area_y_max):
        return False
    stance = getattr(owner, '_serve_stance_last_valid', None)
    if not isinstance(stance, dict) or stance.get('side') != 'near':
        return False
    bbox = stance.get('bbox')
    if bbox is None or not 0 <= owner.frame_count-stance.get('frame',-1000000) <= 45:
        return False
    x,y,w,h = bbox
    top=min(y-h,getattr(owner,'serve_area_y_min',y-h))
    bottom=max(y+h*.5,getattr(owner,'serve_area_y_max',y+h*.5))
    return (x-w <= d[0] <= x+w*2 and top <= d[1] <= bottom)


def falling_near_serve_contact(owner):
    if (not getattr(owner, 'far_serve_image_baseline', False) or
            getattr(owner, '_active_serve_area_end', None) != 'near' or
            not getattr(owner, '_awaiting_serve_bounce', False) or
            getattr(owner, '_point_hit_count', 0) != 0 or
            float(getattr(owner, 'ball_size', 0) or 0) < 80):
        return False
    samples=getattr(owner,'motion_history',[])[-3:]
    if len(samples)!=3 or [s['frame'] for s in samples] != list(range(owner.frame_count-3,owner.frame_count)):
        return False
    if not all(b['pos'][1]-a['pos'][1]>=12 for a,b in zip(samples,samples[1:])):
        return False
    stance=getattr(owner,'_serve_stance_last_valid',None)
    if not isinstance(stance,dict) or stance.get('side')!='near' or not 0<=owner.frame_count-stance['frame']<=45:
        return False
    bbox=stance.get('bbox')
    if bbox is None:return False
    x,y,w,h=bbox;px,py=owner.ball_center
    top=min(y-h,getattr(owner,'serve_area_y_min',y-h))
    bottom=max(y+h*.5,getattr(owner,'serve_area_y_max',y+h*.5))
    return py>owner.net_area_y_max and x-w<=px<=x+w*2 and top<=py<=bottom


def near_partial_strike_area(owner, area):
    """Racket overlap must not inflate the next departing ball's size floor."""
    if not partial_near_serve_contact(owner):
        return area
    reference = getattr(owner, '_configured_flight_area_reference', {})
    measured = float(reference.get('area', 0) or 0)
    if (reference.get('point_start') == getattr(owner, 'point_start_frame_internal', None) and
            reference.get('frame') == owner.frame_count-1 and measured >= 80 and
            measured*1.35 < area <= measured*4):
        return measured
    return area


def far_side_incoming_serve(owner):
    """A small, continuously observed near serve may fade to grey far away."""
    samples=getattr(owner,'motion_history',[])[-3:]
    return bool(getattr(owner,'far_serve_image_baseline',False) and
        getattr(owner,'_active_serve_area_end',None)=='near' and
        (getattr(owner,'_awaiting_serve_bounce',False) or
         (getattr(owner,'_serve_landed_in_current_attempt',False) and
          getattr(owner,'_last_serve_bounce_was_in',False) and
          int(getattr(owner,'_point_hit_count',0))==1)) and
        getattr(owner,'ball_center',None) is not None and
        above_local_net_top(owner,owner.ball_center) and
        3<=float(getattr(owner,'ball_size',0) or 0)<=80 and len(samples)==3 and
        [s['frame'] for s in samples]==list(range(owner.frame_count-3,owner.frame_count)) and
        all(2<=math.dist(a['pos'],b['pos'])<=90 for a,b in zip(samples,samples[1:])) and
        getattr(owner,'_last_tracked_candidate_motion_frame',-1)==owner.frame_count-1 and
        ((getattr(owner,'_last_tracked_candidate_motion_mean',0)>=6 and
          getattr(owner,'_last_tracked_candidate_motion_max',0)>=35) or
         (float(getattr(owner,'ball_size',0) or 0)<=12 and
          getattr(owner,'_last_tracked_candidate_motion_mean',0)>=5 and
          getattr(owner,'_last_tracked_candidate_motion_max',0)>=60)))


def near_serve_impact(samples):
    """Confirm a shallow distant landing using two measured rebound images."""
    def area_continuity(window):
        areas=sorted(s['area'] for s in window)
        middle=(areas[(len(areas)-1)//2]+areas[len(areas)//2])/2
        return bool(all(3<=a<=80 for a in areas) and
            (areas[-1]<=areas[0]*5 or
             (len(areas)>=7 and areas[-1]<=middle*3 and areas[0]>=max(3,middle*.2))))
    if len(samples)>=8:
        window=samples[-8:]
        steps=[(b['pos'][0]-a['pos'][0],b['pos'][1]-a['pos'][1])
               for a,b in zip(window,window[1:])]
        incoming=steps[:3];transition=steps[3:5];outgoing=steps[-2:]
        areas=sorted(s['area'] for s in window)
        partial_area_ok=area_continuity(window)
        if (all(b['frame']-a['frame']==1 for a,b in zip(window,window[1:])) and
                all(-12<=dy<=0 for dx,dy in incoming) and incoming[0][1]<=-4 and
                abs(incoming[-1][1])<=3 and
                all(b[1]>=a[1]-1 for a,b in zip(incoming,incoming[1:])) and
                abs(transition[0][1])<=2 and -6<=transition[1][1]<=0 and
                all(-80<=dy<=-8 for dx,dy in outgoing) and
                abs(outgoing[1][1])<=abs(outgoing[0][1])*3 and
                outgoing[0][1]<=-max(8,abs(incoming[-1][1])*3) and
                all(2<=abs(dx)<=15 for dx,dy in steps) and
                all(dx*steps[0][0]>0 for dx,dy in steps) and
                max(abs(dx) for dx,dy in steps)-min(abs(dx) for dx,dy in steps)<=8 and
                all(3<=a<=80 for a in areas) and partial_area_ok):
            # White paint can leave one partial contour and an extra flattened
            # image at impact. Two subsequent rising images establish the bounce.
            return dict(window[-4],perspective_flattening=True)
    if len(samples)>=7:
        window=samples[-7:]
        steps=[(b['pos'][0]-a['pos'][0],b['pos'][1]-a['pos'][1])
               for a,b in zip(window,window[1:])]
        incoming=steps[:3];outgoing=steps[-2:]
        if (all(b['frame']-a['frame']==1 for a,b in zip(window,window[1:])) and
                all(-12<=dy<=0 for dx,dy in incoming) and
                incoming[0][1]<=-4 and abs(incoming[-1][1])<=3 and
                all(b[1]>=a[1]-1 for a,b in zip(incoming,incoming[1:])) and
                abs(steps[3][1])<=2 and
                all(-80<=dy<=-8 for dx,dy in outgoing) and
                abs(outgoing[1][1])<=abs(outgoing[0][1])*3 and
                outgoing[0][1]<=-max(8,abs(incoming[-1][1])*3) and
                all(2<=abs(dx)<=15 for dx,dy in steps) and
                all(dx*steps[0][0]>0 for dx,dy in steps) and
                max(abs(dx) for dx,dy in steps)-min(abs(dx) for dx,dy in steps)<=8 and
                area_continuity(window)):
            # Perspective may keep image Y decreasing right up to the ground.
            # A flattened approach and two accelerated upward observations
            # establish the late partial-frame impact, without a fabricated
            # downward sample. The caller must exclude player contact.
            return dict(window[-3], perspective_flattening=True)
    for length in (6,7):
        if len(samples)<length:continue
        window=samples[-length:]
        if any(b['frame']-a['frame']!=1 for a,b in zip(window,window[1:])):continue
        steps=[(b['pos'][0]-a['pos'][0],b['pos'][1]-a['pos'][1]) for a,b in zip(window,window[1:])]
        incoming=steps[:3];outgoing=steps[-2:]
        if length==7 and abs(steps[3][1])>3:continue
        if not (all(1<=dy<=12 for dx,dy in incoming) and
                all(-80<=dy<=-8 for dx,dy in outgoing) and
                outgoing[0][1]<=-max(8, incoming[-1][1]*2) and
                abs(outgoing[1][1])<=abs(outgoing[0][1])*1.5 and
                all(abs(dx)<=15 for dx,dy in steps) and
                all(dx*steps[0][0]>0 for dx,dy in steps) and
                max(abs(dx) for dx,dy in steps)-min(abs(dx) for dx,dy in steps)<=8 and
                area_continuity(window)):continue
        return max(window[3:length-2],key=lambda s:s['pos'][1])
    return None


def near_toss_tracking(owner):
    seed=getattr(owner,'_near_serve_toss_seed',None)
    position=getattr(owner,'ball_center',None)
    if (not getattr(owner,'far_serve_image_baseline',False) or
            getattr(owner,'_active_serve_area_end',None)!='near' or not seed or position is None or
            not getattr(owner,'_awaiting_serve_bounce',False) or
            getattr(owner,'_point_hit_count',0)>0 or
            getattr(owner,'_serve_launch_direction_x',0)!=0 or
            not 0<=owner.frame_count-seed['frame']<=45 or seed['area']<100):return False
    return abs(position[0]-seed['position'][0])<=max(100,(owner.serve_area_x_max-owner.serve_area_x_min)*.07)


def near_serve_netward_flight(owner):
    samples=getattr(owner,'motion_history',[])[-3:]
    p=getattr(owner,'ball_center',None)
    return bool(getattr(owner,'far_serve_image_baseline',False) and
        getattr(owner,'_active_serve_area_end',None)=='near' and
        getattr(owner,'_awaiting_serve_bounce',False) and p is not None and
        owner.net_area_y_min<=p[1]<=owner.net_area_y_max+100 and
        20<=float(getattr(owner,'ball_size',0) or 0)<=450 and len(samples)==3 and
        [s['frame'] for s in samples]==list(range(owner.frame_count-3,owner.frame_count)) and
        all(b['pos'][1]-a['pos'][1]<=-8 and 10<=math.dist(a['pos'],b['pos'])<=220
            for a,b in zip(samples,samples[1:])) and
        getattr(owner,'_last_tracked_candidate_motion_frame',-1)==owner.frame_count-1 and
        getattr(owner,'_last_tracked_candidate_motion_mean',0)>=6 and
        getattr(owner,'_last_tracked_candidate_motion_max',0)>=35)
