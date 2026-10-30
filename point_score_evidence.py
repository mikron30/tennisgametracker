"""Do not turn unresolved ball loss into a confident score or game change."""


def unresolved_end_requires_review(owner, reason, measured_terminal=False):
    if (getattr(owner,'far_serve_image_baseline',False) and
            getattr(owner,'_active_serve_area_end',None)=='near' and
            'hit the net' in (reason or '').lower() and
            getattr(owner,'_serve_landed_in_current_attempt',False) and
            getattr(owner,'_last_serve_bounce_was_in',False) and
            int(getattr(owner,'_point_hit_count',0))==1):
        # A legal serve followed by a net return needs a measured receiver
        # hit. Court side alone cannot establish that missing stroke's owner.
        return True
    rewind=getattr(owner,'_short_rewind',None)
    if not (rewind is not None and rewind.point==getattr(owner,'point_start_frame_internal',None)):
        return False
    calibrated=bool(getattr(owner,'far_serve_image_baseline',False))
    pending_failure=(calibrated and getattr(rewind,'first_problem',None) is not None and
                     getattr(rewind,'last_rejection',None) is not None)
    if not (rewind.unresolved or pending_failure):
        return False
    text=(reason or '').lower()
    if rewind.hard_uncertainty or any(word in text for word in ('timeout','ball lost')):
        return True
    if not calibrated:
        return False
    # This label can be a renamed duration timeout using a stale player-side
    # marker; it does not establish an observed double bounce.
    if 'ball stopped' in text:
        return True
    if measured_terminal or not any(word in text for word in ('out of court','hit the net')):
        return False
    # A weak alternate-HSV/background selection cannot resolve the earlier
    # failed replay merely by entering an OUT or NET region.
    fresh=(getattr(owner,'_last_tracked_candidate_motion_frame',-1)==owner.frame_count and
           float(getattr(owner,'_last_tracked_candidate_motion_mean',0))>=6 and
           float(getattr(owner,'_last_tracked_candidate_motion_max',0))>=35 and
           float(getattr(owner,'ball_size',0) or 0)>=3 and
           int(getattr(owner,'stuck_frame_count',0))<=1)
    return not fresh
