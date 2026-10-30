"""Recover an uncounted interval; serve-side parity never awards a point."""
import cv2
from near_serve_scan_recovery import recent_near_stance
from short_ball_rewind import snapshot, restore


def serve_side_model(owner, model):
    if model and model.get('center_line'):return model
    left=getattr(owner,'serve_area_x_min',None)
    right=getattr(owner,'serve_area_x_max',None)
    if left is None or right is None or right-left<200:return None
    # The calibrated search midline is sufficient to REQUEST a rescan when
    # image service-line extraction fails; it is never used to score a bounce.
    return dict(center_line=dict(a=0,b=(left+right)*.5),source='calibrated serve-area midpoint')


class MissedServeReplay:
    def __init__(self):
        self.checkpoint = None
        self.attempted = set()

    def remember(self, owner, point_end_frame):
        if (point_end_frame is None or not getattr(owner,'far_serve_image_baseline',False) or
                getattr(owner,'_active_serve_area_end',None) != 'near' or
                getattr(owner,'_last_point_outcome_category',None) in
                ('serve_let','tracking_uncertain')):
            return
        index=getattr(owner,'_point_history_point_index',0)
        if index<=0 or (self.checkpoint and self.checkpoint['index']==index): return
        self.checkpoint=dict(index=index,frame=owner.frame_count,end=point_end_frame,
            source=int(round(owner.cap.get(cv2.CAP_PROP_POS_FRAMES)))-1,
            state=snapshot(owner),score=tuple(owner.score_games)+tuple(owner.score_points),
            server=owner._current_server_index())

    def rewind_if_missing(self, owner, history, details, model):
        saved=self.checkpoint
        model=serve_side_model(owner,model)
        if (saved is None or saved['index'] in self.attempted or
                not 30<=owner.frame_count-saved['frame']<=2000 or
                not recent_near_stance(owner) or not history or len(details)!=len(history) or
                getattr(owner,'_point_history_point_index',0)!=saved['index'] or
                tuple(owner.score_games)+tuple(owner.score_points)!=saved['score'] or
                owner._current_server_index()!=saved['server'] or
                not model or not model.get('center_line')): return None
        last=details[-1]
        clear_second_serve_contradiction = (
            getattr(owner,'current_serve_attempt',1)==2 and
            float(owner._serve_stance_last_valid.get('confidence') or 0)>=.65 and
            owner.frame_count-last.get('frame',-1000000)<=1 and
            float(last.get('area') or 0)>=80 and
            float(last.get('aspect_ratio') or 100)<=2.5 and
            float(last.get('circularity') or 0)>=.2 and
            float(last.get('motion_delta') or 0)>=20)
        segment,evidence=history[-4:],details[-4:]
        if not clear_second_serve_contradiction and (len(segment)<4 or
                any(b.get('frame',0)-a.get('frame',0) not in (1,2) for a,b in zip(evidence,evidence[1:])) or
                max(p[0] for p in segment)-min(p[0] for p in segment)>100 or
                max(p[1] for p in segment)-min(p[1] for p in segment)<20 or
                any(float(d.get('aspect_ratio') or 100)>3.2 or
                    float(d.get('circularity') or 0)<.12 or
                    float(d.get('motion_delta') or 0)<12 for d in evidence)): return None
        expected='right' if sum(owner.score_points)%2==0 else 'left'
        line=model['center_line']
        center=owner._serve_stance_last_valid.get('center')
        # A clipped toss may already be falling when first segmented. A clear
        # legal serving stance on the wrong half is enough to REQUEST replay;
        # neither the stance nor parity is evidence of who won the lost point.
        positions=[center] if center is not None else segment
        offsets=[p[0]-(line['a']*p[1]+line['b']) for p in positions]
        if not (all(x>60 for x in offsets) or all(x < -60 for x in offsets)): return None
        observed='right' if offsets[-1]>0 else 'left'
        if observed==expected: return None
        until=owner.frame_count; self.attempted.add(saved['index'])
        resume=owner.cap.get(cv2.CAP_PROP_POS_FRAMES)
        live=snapshot(owner)
        if not owner.cap.set(cv2.CAP_PROP_POS_FRAMES,saved['source']+1):
            owner.cap.set(cv2.CAP_PROP_POS_FRAMES,resume)
            return None
        try:
            restore(owner,saved['state'])
        except Exception as error:
            restore(owner,live)
            owner.cap.set(cv2.CAP_PROP_POS_FRAMES,resume)
            print(f'[MISSED_SERVE_REPLAY_REJECTED] f{until}: state restore failed: {error}')
            return None
        owner._missed_serve_retry_until=until
        for key in ('_prev_serve_gray','_prev_frame_gray','_terminal_previous_gray',
                    '_terminal_current_gray','_terminal_current_frame'): setattr(owner,key,None)
        buffer=getattr(owner,'_local_ai_frame_buffer',None)
        if buffer is not None: buffer.clear()
        tracker=getattr(owner,'player_tracker',None)
        if tracker is not None: tracker.prev_gray=None
        rewind=getattr(owner,'_short_rewind',None)
        if rewind is not None: rewind.invalidate()
        recovery=getattr(owner,'local_ai_recovery',None)
        if recovery is not None:
            recovery.last_attempt_frame=-1000000; recovery._urgent_retry_until_frame=-1000000
            if hasattr(recovery,'_inference_budget'): recovery._inference_budget.frames.clear()
        print(f'[MISSED_SERVE_REPLAY] f{until}: expected={expected} observed={observed}; '
              f'restored f{saved["frame"]}; retrying fresh near-toss colours; '
              'score unchanged until recovered point is verified')
        return saved['end']
