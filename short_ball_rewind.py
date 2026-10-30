"""Bounded, transactional Local AI replay within the current tennis point.

The live capture never seeks. A separate decoder reads only the short suspect
segment; accepted observations are replayed through the existing ball/event
committers. Scoring and external writes are forbidden during that transaction.
"""
from collections import deque
from dataclasses import dataclass
import copy
import math
import time

import cv2
import numpy as np
from local_ai_budget import InferenceBudget


class ReplayRejected(RuntimeError):
    pass


def clone(value):
    # Images/config arrays are read-only here. Retaining a reference avoids
    # copying several 4K images into every checkpoint.
    if isinstance(value, np.ndarray):
        return value if value.size > 100_000 else value.copy()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v) for v in value)
    if isinstance(value, deque):
        return deque((clone(v) for v in value), maxlen=value.maxlen)
    return copy.deepcopy(value)


_EXCLUDE = {'_local_ai_frame_buffer', '_short_rewind', '_terminal_current_frame',
            '_terminal_current_gray', '_terminal_previous_gray', '_prev_frame_gray',
            '_local_ai_tight_roi_previous_gray', '_short_rewind_replaying'}
_VALUES = (str, int, float, bool, type(None), list, dict, tuple, set, deque, np.ndarray)


def snapshot(owner):
    """Copy semantic state, keeping runtime handles, models and files external."""
    state = {}
    for key, value in vars(owner).items():
        if key in _EXCLUDE or key.startswith(('_short_rewind_', '_debug')):
            continue
        if isinstance(value, np.ndarray) and value.size > 100_000:
            continue
        if isinstance(value, _VALUES):
            state[key] = clone(value)
    tracker = getattr(owner, 'player_tracker', None)
    player = None
    if tracker is not None:
        player = {k: clone(v) for k, v in vars(tracker).items()
                  if k != 'prev_gray' and isinstance(v, _VALUES)}
        # PlayerTrack instances live inside this small dictionary.
        player['tracks'] = copy.deepcopy(tracker.tracks)
    return state, player


def restore(owner, saved):
    state, player = saved
    for key in list(vars(owner)):
        value = vars(owner)[key]
        if (key not in _EXCLUDE and not key.startswith(('_short_rewind_', '_debug')) and
                isinstance(value, _VALUES) and
                not (isinstance(value, np.ndarray) and value.size > 100_000) and key not in state):
            delattr(owner, key)
    for key, value in state.items():
        setattr(owner, key, clone(value))
    tracker = getattr(owner, 'player_tracker', None)
    if tracker is not None and player is not None:
        for key in list(vars(tracker)):
            if key != 'prev_gray' and isinstance(vars(tracker)[key],_VALUES) and key not in player:
                delattr(tracker,key)
        for key, value in player.items():
            setattr(tracker, key, clone(value))


def prestrike_toss_checkpoint(owner):
    """Only a coherent, freshly observed calibrated toss may seed recovery."""
    if not (getattr(owner, 'far_serve_image_baseline', False) and
            getattr(owner, '_active_serve_area_end', None) == 'near' and
            getattr(owner, '_serve_phase_active', False)):
        return False
    from near_serve_scan_recovery import near_toss_tracking
    if not near_toss_tracking(owner):
        return False
    seed = owner._near_serve_toss_seed
    area = float(getattr(owner, 'ball_size', 0) or 0)
    if not max(80, seed['area'] * .25) <= area <= seed['area'] * 1.8:
        return False
    samples = getattr(owner, 'motion_history', [])[-3:]
    if (len(samples) != 3 or
            [s['frame'] for s in samples] != list(range(owner.frame_count-2, owner.frame_count+1)) or
            tuple(samples[-1]['pos']) != tuple(owner.ball_center)):
        return False
    steps = [(b['pos'][0]-a['pos'][0], b['pos'][1]-a['pos'][1])
             for a,b in zip(samples,samples[1:])]
    return bool(all(abs(dx) <= 20 and 2 <= math.hypot(dx,dy) <= 90 for dx,dy in steps) and
                (steps[0][1] * steps[1][1] >= 0 or all(abs(dy) <= 8 for _,dy in steps)))


def fresh_candidates(image, previous, anchor, velocity, elapsed, area, limit=64):
    """Motion candidates independent of configured HSV contour admission."""
    height, width = image.shape[:2]
    speed = math.hypot(*velocity)
    radius = min(1100, max(180, speed * min(elapsed, 5) * 2 + elapsed * 30))
    # Include both the anchor and projected flight. A contact can reverse it.
    px, py = anchor[0] + velocity[0] * elapsed, anchor[1] + velocity[1] * elapsed
    x1, x2 = max(0, int(min(anchor[0], px)-radius)), min(width, int(max(anchor[0], px)+radius+1))
    y1, y2 = max(0, int(min(anchor[1], py)-radius)), min(height, int(max(anchor[1], py)+radius+1))
    if x2 <= x1 or y2 <= y1:
        return []
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    old = previous[y1:y2, x1:x2]
    delta = cv2.absdiff(gray, old)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Include pale, yellow and camera-shifted balls, but only fresh pixels.
    changed = ((gray.astype(np.int16)-old.astype(np.int16) >= 12) |
               ((old >= 170) & (delta >= 18) & (hsv[:, :, 1] >= 10)))
    mask = np.where(changed & (hsv[:, :, 2] >= 70), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    found = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        size = float(cv2.contourArea(contour))
        if not 3 <= size <= min(2000, max(160, area * 8)):
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if max(bw, bh) > max(10, min(bw, bh)*4):
            continue
        moments = cv2.moments(contour)
        cx, cy = int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])
        patch = delta[max(0, cy-6):cy+7, max(0, cx-6):cx+7]
        mean, peak = float(patch.mean()), float(patch.max())
        if mean < 4 or peak < 35:
            continue
        x, y = cx+x1, cy+y1
        found.append(dict(x=x, y=y, area=size, bbox=[bx+x1, by+y1, bw, bh],
                          motion_mean=mean, motion_max=peak, mode='rewind-fresh-motion',
                          search_cost=min(math.dist((x, y), (px, py)),
                                          math.dist((x, y), anchor)+speed*.5)))
    return sorted(found, key=lambda c:c['search_cost'])[:limit]


def best_path(groups, anchor_frame, anchor, velocity, minimum_score=.985, *, occlusion_frames=()):
    """Beam search: appearance cannot excuse teleportation or stationary locks."""
    beams = [([], 0., anchor_frame, anchor, velocity)]
    occlusion_frames=set(occlusion_frames)
    for frame, candidates in groups:
        expanded = []
        for path, cost, last_frame, last_pos, last_velocity in beams:
            gap = frame-last_frame
            allowed_gap=3+min(4,sum(n in occlusion_frames for n in range(last_frame+1,frame+1)))
            if gap <= 0 or gap > allowed_gap:
                continue
            # At most two genuinely unobserved images, never invented positions.
            if gap < allowed_gap:
                expanded.append((path, cost+4, last_frame, last_pos, last_velocity))
            for candidate in candidates:
                if float(candidate.get('ai_score', 0)) < minimum_score:
                    continue
                pos = (int(candidate['x']), int(candidate['y']))
                dx, dy = (pos[0]-last_pos[0])/gap, (pos[1]-last_pos[1])/gap
                speed = math.hypot(dx, dy)
                previous_speed = math.hypot(*last_velocity)
                if len(path)>=2:
                    older=path[-2]
                    before=path[-3] if len(path)>=3 else dict(frame=anchor_frame,pos=anchor)
                    older_speed=math.dist(older['pos'],before['pos'])/max(1,older['frame']-before['frame'])
                    if previous_speed>max(70,older_speed*1.7) and speed<previous_speed*.35:
                        # A single unexplained leap followed by an immediate
                        # stop is not a confirmed outgoing ball flight.
                        continue
                if not 2 <= speed <= min(400, max(70, previous_speed*3.5+30)):
                    continue
                if candidate.get('motion_mean', 0) < 4 or candidate.get('motion_max', 0) < 35:
                    continue
                change = math.hypot(dx-last_velocity[0], dy-last_velocity[1])
                if change > max(120, previous_speed*2.5):
                    continue
                if path:
                    ratio = candidate['area']/max(1, path[-1]['area'])
                    if not .12 <= ratio <= 8:
                        continue
                item = dict(candidate, frame=frame, pos=pos)
                score_cost = -math.log(max(1e-8, float(candidate['ai_score'])))
                expanded.append((path+[item], cost+score_cost+change/max(30, previous_speed)*.25,
                                 frame, pos, (dx, dy)))
        beams = sorted(expanded, key=lambda p:p[1])[:32]
        if not beams:
            return None
    end_frame = groups[-1][0] if groups else anchor_frame
    paths = [p for p in beams if len(p[0]) >= 4 and p[2] == end_frame and
             all(p[0][i+1]['frame']-p[0][i]['frame'] == 1 for i in range(len(p[0])-3, len(p[0])-1)) and
             all(c.get('player_zone') is None for c in p[0][-2:]) and
             all(not c.get('trusted_prefix',False) for c in p[0][-3:]) and
             math.dist(p[0][-1]['pos'], anchor) >= 12]
    return min(paths, key=lambda p:p[1])[0] if paths else None


@dataclass
class Checkpoint:
    frame: int
    source: int
    point: int
    position: tuple
    velocity: tuple
    area: float
    state: tuple
    prestrike_toss: bool = False


def prestrike_toss_path(checkpoint, images, labels, ledger, anchor_image):
    """Re-observe a large unstruck toss; never synthesize AI confidence."""
    if not checkpoint.prestrike_toss or len(images) < 4 or len(images) != len(labels):
        return None
    state = checkpoint.state[0]
    seed = state.get('_near_serve_toss_seed')
    ranges = state.get('tracking_hsv_ranges')
    if (not seed or not ranges or not state.get('far_serve_image_baseline') or
            state.get('_active_serve_area_end') != 'near' or
            state.get('_point_hit_count', 0) or state.get('_serve_launch_direction_x', 0) or
            not 0 <= labels[-1] - seed['frame'] <= 45 or
            labels != list(range(checkpoint.frame+1, checkpoint.frame+1+len(labels)))):
        return None
    from causal_ball_recovery import find_local_ball
    filters = [(np.array([r['h_min'],r['s_min'],r['v_min']],np.uint8),
                np.array([r['h_max'],r['s_max'],r['v_max']],np.uint8))
               for r in ranges if r['h_min'] <= r['h_max']]
    filters.append((np.array([20,50,130],np.uint8),np.array([85,255,255],np.uint8)))
    pos = checkpoint.position
    velocity = checkpoint.velocity
    area = checkpoint.area
    old_gray = cv2.cvtColor(anchor_image,cv2.COLOR_BGR2GRAY)
    lane = max(100,(state.get('serve_area_x_max',0)-state.get('serve_area_x_min',0))*.07)
    path=[]
    for n,image in zip(labels,images):
        candidate=find_local_ball(image,old_gray,pos,dict(dx=velocity[0],dy=velocity[1]),
            max(area,seed['area']),filters,minimum_speed=2,maximum_speed=90,
            minimum_saturation=5,separate_profiles=True)
        if candidate is None:return None
        point=candidate['pos'];dx,dy=point[0]-pos[0],point[1]-pos[1]
        if (abs(dx)>20 or not 2<=math.hypot(dx,dy)<=90 or
                not max(80,seed['area']*.25)<=candidate['area']<=seed['area']*1.8 or
                abs(point[0]-seed['position'][0])>lane or point[1]>=seed['position'][1]-80 or
                candidate['motion_mean']<18 or candidate['motion_max']<70 or
                (velocity[1]*dy<0 and max(abs(velocity[1]),abs(dy))>8)):
            return None
        for track in (ledger[n].get('tracks') or {}).values():
            box=getattr(track,'bbox',None)
            if box is not None:
                x,y,w,h=box
                if x-8<=point[0]<=x+w+8 and y-8<=point[1]<=y+h+8:return None
        path.append(dict(candidate,frame=n,x=point[0],y=point[1],player_zone=None,
                         replay_evidence='prestrike-hsv-motion'))
        pos=point;velocity=(dx,dy);area=candidate['area']
        old_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if path[-1]['pos'][1]>state.get('net_area_y_max',0):return None
    return path


class ShortBallRewind:
    def __init__(self, window=12, maximum_window=24):
        self.window = int(window)
        self.maximum_window = int(maximum_window)
        self.checkpoints = deque(maxlen=maximum_window+1)
        self.observations = deque(maxlen=maximum_window+2)
        self.point = None
        self.attempts = 0
        self.first_problem = None
        self.last_attempt = -1000000
        self.accepted = 0
        self.decoder = None
        self.last_rejection = None
        self.unresolved = False
        self.unresolved_since = None
        self.hard_uncertainty = False
        self.hard_since = None
        self.contact_uncertainty = False
        self.prestrike_uncertainty = False
        self.trusted_run = 0
        self.total_attempts = 0
        self.checkpoint_seconds = 0.
        self.review_points=[]
        # Rejected replay must not consume the fast tracker's inference slots
        # and thereby change its subsequent decisions. This separate, bounded
        # allowance permits two extra batched checks per video second at most.
        self.inference_budget=InferenceBudget(interval_frames=8,window_frames=25,maximum_calls=2)

    def close(self):
        if self.decoder is not None:
            self.decoder.release()
            self.decoder = None

    def invalidate(self):
        """A manual configuration change starts a new checkpoint interval."""
        self.checkpoints.clear()
        self.observations.clear()
        self.first_problem=None
        self.attempts=0
        self.trusted_run=0

    def remember(self, owner, source):
        started=time.perf_counter()
        point = getattr(owner, 'point_start_frame_internal', None)
        if point != self.point:
            self.point = point
            self.checkpoints.clear()
            self.observations.clear()
            self.attempts = 0
            self.first_problem = None
            self.unresolved = False
            self.unresolved_since = None
            self.hard_uncertainty = False
            self.hard_since = None
            self.contact_uncertainty = False
            self.prestrike_uncertainty = False
            self.trusted_run = 0
        if point is None or owner.ball_center is None:
            return
        if self.observations and (
                owner.frame_count <= self.observations[-1]['frame'] or
                int(source)-self.observations[-1]['source'] != owner.frame_count-self.observations[-1]['frame']):
            self.checkpoints.clear()
            self.observations.clear()
            self.first_problem=None
            self.attempts=0
        tracker = getattr(owner, 'player_tracker', None)
        observation = dict(frame=int(owner.frame_count), source=int(source),
                           pos=tuple(owner.ball_center),
                           area=float(owner.ball_size or 0),
                           motion_mean=float(getattr(owner,'_last_tracked_candidate_motion_mean',0)),
                           motion_max=float(getattr(owner,'_last_tracked_candidate_motion_max',0)),
                           tracks=copy.deepcopy(tracker.tracks) if tracker is not None else None)
        self.observations.append(observation)
        if self.hard_uncertainty and not self.contact_uncertainty and not self.prestrike_uncertainty:
            events=(getattr(owner,'_point_history_current',None) or {}).get('shot_events',[])
            contradiction=any(a.get('player') and a.get('player')==b.get('player') and
                              b['frame']-a['frame']>=4 for a,b in zip(events,events[1:]))
            if not contradiction:
                # The normal tracker's own atomic rollback may already have
                # removed the provisional bad hit before this frame completed.
                self.hard_uncertainty=False
                self.hard_since=None
        motion = owner.last_motion or {}
        toss = prestrike_toss_checkpoint(owner)
        observation['prestrike_toss'] = toss
        trusted = (owner.stuck_frame_count == 0 and owner.ball_size is not None and
                   ((getattr(owner,'_serve_landed_in_current_attempt',False) and
                     not getattr(owner,'_awaiting_serve_bounce',True)) or
                    getattr(owner,'_point_hit_count',0) > 0 or toss) and
                   owner.ball_size >= 3 and owner._player_point_zone(owner.ball_center) is None and
                   getattr(owner, '_last_tracked_candidate_motion_frame', -1) == owner.frame_count and
                   getattr(owner, '_last_tracked_candidate_motion_mean', 0) >= 6 and
                   getattr(owner, '_last_tracked_candidate_motion_max', 0) >= 35 and
                   2 <= motion.get('distance', 0) <= 400 and
                   not getattr(owner, '_pending_rally_end_reason', None))
        self.trusted_run = self.trusted_run+1 if trusted else 0
        observation['trusted']=bool(trusted and self.trusted_run >= 2)
        if (self.first_problem is not None and self.trusted_run >= 4 and
                owner.frame_count-self.first_problem > self.maximum_window):
            # Normal tracking can regain the ball independently. Keep the
            # unresolved segment flag, but allow recovery of a later failure.
            self.first_problem = None
            self.attempts = 0
        if trusted and self.trusted_run >= 2 and self.first_problem is None:
            self.checkpoints.append(Checkpoint(owner.frame_count, int(source), point,
                tuple(owner.ball_center), (motion.get('dx', 0), motion.get('dy', 0)),
                float(owner.ball_size), snapshot(owner),prestrike_toss=toss))
        self.checkpoint_seconds+=time.perf_counter()-started

    def reject(self, frame, reason):
        self.last_rejection=reason
        if self.attempts >= 2:
            self.unresolved=True
            self.unresolved_since=self.unresolved_since or self.first_problem or frame
        print(f'[SHORT_REWIND_REJECTED] f{frame}: {reason}')
        return None

    def problem(self, owner, previous, current, previous_stuck):
        if self.point != getattr(owner, 'point_start_frame_internal', None):
            return None
        if (getattr(owner, '_far_contact_pending', None) or getattr(owner, '_near_contact_pending', None) or
                any(getattr(owner,k,-1) == owner.frame_count for k in (
                    '_temporal_near_contact_frame','_temporal_far_contact_frame',
                    '_temporal_top_reentry_frame','_measured_far_net_frame')) or
                owner._top_return_wait_active() or owner._back_return_wait_active()):
            return None
        if max(previous_stuck, owner.stuck_frame_count) >= 2:
            return 'ball-miss'
        if previous is None or current is None:
            return None
        history = (getattr(owner, '_point_history_current', None) or {}).get('shot_events', [])
        if len(history) >= 2:
            a, b = history[-2:]
            if (b.get('frame') == owner.frame_count and a.get('player') and
                    a.get('player') == b.get('player') and 4 <= b['frame']-a['frame'] <= 180):
                return 'repeated-same-player-hit'
        recent = list(self.observations)[-1:]
        if (self.checkpoints and recent and recent[0].get('prestrike_toss') and
                recent[0]['pos'][1] <= getattr(owner, 'net_area_y_max', 0) and
                current[1] <= getattr(owner, 'net_area_y_max', 0) and
                float(getattr(owner, 'ball_size', 0) or 0) < max(8, recent[0]['area'] * .08) and
                math.dist(previous, current) >= 12):
            from near_serve_scan_recovery import near_toss_tracking
            if near_toss_tracking(owner):
                return 'prestrike-toss-fragment'
        if (getattr(owner, '_configured_causal_evidence', None) or {}).get('frame') == owner.frame_count:
            return None
        step = math.dist(previous, current)
        speeds = list(owner.ball_velocity_history)[-6:-1]
        typical = float(np.median(speeds)) if speeds else 0
        zone = owner._player_point_zone(current)
        recent=list(self.observations)[-2:]
        if (zone is not None and len(recent)==2 and self.checkpoints and
                owner.frame_count-getattr(owner,'_last_racket_contact_frame',-1000000) >= 5 and
                0 <= owner.frame_count-self.checkpoints[-1].frame <= 12 and
                math.hypot(*self.checkpoints[-1].velocity) >= 10 and
                all(owner._player_point_zone(o['pos']) is not None for o in recent) and
                max(math.dist(current,o['pos']) for o in recent) <= 40):
            return 'player-lock'
        if zone is not None and step > max(90, typical*2.8):
            return 'player-jump'
        if (getattr(owner, '_last_tracked_candidate_motion_frame', -1) == owner.frame_count and
                getattr(owner, '_last_tracked_candidate_motion_mean', 0) < 2 and step > 40):
            return 'static-jump'
        return None

    def choose(self, current):
        width = self.window if self.attempts == 0 else self.maximum_window
        available = [c for c in self.checkpoints if c.point == self.point and
                     current-width <= c.frame <= current-3]
        if not available and width < self.maximum_window:
            available=[c for c in self.checkpoints if c.point == self.point and
                       current-self.maximum_window <= c.frame <= current-3]
        return available[0] if available else None

    def try_recover(self, owner, frame, reason, source):
        current = int(owner.frame_count)
        if reason in ('repeated-same-player-hit', 'prestrike-toss-fragment'):
            if reason == 'prestrike-toss-fragment':
                self.prestrike_uncertainty = True
            self.unresolved=True
            self.unresolved_since=self.unresolved_since or current
            if not self.hard_uncertainty:
                recorder=getattr(owner,'_record_ball_loss_event',None)
                if recorder is not None:
                    recorder(f'short rewind required: {reason}',recovery='score remains provisional')
            self.hard_uncertainty=True
            self.hard_since=self.hard_since or current
        if self.attempts >= 2 or current-self.last_attempt < 8:
            return None
        recovery = owner.local_ai_recovery
        checkpoint = self.choose(current)
        if checkpoint is None:
            return None
        self.first_problem = self.first_problem or current
        if not self.inference_budget.ready(current):
            return None
        self.attempts += 1
        self.total_attempts += 1
        self.last_attempt = current
        print(f'[SHORT_REWIND_START] f{current}: anchor=f{checkpoint.frame} reason={reason} attempt={self.attempts}')
        recorder=getattr(owner,'_record_ball_loss_event',None)
        if recorder is not None:
            recorder(f'short rewind: {reason}',recovery=f'AI replay from f{checkpoint.frame}')
        if self.decoder is None:
            self.decoder = cv2.VideoCapture(str(owner.video_path))
        self.decoder.set(cv2.CAP_PROP_POS_FRAMES, checkpoint.source)
        ok, anchor_image = self.decoder.read()
        if not ok or int(round(self.decoder.get(cv2.CAP_PROP_POS_FRAMES)))-1 != checkpoint.source:
            return self.reject(current,'anchor-decode')
        # Explicit source indices preserve the caller's frame-label convention.
        ledger = {o['frame']:o for o in self.observations if o['frame'] > checkpoint.frame}
        ledger[current] = dict(frame=current, source=int(source), pos=owner.ball_center,
            tracks=copy.deepcopy(owner.player_tracker.tracks) if owner.player_tracker is not None else None)
        labels = list(range(checkpoint.frame+1, current+1))
        if any(n not in ledger for n in labels):
            return self.reject(current,'noncontiguous-history')
        images, pools, previous = [], [], cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
        occluded=set()
        expected_source = checkpoint.source+1
        for n in labels:
            if ledger[n]['source'] != expected_source:
                return self.reject(current,'noncontiguous-source')
            ok, image = self.decoder.read()
            if not ok or int(round(self.decoder.get(cv2.CAP_PROP_POS_FRAMES)))-1 != expected_source:
                return self.reject(current,'segment-decode')
            pool = fresh_candidates(image, previous, checkpoint.position, checkpoint.velocity,
                                    n-checkpoint.frame, checkpoint.area)
            tracks=ledger[n]['tracks'] or {}
            for candidate in pool:
                candidate['player_zone']=None
                for side, track in tracks.items():
                    box=getattr(track,'bbox',None)
                    if box is not None:
                        x,y,w,h=box
                        if x-8 <= candidate['x'] <= x+w+8 and y-8 <= candidate['y'] <= y+h+8:
                            candidate['player_zone']=side
                            break
            for track in tracks.values():
                box=getattr(track,'bbox',None)
                if box is not None:
                    x,y,w,h=box
                    lx,ly=ledger[n]['pos']
                    if x-20 <= lx <= x+w+20 and y-20 <= ly <= y+h+20:
                        occluded.add(n)
            if (self.attempts == 1 and ledger[n].get('trusted',False) and
                    n <= self.first_problem-3):
                # Reuse independently verified history before the suspected
                # error; spend inference on the ambiguous part of the window.
                lx,ly=ledger[n]['pos']
                pool=[dict(x=lx,y=ly,area=ledger[n]['area'],trusted_prefix=True,
                           motion_mean=ledger[n]['motion_mean'],motion_max=ledger[n]['motion_max'],
                           player_zone=None,ai_score=1.)]
            images.append(image)
            pools.append(pool)
            previous = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            expected_source += 1
        try:
            if reason == 'prestrike-toss-fragment':
                toss_path=prestrike_toss_path(checkpoint,images,labels,ledger,anchor_image)
                if toss_path is not None:
                    print(f'[PRESTRIKE_TOSS_REPLAY] f{current}: verified {len(toss_path)} fresh HSV/motion observations')
                    return self.apply(owner,checkpoint,toss_path,images,labels,ledger,anchor_image)
            scored=[pool if pool and pool[0].get('trusted_prefix',False) else None for pool in pools]
            requested=[i for i,g in enumerate(scored) if g is None]
            ranked = recovery._score_batch([(images[i],labels[i],pools[i]) for i in requested],
                                            inference_budget=self.inference_budget)
            if len(ranked) != len(requested):
                raise ReplayRejected('invalid-score-batch')
            for i, group in zip(requested,ranked):
                scored[i]=group
            # Preserve motion evidence; the worker only returns appearance data.
            for pool, group in zip(pools, scored):
                by_pos = {(c['x'], c['y']):c for c in pool}
                for candidate in group:
                    candidate.update({k:v for k,v in by_pos.get((candidate['x'],candidate['y']),{}).items()
                                      if k not in candidate})
            path = best_path(list(zip(labels, scored)), checkpoint.frame, checkpoint.position,
                             checkpoint.velocity, recovery.minimum_score,occlusion_frames=occluded)
            if path is None:
                raise ReplayRejected('no-coherent-ai-path')
            return self.apply(owner, checkpoint, path, images, labels, ledger, anchor_image)
        except Exception as error:
            return self.reject(current,str(error))

    def apply(self, owner, checkpoint, path, images, labels, ledger, anchor_image):
        """All event changes are provisional until every replayed step passes."""
        before = snapshot(owner)
        buffer_state=[(sample,sample.get('normal_position'),'normal_position' in sample)
                      for sample in getattr(owner,'_local_ai_frame_buffer',[])]
        marker_state={key:(hasattr(owner,key),getattr(owner,key,None)) for key in
                      ('_short_rewind_accepted_frame','_short_rewind_accepted_pos')}
        image_state = {k:getattr(owner,k,None) for k in
                       ('_prev_frame_gray','_terminal_previous_gray','_terminal_current_gray','_terminal_current_frame')}
        tracker = getattr(owner, 'player_tracker', None)
        profile_path = getattr(tracker, 'profile_path', None)
        player_live=({k:clone(v) if isinstance(v,_VALUES) else v for k,v in vars(tracker).items()
                      if k not in ('profile','point_events')} if tracker is not None else {})
        by_frame = {c['frame']:c for c in path}
        final_frame = labels[-1]
        semantic_uncertainty=False
        try:
            restore(owner, checkpoint.state)
            owner._short_rewind_replaying = True
            if tracker is not None:
                tracker.profile_path = None  # Learn provisionally; never write during replay.
            previous_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
            for n, image in zip(labels, images):
                owner.frame_count = n
                owner._terminal_previous_gray = previous_gray
                owner._prev_frame_gray = previous_gray
                owner._terminal_current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                owner._terminal_current_frame = image
                if tracker is not None and ledger[n]['tracks'] is not None:
                    tracker.tracks = copy.deepcopy(ledger[n]['tracks'])
                candidate = by_frame.get(n)
                if candidate is not None:
                    old_pos = owner.ball_center
                    dx, dy = candidate['pos'][0]-old_pos[0], candidate['pos'][1]-old_pos[1]
                    old_motion = owner.last_motion or {}
                    near_contact = (old_motion.get('dy',0) >= 12 and dy <= -12 and
                                    old_pos[1] > owner.net_area_y_max and
                                    owner._player_point_zone(old_pos) in ('player_body','racket_fragment'))
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    x, y = candidate['pos']
                    committed = dict(candidate, hsv=hsv[y,x], verified_contact_departure=near_contact)
                    contact_before=getattr(owner,'_last_racket_contact_frame',-1000000)
                    bounces_before=getattr(owner,'ground_bounce_count',0)
                    hitter_before=getattr(owner,'_last_racket_contact_player',None)
                    opponent_frames=getattr(owner,'_side_recross_opponent_frames',0)
                    owner._commit_local_ball_flight(committed, image)
                    if (old_motion.get('dy',0) >= 4 and dy <= -4 and
                            getattr(owner,'_last_racket_contact_frame',-1000000) == contact_before and
                            getattr(owner,'ground_bounce_count',0) == bounces_before and
                            not getattr(owner,'_pending_rally_end_reason',None)):
                        semantic_uncertainty=True
                    inferred=owner._maybe_infer_return_contact_from_side_recross(owner.ball_center, image, previous_position=old_pos)
                    if (not inferred and hitter_before in (0,1) and opponent_frames >= 3 and
                            owner._player_index_at_point(owner.ball_center,image) == hitter_before and
                            contact_before <= checkpoint.frame):
                        # The ball returned from the opponent, but the contact
                        # was outside this replay or lacked image support.
                        semantic_uncertainty=True
                    events=(owner._point_history_current or {}).get('shot_events',[])
                    if len(events) >= 2 and events[-1].get('frame') == n:
                        a,b=events[-2:]
                        if a.get('player') and a.get('player') == b.get('player') and b['frame']-a['frame'] >= 4:
                            semantic_uncertainty=True
                    if n < final_frame and getattr(owner,'_pending_rally_end_reason',None):
                        raise ReplayRejected('terminal-event-in-prefix')
                    if n < final_frame and owner._point_history_current is not None:
                        owner._point_history_current.setdefault('tracking_trace',[]).append(dict(
                            frame=n,pos=list(owner.ball_center),size=owner.ball_size,stuck=0))
                else:
                    owner.stuck_frame_count += 1
                previous_gray = owner._terminal_current_gray
            if tuple(owner.ball_center) != tuple(path[-1]['pos']):
                raise ReplayRejected('final-position-mismatch')
            if tracker is not None:
                for key,value in player_live.items():
                    setattr(tracker,key,value)
                tracker.profile_path = profile_path
            owner._short_rewind_replaying = False
            owner._short_rewind_accepted_frame = final_frame
            owner._short_rewind_accepted_pos = tuple(owner.ball_center)
            owner._last_motion_reacq_frame = final_frame
            owner._last_motion_reacq_pos = tuple(owner.ball_center)
            hsv_toss_observations=sum(c.get('replay_evidence')=='prestrike-hsv-motion' for c in path)
            if hsv_toss_observations:
                owner._prestrike_toss_replay_count=int(getattr(owner,'_prestrike_toss_replay_count',0))+1
            else:
                owner._local_ai_recovery_count=int(getattr(owner,'_local_ai_recovery_count',0))+1
            owner._local_ai_follow_until_frame=final_frame+2
            for sample in getattr(owner,'_local_ai_frame_buffer',[]):
                if checkpoint.frame < sample.get('frame',-1) <= final_frame:
                    corrected=by_frame.get(sample['frame'])
                    sample['normal_position']=corrected['pos'] if corrected is not None else None
            recorder=getattr(owner,'_record_ball_loss_event',None)
            if recorder is not None:
                recorder('short rewind verified',recovery=f'replaced {len(labels)} frames from f{checkpoint.frame}; '
                         f'event history {"provisional" if semantic_uncertainty else "verified"}')
            ai_observations=sum(not c.get('trusted_prefix',False) and 'ai_score' in c for c in path)
            print(f'[SHORT_REWIND_ACCEPTED] f{final_frame}: replayed={len(labels)} ai_observations={ai_observations} '
                  f'hsv_toss_observations={hsv_toss_observations} pos={owner.ball_center} events_provisional={semantic_uncertainty}')
            self.accepted += 1
            if semantic_uncertainty:
                self.unresolved=True
                self.hard_uncertainty=True
                self.contact_uncertainty=True
                self.hard_since=min(self.hard_since or checkpoint.frame-1,checkpoint.frame-1)
                self.unresolved_since=self.unresolved_since or checkpoint.frame-1
                self.checkpoints=deque((c for c in self.checkpoints if c.frame<=checkpoint.frame),
                                       maxlen=self.maximum_window+1)
                for observed in self.observations:
                    corrected=by_frame.get(observed['frame'])
                    if corrected is not None:
                        observed.update(pos=corrected['pos'],area=corrected['area'],trusted=False)
            else:
                self.checkpoints.clear()
                self.observations.clear()
                self.first_problem = None
                self.attempts = 0
                if self.hard_since is None or checkpoint.frame <= self.hard_since:
                    self.hard_uncertainty=False
                    self.hard_since=None
                    self.contact_uncertainty=False
                    self.prestrike_uncertainty=False
                if self.unresolved_since is None or checkpoint.frame <= self.unresolved_since:
                    self.unresolved = False
                    self.unresolved_since = None
            return tuple(owner.ball_center)
        except Exception:
            restore(owner, before)
            for k,v in image_state.items():
                setattr(owner,k,v)
            for sample,position,present in buffer_state:
                if present:
                    sample['normal_position']=position
                else:
                    sample.pop('normal_position',None)
            for key,(present,value) in marker_state.items():
                if present:
                    setattr(owner,key,value)
                elif hasattr(owner,key):
                    delattr(owner,key)
            if tracker is not None:
                tracker.profile_path = profile_path
            owner._short_rewind_replaying = False
            raise
