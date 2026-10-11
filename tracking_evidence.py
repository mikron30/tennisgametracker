"""Image and trajectory evidence for ambiguous tracking decisions."""
import cv2
import math
import numpy as np


def player_box_motion(analyzer, bbox):
    before = getattr(analyzer, '_terminal_previous_gray', None)
    after = getattr(analyzer, '_terminal_current_gray', None)
    if before is None or after is None or before.shape != after.shape:
        return None
    x,y,w,h = map(int,bbox)
    # Central body excludes court-line motion at the box edges.
    x1,x2=max(0,x+w//5),min(after.shape[1],x+4*w//5)
    y1,y2=max(0,y+h//5),min(after.shape[0],y+4*h//5)
    if x2<=x1 or y2<=y1:
        return None
    diff=cv2.absdiff(before[y1:y2,x1:x2],after[y1:y2,x1:x2])
    return float(np.count_nonzero(diff>=18))/diff.size


def unsupported_size_collapse(override_area, current_area, current, override):
    """Don't replace a moving, closer ball with a tiny off-path fragment."""
    cd,od=current.get('predicted_distance'),override.get('predicted_distance')
    return bool(current_area>=25 and override_area<current_area*.2
                and override_area<10 and cd is not None and od is not None
                and od>cd+35 and current.get('motion_max',0)>=20
                and not current.get('static_hotspot',False)
                and not current.get('frame0_background',False))


def unsupported_fragment_jump(previous_area, candidate_area, metrics, frame_height,
                              previous_pos=None, candidate_pos=None, previous_motion=None):
    """Require motion or trajectory support for a distant, much smaller fragment.

    Compare against the previous tracked ball, independently of whether another
    HSV filter finds a usable candidate in this frame. Strongly moving or
    prediction-consistent small balls remain eligible for recovery.
    """
    if previous_area is None or candidate_area is None:
        return False
    step = float(metrics.get('prev_distance', 0.0))
    predicted_distance = metrics.get('predicted_distance')
    if previous_pos is not None and candidate_pos is not None:
        # A widened search forecast is not evidence that a one-frame teleport
        # is plausible. Use the last measured motion, or require image support.
        predicted_distance = None
        if previous_motion is not None:
            predicted_distance = math.hypot(
                candidate_pos[0] - previous_pos[0] - previous_motion.get('dx', 0.0),
                candidate_pos[1] - previous_pos[1] - previous_motion.get('dy', 0.0),
            )
    return bool(
        previous_area >= 8.0
        and candidate_area <= 10.0
        and candidate_area <= previous_area * 0.35
        and step >= max(90.0, float(frame_height) * 0.075)
        and metrics.get('motion_mean', 0.0) < 12.0
        and metrics.get('motion_max', 0.0) < 60.0
        and (predicted_distance is None
             or predicted_distance > max(45.0, step * 0.35))
    )
