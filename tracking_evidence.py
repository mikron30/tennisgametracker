"""Image and trajectory evidence for ambiguous tracking decisions."""
import cv2
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
