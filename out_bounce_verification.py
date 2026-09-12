"""Verify a night sideline turn against current image motion before scoring."""
import math
import cv2
import numpy as np


def _matching_recent_terminal_recovery(analyzer, rx, ry, current_frame, image):
    """Return a recent terminal-recovery vector when it still owns prev_motion.

    ``_resume_from_terminal_motion_candidate`` can intentionally teleport the
    tracker from a stale terminal marker to a separately moving ball. That jump
    is useful for identity recovery but is not a physical ball-flight vector.
    The recovery is explicitly tagged in the point tracking trace, while the
    matching motion-history entry carries the exact old/new positions.  Use both
    records so ordinary large movements cannot be mistaken for a synthetic
    recovery.
    """
    context = getattr(analyzer, '_point_history_current', None)
    trace = context.get('tracking_trace', []) if isinstance(context, dict) else []
    recovery_frames = {}
    for entry in reversed(trace):
        try:
            frame = int(entry.get('frame', -1000000))
        except (TypeError, ValueError):
            continue
        age = current_frame - frame
        if age < 0:
            continue
        if age > 8:
            break
        if entry.get('source') != 'terminal_motion_recovery':
            continue
        pos = entry.get('pos')
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            recovery_frames[frame] = (float(pos[0]), float(pos[1]))

    if not recovery_frames or image is None:
        return None

    synthetic_limit = max(400.0, float(image.shape[1]) * 0.10)
    for entry in reversed(getattr(analyzer, 'motion_history', []) or []):
        try:
            frame = int(entry.get('frame', -1000000))
        except (TypeError, ValueError):
            continue
        if frame not in recovery_frames:
            continue
        pos = entry.get('pos')
        prev_pos = entry.get('prev_pos')
        if not (
            isinstance(pos, (tuple, list)) and len(pos) >= 2 and
            isinstance(prev_pos, (tuple, list)) and len(prev_pos) >= 2
        ):
            continue
        recovered = recovery_frames[frame]
        if math.hypot(float(pos[0]) - recovered[0], float(pos[1]) - recovered[1]) > 4.0:
            continue
        recovery_dx = float(pos[0]) - float(prev_pos[0])
        recovery_dy = float(pos[1]) - float(prev_pos[1])
        recovery_speed = math.hypot(recovery_dx, recovery_dy)
        if recovery_speed <= synthetic_limit:
            continue
        vector_error = math.hypot(rx - recovery_dx, ry - recovery_dy)
        tolerance = max(4.0, recovery_speed * 0.015)
        if vector_error <= tolerance:
            return {
                'frame': frame,
                'pos': (int(round(recovered[0])), int(round(recovered[1]))),
                'speed': recovery_speed,
                'vector_error': vector_error,
                'limit': synthetic_limit,
            }
    return None


def recover_continuing_ball(analyzer, position):
    if not analyzer._is_night_session_config():
        return False
    incoming = getattr(analyzer, 'prev_motion', None)
    current = getattr(analyzer, 'last_motion', None)
    if not incoming or not current:
        return False
    rx, ry = float(incoming['dx']), float(incoming['dy'])
    dx, dy = float(current['dx']), float(current['dy'])
    speed = math.hypot(rx, ry)
    current_speed = math.hypot(dx, dy)

    # A verified motion reacquisition can legitimately move the tracker a long
    # distance from a poisoned HSV anchor to the real ball.  That correction is
    # an identity repair, not an incoming tennis-ball vector.  On the very next
    # frame, the sideline detector must not use that synthetic vector to infer a
    # direction reversal / bounce.  Require both recovery markers to identify
    # the reconstructed previous-frame anchor, and only suppress when the
    # correction is far beyond an ordinary one-frame motion.
    recovery_frame = int(getattr(analyzer, '_last_motion_reacq_frame', -1000000))
    recovery_pos = getattr(analyzer, '_last_motion_reacq_pos', None)
    current_frame = int(getattr(analyzer, 'frame_count', -1000000))
    image = getattr(analyzer, '_terminal_current_frame', None)
    if (
        recovery_frame == current_frame - 1 and
        isinstance(recovery_pos, (tuple, list)) and
        len(recovery_pos) >= 2 and
        image is not None
    ):
        previous_anchor = (
            float(position[0]) - dx,
            float(position[1]) - dy,
        )
        anchor_error = math.hypot(
            previous_anchor[0] - float(recovery_pos[0]),
            previous_anchor[1] - float(recovery_pos[1]),
        )
        synthetic_limit = max(400.0, float(image.shape[1]) * 0.10)
        if anchor_error <= 4.0 and speed > synthetic_limit:
            analyzer._last_out_bounce_suppressed_frame = current_frame
            analyzer._last_out_bounce_suppressed_point = tuple(position)
            print(
                f'Frame {current_frame}: [OUT VERIFY RECOVERY GRACE] '
                f'ignoring synthetic incoming reacquisition vector '
                f'speed={speed:.1f}px anchor_error={anchor_error:.1f}px '
                f'recovery={tuple(recovery_pos)} limit={synthetic_limit:.1f}px'
            )
            return True

    # A timeout/stuck recovery has the same provenance problem, but its marker
    # lives in the point trace rather than ``_last_motion_reacq_*``.  Suppress
    # only while ``prev_motion`` still exactly matches that explicitly tagged
    # recovery jump.  Once a real local step replaces it, normal bounce logic
    # resumes immediately.
    terminal_recovery = _matching_recent_terminal_recovery(
        analyzer, rx, ry, current_frame, image
    )
    if terminal_recovery is not None:
        analyzer._last_out_bounce_suppressed_frame = current_frame
        analyzer._last_out_bounce_suppressed_point = tuple(position)
        print(
            f'Frame {current_frame}: [OUT VERIFY TERMINAL RECOVERY GRACE] '
            f'ignoring synthetic terminal-recovery vector '
            f"source_f={terminal_recovery['frame']} "
            f"recovery={terminal_recovery['pos']} "
            f"speed={terminal_recovery['speed']:.1f}px "
            f"vector_error={terminal_recovery['vector_error']:.1f}px "
            f"limit={terminal_recovery['limit']:.1f}px"
        )
        return True

    # Only investigate sudden candidate switches, not ordinary rebounds.
    if speed < 6 or current_speed < max(70, 2.5 * speed):
        return False
    before = getattr(analyzer, '_terminal_previous_gray', None)
    after = getattr(analyzer, '_terminal_current_gray', None)
    if image is None or before is None or after is None or before.shape != after.shape:
        return False

    # Night footage can occasionally replace the ball with a 1-3 px court or
    # player fragment. If that tiny fragment also requires a one-frame jump
    # beyond the same physical step bound used by the sideline detector, it
    # cannot establish an OUT by itself. Keep the normal tracker state intact
    # and use the existing short OUT-suppression window; the next frames may
    # still prove a real bounce or another terminal condition.
    current_ball_size = float(getattr(analyzer, 'ball_size', 0.0) or 0.0)
    max_physical_step = max(125.0, float(image.shape[1]) * 0.035)
    if current_ball_size <= 3.0 and current_speed > max_physical_step:
        analyzer._last_out_bounce_suppressed_frame = analyzer.frame_count
        analyzer._last_out_bounce_suppressed_point = tuple(position)
        print(
            f'Frame {analyzer.frame_count}: [OUT VERIFY SUPPRESS] '
            f'tiny high-speed night candidate pos={position} '
            f'size={current_ball_size:.1f}px speed={current_speed:.1f} '
            f'limit={max_physical_step:.1f}'
        )
        return True

    candidate = analyzer._terminal_moving_ball_candidate(
        position, image, before, after,
        allow_static_anywhere=False, allow_small_static=False)
    if candidate is None:
        return False
    x, y = map(int, candidate)
    if not (0 <= x < after.shape[1] and 0 <= y < after.shape[0]):
        return False
    region = np.s_[max(0,y-12):y+13, max(0,x-12):x+13]
    diff = cv2.absdiff(before[region], after[region])
    if np.count_nonzero(diff >= 18) / max(1, diff.size) < .035:
        return False
    # Anchor at the PREVIOUS frame's ball, never at the rejected head/blob.
    origin = (float(position[0])-dx, float(position[1])-dy)
    sx, sy = x-origin[0], y-origin[1]
    predicted = (origin[0]+rx, origin[1]+ry)
    error = math.hypot(x-predicted[0], y-predicted[1])
    bad_error = math.hypot(position[0]-predicted[0], position[1]-predicted[1])
    if (rx*sx+ry*sy <= 0 or (abs(ry) >= 6 and ry*sy <= 0)
            or error > max(35, 1.5*speed) or bad_error-error < 50
            or math.hypot(x-position[0], y-position[1]) < 70):
        return False
    distance = math.hypot(sx, sy)
    direction = math.degrees(math.atan2(sy, sx))
    analyzer.ball_center = (x, y)
    analyzer.last_delta = (sx, sy)
    analyzer.last_motion = dict(dx=sx, dy=sy, distance=distance, direction_deg=direction)
    if distance >= 3:
        analyzer.last_nonzero_motion = dict(analyzer.last_motion)
    analyzer.last_seen_frame = analyzer.frame_count
    analyzer.stuck_frame_count = 0
    velocities = getattr(analyzer, 'ball_velocity_history', None)
    if isinstance(velocities, list) and velocities:
        velocities[-1] = distance
    history = getattr(analyzer, 'motion_history', None)
    if isinstance(history, list):
        history[:] = [e for e in history if e.get('frame') != analyzer.frame_count]
        history.append(dict(frame=analyzer.frame_count, pos=(x,y), prev_pos=origin,
                            distance=distance, direction_deg=direction,
                            source='out_bounce_verification'))
    context = getattr(analyzer, '_point_history_current', None)
    if isinstance(context, dict):
        trace = context.setdefault('tracking_trace', [])
        trace[:] = [e for e in trace if e.get('frame') != analyzer.frame_count]
        trace.append(dict(frame=analyzer.frame_count, pos=[x,y], stuck=0,
                          source='out_bounce_verification'))
    print(f'Frame {analyzer.frame_count}: [OUT VERIFY CONTINUE] rejected={position} '
          f'ball={(x,y)} origin={origin} prediction_error={error:.1f}')
    return True
