"""Verify a night sideline turn against current image motion before scoring."""
import math
import cv2
import numpy as np


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
    # Only investigate sudden candidate switches, not ordinary rebounds.
    if speed < 6 or current_speed < max(70, 2.5 * speed):
        return False
    image = getattr(analyzer, '_terminal_current_frame', None)
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
