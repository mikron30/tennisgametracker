"""Configured far-serve colours, including hue intervals crossing zero."""
import cv2
import numpy as np
import math


def static_flight_candidate(speed, distance, motion_mean, motion_max):
    """A moving ball cannot relocate to an unchanged image patch."""
    return (speed >= 20 and distance >= 8 and
            motion_mean < 2 and motion_max < 12)


def follows_locked_path(point, history, predicted, reentry_limit=None):
    """A moving toss may cross an ignored static mark's neighbourhood."""
    if len(history) < 3:
        return False
    if reentry_limit is not None:
        return (0 <= point[1] <= reentry_limit and
                abs(point[0] - history[-1][0]) <= 12)
    steps = [math.dist(history[i - 1], history[i]) for i in (-2, -1)]
    return (all(5 <= step <= 60 for step in steps) and
            math.dist(point, predicted) <= max(12, steps[-1] * .75))


def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for limits in ranges:
        lo = [limits['h_min'], limits['s_min'], limits['v_min']]
        hi = [limits['h_max'], limits['s_max'], limits['v_max']]
        if lo[0] <= hi[0]:
            part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        else:
            part = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, *lo[1:]]), np.array(hi)),
                cv2.inRange(hsv, np.array(lo), np.array([179, *hi[1:]])))
        mask = cv2.bitwise_or(mask, part)
    return mask


def compact_mask(mask, min_area=3):
    """Keep ball-sized contours without adjacent thin court markings."""
    result = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        _, _, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        if (min_area <= area <= 120 and max(width, height) <= 3.2 * min(width, height)
                and perimeter > 0 and 4 * math.pi * area / perimeter ** 2 >= .12):
            cv2.drawContours(result, [contour], -1, 255, -1)
    return result


def fresh_toss_candidate(contour, point, current_gray, previous_gray):
    """A static exclusion must not hide a newly moving compact ball."""
    if previous_gray is None or current_gray.shape != previous_gray.shape:
        return False
    area = cv2.contourArea(contour)
    _, _, width, height = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)
    if not (3 <= area <= 80 and max(width, height) <= 3.2 * min(width, height)
            and perimeter > 0 and 4 * math.pi * area / perimeter ** 2 >= .12):
        return False
    x, y = point
    top, bottom = max(0, y-2), min(current_gray.shape[0], y+3)
    left, right = max(0, x-2), min(current_gray.shape[1], x+3)
    difference = cv2.absdiff(current_gray[top:bottom, left:right],
                            previous_gray[top:bottom, left:right])
    return bool(difference.size and difference.mean() >= 18 and difference.max() >= 35)
