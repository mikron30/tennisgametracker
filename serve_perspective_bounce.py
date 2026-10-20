"""Confirm a service impact when perspective hides the vertical reversal."""
import math


def impact(samples, projected=False):
    if len(samples) < 6:
        return None
    samples = samples[-9:]
    for start in range(len(samples) - 5):
        landing = _impact_window(samples[start:start+6])
        if landing is not None:
            return landing
    for start in range(len(samples) - 6):
        landing = _impact_window(samples[start:], diagonal=True)
        if landing is not None:
            return landing
    if projected:
        for start in range(len(samples) - 5):
            window = samples[start:]
            if any(b['frame'] - a['frame'] != 1 for a, b in zip(window, window[1:])):
                continue
            steps = [(b['pos'][0] - a['pos'][0], b['pos'][1] - a['pos'][1])
                     for a, b in zip(window, window[1:])]
            incoming = steps[1][1]
            low = min(dy for _, dy in steps[3:-1])
            if (steps[0][1] >= 35 and incoming >= steps[0][1] * .8 and
                    -8 <= steps[2][1] <= incoming * .65 and
                    all(dy >= -8 and abs(dx - steps[1][0]) <= max(12, abs(steps[1][0]) * .6)
                        for dx, dy in steps[2:]) and
                    all(abs(dx) <= dy for dx, dy in steps[:2]) and
                    steps[-1][1] >= max(low + 8, low * 1.3) and
                    min(s['area'] for s in window) > 0 and
                    max(s['area'] for s in window) <= min(s['area'] for s in window) * 4):
                return window[3]
    return None


def _impact_window(samples, diagonal=False):
    if any(samples[i]['frame'] != samples[i-1]['frame'] + 1 for i in range(1, len(samples))):
        return None
    steps = [(samples[i]['pos'][0] - samples[i-1]['pos'][0],
              samples[i]['pos'][1] - samples[i-1]['pos'][1]) for i in range(1, len(samples))]
    speeds = [math.hypot(*step) for step in steps]
    a, b, drop = speeds[:3]
    low, rebound = min(speeds[3:-1]), speeds[-1]
    rebound_ratio = 1.12 if diagonal else 1.3
    if not (a >= 30 and b >= a * .8 and 6 <= drop <= b * .4 and
            6 <= low <= drop * 1.3 and rebound >= max(low * rebound_ratio, low + 4)):
        return None
    # Continuous netward travel, no stop, player fragment, or lateral jump.
    if not diagonal:
        if any(dy <= 0 or abs(dx) > dy * .5 for dx, dy in steps):
            return None
    elif (any(dy <= 0 or abs(dx) > dy * .75 for dx, dy in steps[:2]) or
          any(abs(dy) > max(4, abs(dx)*.15) for dx,dy in steps[2:4])):
        return None
    # A diagonal rebound can move almost horizontally in the camera image.
    # Keep its lateral velocity continuous instead of demanding vertical travel.
    incoming_dx = steps[1][0]
    if any(dy < 0 or abs(dx - incoming_dx) > max(12, abs(incoming_dx) * .6)
           for dx, dy in steps[2:]):
        return None
    sizes = [sample['area'] for sample in samples]
    if min(sizes) <= 0 or max(sizes) > min(sizes) * 4:
        return None
    return samples[3]
