"""Confirm a service impact when perspective hides the vertical reversal."""
import math


def impact(samples):
    if len(samples) < 6:
        return None
    samples = samples[-6:]
    if any(samples[i]['frame'] != samples[i-1]['frame'] + 1 for i in range(1, 6)):
        return None
    steps = [(samples[i]['pos'][0] - samples[i-1]['pos'][0],
              samples[i]['pos'][1] - samples[i-1]['pos'][1]) for i in range(1, 6)]
    speeds = [math.hypot(*step) for step in steps]
    a, b, drop, low, rebound = speeds
    if not (a >= 30 and b >= a * .8 and 6 <= drop <= b * .4 and
            6 <= low <= drop * 1.3 and rebound >= max(low * 1.3, low + 4)):
        return None
    # Continuous netward travel, no stop, player fragment, or lateral jump.
    if any(dy <= 0 or abs(dx) > dy * .5 for dx, dy in steps):
        return None
    sizes = [sample['area'] for sample in samples]
    if min(sizes) <= 0 or max(sizes) > min(sizes) * 4:
        return None
    return samples[3]
