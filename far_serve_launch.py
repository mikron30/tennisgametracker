"""Motion evidence for a far serve whose toss leaves the top of the image."""
import math


def outgoing_candidate_path(samples, origin=None, radius=180):
    """Resolve a serve/shirt ambiguity using four fresh, consecutive images."""
    if len(samples) < 4:
        return None
    samples = samples[-4:]
    if any(b['frame'] - a['frame'] != 1 for a, b in zip(samples, samples[1:])):
        return None
    paths = [([candidate], 0.) for candidate in samples[0]['candidates']
             if origin is None or math.dist(candidate['pos'], origin) <= radius]
    for sample in samples[1:]:
        extended = []
        for path, cost in paths:
            for candidate in sample['candidates']:
                dx = candidate['pos'][0] - path[-1]['pos'][0]
                dy = candidate['pos'][1] - path[-1]['pos'][1]
                if not (dy >= 10 and 18 <= math.hypot(dx, dy) <= 90):
                    continue
                sizes = [item['area'] for item in path] + [candidate['area']]
                if max(sizes) > min(sizes) * 4:
                    continue
                change = 0.
                if len(path) >= 2:
                    previous_dx = path[-1]['pos'][0] - path[-2]['pos'][0]
                    previous_dy = path[-1]['pos'][1] - path[-2]['pos'][1]
                    change = math.hypot(dx - previous_dx, dy - previous_dy)
                    if change > 20:
                        continue
                extended.append((path + [candidate], cost + change))
        # Keep a bounded set when a player produces many moving fragments.
        paths = sorted(extended, key=lambda item: item[1])[:64]
    valid = [(path, cost) for path, cost in paths
             if path[-1].get('player_zone') is None and
             outgoing_flight([c['pos'] for c in path],
                             [s['frame'] for s in samples], allow_diagonal=True)]
    if not valid:
        return None
    path = min(valid, key=lambda item: item[1])[0]
    return [dict(candidate, frame=sample['frame']) for candidate, sample in zip(path, samples)]


def rising_toss(history, frames, serve_height, current_frame):
    """Find a measured rising segment, ignoring later racket/body fragments."""
    if len(history) < 4 or len(frames) != len(history):
        return None
    candidates = []
    for apex in range(3, len(history)):
        start = apex
        while start > 0:
            gap = frames[start] - frames[start - 1]
            dy = history[start - 1][1] - history[start][1]
            if not 1 <= gap <= 3 or dy < -3:
                break
            start -= 1
        segment = history[start:apex + 1]
        if len(segment) < 4:
            continue
        rise = float(segment[0][1] - segment[-1][1])
        steps = sum(segment[i - 1][1] - segment[i][1] >= 5
                    for i in range(1, len(segment)))
        if rise < max(50, serve_height * .35) or steps < 3:
            continue
        if max(p[0] for p in segment) - min(p[0] for p in segment) > max(60, rise * .6):
            continue
        if not 0 <= current_frame - frames[apex] <= 45:
            continue
        candidates.append(dict(low_pos=segment[0], apex_pos=segment[-1],
                               apex_index=apex, rise=rise, up_steps=steps,
                               low_frame=frames[start], apex_frame=frames[apex]))
    return min(candidates, key=lambda item: item['apex_pos'][1]) if candidates else None


def outgoing_flight(history, frames, allow_diagonal=False):
    """Require three consecutive fast netward steps, rather than a falling toss."""
    if len(history) < 4 or len(frames) != len(history):
        return False
    steps = []
    for index in range(len(history) - 3, len(history)):
        dx = history[index][0] - history[index - 1][0]
        dy = history[index][1] - history[index - 1][1]
        if frames[index] - frames[index - 1] != 1:
            return False
        steps.append((dx, dy))
    if all(dy >= 18 and 20 <= math.hypot(dx, dy) <= 90 for dx, dy in steps):
        return True
    if not allow_diagonal:
        return False
    # A diagonal serve can have less vertical motion than the old threshold.
    # Require a sustained lateral departure as well as netward acceleration,
    # so an ordinary nearly vertical falling toss remains ineligible.
    lateral = sum(dx for dx, _ in steps)
    return (all(dy >= 10 and 18 <= math.hypot(dx, dy) <= 90 for dx, dy in steps)
            and all(dx * lateral > 0 for dx, _ in steps)
            and abs(lateral) >= max(35, sum(dy for _, dy in steps) * .4))


def top_reentry_limit(history, frames, serve_height, current_frame):
    context = rising_toss(history, frames, serve_height, current_frame)
    if (context is None or not history or
            history[-1][1] > max(40, serve_height * .3) or
            current_frame - frames[-1] <= 1):
        return None
    # A missed contour near the top is not an exit. A toss already falling
    # back into view must remain searchable below the top strip.
    dy = history[-1][1] - history[-2][1]
    elapsed = frames[-1] - frames[-2]
    if elapsed <= 0 or dy >= 0:
        return None
    # After a rising toss disappears, its first visible sample must enter
    # through the top strip, not at the player's shirt/racket lower down.
    return max(50, serve_height * .35)


def prestance_toss(history, frames, serve_height, current_frame):
    """Retain a live rising toss when a delayed legal stance is certified."""
    context = rising_toss(history, frames, serve_height, current_frame)
    if (context is None or context['rise'] < 90 or
            context['apex_index'] < len(history) - 2 or
            current_frame - context['apex_frame'] > 2 or
            history[-1][1] > context['apex_pos'][1] + 16):
        return None
    return context


def flight_stance_anchor(position, history, frames, serve_height, current_frame):
    """Validate the original toss, not its diagonally departing serve ball."""
    context = rising_toss(history, frames, serve_height, current_frame)
    if context is None or position[1] < serve_height:
        return position
    preceding = history[:-1] if history and history[-1] == position else history
    if len(preceding) < 2:
        return position
    dx = position[0] - preceding[-1][0]
    dy = position[1] - preceding[-1][1]
    previous_dy = preceding[-1][1] - preceding[-2][1]
    if dy >= 8 and previous_dy >= 8 and math.hypot(dx, dy) <= 90:
        return context['low_pos']
    return position
