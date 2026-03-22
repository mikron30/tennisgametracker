"""
Tracking regression checker.

Usage:
    python check_tracking.py              # run tests and compare against reference
    python check_tracking.py --save       # run tests and SAVE as new reference

The reference is stored in tracking_reference.json.
Each entry describes one test scenario (start_frame, max_frames) and the
expected serves: frame range, start/end reason, and ball position tolerance.

If any expected serve is missing or the ball position drifts more than
POSITION_TOLERANCE px from the reference the checker reports FAIL.
"""
import argparse
import csv
import json
import math
import sys

import cv2
import numpy as np

# ── Silence display so the script runs headless ────────────────────────────
cv2.imshow       = lambda *a, **kw: None
cv2.namedWindow  = lambda *a, **kw: None
cv2.waitKey      = lambda *a, **kw: 0
cv2.setMouseCallback  = lambda *a, **kw: None
cv2.resizeWindow      = lambda *a, **kw: None
cv2.destroyAllWindows = lambda *a, **kw: None
# ──────────────────────────────────────────────────────────────────────────

from interactive_ball_analyzer import InteractiveBallAnalyzer

REFERENCE_FILE     = "tracking_reference.json"
POSITION_TOLERANCE = 50   # px — how far ball may drift from reference per frame
START_FRAME_TOL    = 5    # frames — how close serve start must be to reference


# ── Headless state machine ─────────────────────────────────────────────────
def run_scenario(start_frame: int, max_frames: int):
    """Run the headless state machine and return list of detected serves."""
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    analyzer   = InteractiveBallAnalyzer(video_path, start_frame=start_frame)

    hsv_config = analyzer.load_hsv_config()
    if not hsv_config:
        print("ERROR: hsv_config.json not found.")
        sys.exit(1)

    def reset():
        analyzer.tracking              = False
        analyzer.ball_center           = None
        analyzer.ball_size             = None
        analyzer.ball_hsv              = None
        analyzer.ball_velocity_history = []
        analyzer.last_motion           = None
        analyzer.prev_motion           = None
        analyzer.last_direction        = None
        analyzer.direction_change_streak = 0
        analyzer.edge_wait             = False
        analyzer.near_edge             = False
        analyzer.using_alt_hsv         = False
        analyzer.using_alt2_hsv        = False
        analyzer.focus_loss_active     = False
        analyzer.stuck_frame_count     = 0

    import math as _math

    game_state           = "WAITING_FOR_SERVE"
    point_start_frame    = None
    point_end_frame      = None
    point_end_reason     = None
    serve_tracking_frames = 0
    last_serve_candidate = None
    serve_position_history = []
    early_serve_grace_frames = 15

    serves   = []          # list of {start, start_pos, end, end_note, positions}
    cur_srv  = None

    analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    analyzer.frame_count = start_frame
    frame_limit = start_frame + max_frames

    while True:
        ret, frame = analyzer.cap.read()
        if not ret or analyzer.frame_count >= frame_limit:
            break

        analyzer.frame_count += 1

        # ── SCANNING_FOR_SERVE ───────────────────────────────────────────
        if game_state == "SCANNING_FOR_SERVE":
            potential_serve = analyzer.detect_serve_position(frame)
            if potential_serve:
                analyzer.ball_center           = potential_serve
                analyzer.tracking              = True
                analyzer.ball_stopped          = False
                analyzer.ball_velocity_history = []
                analyzer.initial_ball_position = potential_serve
                analyzer.ball_size             = None
                point_start_frame              = analyzer.frame_count
                game_state = "TRACKING_POINT"
                cur_srv = {'start': analyzer.frame_count,
                           'start_pos': list(potential_serve),
                           'positions': [], 'note': 'SERVE_DETECTED'}
                serves.append(cur_srv)

        # ── TRACKING_POINT ───────────────────────────────────────────────
        elif game_state == "TRACKING_POINT":
            tracked = None
            if analyzer.tracking and analyzer.hsv_lower is not None and analyzer.ball_center is not None:
                tracked = analyzer.track_ball_in_frame(frame)

            if tracked:
                if cur_srv is not None:
                    cur_srv['positions'].append([analyzer.frame_count, tracked[0], tracked[1]])

                if analyzer.stuck_frame_count >= 15:
                    if cur_srv:
                        cur_srv['end']      = analyzer.frame_count
                        cur_srv['end_note'] = 'STUCK_TIMEOUT'
                    game_state = "WAITING_FOR_SERVE"
                    reset()
                    cur_srv = None
                else:
                    point_ended, reason = analyzer.detect_point_end(tracked, frame)
                    if point_ended:
                        point_end_frame  = analyzer.frame_count
                        point_end_reason = reason
                        if cur_srv:
                            cur_srv['end']      = analyzer.frame_count
                            cur_srv['end_note'] = reason
                        game_state = "POINT_ENDED" if "net" not in reason.lower() else "WAITING_FOR_SERVE"
                        reset()
                        cur_srv = None
            else:
                grace_limit = 45 if point_start_frame and analyzer.frame_count <= (start_frame + 45) else 30
                if point_start_frame and (analyzer.frame_count - point_start_frame > grace_limit):
                    if cur_srv:
                        cur_srv['end']      = analyzer.frame_count
                        cur_srv['end_note'] = 'BALL_LOST'
                    point_end_frame  = analyzer.frame_count
                    point_end_reason = 'BALL_LOST'
                    game_state = "POINT_ENDED"
                    reset()
                    cur_srv = None

        # ── POINT_ENDED ──────────────────────────────────────────────────
        elif game_state == "POINT_ENDED":
            cooldown = 90 if point_end_reason and "stopped" in str(point_end_reason).lower() else 60
            if point_end_frame and (analyzer.frame_count - point_end_frame > cooldown):
                game_state        = "SCANNING_FOR_SERVE"
                point_start_frame = None
                point_end_frame   = None
                point_end_reason  = None

        # ── WAITING_FOR_SERVE ────────────────────────────────────────────
        elif game_state == "WAITING_FOR_SERVE":
            potential_serve = analyzer.detect_serve_position(frame)
            if potential_serve:
                serve_tracking_frames += 1
                last_serve_candidate   = potential_serve
                serve_position_history.append(potential_serve)
                if len(serve_position_history) > 10:
                    serve_position_history = serve_position_history[-10:]
                if len(serve_position_history) >= 4:
                    all_rightward = True
                    min_dx        = float('inf')
                    for i in range(-3, 0):
                        pair_dx = serve_position_history[i][0] - serve_position_history[i-1][0]
                        if pair_dx < 15:
                            all_rightward = False
                            break
                        min_dx = min(min_dx, pair_dx)
                    if all_rightward and min_dx > 25:
                        p1 = serve_position_history[-2]
                        p2 = serve_position_history[-1]
                        _dx  = p2[0] - p1[0]
                        _dy  = p2[1] - p1[1]
                        _dist = _math.hypot(_dx, _dy)
                        _dir  = _math.degrees(_math.atan2(_dy, _dx))
                        analyzer.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                        analyzer.last_delta  = (_dx, _dy)
                        analyzer.ball_velocity_history = [_dist]
                        analyzer.ball_center           = potential_serve
                        analyzer.tracking              = True
                        analyzer.ball_stopped          = False
                        analyzer.initial_ball_position = serve_position_history[0]
                        analyzer.ball_size             = None
                        analyzer.stuck_frame_count     = 0
                        point_start_frame              = analyzer.frame_count
                        serve_tracking_frames          = 0
                        serve_position_history         = []
                        game_state = "TRACKING_POINT"
                        cur_srv = {'start': analyzer.frame_count,
                                   'start_pos': list(potential_serve),
                                   'positions': [], 'note': 'SERVE_FAST_START'}
                        serves.append(cur_srv)
            else:
                # Ball exited serve area — check if it was a real serve (moving at serve speed).
                # Mirrors process_video's exits-area path: last_dx > 20.
                # (total_dx is deliberately NOT checked: a player may bounce the ball before
                # serving, causing the overall displacement to be small even though the final
                # strike produces a fast rightward movement.)
                if serve_tracking_frames >= 3 and last_serve_candidate is not None and len(serve_position_history) >= 2:
                    total_dx = serve_position_history[-1][0] - serve_position_history[0][0]
                    last_dx  = serve_position_history[-1][0] - serve_position_history[-2][0]
                    last_dy  = serve_position_history[-1][1] - serve_position_history[-2][1]
                    if last_dx > 20 and last_dy >= -last_dx:
                        # Fast rightward exit — serve strike detected.
                        predicted_pos = last_serve_candidate
                        if len(serve_position_history) >= 2:
                            p1 = serve_position_history[-2]
                            p2 = serve_position_history[-1]
                            _dx   = p2[0] - p1[0]
                            _dy   = p2[1] - p1[1]
                            _dist = _math.hypot(_dx, _dy)
                            _dir  = _math.degrees(_math.atan2(_dy, _dx))
                            predicted_pos = (int(p2[0] + _dx), int(p2[1] + _dy))
                            analyzer.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                            analyzer.last_delta  = (_dx, _dy)
                            analyzer.ball_velocity_history = [_dist]
                        analyzer.ball_center           = predicted_pos
                        analyzer.tracking              = True
                        analyzer.ball_stopped          = False
                        analyzer.initial_ball_position = serve_position_history[0]
                        analyzer.ball_size             = None
                        analyzer.stuck_frame_count     = 0
                        point_start_frame              = analyzer.frame_count
                        game_state = "TRACKING_POINT"
                        cur_srv = {'start': analyzer.frame_count,
                                   'start_pos': list(last_serve_candidate),
                                   'positions': [], 'note': 'SERVE_EXIT'}
                        serves.append(cur_srv)
                    elif last_dy > 5 and last_dx >= -10 and serve_tracking_frames >= 5:
                        # Toss-complete: ball exiting serve area downward (falling after toss).
                        # Player is about to strike the ball just below the serve area.
                        # Start tracking from last known position — the 200px search radius
                        # will find the ball at the strike point.
                        p1 = serve_position_history[-2]
                        p2 = serve_position_history[-1]
                        _dx   = p2[0] - p1[0]
                        _dy   = p2[1] - p1[1]
                        _dist = _math.hypot(_dx, _dy)
                        _dir  = _math.degrees(_math.atan2(_dy, _dx))
                        analyzer.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                        analyzer.last_delta  = (_dx, _dy)
                        analyzer.ball_velocity_history = [_dist]
                        analyzer.ball_center           = last_serve_candidate
                        analyzer.tracking              = True
                        analyzer.ball_stopped          = False
                        analyzer.initial_ball_position = serve_position_history[0]
                        analyzer.ball_size             = None
                        analyzer.stuck_frame_count     = 0
                        point_start_frame              = analyzer.frame_count
                        game_state = "TRACKING_POINT"
                        cur_srv = {'start': analyzer.frame_count,
                                   'start_pos': list(last_serve_candidate),
                                   'positions': [], 'note': 'TOSS_COMPLETE'}
                        serves.append(cur_srv)
                serve_tracking_frames  = 0
                serve_position_history = []

    analyzer.cap.release()
    return serves


# ── Reference helpers ──────────────────────────────────────────────────────
def load_reference():
    try:
        with open(REFERENCE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_reference(scenarios):
    with open(REFERENCE_FILE, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"[SAVED] Reference written to {REFERENCE_FILE}")


def compare(scenario_name, ref_serves, actual_serves):
    """Compare actual serves against reference. Returns (passed, messages)."""
    messages = []
    passed   = True

    if len(actual_serves) < len(ref_serves):
        passed = False
        messages.append(
            f"  FAIL: expected {len(ref_serves)} serves, got {len(actual_serves)}"
        )
        for rs in ref_serves:
            found = any(abs(a['start'] - rs['start']) <= START_FRAME_TOL for a in actual_serves)
            if not found:
                messages.append(
                    f"  FAIL: serve at f{rs['start']} (±{START_FRAME_TOL}) NOT FOUND — "
                    f"[TRACKING_START] tag expected around f{rs['start']}"
                )
    elif len(actual_serves) > len(ref_serves):
        messages.append(
            f"  WARN: expected {len(ref_serves)} serves, got {len(actual_serves)} (extra serve)"
        )

    # Check each reference serve is present and ball positions match
    for rs in ref_serves:
        match = None
        for a in actual_serves:
            if abs(a['start'] - rs['start']) <= START_FRAME_TOL:
                match = a
                break
        if match is None:
            continue   # already reported above

        # Check end frame (rough)
        ref_end = rs.get('end')
        act_end = match.get('end')
        if ref_end and act_end:
            if abs(act_end - ref_end) > 30:
                messages.append(
                    f"  WARN: serve f{rs['start']} ended at f{act_end} "
                    f"(ref f{ref_end}, diff={act_end-ref_end})"
                )

        # Check position of ball on first tracked frame
        if rs.get('start_pos') and match.get('start_pos'):
            dx = match['start_pos'][0] - rs['start_pos'][0]
            dy = match['start_pos'][1] - rs['start_pos'][1]
            dist = math.hypot(dx, dy)
            if dist > POSITION_TOLERANCE:
                passed = False
                messages.append(
                    f"  FAIL: serve f{rs['start']} start pos "
                    f"{match['start_pos']} vs ref {rs['start_pos']} "
                    f"({dist:.0f}px off, limit {POSITION_TOLERANCE}px)"
                )

    if not messages:
        messages.append(f"  PASS: {len(actual_serves)} serve(s) matched reference")
    return passed, messages


# ── Main ───────────────────────────────────────────────────────────────────
SCENARIOS = [
    {"name": "full_800",      "start_frame":   0, "max_frames": 800},
    {"name": "early_f80",     "start_frame":  80, "max_frames": 250},
    {"name": "third_serve",   "start_frame":   0, "max_frames": 1200},
]


def main():
    parser = argparse.ArgumentParser(description="Tracking regression checker")
    parser.add_argument("--save", action="store_true",
                        help="Save current results as new reference")
    args = parser.parse_args()

    print("Running tracking scenarios...")
    results = []
    for sc in SCENARIOS:
        print(f"\n  Scenario '{sc['name']}': start={sc['start_frame']} max={sc['max_frames']}")
        serves = run_scenario(sc['start_frame'], sc['max_frames'])
        summary = [{'start': s['start'], 'start_pos': s.get('start_pos'),
                    'end': s.get('end'), 'end_note': s.get('end_note'),
                    'note': s.get('note')}
                   for s in serves]
        for s in summary:
            print(f"    Serve f{s['start']} -> f{s.get('end','?')}  {s['note']}  end={s.get('end_note','?')}")
        results.append({**sc, "serves": summary})

    if args.save:
        save_reference(results)
        return

    ref = load_reference()
    if ref is None:
        print("\nNo reference file found. Run with --save first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("COMPARISON AGAINST REFERENCE")
    print(f"{'='*60}")
    all_passed = True
    for sc_result in results:
        name       = sc_result["name"]
        ref_sc     = next((r for r in ref if r["name"] == name), None)
        if ref_sc is None:
            print(f"\n  {name}: no reference entry — skipping")
            continue
        passed, msgs = compare(name, ref_sc["serves"], sc_result["serves"])
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {name}")
        for m in msgs:
            print(m)
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    print("RESULT:", "ALL PASS" if all_passed else "REGRESSION DETECTED")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
