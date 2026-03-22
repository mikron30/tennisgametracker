"""
Headless tracking test — runs the same state machine as process_video
but without any OpenCV display. Outputs a CSV log and a summary.

Usage:
    python run_headless_test.py [--max-frames N] [--start-frame N]
"""
import argparse
import csv
import math
import sys
import cv2
import numpy as np

# ── Silence all display calls so the script runs without a screen ─────────
_orig_imshow = cv2.imshow
_orig_namedWindow = cv2.namedWindow
_orig_waitKey = cv2.waitKey
_orig_setMouseCallback = cv2.setMouseCallback
_orig_resizeWindow = cv2.resizeWindow
cv2.imshow = lambda *a, **kw: None
cv2.namedWindow = lambda *a, **kw: None
cv2.waitKey = lambda *a, **kw: 0            # never-pressed key → play_mode stays True-like
cv2.setMouseCallback = lambda *a, **kw: None
cv2.resizeWindow = lambda *a, **kw: None
cv2.destroyAllWindows = lambda *a, **kw: None
# ──────────────────────────────────────────────────────────────────────────

from interactive_ball_analyzer import InteractiveBallAnalyzer


def run_test(max_frames: int = 800, start_frame: int = 0, log_file: str = "tracking_log.csv"):
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    analyzer = InteractiveBallAnalyzer(video_path, start_frame=start_frame)

    # Load config (sets HSV, serve area, net area)
    hsv_config = analyzer.load_hsv_config()
    if not hsv_config:
        print("ERROR: hsv_config.json not found – cannot run headless test.")
        sys.exit(1)

    # ── Replicate reset_tracking_state from process_video ─────────────────
    def reset_tracking_state():
        analyzer.tracking = False
        analyzer.ball_center = None
        analyzer.ball_size = None
        analyzer.ball_hsv = None
        analyzer.ball_velocity_history = []
        analyzer.last_motion = None
        analyzer.prev_motion = None
        analyzer.last_direction = None
        analyzer.direction_change_streak = 0
        analyzer.edge_wait = False
        analyzer.near_edge = False
        analyzer.using_alt_hsv = False
        analyzer.using_alt2_hsv = False
        analyzer.focus_loss_active = False
        analyzer.stuck_frame_count = 0
        analyzer.tracking_start_frame = analyzer.frame_count

    # ── State machine variables ────────────────────────────────────────────
    game_state = "WAITING_FOR_SERVE"
    point_start_frame = None
    point_end_frame = None
    point_end_reason = None
    serve_tracking_frames = 0
    last_serve_candidate = None
    serve_position_history = []
    early_serve_grace_frames = 15

    # ── Log storage ────────────────────────────────────────────────────────
    log_rows = []          # (frame, state, ball_x, ball_y, stuck, note)
    loss_events = []       # frames where ball was not tracked

    analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    analyzer.frame_count = start_frame

    frame_limit = start_frame + max_frames
    prev_ball = None

    print(f"Running headless test: frames {start_frame}-{frame_limit-1}")
    print("-" * 60)

    while True:
        ret, frame = analyzer.cap.read()
        if not ret or analyzer.frame_count >= frame_limit:
            break

        note = ""

        # ── Early serve grace ──────────────────────────────────────────────
        if game_state == "WAITING_FOR_SERVE" and analyzer.frame_count <= (start_frame + early_serve_grace_frames):
            analyzer.using_alt_hsv = False
            analyzer.using_alt2_hsv = False
            candidate = analyzer.track_ball_in_frame(frame, allow_inactive=True)
            if candidate is not None:
                if hasattr(analyzer, 'serve_area_x_min'):
                    if not (analyzer.serve_area_x_min <= candidate[0] <= analyzer.serve_area_x_max and
                            analyzer.serve_area_y_min <= candidate[1] <= analyzer.serve_area_y_max):
                        candidate = None
                if candidate is not None:
                    serve_tracking_frames += 1
                    last_serve_candidate = candidate
            else:
                if serve_tracking_frames >= 5 and last_serve_candidate is not None:
                    game_state = "TRACKING_POINT"
                    point_start_frame = analyzer.frame_count
                    analyzer.tracking = True
                    analyzer.tracking_start_frame = analyzer.frame_count
                    analyzer.ball_center = last_serve_candidate
                else:
                    serve_tracking_frames = 0
                    last_serve_candidate = None

        analyzer.frame_count += 1

        # ── SCANNING_FOR_SERVE ────────────────────────────────────────────
        if game_state == "SCANNING_FOR_SERVE":
            potential_serve = analyzer.detect_serve_position(frame)
            if potential_serve:
                analyzer.ball_center = potential_serve
                analyzer.tracking = True
                analyzer.tracking_start_frame = analyzer.frame_count
                analyzer.ball_stopped = False
                analyzer.ball_velocity_history = []
                analyzer.initial_ball_position = potential_serve
                analyzer.ball_size = None
                point_start_frame = analyzer.frame_count
                game_state = "TRACKING_POINT"
                note = "SERVE_DETECTED"

        # ── TRACKING_POINT ────────────────────────────────────────────────
        elif game_state == "TRACKING_POINT":
            tracked_position = None
            if analyzer.tracking and analyzer.hsv_lower is not None and analyzer.ball_center is not None:
                tracked_position = analyzer.track_ball_in_frame(frame)

            if tracked_position:
                if analyzer.stuck_frame_count >= 15:
                    note = "STUCK_TIMEOUT"
                    point_end_frame = analyzer.frame_count
                    point_end_reason = "STUCK_TIMEOUT"
                    # Go to POINT_ENDED (not WAITING_FOR_SERVE) so that the
                    # cooldown → SCANNING_FOR_SERVE chain runs and later serves
                    # are not missed.
                    game_state = "POINT_ENDED"
                    reset_tracking_state()
                else:
                    point_ended, reason = analyzer.detect_point_end(tracked_position, frame)
                    if point_ended:
                        note = f"POINT_END:{reason}"
                        point_end_frame = analyzer.frame_count
                        point_end_reason = reason
                        game_state = "POINT_ENDED" if "net" not in reason.lower() else "WAITING_FOR_SERVE"
                        reset_tracking_state()
                    else:
                        note = "OK"
            else:
                grace_limit = 45 if point_start_frame and analyzer.frame_count <= (start_frame + 45) else 30
                if point_start_frame and (analyzer.frame_count - point_start_frame > grace_limit):
                    note = "BALL_LOST"
                    loss_events.append(analyzer.frame_count)
                    point_end_frame = analyzer.frame_count
                    point_end_reason = "BALL_LOST"
                    game_state = "POINT_ENDED"
                    reset_tracking_state()
                else:
                    note = "GRACE"

        # ── POINT_ENDED ───────────────────────────────────────────────────
        elif game_state == "POINT_ENDED":
            # Wait 90 frames after point ends before scanning for next serve.
            # Do NOT re-acquire immediately — that causes cycling with static
            # false positives that have near-zero velocity.
            cooldown = 90 if point_end_reason and "stopped" in point_end_reason.lower() else 60
            if point_end_frame and (analyzer.frame_count - point_end_frame > cooldown):
                game_state = "SCANNING_FOR_SERVE"
                point_start_frame = None
                point_end_frame = None
                point_end_reason = None

        # ── WAITING_FOR_SERVE ─────────────────────────────────────────────
        elif game_state == "WAITING_FOR_SERVE":
            potential_serve = analyzer.detect_serve_position(frame)
            if potential_serve:
                serve_tracking_frames += 1
                last_serve_candidate = potential_serve
                serve_position_history.append(potential_serve)
                if len(serve_position_history) > 10:
                    serve_position_history = serve_position_history[-10:]
                if len(serve_position_history) >= 4:
                    all_rightward = True
                    min_dx = float('inf')
                    for i in range(-3, 0):
                        pair_dx = serve_position_history[i][0] - serve_position_history[i-1][0]
                        if pair_dx < 15:
                            all_rightward = False
                            break
                        min_dx = min(min_dx, pair_dx)
                    if all_rightward and min_dx > 25:
                        predicted_pos = potential_serve
                        if len(serve_position_history) >= 2:
                            p1 = serve_position_history[-2]
                            p2 = serve_position_history[-1]
                            _dx = p2[0] - p1[0]
                            _dy = p2[1] - p1[1]
                            _dist = math.hypot(_dx, _dy)
                            _dir = math.degrees(math.atan2(_dy, _dx))
                            analyzer.last_motion = {'distance': _dist, 'dx': _dx,
                                                     'dy': _dy, 'direction_deg': _dir}
                        analyzer.ball_center = predicted_pos
                        analyzer.tracking = True
                        analyzer.tracking_start_frame = analyzer.frame_count
                        analyzer.ball_stopped = False
                        analyzer.ball_velocity_history = []
                        analyzer.initial_ball_position = predicted_pos
                        analyzer.ball_size = None
                        point_start_frame = analyzer.frame_count
                        serve_tracking_frames = 0
                        serve_position_history = []
                        game_state = "TRACKING_POINT"
                        note = "SERVE_FAST_START"

        # ── Record this frame ─────────────────────────────────────────────
        bx = analyzer.ball_center[0] if analyzer.ball_center else None
        by = analyzer.ball_center[1] if analyzer.ball_center else None

        # Detect tracking jump: ball moved >200px in one frame (false positive)
        if prev_ball and bx is not None and game_state == "TRACKING_POINT":
            jump = math.hypot(bx - prev_ball[0], by - prev_ball[1])
            if jump > 300 and note == "OK":
                note = f"JUMP_{int(jump)}px"
        prev_ball = (bx, by) if bx is not None else prev_ball

        log_rows.append((analyzer.frame_count, game_state, bx, by,
                         analyzer.stuck_frame_count, note))

    # ── Write CSV ──────────────────────────────────────────────────────────
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "state", "ball_x", "ball_y", "stuck", "note"])
        writer.writerows(log_rows)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRACKING SUMMARY  (frames {start_frame}-{analyzer.frame_count-1})")
    print(f"{'='*60}")

    tracking_frames = [r for r in log_rows if r[1] == "TRACKING_POINT"]
    ok_frames       = [r for r in tracking_frames if r[5] in ("OK", "")]
    jump_frames     = [r for r in tracking_frames if r[5].startswith("JUMP")]
    lost_frames     = [r for r in log_rows if r[5] in ("BALL_LOST",)]
    serve_starts    = [r for r in log_rows if "SERVE" in r[5]]

    print(f"Frames tracked (OK)  : {len(ok_frames)}")
    print(f"Ball lost events     : {len(lost_frames)}  at frames: {[r[0] for r in lost_frames]}")
    print(f"Jump events (>300px) : {len(jump_frames)}  at frames: {[r[0] for r in jump_frames]}")
    print(f"Serve starts         : {len(serve_starts)}  at frames: {[r[0] for r in serve_starts]}")
    print(f"\nLog written to: {log_file}")

    # Print the first 20 TRACKING_POINT rows around any loss/jump event
    problem_frames = set(r[0] for r in lost_frames + jump_frames)
    if problem_frames:
        print(f"\n{'-'*60}")
        print("PROBLEM FRAME CONTEXT (+-5 frames):")
        print(f"{'-'*60}")
        for pf in sorted(problem_frames):
            context = [r for r in log_rows if abs(r[0] - pf) <= 5]
            for r in context:
                marker = " <<<" if r[0] == pf else ""
                print(f"  f{r[0]:04d}  state={r[1]:<16s}  pos=({str(r[2]):>5s},{str(r[3]):>5s})  stuck={r[4]}  {r[5]}{marker}")
            print()

    analyzer.cap.release()
    return log_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--log", default="tracking_log.csv")
    args = parser.parse_args()
    run_test(max_frames=args.max_frames, start_frame=args.start_frame, log_file=args.log)
