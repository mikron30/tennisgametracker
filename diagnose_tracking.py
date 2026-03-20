"""Headless diagnostic script to run the ball tracker and save annotated frames."""
import cv2
import numpy as np
import json
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interactive_ball_analyzer import InteractiveBallAnalyzer

def reset_tracking(analyzer):
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

def run_diagnostic(video_path, start_frame=0, max_frames=500, save_every=5, output_dir="diagnostic_frames"):
    os.makedirs(output_dir, exist_ok=True)
    analyzer = InteractiveBallAnalyzer(video_path, start_frame=start_frame)
    hsv_config = analyzer.load_hsv_config()
    if not hsv_config:
        analyzer.hsv_lower = np.array([20, 20, 70], dtype=np.uint8)
        analyzer.hsv_upper = np.array([90, 255, 255], dtype=np.uint8)

    game_state = "WAITING_FOR_SERVE"
    point_start_frame = None
    point_end_frame = None
    serve_tracking_frames = 0
    last_serve_candidate = None
    early_serve_grace_frames = 15
    serve_position_history = []

    tracking_log = []
    state_transitions = []

    analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    analyzer.frame_count = start_frame
    frames_processed = 0

    while frames_processed < max_frames:
        ret, frame = analyzer.cap.read()
        if not ret:
            break

        prev_state = game_state
        ball_pos = None

        # Early serve detection (mirrors original process_video)
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
                    ball_pos = candidate
            else:
                if serve_tracking_frames >= 5 and last_serve_candidate is not None:
                    game_state = "TRACKING_POINT"
                    point_start_frame = analyzer.frame_count
                    analyzer.point_start_frame_internal = analyzer.frame_count
                    analyzer.tracking = True
                    analyzer.ball_center = last_serve_candidate
                    ball_pos = last_serve_candidate
                else:
                    serve_tracking_frames = 0
                    last_serve_candidate = None

        analyzer.frame_count += 1

        if game_state == "SCANNING_FOR_SERVE":
            potential_serve = analyzer.detect_serve_position(frame)
            if potential_serve:
                analyzer.ball_center = potential_serve
                analyzer.tracking = True
                analyzer.ball_stopped = False
                analyzer.ball_velocity_history = []
                analyzer.initial_ball_position = potential_serve
                analyzer.ball_size = None
                point_start_frame = analyzer.frame_count
                analyzer.point_start_frame_internal = analyzer.frame_count
                game_state = "TRACKING_POINT"
                ball_pos = potential_serve

        elif game_state == "TRACKING_POINT":
            analyzer.detect_and_track_players(frame)
            tracked_position = None
            if analyzer.tracking and analyzer.hsv_lower is not None and analyzer.ball_center is not None:
                tracked_position = analyzer.track_ball_in_frame(frame)

            if tracked_position:
                ball_pos = tracked_position
                # Stuck-ball timeout
                if getattr(analyzer, 'stuck_frame_count', 0) >= 15:
                    point_end_frame = analyzer.frame_count
                    state_transitions.append({
                        'frame': analyzer.frame_count,
                        'from': 'TRACKING_POINT',
                        'to': 'WAITING_FOR_SERVE',
                        'reason': f'Ball stuck for {analyzer.stuck_frame_count} frames',
                        'ball_pos': tracked_position
                    })
                    game_state = "WAITING_FOR_SERVE"
                    reset_tracking(analyzer)
                else:
                    point_ended, reason = analyzer.detect_point_end(tracked_position, frame)
                    if point_ended:
                        point_end_frame = analyzer.frame_count
                        state_transitions.append({
                            'frame': analyzer.frame_count,
                            'from': 'TRACKING_POINT',
                            'to': 'POINT_ENDED' if "net" not in reason.lower() else 'WAITING_FOR_SERVE',
                            'reason': reason,
                            'ball_pos': tracked_position
                        })
                        if "net" in reason.lower():
                            reset_tracking(analyzer)
                            game_state = "WAITING_FOR_SERVE"
                        else:
                            game_state = "POINT_ENDED"
                            reset_tracking(analyzer)
            else:
                grace_limit = 45 if point_start_frame and analyzer.frame_count <= (start_frame + 45) else 30
                if point_start_frame and (analyzer.frame_count - point_start_frame > grace_limit):
                    point_end_frame = analyzer.frame_count
                    state_transitions.append({
                        'frame': analyzer.frame_count,
                        'from': 'TRACKING_POINT',
                        'to': 'POINT_ENDED',
                        'reason': 'Ball lost',
                        'ball_pos': None
                    })
                    game_state = "POINT_ENDED"
                    reset_tracking(analyzer)

        elif game_state == "POINT_ENDED":
            if point_end_frame and analyzer.frame_count - point_end_frame > 60:
                game_state = "SCANNING_FOR_SERVE"
                state_transitions.append({
                    'frame': analyzer.frame_count,
                    'from': 'POINT_ENDED',
                    'to': 'SCANNING_FOR_SERVE',
                    'reason': 'Timeout',
                    'ball_pos': None
                })
                point_start_frame = None
                point_end_frame = None

        elif game_state == "WAITING_FOR_SERVE":
            # Detect ball in serve area, start tracking when ball exits rightward
            potential_serve = analyzer.detect_serve_position(frame)
            if potential_serve:
                serve_tracking_frames += 1
                last_serve_candidate = potential_serve
                serve_position_history.append(potential_serve)
                if len(serve_position_history) > 10:
                    serve_position_history = serve_position_history[-10:]
                ball_pos = potential_serve
            else:
                if serve_tracking_frames >= 3 and last_serve_candidate is not None:
                    is_rightward = True
                    if len(serve_position_history) >= 2:
                        last_pos = serve_position_history[-1]
                        first_pos = serve_position_history[-min(3, len(serve_position_history))]
                        dx = last_pos[0] - first_pos[0]
                        is_rightward = dx > 10
                    if is_rightward:
                        # Predict where ball is now based on serve velocity
                        predicted_pos = last_serve_candidate
                        if len(serve_position_history) >= 2:
                            p1 = serve_position_history[-2]
                            p2 = serve_position_history[-1]
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            dist = math.hypot(dx, dy)
                            direction = math.degrees(math.atan2(dy, dx))
                            predicted_pos = (int(p2[0] + dx), int(p2[1] + dy))
                            analyzer.last_motion = {
                                'distance': dist,
                                'dx': dx,
                                'dy': dy,
                                'direction_deg': direction
                            }
                            analyzer.last_delta = (dx, dy)
                            analyzer.ball_velocity_history = [dist]
                        analyzer.ball_center = predicted_pos
                        analyzer.tracking = True
                        analyzer.ball_stopped = False
                        analyzer.initial_ball_position = last_serve_candidate
                        analyzer.ball_size = None
                        analyzer.stuck_frame_count = 0
                        point_start_frame = analyzer.frame_count
                        analyzer.point_start_frame_internal = analyzer.frame_count
                        game_state = "TRACKING_POINT"
                        ball_pos = predicted_pos
                        state_transitions.append({
                            'frame': analyzer.frame_count,
                            'from': 'WAITING_FOR_SERVE',
                            'to': 'TRACKING_POINT',
                            'reason': f'Serve exit rightward (after {serve_tracking_frames} frames in serve area)',
                            'ball_pos': last_serve_candidate
                        })
                serve_tracking_frames = 0
                last_serve_candidate = None
                serve_position_history = []

        if game_state != prev_state and not any(t['frame'] == analyzer.frame_count for t in state_transitions):
            state_transitions.append({
                'frame': analyzer.frame_count,
                'from': prev_state,
                'to': game_state,
                'reason': 'state change',
                'ball_pos': ball_pos
            })

        entry = {
            'frame': analyzer.frame_count,
            'state': game_state,
            'ball_pos': ball_pos,
            'ball_size': analyzer.ball_size,
            'tracking': analyzer.tracking,
        }
        tracking_log.append(entry)

        is_state_change = game_state != prev_state
        is_save_frame = (frames_processed % save_every == 0) or is_state_change

        if is_save_frame:
            annotated = frame.copy()
            if ball_pos:
                cv2.circle(annotated, (int(ball_pos[0]), int(ball_pos[1])), 15, (0, 0, 255), 2)
                cv2.circle(annotated, (int(ball_pos[0]), int(ball_pos[1])), 3, (0, 255, 0), -1)
            if hasattr(analyzer, 'serve_area_x_min'):
                cv2.rectangle(annotated,
                            (analyzer.serve_area_x_min, analyzer.serve_area_y_min),
                            (analyzer.serve_area_x_max, analyzer.serve_area_y_max),
                            (255, 255, 0), 2)
            if hasattr(analyzer, 'net_area_y_min'):
                cv2.line(annotated, (0, analyzer.net_area_y_min), (frame.shape[1], analyzer.net_area_y_min), (0, 255, 255), 1)
                cv2.line(annotated, (0, analyzer.net_area_y_max), (frame.shape[1], analyzer.net_area_y_max), (0, 255, 255), 1)
            cv2.putText(annotated, f"Frame: {analyzer.frame_count} State: {game_state}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if ball_pos:
                cv2.putText(annotated, f"Ball: ({ball_pos[0]}, {ball_pos[1]})",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            prefix = "STATE_" if is_state_change else ""
            filename = f"{prefix}frame_{analyzer.frame_count:05d}_{game_state}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), annotated)

        frames_processed += 1

    analyzer.cap.release()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Frames processed: {frames_processed}")
    print(f"State transitions: {len(state_transitions)}")
    for t in state_transitions:
        print(f"  Frame {t['frame']}: {t['from']} -> {t['to']} ({t['reason']}) ball={t['ball_pos']}")

    tracking_frames = [e for e in tracking_log if e['state'] == 'TRACKING_POINT' and e['ball_pos'] is not None]
    lost_frames = [e for e in tracking_log if e['state'] == 'TRACKING_POINT' and e['ball_pos'] is None]
    print(f"\nTracking frames: {len(tracking_frames)}")
    print(f"Lost frames during tracking: {len(lost_frames)}")

    log_path = os.path.join(output_dir, "tracking_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            'state_transitions': state_transitions,
            'tracking_summary': {
                'total_frames': frames_processed,
                'tracking_frames': len(tracking_frames),
                'lost_frames': len(lost_frames)
            }
        }, f, indent=2)
    print(f"\nFull log saved to {log_path}")
    return state_transitions, tracking_log


if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    run_diagnostic(video_path, start_frame=0, max_frames=600, save_every=10)
