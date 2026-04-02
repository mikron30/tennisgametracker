import argparse
import cv2
import numpy as np
import json
import os
import math
from typing import Tuple, Optional


class InteractiveBallAnalyzer:
    def __init__(self, video_path: str, start_frame: int = 0, config_file: str = "hsv_config.json"):
        self.video_path = video_path
        self.config_file = config_file
        self.cap = cv2.VideoCapture(video_path)
        self.start_frame = max(0, start_frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.frame_count = self.start_frame
        self.last_seen_frame = self.start_frame
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Ball analysis state
        self.ball_center = None
        self.ball_hsv = None
        self.ball_size = None
        self.tracking = False
        self.ball_stopped = False
        self.stuck_frame_count = 0
        self.ball_velocity_history = []
        self.initial_ball_position = None
        self.last_seen_frame = None
        self.edge_wait = False
        self.last_delta = None
        self.primary_hsv_lower = None
        self.primary_hsv_upper = None
        self.alt_hsv_lower = None
        self.alt_hsv_upper = None
        self.hsv_regular = None
        self.hsv_behind_net = None
        
        # HSV filter (will be set based on clicked ball)
        self.hsv_lower = None
        self.hsv_upper = None
        
        # HSV values table for analysis
        self.hsv_table = []
        
        self.motion_history = []
        self.focus_loss_frame = None
        self.focus_loss_active = False
        self.alt_focus_hsv_lower = None
        self.alt_focus_hsv_upper = None
        self.last_motion = None
        self.pre_focus_hsv_regular = None
        self.pre_focus_hsv_behind_net = None
        self.using_alt_hsv = False
        self.using_alt2_hsv = False
        self.alt2_hsv_lower = None
        self.alt2_hsv_upper = None
        self.using_alt3_hsv = False
        self.alt3_hsv_lower = None
        self.alt3_hsv_upper = None
        self.alt4_hsv_lower = None
        self.alt4_hsv_upper = None
        self.alt5_hsv_lower = None
        self.alt5_hsv_upper = None
        self.alt6_hsv_lower = None
        self.alt6_hsv_upper = None
        self.disable_behind_net_mode = os.environ.get("DISABLE_BEHIND_NET", "0") == "1"
        self.direction_change_points = []
        self.show_event_markers = False
        self.last_direction = None
        self.near_edge = False
        self.net_contact_points = []
        self.prev_motion = None
        self.max_ball_speed = 70  # overridden per-court via config ball_max_speed
        self.direction_change_streak = 0
        # Flag when per-frame movement jumps well beyond recent median
        self.focus_loss_distance_threshold = 60.0
        self.pause_requested = False
        self.serve_direction_dx = 1
        self.serve_direction_dy = 0
        self.serve_directional_search = False
        self.serve_directional_search_frames = 20
        self.serve_contact_y_ratio = 0.7
        self.serve_contact_min_prev_dy = 20
        self.serve_contact_max_prev_dx = 40
        self.serve_contact_min_ball_size = 100
        self.serve_contact_min_dx = 80
        self.serve_contact_min_dy = 0
        self.point_start_frame_internal = None
        self._serve_contact_grace_frames = 0
        self._rally_contact_grace_frames = 0
        self._rally_contact_ref_size = None
        self._rally_contact_origin = None
        self._rally_contact_expected = None
        self._rally_contact_progress = 0.0
        self._ground_bounce_grace_frames = 0
        self._ground_bounce_ref_size = None
        self._ground_bounce_origin = None
        self._ground_bounce_expected = None
        self._ground_bounce_progress = 0.0
        self.ground_bounce_count = 0
        self.last_ground_bounce_frame = -1000000
        self._contact_recovery_frames = 0
        self._upper_exit_wait_frames = 0
        self.serve_width_ratio = None
        self._prev_serve_gray = None
        self._ignored_serve_positions = []
        self._ignored_tracking_positions = []
        self._frame0_hotspot_combined_mask = None
        self._frame0_hotspot_masks = {}
        self._frame0_hotspot_entries = []
        self._frame0_hotspot_signature = None
        self._frame0_hotspot_file = None
        self.waiting_serve_candidate = None
        self.waiting_serve_candidate_frame = -1
        self._last_detected_serve_candidate = None
        self._debug_contour_candidates = []
        self._debug_rejected_contours = []
        
    def log_motion_metrics(self, dx, dy, distance, direction_deg):
        """Log per-frame motion and raise a focus-loss flag when movement spikes."""
        direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
        print(f"Frame {self.frame_count}: Movement {distance:.1f}px (dx={dx}, dy={dy}) Direction: {direction_text}")

        self.prev_motion = self.last_motion
        self.last_motion = {
            'distance': distance,
            'dx': dx,
            'dy': dy,
            'direction_deg': direction_deg
        }
        if direction_deg is not None and distance > 5:
            if self.near_edge or getattr(self, 'edge_wait', False):
                self.last_direction = direction_deg
                return False
            if self.last_direction is not None:
                delta = abs(direction_deg - self.last_direction) % 360
                angle_diff = min(delta, 360 - delta)
                if angle_diff >= 110:
                    if (getattr(self, 'ground_bounce_count', 0) > 0 and
                            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) > 2 and
                            distance > 12):
                        print(f"Frame {self.frame_count}: Resetting bounce count after new shot direction change")
                        self.ground_bounce_count = 0
                        self.last_ground_bounce_frame = -1000000
                    self.direction_change_points.append((self.ball_center[0], self.ball_center[1]))
                    print(f"Frame {self.frame_count}: Direction change detected ({angle_diff:.1f} deg)")
            self.last_direction = direction_deg
        self.motion_history.append({
            'frame': self.frame_count,
            'distance': distance,
            'direction_deg': direction_deg
        })
        if len(self.motion_history) > 200:
            self.motion_history.pop(0)

        # Compare to recent median distance (starting from start_frame)
        if self.focus_loss_frame is None and self.frame_count >= self.start_frame:
            baseline_distances = [
                entry['distance']
                for entry in self.motion_history
                if self.start_frame <= entry['frame'] < self.frame_count
            ]
            median_distance = float(np.median(baseline_distances[-10:])) if baseline_distances else 0.0
            spike_threshold = max(self.focus_loss_distance_threshold, median_distance * 2.5)
            if distance > spike_threshold:
                self.focus_loss_frame = self.frame_count
                self.focus_loss_active = True
                self.using_alt_hsv = True
                if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                    self.hsv_lower = self.alt_focus_hsv_lower
                    self.hsv_upper = self.alt_focus_hsv_upper
                print(f"Frame {self.frame_count}: [FOCUS LOSS FLAG] movement spike detected "
                      f"(distance {distance:.1f}px vs median {median_distance:.1f}px) "
                      f"starting from frame {self.start_frame}")
                return True
        return False

    def serve_direction_label(self):
        horizontal = "right" if self.serve_direction_dx >= 0 else "left"
        if self.serve_direction_dy < 0:
            vertical = "up"
        elif self.serve_direction_dy > 0:
            vertical = "down"
        else:
            vertical = "flat"
        return f"{horizontal}/{vertical}"

    def _signed_serve_dx(self, dx):
        return dx * self.serve_direction_dx

    def _signed_serve_dy(self, dy):
        if self.serve_direction_dy == 0:
            return dy
        return dy * self.serve_direction_dy

    def _serve_pair_moves_forward(self, prev_pos, curr_pos, min_dx=0):
        if prev_pos is None or curr_pos is None:
            return False
        dx = curr_pos[0] - prev_pos[0]
        return self._signed_serve_dx(dx) > min_dx

    def _serve_direction_search_active(self):
        if not self.serve_directional_search or self.ball_center is None or self.last_motion is None:
            return False
        if self.point_start_frame_internal is None:
            return False
        if (self.frame_count - self.point_start_frame_internal) > self.serve_directional_search_frames:
            return False
        if not hasattr(self, 'serve_area_x_min'):
            return False

        x, y = self.ball_center
        if not (self.serve_area_x_min <= x <= self.serve_area_x_max and
                self.serve_area_y_min <= y <= self.serve_area_y_max):
            return False

        serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
        contact_y = self.serve_area_y_min + int(serve_height * self.serve_contact_y_ratio)
        if y < contact_y:
            return False
        if self.last_motion.get('dy', 0) < self.serve_contact_min_prev_dy:
            return False
        if abs(self.last_motion.get('dx', 0)) > self.serve_contact_max_prev_dx:
            return False
        if (self.ball_size or 0) < self.serve_contact_min_ball_size:
            return False
        return True

    def _build_serve_direction_region(self, x, y, frame_shape, radius):
        frame_height, frame_width = frame_shape[:2]

        if self.serve_direction_dx < 0:
            x1 = max(0, x - radius)
            x2 = min(frame_width, x + 1)
        else:
            x1 = max(0, x)
            x2 = min(frame_width, x + radius + 1)

        if self.serve_direction_dy < 0:
            y1 = max(0, y - radius)
            y2 = min(frame_height, y + 1)
        elif self.serve_direction_dy > 0:
            y1 = max(0, y)
            y2 = min(frame_height, y + radius + 1)
        else:
            y1 = max(0, y - radius)
            y2 = min(frame_height, y + radius + 1)

        return x1, y1, x2, y2

    def _prune_ignored_serve_positions(self):
        self._ignored_serve_positions = [
            entry for entry in getattr(self, '_ignored_serve_positions', [])
            if entry['expires'] >= self.frame_count
        ]

    def _prune_ignored_tracking_positions(self):
        self._ignored_tracking_positions = [
            entry for entry in getattr(self, '_ignored_tracking_positions', [])
            if entry['expires'] >= self.frame_count
        ]

    def _find_ignored_tracking_position(self, pos):
        px, py = pos
        for entry in getattr(self, '_ignored_tracking_positions', []):
            dist = math.hypot(px - entry['pos'][0], py - entry['pos'][1])
            if dist <= entry['radius']:
                return entry
        return None

    def _learn_ignored_tracking_position(self, pos, radius=70, ttl=180, reason="static tracking hotspot"):
        self._prune_ignored_tracking_positions()
        for entry in self._ignored_tracking_positions:
            dist = math.hypot(pos[0] - entry['pos'][0], pos[1] - entry['pos'][1])
            if dist <= max(radius, entry['radius']):
                count = entry.get('count', 1) + 1
                old_x, old_y = entry['pos']
                entry['pos'] = (
                    int(round((old_x * (count - 1) + pos[0]) / count)),
                    int(round((old_y * (count - 1) + pos[1]) / count)),
                )
                entry['radius'] = max(entry['radius'], radius)
                entry['expires'] = max(entry['expires'], self.frame_count + ttl)
                entry['count'] = count
                entry['reason'] = reason
                print(f"[TRACKING_FP_LEARN] f{self.frame_count}: refresh hotspot at {entry['pos']} "
                      f"radius={entry['radius']} ttl={entry['expires'] - self.frame_count} reason={reason}")
                return

        self._ignored_tracking_positions.append({
            'pos': pos,
            'radius': radius,
            'expires': self.frame_count + ttl,
            'count': 1,
            'reason': reason,
        })
        print(f"[TRACKING_FP_LEARN] f{self.frame_count}: ignore hotspot at {pos} "
              f"radius={radius} ttl={ttl} reason={reason}")

    def _frame0_filter_specs(self):
        specs = []
        if self.hsv_lower is not None and self.hsv_upper is not None:
            specs.append(("primary", self.hsv_lower, self.hsv_upper))
        if self.hsv_regular is not None:
            specs.append(("regular", self.hsv_regular['lower'], self.hsv_regular['upper']))
        if self._behind_net_enabled():
            specs.append(("behind_net", self.hsv_behind_net['lower'], self.hsv_behind_net['upper']))
        if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
            specs.append(("alt1", self.alt_focus_hsv_lower, self.alt_focus_hsv_upper))
        if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
            specs.append(("alt2", self.alt2_hsv_lower, self.alt2_hsv_upper))
        if self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
            specs.append(("alt3", self.alt3_hsv_lower, self.alt3_hsv_upper))
        return specs

    def _behind_net_enabled(self):
        return (not self.disable_behind_net_mode) and self.hsv_behind_net is not None

    def _frame0_hotspot_signature_value(self):
        def pack(arr):
            if arr is None:
                return None
            return tuple(int(v) for v in arr.tolist())

        return (
            os.path.abspath(self.video_path),
            os.path.abspath(self.config_file),
            tuple((label, pack(lower), pack(upper)) for label, lower, upper in self._frame0_filter_specs()),
        )

    def _build_frame0_background_hotspots(self, force=False):
        """Build a mask/list of frame-0 ball-colored blobs to reject static court spots."""
        if self.hsv_lower is None or self.hsv_upper is None:
            return False

        signature = self._frame0_hotspot_signature_value()
        if (not force and self._frame0_hotspot_signature == signature
                and self._frame0_hotspot_combined_mask is not None):
            return True

        current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame0 = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)

        if not ret or frame0 is None:
            print("Frame-0 hotspot build skipped: could not read frame 0")
            return False

        frame_height, frame_width = frame0.shape[:2]
        hsv_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        masks = {}
        hotspot_entries = []
        filter_counts = {}
        area_cap = max(150, int(getattr(self, 'serve_ball_size_max', 80)))
        kernel = np.ones((2, 2), np.uint8)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for label, lower, upper in self._frame0_filter_specs():
            mask = cv2.inRange(hsv_frame0, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            selected_mask = np.zeros_like(mask)
            kept = 0

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1 or area > area_cap:
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                hotspot_entries.append({
                    'filter': label,
                    'pos': [cx, cy],
                    'radius': int(max(4, round(max(w, h) / 2) + 2)),
                    'area': float(area),
                })
                cv2.drawContours(selected_mask, [contour], -1, 255, -1)
                kept += 1

            selected_mask = cv2.dilate(selected_mask, dilate_kernel, iterations=1)
            masks[label] = selected_mask
            combined_mask = cv2.bitwise_or(combined_mask, selected_mask)
            filter_counts[label] = kept

        merged_hotspots = []
        merged_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in merged_contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            merged_hotspots.append({
                'pos': [int(round(cx)), int(round(cy))],
                'radius': int(max(4, round(radius))),
                'mask_area': float(area),
            })
        merged_hotspots.sort(key=lambda entry: (entry['pos'][1], entry['pos'][0]))

        self._frame0_hotspot_masks = masks
        self._frame0_hotspot_combined_mask = combined_mask
        self._frame0_hotspot_entries = merged_hotspots
        self._frame0_hotspot_signature = signature

        hotspot_base = os.path.splitext(os.path.basename(self.config_file))[0]
        self._frame0_hotspot_file = os.path.join(
            os.path.dirname(os.path.abspath(self.config_file)),
            f"frame0_hotspots_{hotspot_base}.json"
        )

        payload = {
            'video': os.path.basename(self.video_path),
            'config': os.path.basename(self.config_file),
            'frame': 0,
            'filter_counts': filter_counts,
            'merged_hotspots': merged_hotspots,
        }
        try:
            with open(self._frame0_hotspot_file, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Frame-0 hotspot save failed: {e}")

        coverage = float(np.count_nonzero(combined_mask)) / float(combined_mask.size)
        print(f"Frame-0 hotspot baseline built: {len(merged_hotspots)} merged spots, "
              f"coverage={coverage * 100:.2f}% saved to {os.path.basename(self._frame0_hotspot_file)}")
        return True

    def _find_frame0_background_hotspot(self, pos):
        mask = getattr(self, '_frame0_hotspot_combined_mask', None)
        if mask is None:
            return None

        px, py = int(pos[0]), int(pos[1])
        if py < 0 or px < 0 or py >= mask.shape[0] or px >= mask.shape[1]:
            return None
        if mask[py, px] == 0:
            return None

        nearest = None
        nearest_dist = float('inf')
        for entry in getattr(self, '_frame0_hotspot_entries', []):
            radius = max(4, int(entry.get('radius', 4)))
            dist = math.hypot(px - entry['pos'][0], py - entry['pos'][1])
            if dist <= radius and dist < nearest_dist:
                nearest = entry
                nearest_dist = dist

        if nearest is not None:
            return nearest
        return {
            'pos': [px, py],
            'radius': 0,
            'mask_area': 0.0,
        }

    def _candidate_motion_metrics(self, gray_frame, cx, cy, radius=8):
        prev_gray = getattr(self, '_prev_frame_gray', None)
        if gray_frame is None or prev_gray is None:
            return None
        if gray_frame.shape[:2] != prev_gray.shape[:2]:
            return None

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(gray_frame.shape[1], cx + radius + 1)
        y2 = min(gray_frame.shape[0], cy + radius + 1)
        if x2 <= x1 or y2 <= y1:
            return None

        curr_patch = gray_frame[y1:y2, x1:x2]
        prev_patch = prev_gray[y1:y2, x1:x2]
        if curr_patch.size == 0 or prev_patch.shape != curr_patch.shape:
            return None

        diff = cv2.absdiff(prev_patch, curr_patch)
        return {
            'mean': float(np.mean(diff)),
            'max': float(np.max(diff)),
        }

    def _contact_reacquire_bounds(self, frame_shape, reference_pos):
        """Return a plausible reacquire window for upper-court contact recovery."""
        if reference_pos is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        ref_x, ref_y = reference_pos
        upper_contact_limit = max(280, int(frame_height * 0.14))
        if ref_y > upper_contact_limit:
            return None

        recent_max_vel = max(self.ball_velocity_history[-3:]) if getattr(self, 'ball_velocity_history', None) else 0
        if recent_max_vel > 120:
            return None

        return {
            'max_dist': max(260, min(460, self.max_ball_speed + 40)),
            'min_y': max(0, ref_y - 45),
            'max_y': min(frame_height - 1, ref_y + max(220, int(frame_height * 0.12))),
            'min_x': max(0, ref_x - max(420, int(frame_width * 0.12))),
            'max_x': min(frame_width - 1, ref_x + max(420, int(frame_width * 0.12))),
        }

    def _get_lower_contact_launch_context(self, frame_shape):
        """Predict a launch window after a lower-court racket hit using ball-only motion."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        incoming_dx = float(self.last_motion.get('dx', 0.0))
        incoming_dy = float(self.last_motion.get('dy', 0.0))
        incoming_dist = float(self.last_motion.get('distance', 0.0))

        if self.ball_size < 300:
            return None
        if origin_y < int(frame_height * 0.58):
            return None
        if incoming_dy < max(50.0, frame_height * 0.023):
            return None
        if incoming_dist < 60.0:
            return None

        lateral_scale = 2.5
        vertical_scale = 2.4
        raw_expected_dx = -incoming_dx * lateral_scale
        raw_expected_up = abs(incoming_dy) * vertical_scale

        min_lateral = max(70.0, frame_width * 0.018)
        max_lateral = max(220.0, frame_width * 0.065)
        if abs(raw_expected_dx) < min_lateral:
            raw_expected_dx = math.copysign(min_lateral, -incoming_dx if abs(incoming_dx) > 1 else 1.0)
        expected_dx = max(-max_lateral, min(max_lateral, raw_expected_dx))

        min_up = max(95.0, frame_height * 0.042)
        max_up = max(260.0, frame_height * 0.13)
        expected_up = max(min_up, min(max_up, raw_expected_up))

        expected_x = int(max(0, min(frame_width - 1, origin_x + expected_dx)))
        expected_y = int(max(0, min(frame_height - 1, origin_y - expected_up)))

        return {
            'origin': (origin_x, origin_y),
            'expected': (expected_x, expected_y),
            'min_launch_dist': max(120.0, incoming_dist * 1.6),
            'max_launch_dist': max(360.0, incoming_dist * 4.5),
            'min_upward': max(85.0, incoming_dy * 0.95),
            'ref_size': max(35.0, min(self.ball_size * 0.30, 130.0)),
        }

    def _get_ground_bounce_context(self, frame_shape):
        """Predict a short upward continuation after a court bounce using ball-only motion."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        incoming_dx = float(self.last_motion.get('dx', 0.0))
        incoming_dy = float(self.last_motion.get('dy', 0.0))
        incoming_dist = float(self.last_motion.get('distance', 0.0))

        if self.ball_size > 180:
            return None
        if incoming_dy < max(6.0, frame_height * 0.0035):
            return None
        if incoming_dist < 6.0:
            return None
        if self.prev_motion is not None and self.prev_motion.get('dy', 0.0) < max(3.0, frame_height * 0.002):
            return None
        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            if (self.net_area_y_min - 60) <= origin_y <= (self.net_area_y_max + 80):
                return None

        expected_dx = incoming_dx * 0.9
        expected_up = max(8.0, min(max(14.0, abs(incoming_dy) * 1.15), 70.0))
        expected_x = int(max(0, min(frame_width - 1, origin_x + expected_dx)))
        expected_y = int(max(0, min(frame_height - 1, origin_y - expected_up)))

        return {
            'origin': (origin_x, origin_y),
            'expected': (expected_x, expected_y),
            'incoming_dx': incoming_dx,
            'min_launch_dist': max(5.0, incoming_dist * 0.45),
            'max_launch_dist': max(55.0, incoming_dist * 3.2),
            'min_upward': max(5.0, incoming_dy * 0.50),
            'expected_cap': max(28.0, min(85.0, incoming_dist * 2.2)),
            'ref_size': max(8.0, min(max(self.ball_size * 1.10, self.ball_size + 6.0), 90.0)),
        }

    def _recover_contact_phase_ball(self, frame, reference_pos, frame_gray, predicted_point=None,
                                    max_prev_speed=15.0, upper_stuck_mode=False):
        if reference_pos is None or frame_gray is None or self.last_motion is None:
            return None

        ref_x, ref_y = reference_pos
        if ref_y > max(280, int(frame.shape[0] * 0.14)):
            return None
        if self.last_motion.get('distance', 0) > max_prev_speed:
            return None

        if upper_stuck_mode and predicted_point is None:
            predicted_point = (
                int(ref_x + self.last_motion.get('dx', 0.0)),
                int(ref_y + self.last_motion.get('dy', 0.0)),
            )

        radius = max(self.max_ball_speed + 150, int(frame.shape[1] * 0.16))
        x1 = max(0, ref_x - radius)
        y1 = max(0, ref_y - radius)
        x2 = min(frame.shape[1], ref_x + radius)
        y2 = min(frame.shape[0], ref_y + radius)
        search_frame = frame[y1:y2, x1:x2]
        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)

        filters = []
        if self._behind_net_enabled():
            filters.append(("behind_net", self.hsv_behind_net['lower'], self.hsv_behind_net['upper']))
        if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
            filters.append(("alt2", self.alt2_hsv_lower, self.alt2_hsv_upper))
        if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
            filters.append(("alt1", self.alt_focus_hsv_lower, self.alt_focus_hsv_upper))

        local_best = None
        local_best_score = float('inf')
        best = None
        best_score = float('inf')
        target_shift = max(180, int(frame.shape[1] * 0.08))
        vertical_band = max(120, int(frame.shape[0] * 0.06))
        min_lateral_shift = 120
        local_radius = max(90, int(frame.shape[1] * 0.035))
        local_vertical_band = max(35, int(frame.shape[0] * 0.025))
        if upper_stuck_mode:
            local_radius = max(local_radius, 110)
            local_vertical_band = max(local_vertical_band, 80)
            vertical_band = max(vertical_band, 160)
            min_lateral_shift = 60
            target_shift = max(80, int(frame.shape[1] * 0.035))

        for label, lower, upper in filters:
            mask = cv2.inRange(hsv_frame, lower, upper)
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 4 or area > 80:
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                dx = cx - ref_x
                dy = cy - ref_y

                motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                if motion_metrics is not None:
                    motion_mean = motion_metrics['mean']
                    motion_max = motion_metrics['max']
                    if cy < 100 and motion_mean < 2.5 and motion_max < 10.0:
                        continue
                else:
                    motion_mean = None
                    motion_max = 0.0

                frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                if frame0_hotspot is not None and (motion_mean is None or (motion_mean < 8.0 and motion_max < 35.0)):
                    continue

                if abs(dx) <= local_radius and abs(dy) <= local_vertical_band:
                    if motion_mean is None or (motion_mean < 5.0 and motion_max < 30.0):
                        continue

                    local_score = abs(dx) * 1.4
                    local_score += abs(dy) * 2.0
                    local_score -= min(area, 50) * 4.5
                    local_score -= min(motion_max, 200.0) * 0.35
                    local_score -= min(motion_mean, 40.0) * 1.5
                    if predicted_point is not None:
                        predicted_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                        local_score += predicted_distance * (0.9 if upper_stuck_mode else 0.25)
                    if upper_stuck_mode:
                        expected_dx = self.last_motion.get('dx', 0.0)
                        expected_dy = self.last_motion.get('dy', 0.0)
                        local_score += abs(dx - expected_dx) * 0.8
                        local_score += abs(dy - expected_dy) * 1.6
                        if expected_dy < -2.0 and dy > 0:
                            local_score += 160.0

                    if local_score < local_best_score:
                        local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
                        local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
                        local_best_score = local_score
                        local_best = {
                            'pos': (cx, cy),
                            'area': area,
                            'hsv': hsv_frame[local_y, local_x],
                            'label': label,
                            'score': local_score,
                            'dx': dx,
                            'dy': dy,
                            'motion_mean': motion_mean,
                            'motion_max': motion_max,
                            'mode': 'local',
                        }
                    continue

                if upper_stuck_mode:
                    continue

                if abs(dx) < min_lateral_shift or abs(dy) > vertical_band:
                    continue

                score = abs(dy) * 4.0
                score += abs(abs(dx) - target_shift) * 0.8
                score -= min(area, 50) * 6.0
                score -= min(motion_max, 20.0) * 0.5
                if predicted_point is not None:
                    predicted_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    score += predicted_distance * (0.9 if upper_stuck_mode else 0.25)
                if upper_stuck_mode:
                    expected_dx = self.last_motion.get('dx', 0.0)
                    expected_dy = self.last_motion.get('dy', 0.0)
                    score += abs(dx - expected_dx) * 0.8
                    score += abs(dy - expected_dy) * 1.6
                    if expected_dy < -2.0 and dy > 0:
                        score += 160.0

                if score < best_score:
                    local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
                    local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
                    best_score = score
                    best = {
                        'pos': (cx, cy),
                        'area': area,
                        'hsv': hsv_frame[local_y, local_x],
                        'label': label,
                        'score': score,
                        'dx': dx,
                        'dy': dy,
                        'motion_mean': motion_mean,
                        'motion_max': motion_max,
                        'mode': 'lateral',
                    }

        picked = local_best if local_best is not None else best
        if picked is not None:
            motion_text = "n/a" if picked['motion_mean'] is None else f"{picked['motion_mean']:.1f}/{picked['motion_max']:.1f}"
            print(f"Frame {self.frame_count}: [CONTACT RECOVER] picked {picked['pos']} from {picked['label']} "
                  f"mode={picked.get('mode', 'n/a')} dx={picked['dx']:+d} dy={picked['dy']:+d} "
                  f"area={picked['area']:.1f} score={picked['score']:.1f} motion={motion_text}")
        return picked

    def retrack_with_alt_hsv(self, search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                             frame_gray=None):
        """Re-run detection with alternative HSV only after focus loss."""
        if self.alt_focus_hsv_lower is None or self.alt_focus_hsv_upper is None:
            return None

        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        mask_alt = cv2.inRange(hsv_frame, self.alt_focus_hsv_lower, self.alt_focus_hsv_upper)
        kernel = np.ones((2, 2), np.uint8)
        mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_OPEN, kernel)
        mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_CLOSE, kernel)
        contours_alt, _ = cv2.findContours(mask_alt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours_alt:
            print(f"  DEBUG: Focus-loss retrack found no contours with alt HSV")
            return None

        best_score = float('inf')
        best = None
        for i, contour in enumerate(contours_alt):
            area = cv2.contourArea(contour)
            if allow_inactive:
                if area < 1 or area > 80:
                    continue
            else:
                if area < 1 or area > 150:
                    continue

            # Reject tiny noise when ball was previously much larger.
            if self.ball_size and self.ball_size > 40:
                min_area = max(5, int(self.ball_size * 0.08))
                if area < min_area:
                    print(f"  DEBUG: retrack_using_alt skipping too-small contour area={area:.1f} prev_ball_size={self.ball_size:.1f}")
                    continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1

            if not allow_inactive:
                ignored_entry = self._find_ignored_tracking_position((cx, cy))
                if ignored_entry is not None:
                    print(f"  DEBUG: retrack_using_alt skipping learned hotspot at ({cx},{cy}) "
                          f"reason={ignored_entry.get('reason', 'n/a')}")
                    continue
                if getattr(self, '_contact_recovery_frames', 0) > 0 and prev_pos is not None:
                    contact_band = max(120, int((frame_gray.shape[0] if frame_gray is not None else search_frame.shape[0]) * 0.06))
                    if abs(cy - prev_pos[1]) > contact_band:
                        print(f"  DEBUG: retrack_using_alt skipping contact-band outlier at ({cx},{cy}) "
                              f"dy={cy - prev_pos[1]}")
                        continue
                    contact_lateral_cap = 180
                    if abs(cx - prev_pos[0]) > contact_lateral_cap:
                        print(f"  DEBUG: retrack_using_alt skipping contact-lateral outlier at ({cx},{cy}) "
                              f"dx={cx - prev_pos[0]}")
                        continue
                    if predicted_point is not None:
                        predicted_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                        if predicted_distance > 140:
                            print(f"  DEBUG: retrack_using_alt skipping predicted outlier at ({cx},{cy}) "
                                  f"pred_dist={predicted_distance:.1f}")
                            continue

            if prev_pos:
                distance = math.hypot(cx - prev_pos[0], cy - prev_pos[1])
            else:
                distance = 0

            if prev_ball_size and prev_ball_size > 0:
                size_diff = abs(area - prev_ball_size)
                size_ratio = size_diff / prev_ball_size
            else:
                size_ratio = 0

            score = distance + (size_ratio * 30)
            if self.last_motion and distance > 0 and prev_pos:
                lm_dx = self.last_motion['dx']
                lm_dy = self.last_motion['dy']
                lm_dist = self.last_motion['distance']
                mv_dx = cx - prev_pos[0]
                mv_dy = cy - prev_pos[1]
                dot = lm_dx * mv_dx + lm_dy * mv_dy
                if dot < 0:
                    score += 80  # stronger penalty for opposite direction
                if lm_dist and lm_dist > 0:
                    speed_diff = abs(distance - lm_dist)
                    score += speed_diff * 1.5
                    align_bonus = dot / (lm_dist * distance)
                    score -= max(0.0, align_bonus) * 40
            if predicted_point:
                pdx = cx - predicted_point[0]
                pdy = cy - predicted_point[1]
                predicted_distance = math.hypot(pdx, pdy)
                score += predicted_distance * 0.5

            motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
            if not allow_inactive and motion_metrics is not None:
                motion_mean = motion_metrics['mean']
                motion_max = motion_metrics['max']
                frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                if frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0:
                    score += 1800
                    print(f"  DEBUG: retrack_using_alt penalizing frame0 hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                static_hotspot = ((area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                if static_hotspot:
                    score += 1200
                    print(f"  DEBUG: retrack_using_alt penalizing static hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")

            if score < best_score:
                best_score = score
                best = (cx, cy, area, distance)

        if best is None:
            print(f"  DEBUG: Focus-loss retrack found no valid candidate with alt HSV")
            return None

        cx, cy, area, distance = best
        hsv_values = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
        local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
        hsv_at_point = hsv_values[local_y, local_x]

        print(f"  DEBUG: Focus-loss retrack selected contour at ({cx},{cy}) distance={distance:.1f}px")
        return {
            'pos': (cx, cy),
            'area': area,
            'hsv': hsv_at_point
        }

    def retrack_with_alt2_hsv(self, search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                              lower=None, upper=None, frame_gray=None):
        """Re-run detection with alternative 2 HSV (H 46-72) when stuck."""
        hsv_lower = lower if lower is not None else self.alt2_hsv_lower
        hsv_upper = upper if upper is not None else self.alt2_hsv_upper
        if hsv_lower is None or hsv_upper is None:
            return None
        
        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        mask_alt2 = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
        kernel = np.ones((2, 2), np.uint8)
        mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_OPEN, kernel)
        mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_CLOSE, kernel)
        contours_alt2, _ = cv2.findContours(mask_alt2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours_alt2:
            print(f"  DEBUG: Alt2 retrack found no contours")
            return None

        best_score = float('inf')
        best = None
        for contour in contours_alt2:
            area = cv2.contourArea(contour)
            if allow_inactive:
                if area < 1 or area > 80:
                    continue
            else:
                if area < 1 or area > 150:
                    continue

            if self.ball_size and self.ball_size > 40:
                min_area = max(5, int(self.ball_size * 0.08))
                if area < min_area:
                    print(f"  DEBUG: retrack_using_alt2 skipping too-small contour area={area:.1f} prev_ball_size={self.ball_size:.1f}")
                    continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1

            if not allow_inactive:
                ignored_entry = self._find_ignored_tracking_position((cx, cy))
                if ignored_entry is not None:
                    print(f"  DEBUG: retrack_using_alt2 skipping learned hotspot at ({cx},{cy}) "
                          f"reason={ignored_entry.get('reason', 'n/a')}")
                    continue
                if getattr(self, '_contact_recovery_frames', 0) > 0 and prev_pos is not None:
                    contact_band = max(120, int((frame_gray.shape[0] if frame_gray is not None else search_frame.shape[0]) * 0.06))
                    if abs(cy - prev_pos[1]) > contact_band:
                        print(f"  DEBUG: retrack_using_alt2 skipping contact-band outlier at ({cx},{cy}) "
                              f"dy={cy - prev_pos[1]}")
                        continue
                    contact_lateral_cap = 180
                    if abs(cx - prev_pos[0]) > contact_lateral_cap:
                        print(f"  DEBUG: retrack_using_alt2 skipping contact-lateral outlier at ({cx},{cy}) "
                              f"dx={cx - prev_pos[0]}")
                        continue
                    if predicted_point is not None:
                        predicted_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                        if predicted_distance > 140:
                            print(f"  DEBUG: retrack_using_alt2 skipping predicted outlier at ({cx},{cy}) "
                                  f"pred_dist={predicted_distance:.1f}")
                            continue

            if prev_pos:
                distance = math.hypot(cx - prev_pos[0], cy - prev_pos[1])
            else:
                distance = 0

            if prev_ball_size and prev_ball_size > 0:
                size_diff = abs(area - prev_ball_size)
                size_ratio = size_diff / prev_ball_size
            else:
                size_ratio = 0

            score = distance + (size_ratio * 30)
            if self.last_motion and distance > 0 and prev_pos:
                lm_dx = self.last_motion['dx']
                lm_dy = self.last_motion['dy']
                lm_dist = self.last_motion['distance']
                mv_dx = cx - prev_pos[0]
                mv_dy = cy - prev_pos[1]
                dot = lm_dx * mv_dx + lm_dy * mv_dy
                if dot < 0:
                    score += 80  # stronger penalty for opposite direction
                if lm_dist and lm_dist > 0:
                    speed_diff = abs(distance - lm_dist)
                    score += speed_diff * 1.5
                    align_bonus = dot / (lm_dist * distance)
                    score -= max(0.0, align_bonus) * 40
            if predicted_point:
                pdx = cx - predicted_point[0]
                pdy = cy - predicted_point[1]
                predicted_distance = math.hypot(pdx, pdy)
                score += predicted_distance * 0.5

            motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
            if not allow_inactive and motion_metrics is not None:
                motion_mean = motion_metrics['mean']
                motion_max = motion_metrics['max']
                frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                if frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0:
                    score += 1800
                    print(f"  DEBUG: retrack_using_alt2 penalizing frame0 hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                static_hotspot = ((area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                if static_hotspot:
                    score += 1200
                    print(f"  DEBUG: retrack_using_alt2 penalizing static hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")

            if score < best_score:
                best_score = score
                best = (cx, cy, area, distance)

        if best is None:
            print(f"  DEBUG: Alt2 retrack found no valid candidate")
            return None

        cx, cy, area, distance = best
        hsv_values = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
        local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
        hsv_at_point = hsv_values[local_y, local_x]

        print(f"  DEBUG: Alt2 retrack selected contour at ({cx},{cy}) distance={distance:.1f}px")
        return {
            'pos': (cx, cy),
            'area': area,
            'hsv': hsv_at_point
        }

    def _get_upper_exit_low_s_hsv_specs(self):
        """Specialized upper-wall HSV fallbacks for faint top-flight balls."""
        specs = []
        seen = set()

        def _append(label, lower, upper, soften_s=None):
            if lower is None or upper is None:
                return
            adj_lower = lower.copy()
            adj_upper = upper.copy()
            if soften_s is not None:
                adj_lower[1] = min(int(adj_lower[1]), soften_s)
            key = tuple(int(v) for v in adj_lower) + tuple(int(v) for v in adj_upper)
            if key in seen:
                return
            seen.add(key)
            specs.append((label, adj_lower, adj_upper))

        _append("alt4_upper_wall", self.alt4_hsv_lower, self.alt4_hsv_upper)
        _append("alt5_upper_wall", self.alt5_hsv_lower, self.alt5_hsv_upper)
        _append("alt6_upper_far_player", self.alt6_hsv_lower, self.alt6_hsv_upper)
        _append("regular_s25", self.hsv_regular['lower'], self.hsv_regular['upper'], soften_s=25) if self.hsv_regular is not None else None
        _append("alt2_s25", self.alt2_hsv_lower, self.alt2_hsv_upper, soften_s=25)
        if self._behind_net_enabled():
            _append("behind_net_s25", self.hsv_behind_net['lower'], self.hsv_behind_net['upper'], soften_s=25)
        if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
            _append("alt2_s20", self.alt2_hsv_lower, self.alt2_hsv_upper, soften_s=20)
        if hasattr(self, 'hsv_regular') and self.hsv_regular is not None:
            _append("regular_s20", self.hsv_regular['lower'], self.hsv_regular['upper'], soften_s=20)
        elif self.hsv_lower is not None and self.hsv_upper is not None:
            _append("active_s20", self.hsv_lower, self.hsv_upper, soften_s=20)
        return specs

    def _should_try_upper_exit_low_s(self, frame, allow_inactive, upper_exit_transition_context=False):
        if allow_inactive or self.ball_center is None:
            return False
        frame_height = frame.shape[0]
        top_limit = max(240, int(frame_height * 0.14))
        recent_bounce = (
            self.ground_bounce_count > 0 and
            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 20
        )
        return (
            recent_bounce and
            self.ball_center[1] < top_limit and
            (self.last_motion is None or self.last_motion.get('dy', 0.0) <= -2.0) and
            (
                upper_exit_transition_context or
                getattr(self, '_upper_exit_wait_frames', 0) > 0 or
                self.stuck_frame_count >= 1
            )
        )

    def _retrack_with_upper_exit_low_s(self, search_frame, x1, y1, predicted_point, frame_gray=None):
        """Try a lower-saturation HSV only for upper-flight no-detection cases."""
        if self.ball_center is None:
            return None

        prev_pos = self.ball_center
        best = None
        best_score = float('inf')

        for label, lower, upper in self._get_upper_exit_low_s_hsv_specs():
            retrack = self.retrack_with_alt2_hsv(
                search_frame, x1, y1, prev_pos, predicted_point, self.ball_size, False,
                lower=lower, upper=upper, frame_gray=frame_gray
            )
            if retrack is None:
                continue

            cx, cy = retrack['pos']
            if cy > prev_pos[1] + 4:
                continue
            upward_progress = prev_pos[1] - cy
            expected_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1]) if predicted_point else 0.0
            step_distance = math.hypot(cx - prev_pos[0], cy - prev_pos[1])
            score = expected_distance + (step_distance * 0.15)

            if upward_progress > 0:
                score -= min(55.0, upward_progress * 1.6)
            else:
                score += min(220.0, abs(upward_progress) * 10.0 + 80.0)

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None:
                score += 900.0

            if best is None or score < best_score:
                best_score = score
                best = {
                    'label': label,
                    'pos': retrack['pos'],
                    'area': retrack['area'],
                    'hsv': retrack['hsv'],
                    'lower': lower.copy(),
                    'upper': upper.copy(),
                }

        if best is not None:
            print(f"  DEBUG: Upper-exit low-s fallback selected {best['pos']} via {best['label']} score={best_score:.1f}")
        else:
            print("  DEBUG: Upper-exit low-s fallback found no valid candidate")
        return best

    def verify_ball_with_hsvs(self, frame, predicted_point):
        """Try multiple HSV ranges around predicted point to recover the ball."""
        if predicted_point is None:
            return None
        frame_gray = None
        if hasattr(self, '_prev_frame_gray') and self._prev_frame_gray is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = predicted_point
        radius = self.max_ball_speed
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(frame.shape[1], x + radius)
        y2 = min(frame.shape[0], y + radius)
        search_frame = frame[y1:y2, x1:x2]

        hsv_candidates = [
            ("basic", self.hsv_lower, self.hsv_upper),
            ("alt1", self.alt_focus_hsv_lower, self.alt_focus_hsv_upper),
            ("alt2", self.alt2_hsv_lower, self.alt2_hsv_upper),
            ("alt3", self.alt3_hsv_lower, self.alt3_hsv_upper),
            ("alt4", self.alt4_hsv_lower, self.alt4_hsv_upper),
            ("alt5", self.alt5_hsv_lower, self.alt5_hsv_upper),
            ("alt6", self.alt6_hsv_lower, self.alt6_hsv_upper),
        ]
        best = None
        best_distance = None
        for label, lower, upper in hsv_candidates:
            if lower is None or upper is None:
                continue
            retrack = self.retrack_with_alt2_hsv(
                search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, False,
                lower=lower, upper=upper, frame_gray=frame_gray
            )
            if retrack is None:
                continue
            pos = retrack['pos']
            distance = math.hypot(pos[0] - x, pos[1] - y)
            if best is None or distance < best_distance:
                best = (label, retrack)
                best_distance = distance
        return best

    def show_alternative_debug(self, frame, point, lower, upper, title):
        """Show debug windows for a specific HSV range around a point."""
        if point is None or lower is None or upper is None:
            print("No debug point or HSV range available.")
            return
        x, y = point
        region_size = 150
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(frame.shape[1], x + region_size)
        y2 = min(frame.shape[0], y + region_size)
        region = frame[y1:y2, x1:x2].copy()
        if region.size == 0:
            print("Error: Cannot extract debug region")
            return
        if region.shape[0] != 300 or region.shape[1] != 300:
            region = cv2.resize(region, (300, 300))
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.inRange(hsv_region, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_display = region.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1 or area > 150:
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(local_display, (cx, cy), 6, (0, 255, 0), 1)
            cv2.putText(local_display, f"{int(area)}", (cx+7, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        window_title = f"Detected Bulbs - {title}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 800, 600)
        cv2.putText(local_display,
                    f"H={lower[0]}-{upper[0]} S={lower[1]}-{upper[1]} V={lower[2]}-{upper[2]}",
                    (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow(window_title, local_display)

    def get_standard_hsv_for_position(self, y_position):
        """Get the primary HSV range for a position, ignoring alt/focus-loss modes."""
        if self.hsv_regular is not None and self._behind_net_enabled():
            if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
                if self.net_area_y_min <= y_position <= self.net_area_y_max:
                    return self.hsv_behind_net['lower'], self.hsv_behind_net['upper'], "behind_net"
            return self.hsv_regular['lower'], self.hsv_regular['upper'], "regular_court"
        if self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
            return self.primary_hsv_lower, self.primary_hsv_upper, "primary"
        return self.hsv_lower, self.hsv_upper, "current"

    def get_predicted_point(self):
        """Predict next ball position using last known motion."""
        if self.ball_center and self.last_motion:
            dx = int(self.last_motion.get('dx', 0))
            dy = int(self.last_motion.get('dy', 0))
            return (self.ball_center[0] + dx, self.ball_center[1] + dy)
        if self.ball_center and hasattr(self, 'last_delta') and self.last_delta is not None:
            dx, dy = self.last_delta
            return (self.ball_center[0] + int(dx), self.ball_center[1] + int(dy))
        return None

    def open_predicted_hsv_debug_all(self, frame, point, frame_index):
        """Open all HSV debug windows around a predicted point."""
        if frame is None or point is None:
            print("No predicted point available for HSV debug.")
            return
        lower, upper, label = self.get_standard_hsv_for_position(point[1])
        if lower is not None and upper is not None:
            self.show_alternative_debug(frame, point, lower, upper, f"Predicted {frame_index} ({label})")
        if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alt_focus_hsv_lower, self.alt_focus_hsv_upper,
                                        f"Predicted {frame_index} (alt1)")
        if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alt2_hsv_lower, self.alt2_hsv_upper,
                                        f"Predicted {frame_index} (alt2)")
        if self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alt3_hsv_lower, self.alt3_hsv_upper,
                                        f"Predicted {frame_index} (alt3)")
        if self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alt4_hsv_lower, self.alt4_hsv_upper,
                                        f"Predicted {frame_index} (alt4)")
        if self.alt5_hsv_lower is not None and self.alt5_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alt5_hsv_lower, self.alt5_hsv_upper,
                                        f"Predicted {frame_index} (alt5)")
        if self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alt6_hsv_lower, self.alt6_hsv_upper,
                                        f"Predicted {frame_index} (alt6)")

    def open_prev_frame_hsv_debug(self, frame, point, frame_index, hsv_slot):
        """Open HSV debug windows around the last ball in the previous frame."""
        if frame is None or point is None:
            print("No previous frame/ball position available for HSV debug.")
            return
        if hsv_slot == 0:
            lower, upper, label = self.get_standard_hsv_for_position(point[1])
        elif hsv_slot == 1:
            lower, upper, label = self.alt_focus_hsv_lower, self.alt_focus_hsv_upper, "alt1"
        elif hsv_slot == 2:
            lower, upper, label = self.alt2_hsv_lower, self.alt2_hsv_upper, "alt2"
        elif hsv_slot == 3:
            lower, upper, label = self.alt3_hsv_lower, self.alt3_hsv_upper, "alt3"
        else:
            print(f"Unknown HSV slot: {hsv_slot}")
            return
        if lower is None or upper is None:
            print(f"HSV range not available for slot {hsv_slot} ({label}).")
            return
        title = f"Prev frame {frame_index} ({label})"
        self.show_alternative_debug(frame, point, lower, upper, title)
        
    def load_hsv_config(self):
        """Load HSV values from config file."""
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                
                # Check if new format (with regular_court and behind_net)
                if "regular_court" in config and "behind_net" in config:
                    # New format with two HSV sets
                    self.hsv_regular = {
                        'lower': np.array([config["regular_court"]["h_min"], config["regular_court"]["s_min"], config["regular_court"]["v_min"]], dtype=np.uint8),
                        'upper': np.array([config["regular_court"]["h_max"], config["regular_court"]["s_max"], config["regular_court"]["v_max"]], dtype=np.uint8)
                    }
                    self.hsv_behind_net = {
                        'lower': np.array([config["behind_net"]["h_min"], config["behind_net"]["s_min"], config["behind_net"]["v_min"]], dtype=np.uint8),
                        'upper': np.array([config["behind_net"]["h_max"], config["behind_net"]["s_max"], config["behind_net"]["v_max"]], dtype=np.uint8)
                    }
                    self.pre_focus_hsv_regular = {
                        'lower': self.hsv_regular['lower'].copy(),
                        'upper': self.hsv_regular['upper'].copy()
                    }
                    self.pre_focus_hsv_regular['upper'][0] = min(self.pre_focus_hsv_regular['upper'][0], 73)
                    self.pre_focus_hsv_behind_net = {
                        'lower': self.hsv_behind_net['lower'].copy(),
                        'upper': self.hsv_behind_net['upper'].copy()
                    }
                    self.pre_focus_hsv_behind_net['upper'][0] = min(self.pre_focus_hsv_behind_net['upper'][0], 73)
                    self.net_area_y_min = config.get("net_area_y_min", 250)
                    self.net_area_y_max = config.get("net_area_y_max", 350)
                    
                    # Load high/low net boundaries if available
                    if "high_net_y_min" in config:
                        self.high_net_y_min = config["high_net_y_min"]
                        self.high_net_y_max = config["high_net_y_max"]
                    if "low_net_y_min" in config:
                        self.low_net_y_min = config["low_net_y_min"]
                        self.low_net_y_max = config["low_net_y_max"]
                    
                    # Set initial HSV to regular court
                    self.hsv_lower = self.hsv_regular['lower']
                    self.hsv_upper = self.hsv_regular['upper']
                    
                    print(f"Loaded DUAL HSV config:")
                    print(f"  Regular court: H:{config['regular_court']['h_min']}-{config['regular_court']['h_max']}, S:{config['regular_court']['s_min']}-{config['regular_court']['s_max']}, V:{config['regular_court']['v_min']}-{config['regular_court']['v_max']}")
                    print(f"  Behind net: H:{config['behind_net']['h_min']}-{config['behind_net']['h_max']}, S:{config['behind_net']['s_min']}-{config['behind_net']['s_max']}, V:{config['behind_net']['v_min']}-{config['behind_net']['v_max']}")
                    print(f"  Net area: Y={self.net_area_y_min}-{self.net_area_y_max}")
                    if hasattr(self, 'high_net_y_min') and hasattr(self, 'low_net_y_min'):
                        print(f"    HIGH NET: Y={self.high_net_y_min}-{self.high_net_y_max}")
                        print(f"    LOW NET: Y={self.low_net_y_min}-{self.low_net_y_max}")
                    
                    # Load serve area boundaries if available
                    if "serve_area_x_min" in config:
                        self.serve_area_points = config.get('serve_area_points', [])
                        self.serve_area_x_min = config['serve_area_x_min']
                        self.serve_area_x_max = config['serve_area_x_max']
                        self.serve_area_y_min = config['serve_area_y_min']
                        self.serve_area_y_max = config['serve_area_y_max']
                        print(f"  Serve area: X={self.serve_area_x_min}-{self.serve_area_x_max}, Y={self.serve_area_y_min}-{self.serve_area_y_max}")
                    self.serve_direction_dx = int(config.get('serve_direction_dx', 1))
                    if self.serve_direction_dx not in (-1, 1):
                        self.serve_direction_dx = 1
                    self.serve_direction_dy = int(config.get('serve_direction_dy', 0))
                    if self.serve_direction_dy not in (-1, 0, 1):
                        self.serve_direction_dy = 0
                    self.serve_directional_search = bool(config.get('serve_directional_search', False))
                    self.serve_directional_search_frames = int(config.get('serve_directional_search_frames', 20))
                    self.serve_contact_y_ratio = float(config.get('serve_contact_y_ratio', 0.7))
                    self.serve_contact_min_prev_dy = int(config.get('serve_contact_min_prev_dy', 20))
                    self.serve_contact_max_prev_dx = int(config.get('serve_contact_max_prev_dx', 40))
                    self.serve_contact_min_ball_size = int(config.get('serve_contact_min_ball_size', 100))
                    self.serve_contact_min_dx = int(config.get('serve_contact_min_dx', 80))
                    self.serve_contact_min_dy = int(config.get('serve_contact_min_dy', 0))
                    self.serve_width_ratio = config.get('serve_width_ratio')
                    print(f"  Serve direction: {self.serve_direction_label()}")
                    # Per-court ball size range for serve detection (defaults suit far-end small ball)
                    self.serve_ball_size_min = config.get('serve_ball_size_min', 3)
                    self.serve_ball_size_max = config.get('serve_ball_size_max', 80)
                    # Per-court max ball speed (px/frame) used as search-radius cap during tracking
                    if 'ball_max_speed' in config:
                        self.max_ball_speed = config['ball_max_speed']
                        print(f"  Ball max speed: {self.max_ball_speed}px/frame")
                    # Per-court point duration cap (frames) — forces POINT_ENDED if ball stays
                    # lost too long (handles near-end courts where ball leaves frame fast)
                    if 'max_point_frames' in config:
                        self.max_point_frames = config['max_point_frames']
                        print(f"  Max point duration: {self.max_point_frames} frames")
                else:
                    # Old format (single HSV set)
                    self.hsv_lower = np.array([config["h_min"], config["s_min"], config["v_min"]], dtype=np.uint8)
                    self.hsv_upper = np.array([config["h_max"], config["s_max"], config["v_max"]], dtype=np.uint8)
                    self.hsv_regular = None
                    self.hsv_behind_net = None
                    self.pre_focus_hsv_regular = None
                    self.pre_focus_hsv_behind_net = None
                    print(f"Loaded HSV values: H:{config['h_min']}-{config['h_max']}, S:{config['s_min']}-{config['s_max']}, V:{config['v_min']}-{config['v_max']}")
            
                # Set primary/alt HSV ranges (primary = config, alt = capped legacy)
                self.primary_hsv_lower = self.hsv_lower.copy()
                self.primary_hsv_upper = self.hsv_upper.copy()
                self.alt_hsv_lower = self.primary_hsv_lower.copy()
                self.alt_hsv_upper = self.primary_hsv_upper.copy()
                # Legacy cap for alt to keep the older narrower range
                self.alt_hsv_upper[0] = min(self.alt_hsv_upper[0], 73)
                # Alternative for focus-loss recovery: raise H max to 90
                self.alt_focus_hsv_lower = self.primary_hsv_lower.copy()
                self.alt_focus_hsv_upper = self.primary_hsv_upper.copy()
                self.alt_focus_hsv_upper[0] = min(179, max(self.alt_focus_hsv_upper[0], 90))
                # Alternative 2: narrower H range tuned for recovery
                self.alt2_hsv_lower = self.primary_hsv_lower.copy()
                self.alt2_hsv_upper = self.primary_hsv_upper.copy()
                self.alt2_hsv_lower[0] = 30
                self.alt2_hsv_upper[0] = 75
                # Alternative 3: higher S min to reduce false positives
                self.alt3_hsv_lower = self.primary_hsv_lower.copy()
                self.alt3_hsv_upper = self.primary_hsv_upper.copy()
                self.alt3_hsv_lower[1] = 55
                # Load overrides if present in config
                if "alt1" in config:
                    self.alt_focus_hsv_lower = np.array([
                        config["alt1"]["h_min"], config["alt1"]["s_min"], config["alt1"]["v_min"]
                    ], dtype=np.uint8)
                    self.alt_focus_hsv_upper = np.array([
                        config["alt1"]["h_max"], config["alt1"]["s_max"], config["alt1"]["v_max"]
                    ], dtype=np.uint8)
                if "alt2" in config:
                    self.alt2_hsv_lower = np.array([
                        config["alt2"]["h_min"], config["alt2"]["s_min"], config["alt2"]["v_min"]
                    ], dtype=np.uint8)
                    self.alt2_hsv_upper = np.array([
                        config["alt2"]["h_max"], config["alt2"]["s_max"], config["alt2"]["v_max"]
                    ], dtype=np.uint8)
                if "alt3" in config:
                    self.alt3_hsv_lower = np.array([
                        config["alt3"]["h_min"], config["alt3"]["s_min"], config["alt3"]["v_min"]
                    ], dtype=np.uint8)
                    self.alt3_hsv_upper = np.array([
                        config["alt3"]["h_max"], config["alt3"]["s_max"], config["alt3"]["v_max"]
                    ], dtype=np.uint8)
                if "alt4" in config:
                    self.alt4_hsv_lower = np.array([
                        config["alt4"]["h_min"], config["alt4"]["s_min"], config["alt4"]["v_min"]
                    ], dtype=np.uint8)
                    self.alt4_hsv_upper = np.array([
                        config["alt4"]["h_max"], config["alt4"]["s_max"], config["alt4"]["v_max"]
                    ], dtype=np.uint8)
                if "alt5" in config:
                    self.alt5_hsv_lower = np.array([
                        config["alt5"]["h_min"], config["alt5"]["s_min"], config["alt5"]["v_min"]
                    ], dtype=np.uint8)
                    self.alt5_hsv_upper = np.array([
                        config["alt5"]["h_max"], config["alt5"]["s_max"], config["alt5"]["v_max"]
                    ], dtype=np.uint8)
                if "alt6" in config:
                    self.alt6_hsv_lower = np.array([
                        config["alt6"]["h_min"], config["alt6"]["s_min"], config["alt6"]["v_min"]
                    ], dtype=np.uint8)
                    self.alt6_hsv_upper = np.array([
                        config["alt6"]["h_max"], config["alt6"]["s_max"], config["alt6"]["v_max"]
                    ], dtype=np.uint8)
                # Use primary as active by default (full config range)
                self.hsv_lower = self.primary_hsv_lower
                self.hsv_upper = self.primary_hsv_upper
                self._build_frame0_background_hotspots()
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default HSV values")
            return None
    
    def select_hsv_for_position(self, y_position, at_edge=False):
        """Select appropriate HSV config based on ball Y position."""
        if self.using_alt3_hsv and self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
            return self.alt3_hsv_lower, self.alt3_hsv_upper, "alternative_3"
        if self.using_alt2_hsv and self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
            return self.alt2_hsv_lower, self.alt2_hsv_upper, "alternative_2"
        if self.using_alt_hsv and self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
            return self.alt_focus_hsv_lower, self.alt_focus_hsv_upper, "alternative"
        if self.hsv_regular is None or not self._behind_net_enabled():
            # Single HSV mode
            if at_edge:
                # Use relaxed V_min at edges (ball may be darker when off-screen/returning)
                hsv_lower_edge = self.hsv_lower.copy()
                hsv_lower_edge[2] = max(70, self.hsv_lower[2] - 50)  # Lower V_min by 50
                return hsv_lower_edge, self.hsv_upper, "single_edge"
            if self.alt_hsv_lower is not None and self.alt_hsv_upper is not None:
                return self.alt_hsv_lower, self.alt_hsv_upper, "single_prefocus"
            return self.hsv_lower, self.hsv_upper, "single"
        
        # Check if ball is in net area
        if self.net_area_y_min <= y_position <= self.net_area_y_max:
            if not self.focus_loss_active and self.pre_focus_hsv_behind_net is not None:
                return self.pre_focus_hsv_behind_net['lower'], self.pre_focus_hsv_behind_net['upper'], "behind_net_prefocus"
            return self.hsv_behind_net['lower'], self.hsv_behind_net['upper'], "behind_net"
        elif at_edge:
            # Use behind_net HSV (more relaxed) at edges
            if not self.focus_loss_active and self.pre_focus_hsv_behind_net is not None:
                return self.pre_focus_hsv_behind_net['lower'], self.pre_focus_hsv_behind_net['upper'], "at_edge_prefocus"
            return self.hsv_behind_net['lower'], self.hsv_behind_net['upper'], "at_edge"
        else:
            if not self.focus_loss_active and self.pre_focus_hsv_regular is not None:
                return self.pre_focus_hsv_regular['lower'], self.pre_focus_hsv_regular['upper'], "regular_court_prefocus"
            return self.hsv_regular['lower'], self.hsv_regular['upper'], "regular_court"
    
    def mark_net_area(self, frame):
        """Interactive tool to mark the net area with 6 points (3 for high net, 3 for low net)."""
        print("\n" + "=" * 70)
        print("=== MARK NET AREA (6 POINTS) ===")
        print("=" * 70)
        print("Click 6 points clockwise around the net area:")
        print("  1. LEFT edge of HIGH net (top-left)")
        print("  2. MIDDLE of HIGH net (top-middle)")
        print("  3. RIGHT edge of HIGH net (top-right)")
        print("  4. RIGHT edge of LOW net (bottom-right)")
        print("  5. MIDDLE of LOW net (bottom-middle)")
        print("  6. LEFT edge of LOW net (bottom-left)")
        print("Press 'Q' or ESC when done (after marking all 6 points)")
        print("=" * 70)
        
        points = []
        window_name = "Mark Net Area - 6 Points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Resize frame for display
        screen_height = 900
        scale_factor = screen_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale_factor)
        display_frame = cv2.resize(frame, (new_width, screen_height))
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                # Convert to original frame coordinates
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                points.append((orig_x, orig_y))
                
                # Label points based on position
                labels = ["HIGH NET Left", "HIGH NET Middle", "HIGH NET Right", 
                         "LOW NET Right", "LOW NET Middle", "LOW NET Left"]
                print(f"Point {len(points)} ({labels[len(points)-1]}): ({orig_x}, {orig_y})")
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            display = display_frame.copy()
            
            # Draw existing points with color coding
            for i, pt in enumerate(points):
                disp_x = int(pt[0] * scale_factor)
                disp_y = int(pt[1] * scale_factor)
                
                # Color: Green for high net (1-3), Blue for low net (4-6)
                color = (0, 255, 0) if i < 3 else (255, 0, 0)
                
                cv2.circle(display, (disp_x, disp_y), 10, color, -1)
                cv2.putText(display, str(i+1), (disp_x+15, disp_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw lines for high net (points 1-2-3)
            if len(points) >= 2:
                pt1 = (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor))
                pt2 = (int(points[1][0] * scale_factor), int(points[1][1] * scale_factor))
                cv2.line(display, pt1, pt2, (0, 255, 0), 2)
            if len(points) >= 3:
                pt2 = (int(points[1][0] * scale_factor), int(points[1][1] * scale_factor))
                pt3 = (int(points[2][0] * scale_factor), int(points[2][1] * scale_factor))
                cv2.line(display, pt2, pt3, (0, 255, 0), 2)
            
            # Draw lines for low net (points 4-5-6)
            if len(points) >= 5:
                pt4 = (int(points[3][0] * scale_factor), int(points[3][1] * scale_factor))
                pt5 = (int(points[4][0] * scale_factor), int(points[4][1] * scale_factor))
                cv2.line(display, pt4, pt5, (255, 0, 0), 2)
            if len(points) >= 6:
                pt5 = (int(points[4][0] * scale_factor), int(points[4][1] * scale_factor))
                pt6 = (int(points[5][0] * scale_factor), int(points[5][1] * scale_factor))
                cv2.line(display, pt5, pt6, (255, 0, 0), 2)
            
            # Draw connecting lines between high and low net
            if len(points) >= 4:
                pt1 = (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor))
                pt4 = (int(points[3][0] * scale_factor), int(points[3][1] * scale_factor))
                cv2.line(display, pt1, pt4, (0, 255, 255), 2)
            if len(points) >= 6:
                pt3 = (int(points[2][0] * scale_factor), int(points[2][1] * scale_factor))
                pt6 = (int(points[5][0] * scale_factor), int(points[5][1] * scale_factor))
                cv2.line(display, pt3, pt6, (0, 255, 255), 2)
                
                # Fill semi-transparent area
                overlay = display.copy()
                pts_array = np.array([
                    [int(points[0][0] * scale_factor), int(points[0][1] * scale_factor)],
                    [int(points[1][0] * scale_factor), int(points[1][1] * scale_factor)],
                    [int(points[2][0] * scale_factor), int(points[2][1] * scale_factor)],
                    [int(points[5][0] * scale_factor), int(points[5][1] * scale_factor)],
                    [int(points[4][0] * scale_factor), int(points[4][1] * scale_factor)],
                    [int(points[3][0] * scale_factor), int(points[3][1] * scale_factor)]
                ], np.int32)
                cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            
            # Status text
            status = f"Points marked: {len(points)}/6"
            if len(points) < 3:
                status += " (HIGH NET)"
            elif len(points) < 6:
                status += " (LOW NET)"
            else:
                status += " (COMPLETE - Press Q)"
            
            cv2.putText(display, status, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                if len(points) == 6:
                    break
                else:
                    print(f"Please mark all 6 points (currently {len(points)}/6)")
        
        cv2.destroyWindow(window_name)
        
        if len(points) == 6:
            # Points order: 0,1,2 = high net (left, middle, right)
            #               3,4,5 = low net (right, middle, left)
            high_net_y_coords = [points[0][1], points[1][1], points[2][1]]
            low_net_y_coords = [points[3][1], points[4][1], points[5][1]]
            
            self.high_net_y_min = min(high_net_y_coords)
            self.high_net_y_max = max(high_net_y_coords)
            self.low_net_y_min = min(low_net_y_coords)
            self.low_net_y_max = max(low_net_y_coords)
            
            # Overall net area encompasses both high and low
            self.net_area_y_min = min(self.high_net_y_min, self.low_net_y_min)
            self.net_area_y_max = max(self.high_net_y_max, self.low_net_y_max)
            
            print(f"\nNet area marked successfully!")
            print(f"HIGH NET: Y={self.high_net_y_min} - {self.high_net_y_max}")
            print(f"LOW NET: Y={self.low_net_y_min} - {self.low_net_y_max}")
            print(f"OVERALL NET AREA: Y={self.net_area_y_min} - {self.net_area_y_max}")
            
            # Save to config
            self.save_net_area_to_config()
            return True
        
        return False
    
    def save_net_area_to_config(self):
        """Save net area boundaries to config file."""
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            
            # Save overall net area
            config["net_area_y_min"] = int(self.net_area_y_min)
            config["net_area_y_max"] = int(self.net_area_y_max)
            
            # Save detailed high/low net boundaries if available
            if hasattr(self, 'high_net_y_min'):
                config["high_net_y_min"] = int(self.high_net_y_min)
                config["high_net_y_max"] = int(self.high_net_y_max)
            if hasattr(self, 'low_net_y_min'):
                config["low_net_y_min"] = int(self.low_net_y_min)
                config["low_net_y_max"] = int(self.low_net_y_max)
            
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"Net area boundaries saved to hsv_config.json")
        except Exception as e:
            print(f"Error saving net area: {e}")
    
    def open_hsv_tuner(self, frame, point):
        """Open HSV filter tuner with 100x100 region around the ball."""
        x, y = point
        
        # Extract larger region around the ball for debugging - MUST BE FRESH COPY
        region_size = 150  # 150 pixels radius = 300x300 total
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(frame.shape[1], x + region_size)
        y2 = min(frame.shape[0], y + region_size)
        
        print(f"\n=== EXTRACTING NEW REGION ===")
        print(f"Click position: ({x}, {y})")
        print(f"Extracting region: ({x1},{y1}) to ({x2},{y2}) from frame {frame.shape}")
        
        # IMPORTANT: Extract fresh region from the current frame
        region = frame[y1:y2, x1:x2].copy()  # .copy() ensures fresh data
        if region.size == 0:
            print("Error: Cannot extract 100x100 region around ball")
            return False
        
        print(f"Extracted region shape: {region.shape}")
        
        # Resize to exactly 300x300 if needed
        if region.shape[0] != 300 or region.shape[1] != 300:
            region = cv2.resize(region, (300, 300))
            print(f"Resized region to: {region.shape}")
        
        # Get initial HSV values at the clicked point (relative to region)
        rel_x = x - x1
        rel_y = y - y1
        
        # Adjust relative coordinates if resizing happened
        if x2 - x1 != 300:
            rel_x = int((rel_x / (x2 - x1)) * 300)
        if y2 - y1 != 300:
            rel_y = int((rel_y / (y2 - y1)) * 300)
        
        # Ensure relative coordinates are within bounds
        rel_x = max(0, min(299, rel_x))
        rel_y = max(0, min(299, rel_y))
        
        print(f"Relative coordinates: ({rel_x}, {rel_y}) in 100x100 region")
        
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        initial_hsv = hsv_region[rel_y, rel_x]
        
        print(f"Ball position: ({x}, {y})")
        print(f"Initial HSV: H={initial_hsv[0]}, S={initial_hsv[1]}, V={initial_hsv[2]}")
        print("Adjust sliders until ball appears WHITE, then press 'S' to save")
        
        # Create HSV tuner window
        tuner_window = "HSV Filter Tuner - 300x300"
        cv2.namedWindow(tuner_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tuner_window, 800, 600)
        cv2.waitKey(1)  # Allow window to be created
        
        # Use current active HSV values as starting point
        h_min = int(self.hsv_lower[0]) if self.hsv_lower is not None else 20
        h_max = int(self.hsv_upper[0]) if self.hsv_upper is not None else 90
        s_min = int(self.hsv_lower[1]) if self.hsv_lower is not None else 20
        s_max = int(self.hsv_upper[1]) if self.hsv_upper is not None else 255
        v_min = int(self.hsv_lower[2]) if self.hsv_lower is not None else 70
        v_max = int(self.hsv_upper[2]) if self.hsv_upper is not None else 255
        
        # Bulb size filter (wider to catch first frames)
        bulb_min = 1
        bulb_max = 30
        
        # Create trackbars for HSV and bulb size
        cv2.createTrackbar('H Min', tuner_window, h_min, 179, lambda x: None)
        cv2.createTrackbar('H Max', tuner_window, h_max, 179, lambda x: None)
        cv2.createTrackbar('S Min', tuner_window, s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', tuner_window, s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', tuner_window, v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', tuner_window, v_max, 255, lambda x: None)
        cv2.createTrackbar('Bulb Min', tuner_window, bulb_min, 100, lambda x: None)
        cv2.createTrackbar('Bulb Max', tuner_window, bulb_max, 100, lambda x: None)
        cv2.waitKey(1)  # Allow trackbars to be created
        
        def update_display():
            # Get current trackbar values
            h_min = cv2.getTrackbarPos('H Min', tuner_window)
            h_max = cv2.getTrackbarPos('H Max', tuner_window)
            s_min = cv2.getTrackbarPos('S Min', tuner_window)
            s_max = cv2.getTrackbarPos('S Max', tuner_window)
            v_min = cv2.getTrackbarPos('V Min', tuner_window)
            v_max = cv2.getTrackbarPos('V Max', tuner_window)
            bulb_min = cv2.getTrackbarPos('Bulb Min', tuner_window)
            bulb_max = cv2.getTrackbarPos('Bulb Max', tuner_window)
            
            # Apply HSV filter to the fresh region (100x100)
            hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask_region = cv2.inRange(hsv_region, hsv_lower, hsv_upper)
            
            # Clean up mask for region
            kernel = np.ones((2, 2), np.uint8)
            mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_OPEN, kernel)
            mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, kernel)
            
            # Apply HSV filter to the local region only and find bulbs
            mask_local = cv2.inRange(hsv_region, hsv_lower, hsv_upper)
            mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_OPEN, kernel)
            mask_local = cv2.morphologyEx(mask_local, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw detected bulbs on the local region
            local_display = region.copy()
            detected_bulbs = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if bulb_min <= area <= bulb_max:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(local_display, (cx, cy), 5, (0, 255, 0), 1)
                        cv2.putText(local_display, f"{int(area)}", (cx+7, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        detected_bulbs += 1
            
            # Show local region with detected bulbs
            cv2.namedWindow("Detected Bulbs", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected Bulbs", 800, 600)
            cv2.putText(local_display, f"H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max}",
                        (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow("Detected Bulbs", local_display)
            
            # Create side-by-side display for 100x100 region
            display = np.zeros((320, 620, 3), dtype=np.uint8)
            display[10:310, 10:310] = region  # Original on left
            display[10:310, 320:620] = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR)  # Filter on right
            
            # Draw borders
            cv2.rectangle(display, (10, 10), (309, 309), (255, 255, 255), 1)
            cv2.rectangle(display, (320, 10), (619, 309), (255, 255, 255), 1)
            
            # Show coordinates with small + marker
            cv2.line(display, (rel_x + 8, rel_y + 10), (rel_x + 12, rel_y + 10), (0, 255, 0), 1)
            cv2.line(display, (rel_x + 10, rel_y + 8), (rel_x + 10, rel_y + 12), (0, 255, 0), 1)
            
            # Show info
            bulb_size_region = np.sum(mask_region > 0)
            cv2.putText(display, f"Size: {bulb_size_region}px", (10, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"Found: {detected_bulbs}", (320, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow(tuner_window, display)
            return hsv_lower, hsv_upper, bulb_size_region
        
        # Initial display
        update_display()
        
        print("Controls:")
        print("- Adjust sliders to make ball WHITE")
        print("- Press 'S' to SAVE and continue")
        print("- Press 'Q' to CANCEL")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save HSV values
                hsv_lower, hsv_upper, bulb_size = update_display()
                self.hsv_lower = hsv_lower
                self.hsv_upper = hsv_upper
                self.ball_center = point
                self.ball_size = bulb_size
                self.tracking = True
                
                print(f"\nHSV Filter SAVED!")
                print(f"HSV Range: H={hsv_lower[0]}-{hsv_upper[0]}, S={hsv_lower[1]}-{hsv_upper[1]}, V={hsv_lower[2]}-{hsv_upper[2]}")
                print(f"Bulb size: {bulb_size} pixels")
                
                # Add to HSV table
                self.hsv_table.append({
                    'frame': self.frame_count,
                    'position': (x, y),
                    'hsv': initial_hsv.tolist(),
                    'bulb_size': bulb_size,
                    'hsv_range': [hsv_lower[0], hsv_upper[0], hsv_lower[1], hsv_upper[1], hsv_lower[2], hsv_upper[2]]
                })
                
                cv2.destroyWindow(tuner_window)
                cv2.destroyWindow("Detected Bulbs")
                return True
            elif key == ord('q'):
                print("HSV tuning cancelled")
                cv2.destroyWindow(tuner_window)
                cv2.destroyWindow("Detected Bulbs")
                return False
            else:
                # Update display
                update_display()
        
        return False
    
    def analyze_ball_at_point(self, frame, point):
        """Analyze ball at the clicked point - simple version without HSV tuner."""
        x, y = point
        
        # Store the initial marked position to return to after ball stops
        self.initial_ball_position = point
        
        # Extract small region around clicked point to calculate bulb size
        region_size = 10
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(frame.shape[1], x + region_size)
        y2 = min(frame.shape[0], y + region_size)
        
        region = frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_region, self.hsv_lower, self.hsv_upper)
        
        # Calculate bulb size
        bulb_size = np.sum(mask > 0)
        
        # Get HSV values at clicked point
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_values = hsv_frame[y, x]
        
        # Save analysis results
        self.ball_center = point
        self.ball_hsv = hsv_values
        self.ball_size = bulb_size
        self.tracking = True
        self.ball_stopped = False
        self.ball_velocity_history = []  # Track recent velocities to detect stop
        
        # Track if this is a recovery from a lost ball
        if hasattr(self, 'ball_center') and self.ball_center is not None and self.tracking:
            prev_pos = self.ball_center
            print(f"\n{'='*70}")
            print(f"=== BALL TRACKING LOST - MANUAL RECOVERY NEEDED ===")
            print(f"{'='*70}")
            print(f"Previous ball position: {prev_pos}")
            print(f"New marked position: ({x}, {y})")
            print(f"Distance jumped: {np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2):.1f}px")
            print(f"\n>>> ANALYZING WHY BALL WAS LOST <<<")
            
            # Check if HSV of new ball is within our filter range
            print(f"\nHSV Analysis:")
            print(f"  Current HSV filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
            print(f"  Ball HSV at new position: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]}")
            
            # Check each component
            h_in_range = self.hsv_lower[0] <= hsv_values[0] <= self.hsv_upper[0]
            s_in_range = self.hsv_lower[1] <= hsv_values[1] <= self.hsv_upper[1]
            v_in_range = self.hsv_lower[2] <= hsv_values[2] <= self.hsv_upper[2]
            
            print(f"  H in range: {h_in_range} {'[OK]' if h_in_range else '[OUT OF RANGE!]'}")
            print(f"  S in range: {s_in_range} {'[OK]' if s_in_range else '[OUT OF RANGE!]'}")
            print(f"  V in range: {v_in_range} {'[OK]' if v_in_range else '[OUT OF RANGE!]'}")
            
            if not (h_in_range and s_in_range and v_in_range):
                print(f"\n  *** ROOT CAUSE: Ball HSV is OUTSIDE filter range! ***")
                print(f"  *** SOLUTION: Need to widen HSV filter to include this ball ***")
            else:
                print(f"\n  *** Ball HSV is WITHIN filter range - other issue ***")
                print(f"  *** Possible causes: size changed, moved too fast, or occluded ***")
            
            print(f"{'='*70}")
            
            # Show interactive HSV filter tuner with bulb sizes
            self._open_interactive_hsv_tuner(frame, prev_pos)
        else:
            print(f"\n=== BALL MARKED AT ({x}, {y}) ===")
        
        print(f"HSV at point: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]}")
        print(f"HSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]} (Tennis ball optimized)")
        print(f"Bulb size: {bulb_size} pixels")
        print(f"Initial position saved: ({x}, {y})")
        print("Ball tracking enabled! Press 'D' to advance to next frame")
        
        # Add to table
        self.hsv_table.append({
            'frame': self.frame_count,
            'position': (x, y),
            'hsv': hsv_values.tolist(),
            'bulb_size': bulb_size,
            'hsv_range': [self.hsv_lower[0], self.hsv_upper[0], self.hsv_lower[1], self.hsv_upper[1], self.hsv_lower[2], self.hsv_upper[2]]
        })
        
        return True
    
    def _open_interactive_hsv_tuner(self, frame, search_center):
        """Open interactive HSV filter tuner with bulb size controls."""
        print(f"\n>>> OPENING DUAL HSV FILTER TUNER <<<")
        print(f"Showing search region around last known position: {search_center}")
        
        # Check if dual HSV mode is enabled
        is_dual_mode = self.hsv_regular is not None and self._behind_net_enabled()
        
        if is_dual_mode:
            print(f"Opening TWO windows:")
            print(f"  1. REGULAR COURT HSV (left window)")
            print(f"  2. BEHIND NET HSV (right window)")
            print(f"Adjust the appropriate window based on ball location")
        else:
            print(f"Adjust HSV min/max and Bulb size min/max sliders")
        
        print(f"Press 'Q' or ESC to close and continue")
        
        # Get search region (120px radius)
        search_radius = 120
        x_prev, y_prev = search_center
        x1 = max(0, x_prev - search_radius)
        y1 = max(0, y_prev - search_radius)
        x2 = min(frame.shape[1], x_prev + search_radius)
        y2 = min(frame.shape[0], y_prev + search_radius)
        
        search_frame = frame[y1:y2, x1:x2].copy()
        
        if is_dual_mode:
            # Create TWO windows for dual HSV mode
            window_regular = "1. REGULAR COURT HSV"
            window_behind_net = "2. BEHIND NET HSV"
            
            cv2.namedWindow(window_regular)
            cv2.namedWindow(window_behind_net)
            cv2.resizeWindow(window_regular, 800, 600)
            cv2.resizeWindow(window_behind_net, 800, 600)
            cv2.moveWindow(window_regular, 50, 50)
            cv2.moveWindow(window_behind_net, 900, 50)
            cv2.waitKey(1)
            
            # Regular court HSV values
            h_min_reg = self.hsv_regular['lower'][0]
            h_max_reg = self.hsv_regular['upper'][0]
            s_min_reg = self.hsv_regular['lower'][1]
            s_max_reg = self.hsv_regular['upper'][1]
            v_min_reg = self.hsv_regular['lower'][2]
            v_max_reg = self.hsv_regular['upper'][2]
            
            # Behind net HSV values
            h_min_net = self.hsv_behind_net['lower'][0]
            h_max_net = self.hsv_behind_net['upper'][0]
            s_min_net = self.hsv_behind_net['lower'][1]
            s_max_net = self.hsv_behind_net['upper'][1]
            v_min_net = self.hsv_behind_net['lower'][2]
            v_max_net = self.hsv_behind_net['upper'][2]
            
            bulb_min = 3
            bulb_max = 80
            
            # Create trackbars for REGULAR COURT window
            cv2.createTrackbar("H Min", window_regular, h_min_reg, 179, lambda x: None)
            cv2.createTrackbar("H Max", window_regular, h_max_reg, 179, lambda x: None)
            cv2.createTrackbar("S Min", window_regular, s_min_reg, 255, lambda x: None)
            cv2.createTrackbar("S Max", window_regular, s_max_reg, 255, lambda x: None)
            cv2.createTrackbar("V Min", window_regular, v_min_reg, 255, lambda x: None)
            cv2.createTrackbar("V Max", window_regular, v_max_reg, 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", window_regular, bulb_min, 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", window_regular, bulb_max, 200, lambda x: None)
            
            # Create trackbars for BEHIND NET window
            cv2.createTrackbar("H Min", window_behind_net, h_min_net, 179, lambda x: None)
            cv2.createTrackbar("H Max", window_behind_net, h_max_net, 179, lambda x: None)
            cv2.createTrackbar("S Min", window_behind_net, s_min_net, 255, lambda x: None)
            cv2.createTrackbar("S Max", window_behind_net, s_max_net, 255, lambda x: None)
            cv2.createTrackbar("V Min", window_behind_net, v_min_net, 255, lambda x: None)
            cv2.createTrackbar("V Max", window_behind_net, v_max_net, 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", window_behind_net, bulb_min, 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", window_behind_net, bulb_max, 200, lambda x: None)
            cv2.waitKey(1)
            
            while True:
                # Get REGULAR COURT trackbar values
                h_min_reg = cv2.getTrackbarPos("H Min", window_regular)
                h_max_reg = cv2.getTrackbarPos("H Max", window_regular)
                s_min_reg = cv2.getTrackbarPos("S Min", window_regular)
                s_max_reg = cv2.getTrackbarPos("S Max", window_regular)
                v_min_reg = cv2.getTrackbarPos("V Min", window_regular)
                v_max_reg = cv2.getTrackbarPos("V Max", window_regular)
                bulb_min_reg = cv2.getTrackbarPos("Bulb Min", window_regular)
                bulb_max_reg = cv2.getTrackbarPos("Bulb Max", window_regular)
                
                # Get BEHIND NET trackbar values
                h_min_net = cv2.getTrackbarPos("H Min", window_behind_net)
                h_max_net = cv2.getTrackbarPos("H Max", window_behind_net)
                s_min_net = cv2.getTrackbarPos("S Min", window_behind_net)
                s_max_net = cv2.getTrackbarPos("S Max", window_behind_net)
                v_min_net = cv2.getTrackbarPos("V Min", window_behind_net)
                v_max_net = cv2.getTrackbarPos("V Max", window_behind_net)
                bulb_min_net = cv2.getTrackbarPos("Bulb Min", window_behind_net)
                bulb_max_net = cv2.getTrackbarPos("Bulb Max", window_behind_net)
                
                # Apply REGULAR COURT HSV filter
                hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
                hsv_lower_reg = np.array([h_min_reg, s_min_reg, v_min_reg], dtype=np.uint8)
                hsv_upper_reg = np.array([h_max_reg, s_max_reg, v_max_reg], dtype=np.uint8)
                mask_reg = cv2.inRange(hsv_frame, hsv_lower_reg, hsv_upper_reg)
                contours_reg, _ = cv2.findContours(mask_reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                display_reg = search_frame.copy()
                bulb_count_reg = 0
                for contour in contours_reg:
                    area = cv2.contourArea(contour)
                    if area < bulb_min_reg or area > bulb_max_reg:
                        continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_reg, (cx, cy), 8, (0, 255, 0), 2)
                    cv2.putText(display_reg, f"{area:.0f}", (cx + 12, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    bulb_count_reg += 1
                
                mask_reg_bgr = cv2.cvtColor(mask_reg, cv2.COLOR_GRAY2BGR)
                combined_reg = np.hstack([display_reg, mask_reg_bgr])
                info_reg = f"REGULAR COURT | Bulbs: {bulb_count_reg} | H={h_min_reg}-{h_max_reg} S={s_min_reg}-{s_max_reg} V={v_min_reg}-{v_max_reg}"
                cv2.putText(combined_reg, info_reg, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_regular, combined_reg)
                
                # Apply BEHIND NET HSV filter
                hsv_lower_net = np.array([h_min_net, s_min_net, v_min_net], dtype=np.uint8)
                hsv_upper_net = np.array([h_max_net, s_max_net, v_max_net], dtype=np.uint8)
                mask_net = cv2.inRange(hsv_frame, hsv_lower_net, hsv_upper_net)
                contours_net, _ = cv2.findContours(mask_net, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                display_net = search_frame.copy()
                bulb_count_net = 0
                for contour in contours_net:
                    area = cv2.contourArea(contour)
                    if area < bulb_min_net or area > bulb_max_net:
                        continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_net, (cx, cy), 8, (255, 0, 0), 2)
                    cv2.putText(display_net, f"{area:.0f}", (cx + 12, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    bulb_count_net += 1
                
                mask_net_bgr = cv2.cvtColor(mask_net, cv2.COLOR_GRAY2BGR)
                combined_net = np.hstack([display_net, mask_net_bgr])
                info_net = f"BEHIND NET | Bulbs: {bulb_count_net} | H={h_min_net}-{h_max_net} S={s_min_net}-{s_max_net} V={v_min_net}-{v_max_net}"
                cv2.putText(combined_net, info_net, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(window_behind_net, combined_net)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
            
            # Close both windows
            cv2.destroyWindow(window_regular)
            cv2.destroyWindow(window_behind_net)
            cv2.waitKey(1)
            
            # Update both HSV configs
            self.hsv_regular['lower'] = hsv_lower_reg
            self.hsv_regular['upper'] = hsv_upper_reg
            self.hsv_behind_net['lower'] = hsv_lower_net
            self.hsv_behind_net['upper'] = hsv_upper_net
            
            # Determine which HSV to use as active based on ball position
            y_pos = search_center[1]
            is_in_net_area = (hasattr(self, 'net_area_y_min') and 
                             hasattr(self, 'net_area_y_max') and
                             self.net_area_y_min <= y_pos <= self.net_area_y_max)
            
            if is_in_net_area:
                self.hsv_lower = hsv_lower_net
                self.hsv_upper = hsv_upper_net
                h_min, h_max = h_min_net, h_max_net
                s_min, s_max = s_min_net, s_max_net
                v_min, v_max = v_min_net, v_max_net
                bulb_min, bulb_max = bulb_min_net, bulb_max_net
            else:
                self.hsv_lower = hsv_lower_reg
                self.hsv_upper = hsv_upper_reg
                h_min, h_max = h_min_reg, h_max_reg
                s_min, s_max = s_min_reg, s_max_reg
                v_min, v_max = v_min_reg, v_max_reg
                bulb_min, bulb_max = bulb_min_reg, bulb_max_reg
            
        else:
            # Single HSV mode (original behavior)
            tuner_window = "HSV Filter Tuner"
            cv2.namedWindow(tuner_window)
            cv2.resizeWindow(tuner_window, 1200, 600)
            cv2.waitKey(1)
            
            # Current HSV values
            h_min = self.hsv_lower[0]
            h_max = self.hsv_upper[0]
            s_min = self.hsv_lower[1]
            s_max = self.hsv_upper[1]
            v_min = self.hsv_lower[2]
            v_max = self.hsv_upper[2]
            bulb_min = 3
            bulb_max = 80
            
            # Create trackbars
            cv2.createTrackbar("H Min", tuner_window, h_min, 179, lambda x: None)
            cv2.createTrackbar("H Max", tuner_window, h_max, 179, lambda x: None)
            cv2.createTrackbar("S Min", tuner_window, s_min, 255, lambda x: None)
            cv2.createTrackbar("S Max", tuner_window, s_max, 255, lambda x: None)
            cv2.createTrackbar("V Min", tuner_window, v_min, 255, lambda x: None)
            cv2.createTrackbar("V Max", tuner_window, v_max, 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", tuner_window, bulb_min, 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", tuner_window, bulb_max, 200, lambda x: None)
            cv2.waitKey(1)
            
            while True:
                # Get current trackbar values
                h_min = cv2.getTrackbarPos("H Min", tuner_window)
                h_max = cv2.getTrackbarPos("H Max", tuner_window)
                s_min = cv2.getTrackbarPos("S Min", tuner_window)
                s_max = cv2.getTrackbarPos("S Max", tuner_window)
                v_min = cv2.getTrackbarPos("V Min", tuner_window)
                v_max = cv2.getTrackbarPos("V Max", tuner_window)
                bulb_min = cv2.getTrackbarPos("Bulb Min", tuner_window)
                bulb_max = cv2.getTrackbarPos("Bulb Max", tuner_window)
                
                # Apply HSV filter
                hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
                hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
                hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
                mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw detected bulbs on original frame
                display_frame = search_frame.copy()
                bulb_count = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < bulb_min or area > bulb_max:
                        continue
                        
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw circle and size label
                    cv2.circle(display_frame, (cx, cy), 8, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{area:.0f}", (cx + 12, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    bulb_count += 1
                
                # Create side-by-side display
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([display_frame, mask_bgr])
                
                # Add text info
                info_text = f"Bulbs detected: {bulb_count} | HSV: H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max} | Size: {bulb_min}-{bulb_max}px"
                cv2.putText(combined, info_text, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow(tuner_window, combined)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
            
            # Update HSV values with tuned values
            self.hsv_lower = hsv_lower
            self.hsv_upper = hsv_upper
            
            # Close tuner window
            try:
                cv2.destroyWindow(tuner_window)
                cv2.waitKey(1)
            except:
                pass
        
        # Save to config file for persistence
        # Load existing config to preserve structure
        try:
            with open('hsv_config.json', 'r') as f:
                hsv_config = json.load(f)
        except:
            hsv_config = {}
        
        # Update the appropriate section
        if is_dual_mode:
            # In dual mode, save BOTH configurations
            print(f"Saving BOTH HSV configurations:")
            print(f"  Regular Court: H={h_min_reg}-{h_max_reg}, S={s_min_reg}-{s_max_reg}, V={v_min_reg}-{v_max_reg}")
            print(f"  Behind Net: H={h_min_net}-{h_max_net}, S={s_min_net}-{s_max_net}, V={v_min_net}-{v_max_net}")
            
            hsv_config['regular_court'] = {
                'h_min': int(h_min_reg),
                'h_max': int(h_max_reg),
                's_min': int(s_min_reg),
                's_max': int(s_max_reg),
                'v_min': int(v_min_reg),
                'v_max': int(v_max_reg)
            }
            hsv_config['behind_net'] = {
                'h_min': int(h_min_net),
                'h_max': int(h_max_net),
                's_min': int(s_min_net),
                's_max': int(s_max_net),
                'v_min': int(v_min_net),
                'v_max': int(v_max_net)
            }
        else:
            # Single HSV mode - update entire config
            hsv_config = {
                'h_min': int(h_min),
                'h_max': int(h_max),
                's_min': int(s_min),
                's_max': int(s_max),
                'v_min': int(v_min),
                'v_max': int(v_max)
            }
        
        with open('hsv_config.json', 'w') as f:
            json.dump(hsv_config, f, indent=4)
        
        # Close tuner window(s) - already closed in dual mode
        if not is_dual_mode:
            try:
                cv2.destroyWindow(tuner_window)
                cv2.waitKey(1)
            except:
                pass
        
        print(f"Tuner closed. Updated HSV: H={h_min}-{h_max}, S={s_min}-{s_max}, V={v_min}-{v_max}")
        print(f"HSV values saved to hsv_config.json")
        print(f"Bulb size filter: {bulb_min}-{bulb_max}px\n")
    
    def _reacquire_ball_by_motion(self, frame):
        """Re-acquire ball after occlusion using frame differencing + HSV.

        Returns (x, y) of the best ball candidate, or None if not found.
        Uses frame differencing between current frame and the previous frame
        to detect moving objects. Then filters by HSV color to confirm ball candidates.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self._prev_frame_gray, gray)
        # Update stored frame for next call (consecutive frame comparison)
        self._prev_frame_gray = gray
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # Morphological cleanup — minimal dilation to not merge nearby blobs
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Find motion regions
        motion_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_height, frame_width = frame.shape[:2]

        # Build list of HSV filters
        hsv_filters = []
        if hasattr(self, 'hsv_regular') and self.hsv_regular is not None:
            hsv_filters.append(self.hsv_regular['lower'])
            hsv_filters.append(self.hsv_regular['upper'])
        if hasattr(self, 'alt2_hsv_lower') and self.alt2_hsv_lower is not None:
            hsv_filters.append(self.alt2_hsv_lower)
            hsv_filters.append(self.alt2_hsv_upper)

        stuck_x, stuck_y = self.ball_center if self.ball_center else (0, 0)

        # Maximum distance to search from last known position.
        # Tighten cap when recent velocity was slow (toss phase): the ball can't have
        # traveled far while occluded, so a 500px cap prevents jumping to moving players/
        # rackets far from the serve area.  When the ball was already moving fast, allow
        # a larger cap so post-rally re-acquisitions still work.
        _recent_max_vel = max(self.ball_velocity_history[-3:]) if len(self.ball_velocity_history) >= 1 else 0
        if _recent_max_vel < 50:
            # Slow movement = toss/serve phase: stay near last known position
            _max_reacq_dist = max(250, min(500, self.max_ball_speed))
        else:
            _max_reacq_dist = max(250, min(700, self.max_ball_speed * max(self.stuck_frame_count, 2)))

        contact_bounds = self._contact_reacquire_bounds(frame.shape, self.ball_center)
        if contact_bounds is not None:
            _max_reacq_dist = min(_max_reacq_dist, contact_bounds['max_dist'])

        # Track static positions across re-acquisition attempts
        # If a candidate appears at the same spot as previous attempt, it's static noise
        prev_reacq = getattr(self, '_prev_reacq_candidates', [])

        best_candidate = None
        best_score = float('inf')
        candidate_count = 0

        for mc in motion_contours:
            motion_area = cv2.contourArea(mc)
            # The ball creates a moderate motion blob (typically 50-700px in diff image).
            # In 4K video a fast-moving ball after serve contact can produce frame-diff
            # blobs of 400-700px; the old 300px cap was rejecting these real ball blobs.
            # Very small blobs (< 30) are compression artifacts.
            # Very large blobs (> 700) are player body / arm movement.
            if motion_area < 30 or motion_area > 700:
                continue

            M = cv2.moments(mc)
            if M["m00"] == 0:
                continue
            mx = int(M["m10"] / M["m00"])
            my = int(M["m01"] / M["m00"])

            dist_from_stuck = np.sqrt((mx - stuck_x)**2 + (my - stuck_y)**2)

            # Skip motion at the exact stuck spot (likely static noise / same reflective patch)
            if dist_from_stuck < 40:
                continue

            # Skip candidates far outside the expected travel range — prevents jumping to
            # unrelated corners or edges that happen to match ball colour.
            if dist_from_stuck > _max_reacq_dist:
                print(f"  DEBUG: [REACQ] SKIPPED distant blob at ({mx},{my}) dist={dist_from_stuck:.0f}px > cap={_max_reacq_dist:.0f}px")
                continue

            if contact_bounds is not None:
                if (mx < contact_bounds['min_x'] or mx > contact_bounds['max_x'] or
                        my < contact_bounds['min_y'] or my > contact_bounds['max_y']):
                    print(f"  DEBUG: [REACQ] SKIPPED contact-phase blob at ({mx},{my}) outside "
                          f"x={contact_bounds['min_x']}-{contact_bounds['max_x']} "
                          f"y={contact_bounds['min_y']}-{contact_bounds['max_y']}")
                    continue

            # Hard-reject frame-edge artifacts: the tennis ball is always inside the court,
            # never right at the image border.  A 20px margin filters out compression
            # noise at the pixel border without excluding any legitimate ball position.
            if mx < 20 or my < 20 or mx > frame_width - 20 or my > frame_height - 20:
                continue

            # NOTE: Do NOT exclude the bottom half — tennis rallies happen across the
            # whole frame, and a racket hit can send the ball 800+ px in one frame.

            # Check ball color at the motion centroid
            h, s, v = hsv_frame[min(my, frame_height - 1), min(mx, frame_width - 1)]
            is_ball_color = False
            for i in range(0, len(hsv_filters), 2):
                lower = hsv_filters[i]
                upper = hsv_filters[i + 1]
                if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                    is_ball_color = True
                    break

            if not is_ball_color:
                continue

            # Skip candidates that appeared at the same position in previous re-acquisition
            # (static noise that persists across frames)
            is_static = False
            for prev_x, prev_y in prev_reacq:
                if abs(mx - prev_x) < 30 and abs(my - prev_y) < 30:
                    is_static = True
                    break
            if is_static:
                print(f"  DEBUG: [REACQ] SKIPPED static blob at ({mx},{my})")
                continue

            candidate_count += 1

            # Score: favor small motion blobs (closer to ball size) in the playing area.
            # Penalize frame edges heavily.
            edge_penalty = 0
            if my < 20 or mx < 20 or mx > frame_width - 20:
                edge_penalty = 200

            # Prefer motion in the full playing area of the court (top to ~90% of frame)
            y_score = 0
            if 50 < my < int(frame_height * 0.9):
                y_score = -30  # bonus for being in the court area

            # Prefer candidates closer to the last known position so that a ball
            # briefly hidden (e.g. by racket contact) is re-found near where we lost it
            # rather than a far-away false positive.
            dist_penalty = dist_from_stuck * 0.15

            score = motion_area * 0.3 + edge_penalty + y_score + dist_penalty

            print(f"  DEBUG: [REACQ] Motion+HSV at ({mx},{my}) area={motion_area:.0f}, "
                  f"H={h} S={s} V={v}, dist_stuck={dist_from_stuck:.0f}, score={score:.1f}")

            if score < best_score:
                best_score = score
                best_candidate = (mx, my)

        print(f"  DEBUG: [REACQ] Total candidates after filtering: {candidate_count}")
        # Store all candidates as potential static positions for next attempt
        self._prev_reacq_candidates = [(mx, my) for mc in motion_contours
                                        if cv2.contourArea(mc) >= 30 and cv2.contourArea(mc) <= 700
                                        for M in [cv2.moments(mc)]
                                        if M["m00"] > 0
                                        for mx, my in [(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))]]
        return best_candidate

    def track_ball_in_frame(self, frame, allow_inactive=False):
        """Track ball in current frame using HSV filter with debug information."""
        if (not self.tracking and not allow_inactive) or self.hsv_lower is None:
            return None

        frame_height, frame_width = frame.shape[:2]
        self._prune_ignored_tracking_positions()
        frame_gray = None
        if hasattr(self, '_prev_frame_gray') and self._prev_frame_gray is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Default: no saved previous frame for FFS motion check (overwritten in normal
        # tracking path before the reacquire call).
        _ffs_prev_gray = None

        # Post-reacquire flag: True for the window of frames immediately after a
        # motion-based re-acquisition, during which the serve contact can reverse the
        # ball.  Initialised here (False) so FFS and other early-return paths always
        # have it defined.  The normal-tracking block overwrites it before any decrement.
        _in_post_reacq = False
        serve_direction_search = False
        serve_contact_grace = getattr(self, '_serve_contact_grace_frames', 0) > 0
        rally_contact_grace = getattr(self, '_rally_contact_grace_frames', 0) > 0
        ground_bounce_grace = getattr(self, '_ground_bounce_grace_frames', 0) > 0
        contact_recovery_active = getattr(self, '_contact_recovery_frames', 0) > 0
        contact_reacquire_bounds = self._contact_reacquire_bounds(frame.shape, self.ball_center) if self.ball_center else None
        lower_contact_launch_context = None
        ground_bounce_context = None
        if serve_contact_grace:
            self._serve_contact_grace_frames -= 1
        if rally_contact_grace:
            self._rally_contact_grace_frames -= 1
        if ground_bounce_grace:
            self._ground_bounce_grace_frames -= 1
        if contact_recovery_active:
            self._contact_recovery_frames -= 1

        # Store the last search position for when ball is lost
        if not hasattr(self, 'last_search_position'):
            self.last_search_position = None
        
        early_frames = self.frame_count <= (self.start_frame + 10)

        # Check if ball was near edge in previous frame
        self.edge_wait = False
        upper_exit_wait = getattr(self, '_upper_exit_wait_frames', 0) > 0

        if self.ball_center:
            x_prev, y_prev = self.ball_center
            self.last_search_position = (x_prev, y_prev)  # Remember this position
            edge_margin = 50  # pixels from edge (reduced from 100 to avoid false triggers)

            if upper_exit_wait:
                self._upper_exit_wait_frames -= 1
                print(f"\n  DEBUG: [UPPER-EXIT WAIT] holding near top edge from ({x_prev},{y_prev}), "
                      f"frames_left={self._upper_exit_wait_frames}")
                x, y = x_prev, min(40, max(10, y_prev))
                search_radius = max(220, self.max_ball_speed * 2)
                self.edge_wait = True
            # Check if ball went off top edge
            elif y_prev < edge_margin:
                print(f"\n  DEBUG: Ball near TOP edge (y={y_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at top edge for ball to return...")
                # Search along top edge
                x, y = x_prev, 10
                search_radius = 200  # Very wide search along edge for fast balls
                self.edge_wait = True
            # Check if ball went off bottom edge
            elif y_prev > frame_height - edge_margin:
                print(f"\n  DEBUG: Ball near BOTTOM edge (y={y_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at bottom edge for ball to return...")
                x, y = x_prev, frame_height - 10
                search_radius = 200
                self.edge_wait = True
            # Check if ball went off left edge
            elif x_prev < edge_margin:
                print(f"\n  DEBUG: Ball near LEFT edge (x={x_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at left edge for ball to return...")
                x, y = 10, y_prev
                search_radius = 200
                self.edge_wait = True
            # Check if ball went off right edge
            elif x_prev > frame_width - edge_margin:
                print(f"\n  DEBUG: Ball near RIGHT edge (x={x_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at right edge for ball to return...")
                x, y = frame_width - 10, y_prev
                search_radius = 200
                self.edge_wait = True
            else:
                # Normal tracking - velocity-adaptive radius keeps search close to last ball
                # position to avoid jumping to nearby false positives (racket, player arm).
                # Radius grows with recent ball speed so fast-moving served balls are still caught.
                x, y = x_prev, y_prev
                if self.ball_velocity_history:
                    _recent_vel = max(self.ball_velocity_history[-min(3, len(self.ball_velocity_history)):])
                else:
                    _recent_vel = 0
                search_radius = max(80, min(int(_recent_vel * 1.5), 300))
                # After a motion-based re-acquisition the serve hit can instantly fling
                # the ball 200-400 px in one frame.  Use a wide window for the next
                # several frames so the tracker keeps up with this sudden acceleration.
                # Compute _in_post_reacq BEFORE decrementing so all downstream
                # checks (scoring, bypass guards) see the same True/False value
                # for this frame — otherwise the last frame of the window (counter=1)
                # would see True here but False in the scoring section.
                _in_post_reacq = getattr(self, '_post_reacq_frames', 0) > 0
                if _in_post_reacq:
                    self._post_reacq_frames -= 1
                    search_radius = max(search_radius, self.max_ball_speed)
                    print(f"  DEBUG: [POST-REACQ] wide search radius={search_radius}px, frames_left={self._post_reacq_frames}")
                if rally_contact_grace:
                    search_radius = max(search_radius, self.max_ball_speed)
                    print(f"  DEBUG: [RALLY-CONTACT] wide search radius={search_radius}px, frames_left={self._rally_contact_grace_frames}")
                if ground_bounce_grace:
                    search_radius = max(search_radius, max(self.max_ball_speed, 110))
                    print(f"  DEBUG: [GROUND-BOUNCE] wide search radius={search_radius}px, frames_left={self._ground_bounce_grace_frames}")
                if not allow_inactive and not serve_direction_search and not rally_contact_grace and not ground_bounce_grace:
                    lower_contact_launch_context = self._get_lower_contact_launch_context(frame.shape)
                    if lower_contact_launch_context is not None:
                        search_radius = max(search_radius, self.max_ball_speed)
                        print(f"  DEBUG: [LOWER-HIT PREP] wide search radius={search_radius}px "
                              f"expected={lower_contact_launch_context['expected']}")
                    else:
                        ground_bounce_context = self._get_ground_bounce_context(frame.shape)
                        if ground_bounce_context is not None:
                            search_radius = max(search_radius, max(self.max_ball_speed, 110))
                            print(f"  DEBUG: [GROUND-BOUNCE PREP] wide search radius={search_radius}px "
                                  f"expected={ground_bounce_context['expected']}")
                # Stuck expansion radius: during slow-phase (toss/occlusion, vel<50px/f)
                # keep search reasonably tight so we don't grab far player/arm blobs.
                # Once ball was moving fast, expand aggressively to re-catch it after hit.
                _slow_phase = _recent_vel < 50
                _stuck_expand = 300 if _slow_phase else max(300, 150 + self.stuck_frame_count * 50)
                # When ball is lost for several frames (e.g. player occlusion),
                # try motion-based re-acquisition using frame differencing
                if self.stuck_frame_count in (5, 10):
                    vel_h = [round(v,1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
                    print(f"[STUCK_MILESTONE] f{self.frame_count}: stuck={self.stuck_frame_count} pos={self.ball_center} vel_hist={vel_h} slow_phase={_slow_phase}")
                # Save the previous-frame grayscale BEFORE the reacquire overwrites it.
                # Used later in the FFS candidate loop to verify motion (moving ball vs
                # static background/logo that just happens to have ball colour).
                _ffs_prev_gray = self._prev_frame_gray if hasattr(self, '_prev_frame_gray') else None
                if self.stuck_frame_count >= 5 and hasattr(self, '_prev_frame_gray') and self._prev_frame_gray is not None:
                    reacq_pos = self._reacquire_ball_by_motion(frame)
                    if reacq_pos is not None:
                        print(f"Frame {self.frame_count}: [RE-ACQUIRED] Ball found at {reacq_pos} after {self.stuck_frame_count} stuck frames (motion-based)")
                        self.ball_velocity_history = []
                        self.last_motion = None
                        self.last_direction = None
                        self.direction_change_streak = 0
                        self.stuck_frame_count = 0
                        self._recent_max_ball_size = 0
                        self.ball_center = reacq_pos
                        self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # After re-acquisition the serve hit may cause a large jump in the
                        # very next frames.  Arm a wide-search window so the tracker can
                        # follow the ball even when it suddenly accelerates 200-400 px/frame.
                        # 3 frames covers f49/f50/f51 (toss fall + serve contact) without
                        # leaving the wide-search window open long enough to latch onto a
                        # court-1 false positive and block STUCK_TIMEOUT.
                        self._post_reacq_frames = 3
                        # Freeze the ball-size reference at re-acquisition time.  In the
                        # post-reacq window ball_size can drift toward small noise blobs
                        # selected near the ground; keeping the original large-blob reference
                        # keeps the size-primary scoring anchored to the real ball.
                        self._reacq_ref_size = max(self.ball_size, 50) if self.ball_size else 100
                        return self.ball_center
                    else:
                        # Update prev frame for next attempt (consecutive comparison)
                        self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Reacquire found nothing — expand the HSV fallback radius,
                        # but keep it bounded to avoid grabbing distant false positives
                        # (player bodies, logos) that just happen to match ball colour.
                        search_radius = max(search_radius, _stuck_expand)
                elif self.stuck_frame_count >= 1:
                    # Expand search radius so a sudden serve acceleration (ball jumps
                    # 150-200px after racket contact) is caught quickly.
                    # The max_ball_speed cap applied below keeps court-1 unaffected.
                    search_radius = max(search_radius, _stuck_expand)
                if not allow_inactive and self._serve_direction_search_active():
                    serve_direction_search = True
                    search_radius = max(search_radius, self.max_ball_speed)
        # First, check if ball stopped - if so, search from initial position
        elif hasattr(self, 'ball_stopped') and self.ball_stopped and hasattr(self, 'initial_ball_position'):
            print(f"\n  DEBUG: Ball stopped! Searching from initial position {self.initial_ball_position}")
            x, y = self.initial_ball_position
            search_radius = 80  # Larger radius for initial search
        else:
            search_frame = frame
            x1, y1 = 0, 0
            # When tracking is inactive (serve scan), give a hint region around serve area if defined
            if allow_inactive and hasattr(self, 'serve_area_x_min') and hasattr(self, 'serve_area_x_max'):
                # Center search at middle of serve area to reduce false positives
                x = (self.serve_area_x_min + self.serve_area_x_max) // 2
                y = (self.serve_area_y_min + self.serve_area_y_max) // 2
                search_radius = max(self.serve_area_x_max - self.serve_area_x_min,
                                    self.serve_area_y_max - self.serve_area_y_min) // 2
                x1 = max(0, x - search_radius)
                y1 = max(0, y - search_radius)
                x2 = min(frame.shape[1], x + search_radius)
                y2 = min(frame.shape[0], y + search_radius)
                search_frame = frame[y1:y2, x1:x2]
                print(f"\n  DEBUG: Serve-scan hint search around ({x},{y}) radius {search_radius}px region ({x1},{y1})-({x2},{y2})")
            else:
                print(f"\n  DEBUG: No previous ball position, searching entire frame")
        
        predicted_point = None
        # Predict next position using last known motion if available
        if self.ball_center and hasattr(self, 'last_delta') and self.last_delta is not None:
            dx, dy = self.last_delta
            predicted_point = (self.ball_center[0] + int(dx), self.ball_center[1] + int(dy))

        # Search in region around the search position
        if self.ball_center or (hasattr(self, 'initial_ball_position') and self.initial_ball_position):
            if early_frames and 'search_radius' in locals():
                search_radius = max(search_radius, 250)  # wider window for first few frames
            if not allow_inactive and not getattr(self, 'edge_wait', False) and self.stuck_frame_count < 3:
                search_radius = min(search_radius, self.max_ball_speed)
            if serve_direction_search:
                x1, y1, x2, y2 = self._build_serve_direction_region(x, y, frame.shape, search_radius)
            else:
                x1 = max(0, x - search_radius)
                y1 = max(0, y - search_radius)
                x2 = min(frame.shape[1], x + search_radius)
                y2 = min(frame.shape[0], y + search_radius)
            
            search_frame = frame[y1:y2, x1:x2]
            # Determine which HSV config will be used
            _, _, hsv_mode_check = self.select_hsv_for_position(y)
            if serve_direction_search:
                print(f"\n  DEBUG: [SERVE-DIR] Searching {self.serve_direction_label()} from ({x},{y}) "
                      f"with radius {search_radius}px, region: ({x1},{y1})-({x2},{y2}), HSV mode: {hsv_mode_check}")
            else:
                print(f"\n  DEBUG: Searching at ({x},{y}) with radius {search_radius}px, region: ({x1},{y1})-({x2},{y2}), HSV mode: {hsv_mode_check}")
        
        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        
        # Check if we should search with BOTH HSV filters
        # This happens when ball is approaching the net area (within 100px margin)
        is_dual_mode = self.hsv_regular is not None and self._behind_net_enabled()
        net_approach_margin = 100  # pixels before/after net area to check both filters
        
        should_check_both = False
        if is_dual_mode and hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            # Check if ball is approaching, in, or leaving net area
            if (self.net_area_y_min - net_approach_margin) <= y <= (self.net_area_y_max + net_approach_margin):
                should_check_both = True
        if is_dual_mode and contact_recovery_active:
            should_check_both = True
            print(f"  DEBUG: Contact recovery active - checking BOTH HSV filters")

        upper_wall_search_context = (
            not allow_inactive and
            self.ball_center is not None and
            self.ground_bounce_count > 0 and
            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 20 and
            y < max(145, int(frame.shape[0] * 0.08)) and
            (
                self.stuck_frame_count >= 2 or
                getattr(self, '_upper_exit_wait_frames', 0) > 0
            )
        )
        
        if should_check_both:
            # Search with BOTH filters and combine results
            hsv_mode = "dual_net"
            print(f"  DEBUG: Ball near net area (Y={y}, net Y={self.net_area_y_min}-{self.net_area_y_max})")
            print(f"  DEBUG: Checking BOTH HSV filters to find best match")
            
            # Get contours from REGULAR COURT filter
            mask_reg = cv2.inRange(hsv_frame, self.hsv_regular['lower'], self.hsv_regular['upper'])
            kernel = np.ones((2, 2), np.uint8)
            mask_reg = cv2.morphologyEx(mask_reg, cv2.MORPH_OPEN, kernel)
            mask_reg = cv2.morphologyEx(mask_reg, cv2.MORPH_CLOSE, kernel)
            contours_reg, _ = cv2.findContours(mask_reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get contours from BEHIND NET filter
            mask_net = cv2.inRange(hsv_frame, self.hsv_behind_net['lower'], self.hsv_behind_net['upper'])
            mask_net = cv2.morphologyEx(mask_net, cv2.MORPH_OPEN, kernel)
            mask_net = cv2.morphologyEx(mask_net, cv2.MORPH_CLOSE, kernel)
            contours_net, _ = cv2.findContours(mask_net, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Also check alt2 HSV filter (H=30-75) for balls with low Hue near net
            contours_alt2 = []
            if hasattr(self, 'alt2_hsv_lower') and self.alt2_hsv_lower is not None:
                mask_alt2 = cv2.inRange(hsv_frame, self.alt2_hsv_lower, self.alt2_hsv_upper)
                mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_OPEN, kernel)
                mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_CLOSE, kernel)
                contours_alt2, _ = cv2.findContours(mask_alt2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours_alt4 = []
            contours_alt5 = []
            contours_alt6 = []
            if upper_wall_search_context and self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
                mask_alt4 = cv2.inRange(hsv_frame, self.alt4_hsv_lower, self.alt4_hsv_upper)
                mask_alt4 = cv2.morphologyEx(mask_alt4, cv2.MORPH_OPEN, kernel)
                mask_alt4 = cv2.morphologyEx(mask_alt4, cv2.MORPH_CLOSE, kernel)
                contours_alt4, _ = cv2.findContours(mask_alt4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if upper_wall_search_context and self.alt5_hsv_lower is not None and self.alt5_hsv_upper is not None:
                mask_alt5 = cv2.inRange(hsv_frame, self.alt5_hsv_lower, self.alt5_hsv_upper)
                mask_alt5 = cv2.morphologyEx(mask_alt5, cv2.MORPH_OPEN, kernel)
                mask_alt5 = cv2.morphologyEx(mask_alt5, cv2.MORPH_CLOSE, kernel)
                contours_alt5, _ = cv2.findContours(mask_alt5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if upper_wall_search_context and self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
                mask_alt6 = cv2.inRange(hsv_frame, self.alt6_hsv_lower, self.alt6_hsv_upper)
                mask_alt6 = cv2.morphologyEx(mask_alt6, cv2.MORPH_OPEN, kernel)
                mask_alt6 = cv2.morphologyEx(mask_alt6, cv2.MORPH_CLOSE, kernel)
                contours_alt6, _ = cv2.findContours(mask_alt6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Tag each contour with its source
            contours = []
            for c in contours_reg:
                contours.append(('regular', c))
            for c in contours_net:
                contours.append(('behind_net', c))
            for c in contours_alt2:
                contours.append(('alt2', c))
            for c in contours_alt4:
                contours.append(('alt4', c))
            for c in contours_alt5:
                contours.append(('alt5', c))
            for c in contours_alt6:
                contours.append(('alt6', c))

            print(f"  DEBUG: Found {len(contours_reg)} regular + {len(contours_net)} behind_net + {len(contours_alt2)} alt2 + {len(contours_alt4)} alt4 + {len(contours_alt5)} alt5 + {len(contours_alt6)} alt6 = {len(contours)} total contours")
        else:
            # Normal single HSV filter search
            hsv_lower_use, hsv_upper_use, hsv_mode = self.select_hsv_for_position(y)
            if allow_inactive and hasattr(self, 'serve_area_x_min'):
                # Use full configured HSV for serve scan (do not cap H max)
                hsv_lower_use = self.hsv_lower
                hsv_upper_use = self.hsv_upper
                hsv_mode = "serve_area"
            use_alt_first = (self.frame_count == 127)
            primary_lower = self.alt_hsv_lower if use_alt_first and self.alt_hsv_lower is not None else hsv_lower_use
            primary_upper = self.alt_hsv_upper if use_alt_first and self.alt_hsv_upper is not None else hsv_upper_use
            # Use the narrower alt filter (capped H max) alongside primary to preserve pre-focus behavior
            alt_lower = hsv_lower_use if use_alt_first and self.alt_hsv_lower is not None else self.alt_hsv_lower
            alt_upper = hsv_upper_use if use_alt_first and self.alt_hsv_upper is not None else self.alt_hsv_upper
            # Apply primary HSV filter
            mask_primary = cv2.inRange(hsv_frame, primary_lower, primary_upper)
            # Apply alternate HSV filter (extended H) if defined
            mask_alt = None
            if alt_upper is not None and (alt_upper[0] != primary_upper[0] or use_alt_first):
                mask_alt = cv2.inRange(hsv_frame, alt_lower, alt_upper)
            
            # Clean up
            kernel = np.ones((2, 2), np.uint8)
            mask_primary = cv2.morphologyEx(mask_primary, cv2.MORPH_OPEN, kernel)
            mask_primary = cv2.morphologyEx(mask_primary, cv2.MORPH_CLOSE, kernel)
            if mask_alt is not None:
                mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_OPEN, kernel)
                mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_CLOSE, kernel)
            mask_alt4 = None
            mask_alt5 = None
            mask_alt6 = None
            if upper_wall_search_context and self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
                mask_alt4 = cv2.inRange(hsv_frame, self.alt4_hsv_lower, self.alt4_hsv_upper)
                mask_alt4 = cv2.morphologyEx(mask_alt4, cv2.MORPH_OPEN, kernel)
                mask_alt4 = cv2.morphologyEx(mask_alt4, cv2.MORPH_CLOSE, kernel)
            if upper_wall_search_context and self.alt5_hsv_lower is not None and self.alt5_hsv_upper is not None:
                mask_alt5 = cv2.inRange(hsv_frame, self.alt5_hsv_lower, self.alt5_hsv_upper)
                mask_alt5 = cv2.morphologyEx(mask_alt5, cv2.MORPH_OPEN, kernel)
                mask_alt5 = cv2.morphologyEx(mask_alt5, cv2.MORPH_CLOSE, kernel)
            if upper_wall_search_context and self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
                mask_alt6 = cv2.inRange(hsv_frame, self.alt6_hsv_lower, self.alt6_hsv_upper)
                mask_alt6 = cv2.morphologyEx(mask_alt6, cv2.MORPH_OPEN, kernel)
                mask_alt6 = cv2.morphologyEx(mask_alt6, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours_raw, _ = cv2.findContours(mask_primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [('primary', c) for c in contours_raw]
            if mask_alt is not None:
                contours_alt, _ = cv2.findContours(mask_alt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend([('alt', c) for c in contours_alt])
            if mask_alt4 is not None:
                contours_alt4, _ = cv2.findContours(mask_alt4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend([('alt4', c) for c in contours_alt4])
            if mask_alt5 is not None:
                contours_alt5, _ = cv2.findContours(mask_alt5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend([('alt5', c) for c in contours_alt5])
            if mask_alt6 is not None:
                contours_alt6, _ = cv2.findContours(mask_alt6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend([('alt6', c) for c in contours_alt6])
            
            print(f"  DEBUG: Found {len(contours)} total contours in search region (mode: {hsv_mode} incl alt)")
        
        if not contours:
            print("  DEBUG: [PROBLEM] No contours found in search region!")
            if 'hsv_lower_use' in dir() and 'hsv_upper_use' in dir():
                print(f"  DEBUG: HSV filter range ({hsv_mode}): H={hsv_lower_use[0]}-{hsv_upper_use[0]}, S={hsv_lower_use[1]}-{hsv_upper_use[1]}, V={hsv_lower_use[2]}-{hsv_upper_use[2]}")
            else:
                print(f"  DEBUG: HSV filter mode: {hsv_mode}")
            print(f"  DEBUG: Search region: ({x1},{y1})-({x2},{y2}), size: {x2-x1}x{y2-y1}px")
            if self.ball_center:
                print(f"  DEBUG: Previous ball position: {self.ball_center}")
                print(f"  DEBUG: KEEPING marker at last known position: {self.ball_center}")
                print(f"[BALL_LOST] f{self.frame_count}: no contours found, keeping pos={self.ball_center} stuck={self.stuck_frame_count}")
            print(f"  DEBUG: REASON: Ball may have:")
            print(f"  DEBUG:   - Gone off screen (check edge detection)")
            print(f"  DEBUG:   - Changed color/lighting dramatically")
            print(f"  DEBUG:   - Moved faster than {search_radius}px/frame")
            print(f"  DEBUG:   - Be occluded by player/net")
            print(f"  DEBUG: Will continue searching in next frame at same position...")

            if upper_exit_wait and self.ball_center:
                self.stuck_frame_count += 1
                print(f"Frame {self.frame_count}: [UPPER-EXIT WAIT] no top-edge reentry, holding {self.ball_center}")
                return self.ball_center

            if (not allow_inactive and self.ball_center and not getattr(self, 'edge_wait', False)
                    and self.stuck_frame_count < 5
                    and hasattr(self, '_prev_frame_gray') and self._prev_frame_gray is not None):
                saved_prev_gray = self._prev_frame_gray.copy()
                reacq_pos = self._reacquire_ball_by_motion(frame)
                if reacq_pos is not None:
                    self.ball_center = reacq_pos
                    self.ball_size = None
                    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    self.ball_hsv = hsv_full[reacq_pos[1], reacq_pos[0]]
                    self.ball_velocity_history = []
                    self.last_motion = None
                    self.last_direction = None
                    self.direction_change_streak = 0
                    self.stuck_frame_count = 0
                    self._recent_max_ball_size = 0
                    self._post_reacq_frames = max(getattr(self, '_post_reacq_frames', 0), 3)
                    print(f"Frame {self.frame_count}: [EARLY MOTION REACQ] Ball at {reacq_pos}")
                    return self.ball_center
                self._prev_frame_gray = saved_prev_gray

            if not allow_inactive:
                upper_post_bounce_recover = (
                    contact_reacquire_bounds is not None and
                    self.last_motion is not None and
                    self.last_motion.get('dy', 0.0) < -2.0 and
                    self.ground_bounce_count > 0 and
                    self.stuck_frame_count > 0
                )
                contact_reference = self.ball_center if upper_post_bounce_recover else (predicted_point or self.ball_center)
                contact_recover = self._recover_contact_phase_ball(
                    frame, contact_reference, frame_gray,
                    predicted_point=None if upper_post_bounce_recover else predicted_point,
                    max_prev_speed=45.0 if upper_post_bounce_recover else 15.0,
                    upper_stuck_mode=upper_post_bounce_recover,
                )
                if contact_recover is not None:
                    self.ball_center = contact_recover['pos']
                    self.ball_hsv = contact_recover['hsv']
                    self.ball_size = contact_recover['area']
                    if contact_reference is not None:
                        new_dx = self.ball_center[0] - contact_reference[0]
                        new_dy = self.ball_center[1] - contact_reference[1]
                        new_velocity = math.hypot(new_dx, new_dy)
                        new_direction = math.degrees(math.atan2(new_dy, new_dx)) if new_velocity > 0 else 0.0
                        self.last_delta = (new_dx, new_dy)
                        self.last_motion = {
                            'distance': new_velocity,
                            'dx': new_dx,
                            'dy': new_dy,
                            'direction_deg': new_direction,
                        }
                        self.ball_velocity_history = [new_velocity]
                        self.last_direction = new_direction
                    else:
                        self.ball_velocity_history = []
                        self.last_motion = None
                        self.last_direction = None
                    self.direction_change_streak = 0
                    self.stuck_frame_count = 0
                    self._recent_max_ball_size = 0
                    self._post_reacq_frames = max(getattr(self, '_post_reacq_frames', 0), 3)
                    self._contact_recovery_frames = max(getattr(self, '_contact_recovery_frames', 0), 4)
                    return self.ball_center

            # Fallback: try alternative HSV if primary found nothing
            if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                if getattr(self, 'edge_wait', False) and self.last_motion:
                    lm_dx = self.last_motion['dx']
                    lm_dy = self.last_motion['dy']
                    # Only suppress alt recovery if motion is pointing out of frame
                    if (self.ball_center and
                        ((lm_dx < 0 and self.ball_center[0] < 10) or
                         (lm_dx > 0 and self.ball_center[0] > frame.shape[1] - 10) or
                         (lm_dy < 0 and self.ball_center[1] < 10) or
                         (lm_dy > 0 and self.ball_center[1] > frame.shape[0] - 10))):
                        # Ball is outside frame moving away — count as stuck so
                        # STUCK_TIMEOUT can eventually fire and release the tracker.
                        self.stuck_frame_count += 1
                        return self.ball_center
                retrack = self.retrack_with_alt_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    frame_gray=frame_gray
                )
                if retrack is not None:
                    new_pos = retrack['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack['hsv']
                    self.ball_size = retrack['area']
                    self.using_alt_hsv = True
                    self.using_alt2_hsv = False
                    if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                        self.hsv_lower = self.alt_focus_hsv_lower
                        self.hsv_upper = self.alt_focus_hsv_upper
                    print(f"Frame {self.frame_count}: [ALT HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center
            # Fallback: try alternative 2 HSV if alt1 failed
            if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
                retrack2 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    frame_gray=frame_gray
                )
                if retrack2 is not None:
                    new_pos = retrack2['pos']
                    prev_pos = self.ball_center
                    self.ball_center = new_pos
                    self.ball_hsv = retrack2['hsv']
                    self.ball_size = retrack2['area']
                    self.using_alt_hsv = False
                    self.using_alt2_hsv = True
                    self.using_alt3_hsv = False
                    self.hsv_lower = self.alt2_hsv_lower
                    self.hsv_upper = self.alt2_hsv_upper
                    print(f"Frame {self.frame_count}: [ALT2 HSV RECOVER] Ball at {new_pos}")
                    if self.last_motion and prev_pos and self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
                        lm_dist = self.last_motion['distance']
                        mv_dx = new_pos[0] - prev_pos[0]
                        mv_dy = new_pos[1] - prev_pos[1]
                        mv_dist = math.hypot(mv_dx, mv_dy)
                        speed_ratio = (mv_dist / lm_dist) if lm_dist else 1.0
                        angle_jump = 0.0
                        if self.last_direction is not None:
                            direction_deg = math.degrees(math.atan2(mv_dy, mv_dx))
                            delta = abs(direction_deg - self.last_direction) % 360
                            angle_jump = min(delta, 360 - delta)
                        if angle_jump > 45 or speed_ratio > 1.8 or speed_ratio < 0.6:
                            retrack3 = self.retrack_with_alt2_hsv(
                                search_frame, x1, y1, prev_pos, predicted_point, self.ball_size, allow_inactive,
                                lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray
                            )
                            if retrack3 is not None:
                                new_pos = retrack3['pos']
                                self.ball_center = new_pos
                                self.ball_hsv = retrack3['hsv']
                                self.ball_size = retrack3['area']
                                self.using_alt2_hsv = False
                                self.using_alt3_hsv = True
                                self.hsv_lower = self.alt3_hsv_lower
                                self.hsv_upper = self.alt3_hsv_upper
                                print(f"Frame {self.frame_count}: [ALT3 HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center
            if self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
                retrack3 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray
                )
                if retrack3 is not None:
                    new_pos = retrack3['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack3['hsv']
                    self.ball_size = retrack3['area']
                    self.using_alt_hsv = False
                    self.using_alt2_hsv = False
                    self.using_alt3_hsv = True
                    self.hsv_lower = self.alt3_hsv_lower
                    self.hsv_upper = self.alt3_hsv_upper
                    print(f"Frame {self.frame_count}: [ALT3 HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center

            if self._should_try_upper_exit_low_s(frame, allow_inactive):
                upper_exit_low_s = self._retrack_with_upper_exit_low_s(
                    search_frame, x1, y1, predicted_point, frame_gray=frame_gray
                )
                if upper_exit_low_s is not None:
                    new_pos = upper_exit_low_s['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = upper_exit_low_s['hsv']
                    self.ball_size = upper_exit_low_s['area']
                    self.using_alt_hsv = False
                    self.using_alt2_hsv = False
                    self.using_alt3_hsv = False
                    self.hsv_lower = upper_exit_low_s['lower']
                    self.hsv_upper = upper_exit_low_s['upper']
                    self.stuck_frame_count = 0
                    print(f"Frame {self.frame_count}: [UPPER EXIT LOW-S RECOVER] Ball at {new_pos} via {upper_exit_low_s['label']}")
                    return self.ball_center

            if (not allow_inactive and self.ball_center and self.last_seen_frame == (self.frame_count - 1)
                    and not getattr(self, 'auto_play', False)):
                predicted_point = predicted_point or self.get_predicted_point()
                if predicted_point is not None:
                    print("  DEBUG: Ball lost. Opening HSV debug windows at predicted position...")
                    self.open_predicted_hsv_debug_all(frame, predicted_point, self.frame_count)
                    self.pause_requested = True
            
            # Keep the ball_center at last position instead of losing it
            # Increment stuck counter since no contour was found
            self.stuck_frame_count += 1
            return self.ball_center
        
        # Find the best ball candidate using weighted score (distance + size similarity)
        best_contour = None
        best_score = float('inf')
        best_source = None
        candidates = []
        upper_exit_transition_context = (
            not allow_inactive and
            self.ball_center is not None and
            contact_reacquire_bounds is not None and
            self.ground_bounce_count > 0 and
            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 16 and
            self.ball_size is not None and
            self.ball_size >= 35 and
            self.ball_center[1] < max(220, int(frame.shape[0] * 0.12))
        )
        for i, (source, contour) in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Reject huge background regions first.
            # Upper limit scales with serve_ball_size_max so near-end large balls aren't
            # incorrectly classified as background (e.g. court 2 ball area ~280-360px²).
            bg_threshold = max(500, int(self.serve_ball_size_max * 1.5))
            if area > bg_threshold:
                if i < 3:  # Only print first few to avoid spam
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (background region)")
                continue

            # Size filter: tighter in inactive serve scan, looser when tracking.
            # Both limits are court-configurable via serve_ball_size_min/max so that
            # near-end servers (large ball) and far-end servers (tiny ball) both work.
            ball_size_max_tracking = max(150, self.serve_ball_size_max)
            if allow_inactive:
                if area < self.serve_ball_size_min or area > self.serve_ball_size_max:
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (serve scan outside {self.serve_ball_size_min}-{self.serve_ball_size_max})")
                    continue
            else:
                if area < 1 or area > ball_size_max_tracking:
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (outside 1-{ball_size_max_tracking})")
                    continue

            # Reject extremely small candidates when we already have a valid previous ball size
            # to avoid picking spark/noise/racket edges with size 1-3px as the ball.
            if (not upper_exit_transition_context and
                    self.ball_size and self.ball_size > 40 and
                    area < max(5, int(self.ball_size * 0.08))):
                print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (too small relative to previous ball size {self.ball_size:.1f}px)")
                continue

            # Additional reject: belt-and-suspenders guard in case allow_inactive fell through
            if area < 1 or area > ball_size_max_tracking:
                print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (outside 1-{ball_size_max_tracking})")
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            # Calculate position in original frame coordinates
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            
            # Reject candidates outside serve area when inactive
            if allow_inactive and hasattr(self, 'serve_area_x_min'):
                if not (self.serve_area_x_min <= cx <= self.serve_area_x_max and
                        self.serve_area_y_min <= cy <= self.serve_area_y_max):
                    print(f"  DEBUG: Contour {i} REJECTED - pos=({cx},{cy}) outside serve area")
                    continue

            # Calculate distance from previous position
            if self.ball_center:
                distance = np.sqrt((cx - self.ball_center[0])**2 + (cy - self.ball_center[1])**2)
            else:
                distance = 0
            
            # Calculate size difference (penalty for very different sizes)
            if self.ball_size and self.ball_size > 0:
                size_diff = abs(area - self.ball_size)
                size_ratio = size_diff / self.ball_size
            else:
                size_ratio = 0

            if serve_direction_search and self.ball_center:
                signed_dx = self._signed_serve_dx(cx - self.ball_center[0])
                if signed_dx < self.serve_contact_min_dx:
                    print(f"  DEBUG: Contour {i} REJECTED - serve-direction dx={signed_dx:.1f}px < {self.serve_contact_min_dx}")
                    continue
                if self.serve_direction_dy != 0:
                    signed_dy = self._signed_serve_dy(cy - self.ball_center[1])
                    if signed_dy < self.serve_contact_min_dy:
                        print(f"  DEBUG: Contour {i} REJECTED - serve-direction dy={signed_dy:.1f}px < {self.serve_contact_min_dy}")
                        continue

            if not allow_inactive:
                ignored_entry = self._find_ignored_tracking_position((cx, cy))
                if ignored_entry is not None:
                    print(f"  DEBUG: Contour {i} REJECTED - learned hotspot at ({cx},{cy}) "
                          f"reason={ignored_entry.get('reason', 'n/a')}")
                    continue
                if contact_recovery_active and self.ball_center:
                    contact_band = max(120, int(frame_height * 0.06))
                    if abs(cy - self.ball_center[1]) > contact_band:
                        print(f"  DEBUG: Contour {i} REJECTED - contact-recovery band dy={cy - self.ball_center[1]}")
                        continue
                    contact_lateral_cap = max(180, int(frame_width * 0.05))
                    if abs(cx - self.ball_center[0]) > contact_lateral_cap:
                        print(f"  DEBUG: Contour {i} REJECTED - contact-recovery lateral dx={cx - self.ball_center[0]}")
                        continue
                    if predicted_point is not None:
                        predicted_distance = np.sqrt((cx - predicted_point[0])**2 + (cy - predicted_point[1])**2)
                        predicted_cap = max(140, int(frame_width * 0.04))
                        if predicted_distance > predicted_cap:
                            print(f"  DEBUG: Contour {i} REJECTED - contact-recovery predicted-dist={predicted_distance:.1f}px > {predicted_cap}")
                            continue
             
            # Weighted score: distance + size penalty
            # Prefer candidates with similar size to previous ball
            # Distance is primary, but size consistency matters for fast-moving balls
            full_frame_scan = self.stuck_frame_count >= 5
            if full_frame_scan:
                # In full-frame scan mode: prioritize size match over distance.
                # Ball bounced off player so direction is unknown.
                # Reject tiny noise when ball was recently visible at larger size.
                if self.ball_size and self.ball_size >= 10 and area < 3:
                    continue
                # Reject contours near stuck position (noise at old location) —
                # the ball has moved away; anything still within 500 px of the
                # last-known position is almost certainly a static artefact or
                # stuck-phase occlusion noise.
                if self.ball_center:
                    dist_from_stuck = np.sqrt((cx - self.ball_center[0])**2 + (cy - self.ball_center[1])**2)
                    if dist_from_stuck < 500:
                        continue
                # When stuck ≥ 10 frames ball_size may be stale (set at close-range
                # toss, now ball is far away).  Cap the reference at 100 px and add
                # motion-verification to distinguish the moving ball from static logos
                # or court markings that happen to pass the colour filter.
                if self.stuck_frame_count >= 10:
                    _ffs_ref_size = min(self.ball_size if self.ball_size else 100, 100)
                    _ffs_size_ratio = abs(area - _ffs_ref_size) / max(_ffs_ref_size, 1)
                    score = _ffs_size_ratio * 100
                    score += distance * 0.001  # tiebreaker only
                    # Motion verification: a static blob looks the same in consecutive
                    # frames; a flying ball changes the pixel value at its new location.
                    if (_ffs_prev_gray is not None
                            and hasattr(self, '_prev_frame_gray')
                            and self._prev_frame_gray is not None
                            and 0 <= cy < _ffs_prev_gray.shape[0]
                            and 0 <= cx < _ffs_prev_gray.shape[1]):
                        _prev_lum = int(_ffs_prev_gray[cy, cx])
                        _curr_lum = int(self._prev_frame_gray[cy, cx])
                        if abs(_curr_lum - _prev_lum) < 20:
                            score += 2000  # static blob — not the flying ball
                else:
                    # stuck 5–9: use the same scoring as the committed baseline so that
                    # court-1 serve detection (which relies on this path to stop tracking
                    # and enter serve-scan mode) is not disturbed.
                    score = size_ratio * 100  # size match is primary
                    score += distance * 0.1   # very mild distance preference
            elif _in_post_reacq:
                # Immediately after motion-based re-acquisition the serve hit can
                # dramatically change direction and speed.  Prioritise SIZE over
                # DISTANCE so we follow the actual ball rather than a nearby artifact.
                # Use the frozen _reacq_ref_size so ball_size drift in f49-f50 (near ground,
                # small blobs) doesn't corrupt the reference for f51+ (serve contact, large blob).
                _rr_size = getattr(self, '_reacq_ref_size', None) or self.ball_size or 100
                _rr_size = max(_rr_size, 50)
                _post_size_ratio = abs(area - _rr_size) / _rr_size
                score = _post_size_ratio * 100 + distance * 0.1
            elif serve_contact_grace and self.stuck_frame_count >= 1:
                # During serve contact the ball can jump hundreds of pixels and reverse
                # direction instantly, so distance to the previous toss location is a
                # weak signal. Prefer contours that still look like the same ball size.
                _contact_ref_size = max(self.ball_size or area, 50)
                _contact_size_ratio = abs(area - _contact_ref_size) / _contact_ref_size
                score = _contact_size_ratio * 100 + distance * 0.1
            elif ground_bounce_context is not None:
                origin_x, origin_y = ground_bounce_context['origin']
                expected_x, expected_y = ground_bounce_context['expected']
                bounce_dist = math.hypot(cx - origin_x, cy - origin_y)
                upward_progress = origin_y - cy
                expected_distance = math.hypot(cx - expected_x, cy - expected_y)
                _bounce_ref_size = ground_bounce_context['ref_size']
                _bounce_size_ratio = abs(area - _bounce_ref_size) / max(_bounce_ref_size, 1)
                score = expected_distance * 0.9
                score += _bounce_size_ratio * 22.0
                if bounce_dist < ground_bounce_context['min_launch_dist']:
                    score += (ground_bounce_context['min_launch_dist'] - bounce_dist) * 4.0
                if upward_progress < ground_bounce_context['min_upward']:
                    score += (ground_bounce_context['min_upward'] - upward_progress) * 3.5
                if bounce_dist > ground_bounce_context['max_launch_dist']:
                    score += (bounce_dist - ground_bounce_context['max_launch_dist']) * 1.8
            elif lower_contact_launch_context is not None:
                origin_x, origin_y = lower_contact_launch_context['origin']
                expected_x, expected_y = lower_contact_launch_context['expected']
                launch_dist = math.hypot(cx - origin_x, cy - origin_y)
                upward_progress = origin_y - cy
                expected_distance = math.hypot(cx - expected_x, cy - expected_y)
                _launch_ref_size = lower_contact_launch_context['ref_size']
                _launch_size_ratio = abs(area - _launch_ref_size) / max(_launch_ref_size, 1)
                score = expected_distance * 1.1
                score += _launch_size_ratio * 30.0
                if launch_dist < lower_contact_launch_context['min_launch_dist']:
                    score += (lower_contact_launch_context['min_launch_dist'] - launch_dist) * 4.0
                if upward_progress < lower_contact_launch_context['min_upward']:
                    score += (lower_contact_launch_context['min_upward'] - upward_progress) * 3.0
                if launch_dist > lower_contact_launch_context['max_launch_dist']:
                    score += (launch_dist - lower_contact_launch_context['max_launch_dist']) * 2.0
            elif rally_contact_grace:
                # After a lower-court racket hit the ball often shrinks quickly from a
                # very large pre-contact blob. Use a capped rolling reference size so
                # we prefer the launched ball instead of nearby racket/string contours.
                _rally_ref_size = getattr(self, '_rally_contact_ref_size', None)
                if _rally_ref_size is None:
                    _rally_ref_size = max(40, min(self.ball_size or area, 140))
                _rally_size_ratio = abs(area - _rally_ref_size) / max(_rally_ref_size, 1)
                score = _rally_size_ratio * 85 + distance * 0.08
                if predicted_point is not None:
                    _rally_pred_dist = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    score += _rally_pred_dist * 0.7
                if getattr(self, '_rally_contact_origin', None) is not None:
                    origin_x, origin_y = self._rally_contact_origin
                    radial_dist = math.hypot(cx - origin_x, cy - origin_y)
                    min_progress = max(90.0, getattr(self, '_rally_contact_progress', 0.0) - 20.0)
                    if radial_dist < min_progress:
                        score += (min_progress - radial_dist) * 3.5
                    upward_progress = origin_y - cy
                    if upward_progress < 70.0:
                        score += (70.0 - upward_progress) * 3.0
            elif ground_bounce_grace:
                _bounce_ref_size = getattr(self, '_ground_bounce_ref_size', None)
                if _bounce_ref_size is None:
                    _bounce_ref_size = max(8.0, min(self.ball_size or area, 90.0))
                _bounce_size_ratio = abs(area - _bounce_ref_size) / max(_bounce_ref_size, 1)
                score = _bounce_size_ratio * 55.0 + distance * 0.08
                if predicted_point is not None:
                    _bounce_pred_dist = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    score += _bounce_pred_dist * 0.55
                if getattr(self, '_ground_bounce_expected', None) is not None:
                    expected_x, expected_y = self._ground_bounce_expected
                    score += math.hypot(cx - expected_x, cy - expected_y) * 0.45
                if getattr(self, '_ground_bounce_origin', None) is not None:
                    origin_x, origin_y = self._ground_bounce_origin
                    radial_dist = math.hypot(cx - origin_x, cy - origin_y)
                    min_progress = max(5.0, getattr(self, '_ground_bounce_progress', 0.0) - 10.0)
                    if radial_dist < min_progress:
                        score += (min_progress - radial_dist) * 3.0
                    if cy > origin_y + 4:
                        score += (cy - origin_y) * 2.5
            else:
                score = distance + (size_ratio * 30)  # 30px penalty per 100% size change
            # Direction / speed consistency penalties: skip in post-reacquire window
            # because serve contact reverses the ball direction instantly.
            # _in_post_reacq was computed before the counter decrement (see above) so it
            # is correct even on the last frame of the window (counter was 1, now 0).
            if (not full_frame_scan and not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace
                    and lower_contact_launch_context is None and ground_bounce_context is None and not _in_post_reacq
                    and self.last_motion and distance > 0):
                lm_dx = self.last_motion['dx']
                lm_dy = self.last_motion['dy']
                lm_dist = self.last_motion['distance']
                mv_dx = cx - self.ball_center[0]
                mv_dy = cy - self.ball_center[1]
                dot = lm_dx * mv_dx + lm_dy * mv_dy
                if dot < 0:
                    score += 80  # stronger penalty for opposite direction
                if lm_dist and lm_dist > 0:
                    speed_diff = abs(distance - lm_dist)
                    score += speed_diff * 1.5
                    align_bonus = dot / (lm_dist * distance)
                    score -= max(0.0, align_bonus) * 40
            if (not full_frame_scan and not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace
                    and lower_contact_launch_context is None and ground_bounce_context is None and not _in_post_reacq and predicted_point):
                pdx = cx - predicted_point[0]
                pdy = cy - predicted_point[1]
                predicted_distance = np.sqrt(pdx * pdx + pdy * pdy)
                # Give mild preference to being near predicted point
                score += predicted_distance * 0.5

            motion_note = ""
            motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
            if upper_exit_transition_context and self.ball_center is not None:
                prev_x, prev_y = self.ball_center
                upward_progress = prev_y - cy
                if upward_progress >= 12:
                    score -= min(140.0, upward_progress * 3.0)
                    if area <= max(14.0, self.ball_size * 0.25):
                        score -= 25.0
                elif cy >= prev_y - 8:
                    score += 140.0
            if not allow_inactive and contact_reacquire_bounds is not None:
                if upper_exit_transition_context:
                    if (cx < contact_reacquire_bounds['min_x'] or
                            cx > contact_reacquire_bounds['max_x'] or
                            cy > contact_reacquire_bounds['max_y']):
                        print(f"  DEBUG: Contour {i} REJECTED - upper-exit bounds "
                              f"x={contact_reacquire_bounds['min_x']}-{contact_reacquire_bounds['max_x']} "
                              f"max_y={contact_reacquire_bounds['max_y']}")
                        continue
                else:
                    if (cx < contact_reacquire_bounds['min_x'] or cx > contact_reacquire_bounds['max_x'] or
                            cy < contact_reacquire_bounds['min_y']):
                        print(f"  DEBUG: Contour {i} REJECTED - upper-contact bounds "
                              f"x={contact_reacquire_bounds['min_x']}-{contact_reacquire_bounds['max_x']} "
                              f"min_y={contact_reacquire_bounds['min_y']}")
                        continue
            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy)) if not allow_inactive else None
            if not allow_inactive and motion_metrics is not None:
                motion_mean = motion_metrics['mean']
                motion_max = motion_metrics['max']
                motion_note = f", motion_mean={motion_mean:.1f}, motion_max={motion_max:.1f}"
                if upper_exit_transition_context:
                    if motion_max >= 40.0:
                        score -= min(55.0, motion_max * 0.3)
                    elif motion_mean < 1.0 and motion_max < 8.0:
                        score += 80.0
                if frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0:
                    score += 1800
                    print(f"  DEBUG: Contour {i} PENALIZED - frame0 hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                static_hotspot = ((area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                if static_hotspot:
                    score += 1200
            if upper_wall_search_context and self.ball_center is not None:
                prev_x, prev_y = self.ball_center
                if source == 'alt6':
                    score -= 20.0
                    if cy <= prev_y - 6:
                        score -= 40.0
                    if abs(cx - prev_x) <= 40:
                        score -= 10.0
                elif source == 'behind_net' and cy >= prev_y + 8:
                    score += 140.0
             
            candidates.append((i, source, cx, cy, area, distance, size_ratio, score))
            source_label = f"[{source}]" if source != 'single' else ""
            print(f"  DEBUG: Contour {i} {source_label} CANDIDATE - pos=({cx},{cy}), area={area:.1f}px, distance={distance:.1f}px, size_ratio={size_ratio:.2f}, score={score:.1f}{motion_note}")
            
            if score < best_score:
                best_score = score
                best_contour = contour
                best_source = source

        # Always persist candidates for C-key contour debug overlay (lightweight list).
        self._debug_contour_candidates = list(candidates)

        # Early-serve bias: when starting and no previous ball, favor the highest (smallest y) valid contour
        if self.ball_center is None and self.frame_count <= self.start_frame + 10 and candidates:
            highest = min(candidates, key=lambda c: (c[3], c[4]))  # prioritize lowest y (higher on screen), then smaller area
            _, best_source, _, _, _, _, _, _ = highest
            best_contour = contours[highest[0]][1]
            print(f"  DEBUG: Early-serve bias -> picking highest contour (y={highest[3]}, area={highest[4]:.1f})")

        if best_contour is not None:
            # Update ball position
            M = cv2.moments(best_contour)
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            self.last_seen_frame = self.frame_count
            self.edge_wait = False
            if upper_exit_wait:
                self._upper_exit_wait_frames = 0
            if hasattr(self, 'ball_center') and self.ball_center:
                prev_x, prev_y = self.ball_center
                self.last_delta = (cx - prev_x, cy - prev_y)
            else:
                self.last_delta = None
            
            # Check if this is likely a false positive jump
            # If ball was at edge and closest match is far away, ball likely went off-screen
            # Skip this check during full-frame scan (re-acquisition after occlusion)
            should_guard_selected = self.ball_center and not allow_inactive and (
                self.stuck_frame_count < 5 or contact_reacquire_bounds is not None or
                self._find_frame0_background_hotspot((cx, cy)) is not None
            )
            if should_guard_selected:
                x_prev, y_prev = self.ball_center
                actual_distance = np.sqrt((cx - x_prev)**2 + (cy - y_prev)**2)
                edge_threshold = 5  # pixels from edge
                jump_threshold = 50  # pixels - suspicious if ball "moves" more than this
                
                frame_height, frame_width = frame.shape[:2]
                at_edge = (y_prev < edge_threshold or y_prev > frame_height - edge_threshold or
                          x_prev < edge_threshold or x_prev > frame_width - edge_threshold)
                
                if at_edge and actual_distance > jump_threshold:
                    self._record_rejected_contour_debug(
                        best_contour,
                        x1,
                        y1,
                        cx,
                        cy,
                        cv2.contourArea(best_contour),
                        f"edge jump {actual_distance:.1f}px from {self.ball_center}",
                        source=best_source,
                    )
                    print(f"  DEBUG: [REJECTED] Candidate at ({cx},{cy}), distance={actual_distance:.1f}px")
                    print(f"  DEBUG: Ball was at edge (y={y_prev}), closest match is {actual_distance:.1f}px away")
                    print(f"  DEBUG: This is likely a FALSE POSITIVE - ball probably went off-screen")
                    print(f"  DEBUG: KEEPING marker at last edge position: {self.ball_center}")
                    print(f"  DEBUG: Will wait for ball to return...")
                    return self.ball_center

                selected_motion = self._candidate_motion_metrics(frame_gray, cx, cy)
                motion_mean = selected_motion['mean'] if selected_motion is not None else 0.0
                motion_max = selected_motion['max'] if selected_motion is not None else 0.0
                selected_area = cv2.contourArea(best_contour)
                static_hotspot = ((selected_area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                suspicious_upper_static_jump = (
                    actual_distance > max(180.0, frame_width * 0.045) and
                    cy < max(220, int(frame_height * 0.11)) and
                    motion_mean < 4.0 and
                    motion_max < 12.0
                )
                outside_contact_bounds = (
                    contact_reacquire_bounds is not None and (
                        cx < contact_reacquire_bounds['min_x'] or
                        cx > contact_reacquire_bounds['max_x'] or
                        cy < contact_reacquire_bounds['min_y']
                    )
                )
                frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                frame0_background = (
                    frame0_hotspot is not None and (
                        selected_motion is None or (motion_mean < 8.0 and motion_max < 35.0)
                    )
                )
                upper_exit_non_reentry = (
                    not upper_exit_wait and
                    contact_reacquire_bounds is not None and
                    self.ground_bounce_count > 0 and
                    self.ball_size is not None and
                    self.ball_size >= 35 and
                    selected_area <= max(14.0, self.ball_size * 0.35) and
                    y_prev < max(220, int(frame_height * 0.12)) and
                    actual_distance <= max(90.0, frame_width * 0.025) and
                    cy >= (y_prev - 12)
                )
                if static_hotspot or suspicious_upper_static_jump or outside_contact_bounds or frame0_background:
                    reason = f"static patch mean={motion_mean:.1f} max={motion_max:.1f}"
                    if outside_contact_bounds:
                        reason = (
                            f"upper-contact bounds x={contact_reacquire_bounds['min_x']}-"
                            f"{contact_reacquire_bounds['max_x']} min_y={contact_reacquire_bounds['min_y']}"
                        )
                    elif suspicious_upper_static_jump:
                        reason = (
                            f"upper static jump dist={actual_distance:.1f} "
                            f"mean={motion_mean:.1f} max={motion_max:.1f}"
                        )
                    elif frame0_background:
                        hotspot_pos = tuple(frame0_hotspot.get('pos', [cx, cy]))
                        hotspot_radius = frame0_hotspot.get('radius', 0)
                        reason = f"frame0 hotspot at {hotspot_pos} r={hotspot_radius}"
                    self._record_rejected_contour_debug(
                        best_contour,
                        x1,
                        y1,
                        cx,
                        cy,
                        selected_area,
                        reason,
                        source=best_source,
                    )
                    self._learn_ignored_tracking_position((cx, cy), radius=80, ttl=200, reason=reason)
                    self.stuck_frame_count += 1
                    print(f"Frame {self.frame_count}: [TRACKING_FP_REJECT] holding {self.ball_center} "
                          f"instead of false hotspot ({cx},{cy}) motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                    return self.ball_center
                if upper_exit_non_reentry:
                    self._record_rejected_contour_debug(
                        best_contour,
                        x1,
                        y1,
                        cx,
                        cy,
                        selected_area,
                        f"upper-exit non-reentry area={selected_area:.1f}px",
                        source=best_source,
                    )
                    self._upper_exit_wait_frames = max(getattr(self, '_upper_exit_wait_frames', 0), 4)
                    self.edge_wait = True
                    self.stuck_frame_count += 1
                    print(f"Frame {self.frame_count}: [UPPER-EXIT WAIT] holding {self.ball_center} "
                          f"instead of weak upper blob ({cx},{cy}) area={selected_area:.1f}px")
                    return self.ball_center
                if upper_exit_wait:
                    reentry_y_limit = max(110, min(y_prev - 15, 150))
                    if cy > reentry_y_limit:
                        self._record_rejected_contour_debug(
                            best_contour,
                            x1,
                            y1,
                            cx,
                            cy,
                            selected_area,
                            f"upper-exit reentry y_limit={reentry_y_limit}",
                            source=best_source,
                        )
                        self.stuck_frame_count += 1
                        print(f"Frame {self.frame_count}: [UPPER-EXIT WAIT] ignoring non-reentry blob ({cx},{cy}) "
                              f"y_limit={reentry_y_limit}")
                        return self.ball_center

            source_label = f" from [{best_source}] filter" if best_source and best_source != 'single' else ""
            print(f"  DEBUG: [SELECTED] contour at ({cx},{cy}), score={best_score:.1f}{source_label}")

            selected_override = None
            upper_far_player_override_context = (
                not allow_inactive and
                self.ball_center is not None and
                self.ground_bounce_count > 0 and
                (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 20 and
                self.ball_center[1] < max(185, int(frame.shape[0] * 0.09)) and
                (self.ball_size is not None and self.ball_size <= 35)
            )
            if ((upper_wall_search_context or upper_far_player_override_context)
                    and best_source == 'behind_net' and self.ball_center is not None
                    and cy >= self.ball_center[1] + 8):
                upper_exit_alt = self._retrack_with_upper_exit_low_s(
                    search_frame, x1, y1, predicted_point, frame_gray=frame_gray
                )
                if (upper_exit_alt is not None and
                        upper_exit_alt['pos'][1] <= self.ball_center[1] - 6 and
                        abs(upper_exit_alt['pos'][0] - self.ball_center[0]) <= 120):
                    selected_override = upper_exit_alt
                    cx, cy = upper_exit_alt['pos']
                    best_source = upper_exit_alt['label']
                    print(f"Frame {self.frame_count}: [UPPER ALT OVERRIDE] Replacing behind_net pick with {best_source} at ({cx},{cy})")

            # Get HSV values at new position
            hsv_values = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[cy, cx]
            bulb_size = selected_override['area'] if selected_override is not None else cv2.contourArea(best_contour)
            if best_source == 'alt4' and self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
                self.hsv_lower = self.alt4_hsv_lower
                self.hsv_upper = self.alt4_hsv_upper
            elif best_source == 'alt5' and self.alt5_hsv_lower is not None and self.alt5_hsv_upper is not None:
                self.hsv_lower = self.alt5_hsv_lower
                self.hsv_upper = self.alt5_hsv_upper
            elif best_source == 'alt6' and self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
                self.hsv_lower = self.alt6_hsv_lower
                self.hsv_upper = self.alt6_hsv_upper
            
            # Calculate velocity (distance moved)
            prev_pos = self.ball_center if self.ball_center else None
            prev_ball_size = self.ball_size
            dx = dy = 0
            direction_deg = None
            if prev_pos:
                dx = cx - prev_pos[0]
                dy = cy - prev_pos[1]
                velocity = math.hypot(dx, dy)
                direction_deg = math.degrees(math.atan2(dy, dx))
            else:
                velocity = 0
            
            # If direction/speed look wrong, try alternative HSV before committing
            # Skip this during full-frame scan recovery - ball direction changed after player hit.
            # Also skip during post-reacquire window: the serve contact can instantly reverse
            # the ball, so a direction / speed change here is expected and correct.
            skip_upper_wall_override = upper_wall_search_context and best_source in ('alt4', 'alt5', 'alt6')
            if (not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace and not serve_direction_search
                    and lower_contact_launch_context is None and not _in_post_reacq and self.last_motion
                    and not skip_upper_wall_override
                    and self.ball_center and self.stuck_frame_count < 5):
                lm_dx = self.last_motion['dx']
                lm_dy = self.last_motion['dy']
                lm_dist = self.last_motion['distance']
                mv_dx = cx - self.ball_center[0]
                mv_dy = cy - self.ball_center[1]
                dot = lm_dx * mv_dx + lm_dy * mv_dy
                speed_diff = abs(velocity - lm_dist) if lm_dist else 0
                angle_jump = 0.0
                if self.last_direction is not None and direction_deg is not None:
                    delta = abs(direction_deg - self.last_direction) % 360
                    angle_jump = min(delta, 360 - delta)
                speed_ratio = (velocity / lm_dist) if lm_dist else 1.0
                if (dot < 0 or angle_jump > 45 or speed_ratio > 1.8 or speed_ratio < 0.6 or
                        (lm_dist and speed_diff > lm_dist * 0.6)):
                    retrack = self.retrack_with_alt_hsv(
                        search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                        frame_gray=frame_gray
                    )
                    if retrack is not None:
                        cx, cy = retrack['pos']
                        hsv_values = retrack['hsv']
                        bulb_size = retrack['area']
                        self.using_alt_hsv = True
                        self.using_alt2_hsv = False
                        if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                            self.hsv_lower = self.alt_focus_hsv_lower
                            self.hsv_upper = self.alt_focus_hsv_upper
                        print(f"Frame {self.frame_count}: [ALT HSV OVERRIDE] Ball at ({cx}, {cy})")
                    elif self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
                        retrack2 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            frame_gray=frame_gray
                        )
                        if retrack2 is not None:
                            cx, cy = retrack2['pos']
                            hsv_values = retrack2['hsv']
                            bulb_size = retrack2['area']
                            self.using_alt_hsv = False
                            self.using_alt2_hsv = True
                            self.using_alt3_hsv = False
                            self.hsv_lower = self.alt2_hsv_lower
                            self.hsv_upper = self.alt2_hsv_upper
                            print(f"Frame {self.frame_count}: [ALT2 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            if self.last_motion and self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
                                lm_dist = self.last_motion['distance']
                                mv_dx = cx - self.ball_center[0]
                                mv_dy = cy - self.ball_center[1]
                                mv_dist = math.hypot(mv_dx, mv_dy)
                                speed_ratio = (mv_dist / lm_dist) if lm_dist else 1.0
                                angle_jump = 0.0
                                if self.last_direction is not None:
                                    direction_deg = math.degrees(math.atan2(mv_dy, mv_dx))
                                    delta = abs(direction_deg - self.last_direction) % 360
                                    angle_jump = min(delta, 360 - delta)
                                if angle_jump > 45 or speed_ratio > 1.8 or speed_ratio < 0.6:
                                    retrack3 = self.retrack_with_alt2_hsv(
                                        search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                                        lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray
                                    )
                                    if retrack3 is not None:
                                        cx, cy = retrack3['pos']
                                        hsv_values = retrack3['hsv']
                                        bulb_size = retrack3['area']
                                        self.using_alt2_hsv = False
                                        self.using_alt3_hsv = True
                                        self.hsv_lower = self.alt3_hsv_lower
                                        self.hsv_upper = self.alt3_hsv_upper
                                        print(f"Frame {self.frame_count}: [ALT3 HSV OVERRIDE] Ball at ({cx}, {cy})")
                    elif self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
                        retrack3 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray
                        )
                        if retrack3 is not None:
                            cx, cy = retrack3['pos']
                            hsv_values = retrack3['hsv']
                            bulb_size = retrack3['area']
                            self.using_alt_hsv = False
                            self.using_alt2_hsv = False
                            self.using_alt3_hsv = True
                            self.hsv_lower = self.alt3_hsv_lower
                            self.hsv_upper = self.alt3_hsv_upper
                            print(f"Frame {self.frame_count}: [ALT3 HSV OVERRIDE] Ball at ({cx}, {cy})")

            # Detect sudden ball size drop (occlusion by player)
            # If ball was > 30px and now < 5px, it's being occluded — don't trust this detection
            soft_upper_contact_recover = (
                not allow_inactive and
                contact_reacquire_bounds is not None and
                self.ball_center is not None and
                self.last_motion is not None and
                self.last_motion.get('dy', 0.0) < -2.0 and
                self.ground_bounce_count > 0 and
                prev_ball_size is not None and prev_ball_size <= 20
            )
            if soft_upper_contact_recover:
                contact_reference = self.ball_center
                soft_recover = self._recover_contact_phase_ball(
                    frame, contact_reference, frame_gray,
                    predicted_point=None,
                    max_prev_speed=45.0,
                    upper_stuck_mode=True,
                )
                if soft_recover is not None:
                    trusted_predicted_point = (
                        int(self.ball_center[0] + self.last_motion.get('dx', 0.0)),
                        int(self.ball_center[1] + self.last_motion.get('dy', 0.0)),
                    )
                    current_predicted_distance = float('inf')
                    recovered_predicted_distance = float('inf')
                    if trusted_predicted_point is not None:
                        current_predicted_distance = math.hypot(
                            cx - trusted_predicted_point[0],
                            cy - trusted_predicted_point[1],
                        )
                        recovered_predicted_distance = math.hypot(
                            soft_recover['pos'][0] - trusted_predicted_point[0],
                            soft_recover['pos'][1] - trusted_predicted_point[1],
                        )
                    current_upward = self.ball_center[1] - cy
                    recovered_upward = self.ball_center[1] - soft_recover['pos'][1]
                    if (recovered_upward >= current_upward + 12 or
                            recovered_predicted_distance + 10 < current_predicted_distance):
                        cx, cy = soft_recover['pos']
                        hsv_values = soft_recover['hsv']
                        bulb_size = soft_recover['area']
                        dx = cx - self.ball_center[0]
                        dy = cy - self.ball_center[1]
                        velocity = math.hypot(dx, dy)
                        direction_deg = math.degrees(math.atan2(dy, dx)) if velocity > 0 else 0.0
                        print(f"Frame {self.frame_count}: [UPPER CONTACT SOFT RECOVER] Ball at ({cx}, {cy}) "
                              f"from {soft_recover.get('label', 'n/a')} mode={soft_recover.get('mode', 'n/a')}")

            upper_post_bounce_size_spike = (
                not allow_inactive and
                contact_reacquire_bounds is not None and
                self.ball_center is not None and
                self.ground_bounce_count > 0 and
                prev_ball_size is not None and prev_ball_size <= 15 and
                bulb_size > max(18, prev_ball_size * 4.0)
            )
            if upper_post_bounce_size_spike:
                spike_recover = self._recover_contact_phase_ball(
                    frame, self.ball_center, frame_gray,
                    predicted_point=None,
                    max_prev_speed=45.0,
                    upper_stuck_mode=True,
                )
                if spike_recover is not None:
                    recover_x, recover_y = spike_recover['pos']
                    recover_area = spike_recover['area']
                    if recover_area <= bulb_size * 0.65 and recover_y <= cy - 10:
                        cx, cy = spike_recover['pos']
                        hsv_values = spike_recover['hsv']
                        bulb_size = spike_recover['area']
                        dx = cx - self.ball_center[0]
                        dy = cy - self.ball_center[1]
                        velocity = math.hypot(dx, dy)
                        direction_deg = math.degrees(math.atan2(dy, dx)) if velocity > 0 else 0.0
                        print(f"Frame {self.frame_count}: [UPPER SIZE-SPIKE RECOVER] Ball at ({cx}, {cy}) "
                              f"from {spike_recover.get('label', 'n/a')} mode={spike_recover.get('mode', 'n/a')}")

            if prev_ball_size and prev_ball_size > 30 and bulb_size < 5:
                print(f"Frame {self.frame_count}: Ball size dropped {prev_ball_size:.0f}->{bulb_size:.0f}px - likely occluded by player")
                self.stuck_frame_count += 2  # accelerate stuck detection
                return self.ball_center

            # Gate large direction/velocity changes for a few frames
            # Skip this gate during full-frame scan recovery (ball changed direction after player hit).
            # Also skip during the post-reacquire window: serve contact reverses direction immediately.
            if (not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace and not serve_direction_search
                    and not upper_exit_transition_context and not _in_post_reacq and self.last_motion
                    and self.ball_center and self.stuck_frame_count < 5):
                lm_dist = self.last_motion['distance']
                angle_jump = 0.0
                if self.last_direction is not None and direction_deg is not None:
                    delta = abs(direction_deg - self.last_direction) % 360
                    angle_jump = min(delta, 360 - delta)

                # Net-area transitions often produce genuine direction changes.
                near_net = False
                if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
                    near_net = (self.net_area_y_min - 100 <= self.ball_center[1] <= self.net_area_y_max + 100)

                angle_threshold = 90 if near_net else 70
                speed_ratio = (velocity / lm_dist) if lm_dist else 1.0
                change_detected = (angle_jump > angle_threshold) or (speed_ratio > 2.0) or (speed_ratio < 0.5)
                predicted_turn_candidate = False
                predicted_continuation_candidate = False
                small_ball_upper_flight = (
                    prev_ball_size is not None and prev_ball_size <= 12 and bulb_size <= 12
                )
                if (predicted_point is not None and small_ball_upper_flight and
                        240 <= cy < max(420, int(frame_height * 0.20))):
                    predicted_turn_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    predicted_turn_cap = max(90, int(frame_width * 0.025))
                    predicted_turn_candidate = predicted_turn_distance <= predicted_turn_cap
                lower_contact_launch_candidate = False
                if lower_contact_launch_context is not None and self.ball_center is not None:
                    origin_x, origin_y = lower_contact_launch_context['origin']
                    launch_upward = dy <= -max(45, int(frame_height * 0.02))
                    launch_dist = math.hypot(cx - origin_x, cy - origin_y)
                    expected_x, expected_y = lower_contact_launch_context['expected']
                    expected_distance = math.hypot(cx - expected_x, cy - expected_y)
                    candidate_not_tiny = bulb_size >= max(20, min(int(lower_contact_launch_context['ref_size'] * 0.50), 80))
                    lower_contact_launch_candidate = (
                        launch_upward and
                        candidate_not_tiny and
                        launch_dist >= lower_contact_launch_context['min_launch_dist'] and
                        launch_dist <= lower_contact_launch_context['max_launch_dist'] and
                        expected_distance <= max(140, int(frame_width * 0.04))
                    )
                ground_bounce_candidate = False
                if ground_bounce_context is not None and self.ball_center is not None:
                    origin_x, origin_y = ground_bounce_context['origin']
                    expected_x, expected_y = ground_bounce_context['expected']
                    bounce_dist = math.hypot(cx - origin_x, cy - origin_y)
                    upward_progress = origin_y - cy
                    expected_distance = math.hypot(cx - expected_x, cy - expected_y)
                    lateral_dx = cx - origin_x
                    incoming_dx = ground_bounce_context['incoming_dx']
                    same_direction_x = abs(incoming_dx) <= 2.0 or (lateral_dx * incoming_dx) >= -8.0
                    candidate_not_tiny = bulb_size >= max(1, min(int(ground_bounce_context['ref_size'] * 0.08), 4))
                    ground_bounce_candidate = (
                        same_direction_x and
                        candidate_not_tiny and
                        upward_progress >= ground_bounce_context['min_upward'] and
                        bounce_dist >= ground_bounce_context['min_launch_dist'] and
                        bounce_dist <= ground_bounce_context['max_launch_dist'] and
                        expected_distance <= ground_bounce_context['expected_cap']
                    )
                if predicted_point is not None and not predicted_turn_candidate:
                    predicted_path_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    predicted_path_cap = max(70, int(frame_width * 0.02))
                    smooth_angle_continuation = angle_jump <= 30
                    moderate_speed_drop = speed_ratio >= 0.30
                    predicted_continuation_candidate = (
                        smooth_angle_continuation and
                        moderate_speed_drop and
                        predicted_path_distance <= predicted_path_cap
                    )
                    if (not predicted_continuation_candidate and contact_reacquire_bounds is not None and
                            self.ground_bounce_count > 0 and self.last_motion is not None and
                            self.last_motion.get('dy', 0.0) < -2.0 and prev_ball_size is not None and
                            prev_ball_size <= 12 and bulb_size <= 12):
                        bounce_soft_angle = angle_jump <= 40
                        bounce_soft_speed = speed_ratio >= 0.20
                        bounce_soft_cap = max(110, int(frame_width * 0.03))
                        predicted_continuation_candidate = (
                            bounce_soft_angle and
                            bounce_soft_speed and
                            predicted_path_distance <= bounce_soft_cap
                        )

                if (change_detected and not predicted_turn_candidate and not predicted_continuation_candidate
                        and not lower_contact_launch_candidate and not ground_bounce_candidate):
                    self.direction_change_streak += 1
                    max_hold = 2 if near_net else 3
                    if self.direction_change_streak < max_hold:
                        print(f"Frame {self.frame_count}: Direction change candidate (holding {self.direction_change_streak}/{max_hold}) angle_jump={angle_jump:.1f}")
                        self.stuck_frame_count += 1
                        return self.ball_center
                else:
                    if (change_detected and not ground_bounce_candidate and not predicted_turn_candidate
                            and not predicted_continuation_candidate):
                        strong_x_reversal = (self.last_motion['dx'] * dx) < -12 if self.last_motion is not None else False
                        if lower_contact_launch_candidate or strong_x_reversal or angle_jump >= 120 or speed_ratio > 1.8:
                            if self.ground_bounce_count > 0:
                                print(f"Frame {self.frame_count}: Resetting bounce count after non-bounce shot change")
                            self.ground_bounce_count = 0
                            self.last_ground_bounce_frame = -1000000
                    if change_detected and predicted_turn_candidate:
                        print(f"Frame {self.frame_count}: Allowing upper-flight turn near predicted path")
                    elif change_detected and predicted_continuation_candidate:
                        print(f"Frame {self.frame_count}: Allowing predicted-path continuation after speed drop")
                    elif change_detected and lower_contact_launch_candidate:
                        self._rally_contact_grace_frames = max(getattr(self, '_rally_contact_grace_frames', 0), 3)
                        self._rally_contact_ref_size = max(40.0, min(float(bulb_size), 140.0))
                        self._rally_contact_origin = lower_contact_launch_context['origin']
                        self._rally_contact_expected = lower_contact_launch_context['expected']
                        self._rally_contact_progress = math.hypot(
                            cx - self._rally_contact_origin[0],
                            cy - self._rally_contact_origin[1],
                        )
                        print(f"Frame {self.frame_count}: Allowing lower-racket contact launch")
                    elif change_detected and ground_bounce_candidate:
                        if (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) > 3:
                            self.ground_bounce_count += 1
                            self.last_ground_bounce_frame = self.frame_count
                            print(f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} detected")
                        self._ground_bounce_grace_frames = max(getattr(self, '_ground_bounce_grace_frames', 0), 3)
                        self._ground_bounce_ref_size = max(8.0, min(max(float(bulb_size), ground_bounce_context['ref_size']), 90.0))
                        self._ground_bounce_origin = ground_bounce_context['origin']
                        self._ground_bounce_expected = ground_bounce_context['expected']
                        self._ground_bounce_progress = math.hypot(
                            cx - self._ground_bounce_origin[0],
                            cy - self._ground_bounce_origin[1],
                        )
                        print(f"Frame {self.frame_count}: Allowing ground-bounce continuation")
                    self.direction_change_streak = 0

            # Update tracking data
            # If re-acquiring after full-frame scan, reset velocity/direction state
            if self.stuck_frame_count >= 5:
                print(f"Frame {self.frame_count}: [RE-ACQUIRED] Ball found at ({cx},{cy}) after {self.stuck_frame_count} stuck frames")
                self.ball_velocity_history = []
                self.last_motion = None
                self.last_direction = None
                self.direction_change_streak = 0
                self.stuck_frame_count = 0
                self._recent_max_ball_size = 0
                self.ball_center = (cx, cy)
                self.ball_hsv = hsv_values
                self.ball_size = bulb_size
                self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Skip all correction mechanisms (focus loss, alt HSV) on re-acquisition
                return self.ball_center
            self.ball_center = (cx, cy)
            self.ball_hsv = hsv_values
            self.ball_size = bulb_size
            if serve_direction_search:
                # The first few frames after serve contact contain the biggest
                # legitimate direction/speed change of the point.
                self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 3)
            if rally_contact_grace:
                current_ref = getattr(self, '_rally_contact_ref_size', None)
                if current_ref is None:
                    current_ref = bulb_size
                self._rally_contact_ref_size = max(40.0, min(max(float(current_ref), float(bulb_size)), 140.0))
                if getattr(self, '_rally_contact_origin', None) is not None:
                    origin_x, origin_y = self._rally_contact_origin
                    self._rally_contact_progress = max(
                        getattr(self, '_rally_contact_progress', 0.0),
                        math.hypot(cx - origin_x, cy - origin_y),
                    )
                self._rally_contact_expected = self.get_predicted_point()
            if ground_bounce_grace:
                current_ref = getattr(self, '_ground_bounce_ref_size', None)
                if current_ref is None:
                    current_ref = bulb_size
                self._ground_bounce_ref_size = max(8.0, min(max(float(current_ref), float(bulb_size)), 90.0))
                if getattr(self, '_ground_bounce_origin', None) is not None:
                    origin_x, origin_y = self._ground_bounce_origin
                    self._ground_bounce_progress = max(
                        getattr(self, '_ground_bounce_progress', 0.0),
                        math.hypot(cx - origin_x, cy - origin_y),
                    )
                self._ground_bounce_expected = self.get_predicted_point()
            # Track max ball size over recent frames for occlusion detection
            if bulb_size > 5:
                self._recent_max_ball_size = max(getattr(self, '_recent_max_ball_size', 0), bulb_size)
            # Store frame for motion-based re-acquisition (every 3rd frame to save CPU)
            if self.frame_count % 3 == 0:
                self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_height, frame_width = frame.shape[:2]
            edge_margin = 50
            self.near_edge = (
                self.ball_center[0] < edge_margin or self.ball_center[0] > frame_width - edge_margin or
                self.ball_center[1] < edge_margin or self.ball_center[1] > frame_height - edge_margin
            )

            # Log motion metrics and detect focus loss spikes
            focus_loss_triggered = False
            if not allow_inactive:
                focus_loss_triggered = self.log_motion_metrics(dx, dy, velocity, direction_deg)
                if focus_loss_triggered and (serve_contact_grace or rally_contact_grace):
                    self.focus_loss_active = False
                    self.focus_loss_frame = None
                    print(f"Frame {self.frame_count}: Ignoring focus-loss spike during contact grace")
            
            # Track velocity history (last 5 frames)
            if not hasattr(self, 'ball_velocity_history'):
                self.ball_velocity_history = []
            self.ball_velocity_history.append(velocity)
            if len(self.ball_velocity_history) > 5:
                self.ball_velocity_history.pop(0)

            # If focus loss triggered, re-run detection using alternative HSV only.
            # Skip during serve-contact grace / post-reacquire: a large jump here is
            # usually the racket launch, not a camera focus change, so re-running can
            # replace the correct ball with a nearby racket/blob false positive.
            if focus_loss_triggered and not serve_contact_grace and not rally_contact_grace and not _in_post_reacq:
                print(f"Frame {self.frame_count}: Re-running detection with alt HSV after focus loss")
                retrack = self.retrack_with_alt_hsv(
                    search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                    frame_gray=frame_gray
                )
                if retrack is not None:
                    new_pos = retrack['pos']
                    new_area = retrack['area']
                    new_hsv = retrack['hsv']
                    self.ball_center = new_pos
                    self.ball_hsv = new_hsv
                    self.ball_size = new_area
                    if prev_pos:
                        new_dx = new_pos[0] - prev_pos[0]
                        new_dy = new_pos[1] - prev_pos[1]
                        new_velocity = math.hypot(new_dx, new_dy)
                        new_direction = math.degrees(math.atan2(new_dy, new_dx))
                        self.last_delta = (new_dx, new_dy)
                        self.last_motion = {
                            'distance': new_velocity,
                            'dx': new_dx,
                            'dy': new_dy,
                            'direction_deg': new_direction
                        }
                        if self.ball_velocity_history:
                            self.ball_velocity_history[-1] = new_velocity
                        print(f"Frame {self.frame_count}: [RETRACK ALT] Ball at ({new_pos[0]}, {new_pos[1]}) "
                              f"- Size: {new_area:.1f}px - Movement: {new_velocity:.1f}px Dir: {new_direction:+.1f} deg")
                # Clear focus lost indicator after switching to alternative HSV
                self.focus_loss_active = False

            # If ball did not move while on alternative HSV, switch to alternative 2
            if self.using_alt_hsv and not self.using_alt2_hsv and self.last_motion and self.last_motion['distance'] == 0:
                print(f"Frame {self.frame_count}: No movement detected, switching to alternative 2 HSV")
                self.using_alt2_hsv = True
                if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
                    self.hsv_lower = self.alt2_hsv_lower
                    self.hsv_upper = self.alt2_hsv_upper
                retrack2 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                    frame_gray=frame_gray
                )
                if retrack2 is not None:
                    new_pos = retrack2['pos']
                    new_area = retrack2['area']
                    new_hsv = retrack2['hsv']
                    self.ball_center = new_pos
                    self.ball_hsv = new_hsv
                    self.ball_size = new_area
                    if prev_pos:
                        new_dx = new_pos[0] - prev_pos[0]
                        new_dy = new_pos[1] - prev_pos[1]
                        new_velocity = math.hypot(new_dx, new_dy)
                        new_direction = math.degrees(math.atan2(new_dy, new_dx))
                        self.last_delta = (new_dx, new_dy)
                        self.last_motion = {
                            'distance': new_velocity,
                            'dx': new_dx,
                            'dy': new_dy,
                            'direction_deg': new_direction
                        }
                        if self.ball_velocity_history:
                            self.ball_velocity_history[-1] = new_velocity
                        print(f"Frame {self.frame_count}: [RETRACK ALT2] Ball at ({new_pos[0]}, {new_pos[1]}) "
                              f"- Size: {new_area:.1f}px - Movement: {new_velocity:.1f}px Dir: {new_direction:+.1f} deg")
            
            # Detect if ball stopped (average velocity < 2 pixels/frame for 5 frames)
            if len(self.ball_velocity_history) >= 5:
                avg_velocity = sum(self.ball_velocity_history) / len(self.ball_velocity_history)
                if avg_velocity < 2.0:
                    if not hasattr(self, 'ball_stopped') or not self.ball_stopped:
                        print(f"  DEBUG: [BALL STOPPED] Avg velocity: {avg_velocity:.1f}px/frame")
                        print(f"  DEBUG: Will search from initial position {self.initial_ball_position} for next ball")
                        self.ball_stopped = True
                    self.stuck_frame_count += 1
                else:
                    self.ball_stopped = False
                    self.stuck_frame_count = 0

            # Also count stuck frames when position hasn't moved
            if self.last_motion and self.last_motion['distance'] < 1.5:
                self.stuck_frame_count += 1
            elif self.last_motion and self.last_motion['distance'] >= 3.0:
                self.stuck_frame_count = 0
            
            # Add to HSV table using the final (possibly retracked) values
            final_pos = self.ball_center
            final_hsv = self.ball_hsv
            final_size = self.ball_size
            final_velocity = self.last_motion['distance'] if self.last_motion else velocity
            self.hsv_table.append({
                'frame': self.frame_count,
                'position': final_pos,
                'hsv': final_hsv.tolist() if final_hsv is not None else hsv_values.tolist(),
                'bulb_size': final_size,
                'velocity': final_velocity,
                'hsv_range': [self.hsv_lower[0], self.hsv_upper[0], 
                             self.hsv_lower[1], self.hsv_upper[1], 
                             self.hsv_lower[2], self.hsv_upper[2]]
            })
            
            if final_pos == (cx, cy):
                print(f"Frame {self.frame_count}: Ball at ({cx}, {cy}) - HSV: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]} - Size: {bulb_size:.1f}px - Velocity: {velocity:.1f}px/frame")
            
            return self.ball_center
        
        print(f"  DEBUG: [PROBLEM] No valid candidate found!")
        print(f"  DEBUG: Total contours: {len(contours)}, Valid candidates: {len(candidates)}")
        if len(contours) > 0 and len(candidates) == 0:
            size_cap = f"{self.serve_ball_size_min}-{self.serve_ball_size_max}px (serve scan)" if allow_inactive else f"1-{max(150, self.serve_ball_size_max)}px"
            print(f"  DEBUG: All {len(contours)} contours were rejected by size filter ({size_cap})")
            # Show the actual sizes that were rejected
            rejected_sizes = []
            for source, contour in contours[:5]:  # Show first 5
                rejected_sizes.append(f"{cv2.contourArea(contour):.1f}px")
            print(f"  DEBUG: Rejected sizes (first 5): {', '.join(rejected_sizes)}")
            print(f"  DEBUG: REASON: Ball size changed outside 1-150px range")
            print(f"  DEBUG:   - Ball may be too small (far away) or too large (very close)")
            print(f"  DEBUG:   - Consider adjusting size filter if ball is visible")
            if predicted_point:
                print(f"  DEBUG: Predicted point was {predicted_point}, consider widening search around it")
        # Early-frame fallback: if nothing selected, try full-frame smallest contour
        if best_contour is None and early_frames:
            fallback_mask = cv2.inRange(frame, hsv_lower_use, hsv_upper_use) if 'hsv_lower_use' in locals() else None
            if fallback_mask is not None:
                fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
                fallback_contours, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                viable = [c for c in fallback_contours if 1 <= cv2.contourArea(c) <= 150]
                if viable:
                    smallest = min(viable, key=cv2.contourArea)
                    M = cv2.moments(smallest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self.ball_center = (cx, cy)
                        print(f"  DEBUG: Fallback picked smallest full-frame contour at ({cx},{cy}) area={cv2.contourArea(smallest):.1f}")
                        return self.ball_center
        if best_contour is None and self._should_try_upper_exit_low_s(frame, allow_inactive, upper_exit_transition_context):
            upper_exit_low_s = self._retrack_with_upper_exit_low_s(
                search_frame, x1, y1, predicted_point, frame_gray=frame_gray
            )
            if upper_exit_low_s is not None:
                new_pos = upper_exit_low_s['pos']
                self.ball_center = new_pos
                self.ball_hsv = upper_exit_low_s['hsv']
                self.ball_size = upper_exit_low_s['area']
                self.using_alt_hsv = False
                self.using_alt2_hsv = False
                self.using_alt3_hsv = False
                self.hsv_lower = upper_exit_low_s['lower']
                self.hsv_upper = upper_exit_low_s['upper']
                self.stuck_frame_count = 0
                print(f"Frame {self.frame_count}: [UPPER EXIT LOW-S RECOVER] Ball at {new_pos} via {upper_exit_low_s['label']}")
                return self.ball_center
        if self.ball_center:
            print(f"  DEBUG: KEEPING marker at last known position: {self.ball_center}")
            print(f"  DEBUG: Will continue searching in next frame at same position...")
            # Increment so the state-machine STUCK_TIMEOUT (stuck≥15) can fire and
            # return to serve-scan mode even when every FFS candidate is filtered out
            # (e.g. all within the 500-px exclusion zone).
            self.stuck_frame_count += 1
            print(f"[BALL_LOST] f{self.frame_count}: no valid candidate, keeping pos={self.ball_center} stuck={self.stuck_frame_count}")
        # Keep the ball_center at last position instead of losing it
        return self.ball_center
    
    def _record_rejected_contour_debug(self, contour, x_offset, y_offset, cx, cy, area, reason, source=None):
        """Persist a rejected contour so paused playback can still show it."""
        if contour is None:
            return

        contour_global = contour.copy().astype(np.int32)
        contour_global[:, 0, 0] += int(x_offset)
        contour_global[:, 0, 1] += int(y_offset)
        self._debug_rejected_contours.append({
            'contour': contour_global,
            'center': (int(cx), int(cy)),
            'area': float(area),
            'reason': str(reason),
            'source': source,
        })
        self._debug_rejected_contours = self._debug_rejected_contours[-3:]

    def _draw_rejected_contour_debug(self, frame, scale=1.0):
        rejected_contours = getattr(self, '_debug_rejected_contours', [])
        if not rejected_contours:
            return frame

        result = frame
        for idx, entry in enumerate(rejected_contours):
            contour = entry.get('contour')
            if contour is not None and len(contour) > 0:
                scaled_contour = contour.astype(np.float32).copy()
                scaled_contour[:, 0, 0] *= scale
                scaled_contour[:, 0, 1] *= scale
                scaled_contour = np.round(scaled_contour).astype(np.int32)
                thickness = 2 if idx == (len(rejected_contours) - 1) else 1
                cv2.drawContours(result, [scaled_contour], -1, (0, 0, 255), thickness)

        latest = rejected_contours[-1]
        rx = int(latest['center'][0] * scale)
        ry = int(latest['center'][1] * scale)
        cv2.circle(result, (rx, ry), 12, (0, 0, 255), 2)
        cv2.circle(result, (rx, ry), 2, (0, 0, 255), -1)

        source = latest.get('source')
        source_text = f" [{source}]" if source else ""
        reason = latest.get('reason', 'rejected')
        if len(reason) > 70:
            reason = reason[:67] + "..."
        cv2.putText(result, f"Rejected contour{source_text}: {reason}",
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(result,
                    f"Rejected At: ({latest['center'][0]}, {latest['center'][1]}) area={latest['area']:.1f}px",
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return result

    def draw_analysis_info(self, frame, scale=1.0, show_paused_rejected=False):
        """Draw analysis information on the frame with proper scaling."""
        result = frame.copy()
        
        if self.ball_center:
            # Scale ball coordinates for display
            x = int(self.ball_center[0] * scale)
            y = int(self.ball_center[1] * scale)
            
            # Draw green circle around the ball
            cv2.circle(result, (x, y), 10, (0, 255, 0), 2)  # Circle outline
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)  # Center dot
            
            # Draw info text at the top
            if self.ball_hsv is not None and self.ball_size is not None:
                # Show current HSV values at clicked point
                hsv_text = f"Ball HSV: H={self.ball_hsv[0]}, S={self.ball_hsv[1]}, V={self.ball_hsv[2]}"
                cv2.putText(result, hsv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show HSV filter range
                filter_text = f"Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}"
                cv2.putText(result, filter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show bulb size
                size_text = f"Bulb Size: {self.ball_size:.0f}px"
                cv2.putText(result, size_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show ball position
                pos_text = f"Ball Pos: ({self.ball_center[0]}, {self.ball_center[1]})"
                cv2.putText(result, pos_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        waiting_candidate = getattr(self, 'waiting_serve_candidate', None)
        waiting_candidate_frame = getattr(self, 'waiting_serve_candidate_frame', -1)
        if (waiting_candidate is not None and not self.tracking and
                abs(self.frame_count - waiting_candidate_frame) <= 2):
            sx = int(waiting_candidate[0] * scale)
            sy = int(waiting_candidate[1] * scale)
            cv2.circle(result, (sx, sy), 10, (0, 255, 255), 2)
            cv2.circle(result, (sx, sy), 2, (0, 255, 255), -1)
            serve_pos_text = f"Serve Cand: ({waiting_candidate[0]}, {waiting_candidate[1]})"
            cv2.putText(result, serve_pos_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if getattr(self, 'show_event_markers', False):
            # Draw historical direction-change points, but do not paint over the
            # live ball marker or it looks like the current ball "stopped" there.
            for point in self.direction_change_points:
                if self.ball_center is not None and math.hypot(
                    point[0] - self.ball_center[0],
                    point[1] - self.ball_center[1],
                ) <= 12:
                    continue
                px = int(point[0] * scale)
                py = int(point[1] * scale)
                cv2.circle(result, (px, py), 10, (0, 0, 255), 2)
            # Draw net contact points
            for point in self.net_contact_points:
                px = int(point[0] * scale)
                py = int(point[1] * scale)
                cv2.circle(result, (px, py), 12, (0, 0, 255), 3)
        
        # Draw serve detection area (where toss detection watches) in WAITING_FOR_SERVE
        if hasattr(self, 'serve_area_x_min') and hasattr(self, 'serve_area_y_min'):
            sx1 = int(self.serve_area_x_min * scale)
            sy1 = int(self.serve_area_y_min * scale)
            sx2 = int(self.serve_area_x_max * scale)
            sy2 = int(self.serve_area_y_max * scale)
            # Orange dashed rectangle: draw as two overlapping rects for a dashed look
            cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 128, 255), 2)
            cv2.putText(result, "SERVE ZONE", (sx1 + 4, sy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 2)

        # Show motion metrics and focus-loss status
        if self.last_motion is not None:
            distance = self.last_motion['distance']
            direction_deg = self.last_motion['direction_deg']
            direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
            motion_text = f"Ball Move: {distance:.1f}px | Dir: {direction_text}"
            cv2.putText(result, motion_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.focus_loss_active:
            cv2.putText(result, "FOCUS LOST", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.using_alt2_hsv:
            cv2.putText(result, "USING ALTERNATIVE 2 HSV", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.using_alt_hsv:
            cv2.putText(result, "USING ALTERNATIVE HSV", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if show_paused_rejected:
            result = self._draw_rejected_contour_debug(result, scale=scale)
        
        return result
    
    def print_hsv_table(self):
        """Print the HSV values table."""
        if not self.hsv_table:
            print("No HSV data collected yet.")
            return
        
        print(f"\n=== HSV VALUES TABLE ({len(self.hsv_table)} entries) ===")
        print("Frame | Position | HSV Values | Bulb Size | HSV Range")
        print("-" * 80)
        
        for entry in self.hsv_table:
            hsv_str = f"H={entry['hsv'][0]}, S={entry['hsv'][1]}, V={entry['hsv'][2]}"
            range_str = f"H={entry['hsv_range'][0]}-{entry['hsv_range'][1]}, S={entry['hsv_range'][2]}-{entry['hsv_range'][3]}, V={entry['hsv_range'][4]}-{entry['hsv_range'][5]}"
            print(f"{entry['frame']:5d} | ({entry['position'][0]:3d},{entry['position'][1]:3d}) | {hsv_str} | {entry['bulb_size']:8.1f} | {range_str}")
        
        print("-" * 80)
    
    def detect_serve_position(self, frame):
        """Detect potential serve positions by looking for balls in the serve area."""
        # Check if serve area is configured
        if not hasattr(self, 'serve_area_x_min'):
            return None
        self._last_detected_serve_candidate = None
        
        self._prune_ignored_serve_positions()
        frame_height, frame_width = frame.shape[:2]
        serve_search_x_min = self.serve_area_x_min
        serve_search_x_max = self.serve_area_x_max
        serve_search_y_min = self.serve_area_y_min
        server_center_x = None
        server_x_gate = None

        # Player recognition is disabled, so serve detection uses only the
        # configured serve area and a calibrated top-height cut.
        serve_search_y_max = int(self.serve_area_y_min +
                                 (self.serve_area_y_max - self.serve_area_y_min) * 0.85)
        
        if serve_search_x_max <= serve_search_x_min or serve_search_y_max <= serve_search_y_min:
            return None
        
        # Apply HSV filter to find potential balls
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_serve_gray = self._prev_serve_gray if hasattr(self, '_prev_serve_gray') else None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
        
        # Find contours in the calibrated upper portion of the serve area.
        serve_roi = mask[serve_search_y_min:serve_search_y_max,
                        serve_search_x_min:serve_search_x_max]
        contours, _ = cv2.findContours(serve_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for ball-sized contours. With a wider serve band, prefer blobs that
        # stay near the detected server horizontally while still rewarding size.
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Tennis ball size range: configurable per court (small for far-end, large for near-end)
            if self.serve_ball_size_min <= area <= self.serve_ball_size_max:
                # Calculate center in full frame coordinates
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + serve_search_x_min
                    cy = int(M["m01"] / M["m00"]) + serve_search_y_min
                    ignored = False
                    for entry in self._ignored_serve_positions:
                        ix, iy = entry['pos']
                        if math.hypot(cx - ix, cy - iy) <= entry['radius']:
                            ignored = True
                            break
                    if ignored:
                        continue

                    if server_center_x is not None and server_x_gate is not None:
                        server_dx = abs(cx - server_center_x)
                        if server_dx > server_x_gate:
                            continue

                    frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                    
                    # Only accept balls in the calibrated upper portion of the serve area.
                    if cy < serve_search_y_max:
                        x_penalty = 0.0
                        if server_center_x is not None:
                            x_penalty = abs(cx - server_center_x) * 0.35
                        score = x_penalty - (area * 0.25)
                        motion_delta = None
                        if prev_serve_gray is not None and prev_serve_gray.shape == gray_frame.shape:
                            px1 = max(0, cx - 2)
                            py1 = max(0, cy - 2)
                            px2 = min(frame_width, cx + 3)
                            py2 = min(frame_height, cy + 3)
                            curr_patch = gray_frame[py1:py2, px1:px2]
                            prev_patch = prev_serve_gray[py1:py2, px1:px2]
                            if curr_patch.size and prev_patch.size:
                                motion_delta = float(np.mean(cv2.absdiff(curr_patch, prev_patch)))
                                if motion_delta < 12:
                                    score += 120
                                else:
                                    score -= min(motion_delta, 40.0)
                        if frame0_hotspot is not None and (motion_delta is None or motion_delta < 18.0):
                            continue
                        if score < best_score:
                            best_score = score
                            best_contour = (cx, cy)
                            local_hsv = hsv_frame[cy, cx]
                            self._last_detected_serve_candidate = {
                                'pos': (cx, cy),
                                'area': float(area),
                                'hsv': (int(local_hsv[0]), int(local_hsv[1]), int(local_hsv[2])),
                                'motion_delta': motion_delta,
                            }
        
        self._prev_serve_gray = gray_frame
        return best_contour
    
    def mark_serve_area(self, frame):
        """Mark the serve area with 4 points and save to config."""
        print("\n=== MARKING SERVE AREA ===")
        print("Click 4 points to define the serve area:")
        print("1. Top-left corner of serve area")
        print("2. Top-right corner of serve area") 
        print("3. Bottom-right corner of serve area")
        print("4. Bottom-left corner of serve area")
        print("Press ESC to cancel")
        
        # Create a copy of the frame for marking
        display_frame = frame.copy()
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"{len(points)}", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Mark Serve Area", display_frame)
                print(f"Point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow("Mark Serve Area", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mark Serve Area", mouse_callback)
        
        while len(points) < 4:
            cv2.imshow("Mark Serve Area", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow("Mark Serve Area")
                return False
        
        cv2.destroyWindow("Mark Serve Area")
        
        # Calculate serve area boundaries
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        serve_area_x_min = min(x_coords)
        serve_area_x_max = max(x_coords)
        serve_area_y_min = min(y_coords)
        serve_area_y_max = max(y_coords)
        
        print(f"\nServe area boundaries:")
        print(f"X: {serve_area_x_min} to {serve_area_x_max}")
        print(f"Y: {serve_area_y_min} to {serve_area_y_max}")
        
        # Save to config
        self.serve_area_points = points
        self.serve_area_x_min = serve_area_x_min
        self.serve_area_x_max = serve_area_x_max
        self.serve_area_y_min = serve_area_y_min
        self.serve_area_y_max = serve_area_y_max
        
        # Save to config file
        self.save_serve_area_to_config()
        
        return True
    
    def save_serve_area_to_config(self):
        """Save serve area configuration to hsv_config.json."""
        try:
            # Load existing config
            config = {}
            if os.path.exists('hsv_config.json'):
                with open('hsv_config.json', 'r') as f:
                    config = json.load(f)
            
            # Add serve area data
            config['serve_area_points'] = self.serve_area_points
            config['serve_area_x_min'] = self.serve_area_x_min
            config['serve_area_x_max'] = self.serve_area_x_max
            config['serve_area_y_min'] = self.serve_area_y_min
            config['serve_area_y_max'] = self.serve_area_y_max
            
            # Save updated config
            with open('hsv_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print("Serve area configuration saved to hsv_config.json")
            
        except Exception as e:
            print(f"Error saving serve area config: {e}")
    
    def detect_point_end(self, ball_position, frame):
        """Detect if a point has ended based on ball position and behavior."""
        height, width = frame.shape[:2]
        x, y = ball_position
        recent_ground_bounce = (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 2
        
        # Early serve grace: do not end point in first frames after start_frame
        if self.frame_count <= (self.start_frame + 15):
            return False, "Early-serve grace"
        # If we're waiting near an edge, don't end the point
        if getattr(self, 'edge_wait', False):
            return False, "Edge wait"
        if getattr(self, 'ground_bounce_count', 0) >= 2:
            return True, "Ball bounced twice on court"
        
        # Check if ball is out of court bounds
        if x < 0 or x > width or y < 0 or y > height:
            return True, "Ball out of court bounds"
        
        # Check if ball is in or just above the marked net area.
        # A little extra top margin helps catch real tape clips when the marked
        # net band sits slightly low in the frame.
        net_x_limit = int(width * 0.60)
        net_y_top_margin = 220
        net_y_bottom_margin = 80
        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            expanded_net_zone = (
                (self.net_area_y_min - net_y_top_margin) <= y <= (self.net_area_y_max + net_y_bottom_margin)
                and x < net_x_limit
            )
            core_net_zone = (
                max(0, self.net_area_y_min - 20) <= y <= (self.net_area_y_max + 40)
                and x < net_x_limit
            )
            if expanded_net_zone and not recent_ground_bounce:
                ball_is_fast = self.last_motion and self.last_motion['distance'] > 15
                # Only count "lingering" inside the tighter core net band.
                # A normal descending ball can legitimately pass through the larger
                # expanded zone for many frames without touching the net.
                if core_net_zone and not ball_is_fast:
                    if not hasattr(self, 'net_area_frames'):
                        self.net_area_frames = 0
                    self.net_area_frames += 1
                else:
                    self.net_area_frames = 0
                # Consider it a net hit if it lingers - use longer threshold for fast balls
                net_linger_limit = 8 if ball_is_fast else 6
                if getattr(self, 'net_area_frames', 0) > net_linger_limit:
                    return True, "Ball hit the net"
                # Immediate stop near net
                if self.last_motion and self.last_motion['distance'] < 2.5:
                    # Only if previous frame was also slow (not a tracking glitch)
                    prev_was_slow = self.prev_motion and self.prev_motion['distance'] < 10
                    if prev_was_slow:
                        return True, "Ball hit the net"
                # Direction change with deceleration — classic net hit signature
                if self.prev_motion and self.last_motion:
                    prev_dir = self.prev_motion.get('direction_deg')
                    curr_dir = self.last_motion.get('direction_deg')
                    prev_dist = self.prev_motion.get('distance', 0)
                    curr_dist = self.last_motion.get('distance', 0)
                    prev_dx = self.prev_motion.get('dx', 0)
                    prev_dy = self.prev_motion.get('dy', 0)
                    curr_dx = self.last_motion.get('dx', 0)
                    curr_dy = self.last_motion.get('dy', 0)
                    if prev_dir is not None and curr_dir is not None and prev_dist > 0:
                        delta = abs(curr_dir - prev_dir) % 360
                        angle_diff = min(delta, 360 - delta)
                        speed_ratio = curr_dist / prev_dist if prev_dist else 1.0
                        reversed_y = (
                            prev_dy * curr_dy < 0
                            and abs(prev_dy) >= 3
                            and abs(curr_dy) >= 3
                        )
                        strong_x_reversal = (
                            prev_dx * curr_dx < 0
                            and abs(prev_dx) >= 6
                            and abs(curr_dx) >= 6
                        )
                        direction_reversed = reversed_y or strong_x_reversal
                        slowed_after_impact = curr_dist <= max(8.0, prev_dist * 0.80)
                        if direction_reversed and slowed_after_impact:
                            return True, "Ball hit the net"
                        if angle_diff >= 60 and slowed_after_impact:
                            return True, "Ball hit the net"
                        if angle_diff >= 90 and speed_ratio < 0.6:
                            if prev_dist < 30:
                                return True, "Ball hit the net"
            else:
                # Reset counter if ball is not in net area
                if hasattr(self, 'net_area_frames'):
                    self.net_area_frames = 0
        
        # Check if ball has been stationary for too long (double bounce)
        if hasattr(self, 'ball_velocity_history') and len(self.ball_velocity_history) > 10:
            recent_velocities = self.ball_velocity_history[-10:]
            avg_velocity = sum(recent_velocities) / len(recent_velocities)
            print(f"[BALL_STOPPED_CHECK] f{self.frame_count}: avg_vel={avg_velocity:.1f} hist={[round(v,1) for v in recent_velocities]}")
            if avg_velocity < 5:  # Very slow movement
                print(f"[BALL_STOPPED] f{self.frame_count}: avg_vel={avg_velocity:.1f} < 5 → POINT ENDS")
                return True, "Ball stopped (possible double bounce)"
        
        # Check if ball is near court edges (likely out)
        edge_margin = 50
        if (x < edge_margin or x > width - edge_margin or 
            y < edge_margin or y > height - edge_margin):
            if getattr(self, 'edge_wait', False) or getattr(self, 'near_edge', False):
                return False, "Edge return grace"
            self.edge_wait = True
            return False, "Edge return grace"
        
        return False, "Point continues"
    
    def _open_serve_area_hsv_tuner(self, frame):
        """Open HSV tuner specifically for the serve area."""
        print("Opening HSV tuner for SERVE AREA...")
        
        # Extract serve area from frame
        serve_roi = frame[self.serve_area_y_min:self.serve_area_y_max, 
                         self.serve_area_x_min:self.serve_area_x_max]
        
        # Create tuner window
        tuner_window = "SERVE AREA HSV TUNER"
        cv2.namedWindow(tuner_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tuner_window, 800, 600)
        
        # Initialize trackbars with current HSV values
        cv2.createTrackbar("H Min", tuner_window, int(self.hsv_lower[0]), 179, lambda x: None)
        cv2.createTrackbar("H Max", tuner_window, int(self.hsv_upper[0]), 179, lambda x: None)
        cv2.createTrackbar("S Min", tuner_window, int(self.hsv_lower[1]), 255, lambda x: None)
        cv2.createTrackbar("S Max", tuner_window, int(self.hsv_upper[1]), 255, lambda x: None)
        cv2.createTrackbar("V Min", tuner_window, int(self.hsv_lower[2]), 255, lambda x: None)
        cv2.createTrackbar("V Max", tuner_window, int(self.hsv_upper[2]), 255, lambda x: None)
        cv2.createTrackbar("Bulb Min", tuner_window, 2, 50, lambda x: None)
        cv2.createTrackbar("Bulb Max", tuner_window, 80, 200, lambda x: None)
        
        print("Adjust HSV values to detect balls in the serve area")
        print("Press 'Q' or ESC to close and save settings")
        
        while True:
            # Get trackbar values
            h_min = cv2.getTrackbarPos("H Min", tuner_window)
            h_max = cv2.getTrackbarPos("H Max", tuner_window)
            s_min = cv2.getTrackbarPos("S Min", tuner_window)
            s_max = cv2.getTrackbarPos("S Max", tuner_window)
            v_min = cv2.getTrackbarPos("V Min", tuner_window)
            v_max = cv2.getTrackbarPos("V Max", tuner_window)
            bulb_min = cv2.getTrackbarPos("Bulb Min", tuner_window)
            bulb_max = cv2.getTrackbarPos("Bulb Max", tuner_window)
            
            # Apply HSV filter to serve area
            hsv_serve = cv2.cvtColor(serve_roi, cv2.COLOR_BGR2HSV)
            hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv_serve, hsv_lower, hsv_upper)
            
            # Find contours in serve area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw detected balls on serve area
            display_serve = serve_roi.copy()
            ball_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if bulb_min < area < bulb_max:
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw circle and size label
                        cv2.circle(display_serve, (cx, cy), 8, (0, 255, 0), 2)
                        cv2.putText(display_serve, f"{area:.0f}", (cx + 12, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        ball_count += 1
            
            # Create side-by-side display
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([display_serve, mask_bgr])
            
            # Add text info
            info_text = f"SERVE AREA | Balls: {ball_count} | HSV: H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max} | Size: {bulb_min}-{bulb_max}px"
            cv2.putText(combined, info_text, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(tuner_window, combined)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        # Update HSV values with tuned values
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        
        # Save updated HSV values to config
        self.save_hsv_to_config()
        
        cv2.destroyWindow(tuner_window)
        print("Serve area HSV tuner closed. Settings saved.")
    
    def save_hsv_to_config(self):
        """Save current HSV values to config file."""
        try:
            # Load existing config
            config = {}
            if os.path.exists('hsv_config.json'):
                with open('hsv_config.json', 'r') as f:
                    config = json.load(f)
            
            # Update HSV values
            if "regular_court" in config:
                # Update regular court HSV values
                config["regular_court"]["h_min"] = int(self.hsv_lower[0])
                config["regular_court"]["h_max"] = int(self.hsv_upper[0])
                config["regular_court"]["s_min"] = int(self.hsv_lower[1])
                config["regular_court"]["s_max"] = int(self.hsv_upper[1])
                config["regular_court"]["v_min"] = int(self.hsv_lower[2])
                config["regular_court"]["v_max"] = int(self.hsv_upper[2])
            else:
                # Old format
                config["h_min"] = int(self.hsv_lower[0])
                config["h_max"] = int(self.hsv_upper[0])
                config["s_min"] = int(self.hsv_lower[1])
                config["s_max"] = int(self.hsv_upper[1])
                config["v_min"] = int(self.hsv_lower[2])
                config["v_max"] = int(self.hsv_upper[2])
            
            # Save updated config
            with open('hsv_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print("HSV values saved to hsv_config.json")
            
        except Exception as e:
            print(f"Error saving HSV config: {e}")
    
    def process_video(self, auto_play=False, max_frames=0):
        """Process video with intelligent tennis game analysis."""
        self.auto_play = auto_play  # store so detect_ball can skip GUI debug in batch mode
        print("Intelligent Tennis Game Tracker")
        print("=" * 50)
        print("1. Waits for first serve position")
        print("2. Tracks ball through each point")
        print("3. Detects when point ends (out, net, double bounce)")
        print("4. Automatically waits for next serve")
        print("5. Press 'D' to advance frame by frame")
        print("6. Press 'N' to mark net area (6 points clockwise)")
        print("7. Press 'S' to mark serve area (4 points)")
        print("8. Press 'B' to open ball HSV debug window")
        print("9. Press SPACE to play/pause")
        print("10. Press BACKSPACE to step back one frame")
        print("11. Press 0/1/2/3 for HSV debug on previous frame (standard/alt1/alt2/alt3)")
        print("12. Press 'Q' to quit")
        print("=" * 50)
        if self.start_frame > 0:
            print(f"Starting at frame {self.start_frame}")
        
        if not auto_play:
            cv2.namedWindow("Tennis Game Tracker", cv2.WINDOW_NORMAL)
        
        # Store scale factor for mouse coordinate conversion
        scale_factor = 1.0
        current_frame = None
        
        def reset_tracking_state():
            self.tracking = False
            self.ball_stopped = False
            self.ball_center = None
            self.ball_size = None
            self.ball_hsv = None
            self.ball_velocity_history = []
            self.last_motion = None
            self.prev_motion = None
            self.last_direction = None
            self.direction_change_streak = 0
            self.edge_wait = False
            self.near_edge = False
            self.using_alt_hsv = False
            self.using_alt2_hsv = False
            self.focus_loss_active = False
            self.stuck_frame_count = 0
            self.point_start_frame_internal = None
            self._serve_contact_grace_frames = 0
            self._rally_contact_grace_frames = 0
            self._rally_contact_ref_size = None
            self._rally_contact_origin = None
            self._rally_contact_expected = None
            self._rally_contact_progress = 0.0
            self._ground_bounce_grace_frames = 0
            self._ground_bounce_ref_size = None
            self._ground_bounce_origin = None
            self._ground_bounce_expected = None
            self._ground_bounce_progress = 0.0
            self.ground_bounce_count = 0
            self.last_ground_bounce_frame = -1000000
            self._contact_recovery_frames = 0
            self._upper_exit_wait_frames = 0
            self._prev_serve_gray = None
            self._ignored_serve_positions = []
            self.waiting_serve_candidate = None
            self.waiting_serve_candidate_frame = -1
            self._last_detected_serve_candidate = None
            if hasattr(self, 'net_area_frames'):
                self.net_area_frames = 0

        # Game state variables
        game_state = "WAITING_FOR_SERVE"  # SCANNING_FOR_SERVE, TRACKING_POINT, POINT_ENDED, WAITING_FOR_SERVE
        point_start_frame = None
        point_end_frame = None
        serve_positions = []
        serve_tracking_frames = 0
        last_serve_candidate = None
        serve_position_history = []
        serve_candidate_details_history = []
        scan_position_history = []  # motion history for SCANNING_FOR_SERVE rightward check
        play_mode = auto_play  # start playing immediately if --auto-play flag is set
        last_frame_for_debug = None
        last_ball_center_for_debug = None
        last_frame_index_for_debug = None
        # Ensure capture starts at requested frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        early_serve_grace_frames = 15  # aggressively accept serve during first frames after start_frame

        def seed_tracking_from_serve_history(target_pos):
            nonlocal serve_candidate_details_history
            if target_pos is None or not serve_candidate_details_history:
                return
            best_detail = None
            best_dist = float('inf')
            for detail in reversed(serve_candidate_details_history):
                pos = detail.get('pos') if detail else None
                if pos is None:
                    continue
                dist = math.hypot(pos[0] - target_pos[0], pos[1] - target_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_detail = detail
                    if dist <= 2:
                        break
            if best_detail is None or best_dist > 80:
                return
            area = best_detail.get('area')
            hsv = best_detail.get('hsv')
            if area is not None:
                self.ball_size = float(area)
            if hsv is not None:
                self.ball_hsv = np.array(hsv, dtype=np.uint8)

        def clear_waiting_serve_history():
            nonlocal serve_tracking_frames, last_serve_candidate
            nonlocal serve_position_history, serve_candidate_details_history
            serve_tracking_frames = 0
            last_serve_candidate = None
            serve_position_history = []
            serve_candidate_details_history = []
        
        # Modified mouse callback that converts resized coordinates to original
        def mouse_callback_with_scale(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert clicked coordinates from resized frame to original frame
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                # Analyze ball at original coordinates
                self.analyze_ball_at_point(current_frame, (orig_x, orig_y))
        
        if not auto_play:
            cv2.setMouseCallback("Tennis Game Tracker", mouse_callback_with_scale)
        
        # Start from configured frame to wait for first serve
        print(f"\nStarting from frame {self.start_frame} to wait for first serve...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.frame_count = self.start_frame
        
        # Load HSV values from config file (or use defaults if not present)
        hsv_config = self.load_hsv_config()
        
        # Check if net area needs to be marked (first time setup)
        if hsv_config and "net_area_y_min" not in hsv_config:
            print("\n" + "=" * 70)
            print("FIRST TIME SETUP: Please mark the net area")
            print("=" * 70)
            ret, frame = self.cap.read()
            if ret:
                if self.mark_net_area(frame):
                    print("Net area saved! This will be used for all future runs.")
                    # Reload config after marking net area
                    hsv_config = self.load_hsv_config()
                else:
                    print("Net area marking skipped. Using default values.")
        
        # Check if serve area needs to be marked (first time setup)
        if hsv_config and "serve_area_points" not in hsv_config:
            print("\n" + "=" * 70)
            print("FIRST TIME SETUP: Please mark the serve area")
            print("=" * 70)
            ret, frame = self.cap.read()
            if ret:
                if self.mark_serve_area(frame):
                    print("Serve area saved! This will be used for all future runs.")
                    # Reload config after marking serve area
                    hsv_config = self.load_hsv_config()
                else:
                    print("Serve area marking skipped. Using default values.")
        
        # Initialize HSV values
        if not hsv_config:
            # Default values if config not found
            self.hsv_lower = np.array([20, 20, 70], dtype=np.uint8)
            self.hsv_upper = np.array([90, 255, 255], dtype=np.uint8)
        # If old format, set hsv_lower and hsv_upper (already done in load_hsv_config for new format)
        elif 'h_min' in hsv_config:
            self.hsv_lower = np.array([hsv_config['h_min'], hsv_config['s_min'], hsv_config['v_min']], dtype=np.uint8)
            self.hsv_upper = np.array([hsv_config['h_max'], hsv_config['s_max'], hsv_config['v_max']], dtype=np.uint8)
        # New format already loaded in load_hsv_config, just use the regular court as default
        
        print(f"\nHSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
        print(f"Game State: {game_state}")
        print(f"Press 'D' to advance frame by frame")
        
        _consecutive_read_failures = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # H.265/HEVC videos sometimes have corrupt slices mid-file.
                # Try to skip up to 10 bad frames before giving up.
                _consecutive_read_failures += 1
                if _consecutive_read_failures <= 10:
                    print(f"[SKIP_BAD_FRAME] f{self.frame_count}: read failed, skipping (attempt {_consecutive_read_failures}/10)")
                    self.frame_count += 1
                    continue
                print(f"[VIDEO_END] Stopped after {_consecutive_read_failures} consecutive read failures at f{self.frame_count}")
                break
            _consecutive_read_failures = 0  # reset on successful read
            if max_frames > 0 and (self.frame_count - self.start_frame) >= max_frames:
                print(f"[MAX_FRAMES] Reached {max_frames} frames limit, stopping.")
                break
            self._debug_contour_candidates = []
            self._debug_rejected_contours = []

            # Early serve detection: while waiting and within grace window, attempt ball track and enter tracking
            if game_state == "WAITING_FOR_SERVE" and self.frame_count <= (self.start_frame + early_serve_grace_frames):
                self.using_alt_hsv = False
                self.using_alt2_hsv = False
                candidate = self.track_ball_in_frame(frame, allow_inactive=True)
                if candidate is not None:
                    if hasattr(self, 'serve_area_x_min'):
                        if not (self.serve_area_x_min <= candidate[0] <= self.serve_area_x_max and
                                self.serve_area_y_min <= candidate[1] <= self.serve_area_y_max):
                            print(f"Frame {self.frame_count}: Early serve candidate outside serve area, ignoring")
                            candidate = None
                    if candidate is not None:
                        serve_tracking_frames += 1
                        last_serve_candidate = candidate
                        self.waiting_serve_candidate = candidate
                        self.waiting_serve_candidate_frame = self.frame_count
                        print(f"Frame {self.frame_count}: Serve candidate {candidate} (holding for confirmation)")
                else:
                    if serve_tracking_frames >= 5 and last_serve_candidate is not None:
                        print(f"Frame {self.frame_count}: Serve exited area, entering TRACKING_POINT from {last_serve_candidate}")
                        game_state = "TRACKING_POINT"
                        point_start_frame = self.frame_count
                        self.point_start_frame_internal = self.frame_count
                        self.tracking = True
                        self.ball_center = last_serve_candidate
                        self.waiting_serve_candidate = None
                        self.waiting_serve_candidate_frame = -1
                    else:
                        serve_tracking_frames = 0
                        last_serve_candidate = None
            
            self.frame_count += 1
            current_frame = frame  # Store current frame for mouse callback
            
            # Handle different game states
            if game_state == "SCANNING_FOR_SERVE":
                # Look for potential serve positions (ball in serve area)
                # Require consistent motion in the configured serve direction before starting.
                # A single detection (ball just sitting in serve area) must NOT trigger tracking.
                potential_serve = self.detect_serve_position(frame)
                if potential_serve:
                    scan_position_history.append(potential_serve)
                    if len(scan_position_history) > 10:
                        scan_position_history = scan_position_history[-10:]
                    if len(scan_position_history) >= 4:
                        all_forward = True
                        min_signed_dx = float('inf')
                        for i in range(-3, 0):
                            pair_dx = scan_position_history[i][0] - scan_position_history[i-1][0]
                            signed_dx = self._signed_serve_dx(pair_dx)
                            if signed_dx < 15:
                                all_forward = False
                                break
                            min_signed_dx = min(min_signed_dx, signed_dx)
                        if all_forward and min_signed_dx > 25:
                            print(f"\n{'='*70}")
                            print(f"SERVE DETECTED at frame {self.frame_count}!")
                            print(f"Ball position: {potential_serve}")
                            print(f"Starting to track the ball...")
                            print(f"{'='*70}\n")
                            print(f"[TRACKING_START] f{self.frame_count}: serve detected at {potential_serve}")
                            serve_positions.append((self.frame_count, potential_serve))
                            p1 = scan_position_history[-2]
                            p2 = scan_position_history[-1]
                            _dx = p2[0] - p1[0]
                            _dy = p2[1] - p1[1]
                            _dist = math.hypot(_dx, _dy)
                            _dir = math.degrees(math.atan2(_dy, _dx))
                            self.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                            self.last_delta = (_dx, _dy)
                            self.ball_velocity_history = [_dist]
                            self.ball_center = potential_serve
                            self.tracking = True
                            self.ball_stopped = False
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 18)
                            self.initial_ball_position = scan_position_history[0]
                            self.ball_size = None
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            scan_position_history = []
                            game_state = "TRACKING_POINT"
                else:
                    scan_position_history = []
            
            elif game_state == "TRACKING_POINT":
                # Point duration timeout: if we stay in TRACKING_POINT too long without
                # a natural end condition, the ball is truly lost (e.g. false-positive at
                # edge, or post-strike tracking failure on near-end court).
                # Default: 500 frames (~20 s at 25 fps).  Override via config max_point_frames.
                _max_point_frames = getattr(self, 'max_point_frames', 500)
                if point_start_frame and (self.frame_count - point_start_frame) > _max_point_frames:
                    dur = self.frame_count - point_start_frame
                    print(f"[POINT_END] f{self.frame_count}: reason=POINT_TIMEOUT duration={dur}f — returning to serve detection")
                    game_state = "WAITING_FOR_SERVE"
                    reset_tracking_state()

                # Track ball through the point
                tracked_position = None
                if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
                    prev_ball_center = self.ball_center
                    prev_stuck = self.stuck_frame_count
                    tracked_position = self.track_ball_in_frame(frame)
                    # Reject any position that jumps impossibly far in one frame (false positive).
                    # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
                    # a larger jump because the ball may have traveled far while lost.
                    if tracked_position and prev_ball_center:
                        jump = math.hypot(tracked_position[0] - prev_ball_center[0],
                                          tracked_position[1] - prev_ball_center[1])
                        # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
                        # a racket hit can send the ball 800+ px in one frame, so we use 1500px
                        # to let motion-based re-acquisition recover across the full court.
                        max_jump = 1500 if prev_stuck >= 5 else 400
                        if jump > max_jump:
                            print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
                            self.ball_center = prev_ball_center
                            tracked_position = None
                if tracked_position:
                    vel = self.last_motion['distance'] if self.last_motion else 0
                    size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
                    vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
                    print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
                    print(f"[TRACK] f{self.frame_count}: pos={tracked_position} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")

                    # Stuck-ball timeout: if ball hasn't moved for 15+ frames, end point
                    if self.stuck_frame_count >= 15:
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        print(f"Frame {self.frame_count}: POINT ENDED - Ball stuck for {self.stuck_frame_count} frames")
                        print(f"Point duration: {dur} frames")
                        print(f"[POINT_END] f{self.frame_count}: reason=STUCK_TIMEOUT stuck={self.stuck_frame_count} duration={dur}f pos={tracked_position}")
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state()
                    else:
                        # Check if point has ended
                        point_ended, reason = self.detect_point_end(tracked_position, frame)
                        if point_ended:
                            point_end_frame = self.frame_count
                            dur = point_end_frame - point_start_frame if point_start_frame else 0
                            print(f"Frame {self.frame_count}: POINT ENDED - {reason}")
                            print(f"Point duration: {dur} frames")
                            print(f"[POINT_END] f{self.frame_count}: reason={reason} duration={dur}f pos={tracked_position} vel={vel:.1f}px vel_hist={vel_hist_tail}")
                            wait_for_next_serve = any(
                                token in reason.lower()
                                for token in ("net", "bounce", "stopped", "lost")
                            )
                            if wait_for_next_serve:
                                if "net" in reason.lower():
                                    self.net_contact_points.append(tracked_position)
                                game_state = "WAITING_FOR_SERVE"
                            else:
                                game_state = "POINT_ENDED"
                            reset_tracking_state()
                        else:
                            print(f"Frame {self.frame_count}: Ball tracking continued")
                else:
                    # Ball lost - might be end of point
                    grace_limit = 45 if point_start_frame and self.frame_count <= (self.start_frame + 45) else 30
                    if point_start_frame and (self.frame_count - point_start_frame > grace_limit):
                        point_end_frame = self.frame_count
                        print(f"Frame {self.frame_count}: POINT ENDED - Ball lost (likely out of court)")
                        print(f"Point duration: {point_end_frame - point_start_frame} frames")
                        if not getattr(self, 'auto_play', False):
                            predicted_point = self.get_predicted_point() or self.ball_center
                            if predicted_point is not None:
                                print("  DEBUG: Point ended. Opening HSV debug windows at predicted position...")
                                self.open_predicted_hsv_debug_all(frame, predicted_point, self.frame_count)
                                self.pause_requested = True
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state()
            
            elif game_state == "POINT_ENDED":
                # Wait a few frames then start scanning for next serve
                if self.frame_count - point_end_frame > 60:  # Wait 2 seconds (60 frames at 30fps)
                    print(f"Frame {self.frame_count}: Starting to scan for next serve...")
                    game_state = "SCANNING_FOR_SERVE"
                    point_start_frame = None
                    point_end_frame = None
                # If we are still within early grace and see the ball, resume tracking
                elif (self.frame_count <= (self.start_frame + 15)) or getattr(self, 'edge_wait', False) or (self.last_seen_frame and self.frame_count - self.last_seen_frame <= 200):
                    candidate = self.track_ball_in_frame(frame, allow_inactive=True)
                    if candidate is not None:
                        print(f"Frame {self.frame_count}: Ball re-found during grace, resuming TRACKING_POINT at {candidate}")
                        game_state = "TRACKING_POINT"
                        point_start_frame = self.frame_count
                        self.tracking = True
        
            elif game_state == "WAITING_FOR_SERVE":
                # Detect ball in serve area, accumulate position history,
                # start tracking only when ball exits serve area in the configured serve direction
                import math as _math
                potential_serve = self.detect_serve_position(frame)
                if potential_serve:
                    self.waiting_serve_candidate = potential_serve
                    self.waiting_serve_candidate_frame = self.frame_count
                    if last_serve_candidate is not None:
                        candidate_jump = _math.hypot(
                            potential_serve[0] - last_serve_candidate[0],
                            potential_serve[1] - last_serve_candidate[1],
                        )
                        candidate_jump_reset = max(
                            220,
                            int((self.serve_area_x_max - self.serve_area_x_min) * 0.12),
                        )
                        if candidate_jump > candidate_jump_reset:
                            print(f"[SERVE_CANDIDATE_RESET] f{self.frame_count}: jump={candidate_jump:.1f}px "
                                  f"from {last_serve_candidate} to {potential_serve} — restarting serve history")
                            clear_waiting_serve_history()
                    serve_tracking_frames += 1
                    last_serve_candidate = potential_serve
                    serve_position_history.append(potential_serve)
                    current_serve_detail = self._last_detected_serve_candidate
                    if current_serve_detail is not None:
                        serve_candidate_details_history.append(dict(current_serve_detail))
                    else:
                        serve_candidate_details_history.append({'pos': potential_serve})
                    if len(serve_position_history) > 20:
                        serve_position_history = serve_position_history[-20:]
                    if len(serve_candidate_details_history) > 20:
                        serve_candidate_details_history = serve_candidate_details_history[-20:]
                    # Static false-positive filter: a real toss ball moves significantly
                    # through the serve area; a static artifact (line, shadow, court marking)
                    # stays at nearly the same pixel for many frames.
                    # After 8 frames with total displacement < 10px → reset and ignore.
                    if serve_tracking_frames >= 15 and len(serve_position_history) >= 4:
                        xs = [p[0] for p in serve_position_history]
                        ys = [p[1] for p in serve_position_history]
                        total_disp = _math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                        if total_disp < 10:
                            print(f"[SERVE_FP_RESET] f{self.frame_count}: static false positive at {potential_serve} "
                                  f"total_disp={total_disp:.1f}px over {serve_tracking_frames}f — resetting")
                            self._ignored_serve_positions.append({
                                'pos': potential_serve,
                                'radius': 45,
                                'expires': self.frame_count + 120,
                            })
                            clear_waiting_serve_history()
                    # Toss-in-flight: when a persistent false positive keeps the ball "in"
                    # the serve area for far longer than a normal serve (3-8 frames), but
                    # the position history shows the real ball was briefly near the top of
                    # the serve area (toss apex), trigger tracking from that top position.
                    # This handles the cold-start case (--start-frame N) where the player
                    # detector hasn't warmed up yet and allows a wider serve search Y range
                    # that catches a persistent false positive at the top of the search box.
                    # serve_tracking_frames >= 10 ensures normal serves (3-8 frames) are
                    # never affected by this path.
                    if serve_tracking_frames >= 10 and len(serve_position_history) >= 3:
                        toss_high_y = self.serve_area_y_min + int(
                            (self.serve_area_y_max - self.serve_area_y_min) * 0.45
                        )
                        recent_steps = [
                            _math.hypot(
                                serve_position_history[i][0] - serve_position_history[i - 1][0],
                                serve_position_history[i][1] - serve_position_history[i - 1][1],
                            )
                            for i in range(1, len(serve_position_history))
                        ]
                        recent_step_max = max(recent_steps[-4:], default=0.0)
                        recent_min_y = min(p[1] for p in serve_position_history)
                        recent_max_y = max(p[1] for p in serve_position_history)
                        total_disp = _math.hypot(
                            max(p[0] for p in serve_position_history) - min(p[0] for p in serve_position_history),
                            recent_max_y - recent_min_y
                        )
                        toss_drop_threshold = max(50, int((self.serve_area_y_max - self.serve_area_y_min) * 0.12))
                        toss_speed_threshold = max(18, int((self.serve_area_y_max - self.serve_area_y_min) * 0.05))
                        apex_index = min(
                            range(len(serve_position_history)),
                            key=lambda idx: serve_position_history[idx][1]
                        )
                        pre_apex_positions = serve_position_history[:apex_index + 1]
                        pre_apex_rise = 0.0
                        pre_apex_up_steps = []
                        if pre_apex_positions:
                            pre_apex_rise = max(p[1] for p in pre_apex_positions) - recent_min_y
                        if len(pre_apex_positions) >= 2:
                            pre_apex_up_steps = [
                                self._signed_serve_dy(pre_apex_positions[i][1] - pre_apex_positions[i - 1][1])
                                for i in range(1, len(pre_apex_positions))
                            ]
                        pre_apex_rise_threshold = max(90, int((self.serve_area_y_max - self.serve_area_y_min) * 0.20))
                        strong_up_steps = [step for step in pre_apex_up_steps if step >= 12]
                        has_real_toss_rise = (
                            pre_apex_rise >= pre_apex_rise_threshold and
                            len(strong_up_steps) >= 3
                        )
                        vertical_drop = potential_serve[1] - recent_min_y
                        if (recent_min_y < toss_high_y and vertical_drop >= toss_drop_threshold
                                and total_disp >= toss_drop_threshold
                                and recent_step_max >= toss_speed_threshold
                                and has_real_toss_rise):
                            # Find the most recent position in the top portion (actual toss
                            # ball, not the false positive which sits at a higher Y value)
                            top_positions = [p for p in serve_position_history if p[1] < toss_high_y]
                            toss_start = top_positions[-1] if top_positions else last_serve_candidate
                            _dx = potential_serve[0] - toss_start[0]
                            _dy = potential_serve[1] - toss_start[1]
                            _dist = _math.hypot(_dx, _dy)
                            _dir = _math.degrees(_math.atan2(_dy, _dx))
                            self.last_motion = {
                                'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir
                            }
                            self.last_delta = (_dx, _dy)
                            self.ball_velocity_history = [_dist]
                            print(f"[TRACKING_START] f{self.frame_count}: toss-in-flight detected at {toss_start} "
                                  f"serve_tracking_frames={serve_tracking_frames} recent_min_y={recent_min_y} "
                                  f"vertical_drop={vertical_drop:.0f}px total_disp={total_disp:.0f}px "
                                  f"recent_step_max={recent_step_max:.1f}px toss_high_y={toss_high_y}")
                            self.ball_center = toss_start
                            self.tracking = True
                            self.ball_stopped = False
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 6)
                            self.initial_ball_position = serve_position_history[0]
                            self.ball_size = None
                            self.ball_hsv = None
                            seed_tracking_from_serve_history(last_serve_candidate)
                            self.stuck_frame_count = 0
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            clear_waiting_serve_history()
                    # Early serve start: if ball is moving fast in the configured serve direction within the serve area,
                    # start tracking immediately (don't wait for serve area exit)
                    # Require ALL consecutive pairs to move in the serve direction (not just first-to-last)
                    if len(serve_position_history) >= 4:
                        # Check last 3 consecutive pairs all move forward with signed dx > 15
                        all_forward = True
                        min_signed_dx = float('inf')
                        for i in range(-3, 0):
                            pair_dx = serve_position_history[i][0] - serve_position_history[i-1][0]
                            signed_dx = self._signed_serve_dx(pair_dx)
                            if signed_dx < 15:
                                all_forward = False
                                break
                            min_signed_dx = min(min_signed_dx, signed_dx)
                        avg_dx = min_signed_dx  # Use minimum signed dx as the threshold
                        if all_forward and avg_dx >= 25:
                            predicted_pos = potential_serve
                            if len(serve_position_history) >= 2:
                                p1 = serve_position_history[-2]
                                p2 = serve_position_history[-1]
                                _dx = p2[0] - p1[0]
                                _dy = p2[1] - p1[1]
                                _dist = _math.hypot(_dx, _dy)
                                _dir = _math.degrees(_math.atan2(_dy, _dx))
                                self.last_motion = {
                                    'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir
                                }
                                self.last_delta = (_dx, _dy)
                                self.ball_velocity_history = [_dist]
                            print(f"\n{'='*70}")
                            print(f"SERVE IN PROGRESS at frame {self.frame_count}!")
                            print(f"Ball position: {potential_serve}, signed dx={avg_dx:.1f}px/frame")
                            print(f"{'='*70}\n")
                            print(f"[TRACKING_START] f{self.frame_count}: fast-serve detected at {potential_serve} signed_dx={avg_dx:.1f}px")
                            self.ball_center = potential_serve
                            self.tracking = True
                            self.ball_stopped = False
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 6)
                            self.initial_ball_position = serve_position_history[0]
                            self.ball_size = None
                            self.ball_hsv = None
                            seed_tracking_from_serve_history(potential_serve)
                            self.stuck_frame_count = 0
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            clear_waiting_serve_history()
                else:
                    # Ball exited serve area — only start tracking if it was moving at serve speed
                    # in the configured serve direction.
                    # A ball just sitting/bouncing in the area (false positive like f492) has near-zero speed.
                    if serve_tracking_frames >= 3 and last_serve_candidate is not None and len(serve_position_history) >= 2:
                        total_dx = serve_position_history[-1][0] - serve_position_history[0][0]
                        last_dx = serve_position_history[-1][0] - serve_position_history[-2][0]
                        last_dy = serve_position_history[-1][1] - serve_position_history[-2][1]
                        print(f"[SERVE_EXIT_CHECK] f{self.frame_count}: tracked {serve_tracking_frames}f, total_dx={total_dx:.0f}px, last_dx={last_dx:.0f}px, last_dy={last_dy:.0f}px")
                        signed_total_dx = self._signed_serve_dx(total_dx)
                        signed_last_dx = self._signed_serve_dx(last_dx)
                        signed_last_dy = self._signed_serve_dy(last_dy) if self.serve_direction_dy != 0 else last_dy
                        # Reject toss motion that still moves strongly opposite the configured serve launch.
                        if signed_last_dx > 20 and (self.serve_direction_dy == 0 or signed_last_dy >= -signed_last_dx):
                            predicted_pos = last_serve_candidate
                            if len(serve_position_history) >= 2:
                                p1 = serve_position_history[-2]
                                p2 = serve_position_history[-1]
                                _dx = p2[0] - p1[0]
                                _dy = p2[1] - p1[1]
                                _dist = _math.hypot(_dx, _dy)
                                _dir = _math.degrees(_math.atan2(_dy, _dx))
                                raw_pred = (int(p2[0] + _dx), int(p2[1] + _dy))
                                # Clamp predicted position to within frame bounds so edge-wait
                                # mode does not trap the tracker off-screen forever.
                                frame_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 2160
                                frame_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 3840
                                predicted_pos = (
                                    int(max(0, min(frame_w - 1, raw_pred[0]))),
                                    int(max(0, min(frame_h - 1, raw_pred[1]))),
                                )
                                self.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                                self.last_delta = (_dx, _dy)
                                self.ball_velocity_history = [_dist]
                            print(f"[TRACKING_START] f{self.frame_count}: serve-exit detected at {last_serve_candidate} "
                                  f"signed_total_dx={signed_total_dx:.0f}px signed_last_dx={signed_last_dx:.0f}px, predicted={predicted_pos}")
                            self.ball_center = predicted_pos
                            self.tracking = True
                            self.ball_stopped = False
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 6)
                            self.initial_ball_position = last_serve_candidate
                            self.ball_size = None
                            self.ball_hsv = None
                            seed_tracking_from_serve_history(last_serve_candidate)
                            self.stuck_frame_count = 0
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                        elif last_dy > 5 and last_dx >= -10 and serve_tracking_frames >= 5:
                            # Toss-complete: ball exiting serve area downward (falling after toss).
                            # Player is about to strike the ball just below the serve area.
                            # Start tracking from last known position — the 200px search radius
                            # will find the ball at the strike point.
                            p1 = serve_position_history[-2]
                            p2 = serve_position_history[-1]
                            _dx = p2[0] - p1[0]
                            _dy = p2[1] - p1[1]
                            _dist = _math.hypot(_dx, _dy)
                            _dir = _math.degrees(_math.atan2(_dy, _dx))
                            self.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                            self.last_delta = (_dx, _dy)
                            self.ball_velocity_history = [_dist]
                            print(f"[TRACKING_START] f{self.frame_count}: toss-complete detected at {last_serve_candidate} serve_tracking_frames={serve_tracking_frames} last_dy={last_dy:.0f}px")
                            self.ball_center = last_serve_candidate
                            self.tracking = True
                            self.ball_stopped = False
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 6)
                            self.initial_ball_position = serve_position_history[0]
                            self.ball_size = None
                            self.ball_hsv = None
                            seed_tracking_from_serve_history(last_serve_candidate)
                            self.stuck_frame_count = 0
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                    clear_waiting_serve_history()
            
            # Resize frame to fit screen
            height, width = frame.shape[:2]
            scale_factor = min(1920/width, 1080/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # In auto_play (headless batch) mode skip all GUI rendering — cv2.imshow/waitKey can
            # freeze when no display message pump is available (background subprocess on Windows).
            if not getattr(self, 'auto_play', False):
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Draw analysis info with scale factor
                display_frame = self.draw_analysis_info(
                    resized_frame,
                    scale=scale_factor,
                    show_paused_rejected=((not play_mode) or self.pause_requested),
                )

                # Show frame info
                cv2.putText(display_frame, f"Frame: {self.frame_count}/{self.total_frames}",
                           (10, new_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show game state and controls
                if game_state == "SCANNING_FOR_SERVE":
                    cv2.putText(display_frame, f"SCANNING FOR SERVE | D=Next | N=Net | S=Serve | Q=Quit",
                               (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif game_state == "TRACKING_POINT":
                    cv2.putText(display_frame, f"TRACKING POINT (started frame {point_start_frame}) | D=Next | N=Net | S=Serve | Q=Quit",
                               (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif game_state == "POINT_ENDED":
                    cv2.putText(display_frame, f"POINT ENDED (waiting for next serve) | D=Next | N=Net | S=Serve | Q=Quit",
                               (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(display_frame, f"Game State: {game_state} | D=Next | N=Net | S=Serve | Q=Quit",
                               (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # C-key contour debug overlay: draw candidate index + area on screen
                if getattr(self, 'show_contour_debug', False) and hasattr(self, '_debug_contour_candidates'):
                    for idx, cand in enumerate(self._debug_contour_candidates):
                        _, _, ccx, ccy, carea, cdist, _, cscore = cand
                        sx = int(ccx * scale_factor)
                        sy = int(ccy * scale_factor)
                        # Yellow circle + index number + area
                        cv2.circle(display_frame, (sx, sy), 10, (0, 255, 255), 1)
                        cv2.putText(display_frame, str(idx),
                                    (sx + 6, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(display_frame, f"a={carea:.0f} s={cscore:.0f}",
                                    (sx + 6, sy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

                cv2.imshow("Tennis Game Tracker", display_frame)

            prev_frame_for_debug = last_frame_for_debug
            prev_ball_center_for_debug = last_ball_center_for_debug
            prev_frame_index_for_debug = last_frame_index_for_debug
            last_frame_for_debug = frame.copy()
            last_ball_center_for_debug = self.ball_center
            last_frame_index_for_debug = self.frame_count

            if not getattr(self, 'auto_play', False):
                if self.pause_requested:
                    play_mode = False
                key = cv2.waitKey(30 if play_mode else 0) & 0xFF
            else:
                key = 0xFF  # no-op key in headless auto_play mode
            if key == ord('q'):
                break
            elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
                self.open_prev_frame_hsv_debug(
                    prev_frame_for_debug,
                    prev_ball_center_for_debug,
                    prev_frame_index_for_debug,
                    int(chr(key))
                )
                continue
            elif key == ord(' '):
                play_mode = not play_mode
                continue
            elif key == ord('n'):
                # Mark net area
                print("\n=== MARKING NET AREA ===")
                if self.mark_net_area(frame):
                    print("Net area marked successfully! Updated config file.")
                    # Reload HSV config to get updated net area
                    self.load_hsv_config()
                else:
                    print("Net area marking cancelled.")
                # Redisplay current frame
                continue
            elif key == ord('s'):
                # Show HSV tuner for serve area
                print("\n=== SERVE AREA HSV TUNER ===")
                if hasattr(self, 'serve_area_x_min'):
                    print(f"Serve area: X={self.serve_area_x_min}-{self.serve_area_x_max}, Y={self.serve_area_y_min}-{self.serve_area_y_max}")
                    self._open_serve_area_hsv_tuner(frame)
                else:
                    print("No serve area configured. Marking serve area first...")
                    if self.mark_serve_area(frame):
                        print("Serve area marked successfully! Updated config file.")
                        # Reload HSV config to get updated serve area
                        self.load_hsv_config()
                        # Now open the HSV tuner for the serve area
                        self._open_serve_area_hsv_tuner(frame)
                    else:
                        print("Serve area marking cancelled.")
                # Redisplay current frame
                continue
            elif key == ord('b'):
                # Show HSV tuner for current ball position
                if self.ball_center is None:
                    print("No ball position available yet.")
                    continue
                print("\n=== BALL HSV DEBUG ===")
                self.open_hsv_tuner(frame, self.ball_center)
                continue
            elif key == ord('d'):
                # Advance to next frame and automatically track
                if self.tracking and self.hsv_lower is not None:
                    print(f"\n--- ADVANCING TO FRAME {self.frame_count + 1} ---")
                    print("Automatically searching for ball in next frame...")
                    # Continue to next iteration to read next frame
                    # The ball will be tracked automatically
                    continue
                else:
                    print("\n--- ADVANCING TO FRAME (no tracking) ---")
                    print("No ball marked yet - click on ball first to enable tracking")
                    # Just continue to next frame without tracking
                    continue
            elif key == 8:  # BACKSPACE
                target_frame = max(self.start_frame, self.frame_count - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                self.frame_count = target_frame
                print(f"\n--- STEPPED BACK TO FRAME {self.frame_count} ---")
                continue
            elif key == ord('t'):
                # Show HSV table
                self.print_hsv_table()
            elif key == ord('c'):
                # Toggle contour-number debug overlay
                self.show_contour_debug = not getattr(self, 'show_contour_debug', False)
                status = "ON" if self.show_contour_debug else "OFF"
                print(f"[C KEY] Contour debug overlay: {status}")
                continue
            elif key == ord('r'):
                self.show_event_markers = not getattr(self, 'show_event_markers', False)
                status = "ON" if self.show_event_markers else "OFF"
                print(f"[R KEY] Red event markers: {status}")
                continue

            if self.pause_requested:
                self.pause_requested = False
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nAnalysis complete! Processed {self.frame_count} frames")


if __name__ == "__main__":
    # Per-court configuration: video file + HSV/serve config file
    COURT_CONFIGS = {
        "1": {
            "video":  "20251011124747503_FV3553362380_FV3553362.mp4",
            "config": "hsv_config.json",
            "label":  "Court 1 (far-end server)",
        },
        "2": {
            "video":  "20260322115648717_FV3553362360_FV3553362.mp4",
            "config": "hsv_config_court2.json",
            "label":  "Court 2 / 04 Left (near-end server)",
        },
    }

    parser = argparse.ArgumentParser(description="Interactive tennis ball analyzer/tracker")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame index to start from (default 0)")
    parser.add_argument("--auto-play", action="store_true",
                        help="Start playing immediately without waiting for SPACE")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after processing this many frames (0 = no limit)")
    parser.add_argument("--court", choices=list(COURT_CONFIGS.keys()), default="1",
                        help="Which court/video to analyse (default: 1)")
    args = parser.parse_args()

    court = COURT_CONFIGS[args.court]
    print(f"[COURT] {court['label']}")
    analyzer = InteractiveBallAnalyzer(court["video"], start_frame=args.start_frame,
                                       config_file=court["config"])
    analyzer.process_video(auto_play=args.auto_play, max_frames=args.max_frames)
