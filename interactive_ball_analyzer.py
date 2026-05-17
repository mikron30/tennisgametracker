import argparse
import cv2
import numpy as np
import json
import os
import math
from typing import Tuple, Optional


class InteractiveBallAnalyzer:
    def __init__(
        self,
        video_path: str,
        start_frame: int = 0,
        config_file: str = "hsv_config.json",
        headless: bool = False,
        disable_false_points: bool = False,
    ):
        self.video_path = video_path
        self.config_file = config_file
        self.headless = headless
        self.disable_false_points = disable_false_points
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
        self.using_alt4_hsv = False
        self.alt4_hsv_lower = None
        self.alt4_hsv_upper = None
        self.using_alt6_hsv = False
        self.alt5_hsv_lower = None
        self.alt5_hsv_upper = None
        self.alt6_hsv_lower = None
        self.alt6_hsv_upper = None
        self.s30_hsv_lower = None
        self.s30_hsv_upper = None
        self.h10_hsv_lower = None
        self.h10_hsv_upper = None
        self.alts20_hsv_lower = None
        self.alts20_hsv_upper = None
        self.alts9_11_hsv_lower = None
        self.alts9_11_hsv_upper = None
        # Keep behind-net / near-net HSV fully disabled by default.
        # Set DISABLE_BEHIND_NET=0 to temporarily re-enable it if needed.
        self.disable_behind_net_mode = os.environ.get("DISABLE_BEHIND_NET", "1") == "1"
        self.direction_change_points = []
        self.direction_change_events = []
        self.show_event_markers = True
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
        self._serve_scan_block_until_frame = -1
        self.point_start_frame_internal = None
        self._serve_contact_grace_frames = 0
        self._rally_contact_grace_frames = 0
        self._rally_contact_large_ref_frames = 0
        self._rally_contact_ref_size = None
        self._rally_contact_origin = None
        self._rally_contact_expected = None
        self._rally_contact_progress = 0.0
        self._rally_contact_high_launch_until_frame = -1
        self._last_racket_contact_frame = -1000000
        self._last_racket_contact_point = None
        self._ground_bounce_grace_frames = 0
        self._ground_bounce_ref_size = None
        self._ground_bounce_origin = None
        self._ground_bounce_expected = None
        self._ground_bounce_progress = 0.0
        self._ground_bounce_debug_history = []
        self.ground_bounce_count = 0
        self.last_ground_bounce_frame = -1000000
        self._contact_recovery_frames = 0
        self._upper_exit_wait_frames = 0
        self._top_return_wait_frames = 0
        self._top_return_anchor = None
        self._top_return_origin_frame = -1
        self._top_return_reentry_grace_frames = 0
        self._top_return_mode = None
        self._top_return_exit_dx = 0.0
        self._recent_offscreen_return_frame = -1000000
        self._recent_return_bounce_recover_frame = -1000000
        self._recent_racket_rebound_bounce_frame = -1000000
        self._late_contact_prior_bounce_until_frame = -1000000
        self._late_contact_prior_bounce_count = 0
        self._back_return_wait_frames = 0
        # Court-2 back/back-screen exits can be invisible for a while before returning.
        # Keep a real offscreen wait active instead of dropping into normal lost-ball
        # timeout as soon as the large close ball leaves the bottom edge.
        self._back_return_timeout_frames = 90
        self._back_return_anchor = None
        self._back_return_origin_frame = -1
        self._back_return_reentry_grace_frames = 0
        self._back_return_timed_out = False
        self.serve_width_ratio = None
        self._prev_serve_gray = None
        self._ignored_serve_positions = []
        self._ignored_tracking_positions = []
        self._persistent_false_points = []
        self._steady_false_point_observations = []
        self._steady_false_point_file = None
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
        self._deferred_motion_anchor = None
        self._held_direction_candidate = None
        self._last_motion_reacq_frame = -1000000
        self._last_motion_reacq_pos = None
        self._upper_slow_arc_until_frame = -1
        self._upper_fence_fall_frames = 0
        self.last_nonzero_motion = None
        self._singles_sideline_model = None
        self._singles_sideline_frame_shape = None
        self._singles_court_model = None
        self._singles_court_frame_shape = None
        self._service_box_model = None
        self._service_box_frame_shape = None
        self._white_line_visual_model = None
        self._white_line_visual_frame_shape = None
        self.net_area_points = []
        self.current_game_state = "WAITING_FOR_SERVE"
        self.point_end_marker_hold_frames = 10
        self._ended_point_visual_snapshot = None
        self.recent_bounce_markers = []
        self._last_impact_marker_frame = -1000000
        self._last_impact_marker_pos = None
        self._last_impact_marker_kind = None
        self._pending_rally_end_reason = None
        self._pending_rally_end_frame = -1
        self._awaiting_serve_bounce = False
        self._point_serve_start_side = None
        self._point_target_service_side = None
        self._last_direction_change_frame = -1000000
        self._last_direction_change_point = None
        self._last_direction_change_angle = 0.0
        self._last_direction_change_vertical_reversal = False
        self.direction_change_min_degrees = 20.0
        self.motion_debug_vectors = []
        self.motion_debug_vector_limit = 8

    def log_motion_metrics(self, prev_pos, dx, dy, distance, direction_deg):
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
        if distance >= 3.0:
            self.last_nonzero_motion = dict(self.last_motion)
        angle_diff = None
        forward_vertical_reversal = False
        direction_change_detected = False
        if direction_deg is not None and distance > 5:
            if self.near_edge or getattr(self, 'edge_wait', False):
                self._record_motion_vector(prev_pos, self.ball_center, direction_deg)
                self.last_direction = direction_deg
                return False
            if self.last_direction is not None:
                threshold_deg = max(0.0, float(getattr(self, 'direction_change_min_degrees', 20.0)))
                prev_dx = float(self.prev_motion.get('dx', 0.0)) if self.prev_motion is not None else 0.0
                prev_dy = float(self.prev_motion.get('dy', 0.0)) if self.prev_motion is not None else 0.0
                delta = abs(direction_deg - self.last_direction) % 360
                angle_diff = min(delta, 360 - delta)
                same_forward_x = abs(prev_dx) <= 3.0 or abs(dx) <= 3.0 or (prev_dx * dx) >= -6.0
                forward_vertical_reversal = (
                    prev_dy >= 4.0 and
                    dy <= -4.0 and
                    same_forward_x
                )
                direction_change_detected = forward_vertical_reversal or angle_diff >= threshold_deg
                if direction_change_detected:
                    change_point = prev_pos if prev_pos is not None else self.ball_center
                    if (getattr(self, 'ground_bounce_count', 0) > 0 and
                            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) > 2 and
                            distance > 12 and
                            (angle_diff >= 110.0 or (prev_dx * dx) < -12.0)):
                        print(f"Frame {self.frame_count}: Resetting bounce count after new shot direction change")
                        self.ground_bounce_count = 0
                        self.last_ground_bounce_frame = -1000000
                    self.direction_change_points.append((change_point[0], change_point[1]))
                    self._record_direction_change_event(
                        change_point,
                        prev_dx,
                        prev_dy,
                        dx,
                        dy,
                        float(self.prev_motion.get('distance', 0.0)) if self.prev_motion is not None else 0.0,
                        distance,
                        angle_diff,
                    )
                    self._last_direction_change_frame = self.frame_count
                    self._last_direction_change_point = (int(change_point[0]), int(change_point[1]))
                    self._last_direction_change_angle = float(angle_diff)
                    self._last_direction_change_vertical_reversal = bool(forward_vertical_reversal)
                    if getattr(self, '_last_impact_marker_frame', -1000000) != self.frame_count:
                        self._add_impact_marker(
                            change_point,
                            kind="direction_change",
                            ttl=7,
                            label="direction change",
                        )
                    extra_tag = " vertical-reversal" if forward_vertical_reversal else ""
                    print(
                        f"Frame {self.frame_count}: Direction change detected "
                        f"({angle_diff:.1f} deg, threshold={threshold_deg:.1f}{extra_tag})"
                    )
            self._record_motion_vector(
                prev_pos,
                self.ball_center,
                direction_deg,
                angle_diff=angle_diff,
                changed=direction_change_detected,
                vertical_reversal=forward_vertical_reversal,
            )
            self.last_direction = direction_deg
        self.motion_history.append({
            'frame': self.frame_count,
            'distance': distance,
            'direction_deg': direction_deg,
            'pos': tuple(self.ball_center) if self.ball_center is not None else None,
            'prev_pos': tuple(prev_pos) if prev_pos is not None else None,
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
                self.using_alt2_hsv = False
                self.using_alt3_hsv = False
                self.using_alt6_hsv = False
                if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                    self.hsv_lower = self.alt_focus_hsv_lower
                    self.hsv_upper = self.alt_focus_hsv_upper
                print(f"Frame {self.frame_count}: [FOCUS LOSS FLAG] movement spike detected "
                      f"(distance {distance:.1f}px vs median {median_distance:.1f}px) "
                      f"starting from frame {self.start_frame}")
                return True
        return False

    def _record_motion_vector(self, prev_pos, curr_pos, direction_deg, angle_diff=None,
                              changed=False, vertical_reversal=False):
        if prev_pos is None or curr_pos is None or direction_deg is None:
            return
        self.motion_debug_vectors.append({
            'frame': int(self.frame_count),
            'from': (int(prev_pos[0]), int(prev_pos[1])),
            'to': (int(curr_pos[0]), int(curr_pos[1])),
            'direction_deg': float(direction_deg),
            'angle_diff': float(angle_diff) if angle_diff is not None else None,
            'changed': bool(changed),
            'vertical_reversal': bool(vertical_reversal),
        })
        max_vectors = max(4, int(getattr(self, 'motion_debug_vector_limit', 8)))
        if len(self.motion_debug_vectors) > max_vectors:
            self.motion_debug_vectors = self.motion_debug_vectors[-max_vectors:]

    def _classify_direction_change_review(self, point, prev_dx, prev_dy, dx, dy, prev_distance, distance):
        serve_like = (
            getattr(self, '_serve_contact_grace_frames', 0) > 0 or
            self._serve_direction_search_active()
        )
        ground_reversal_like = prev_dy >= 4.0 and dy <= -4.0
        # Some real court bounces are first visible as a strong downward path
        # flattening at court level before the next frame clearly moves upward.
        ground_flatten_like = (
            point is not None and
            int(point[1]) >= 420 and
            prev_dy >= 18.0 and
            -4.0 <= dy <= max(8.0, prev_dy * 0.35) and
            prev_distance >= 18.0 and
            distance >= 12.0
        )
        ground_like = ground_reversal_like or ground_flatten_like
        racket_like = (prev_dx * dx) < -12.0 and max(abs(prev_dx), abs(dx)) >= 6.0
        net_like = False
        if point is not None:
            net_geometry = self._net_contact_geometry(point)
            if net_geometry is not None:
                net_like = (
                    net_geometry['inside'] or
                    net_geometry['near_top_tape'] or
                    (net_geometry['near_polygon'] and (dy >= 4.0 or prev_dy <= -4.0))
                )
        slow_suspect = min(prev_distance, distance) < 12.0

        if serve_like:
            return "good", "serve"
        if ground_like:
            return "good", "ground"
        if racket_like:
            return "good", "racket"
        if net_like:
            return "good", "net"
        if slow_suspect:
            return "problematic", "slow"
        return "problematic", "other"

    def _append_direction_change_review_event(self, point, status, reason, angle_diff=None, frame=None):
        if point is None:
            return
        event_frame = int(self.frame_count if frame is None else frame)
        pos = (int(point[0]), int(point[1]))
        for existing in reversed(self.direction_change_events[-8:]):
            if existing.get('frame') != event_frame:
                continue
            existing_pos = existing.get('pos')
            if existing_pos is None:
                continue
            if math.hypot(existing_pos[0] - pos[0], existing_pos[1] - pos[1]) > 18:
                continue
            if existing.get('status') != 'good' and status == 'good':
                existing['status'] = status
                existing['reason'] = reason
                if angle_diff is not None:
                    existing['angle_diff'] = float(angle_diff)
            return
        self.direction_change_events.append({
            'frame': event_frame,
            'pos': pos,
            'status': status,
            'reason': reason,
            'angle_diff': float(angle_diff) if angle_diff is not None else None,
        })

    def _record_direction_change_event(self, point, prev_dx, prev_dy, dx, dy, prev_distance, distance, angle_diff):
        if point is None:
            return
        status, reason = self._classify_direction_change_review(
            point,
            prev_dx, prev_dy, dx, dy, prev_distance, distance
        )
        self._append_direction_change_review_event(point, status, reason, angle_diff=angle_diff)

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

    def _is_descending_serve_contact_motion(self, motion=None):
        if motion is None:
            motion = self.last_motion
        if motion is None:
            return False
        dy = float(motion.get('dy', 0.0))
        if self.serve_direction_dy == 0:
            return dy > 12.0
        return self._signed_serve_dy(dy) < -12.0

    def _is_forward_serve_launch_motion(self, motion=None):
        if motion is None:
            motion = self.last_motion
        if motion is None:
            return False
        dx = float(motion.get('dx', 0.0))
        dy = float(motion.get('dy', 0.0))
        signed_dx = self._signed_serve_dx(dx)
        min_signed_dx = max(32.0, float(self.serve_contact_min_dx) * 0.35)
        if signed_dx < min_signed_dx:
            return False
        if self.serve_direction_dy == 0:
            return True
        signed_dy = self._signed_serve_dy(dy)
        min_signed_dy = max(8.0, float(self.serve_contact_min_dy) * 0.25)
        return signed_dy >= min_signed_dy

    def _min_area_for_previous_ball_size(self, prev_size, prev_pos=None, frame_shape=None):
        """Reject tiny fragments while a close lower-court ball should still be large."""
        if not prev_size or prev_size <= 40:
            return 5

        return max(5, int(prev_size * 0.08))

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

        post_contact_launch_search = (
            getattr(self, '_serve_contact_grace_frames', 0) > 0 and
            self._is_forward_serve_launch_motion(self.last_motion) and
            (
                self._is_descending_serve_contact_motion(self.prev_motion) or
                self._is_descending_serve_contact_motion(getattr(self, 'prev_prev_motion', None))
            ) and
            (self.ball_size or 0) >= max(45, int(self.serve_contact_min_ball_size * 0.45))
        )
        if post_contact_launch_search:
            return True

        serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
        contact_y = self.serve_area_y_min + int(serve_height * self.serve_contact_y_ratio)
        # On the final descending-contact frame, the ball can still be a little above the
        # nominal contact band just before it launches back up/forward.  Allow a small
        # early-entry margin here so we search the serve-direction wedge instead of
        # forcing the generic continuation heuristic to keep following the downward toss.
        contact_y_margin = 0
        if (
            getattr(self, '_serve_contact_grace_frames', 0) <= 1 and
            self._is_descending_serve_contact_motion(self.last_motion)
        ):
            contact_y_margin = max(
                20,
                min(36, int(abs(float(self.last_motion.get('dy', 0.0) or 0.0)) * 0.6))
            )
        if y < (contact_y - contact_y_margin):
            return False
        if self.last_motion.get('dy', 0) < self.serve_contact_min_prev_dy:
            return False
        if abs(self.last_motion.get('dx', 0)) > self.serve_contact_max_prev_dx:
            return False
        if (self.ball_size or 0) < self.serve_contact_min_ball_size:
            return False
        return True

    def _build_serve_direction_region(self, x, y, frame_shape, radius, descending_contact=False):
        frame_height, frame_width = frame_shape[:2]
        serve_contact_drop_band = 0
        if self.serve_direction_dy < 0 and descending_contact:
            serve_contact_drop_band = max(70, int(radius * 0.18))

        if self.serve_direction_dx < 0:
            x1 = max(0, x - radius)
            x2 = min(frame_width, x + 1)
        else:
            x1 = max(0, x)
            x2 = min(frame_width, x + radius + 1)

        if self.serve_direction_dy < 0:
            y1 = max(0, y - radius)
            y2 = min(frame_height, y + serve_contact_drop_band + 1)
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

    def _prune_steady_false_point_observations(self):
        self._steady_false_point_observations = [
            entry for entry in getattr(self, '_steady_false_point_observations', [])
            if (self.frame_count - entry.get('last_frame', self.frame_count)) <= 90
        ]

    def _steady_false_point_file_path(self):
        config_path = os.path.abspath(self.config_file or "hsv_config.json")
        base = os.path.splitext(os.path.basename(config_path))[0]
        return os.path.join(
            os.path.dirname(config_path),
            f"steady_false_points_{base}.json"
        )

    def _normalize_false_point_filter_key(self, filter_key):
        if filter_key is None:
            return None
        key = str(filter_key).strip().lower()
        if not key:
            return None
        if key == "primary":
            return "primary"
        if key in ("alt", "alternative", "alternative_1", "alt1"):
            return "alt1"
        if key.startswith("alternative_2") or key.startswith("alt2"):
            return "alt2"
        if key.startswith("alternative_3") or key.startswith("alt3"):
            return "alt3"
        if key.startswith("alt4"):
            return "alt4"
        if key.startswith("alt5"):
            return "alt5"
        if key.startswith("alt6"):
            return "alt6"
        if key.startswith("s_30") or key.startswith("s30"):
            return "regular_court"
        if key.startswith("h_10") or key.startswith("h10"):
            return "h_10"
        if key.startswith("alts_20") or key.startswith("alts20"):
            return "alts_20"
        if key.startswith("alts9_11") or key.startswith("alts911"):
            return "alts9_11"
        if key.startswith("behind_net") or key.startswith("at_edge"):
            return "behind_net"
        if key.startswith("regular") or key in ("single", "single_prefocus", "serve_area", "current"):
            return "regular_court"
        return key

    def _candidate_false_point_filter_key(self, source, hsv_mode=None):
        source_key = self._normalize_false_point_filter_key(source)
        if source_key == "primary":
            mode_key = self._normalize_false_point_filter_key(hsv_mode)
            if mode_key not in (None, "primary"):
                return mode_key
            if self.using_alt3_hsv:
                return "alt3"
            if self.using_alt2_hsv:
                return "alt2"
            if self.using_alt_hsv:
                return "alt1"
            return "regular_court"
        if source_key == "alt":
            return "alt1"
        return source_key

    def _load_persistent_false_points(self):
        self._persistent_false_points = []
        self._steady_false_point_file = self._steady_false_point_file_path()
        if not os.path.exists(self._steady_false_point_file):
            return

        try:
            with open(self._steady_false_point_file, "r") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"Persistent false-point load failed: {e}")
            return

        raw_entries = payload.get("false_points", payload if isinstance(payload, list) else [])
        loaded = []
        for raw in raw_entries:
            pos = raw.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            filter_keys = raw.get("filter_keys", raw.get("filters", []))
            if not isinstance(filter_keys, list):
                filter_keys = [filter_keys]
            normalized_keys = []
            for key in filter_keys:
                norm = self._normalize_false_point_filter_key(key)
                if norm is not None and norm not in normalized_keys:
                    normalized_keys.append(norm)
            loaded.append({
                "pos": (int(pos[0]), int(pos[1])),
                "radius": int(raw.get("radius", 20)),
                "count": int(raw.get("count", 1)),
                "reason": raw.get("reason", "persistent false point"),
                "filter_keys": normalized_keys,
            })

        self._persistent_false_points = loaded
        print(
            f"Loaded {len(self._persistent_false_points)} persistent false points "
            f"from {os.path.basename(self._steady_false_point_file)}"
        )

    def _save_persistent_false_points(self):
        if self._steady_false_point_file is None:
            self._steady_false_point_file = self._steady_false_point_file_path()
        payload = {
            "config": os.path.basename(self.config_file or "hsv_config.json"),
            "false_points": [
                {
                    "pos": [int(entry["pos"][0]), int(entry["pos"][1])],
                    "radius": int(entry.get("radius", 20)),
                    "count": int(entry.get("count", 1)),
                    "reason": entry.get("reason", "persistent false point"),
                    "filter_keys": list(entry.get("filter_keys", [])),
                }
                for entry in getattr(self, "_persistent_false_points", [])
            ],
        }
        try:
            with open(self._steady_false_point_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Persistent false-point save failed: {e}")

    def _find_persistent_false_point(self, pos, filter_key=None):
        px, py = pos
        normalized_key = self._normalize_false_point_filter_key(filter_key)
        alt_family = {"alt2", "alt4", "alt5", "alt6", "alts9_11"}
        for entry in getattr(self, "_persistent_false_points", []):
            entry_keys = entry.get("filter_keys", [])
            if normalized_key is not None and entry_keys:
                if normalized_key not in entry_keys:
                    if normalized_key not in alt_family or not alt_family.intersection(entry_keys):
                        continue
            dist = math.hypot(px - entry["pos"][0], py - entry["pos"][1])
            if dist <= entry["radius"]:
                return entry
        return None

    def _find_false_point_for_diagnostics(self, pos):
        return self._find_persistent_false_point(pos, filter_key=None)

    def _learn_persistent_false_point(self, pos, filter_key, radius=22, reason="steady false point", count=1):
        normalized_key = self._normalize_false_point_filter_key(filter_key)
        if normalized_key is None:
            return None
        if int(getattr(self, 'start_frame', 0) or 0) > 0:
            print(
                f"[FALSE_POINT_LEARN_SKIPPED] f{self.frame_count}: partial-run false point "
                f"at {pos} filter={normalized_key} reason={reason}"
            )
            return None

        for entry in self._persistent_false_points:
            dist = math.hypot(pos[0] - entry["pos"][0], pos[1] - entry["pos"][1])
            if dist > max(radius, entry.get("radius", 0)):
                continue
            keys = entry.setdefault("filter_keys", [])
            if normalized_key not in keys:
                keys.append(normalized_key)
            merged_count = max(int(entry.get("count", 1)), int(count))
            old_x, old_y = entry["pos"]
            entry["pos"] = (
                int(round((old_x * merged_count + pos[0]) / (merged_count + 1))),
                int(round((old_y * merged_count + pos[1]) / (merged_count + 1))),
            )
            entry["radius"] = max(int(entry.get("radius", radius)), int(radius))
            entry["count"] = merged_count + 1
            entry["reason"] = reason
            self._save_persistent_false_points()
            print(
                f"[FALSE_POINT_LEARN] f{self.frame_count}: refresh {normalized_key} false point "
                f"at {entry['pos']} radius={entry['radius']} reason={reason}"
            )
            return entry

        entry = {
            "pos": (int(pos[0]), int(pos[1])),
            "radius": int(radius),
            "count": int(max(1, count)),
            "reason": reason,
            "filter_keys": [normalized_key],
        }
        self._persistent_false_points.append(entry)
        self._save_persistent_false_points()
        print(
            f"[FALSE_POINT_LEARN] f{self.frame_count}: learned {normalized_key} false point "
            f"at {entry['pos']} radius={entry['radius']} reason={reason}"
        )
        return entry

    def _observe_steady_false_point(self, pos, filter_key, area, motion_mean, motion_max, reason):
        normalized_key = self._normalize_false_point_filter_key(filter_key)
        if normalized_key not in ("alt2", "alt6", "alt4", "alt5", "alts9_11"):
            return False
        if area > 18.0 or motion_mean > 3.5 or motion_max > 12.0:
            return False
        if self.ball_center is not None and math.hypot(pos[0] - self.ball_center[0], pos[1] - self.ball_center[1]) < 35.0:
            return False
        if self._find_persistent_false_point(pos, filter_key=normalized_key) is not None:
            return True

        self._prune_steady_false_point_observations()
        for entry in self._steady_false_point_observations:
            if entry["filter_key"] != normalized_key:
                continue
            dist = math.hypot(pos[0] - entry["pos"][0], pos[1] - entry["pos"][1])
            if dist > entry["radius"]:
                continue
            if entry["last_frame"] != self.frame_count:
                old_count = entry["count"]
                entry["count"] = old_count + 1
                old_x, old_y = entry["pos"]
                entry["pos"] = (
                    int(round((old_x * old_count + pos[0]) / entry["count"])),
                    int(round((old_y * old_count + pos[1]) / entry["count"])),
                )
            entry["last_frame"] = self.frame_count
            entry["reason"] = reason
            if entry["count"] >= 3:
                self._learn_persistent_false_point(
                    entry["pos"],
                    normalized_key,
                    radius=entry["radius"] + 4,
                    reason=reason,
                    count=entry["count"],
                )
                self._steady_false_point_observations.remove(entry)
                return True
            return False

        self._steady_false_point_observations.append({
            "pos": (int(pos[0]), int(pos[1])),
            "radius": 18,
            "count": 1,
            "last_frame": self.frame_count,
            "filter_key": normalized_key,
            "reason": reason,
        })
        return False

    def _erase_diagnostic_false_points(self, mask, x1, y1):
        if self.disable_false_points:
            return 0
        if mask is None or mask.size == 0:
            return 0
        hidden = 0
        for entry in getattr(self, "_persistent_false_points", []):
            px, py = entry["pos"]
            local_x = int(px - x1)
            local_y = int(py - y1)
            if local_x < 0 or local_y < 0 or local_x >= mask.shape[1] or local_y >= mask.shape[0]:
                continue
            radius = max(6, int(entry.get("radius", 12)))
            cv2.circle(mask, (local_x, local_y), radius, 0, -1)
            hidden += 1
        return hidden

    def _prefer_regular_tracking_candidate(self, candidate_meta, predicted_point):
        if self.ball_center is None or not candidate_meta:
            return None
        if getattr(self, "ground_bounce_count", 0) <= 0:
            return None
        if self.ball_center[1] > 260:
            return None
        if self.last_motion is not None and self.last_motion.get("distance", 0.0) > 45.0:
            return None

        best_entry = min(candidate_meta, key=lambda entry: entry["score"])
        if best_entry.get("filter_key") == "regular_court":
            return None

        alt_family = {"alt1", "alt2", "alt3", "alt4", "alt5", "alt6"}
        if best_entry.get("filter_key") not in alt_family:
            return None

        regular_candidates = [
            entry for entry in candidate_meta
            if entry.get("filter_key") == "regular_court"
        ]
        if not regular_candidates:
            return None

        regular_best = min(regular_candidates, key=lambda entry: (entry["score"], entry["distance"]))
        pos_gap = math.hypot(
            regular_best["pos"][0] - best_entry["pos"][0],
            regular_best["pos"][1] - best_entry["pos"][1],
        )
        regular_pred = regular_best.get("predicted_distance")
        best_pred = best_entry.get("predicted_distance")
        regular_close_to_track = regular_best["distance"] <= best_entry["distance"] + 18.0
        regular_close_to_pred = (
            predicted_point is None or
            regular_pred is None or
            best_pred is None or
            regular_pred <= best_pred + 24.0
        )
        regular_only = len(regular_candidates) == 1
        score_margin = 85.0 if best_entry["filter_key"] in {"alt4", "alt5", "alt6"} else 42.0
        if regular_only:
            score_margin += 15.0

        if (pos_gap <= 32.0 and
                regular_best["score"] <= best_entry["score"] + score_margin and
                regular_close_to_track and regular_close_to_pred):
            print(
                f"  DEBUG: Regular precedence -> preferring {regular_best['pos']} "
                f"score={regular_best['score']:.1f} over {best_entry['pos']} "
                f"from {best_entry['filter_key']}"
            )
            return regular_best

        return None

    def _prefer_predicted_continuation_candidate(self, candidate_meta, predicted_point):
        if self.ball_center is None or predicted_point is None or not candidate_meta:
            return None
        if self.last_motion is None:
            return None

        lm_dx = float(self.last_motion.get("dx", 0.0) or 0.0)
        lm_dy = float(self.last_motion.get("dy", 0.0) or 0.0)
        lm_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if lm_dist < 20.0:
            return None

        best_entry = min(candidate_meta, key=lambda entry: (entry["score"], entry["distance"]))
        best_distance = float(best_entry.get("distance", 0.0) or 0.0)
        stale_cap = max(1.5, min(8.0, lm_dist * 0.12))

        best_pred = best_entry.get("predicted_distance")
        best_is_stale = best_distance <= stale_cap
        best_is_already_on_path = (
            best_pred is not None and
            best_pred <= max(36.0, min(110.0, lm_dist * 0.32))
        )
        if not best_is_stale and best_is_already_on_path:
            return None
        if best_is_stale and best_pred is not None and best_pred <= max(20.0, lm_dist * 0.18):
            return None

        forward_candidates = []
        for entry in candidate_meta:
            if entry is best_entry:
                continue

            pred_dist = entry.get("predicted_distance")
            if pred_dist is None:
                continue

            move_dist = float(entry.get("distance", 0.0) or 0.0)
            if move_dist < max(28.0, lm_dist * 0.35):
                continue
            if move_dist > max(240.0, lm_dist * 1.55):
                continue

            mv_dx = entry["pos"][0] - self.ball_center[0]
            mv_dy = entry["pos"][1] - self.ball_center[1]
            dot = lm_dx * mv_dx + lm_dy * mv_dy
            if dot <= 0.0:
                continue

            pred_cap = max(40.0, min(165.0, lm_dist * 0.85))
            if pred_dist > pred_cap:
                continue

            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            area = float(entry.get("area", 0.0) or 0.0)
            if motion_mean < 3.0 and motion_max < 18.0 and area < 5.0:
                continue

            min_pred_improvement = 40.0 if best_is_stale else max(70.0, min(150.0, lm_dist * 0.30))
            if best_pred is not None and pred_dist > best_pred - min_pred_improvement:
                continue

            align_bonus = dot / max(1.0, lm_dist * move_dist)
            if align_bonus < 0.45:
                continue
            adjusted_score = (
                entry["score"]
                - min(120.0, max(0.0, best_pred - pred_dist) * 1.8) if best_pred is not None
                else entry["score"]
            )
            adjusted_score -= max(0.0, align_bonus) * 55.0
            adjusted_score -= min(55.0, motion_max * 0.45)
            adjusted_score -= min(30.0, motion_mean * 2.0)
            if entry.get("source") in ("primary", "regular"):
                adjusted_score -= 18.0

            forward_candidates.append((adjusted_score, pred_dist, -move_dist, entry))

        if not forward_candidates:
            return None

        _, _, _, chosen = min(forward_candidates, key=lambda item: (item[0], item[1], item[2]))
        best_label = "stale" if best_is_stale else "off-path"
        print(
            f"  DEBUG: Predicted continuation -> preferring {chosen['pos']} "
            f"score={chosen['score']:.1f} pred_dist={chosen.get('predicted_distance', float('nan')):.1f} "
            f"over {best_label} {best_entry['pos']} score={best_entry['score']:.1f}"
        )
        return chosen

    def _prefer_recent_return_dynamic_candidate(self, candidate_meta, selected_contour, frame_shape):
        if self.ball_center is None or not candidate_meta or self.last_motion is None:
            return None
        if not self._recent_offscreen_return_hold_active(window_frames=80):
            return None

        selected_entry = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get("contour") is selected_contour:
                    selected_entry = entry
                    break
        if selected_entry is None:
            selected_entry = min(candidate_meta, key=lambda entry: (entry["score"], entry["distance"]))

        selected_area = float(selected_entry.get("area", 0.0) or 0.0)
        selected_motion_mean = float(selected_entry.get("motion_mean", 0.0) or 0.0)
        selected_motion_max = float(selected_entry.get("motion_max", 0.0) or 0.0)
        selected_frame0 = self._find_frame0_background_hotspot(selected_entry["pos"]) is not None
        selected_static = (
            selected_area <= 6.0 and
            selected_motion_mean < 4.0 and
            selected_motion_max < 22.0
        ) or (
            selected_frame0 and
            selected_motion_mean < 8.0 and
            selected_motion_max < 35.0
        )
        if not selected_static:
            return None

        last_dx = float(self.last_motion.get("dx", 0.0) or 0.0)
        last_dy = float(self.last_motion.get("dy", 0.0) or 0.0)
        last_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if last_dist < 70.0:
            return None

        prev_x, prev_y = self.ball_center
        selected_x, selected_y = selected_entry["pos"]
        frame_height, _ = frame_shape[:2]
        prev_size = float(self.ball_size or 0.0)
        min_area = max(70.0, min(140.0, prev_size * 0.10 if prev_size else 70.0))
        max_distance = max(330.0, min(430.0, last_dist * 1.65))
        min_distance = max(45.0, last_dist * 0.22)

        rescue_candidates = []
        for entry in candidate_meta:
            if entry is selected_entry:
                continue
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            cx, cy = entry["pos"]
            area = float(entry.get("area", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            distance = float(entry.get("distance", 0.0) or 0.0)

            if area < min_area:
                continue
            if prev_size >= 500.0 and area > prev_size * 1.65:
                continue
            if motion_mean < 18.0 or motion_max < 70.0:
                continue
            if distance < min_distance or distance > max_distance:
                continue

            # The real ball can lag vertically after the offscreen return; it
            # should sit just below the predicted static speck, not elsewhere.
            if abs(cx - selected_x) > 65:
                continue
            if cy < selected_y + 18 or cy > selected_y + 130:
                continue
            if cy < max(280, int(frame_height * 0.14)):
                continue

            move_dx = float(cx - prev_x)
            move_dy = float(cy - prev_y)
            move_dist = max(1.0, math.hypot(move_dx, move_dy))
            alignment = (last_dx * move_dx + last_dy * move_dy) / max(1.0, last_dist * move_dist)
            if alignment < 0.72:
                continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None and motion_mean < 12.0 and motion_max < 55.0:
                continue

            adjusted_score = (
                float(entry.get("score", 0.0) or 0.0) -
                min(250.0, area * 0.08) -
                min(90.0, motion_mean * 1.2) -
                min(80.0, motion_max * 0.25) +
                abs(cx - selected_x) * 0.8 +
                abs(cy - (selected_y + 42)) * 0.8
            )
            rescue_candidates.append((adjusted_score, -area, entry))

        if not rescue_candidates:
            return None

        _, _, chosen = min(rescue_candidates, key=lambda item: (item[0], item[1]))
        print(
            f"  DEBUG: [RECENT-RETURN DYNAMIC] preferring moving candidate at "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"score={chosen['score']:.1f} motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over static {selected_entry['pos']} score={selected_entry['score']:.1f}"
        )
        return chosen

    def _prefer_large_descending_continuation_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Keep a large descending ball when a smaller off-path blob tries to steal focus."""
        if self.ball_center is None or self.last_motion is None or not candidate_meta:
            return None

        frame_height, _ = frame_shape[:2]
        _, prev_y = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        if prev_size < 180.0 or prev_y < max(1600, int(frame_height * 0.74)):
            return None

        last_dx = float(self.last_motion.get("dx", 0.0) or 0.0)
        last_dy = float(self.last_motion.get("dy", 0.0) or 0.0)
        last_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if last_dy < 35.0 or last_dist < 55.0:
            return None

        selected_entry = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get("contour") is selected_contour:
                    selected_entry = entry
                    break
        if selected_entry is None:
            selected_entry = min(candidate_meta, key=lambda entry: (entry["score"], entry["distance"]))

        selected_area = float(selected_entry.get("area", 0.0) or 0.0)
        selected_distance = float(selected_entry.get("distance", 0.0) or 0.0)
        selected_size_ratio = abs(selected_area - prev_size) / max(prev_size, 1.0)
        selected_keeps_shape = (
            selected_area >= prev_size * 0.55 and
            selected_size_ratio <= 0.55 and
            selected_distance <= max(145.0, last_dist * 1.35)
        )
        if selected_keeps_shape:
            return None

        prev_x, prev_y = self.ball_center
        continuation_candidates = []
        for entry in candidate_meta:
            if entry is selected_entry:
                continue
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            cx, cy = entry["pos"]
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            distance = float(entry.get("distance", 0.0) or 0.0)
            area = float(entry.get("area", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)

            if area < prev_size * 0.58 or area > prev_size * 1.55:
                continue
            if dy < max(10.0, last_dy * 0.18):
                continue
            if dy > max(150.0, last_dy * 1.55):
                continue
            if distance > max(155.0, last_dist * 1.45):
                continue
            if motion_max < 45.0 and motion_mean < 10.0:
                continue

            # Keep roughly the same descent lane, but allow the ball to start
            # flattening laterally as it nears the floor.
            if abs(last_dx) >= 18.0 and abs(dx - last_dx) > max(48.0, abs(last_dx) * 2.2):
                continue

            area_ratio = abs(area - prev_size) / max(prev_size, 1.0)
            adjusted_score = (
                area_ratio * 120.0 +
                distance * 0.35 +
                abs(distance - last_dist) * 0.28 +
                abs(dy - last_dy) * 0.20 -
                min(35.0, motion_mean * 0.55) -
                min(42.0, motion_max * 0.18)
            )
            if entry.get("source") in ("primary", "regular"):
                adjusted_score -= 10.0
            continuation_candidates.append((adjusted_score, entry))

        if not continuation_candidates:
            return None

        _, chosen = min(continuation_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [LARGE-DESCENT CONTINUE] preferring same-flight candidate at "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"score={chosen['score']:.1f} over {selected_entry['pos']} "
            f"area={selected_area:.1f}px score={selected_entry['score']:.1f}"
        )
        return chosen

    def _prefer_large_lower_bounce_launch_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Prefer the first strong upward rebound from a very large near-camera bounce."""
        if self.ball_center is None or self.last_motion is None or not candidate_meta:
            return None

        frame_height, _ = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        last_dy = float(self.last_motion.get("dy", 0.0) or 0.0)
        last_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if (
                prev_size < 180.0 or
                prev_y < max(1700, int(frame_height * 0.78)) or
                last_dy < 28.0 or
                last_dist < 32.0):
            return None

        selected_entry = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get("contour") is selected_contour:
                    selected_entry = entry
                    break

        min_upward = max(115.0, last_dy * 1.8)
        max_distance = max(260.0, min(360.0, last_dist * 6.5))
        min_area = max(70.0, prev_size * 0.45)
        max_area = max(prev_size * 1.45, min_area + 40.0)
        launch_candidates = []

        for entry in candidate_meta:
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            cx, cy = entry["pos"]
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            upward = -dy
            distance = float(entry.get("distance", 0.0) or 0.0)
            area = float(entry.get("area", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)

            if upward < min_upward:
                continue
            if distance < min_upward * 0.75 or distance > max_distance:
                continue
            if area < min_area or area > max_area:
                continue
            if motion_max < 70.0 and motion_mean < 18.0:
                continue
            if abs(dx) > max(180.0, last_dist * 3.5):
                continue

            area_ratio = abs(area - prev_size) / max(prev_size, 1.0)
            adjusted_score = (
                upward * 1.10 +
                min(160.0, area * 0.20) +
                min(70.0, motion_mean * 1.1) +
                min(85.0, motion_max * 0.32) -
                area_ratio * 45.0 -
                distance * 0.18
            )
            if entry.get("source") in ("primary", "regular"):
                adjusted_score += 12.0
            launch_candidates.append((adjusted_score, entry))

        if not launch_candidates:
            return None

        _, chosen = max(launch_candidates, key=lambda item: item[0])
        selected_text = (
            f" over {selected_entry['pos']} area={selected_entry['area']:.1f}px"
            if selected_entry is not None else ""
        )
        print(
            f"  DEBUG: [LARGE-LOWER BOUNCE LAUNCH] preferring upward rebound at "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"score={chosen['score']:.1f}{selected_text}"
        )
        return chosen

    def _prefer_top_return_downward_continuation(self, candidate_meta, frame_shape):
        if self.ball_center is None or not candidate_meta:
            return None
        if not self._recent_offscreen_return_hold_active(window_frames=12):
            return None

        prev_x, prev_y = self.ball_center
        frame_height, frame_width = frame_shape[:2]
        if prev_y < 50 or prev_y > max(125, int(frame_height * 0.065)):
            return None

        motion = self.last_motion
        last_dist = float(motion.get("distance", 0.0) or 0.0) if motion is not None else 0.0
        last_dy = float(motion.get("dy", 0.0) or 0.0) if motion is not None else 0.0
        prev_size = float(self.ball_size or 0.0)
        if prev_size > 80.0 or last_dist < 140.0 or last_dist > 420.0:
            return None
        if last_dy > 0.0 and last_dist <= 120.0:
            expected_dy = max(12.0, min(58.0, last_dy * 1.15))
        else:
            expected_dy = 24.0

        max_area = max(180.0, prev_size * 4.0 if prev_size else 180.0)
        x_cap = max(42.0, min(76.0, frame_width * 0.02))
        max_dy = 112.0 if prev_y >= 78 else 96.0
        best = None
        best_score = float("inf")

        for entry in candidate_meta:
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            cx, cy = entry["pos"]
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            area = float(entry.get("area", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            distance = float(entry.get("distance", 0.0) or 0.0)

            if dy < 6.0 or dy > max_dy:
                continue
            if abs(dx) > x_cap:
                continue
            if distance < 7.0 or distance > max(95.0, min(120.0, max_dy * 1.15)):
                continue
            if area < 3.0 or area > max_area:
                continue
            if motion_max < 45.0 and motion_mean < 12.0 and area < 10.0:
                continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None and motion_mean < 12.0 and motion_max < 55.0:
                continue

            size_ratio = (
                abs(area - prev_size) / max(prev_size, 1.0)
                if prev_size else 0.0
            )
            score = (
                abs(dx) * 2.0 +
                abs(dy - expected_dy) * 1.2 +
                distance * 0.55 +
                size_ratio * 12.0 -
                min(45.0, motion_mean * 0.8) -
                min(55.0, motion_max * 0.22) -
                min(35.0, area * 0.45)
            )
            if entry.get("source") in ("primary", "regular"):
                score -= 8.0

            if score < best_score:
                best_score = score
                best = entry

        return best

    def _prefer_serve_contact_launch_candidate(self, candidate_meta, frame_shape):
        """At serve contact, prefer the ball launching in serve direction over the racket."""
        if self.ball_center is None or not candidate_meta:
            return None
        if self.last_motion is None or not self._is_descending_serve_contact_motion(self.last_motion):
            return None
        if not hasattr(self, 'serve_area_y_min') or not hasattr(self, 'serve_area_y_max'):
            return None

        prev_x, prev_y = self.ball_center
        frame_height = frame_shape[0]
        serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
        contact_y = self.serve_area_y_min + int(serve_height * self.serve_contact_y_ratio)
        contact_margin = max(55, min(130, int(serve_height * 0.18)))
        if prev_y < contact_y - contact_margin:
            return None

        prev_size = float(self.ball_size or 0.0)
        if prev_size and prev_size < max(40.0, float(self.serve_contact_min_ball_size) * 0.35):
            return None

        last_dist = float(self.last_motion.get('distance', 0.0) or 0.0)
        min_forward_dx = max(55.0, float(self.serve_contact_min_dx) * 0.55)
        min_forward_dy = max(40.0, frame_height * 0.018)
        min_launch_dist = max(90.0, last_dist * 1.55)
        max_launch_dist = max(240.0, min(float(self.max_ball_speed), last_dist * 6.0))
        min_area = max(8.0, min(35.0, prev_size * 0.055 if prev_size else 8.0))
        max_area = max(90.0, prev_size * 1.35 if prev_size else 120.0)

        launch_candidates = []
        for entry in candidate_meta:
            cx, cy = entry['pos']
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            signed_dx = self._signed_serve_dx(dx)
            if signed_dx < min_forward_dx:
                continue
            if self.serve_direction_dy != 0:
                signed_dy = self._signed_serve_dy(dy)
                if signed_dy < min_forward_dy:
                    continue
            else:
                signed_dy = 0.0

            move_dist = float(entry.get('distance', 0.0) or 0.0)
            if move_dist < min_launch_dist or move_dist > max_launch_dist:
                continue

            area = float(entry.get('area', 0.0) or 0.0)
            if area < min_area or area > max_area:
                continue

            motion_mean = float(entry.get('motion_mean', 0.0) or 0.0)
            motion_max = float(entry.get('motion_max', 0.0) or 0.0)
            if motion_max < 55.0 and motion_mean < 10.0 and area < 18.0:
                continue

            upward_bonus = signed_dy if self.serve_direction_dy != 0 else max(0.0, -dy)
            area_match = abs(area - prev_size) / max(prev_size, 1.0) if prev_size else 0.0
            adjusted_score = (
                signed_dx * 1.15 +
                upward_bonus * 1.05 +
                min(160.0, area * 0.45) +
                min(90.0, motion_max * 0.45) +
                min(55.0, motion_mean * 1.6) -
                move_dist * 0.22 -
                area_match * 24.0
            )
            launch_candidates.append((adjusted_score, entry, signed_dx, upward_bonus))

        if not launch_candidates:
            return None

        _, chosen, signed_dx, signed_dy = max(launch_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [SERVE-CONTACT LAUNCH] prioritizing serve-direction candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px signed_dx={signed_dx:.1f} "
            f"signed_dy={signed_dy:.1f} score={chosen['score']:.1f} motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f}"
        )
        return chosen

    def _prefer_post_bounce_racket_launch_candidate(self, candidate_meta, frame_shape):
        """After a low bounce near the player, prefer the outgoing hit over racket follow-through."""
        if self.ball_center is None or self.last_motion is None or not candidate_meta:
            return None
        if getattr(self, '_ground_bounce_grace_frames', 0) <= 0:
            return None
        if getattr(self, 'ground_bounce_count', 0) <= 0:
            return None

        frame_height, frame_width = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        bounce_origin = getattr(self, '_ground_bounce_origin', None)
        if bounce_origin is None:
            return None
        origin_y = int(bounce_origin[1])
        if origin_y < max(980, int(frame_height * 0.48)):
            return None
        if prev_y < max(780, int(frame_height * 0.38)):
            return None

        last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        last_dist = float(self.last_motion.get('distance', 0.0) or 0.0)
        if last_dy > -55.0 or last_dist < 85.0 or abs(last_dx) < 25.0:
            return None

        launch_candidates = []
        max_launch_dist = max(260.0, min(float(self.max_ball_speed), last_dist * 2.2))
        min_reverse_dx = max(55.0, min(115.0, abs(last_dx) * 0.70))
        min_upward = max(70.0, min(145.0, abs(last_dy) * 0.70))
        min_area = max(10.0, min(float(self.ball_size or 0.0) * 0.20, 28.0))
        max_area = max(120.0, min(float(self.ball_size or 0.0) * 5.0, 260.0))

        for entry in candidate_meta:
            cx, cy = entry['pos']
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            move_dist = float(entry.get('distance', 0.0) or 0.0)
            area = float(entry.get('area', 0.0) or 0.0)
            motion_mean = float(entry.get('motion_mean', 0.0) or 0.0)
            motion_max = float(entry.get('motion_max', 0.0) or 0.0)

            if (last_dx * dx) >= -300.0:
                continue
            if abs(dx) < min_reverse_dx:
                continue
            if -dy < min_upward:
                continue
            if move_dist < max(95.0, last_dist * 0.65) or move_dist > max_launch_dist:
                continue
            if area < min_area or area > max_area:
                continue
            if motion_max < 75.0 and motion_mean < 18.0:
                continue
            if not (0 <= cx < frame_width and 0 <= cy < frame_height):
                continue

            _, _, w, h = cv2.boundingRect(entry['contour'])
            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect > 4.2:
                continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None and motion_max < 120.0 and motion_mean < 35.0:
                continue

            area_match = abs(area - float(self.ball_size or area)) / max(float(self.ball_size or area), 1.0)
            adjusted_score = (
                abs(dx) * 1.0 +
                (-dy) * 1.25 +
                min(170.0, area * 0.45) +
                min(100.0, motion_max * 0.45) +
                min(65.0, motion_mean * 1.6) -
                move_dist * 0.18 -
                area_match * 18.0
            )
            if entry.get('source') in ('primary', 'regular'):
                adjusted_score += 12.0
            launch_candidates.append((adjusted_score, entry, dx, dy))

        if not launch_candidates:
            return None

        _, chosen, dx, dy = max(launch_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [POST-BOUNCE RACKET LAUNCH] prioritizing outgoing candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px dx={dx:.1f} dy={dy:.1f} "
            f"score={chosen['score']:.1f} motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f}"
        )
        return chosen

    def _prefer_upper_far_player_escape_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Avoid following the far player's racket when the ball peels left after contact."""
        if self.ball_center is None or not candidate_meta or selected_contour is None:
            return None
        if self.last_motion is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        if (
                prev_x < int(frame_width * 0.55) or
                prev_y > max(320, int(frame_height * 0.16)) or
                prev_y < max(120, int(frame_height * 0.055)) or
                prev_size > 35.0):
            return None

        lm_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if lm_dist < 3.0 or lm_dist > 55.0:
            return None

        selected = None
        for entry in candidate_meta:
            if entry.get("contour") is selected_contour:
                selected = entry
                break
        if selected is None or not selected.get("predicted_path_hotspot_override"):
            return None

        selected_pos = selected["pos"]
        selected_distance = float(selected.get("distance", 0.0) or 0.0)
        selected_dx = selected_pos[0] - prev_x
        selected_area = float(selected.get("area", 0.0) or 0.0)
        if selected_distance > max(62.0, lm_dist * 2.8):
            return None
        if selected_dx < -8:
            return None

        escape_candidates = []
        for entry in candidate_meta:
            if entry is selected:
                continue
            if entry.get("predicted_path_hotspot_override"):
                continue
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            cx, cy = entry["pos"]
            area = float(entry.get("area", 0.0) or 0.0)
            distance = float(entry.get("distance", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            dx = cx - prev_x
            dy = cy - prev_y

            if dx > selected_dx - max(28.0, selected_distance * 0.85):
                continue
            if distance < max(16.0, selected_distance + 10.0):
                continue
            if distance > max(105.0, lm_dist * 4.0):
                continue
            if cy < prev_y - 75 or cy > prev_y + 32:
                continue
            if area < max(8.0, selected_area * 1.7, prev_size * 2.0):
                continue
            if area > max(55.0, prev_size * 8.0):
                continue
            if motion_max < 70.0 and motion_mean < 12.0:
                continue

            adjusted_score = (
                distance * 0.35 +
                abs(dy) * 0.45 -
                min(90.0, area * 3.2) -
                min(80.0, motion_max * 0.45) -
                min(35.0, motion_mean * 1.4)
            )
            escape_candidates.append((adjusted_score, entry))

        if not escape_candidates:
            return None

        _, chosen = min(escape_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [UPPER FAR PLAYER ESCAPE] preferring ball candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over predicted hotspot {selected_pos}"
        )
        return chosen

    def _prefer_upper_dynamic_departure_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Prefer a moving upper-court ball over a tiny stale patch after contact."""
        if self.ball_center is None or not candidate_meta or selected_contour is None:
            return None
        if self.last_motion is None:
            return None

        frame_height, _ = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        last_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if (
                prev_y > max(280, int(frame_height * 0.14)) or
                prev_size > 12.0 or
                last_dist > 18.0):
            return None

        selected = None
        for entry in candidate_meta:
            if entry.get("contour") is selected_contour:
                selected = entry
                break
        if selected is None:
            return None

        selected_area = float(selected.get("area", 0.0) or 0.0)
        selected_motion_mean = float(selected.get("motion_mean", 0.0) or 0.0)
        selected_motion_max = float(selected.get("motion_max", 0.0) or 0.0)
        selected_distance = float(selected.get("distance", 0.0) or 0.0)
        if (
                selected_area > max(6.0, prev_size * 6.0) or
                selected_motion_mean > 3.0 or
                selected_motion_max > 25.0):
            return None

        departure_candidates = []
        for entry in candidate_meta:
            if entry is selected:
                continue
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            cx, cy = entry["pos"]
            area = float(entry.get("area", 0.0) or 0.0)
            distance = float(entry.get("distance", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)

            if area < max(10.0, selected_area * 3.0, prev_size * 8.0):
                continue
            if area > 90.0:
                continue
            if distance < max(28.0, selected_distance + 10.0):
                continue
            if distance > max(120.0, last_dist * 14.0):
                continue
            if cy > prev_y + 20 or cy > selected["pos"][1] - 28:
                continue
            if abs(dx) < 18.0 and abs(dy) < 18.0:
                continue
            if motion_max < 90.0 or motion_mean < 14.0:
                continue

            adjusted_score = (
                distance * 0.30 +
                abs(dy) * 0.20 -
                min(95.0, area * 2.4) -
                min(90.0, motion_max * 0.42) -
                min(45.0, motion_mean * 1.2)
            )
            if entry.get("source") in ("primary", "regular"):
                adjusted_score -= 10.0
            departure_candidates.append((adjusted_score, entry))

        if not departure_candidates:
            return None

        _, chosen = min(departure_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [UPPER DYNAMIC DEPARTURE] preferring moving candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over stale patch {selected['pos']}"
        )
        return chosen

    def _find_single_standard_candidate(self, search_frame, x1, y1, prev_pos, predicted_point, frame_gray):
        if prev_pos is None:
            return None

        lower, upper, label = self.get_standard_hsv_for_position(prev_pos[1])
        if label not in {"regular_court", "primary"} or lower is None or upper is None:
            return None

        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        contact_bounds = self._contact_reacquire_bounds(frame_gray.shape if frame_gray is not None else search_frame.shape, prev_pos)
        max_area = max(150, self.serve_ball_size_max)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1 or area > max_area:
                continue
            if self.ball_size and self.ball_size > 40 and area < max(5, int(self.ball_size * 0.08)):
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1

            motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
            motion_mean = motion_metrics["mean"] if motion_metrics is not None else 0.0
            motion_max = motion_metrics["max"] if motion_metrics is not None else 0.0
            distance = math.hypot(cx - prev_pos[0], cy - prev_pos[1])
            predicted_distance = (
                math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                if predicted_point is not None else None
            )

            ignored_entry = self._find_ignored_tracking_position((cx, cy), filter_key="regular_court")
            ignored_ball_override = (
                ignored_entry is not None and
                motion_max >= 40.0 and
                motion_mean >= 10.0 and
                (predicted_distance is None or predicted_distance <= 40.0)
            )
            if ignored_entry is not None and not ignored_ball_override:
                continue
            if ignored_ball_override:
                print(
                    f"  DEBUG: Single regular candidate bypassing ignored hotspot at ({cx},{cy}) "
                    f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}"
                )

            relaxed_min_y = None
            if contact_bounds is not None:
                relaxed_min_y = self._contact_reacquire_min_y(
                    contact_bounds, predicted_distance, motion_mean, motion_max
                )
                if (cx < contact_bounds["min_x"] or
                        cx > contact_bounds["max_x"] or
                        cy < relaxed_min_y):
                    continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0:
                continue
            if cy < 100 and motion_mean < 2.5 and motion_max < 10.0:
                continue

            size_ratio = 0.0
            if self.ball_size and self.ball_size > 0:
                size_ratio = abs(area - self.ball_size) / self.ball_size
            score = distance + (size_ratio * 30.0)
            if predicted_distance is not None:
                score += predicted_distance * 0.4

            valid.append({
                "pos": (cx, cy),
                "area": area,
                "score": score,
                "distance": distance,
                "predicted_distance": predicted_distance,
                "contour": contour,
            })

        if not valid:
            return None

        if len(valid) > 1:
            xs = [entry["pos"][0] for entry in valid]
            ys = [entry["pos"][1] for entry in valid]
            if (max(xs) - min(xs) <= 8) and (max(ys) - min(ys) <= 8):
                candidate = min(
                    valid,
                    key=lambda entry: (
                        -entry["area"],
                        entry["score"],
                        entry["distance"],
                        entry["predicted_distance"] if entry["predicted_distance"] is not None else float("inf"),
                    )
                )
                print(
                    f"  DEBUG: Single regular candidate collapsing {len(valid)} adjacent components "
                    f"-> {candidate['pos']} area={candidate['area']:.1f}"
                )
            else:
                return None
        else:
            candidate = valid[0]

        print(
            f"  DEBUG: Single regular candidate -> {candidate['pos']} "
            f"area={candidate['area']:.1f} score={candidate['score']:.1f}"
        )
        return candidate

    def _activate_regular_hsv(self):
        self.using_alt_hsv = False
        self.using_alt2_hsv = False
        self.using_alt3_hsv = False
        self.using_alt4_hsv = False
        self.using_alt6_hsv = False
        if self.hsv_regular is not None:
            self.hsv_lower = self.hsv_regular['lower']
            self.hsv_upper = self.hsv_regular['upper']
        elif self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
            self.hsv_lower = self.primary_hsv_lower
            self.hsv_upper = self.primary_hsv_upper

    def _should_use_single_regular_candidate(self, regular_single, predicted_point, search_radius,
                                             current_pos=None, current_area=0.0,
                                             current_score=None, current_distance=None,
                                             current_predicted_distance=None):
        if regular_single is None or self.ball_center is None:
            return False

        min_regular_area = 4.0
        if self.ball_size is not None and self.ball_size > 0:
            min_regular_area = max(min_regular_area, min(18.0, self.ball_size * 0.28))
        if current_area is not None and current_area > 0:
            min_regular_area = max(
                min_regular_area,
                min(14.0, current_area * 0.38),
            )
        if regular_single["area"] < min_regular_area:
            return False

        regular_predicted_distance = regular_single.get("predicted_distance")
        if current_pos is None:
            if search_radius is None:
                search_radius = float(self.max_ball_speed or 95)
            distance_cap = max(95.0, min(float(search_radius), 135.0))
            predicted_cap = max(28.0, min(72.0, float(search_radius) * 0.22))
            if regular_single["distance"] > distance_cap:
                return False
            if (predicted_point is not None and regular_predicted_distance is not None and
                    regular_predicted_distance > predicted_cap):
                return False
            return True

        regular_pos_gap = math.hypot(
            regular_single["pos"][0] - current_pos[0],
            regular_single["pos"][1] - current_pos[1],
        )
        return (
            regular_pos_gap <= 24.0 and
            regular_single["score"] <= current_score + 30.0 and
            regular_single["distance"] <= current_distance + 16.0 and
            (
                predicted_point is None or
                current_predicted_distance is None or
                regular_predicted_distance is None or
                regular_predicted_distance <= current_predicted_distance + 20.0
            )
        )

    def _find_ignored_tracking_position(self, pos, filter_key=None):
        px, py = pos
        for entry in getattr(self, '_ignored_tracking_positions', []):
            dist = math.hypot(px - entry['pos'][0], py - entry['pos'][1])
            if dist <= entry['radius']:
                return entry
        if filter_key is not None:
            return self._find_persistent_false_point(pos, filter_key=filter_key)
        return None

    def _predicted_path_hotspot_override(self, pos, area, prev_pos, predicted_point, frame_gray):
        """Allow a learned hotspot only when it is tightly on the live ball trajectory."""
        if prev_pos is None or predicted_point is None:
            return False, None

        motion = self.last_motion
        if motion is None or float(motion.get('distance', 0.0) or 0.0) <= 0:
            motion = getattr(self, 'last_nonzero_motion', None)
        if motion is None:
            return False, None

        lm_dx = float(motion.get('dx', 0.0) or 0.0)
        lm_dy = float(motion.get('dy', 0.0) or 0.0)
        lm_dist = float(motion.get('distance', 0.0) or 0.0)
        if lm_dist < 6.0 or lm_dist > 120.0:
            return False, None

        px, py = pos
        mv_dx = float(px - prev_pos[0])
        mv_dy = float(py - prev_pos[1])
        prev_distance = math.hypot(mv_dx, mv_dy)
        if prev_distance < 3.0:
            return False, None

        predicted_distance = math.hypot(px - predicted_point[0], py - predicted_point[1])
        predicted_cap = max(10.0, min(36.0, lm_dist * 0.9 + 6.0))
        prev_cap = max(18.0, min(70.0, lm_dist * 1.7 + 10.0))
        if predicted_distance > predicted_cap or prev_distance > prev_cap:
            return False, None

        dot = lm_dx * mv_dx + lm_dy * mv_dy
        if dot < -0.10 * lm_dist * prev_distance:
            return False, None
        if abs(lm_dx) >= 6.0 and (mv_dx * lm_dx) < -8.0:
            return False, None
        if abs(lm_dy) >= 6.0 and (mv_dy * lm_dy) < -8.0:
            return False, None

        motion_metrics = self._candidate_motion_metrics(frame_gray, int(px), int(py))
        motion_mean = motion_metrics['mean'] if motion_metrics is not None else 0.0
        motion_max = motion_metrics['max'] if motion_metrics is not None else 0.0
        if area < 2.0 and motion_mean < 3.0 and motion_max < 12.0:
            return False, None

        return True, {
            'predicted_distance': predicted_distance,
            'predicted_cap': predicted_cap,
            'prev_distance': prev_distance,
            'prev_cap': prev_cap,
            'motion_mean': motion_mean,
            'motion_max': motion_max,
            'last_motion_distance': lm_dist,
        }

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
        if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
            specs.append(("s_30", self.s30_hsv_lower, self.s30_hsv_upper))
        if self.alts9_11_hsv_lower is not None and self.alts9_11_hsv_upper is not None:
            specs.append(("alts9_11", self.alts9_11_hsv_lower, self.alts9_11_hsv_upper))
        return specs

    def _build_sparse_low_s_mask(self, hsv_frame, lower, upper):
        if lower is None or upper is None:
            return None
        mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        return mask

    def _should_try_alts9_11_override(self, frame_shape, allow_inactive):
        if allow_inactive:
            return False
        if self.alts9_11_hsv_lower is None or self.alts9_11_hsv_upper is None:
            return False
        if self.ball_center is None or self.last_motion is None:
            return False
        if self.ball_center[1] > max(300, int(frame_shape[0] * 0.14)):
            return False
        if self.ball_size is not None and self.ball_size > 20:
            return False
        return self.last_motion.get('distance', 0.0) <= 8.0

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

    def _collect_override_candidate_metrics(self, pos, area, prev_pos, predicted_point, frame_gray):
        motion_metrics = self._candidate_motion_metrics(frame_gray, pos[0], pos[1])
        motion_mean = motion_metrics['mean'] if motion_metrics is not None else 0.0
        motion_max = motion_metrics['max'] if motion_metrics is not None else 0.0
        prev_distance = math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) if prev_pos is not None else 0.0
        predicted_distance = (
            math.hypot(pos[0] - predicted_point[0], pos[1] - predicted_point[1])
            if predicted_point is not None else None
        )
        frame0_hotspot = self._find_frame0_background_hotspot(pos)
        static_hotspot = (
            (area <= 3 and motion_mean < 1.0 and motion_max <= 5.0) or
            (pos[1] < 100 and motion_mean < 2.5 and motion_max < 10.0)
        )
        frame0_background = (
            frame0_hotspot is not None and
            (motion_metrics is None or (motion_mean < 8.0 and motion_max < 35.0))
        )
        return {
            'motion_mean': motion_mean,
            'motion_max': motion_max,
            'prev_distance': prev_distance,
            'predicted_distance': predicted_distance,
            'frame0_hotspot': frame0_hotspot,
            'frame0_background': frame0_background,
            'static_hotspot': static_hotspot,
        }

    def _should_accept_hsv_override(self, label, override, current_pos, current_area, prev_pos,
                                    predicted_point, frame_gray, current_filter_key=None):
        if override is None or current_pos is None or prev_pos is None:
            return override is not None

        override_metrics = self._collect_override_candidate_metrics(
            override['pos'], override['area'], prev_pos, predicted_point, frame_gray
        )
        if override_metrics['static_hotspot'] or override_metrics['frame0_background']:
            reason = (
                f"static patch mean={override_metrics['motion_mean']:.1f} "
                f"max={override_metrics['motion_max']:.1f}"
            )
            hotspot = override_metrics['frame0_hotspot']
            if override_metrics['frame0_background'] and hotspot is not None:
                hotspot_pos = tuple(hotspot.get('pos', list(override['pos'])))
                hotspot_radius = hotspot.get('radius', 0)
                reason = f"frame0 hotspot at {hotspot_pos} r={hotspot_radius}"
            print(f"  DEBUG: Rejecting {label} override at {override['pos']} - {reason}")
            return False

        frame_height = frame_gray.shape[0] if frame_gray is not None else 0
        recent_upper_return = (
            self._recent_offscreen_return_hold_active(window_frames=36) or
            getattr(self, '_top_return_reentry_grace_frames', 0) > 0
        )
        if recent_upper_return and frame_height:
            upper_path_limit = max(360, int(frame_height * 0.18))
            override_dx = float(override['pos'][0] - prev_pos[0])
            override_dy = float(override['pos'][1] - prev_pos[1])
            override_step = math.hypot(override_dx, override_dy)
            weak_override_motion = (
                override_metrics['motion_mean'] < 2.5 and
                override_metrics['motion_max'] < 12.0
            )
            if (
                    prev_pos[1] <= upper_path_limit and
                    override_step >= max(180.0, frame_height * 0.10) and
                    override_dy >= max(120.0, frame_height * 0.055) and
                    weak_override_motion):
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"static post-return jump from {prev_pos} "
                    f"motion={override_metrics['motion_mean']:.1f}/{override_metrics['motion_max']:.1f}"
                )
                return False

        current_metrics = self._collect_override_candidate_metrics(
            current_pos, current_area, prev_pos, predicted_point, frame_gray
        )
        near_camera_track_lock = self._near_camera_large_continuation_lock(
            current_pos,
            current_area,
            prev_pos,
            frame_gray.shape if frame_gray is not None else None,
        )
        current_dynamic = (
            current_metrics['motion_max'] >= 20.0 or
            current_metrics['motion_mean'] >= 6.0
        )
        override_weaker_motion = (
            override_metrics['motion_max'] + 15.0 < current_metrics['motion_max'] and
            override_metrics['motion_mean'] + 3.0 < current_metrics['motion_mean']
        )
        override_farther_from_prev = (
            override_metrics['prev_distance'] > current_metrics['prev_distance'] + 35.0
        )
        override_farther_from_predicted = (
            predicted_point is not None and
            current_metrics['predicted_distance'] is not None and
            override_metrics['predicted_distance'] is not None and
            override_metrics['predicted_distance'] > current_metrics['predicted_distance'] + 25.0
        )
        override_much_smaller = (
            current_area is not None and current_area >= 8.0 and
            override.get('area', 0.0) <= max(3.0, current_area * 0.35)
        )
        override_area = float(override.get('area', 0.0) or 0.0)
        if near_camera_track_lock is not None:
            last_dist = near_camera_track_lock['last_dist']
            current_step = current_metrics['prev_distance']
            override_step = override_metrics['prev_distance']
            override_farther_lower_blob = (
                override_step >= current_step + max(70.0, last_dist * 0.45) and
                override['pos'][1] >= current_pos[1] + 70 and
                override_area <= max(90.0, float(current_area or 0.0) * 0.88)
            )
            if override_farther_lower_blob:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping large near-camera continuation {current_pos} "
                    f"area={float(current_area or 0.0):.1f}px"
                )
                return False
        if (
                label in ("alt1", "alt2", "alt4", "alt6") and
                frame_height and
                prev_pos[1] <= max(320, int(frame_height * 0.16)) and
                predicted_point is not None and
                current_metrics['predicted_distance'] is not None and
                override_metrics['predicted_distance'] is not None
        ):
            current_near_predicted_lane = (
                current_metrics['predicted_distance'] <= max(28.0, current_metrics['prev_distance'] * 2.5)
            )
            override_leaves_lane = (
                override_metrics['prev_distance'] >= current_metrics['prev_distance'] + 35.0 and
                override_metrics['predicted_distance'] >= current_metrics['predicted_distance'] + 35.0
            )
            override_weak_motion = (
                override_metrics['motion_mean'] < 2.5 and
                override_metrics['motion_max'] < 12.0
            )
            override_is_tiny_blob = override_area <= max(3.0, float(current_area or 0.0) * 0.55)
            if (
                    current_near_predicted_lane and
                    override_leaves_lane and
                    (override_weak_motion or override_is_tiny_blob)):
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"upper-court jump away from predicted lane "
                    f"motion={override_metrics['motion_mean']:.1f}/{override_metrics['motion_max']:.1f}"
                )
                return False
            upper_lane_override_much_smaller = (
                current_area is not None and
                current_area >= 5.0 and
                override_area <= max(3.0, float(current_area) * 0.55)
            )
            upper_lane_smaller_override = (
                current_dynamic and
                upper_lane_override_much_smaller and
                override_metrics['prev_distance'] >=
                current_metrics['prev_distance'] + max(22.0, current_metrics['prev_distance']) and
                override_metrics['predicted_distance'] >=
                current_metrics['predicted_distance'] + max(16.0, current_metrics['predicted_distance'])
            )
            if current_near_predicted_lane and upper_lane_smaller_override:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping upper-court predicted-lane candidate {current_pos}"
                )
                return False
            upper_static_override_vs_dynamic_current = (
                current_dynamic and
                current_metrics['motion_mean'] >= 10.0 and
                current_metrics['motion_max'] >= 60.0 and
                override_weak_motion and
                override_metrics['prev_distance'] >= current_metrics['prev_distance'] - 12.0 and
                override_metrics['predicted_distance'] >= current_metrics['predicted_distance'] - 12.0
            )
            if upper_static_override_vs_dynamic_current:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping dynamic upper-court candidate {current_pos}"
                )
                return False
            upper_visible_ball_vs_static_override = (
                current_dynamic and
                current_area is not None and
                current_area >= 10.0 and
                current_metrics['motion_mean'] >= 14.0 and
                current_metrics['motion_max'] >= 80.0 and
                override_metrics['motion_mean'] < 2.5 and
                override_metrics['motion_max'] <= 25.0 and
                override_area <= max(5.0, float(current_area) * 0.35) and
                current_metrics['prev_distance'] <= override_metrics['prev_distance'] + 28.0
            )
            if upper_visible_ball_vs_static_override:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping visible moving upper-court ball {current_pos}"
                )
                return False
        if (label in ("alt2", "alt6") and
                getattr(self, 'stuck_frame_count', 0) > 0 and
                override_area <= max(8.0, float(current_area or 0.0) * 4.0) and
                predicted_point is not None and
                self.ball_center is not None and
                self.ball_size is not None and self.ball_size <= 20.0 and
                self.last_motion is not None and
                float(self.last_motion.get('dy', 0.0) or 0.0) < -2.0):
            frame_height = frame_gray.shape[0] if frame_gray is not None else 0
            upper_contact_limit = max(260, int(frame_height * 0.13)) if frame_height else 260
            if self.ball_center[1] <= upper_contact_limit:
                last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
                last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
                last_dist = float(self.last_motion.get('distance', 0.0) or 0.0)
                override_dx = float(override['pos'][0] - prev_pos[0])
                override_dy = float(override['pos'][1] - prev_pos[1])
                override_dot = last_dx * override_dx + last_dy * override_dy
                current_pred_distance = current_metrics['predicted_distance']
                override_pred_distance = override_metrics['predicted_distance']
                current_near_prediction = (
                    current_pred_distance is not None and
                    current_pred_distance <= max(42.0, last_dist * 1.25)
                )
                override_worse_prediction = (
                    current_pred_distance is not None and
                    override_pred_distance is not None and
                    override_pred_distance >= current_pred_distance + max(28.0, current_metrics['prev_distance'] * 0.8)
                )
                override_farther_step = (
                    override_metrics['prev_distance'] >= max(45.0, current_metrics['prev_distance'] * 2.2)
                )
                override_opposes_upper_flight = (
                    override_dy > max(6.0, abs(last_dy) * 0.15) or
                    override_dot < 0.0
                )
                if (current_near_prediction and override_opposes_upper_flight and
                        (override_worse_prediction or override_farther_step)):
                    print(
                        f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                        f"upper-return override loses predicted track from {current_pos}"
                    )
                    return False
        if override_much_smaller and (
                override_farther_from_prev or override_farther_from_predicted):
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                f"override area {override.get('area', 0.0):.1f}px too small vs current {current_area:.1f}px"
            )
            return False
        if (self._upper_slow_arc_active() and override_much_smaller and current_dynamic and
                override_area <= 3.0 and current_area is not None and current_area >= 6.0):
            current_pred = current_metrics['predicted_distance']
            override_pred = override_metrics['predicted_distance']
            override_not_better = (
                override_metrics['prev_distance'] >= current_metrics['prev_distance'] - 10.0 and
                (
                    predicted_point is None or
                    current_pred is None or
                    override_pred is None or
                    override_pred >= current_pred - 12.0
                )
            )
            if override_not_better:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping visible upper-arc candidate {current_pos}"
                )
                return False

        if self._upper_slow_arc_active() and override_much_smaller and current_dynamic:
            current_close_to_track = current_metrics['prev_distance'] <= 20.0
            current_close_to_prediction = (
                predicted_point is None or
                current_metrics['predicted_distance'] is None or
                current_metrics['predicted_distance'] <= 32.0
            )
            if current_close_to_track and current_close_to_prediction:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping dynamic same-track candidate {current_pos}"
                )
                return False

        if label == "alts9_11":
            current_jump = current_metrics['prev_distance'] >= 55.0
            override_track_lock = override_metrics['prev_distance'] + 30.0 < current_metrics['prev_distance']
            override_dynamic = (
                override_metrics['motion_max'] >= 18.0 or
                override_metrics['motion_mean'] >= 6.0
            )
            if current_jump and override_track_lock and override_dynamic:
                print(
                    f"  DEBUG: Accepting {label} override at {override['pos']} - "
                    f"closer to previous track than current jump {current_pos}"
                )
                return True
            reject_reason = "current pick is not a suspicious jump"
            if current_jump and not override_track_lock:
                reject_reason = "override is not close enough to previous track"
            elif current_jump and not override_dynamic:
                reject_reason = "override is too static"
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - {reject_reason}"
            )
            return False

        if current_dynamic and override_weaker_motion and (
                override_farther_from_prev or override_farther_from_predicted):
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                f"current pick {current_pos} is more dynamic and closer to track"
            )
            return False

        if current_filter_key == "regular_court":
            override_much_closer = (
                override_metrics['prev_distance'] + 18.0 < current_metrics['prev_distance']
            )
            if (predicted_point is not None and
                    current_metrics['predicted_distance'] is not None and
                    override_metrics['predicted_distance'] is not None and
                    override_metrics['predicted_distance'] + 15.0 < current_metrics['predicted_distance']):
                override_much_closer = True
            override_much_more_dynamic = (
                override_metrics['motion_max'] > current_metrics['motion_max'] + 20.0 or
                override_metrics['motion_mean'] > current_metrics['motion_mean'] + 4.0
            )
            if not override_much_closer and not override_much_more_dynamic:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping regular candidate {current_pos}"
                )
                return False

        return True

    def _contact_reacquire_bounds(self, frame_shape, reference_pos):
        """Return a plausible reacquire window for upper-court contact recovery."""
        if reference_pos is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        ref_x, ref_y = reference_pos
        upper_contact_limit = max(280, int(frame_height * 0.14))
        if (
            getattr(self, 'ground_bounce_count', 0) > 0 and
            self.last_motion is not None and
            self.last_motion.get('dy', 0.0) < -2.0 and
            (self.ball_size is None or self.ball_size <= 35)
        ):
            upper_contact_limit = max(340, int(frame_height * 0.17))
        if ref_y > upper_contact_limit:
            return None

        recent_max_vel = max(self.ball_velocity_history[-3:]) if getattr(self, 'ball_velocity_history', None) else 0
        if recent_max_vel > 120:
            return None

        upward_expand = 0.0
        if (self.last_motion is not None and
                self.last_motion.get('dy', 0.0) < -2.0 and
                (self.ball_size is None or self.ball_size <= 18)):
            upward_expand = min(28.0, max(0.0, -float(self.last_motion.get('dy', 0.0))))

        return {
            'max_dist': max(260, min(460, self.max_ball_speed + 40)),
            'min_y': max(0, int(round(ref_y - 45 - upward_expand))),
            'max_y': min(frame_height - 1, ref_y + max(220, int(frame_height * 0.12))),
            'min_x': max(0, ref_x - max(420, int(frame_width * 0.12))),
            'max_x': min(frame_width - 1, ref_x + max(420, int(frame_width * 0.12))),
        }

    def _contact_reacquire_min_y(self, contact_bounds, predicted_distance, motion_mean, motion_max):
        """Allow a modest upward extension when motion strongly supports the predicted path."""
        if contact_bounds is None:
            return None

        relaxed_min_y = contact_bounds["min_y"]
        if (predicted_distance is not None and predicted_distance <= 40.0 and
                motion_max >= 40.0 and motion_mean >= 10.0):
            relaxed_min_y = max(0, relaxed_min_y - 28)
        if (self._upper_slow_arc_active() and
                getattr(self, "ground_bounce_count", 0) > 0 and
                (self.ball_size is None or self.ball_size <= 35.0)):
            if motion_max >= 70.0 or (motion_mean >= 14.0 and motion_max >= 35.0):
                relaxed_min_y = max(0, relaxed_min_y - 90)
        return relaxed_min_y

    def _recent_upper_racket_top_exit_context(self, frame_shape, max_frames=45):
        """True when a recent upper-racket contact is now leaving through the top band."""
        if self.ball_center is None:
            return False
        contact_frame = getattr(self, '_last_racket_contact_frame', -1000000)
        frames_since_contact = self.frame_count - contact_frame
        if frames_since_contact < 0 or frames_since_contact > max_frames:
            return False

        contact_point = getattr(self, '_last_racket_contact_point', None)
        if contact_point is None:
            return False

        frame_height, _ = frame_shape[:2]
        ball_x, ball_y = self.ball_center
        contact_x, contact_y = contact_point
        if contact_y > max(380, int(frame_height * 0.20)):
            return False
        if ball_y > max(190, int(frame_height * 0.10)):
            return False
        if ball_y > contact_y - 55:
            return False
        if abs(ball_x - contact_x) > max(420, int(frame_shape[1] * 0.16)):
            return False
        if self.ball_size is not None and self.ball_size > 35:
            return False

        motion = self.last_motion
        if motion is None or motion.get('distance', 0.0) < 3.0:
            motion = getattr(self, 'last_nonzero_motion', None)
        if motion is None:
            return False

        dy = float(motion.get('dy', 0.0) or 0.0)
        dist = float(motion.get('distance', 0.0) or 0.0)
        recent_vel = max(self.ball_velocity_history[-3:]) if getattr(self, 'ball_velocity_history', None) else 0.0
        return dy <= -5.0 or dist >= 20.0 or recent_vel >= 18.0

    def _should_start_top_return_wait(self, frame_shape):
        """Return the top-return mode for long upper exits that can re-enter later."""
        if self.ball_center is None:
            return None
        motion = self.last_motion
        if motion is None or motion.get('distance', 0.0) < 3.0:
            motion = getattr(self, 'last_nonzero_motion', None)
        if motion is None:
            return None
        if getattr(self, '_top_return_reentry_grace_frames', 0) > 0:
            return None
        frame_height, _ = frame_shape[:2]
        x_prev, y_prev = self.ball_center
        dy = float(motion.get('dy', 0.0) or 0.0)
        dist = float(motion.get('distance', 0.0) or 0.0)
        predicted_y = y_prev + dy
        recent_vel = max(self.ball_velocity_history[-3:]) if getattr(self, 'ball_velocity_history', None) else 0.0
        ball_size = float(self.ball_size or 0.0)
        recent_upper_racket_exit = self._recent_upper_racket_top_exit_context(frame_shape)

        if recent_upper_racket_exit:
            clipped_recent_exit = (
                max(34, int(frame_height * 0.015)) <= y_prev <= 40 and
                dy < -12.0 and
                dist >= 18.0 and
                predicted_y < 12.0
            )
            if clipped_recent_exit:
                return "upper_racket"

        if y_prev <= 40 and dy < -14.0 and dist >= 20.0:
            if self.ground_bounce_count > 0 and (self.ball_size is None or self.ball_size <= 18.0):
                return "edge"
            if (
                dy <= -35.0 and
                dist >= 45.0 and
                predicted_y < 8.0 and
                (ball_size >= 60.0 or recent_vel >= 50.0)
            ):
                return "upper_side"
        projected_top_exit = (
            y_prev <= max(72, int(frame_height * 0.035)) and
            dy <= -35.0 and
            dist >= 45.0 and
            predicted_y <= max(8, int(frame_height * 0.006)) and
            (
                self.ground_bounce_count > 0 or
                ball_size >= 60.0 or
                recent_vel >= 50.0
            )
        )
        if projected_top_exit:
            return "edge"
        projected_top_exit_after_miss = (
            getattr(self, 'stuck_frame_count', 0) >= 1 and
            y_prev <= max(60, int(frame_height * 0.035)) and
            dy <= -14.0 and
            dist >= 20.0 and
            predicted_y <= max(18, int(frame_height * 0.012)) and
            (
                self.ground_bounce_count > 0 or
                ball_size <= 18.0 or
                recent_vel >= 25.0
            )
        )
        if projected_top_exit_after_miss:
            return "edge"
        upper_band_limit = max(120, int(frame_height * 0.11))
        strong_upper_exit = (
            y_prev <= upper_band_limit and
            dy <= -60.0 and
            dist >= 120.0 and
            predicted_y < -35.0
        )
        if strong_upper_exit and (
                self.ground_bounce_count > 0 or
                ball_size >= 80.0 or
                recent_vel >= 120.0):
            return "upper_side"
        return None

    def _should_start_top_edge_return_wait(self, frame_shape):
        """Return True once the tracked ball has actually clipped/exited the top edge."""
        if self.ball_center is None:
            return False
        if getattr(self, '_top_return_reentry_grace_frames', 0) > 0:
            return False
        frame_height, _ = frame_shape[:2]
        _, y_prev = self.ball_center
        motion = self.last_motion
        if motion is None or motion.get('distance', 0.0) < 3.0:
            motion = getattr(self, 'last_nonzero_motion', None)
        dy = float(motion.get('dy', 0.0) or 0.0) if motion is not None else 0.0
        predicted_y = float(y_prev) + dy
        clipped_edge_y = max(6, int(frame_height * 0.003))
        projected_edge_y = max(14, int(frame_height * 0.007))
        return (
            y_prev <= clipped_edge_y or
            (y_prev <= projected_edge_y and predicted_y < -4.0)
        )

    def _activate_offscreen_return_wait(self, edge, frame_shape, mode=None):
        """Use the shared off-screen return waiter for edge exits."""
        if edge != "top":
            return False
        self._activate_top_return_wait(mode or "edge")
        return self._top_return_wait_active()

    def _activate_top_return_wait(self, mode="edge"):
        """Arm a longer top-band wait so the tracker can catch a delayed re-entry."""
        if self.ball_center is None:
            return
        wait_frames = 80 if mode in ("upper_side", "upper_racket") else 60
        self._top_return_wait_frames = max(self._top_return_wait_frames, wait_frames)
        self._top_return_anchor = tuple(self.ball_center)
        self._top_return_origin_frame = self.frame_count
        self._top_return_mode = mode
        self._top_return_exit_dx = float(
            (self.last_motion.get('dx', 0.0) if self.last_motion is not None else 0.0) or 0.0
        )

    def _top_return_wait_active(self):
        anchor = getattr(self, '_top_return_anchor', None)
        return (
            getattr(self, '_top_return_wait_frames', 0) > 0 and
            anchor is not None and
            self.ball_center is not None
        )

    def _top_return_reentry_ok(self, pos, area, motion_mean, motion_max, frame_shape):
        """Validate a top-band re-entry candidate while waiting for a delayed return."""
        anchor = getattr(self, '_top_return_anchor', None)
        if anchor is None:
            return True, None

        cx, cy = pos
        frame_height, frame_width = frame_shape[:2]
        elapsed = max(0, self.frame_count - getattr(self, '_top_return_origin_frame', self.frame_count))
        mode = getattr(self, '_top_return_mode', 'edge')
        if mode in ("upper_side", "upper_racket"):
            exit_dx = float(getattr(self, '_top_return_exit_dx', 0.0) or 0.0)
            min_reentry_y = 56 if mode == "upper_racket" else 50
            partial_top_reentry_y = 14
            max_reentry_y = min(
                frame_height - 1,
                max(210 if mode == "upper_racket" else 150, anchor[1] + (170 if mode == "upper_racket" else 55)),
            )
            strong_motion = motion_max >= 45.0 or motion_mean >= 8.0
            strong_visible_reentry = (
                cy >= min_reentry_y and
                area >= 14.0 and
                motion_mean >= 18.0 and
                motion_max >= 90.0
            )
            # True upper-edge returns can appear as a clipped sliver; ordinary re-entry
            # must clear the top noise band before it can restart tracking.
            partial_top_reentry = (
                mode != "upper_racket" and
                elapsed >= 30 and
                0 <= cy < partial_top_reentry_y and
                area >= 8.0 and
                motion_mean >= 14.0 and
                motion_max >= 120.0
            )
            lane_min_x = int(frame_width * 0.18)
            lane_max_x = int(frame_width * 0.72)
            if abs(exit_dx) < 12.0:
                x_drift_cap = 260.0
            elif abs(exit_dx) < 28.0:
                x_drift_cap = max(360.0, min(560.0, abs(exit_dx) * 18.0))
            else:
                x_drift_cap = max(700.0, min(1100.0, max(abs(exit_dx) * 9.0, 900.0)))
            min_directional_progress = max(260.0, min(520.0, abs(exit_dx) * 8.0))
            blind_wait_frames = 20 if strong_visible_reentry else 30
            if strong_visible_reentry:
                min_directional_progress = min(min_directional_progress, 235.0)
            if elapsed < blind_wait_frames:
                return False, f"top-return blind wait elapsed={elapsed}f"
            if cx < lane_min_x or cx > lane_max_x:
                return False, f"top-return x {cx} outside top-band lane"
            if abs(cx - anchor[0]) > x_drift_cap:
                return False, f"top-return x drift {abs(cx - anchor[0]):.1f}px > {x_drift_cap:.1f}px"
            if exit_dx >= 20.0 and cx < anchor[0] + min_directional_progress:
                return False, f"top-return x {cx} lacks rightward reentry progress from {anchor[0]}"
            if exit_dx <= -20.0 and cx > anchor[0] - min_directional_progress:
                return False, f"top-return x {cx} lacks leftward reentry progress from {anchor[0]}"
            if exit_dx <= -40.0 and cx > anchor[0] + 120:
                return False, f"top-return x {cx} opposes leftward exit from {anchor[0]}"
            if exit_dx >= 40.0 and cx < anchor[0] - 120:
                return False, f"top-return x {cx} opposes rightward exit from {anchor[0]}"
            if cy < min_reentry_y and not partial_top_reentry:
                return False, f"top-return y {cy} < min_reentry_y {min_reentry_y}"
            if cy > max_reentry_y:
                return False, f"top-return y {cy} > max_reentry_y {max_reentry_y}"
            if not strong_motion:
                return False, f"top-return weak motion mean={motion_mean:.1f} max={motion_max:.1f} area={area:.1f}"
            return True, None

        min_reentry_y = max(50, anchor[1] + 18)
        base_max_reentry_y = max(110, int(frame_height * 0.11))
        dynamic_max_reentry_y = base_max_reentry_y
        if elapsed >= 22:
            dynamic_max_reentry_y = max(
                dynamic_max_reentry_y,
                int(anchor[1] + 150 + ((elapsed - 22) * 75)),
            )
        max_reentry_y = min(frame_height - 1, dynamic_max_reentry_y)
        deep_reentry_y = max(220, anchor[1] + 165)
        exit_dx = float(getattr(self, '_top_return_exit_dx', 0.0) or 0.0)
        if mode == "edge_clip":
            expected_x = anchor[0] + exit_dx * max(1.0, min(float(elapsed), 18.0))
            expected_x_gap = abs(cx - expected_x)
            top_edge_band_y = max(88, min(130, int(anchor[1]) + 92, int(frame_height * 0.060)))
            if (
                    elapsed >= 4 and
                    cy <= top_edge_band_y and
                    area >= 1.0 and
                    motion_max >= 55.0 and
                    motion_mean >= 6.0 and
                    expected_x_gap <= max(105.0, min(240.0, 85.0 + abs(exit_dx) * 3.5 + elapsed * 7.0))):
                return True, None
            return (
                False,
                f"top-edge clipped wait elapsed={elapsed} y={cy} "
                f"x_gap={expected_x_gap:.1f} motion={motion_mean:.1f}/{motion_max:.1f}"
            )
        if abs(exit_dx) < 12.0:
            x_cap = 260.0
        elif abs(exit_dx) < 25.0:
            x_cap = 340.0
        else:
            projected_x_cap = abs(exit_dx) * max(10.0, min(float(elapsed), 14.0))
            x_cap = max(320.0, min(900.0, projected_x_cap))
        deep_reentry = (
            elapsed >= 28 and
            cy >= deep_reentry_y and
            area >= 18.0
        )
        clipped_top_reentry = (
            elapsed >= 28 and
            cy >= max(24, anchor[1] - 32) and
            cy < min_reentry_y and
            area >= 8.0 and
            (motion_max >= 80.0 or motion_mean >= 18.0)
        )
        shallow_visible_reentry = (
            cy >= min_reentry_y and
            cy < deep_reentry_y and
            area >= 18.0 and
            (motion_max >= 70.0 or motion_mean >= 14.0)
        )
        strong_motion = (
            motion_max >= 40.0 or
            motion_mean >= 8.0 or
            deep_reentry or
            clipped_top_reentry or
            shallow_visible_reentry
        )
        strong_visible_reentry = (
            elapsed >= 24 and
            cy >= max(20, anchor[1] - 10) and
            area >= 70.0 and
            motion_mean >= 25.0 and
            motion_max >= 120.0
        )
        strong_partial_reentry = strong_visible_reentry and cy < min_reentry_y
        reentry_x_cap = x_cap
        if strong_visible_reentry:
            reentry_x_cap = max(reentry_x_cap, 420.0)
        if deep_reentry:
            reentry_x_cap = max(reentry_x_cap, 650.0)

        if elapsed < 24:
            return False, f"top-return blind wait elapsed={elapsed}f"
        if abs(cx - anchor[0]) > reentry_x_cap:
            return False, f"top-return x drift {abs(cx - anchor[0]):.1f}px > {reentry_x_cap:.1f}px"
        directional_progress = max(60.0, min(260.0, abs(exit_dx) * 3.0))
        if exit_dx >= 12.0 and cx < anchor[0] + directional_progress:
            return False, f"top-return x {cx} lacks rightward reentry progress from {anchor[0]}"
        if exit_dx <= -12.0 and cx > anchor[0] - directional_progress:
            return False, f"top-return x {cx} lacks leftward reentry progress from {anchor[0]}"
        if cy < min_reentry_y and not (strong_partial_reentry or clipped_top_reentry):
            return False, f"top-return y {cy} < min_reentry_y {min_reentry_y}"
        if cy > max_reentry_y:
            return False, f"top-return y {cy} > max_reentry_y {max_reentry_y}"
        if cy < deep_reentry_y and not (
            clipped_top_reentry or
            shallow_visible_reentry or
            strong_visible_reentry or
            strong_partial_reentry
        ):
            return False, (
                f"top-return shallow speck y={cy} area={area:.1f} "
                f"motion={motion_mean:.1f}/{motion_max:.1f}"
            )
        if not strong_motion:
            return False, f"top-return weak motion mean={motion_mean:.1f} max={motion_max:.1f} area={area:.1f}"
        return True, None

    def _build_top_return_search_region(self, frame_shape):
        """Build a thin top-band/lane search region for delayed upper re-entry."""
        frame_height, frame_width = frame_shape[:2]
        anchor = getattr(self, '_top_return_anchor', None) or self.ball_center
        if anchor is None:
            anchor = (frame_width // 2, 0)
        anchor_x, anchor_y = anchor
        mode = getattr(self, '_top_return_mode', 'edge')
        band_height = max(70, min(140, int(frame_height * 0.06)))

        if mode in ("upper_side", "upper_racket"):
            y2 = min(
                frame_height,
                max(band_height, min(230 if mode == "upper_racket" else 170, max(110, int(anchor_y) + (190 if mode == "upper_racket" else 70)))),
            )
            x1 = max(0, int(frame_width * 0.16))
            x2 = min(frame_width, int(frame_width * 0.74))
            center_x = (x1 + x2) // 2
        elif mode == "edge_clip":
            elapsed = max(0, self.frame_count - getattr(self, '_top_return_origin_frame', self.frame_count))
            y2 = min(
                frame_height,
                max(band_height, min(130, max(90, int(anchor_y) + 60))),
            )
            exit_dx = float(getattr(self, '_top_return_exit_dx', 0.0) or 0.0)
            projected_x = int(round(anchor_x + exit_dx * max(1.0, min(float(elapsed + 1), 18.0))))
            half_width = int(max(170, min(380, 120 + abs(exit_dx) * 3.2 + elapsed * 8.0)))
            x1 = max(0, projected_x - half_width)
            x2 = min(frame_width, projected_x + half_width)
            if x2 - x1 < 160:
                center = max(0, min(frame_width - 1, projected_x))
                x1 = max(0, center - 80)
                x2 = min(frame_width, center + 80)
            center_x = (x1 + x2) // 2 if x2 > x1 else frame_width // 2
        else:
            y2 = min(
                frame_height,
                max(band_height, min(130, max(90, int(anchor_y) + 60))),
            )
            exit_dx = float(getattr(self, '_top_return_exit_dx', 0.0) or 0.0)
            speed_extent = abs(exit_dx) * 14.0 if abs(exit_dx) >= 12.0 else self.max_ball_speed * 6.0
            half_width = int(max(240, min(920, speed_extent)))
            direction_margin = 50
            if exit_dx >= 12.0:
                x1 = max(0, int(anchor_x) + direction_margin)
                x2 = min(frame_width, int(anchor_x) + half_width)
                if x2 - x1 < 160:
                    x1 = max(0, x2 - 160)
            elif exit_dx <= -12.0:
                x1 = max(0, int(anchor_x) - half_width)
                x2 = min(frame_width, int(anchor_x) - direction_margin)
                if x2 - x1 < 160:
                    x2 = min(frame_width, x1 + 160)
            else:
                x1 = max(0, int(anchor_x) - half_width)
                x2 = min(frame_width, int(anchor_x) + half_width)
            center_x = (x1 + x2) // 2 if x2 > x1 else frame_width // 2

        if x2 <= x1:
            x1, x2 = 0, frame_width
            center_x = frame_width // 2

        y1 = 0
        if y2 <= y1:
            y2 = min(frame_height, band_height)
        center_y = int(max(y1 + 10, min(y2 - 1, y1 + ((y2 - y1) // 2)))) if y2 > y1 else 10
        return center_x, center_y, x1, y1, x2, y2

    def _find_top_return_motion_reentry(self, frame, frame_gray):
        """Find a delayed top re-entry by fresh motion when learned hotspots mask it."""
        if frame is None or frame_gray is None or not self._top_return_wait_active():
            return None
        mode = getattr(self, '_top_return_mode', 'edge')
        if mode not in ("upper_side", "upper_racket", "edge_clip"):
            return None

        _, _, x1, y1, x2, y2 = self._build_top_return_search_region(frame.shape)
        if x2 <= x1 or y2 <= y1:
            return None

        search_frame = frame[y1:y2, x1:x2]
        if search_frame.size == 0:
            return None
        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)

        filters = []
        seen = set()

        def add_filter(label, lower, upper):
            if lower is None or upper is None:
                return
            key = (label, tuple(int(v) for v in lower), tuple(int(v) for v in upper))
            if key in seen:
                return
            seen.add(key)
            filters.append((label, lower, upper))

        if self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
            add_filter("primary", self.primary_hsv_lower, self.primary_hsv_upper)
        if hasattr(self, 'hsv_regular') and self.hsv_regular is not None:
            add_filter("regular", self.hsv_regular['lower'], self.hsv_regular['upper'])
        add_filter("s_30", self.s30_hsv_lower, self.s30_hsv_upper)

        best = None
        best_score = float('inf')
        kernel = np.ones((2, 2), np.uint8)
        elapsed = max(0, self.frame_count - getattr(self, '_top_return_origin_frame', self.frame_count))
        anchor = getattr(self, '_top_return_anchor', None) or self.ball_center or (0, 0)
        exit_dx = float(getattr(self, '_top_return_exit_dx', 0.0) or 0.0)
        expected_x = anchor[0] + exit_dx * max(1.0, min(float(elapsed), 18.0))
        edge_max_y = y2 - 1

        for label, lower, upper in filters:
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = 1.0 if mode == "edge_clip" else 4.0
                if area < min_area or area > 90.0:
                    continue
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                if mode == "upper_side" and cy >= 14:
                    continue
                if mode == "edge_clip" and cy > edge_max_y:
                    continue
                motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                if motion_metrics is None:
                    continue
                motion_mean = motion_metrics['mean']
                motion_max = motion_metrics['max']
                if motion_mean < 6.0 or motion_max < 55.0:
                    continue
                ok, reason = self._top_return_reentry_ok(
                    (cx, cy), area, motion_mean, motion_max, frame.shape
                )
                if not ok:
                    continue

                if mode == "edge_clip":
                    score = abs(cx - expected_x) * 0.9
                    score += cy * 1.8
                else:
                    score = abs(cy - 28) * 2.2
                    score += abs(cx - (self._top_return_anchor[0] + self._top_return_exit_dx * 13.0)) * 0.05
                score -= min(220.0, motion_max * 1.2)
                score -= min(120.0, motion_mean * 3.0)
                score -= min(90.0, area * 4.0)
                if score < best_score:
                    local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
                    local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
                    best_score = score
                    best = {
                        'pos': (cx, cy),
                        'area': area,
                        'hsv': hsv_frame[local_y, local_x],
                        'label': label,
                        'motion_mean': motion_mean,
                        'motion_max': motion_max,
                        'score': score,
                    }

        return best

    def _commit_top_return_reentry(self, reentry, label):
        prev_pos = self.ball_center
        new_pos = reentry['pos']
        self.ball_center = new_pos
        self.ball_hsv = reentry['hsv']
        self.ball_size = reentry['area']
        self._update_recovered_motion(prev_pos, new_pos)
        self.stuck_frame_count = 0
        self._recent_offscreen_return_frame = self.frame_count
        self._top_return_reentry_grace_frames = max(
            getattr(self, '_top_return_reentry_grace_frames', 0), 4
        )
        self._top_return_wait_frames = 0
        self._top_return_anchor = None
        self._top_return_origin_frame = -1
        self._top_return_mode = None
        self._top_return_exit_dx = 0.0
        self.last_seen_frame = self.frame_count
        print(
            f"Frame {self.frame_count}: [{label}] Ball at {new_pos} "
            f"via {reentry['label']} motion={reentry['motion_mean']:.1f}/"
            f"{reentry['motion_max']:.1f}"
        )
        return self.ball_center

    def _active_top_return_overlay_region(self, frame_shape):
        if getattr(self, '_top_return_anchor', None) is None:
            return None
        if not (
            self._top_return_wait_active() or
            getattr(self, '_top_return_reentry_grace_frames', 0) > 0
        ):
            return None
        _, _, x1, y1, x2, y2 = self._build_top_return_search_region(frame_shape)
        return x1, y1, x2, y2

    def _prune_bounce_markers(self):
        self.recent_bounce_markers = [
            marker for marker in getattr(self, 'recent_bounce_markers', [])
            if marker.get('expires', -1) >= self.frame_count
        ]

    def _impact_marker_color(self, kind):
        palette = {
            'ground_bounce': (255, 0, 0),      # blue
            'serve_bounce': (255, 80, 0),      # blue-orange
            'racket_contact': (0, 165, 255),   # orange
            'net_contact': (0, 0, 255),        # red
            'fence_contact': (255, 0, 255),    # magenta
            'direction_change': (0, 255, 255), # yellow
        }
        return palette.get(kind, (255, 255, 0))

    def _add_impact_marker(self, point, kind="direction_change", ttl=7, color=None, label=None):
        if point is None:
            return False
        self._prune_bounce_markers()
        pos = (int(point[0]), int(point[1]))
        last_pos = getattr(self, '_last_impact_marker_pos', None)
        last_frame = getattr(self, '_last_impact_marker_frame', -1000000)
        last_kind = getattr(self, '_last_impact_marker_kind', None)
        if (
            last_pos is not None and
            last_kind == kind and
            (self.frame_count - last_frame) <= 3 and
            math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1]) <= 28
        ):
            return False
        if color is None:
            color = self._impact_marker_color(kind)
        self.recent_bounce_markers.append({
            'pos': pos,
            'color': tuple(int(v) for v in color),
            'kind': kind,
            'label': label,
            'start_frame': self.frame_count,
            'ttl': max(1, int(ttl)),
            'expires': self.frame_count + max(1, int(ttl)) - 1,
        })
        self._last_impact_marker_frame = self.frame_count
        self._last_impact_marker_pos = pos
        self._last_impact_marker_kind = kind
        return True

    def _add_bounce_marker(self, point, color, ttl=7):
        self._add_impact_marker(point, kind="ground_bounce", ttl=ttl, color=color)

    def _capture_point_end_visuals(self, end_position=None, hold_frames=None):
        hold = self.point_end_marker_hold_frames if hold_frames is None else hold_frames
        hold = max(0, int(hold))
        if hold <= 0:
            self._ended_point_visual_snapshot = None
            return

        pos = end_position if end_position is not None else self.ball_center
        ball_center = (int(pos[0]), int(pos[1])) if pos is not None else None
        ball_hsv = self.ball_hsv.copy() if isinstance(self.ball_hsv, np.ndarray) else self.ball_hsv
        hsv_lower = self.hsv_lower.copy() if isinstance(self.hsv_lower, np.ndarray) else self.hsv_lower
        hsv_upper = self.hsv_upper.copy() if isinstance(self.hsv_upper, np.ndarray) else self.hsv_upper
        expires_frame = self.frame_count + hold

        held_markers = []
        for marker in getattr(self, 'recent_bounce_markers', []):
            held_marker = dict(marker)
            held_marker['start_frame'] = self.frame_count
            held_marker['ttl'] = hold + 1
            held_marker['expires'] = expires_frame
            held_markers.append(held_marker)

        self._ended_point_visual_snapshot = {
            'expires_frame': expires_frame,
            'ball_center': ball_center,
            'ball_size': self.ball_size,
            'ball_hsv': ball_hsv,
            'hsv_lower': hsv_lower,
            'hsv_upper': hsv_upper,
            'direction_change_events': [dict(event) for event in getattr(self, 'direction_change_events', [])],
            'net_contact_points': [tuple(point) for point in getattr(self, 'net_contact_points', [])],
            'recent_bounce_markers': held_markers,
        }

    def _active_point_end_visuals(self):
        snapshot = getattr(self, '_ended_point_visual_snapshot', None)
        if not snapshot:
            return None
        if self.frame_count <= snapshot.get('expires_frame', -1):
            return snapshot
        self._ended_point_visual_snapshot = None
        return None

    def _is_fence_contact_candidate(self, prev_pos, new_pos, frame_shape, dx, dy, angle_jump, speed_ratio):
        if prev_pos is None or self.last_motion is None:
            return False
        frame_height, frame_width = frame_shape[:2]
        prev_dx = float(self.last_motion.get('dx', 0.0))
        prev_dy = float(self.last_motion.get('dy', 0.0))
        px, py = prev_pos
        nx, ny = new_pos
        side_margin = max(35, int(frame_width * 0.025))
        top_margin = max(60, int(frame_height * 0.10))
        bottom_margin = max(45, int(frame_height * 0.06))

        near_left = min(px, nx) <= side_margin
        near_right = max(px, nx) >= (frame_width - side_margin)
        near_top = min(py, ny) <= top_margin or py <= max(110, int(frame_height * 0.18))
        near_bottom = max(py, ny) >= (frame_height - bottom_margin)

        reversed_x = prev_dx * dx < -12 and abs(prev_dx) >= 6 and abs(dx) >= 6
        reversed_y = prev_dy * dy < -12 and abs(prev_dy) >= 6 and abs(dy) >= 6
        if (near_left or near_right) and reversed_x and angle_jump >= 45:
            return True
        if (near_top or near_bottom) and reversed_y and (angle_jump >= 35 or speed_ratio <= 0.95):
            return True
        return False

    def _classify_direction_change_impact(
        self,
        new_pos,
        frame_shape,
        dx,
        dy,
        angle_jump,
        speed_ratio,
        serve_contact_grace=False,
        predicted_turn_candidate=False,
        predicted_continuation_candidate=False,
        lower_contact_launch_candidate=False,
        lower_contact_launch_context=None,
        ground_bounce_candidate=False,
        ground_bounce_context=None,
    ):
        prev_pos = self.ball_center
        if ground_bounce_candidate and ground_bounce_context is not None:
            return {
                'kind': 'ground_bounce',
                'point': ground_bounce_context['origin'],
                'label': 'ground bounce',
            }
        if serve_contact_grace and self.last_motion is not None:
            prev_dy = float(self.last_motion.get('dy', 0.0))
            serve_launch_turn = (
                prev_dy >= 12.0 and
                dy <= -12.0 and
                angle_jump >= 70.0
            )
            serve_launch_spike = angle_jump >= 110.0 or speed_ratio >= 1.8
            if serve_launch_turn or serve_launch_spike:
                return {
                    'kind': 'racket_contact',
                    'point': prev_pos or new_pos,
                    'label': 'serve/racket contact',
                }
        if lower_contact_launch_candidate and lower_contact_launch_context is not None:
            return {
                'kind': 'racket_contact',
                'point': lower_contact_launch_context['origin'],
                'label': 'racket contact',
            }

        probe_points = [p for p in (prev_pos, new_pos) if p is not None]
        for probe in probe_points:
            net_geometry = self._net_contact_geometry(probe)
            if net_geometry is None:
                continue
            direct_contact_zone = net_geometry['inside'] or net_geometry['near_top_tape']
            if direct_contact_zone and (angle_jump >= 45 or speed_ratio <= 0.90):
                return {
                    'kind': 'net_contact',
                    'point': probe,
                    'label': 'net contact',
                }

        if self._is_fence_contact_candidate(prev_pos, new_pos, frame_shape, dx, dy, angle_jump, speed_ratio):
            return {
                'kind': 'fence_contact',
                'point': prev_pos or new_pos,
                'label': 'fence contact',
            }

        if predicted_turn_candidate or predicted_continuation_candidate:
            return None
        if angle_jump >= 75 or speed_ratio > 1.8 or speed_ratio < 0.55:
            return {
                'kind': 'direction_change',
                'point': prev_pos or new_pos,
                'label': 'direction change',
            }
        return None

    def _start_point_context(self, origin_pos):
        self._pending_rally_end_reason = None
        self._pending_rally_end_frame = -1
        self._awaiting_serve_bounce = False
        self._point_serve_start_side = None
        self._point_target_service_side = None
        if origin_pos is None or not hasattr(self, 'serve_area_x_min'):
            return
        serve_mid_x = (self.serve_area_x_min + self.serve_area_x_max) / 2.0
        start_side = "left" if origin_pos[0] < serve_mid_x else "right"
        target_side = "right" if start_side == "left" else "left"
        self._point_serve_start_side = start_side
        self._point_target_service_side = target_side
        self._awaiting_serve_bounce = True

    def _classify_ground_bounce(self, point, frame, conservative_sideline_override=False):
        outside_singles, side, left_x, right_x = self._point_outside_singles_sidelines(point, frame)
        if outside_singles:
            outside_px = (
                (float(left_x) - float(point[0]))
                if side == "left" else
                (float(point[0]) - float(right_x))
            )
            if conservative_sideline_override and outside_px > 42.0:
                return False, f"Ball bounce outside singles court ({side} sideline)", (0, 0, 255)
            if self._sideline_line_contact_override(
                    point, side, left_x, right_x, frame, self.ball_size,
                    emit_debug=True):
                return True, f"Bounce on singles {side} sideline", (255, 0, 0)
            return False, f"Ball bounce outside singles court ({side} sideline)", (0, 0, 255)

        upper_recent_return_bounce = (
            not getattr(self, '_awaiting_serve_bounce', False) and
            point[1] <= max(320, int(frame.shape[0] * 0.16)) and
            (
                self._recent_offscreen_return_hold_active(window_frames=12) or
                self._recent_return_bounce_recover_active(window_frames=3)
            )
        )
        if upper_recent_return_bounce:
            outside_far_baseline, far_y = self._point_outside_top_singles_baseline(point, frame)
            if outside_far_baseline:
                print(
                    f"Frame {self.frame_count}: [TOP BASELINE OUT] point={point} "
                    f"far_baseline_y={far_y:.1f}"
                )
                return False, "Ball bounce outside singles court (far baseline)", (0, 0, 255)

        if getattr(self, '_awaiting_serve_bounce', False):
            serve_bounce_window_active = (
                self.point_start_frame_internal is not None and
                (self.frame_count - self.point_start_frame_internal) <= 45
            )
            if not serve_bounce_window_active:
                self._awaiting_serve_bounce = False
                return True, "Bounce in singles court", (255, 0, 0)
            target_side = getattr(self, '_point_target_service_side', None)
            service_box_ok, service_box_reason = self._point_in_target_service_box(point, frame, target_side)
            if service_box_ok is False:
                return False, service_box_reason, (0, 0, 255)
            return True, "Serve bounce in", (255, 0, 0)

        return True, "Bounce in singles court", (255, 0, 0)

    def _handle_ground_bounce_event(
        self,
        point,
        frame,
        accepted_in_reason=None,
        conservative_sideline_override=False,
    ):
        if accepted_in_reason is not None:
            in_bounds, reason, color = True, accepted_in_reason, (255, 0, 0)
        else:
            in_bounds, reason, color = self._classify_ground_bounce(
                point,
                frame,
                conservative_sideline_override=conservative_sideline_override,
            )
        serve_bounce_active = getattr(self, '_awaiting_serve_bounce', False)
        bounce_kind = "serve_bounce" if getattr(self, '_awaiting_serve_bounce', False) else "ground_bounce"
        self._add_impact_marker(point, kind=bounce_kind, color=color, label=reason)
        if getattr(self, '_awaiting_serve_bounce', False):
            self._awaiting_serve_bounce = False
        if in_bounds:
            review_reason = "serve" if serve_bounce_active else "ground"
            self._append_direction_change_review_event(point, "good", review_reason, frame=self.frame_count)
            if not serve_bounce_active and getattr(self, 'ground_bounce_count', 0) >= 2:
                self._pending_rally_end_reason = "Ball bounced twice on court"
                self._pending_rally_end_frame = self.frame_count
                print(f"Frame {self.frame_count}: [DOUBLE BOUNCE] point will end at {point}")
                self._print_double_bounce_debug()
            print(f"Frame {self.frame_count}: [BOUNCE IN] {reason} at {point}")
            return
        self._pending_rally_end_reason = reason
        self._pending_rally_end_frame = self.frame_count
        print(f"Frame {self.frame_count}: [BOUNCE OUT] {reason} at {point}")

    def _format_motion_debug(self, motion):
        if motion is None:
            return "none"
        return (
            f"dx={float(motion.get('dx', 0.0) or 0.0):.1f},"
            f"dy={float(motion.get('dy', 0.0) or 0.0):.1f},"
            f"dist={float(motion.get('distance', 0.0) or 0.0):.1f}"
        )

    def _racket_below_ball_motion_debug(self, point, frame):
        if point is None or frame is None:
            return {'available': False, 'reason': 'missing frame/point'}
        prev_gray = getattr(self, '_prev_frame_gray', None)
        if prev_gray is None:
            return {'available': False, 'reason': 'missing previous gray frame'}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != prev_gray.shape[:2]:
            return {'available': False, 'reason': 'previous gray shape mismatch'}

        frame_height, frame_width = gray.shape[:2]
        x, y = int(point[0]), int(point[1])
        half_width = max(90, int(frame_width * 0.030))
        top_pad = max(8, int(frame_height * 0.004))
        bottom_depth = max(150, int(frame_height * 0.075))
        x1 = max(0, x - half_width)
        x2 = min(frame_width, x + half_width + 1)
        y1 = max(0, y + top_pad)
        y2 = min(frame_height, y + bottom_depth + 1)
        if x2 <= x1 or y2 <= y1:
            return {'available': False, 'reason': 'empty below-ball region'}

        curr_patch = gray[y1:y2, x1:x2]
        prev_patch = prev_gray[y1:y2, x1:x2]
        diff = cv2.absdiff(curr_patch, prev_patch)
        motion_mask = diff >= 25
        edges = cv2.Canny(curr_patch, 50, 150)
        hsv_patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        sat_patch = hsv_patch[:, :, 1]
        motion_pixels = int(np.count_nonzero(motion_mask))
        motion_ratio = motion_pixels / float(diff.size)
        mean_motion = float(np.mean(diff)) if diff.size else 0.0
        max_motion = float(np.max(diff)) if diff.size else 0.0
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        high_sat_ratio = float(np.count_nonzero(sat_patch > 80)) / float(sat_patch.size)
        dark_ratio = float(np.count_nonzero(curr_patch < 90)) / float(curr_patch.size)
        object_hint = edge_ratio >= 0.08 and (high_sat_ratio >= 0.06 or dark_ratio >= 0.04)
        motion_hint = mean_motion >= 5.0 or max_motion >= 80.0 or motion_ratio >= 0.018

        return {
            'available': True,
            'roi': (x1, y1, x2, y2),
            'mean': mean_motion,
            'max': max_motion,
            'motion_ratio': motion_ratio,
            'motion_pixels': motion_pixels,
            'edge_ratio': edge_ratio,
            'high_sat_ratio': high_sat_ratio,
            'dark_ratio': dark_ratio,
            'motion_hint': motion_hint,
            'object_hint': object_hint,
            'hint': object_hint,
        }

    def _ground_bounce_has_racket_below_contact(self, ground_bounce_context, frame):
        if ground_bounce_context is None or frame is None:
            return False, {'available': False, 'reason': 'missing bounce context'}
        if getattr(self, '_awaiting_serve_bounce', False):
            return False, {'available': False, 'reason': 'awaiting serve bounce'}
        origin = ground_bounce_context.get('origin')
        if origin is None:
            return False, {'available': False, 'reason': 'missing bounce origin'}

        frame_height = frame.shape[0]
        origin_y = int(origin[1])
        if origin_y > max(650, int(frame_height * 0.32)):
            return False, {'available': False, 'reason': 'not upper-court contact zone'}

        incoming_dy = float(ground_bounce_context.get('incoming_dy', 0.0) or 0.0)
        incoming_dist = float(ground_bounce_context.get('incoming_dist', 0.0) or 0.0)
        if incoming_dy < max(6.0, frame_height * 0.0035) or incoming_dist < 6.0:
            return False, {'available': False, 'reason': 'weak incoming bounce motion'}

        racket_debug = self._racket_below_ball_motion_debug(origin, frame)
        return bool(racket_debug.get('object_hint')), racket_debug

    def _handle_racket_below_contact_instead_of_bounce(
        self,
        ground_bounce_context,
        frame,
        new_pos,
        area,
        racket_debug,
    ):
        origin = ground_bounce_context.get('origin') if ground_bounce_context else self.ball_center
        if origin is None:
            origin = self.ball_center
        if self.ground_bounce_count > 0:
            print(f"Frame {self.frame_count}: Resetting bounce count after racket-below contact")
        self.ground_bounce_count = 0
        self.last_ground_bounce_frame = -1000000
        self._rally_contact_grace_frames = max(getattr(self, '_rally_contact_grace_frames', 0), 3)
        self._rally_contact_ref_size = max(20.0, min(float(area or self.ball_size or 0.0), 120.0))
        self._rally_contact_origin = origin
        self._rally_contact_expected = new_pos
        self._rally_contact_progress = math.hypot(new_pos[0] - origin[0], new_pos[1] - origin[1])
        self._rally_contact_high_launch_until_frame = -1
        self._last_racket_contact_frame = self.frame_count
        self._last_racket_contact_point = origin
        self._add_impact_marker(origin, kind='racket_contact', ttl=7, label='racket contact')
        edge_ratio = racket_debug.get('edge_ratio', 0.0) * 100.0 if racket_debug else 0.0
        high_sat_ratio = racket_debug.get('high_sat_ratio', 0.0) * 100.0 if racket_debug else 0.0
        dark_ratio = racket_debug.get('dark_ratio', 0.0) * 100.0 if racket_debug else 0.0
        print(
            f"Frame {self.frame_count}: [RACKET-BELOW CONTACT] treating bounce candidate "
            f"at {origin} as racket contact, new_pos={new_pos} "
            f"below_edges={edge_ratio:.1f}% high_sat={high_sat_ratio:.1f}% dark={dark_ratio:.1f}%"
        )

    def _record_ground_bounce_debug(self, ground_bounce_context, source_label, prior_count, frame):
        context = ground_bounce_context or {}
        incoming_dx = float(context.get('incoming_dx', 0.0) or 0.0)
        incoming_dy = float(context.get('incoming_dy', 0.0) or 0.0)
        incoming_dist = float(context.get('incoming_dist', 0.0) or 0.0)
        racket_hint = self._racket_below_ball_motion_debug(context.get('origin'), frame)
        contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        frames_since_contact = self.frame_count - contact_frame
        recent_contact = frames_since_contact if frames_since_contact < 100000 else None
        entry = {
            'frame': self.frame_count,
            'source': source_label or 'predicted launch',
            'prior_count': int(prior_count),
            'count_after': int(getattr(self, 'ground_bounce_count', prior_count)),
            'origin': context.get('origin'),
            'expected': context.get('expected'),
            'incoming_dx': incoming_dx,
            'incoming_dy': incoming_dy,
            'incoming_dist': incoming_dist,
            'last_motion': self._format_motion_debug(self.last_motion),
            'prev_motion': self._format_motion_debug(self.prev_motion),
            'racket_below': racket_hint,
            'last_racket_contact_frame': contact_frame if recent_contact is not None else None,
            'frames_since_racket_contact': recent_contact,
            'last_racket_contact_point': getattr(self, '_last_racket_contact_point', None),
        }
        history = getattr(self, '_ground_bounce_debug_history', None)
        if history is None:
            self._ground_bounce_debug_history = []
            history = self._ground_bounce_debug_history
        history.append(entry)
        del history[:-6]
        return entry

    def _print_ground_bounce_debug(self, entry):
        if entry is None:
            return
        racket = entry.get('racket_below') or {}
        if racket.get('available'):
            racket_text = (
                f"racket_below_hint={'yes' if racket.get('hint') else 'no'} "
                f"below_motion(mean={racket.get('mean', 0.0):.1f},"
                f"max={racket.get('max', 0.0):.1f},"
                f"ratio={racket.get('motion_ratio', 0.0) * 100:.1f}%,"
                f"edges={racket.get('edge_ratio', 0.0) * 100:.1f}%,"
                f"high_sat={racket.get('high_sat_ratio', 0.0) * 100:.1f}%,"
                f"dark={racket.get('dark_ratio', 0.0) * 100:.1f}%,"
                f"roi={racket.get('roi')})"
            )
        else:
            racket_text = f"racket_below_hint=unknown ({racket.get('reason', 'unavailable')})"
        contact_frames = entry.get('frames_since_racket_contact')
        contact_text = (
            f"{contact_frames}f ago at {entry.get('last_racket_contact_point')}"
            if contact_frames is not None else "none"
        )
        print(
            f"Frame {entry['frame']}: [BOUNCE DEBUG] source={entry['source']} "
            f"count={entry['prior_count']}->{entry['count_after']} "
            f"origin={entry.get('origin')} expected={entry.get('expected')} "
            f"incoming=({entry['incoming_dx']:.1f},{entry['incoming_dy']:.1f},"
            f"{entry['incoming_dist']:.1f}) last_motion=({entry['last_motion']}) "
            f"prev_motion=({entry['prev_motion']}) last_racket_contact={contact_text} "
            f"{racket_text}"
        )

    def _print_double_bounce_debug(self):
        history = getattr(self, '_ground_bounce_debug_history', [])[-2:]
        if not history:
            print(f"Frame {self.frame_count}: [DOUBLE BOUNCE DEBUG] no bounce history recorded")
            return
        for idx, entry in enumerate(history, start=1):
            print(
                f"Frame {self.frame_count}: [DOUBLE BOUNCE DEBUG] bounce{idx} "
                f"registered_at={entry.get('frame')} source={entry.get('source')} "
                f"origin={entry.get('origin')} count={entry.get('prior_count')}->"
                f"{entry.get('count_after')}"
            )

    def _register_ground_bounce_from_context(self, ground_bounce_context, frame, source_label=None):
        if ground_bounce_context is None or frame is None:
            return False
        if (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 3:
            return False
        origin_y = ground_bounce_context['origin'][1]
        late_contact_until = int(getattr(self, '_late_contact_prior_bounce_until_frame', -1000000))
        late_contact_saved_frame = late_contact_until - 90
        post_preserve_contact = (
            late_contact_until > -1000000 and
            int(getattr(self, '_last_racket_contact_frame', -1000000)) >= late_contact_saved_frame
        )
        carried_prior_bounce = (
            self.ground_bounce_count == 0 and
            self._late_contact_prior_bounce_active() and
            origin_y <= max(360, int(frame.shape[0] * 0.18)) and
            not post_preserve_contact
        )
        if carried_prior_bounce:
            self.ground_bounce_count = int(getattr(self, '_late_contact_prior_bounce_count', 0))
            print(
                f"Frame {self.frame_count}: Carrying prior bounce count "
                f"{self.ground_bounce_count} through late contact before upper bounce"
            )
        is_racket_rebound = source_label == "racket rebound"
        if (not is_racket_rebound and
                self.ground_bounce_count > 0 and
                (self.frame_count - getattr(self, '_recent_racket_rebound_bounce_frame', -1000000)) <= 90):
            print(f"Frame {self.frame_count}: Resetting bounce count after racket rebound before new bounce")
            self.ground_bounce_count = 0
        prior_count = self.ground_bounce_count
        self.ground_bounce_count += 1
        self.last_ground_bounce_frame = self.frame_count
        if is_racket_rebound:
            self._recent_racket_rebound_bounce_frame = self.frame_count
        label_suffix = f" ({source_label})" if source_label else ""
        print(f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} detected{label_suffix}")
        bounce_debug = self._record_ground_bounce_debug(
            ground_bounce_context, source_label, prior_count, frame
        )
        bounce_debug['count_after'] = self.ground_bounce_count
        self._print_ground_bounce_debug(bounce_debug)
        self._handle_ground_bounce_event(
            ground_bounce_context['origin'],
            frame,
            conservative_sideline_override=(
                source_label == "predicted launch" and
                getattr(self, 'direction_change_streak', 0) >= 2
            ),
        )
        return True

    def _near_camera_large_continuation_lock(self, current_pos, current_area, prev_pos, frame_shape):
        """Return motion context when a visible large lower-court ball is still on track."""
        if (
                current_pos is None or
                prev_pos is None or
                frame_shape is None or
                self.last_motion is None or
                self.ball_size is None):
            return None

        frame_height = frame_shape[0]
        prev_size = float(self.ball_size or 0.0)
        if (
                prev_size < 120.0 or
                prev_pos[1] < max(850, int(frame_height * 0.40))):
            return None

        last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        last_dist = float(self.last_motion.get('distance', 0.0) or 0.0)
        if last_dy < max(28.0, frame_height * 0.012) or last_dist < 45.0:
            return None

        area = float(current_area or 0.0)
        if area < max(90.0, prev_size * 0.72):
            return None

        dx = float(current_pos[0] - prev_pos[0])
        dy = float(current_pos[1] - prev_pos[1])
        distance = math.hypot(dx, dy)
        if (
                dy < max(12.0, last_dy * 0.15) or
                distance > max(185.0, last_dist * 1.35)):
            return None

        dot = last_dx * dx + last_dy * dy
        if dot <= 0.0:
            return None
        if abs(last_dx) >= 8.0 and (dx * last_dx) < -8.0:
            return None

        return {
            'last_dist': last_dist,
            'dx': dx,
            'dy': dy,
            'distance': distance,
        }

    def _should_start_back_return_wait(self, frame_shape):
        """Return True for large lower-right exits that can re-enter from the right side later."""
        if self.ball_center is None or self.last_motion is None:
            return False
        frame_height, frame_width = frame_shape[:2]
        x_prev, y_prev = self.ball_center
        large_bottom_exit = (
            self.ball_size is not None and
            self.ball_size >= 700.0 and
            y_prev >= frame_height - max(90, int(frame_height * 0.04)) and
            x_prev >= int(frame_width * 0.58)
        )
        lower_right_exit = x_prev >= int(frame_width * 0.84)
        if not (lower_right_exit or large_bottom_exit):
            return False
        if y_prev < frame_height - 160:
            return False
        if self.last_motion.get('dx', 0.0) < 8.0:
            return False
        if self.last_motion.get('dy', 0.0) < 35.0:
            return False
        if self.last_motion.get('distance', 0.0) < 70.0:
            return False
        if self.ball_size is None or self.ball_size < 250.0:
            return False
        return True

    def _lower_right_edge_exit_out_reason(self, frame_shape):
        """Return an out reason when a lower-right edge exit is no longer recoverable."""
        if self.ball_center is None or self.last_motion is None:
            return None
        if self._back_return_wait_active() or self._should_start_back_return_wait(frame_shape):
            return None
        if self.point_start_frame_internal is not None and (self.frame_count - self.point_start_frame_internal) <= 20:
            return None

        frame_height, frame_width = frame_shape[:2]
        x_prev, y_prev = self.ball_center
        dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        distance = float(self.last_motion.get('distance', 0.0) or 0.0)
        near_right_edge = x_prev >= frame_width - 20
        near_bottom_edge = y_prev >= frame_height - 20
        moving_out = dx >= 8.0 and dy >= 35.0 and distance >= 45.0
        partial_edge_blob = self.ball_size is None or self.ball_size <= 180.0
        if near_right_edge and near_bottom_edge and moving_out and partial_edge_blob:
            return "Ball bounce outside singles court (right sideline)"
        return None

    def _activate_back_return_wait(self):
        """Arm a delayed lower-right return search after the large ball leaves frame."""
        if self.ball_center is None:
            return
        self._back_return_wait_frames = max(
            self._back_return_wait_frames,
            int(getattr(self, '_back_return_timeout_frames', 8)),
        )
        self._back_return_anchor = tuple(self.ball_center)
        self._back_return_origin_frame = self.frame_count
        self._back_return_timed_out = False

    def _back_return_wait_active(self):
        anchor = getattr(self, '_back_return_anchor', None)
        return (
            getattr(self, '_back_return_wait_frames', 0) > 0 and
            anchor is not None and
            self.ball_center is not None
        )

    def _recent_offscreen_return_hold_active(self, window_frames=8):
        last_frame = int(getattr(self, '_recent_offscreen_return_frame', -1000000))
        return (self.frame_count - last_frame) <= max(1, int(window_frames))

    def _recent_return_bounce_recover_active(self, window_frames=2):
        last_frame = int(getattr(self, '_recent_return_bounce_recover_frame', -1000000))
        return (self.frame_count - last_frame) <= max(1, int(window_frames))

    def _late_contact_prior_bounce_active(self):
        return (
            int(getattr(self, '_late_contact_prior_bounce_count', 0)) > 0 and
            self.frame_count <= int(getattr(self, '_late_contact_prior_bounce_until_frame', -1000000))
        )

    def _update_recovered_motion(self, prev_pos, new_pos):
        if prev_pos is None or new_pos is None:
            return
        new_dx = int(new_pos[0] - prev_pos[0])
        new_dy = int(new_pos[1] - prev_pos[1])
        new_velocity = math.hypot(new_dx, new_dy)
        new_direction = math.degrees(math.atan2(new_dy, new_dx)) if new_velocity > 0 else 0.0
        self.last_delta = (new_dx, new_dy)
        self.last_motion = {
            'distance': new_velocity,
            'dx': new_dx,
            'dy': new_dy,
            'direction_deg': new_direction,
        }
        vel_hist = list(getattr(self, 'ball_velocity_history', []))
        vel_hist.append(new_velocity)
        self.ball_velocity_history = vel_hist[-5:]
        self.last_direction = new_direction

    def _back_return_reentry_ok(self, pos, area, motion_mean, motion_max, frame_shape):
        """Validate a delayed re-entry from the right/back side after a lower-right exit."""
        anchor = getattr(self, '_back_return_anchor', None)
        if anchor is None:
            return True, None

        cx, cy = pos
        frame_height, frame_width = frame_shape[:2]
        elapsed = max(0, self.frame_count - getattr(self, '_back_return_origin_frame', self.frame_count))
        x_floor = max(int(frame_width * 0.60), anchor[0] - 1000)
        x_ceil = min(frame_width - 180, anchor[0] + 120)
        min_reentry_y = max(int(frame_height * 0.12), anchor[1] - 1450)
        max_reentry_y = min(int(frame_height * 0.92), anchor[1] - 80)
        strong_motion = motion_max >= 35.0 or motion_mean >= 6.0 or area >= 8.0

        early_bottom_return = (
            elapsed >= 4 and
            abs(cx - anchor[0]) <= 180 and
            cy >= anchor[1] - 340 and
            cy <= frame_height - 2 and
            (motion_max >= 25.0 or motion_mean >= 6.0 or area >= 15.0)
        )
        if early_bottom_return:
            return True, None

        if elapsed < 4:
            return False, f"back-return blind wait elapsed={elapsed}f"
        if cx < x_floor:
            return False, f"back-return x {cx} < min_x {x_floor}"
        if cx > x_ceil:
            return False, f"back-return x {cx} > max_x {x_ceil}"
        if cy < min_reentry_y:
            return False, f"back-return y {cy} < min_y {min_reentry_y}"
        if cy > max_reentry_y:
            return False, f"back-return y {cy} > max_y {max_reentry_y}"
        if not strong_motion:
            return False, f"back-return weak motion mean={motion_mean:.1f} max={motion_max:.1f} area={area:.1f}"
        return True, None

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

    def _large_lower_launch_candidate_ok(self, pos, area, frame_shape):
        """Recognize the close-ball launch that follows a one-frame lower-court hold."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False
        if self.stuck_frame_count < 1:
            return False

        frame_height, frame_width = frame_shape[:2]
        prev_size = float(self.ball_size)
        origin_x, origin_y = self.ball_center
        if not (145.0 <= prev_size <= 260.0):
            return False
        if not (int(frame_height * 0.48) <= origin_y <= int(frame_height * 0.56)):
            return False

        incoming_dy = float(self.last_motion.get('dy', 0.0))
        incoming_dist = float(self.last_motion.get('distance', 0.0))
        if incoming_dy < max(35.0, frame_height * 0.016) or incoming_dist < 35.0:
            return False

        cx, cy = pos
        dx = cx - origin_x
        dy = cy - origin_y
        launch_dist = math.hypot(dx, dy)
        upward_progress = origin_y - cy
        min_area = max(95.0, min(155.0, prev_size * 0.50))
        min_upward = max(120.0, incoming_dy * 2.2)
        min_launch_dist = max(115.0, incoming_dist * 2.0)
        max_launch_dist = max(360.0, incoming_dist * 6.0)
        lateral_cap = max(180.0, frame_width * 0.05)

        return (
            area >= min_area and
            upward_progress >= min_upward and
            min_launch_dist <= launch_dist <= max_launch_dist and
            abs(dx) <= lateral_cap
        )

    def _near_side_large_racket_hit_prep_active(self, frame_shape):
        """Open the search window before a near-side racket hit from a large incoming ball."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False

        frame_height = frame_shape[0]
        origin_y = self.ball_center[1]
        prev_size = float(self.ball_size)
        prev_dy = float(self.last_motion.get('dy', 0.0))
        prev_speed = float(self.last_motion.get('distance', 0.0))

        return (
            prev_size >= 240.0 and
            origin_y >= int(frame_height * 0.535) and
            prev_dy >= max(26.0, frame_height * 0.012) and
            prev_speed >= 35.0 and
            self.stuck_frame_count <= 1
        )

    def _near_camera_large_racket_turn_candidate_ok(
        self,
        pos,
        area,
        frame_shape,
        motion_mean=None,
        motion_max=None,
    ):
        """Prefer the large close-camera ball blob at a lower-court racket turn."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        prev_size = float(self.ball_size)
        prev_dy = float(self.last_motion.get('dy', 0.0))
        prev_speed = float(self.last_motion.get('distance', 0.0))
        cx, cy = pos
        dx = cx - origin_x
        dy = cy - origin_y
        turn_dist = math.hypot(dx, dy)
        upward_progress = origin_y - cy
        large_edge_launch = prev_size >= 240.0

        if prev_size < 180.0:
            return False
        min_origin_y = int(frame_height * (0.535 if large_edge_launch else 0.56))
        if origin_y < min_origin_y:
            return False
        if prev_dy < max(28.0, frame_height * 0.013) or prev_speed < 35.0:
            return False
        if area < max(180.0, prev_size * 0.72):
            return False
        if upward_progress < max(85.0, min(prev_dy * 1.8, 135.0)):
            return False
        if dy > -max(55.0, frame_height * 0.025):
            return False
        turn_cap = max(360.0, prev_speed * 5.5) if large_edge_launch else max(260.0, prev_speed * 4.0)
        if turn_dist > turn_cap:
            return False
        lateral_cap = max(240.0, frame_width * 0.075) if large_edge_launch else max(170.0, frame_width * 0.045)
        if abs(dx) > lateral_cap:
            return False
        if motion_mean is not None and motion_max is not None:
            if motion_mean < 25.0 and motion_max < 100.0:
                return False
        return True

    def _lower_contact_launch_candidate_ok(self, pos, area, lower_contact_launch_context, frame_shape):
        """Check whether a contour matches the predicted launch after a lower racket contact."""
        if lower_contact_launch_context is None or self.ball_center is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        cx, cy = pos
        origin_x, origin_y = lower_contact_launch_context['origin']
        dx = cx - origin_x
        dy = cy - origin_y
        launch_upward = dy <= -max(45, int(frame_height * 0.02))
        launch_dist = math.hypot(dx, dy)
        expected_x, expected_y = lower_contact_launch_context['expected']
        expected_distance = math.hypot(cx - expected_x, cy - expected_y)
        candidate_not_tiny = area >= max(
            20,
            min(int(lower_contact_launch_context['ref_size'] * 0.50), 80)
        )

        return (
            launch_upward and
            candidate_not_tiny and
            launch_dist >= lower_contact_launch_context['min_launch_dist'] and
            launch_dist <= lower_contact_launch_context['max_launch_dist'] and
            expected_distance <= max(140, int(frame_width * 0.04))
        )

    def _lower_racket_contact_turn_candidate_ok(
        self,
        pos,
        area,
        frame_shape,
        dx,
        dy,
        angle_jump,
        motion_mean=None,
        motion_max=None,
    ):
        """Recognize the close lower-court turn at the racket before launch opens up."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False
        if self.stuck_frame_count < 1:
            return False

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        prev_size = float(self.ball_size)
        prev_dx = float(self.last_motion.get('dx', 0.0))
        prev_dy = float(self.last_motion.get('dy', 0.0))
        prev_speed = float(self.last_motion.get('distance', 0.0))
        cx, cy = pos
        turn_dist = math.hypot(dx, dy)
        upward_progress = origin_y - cy

        if origin_y < int(frame_height * 0.56) or cy < int(frame_height * 0.50):
            return False
        if prev_dy < max(28.0, frame_height * 0.013) or prev_speed < 35.0:
            return False
        if prev_size > 300.0:
            return False
        if prev_size < 170.0 and area < 85.0:
            return False
        if area < max(35.0, min(prev_size * 0.25, 90.0)):
            return False
        if upward_progress < max(32.0, frame_height * 0.015):
            return False
        if dy > -max(30.0, frame_height * 0.014):
            return False
        if angle_jump < 95.0:
            return False
        if turn_dist > max(135.0, prev_speed * 1.45):
            return False
        if abs(dx) > max(105.0, frame_width * 0.035):
            return False
        if motion_mean is not None and motion_max is not None:
            if motion_mean < 30.0 and motion_max < 100.0:
                return False
        if prev_dx * dx < -80.0 and abs(dx) > 55.0:
            return False
        return True

    def _immediate_lower_racket_contact_turn_candidate_ok(
        self,
        pos,
        area,
        frame_shape,
        dx,
        dy,
        angle_jump,
        motion_mean=None,
        motion_max=None,
    ):
        """Catch the first post-impact frame when a large near-court ball reverses at the racket."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        prev_size = float(self.ball_size)
        prev_dx = float(self.last_motion.get('dx', 0.0))
        prev_dy = float(self.last_motion.get('dy', 0.0))
        prev_speed = float(self.last_motion.get('distance', 0.0))
        cx, cy = pos
        turn_dist = math.hypot(dx, dy)
        upward_progress = origin_y - cy

        if self.stuck_frame_count > 1:
            return False
        if not (240.0 <= prev_size <= 330.0):
            return False
        if origin_y < int(frame_height * 0.535) or cy < int(frame_height * 0.495):
            return False
        if prev_dy < max(26.0, frame_height * 0.012) or prev_speed < 35.0:
            return False
        if angle_jump < 135.0:
            return False
        if prev_dx * dx >= -200.0:
            return False
        if area < max(35.0, min(prev_size * 0.14, 60.0)):
            return False
        if upward_progress < max(30.0, frame_height * 0.013):
            return False
        if dy > -max(28.0, frame_height * 0.013):
            return False
        if turn_dist > max(120.0, prev_speed * 1.60):
            return False
        if abs(dx) > max(115.0, frame_width * 0.040):
            return False
        if motion_mean is not None and motion_max is not None:
            if motion_mean < 25.0 and motion_max < 100.0:
                return False
        return True

    def _upper_racket_contact_turn_candidate_ok(
        self,
        pos,
        area,
        frame_shape,
        dx,
        dy,
        angle_jump,
        speed_ratio,
        motion_mean=None,
        motion_max=None,
    ):
        """Recognize the far-player hit after the ball stalls near its upper apex."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False
        if getattr(self, '_awaiting_serve_bounce', False):
            return False
        if self.ground_bounce_count <= 0:
            return False
        if getattr(self, 'direction_change_streak', 0) < 2:
            return False

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        prev_size = float(self.ball_size)
        prev_dy = float(self.last_motion.get('dy', 0.0))
        prev_speed = float(self.last_motion.get('distance', 0.0))
        cx, cy = pos
        turn_dist = math.hypot(dx, dy)

        if origin_y < max(150, int(frame_height * 0.07)):
            return False
        if origin_y > max(360, int(frame_height * 0.18)):
            return False
        if cy > max(430, int(frame_height * 0.22)):
            return False
        if prev_size > 45.0 or area > 60.0:
            return False
        if prev_dy > 2.0 or prev_speed > 10.0:
            return False
        if dy < max(6.0, frame_height * 0.0035):
            return False
        if angle_jump < 70.0:
            return False
        if speed_ratio < 1.55:
            return False
        if turn_dist < 8.0 or turn_dist > max(60.0, prev_speed * 5.0):
            return False
        if abs(dx) > max(90.0, frame_width * 0.03):
            return False
        if motion_mean is not None and motion_max is not None:
            if motion_mean < 12.0 and motion_max < 70.0:
                return False
        return True

    def _upper_slow_arc_active(self):
        return self.frame_count <= getattr(self, '_upper_slow_arc_until_frame', -1)

    def _upper_fence_fall_end_candidate(self, ball_position, frame_shape):
        if ball_position is None or self.last_motion is None:
            self._upper_fence_fall_frames = 0
            return False
        if not self._upper_slow_arc_active():
            self._upper_fence_fall_frames = 0
            return False

        frame_height, frame_width = frame_shape[:2]
        x, y = ball_position
        upper_fence_y_min = max(150, int(frame_height * 0.070))
        upper_fence_y_max = max(205, int(frame_height * 0.095))
        lane_min_x = int(frame_width * 0.35)
        lane_max_x = int(frame_width * 0.70)

        ball_size = float(self.ball_size or 0.0)
        speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        in_upper_fence_band = (
            lane_min_x <= x <= lane_max_x and
            upper_fence_y_min <= y <= upper_fence_y_max
        )
        tiny_visible_ball = 3.0 <= ball_size <= 16.0
        settled_or_falling = (speed <= 6.5 and dy >= -3.0) or getattr(self, 'stuck_frame_count', 0) >= 1

        if not (in_upper_fence_band and tiny_visible_ball and settled_or_falling):
            self._upper_fence_fall_frames = 0
            return False

        self._upper_fence_fall_frames += 1
        if self._upper_fence_fall_frames < 2:
            return False

        self._add_impact_marker(
            ball_position,
            kind="fence_contact",
            ttl=12,
            label="upper fence / ball fell down",
        )
        print(
            f"Frame {self.frame_count}: [UPPER FENCE FALL] "
            f"pos={ball_position} size={ball_size:.1f}px speed={speed:.1f}px dy={dy:.1f}"
        )
        return True

    def _top_far_baseline_fall_out_candidate(self, ball_position, frame):
        """End upper returns that have clearly fallen out beyond the far baseline."""
        if ball_position is None or self.last_motion is None:
            return False, None
        if getattr(self, '_awaiting_serve_bounce', False):
            return False, None

        frame_height, _ = frame.shape[:2]
        x, y = ball_position
        if not (max(175, int(frame_height * 0.081)) <= y <= max(270, int(frame_height * 0.13))):
            return False, None

        ball_size = float(self.ball_size or 0.0)
        if self.ball_size is not None and ball_size > 22.0:
            return False, None

        dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        if not (dy >= max(10.0, frame_height * 0.006) and speed <= max(130.0, frame_height * 0.060)):
            return False, None

        recent_upper_speed = max(
            [
                float(entry.get('distance', 0.0) or 0.0)
                for entry in getattr(self, 'motion_history', [])
                if 0 <= self.frame_count - int(entry.get('frame', -1000000)) <= 12
            ] or [0.0]
        )
        if recent_upper_speed < 145.0:
            return False, None

        outside_far_baseline, far_y = self._point_outside_top_singles_baseline(ball_position, frame)
        if not outside_far_baseline:
            return False, None

        if self.frame_count <= int(getattr(self, '_top_far_out_defer_until_frame', -1000000)):
            return False, None

        recent_top_origin_frame = int(getattr(self, '_top_return_origin_frame', -1000000))
        recent_top_wait_context = 0 <= (self.frame_count - recent_top_origin_frame) <= 110
        upper_arc_recently_active = (
            self.frame_count <= int(getattr(self, '_upper_slow_arc_until_frame', -1000000)) + 70
        )
        top_or_return_context = (
            self._top_return_wait_active() or
            getattr(self, '_top_return_reentry_grace_frames', 0) > 0 or
            self._recent_offscreen_return_hold_active(window_frames=48) or
            recent_top_wait_context or
            upper_arc_recently_active
        )
        if not top_or_return_context:
            return False, None

        print(
            f"Frame {self.frame_count}: [TOP-FAR-OUT] pos={ball_position} "
            f"far_baseline_y={far_y:.1f} dy={dy:.1f} speed={speed:.1f} size={ball_size:.1f}"
        )
        return True, "Ball bounced out of court (far baseline)"

    def _upper_slow_arc_candidate_ok(self, pos, area, velocity, predicted_point, frame_shape, frame_gray=None):
        """Allow tiny upper-court flight changes near the visible apex without freezing."""
        if not self._upper_slow_arc_active():
            return False
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        cx, cy = pos
        if not (max(180, int(frame_height * 0.10)) <= origin_y <= max(360, int(frame_height * 0.18))):
            return False
        if not (max(180, int(frame_height * 0.10)) <= cy <= max(360, int(frame_height * 0.18))):
            return False
        if self.ball_size > 90 or area < max(5.0, min(float(self.ball_size) * 0.25, 12.0)):
            return False

        last_dist = float(self.last_motion.get('distance', 0.0) or 0.0)
        if last_dist > 18.0 or velocity > 22.0:
            return False
        if abs(cx - origin_x) > max(18.0, frame_width * 0.006):
            return False

        if predicted_point is not None:
            predicted_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
            if predicted_distance > max(34.0, last_dist * 4.0 + 12.0):
                return False

        motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
        if motion_metrics is None:
            return True
        return motion_metrics['max'] >= 35.0 or motion_metrics['mean'] >= 8.0

    def _get_ground_bounce_context(self, frame_shape, allow_near_net=False):
        """Predict a short upward continuation after a court bounce using ball-only motion."""
        if self.ball_center is None or self.last_motion is None or self.ball_size is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        origin_x, origin_y = self.ball_center
        incoming_dx = float(self.last_motion.get('dx', 0.0))
        incoming_dy = float(self.last_motion.get('dy', 0.0))
        incoming_dist = float(self.last_motion.get('distance', 0.0))

        large_lower_near_court = (
            self.ball_size > 180 and
            origin_y >= max(1080, int(frame_height * 0.55)) and
            incoming_dy >= max(18.0, frame_height * 0.012) and
            incoming_dist >= 24.0
        )
        if self.ball_size > 180 and not large_lower_near_court:
            return None
        min_incoming_dy = max(6.0, frame_height * 0.0035)
        upper_soft_bounce = False
        if incoming_dy < min_incoming_dy:
            prev_dy = float(self.prev_motion.get('dy', 0.0)) if self.prev_motion is not None else 0.0
            prev_dx = float(self.prev_motion.get('dx', 0.0)) if self.prev_motion is not None else 0.0
            prev_dist = float(self.prev_motion.get('distance', 0.0)) if self.prev_motion is not None else 0.0
            serve_soft_bounce = (
                getattr(self, '_awaiting_serve_bounce', False) and
                self.point_start_frame_internal is not None and
                (self.frame_count - self.point_start_frame_internal) <= 45 and
                6.0 <= self.ball_size <= 90 and
                origin_y >= max(300, int(frame_height * 0.14)) and
                incoming_dy >= max(3.0, frame_height * 0.0018) and
                prev_dy >= max(3.0, frame_height * 0.0018) and
                incoming_dist >= 6.0 and
                prev_dist >= 6.0
            )
            upper_soft_bounce = (
                self._upper_slow_arc_active() and
                (
                    getattr(self, 'ground_bounce_count', 0) > 0 or
                    self._late_contact_prior_bounce_active()
                ) and
                self.ball_size <= 90 and
                origin_y <= max(360, int(frame_height * 0.18)) and
                incoming_dy >= 1.0 and
                prev_dy >= 2.0 and
                (incoming_dy + prev_dy) >= 5.0 and
                prev_dist >= 2.5
            )
            if not (serve_soft_bounce or upper_soft_bounce):
                return None
            incoming_dx = (incoming_dx + prev_dx) * 0.5
            incoming_dy = max(incoming_dy, prev_dy * 0.75)
            incoming_dist = max(incoming_dist, prev_dist * 0.75)
        min_context_dist = 2.5 if upper_soft_bounce else 6.0
        if incoming_dist < min_context_dist:
            return None
        if (not upper_soft_bounce and self.prev_motion is not None and
                self.prev_motion.get('dy', 0.0) < max(3.0, frame_height * 0.002)):
            return None
        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            if allow_near_net:
                net_geometry = self._net_contact_geometry((origin_x, origin_y))
                if net_geometry is not None and (net_geometry['inside'] or net_geometry['near_top_tape']):
                    return None
            elif (self.net_area_y_min - 60) <= origin_y <= (self.net_area_y_max + 80):
                return None

        expected_dx = incoming_dx * 0.9
        expected_up = max(8.0, min(max(14.0, abs(incoming_dy) * 1.15), 70.0))
        expected_x = int(max(0, min(frame_width - 1, origin_x + expected_dx)))
        expected_y = int(max(0, min(frame_height - 1, origin_y - expected_up)))

        if upper_soft_bounce:
            min_launch_dist = max(2.5, incoming_dist * 0.25)
            min_upward = max(3.0, incoming_dy * 0.35)
            expected_cap = max(28.0, min(85.0, incoming_dist * 3.0))
        else:
            min_launch_dist = max(5.0, incoming_dist * 0.45)
            min_upward = max(5.0, incoming_dy * 0.50)
            expected_cap = max(28.0, min(85.0, incoming_dist * 2.2))

        max_launch_dist = max(55.0, incoming_dist * 3.2)
        lower_near_court_rebound = (
            not upper_soft_bounce and
            self.ball_size >= 80.0 and
            origin_y >= max(1080, int(frame_height * 0.55)) and
            incoming_dy >= max(18.0, frame_height * 0.012) and
            incoming_dist >= 24.0
        )
        if lower_near_court_rebound:
            max_launch_dist = max(max_launch_dist, min(220.0, incoming_dist * 4.25))
            expected_cap = max(expected_cap, min(155.0, incoming_dist * 3.6))
        if large_lower_near_court:
            max_launch_dist = max(max_launch_dist, min(360.0, incoming_dist * 6.5))
            expected_cap = max(expected_cap, min(260.0, incoming_dist * 5.5))

        return {
            'origin': (origin_x, origin_y),
            'expected': (expected_x, expected_y),
            'incoming_dx': incoming_dx,
            'incoming_dy': incoming_dy,
            'incoming_dist': incoming_dist,
            'min_launch_dist': min_launch_dist,
            'max_launch_dist': max_launch_dist,
            'min_upward': min_upward,
            'expected_cap': expected_cap,
            'ref_size': max(8.0, min(max(self.ball_size * 1.10, self.ball_size + 6.0), 90.0)),
        }

    def _maybe_handle_reacquire_ground_bounce(self, reacq_pos, frame):
        """Infer a missed ground bounce when reacquire finds the first post-bounce blob."""
        if reacq_pos is None or frame is None or self.ball_center is None:
            return False
        if self._top_return_wait_active() or self._back_return_wait_active():
            return False

        ground_bounce_context = self._get_ground_bounce_context(frame.shape, allow_near_net=True)
        if ground_bounce_context is None:
            return False

        cx, cy = int(reacq_pos[0]), int(reacq_pos[1])
        origin_x, origin_y = ground_bounce_context['origin']
        bounce_dist = math.hypot(cx - origin_x, cy - origin_y)
        upward_progress = origin_y - cy
        expected_distance = math.hypot(
            cx - ground_bounce_context['expected'][0],
            cy - ground_bounce_context['expected'][1],
        )
        lateral_dx = cx - origin_x
        incoming_dx = ground_bounce_context['incoming_dx']
        same_direction_x = abs(incoming_dx) <= 2.0 or (lateral_dx * incoming_dx) >= -8.0

        min_launch_dist = max(8.0, ground_bounce_context['min_launch_dist'] * 0.75)
        max_launch_dist = max(ground_bounce_context['max_launch_dist'], ground_bounce_context['max_launch_dist'] + 18.0)
        min_upward = max(6.0, ground_bounce_context['min_upward'] * 0.75)
        expected_cap = max(ground_bounce_context['expected_cap'], ground_bounce_context['expected_cap'] + 18.0)
        candidate_not_tiny = (
            self.ball_size is None or
            self.ball_size <= 160.0
        )

        inferred_bounce = (
            same_direction_x and
            candidate_not_tiny and
            upward_progress >= min_upward and
            bounce_dist >= min_launch_dist and
            bounce_dist <= max_launch_dist and
            expected_distance <= expected_cap
        )
        if not inferred_bounce:
            return False

        if self._register_ground_bounce_from_context(
                ground_bounce_context, frame, source_label=f"reacquire from {reacq_pos}"):
            print(
                f"Frame {self.frame_count}: [REACQ BOUNCE] inferred ground bounce at "
                f"{ground_bounce_context['origin']} from reacquire {reacq_pos}"
            )

        self._ground_bounce_grace_frames = max(getattr(self, '_ground_bounce_grace_frames', 0), 3)
        self._ground_bounce_ref_size = max(
            8.0,
            min(max(float(self.ball_size or 0.0), ground_bounce_context['ref_size']), 90.0),
        )
        self._ground_bounce_origin = ground_bounce_context['origin']
        self._ground_bounce_expected = ground_bounce_context['expected']
        self._ground_bounce_progress = math.hypot(
            cx - self._ground_bounce_origin[0],
            cy - self._ground_bounce_origin[1],
        )
        return True

    def _recover_contact_phase_ball(self, frame, reference_pos, frame_gray, predicted_point=None,
                                    max_prev_speed=15.0, upper_stuck_mode=False):
        if reference_pos is None or frame_gray is None or self.last_motion is None:
            return None

        ref_x, ref_y = reference_pos
        upper_limit = max(280, int(frame.shape[0] * 0.14))
        if upper_stuck_mode:
            upper_limit = max(340, int(frame.shape[0] * 0.17))
        if ref_y > upper_limit:
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
        lateral_reject_counts = {
            'weak_motion': 0,
            'early_weak': 0,
            'far_predicted': 0,
        }
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

                predicted_distance = None
                if predicted_point is not None:
                    predicted_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])

                lateral_weak_motion = (
                    motion_mean is None or
                    (motion_mean < 6.0 and motion_max < 60.0)
                )
                if lateral_weak_motion:
                    lateral_reject_counts['weak_motion'] += 1
                    continue

                if (getattr(self, 'stuck_frame_count', 0) < 2 and
                        motion_mean < 10.0 and motion_max < 80.0):
                    lateral_reject_counts['early_weak'] += 1
                    continue

                if predicted_distance is not None and getattr(self, 'stuck_frame_count', 0) < 2:
                    lateral_predicted_cap = max(180.0, int(frame.shape[1] * 0.055))
                    if predicted_distance > lateral_predicted_cap:
                        lateral_reject_counts['far_predicted'] += 1
                        continue

                score = abs(dy) * 4.0
                score += abs(abs(dx) - target_shift) * 0.8
                score -= min(area, 50) * 6.0
                score -= min(motion_max, 20.0) * 0.5
                if predicted_distance is not None:
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
        elif any(lateral_reject_counts.values()):
            print(
                "  DEBUG: [CONTACT RECOVER] rejected lateral candidates "
                f"weak_motion={lateral_reject_counts['weak_motion']} "
                f"early_weak={lateral_reject_counts['early_weak']} "
                f"far_predicted={lateral_reject_counts['far_predicted']}"
            )
        return picked

    def retrack_with_alt_hsv(self, search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                             frame_gray=None, filter_key="alt1"):
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
                frame_shape = frame_gray.shape if frame_gray is not None else None
                min_area = self._min_area_for_previous_ball_size(self.ball_size, prev_pos, frame_shape)
                if area < min_area:
                    print(f"  DEBUG: retrack_using_alt skipping too-small contour area={area:.1f} "
                          f"prev_ball_size={self.ball_size:.1f} min_area={min_area}")
                    continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1

            if not allow_inactive:
                ignored_entry = self._find_ignored_tracking_position((cx, cy), filter_key=filter_key)
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
                    print(f"  DEBUG: retrack_using_alt skipping frame0 hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                    continue
                static_hotspot = ((area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                if static_hotspot:
                    print(f"  DEBUG: retrack_using_alt skipping static hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                    continue

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
        selected_motion = self._candidate_motion_metrics(frame_gray, cx, cy)
        motion_mean = selected_motion['mean'] if selected_motion is not None else 0.0
        motion_max = selected_motion['max'] if selected_motion is not None else 0.0
        prev_size_text = f"{prev_ball_size:.1f}" if prev_ball_size is not None else "None"

        print(
            f"  DEBUG: Focus-loss retrack selected contour at ({cx},{cy}) "
            f"area={area:.1f}px distance={distance:.1f}px score={best_score:.1f} "
            f"prev_ball_size={prev_size_text} filter={filter_key} "
            f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}"
        )
        return {
            'pos': (cx, cy),
            'area': area,
            'hsv': hsv_at_point,
            'distance': distance,
            'score': best_score,
            'motion_mean': motion_mean,
            'motion_max': motion_max,
            'filter_key': filter_key,
        }

    def retrack_with_alt2_hsv(self, search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                              lower=None, upper=None, frame_gray=None, filter_key="alt2", sparse_mode=False,
                              ignore_false_points=False, prefer_predicted_path=False):
        """Re-run detection with alternative 2 HSV (H 46-72) when stuck."""
        debug_label = "Alt2" if filter_key == "alt2" else str(filter_key)
        hsv_lower = lower if lower is not None else self.alt2_hsv_lower
        hsv_upper = upper if upper is not None else self.alt2_hsv_upper
        if hsv_lower is None or hsv_upper is None:
            return None
        
        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        if sparse_mode:
            mask_alt2 = self._build_sparse_low_s_mask(hsv_frame, hsv_lower, hsv_upper)
        else:
            mask_alt2 = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
            kernel = np.ones((2, 2), np.uint8)
            if filter_key in ("alts_20", "h_10"):
                mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_CLOSE, kernel)
                mask_alt2 = cv2.dilate(mask_alt2, kernel, iterations=1)
            else:
                mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_OPEN, kernel)
                mask_alt2 = cv2.morphologyEx(mask_alt2, cv2.MORPH_CLOSE, kernel)
        contours_alt2, _ = cv2.findContours(mask_alt2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours_alt2:
            print(f"  DEBUG: {debug_label} retrack found no contours")
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
                frame_shape = frame_gray.shape if frame_gray is not None else None
                min_area = self._min_area_for_previous_ball_size(self.ball_size, prev_pos, frame_shape)
                if area < min_area:
                    print(f"  DEBUG: retrack_using_alt2 skipping too-small contour area={area:.1f} "
                          f"prev_ball_size={self.ball_size:.1f} min_area={min_area}")
                    continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1

            if not allow_inactive:
                ignored_entry = self._find_ignored_tracking_position((cx, cy), filter_key=filter_key)
                if ignored_entry is not None and not ignore_false_points:
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
            predicted_distance = None
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
                trajectory_priority_candidate = False
                if (prefer_predicted_path and predicted_distance is not None and prev_pos is not None and
                        self.last_motion is not None):
                    predicted_cap = max(14.0, min(34.0, self.last_motion.get('distance', 0.0) * 3.0 + 8.0))
                    trajectory_priority_candidate = (
                        predicted_distance <= predicted_cap and
                        distance <= max(50.0, predicted_cap + 12.0) and
                        (motion_mean >= 4.0 or motion_max >= 20.0 or area >= 3.0)
                    )
                    if trajectory_priority_candidate:
                        score -= min(60.0, max(0.0, predicted_cap - predicted_distance) * 4.0)
                if frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0 and not trajectory_priority_candidate:
                    print(f"  DEBUG: retrack_using_alt2 skipping frame0 hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                    continue
                static_hotspot = ((area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                if static_hotspot and not trajectory_priority_candidate:
                    print(f"  DEBUG: retrack_using_alt2 skipping static hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                    continue

            if score < best_score:
                best_score = score
                best = (cx, cy, area, distance)

        if best is None:
            print(f"  DEBUG: {debug_label} retrack found no valid candidate")
            return None

        cx, cy, area, distance = best
        hsv_values = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        local_x = max(0, min(search_frame.shape[1] - 1, cx - x1))
        local_y = max(0, min(search_frame.shape[0] - 1, cy - y1))
        hsv_at_point = hsv_values[local_y, local_x]
        selected_motion = self._candidate_motion_metrics(frame_gray, cx, cy)
        motion_mean = selected_motion['mean'] if selected_motion is not None else 0.0
        motion_max = selected_motion['max'] if selected_motion is not None else 0.0
        prev_size_text = f"{prev_ball_size:.1f}" if prev_ball_size is not None else "None"

        print(
            f"  DEBUG: {debug_label} retrack selected contour at ({cx},{cy}) "
            f"area={area:.1f}px distance={distance:.1f}px score={best_score:.1f} "
            f"prev_ball_size={prev_size_text} filter={filter_key} "
            f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}"
        )
        return {
            'pos': (cx, cy),
            'area': area,
            'hsv': hsv_at_point,
            'distance': distance,
            'score': best_score,
            'motion_mean': motion_mean,
            'motion_max': motion_max,
            'filter_key': filter_key,
        }

    def _retrack_local_alts20_hsv(self, frame, frame_gray, predicted_point=None, radius=80):
        """Try the low-saturation bright ball HSV only in a tight local window."""
        if self.ball_center is None or self.alts20_hsv_lower is None or self.alts20_hsv_upper is None:
            return None

        prev_pos = self.ball_center
        ref_x, ref_y = prev_pos
        local_radius = int(radius)
        x1 = max(0, ref_x - local_radius)
        y1 = max(0, ref_y - local_radius)
        x2 = min(frame.shape[1], ref_x + local_radius)
        y2 = min(frame.shape[0], ref_y + local_radius)
        local_frame = frame[y1:y2, x1:x2]
        if local_frame.size == 0:
            return None

        retrack = self.retrack_with_alt2_hsv(
            local_frame, x1, y1, prev_pos, predicted_point, self.ball_size, False,
            lower=self.alts20_hsv_lower, upper=self.alts20_hsv_upper,
            frame_gray=frame_gray, filter_key="alts_20"
        )
        if retrack is None:
            return None

        cx, cy = retrack['pos']
        step_distance = math.hypot(cx - prev_pos[0], cy - prev_pos[1])
        predicted_distance = (
            math.hypot(cx - predicted_point[0], cy - predicted_point[1])
            if predicted_point is not None else None
        )
        if step_distance > local_radius:
            print(
                f"  DEBUG: Rejecting alts_20 local recover at {retrack['pos']} - "
                f"distance={step_distance:.1f}px > local_radius={local_radius}px"
            )
            return None
        if predicted_distance is not None and predicted_distance > max(95.0, local_radius + 20.0):
            print(
                f"  DEBUG: Rejecting alts_20 local recover at {retrack['pos']} - "
                f"predicted_distance={predicted_distance:.1f}px"
            )
            return None
        if retrack['area'] > 80.0:
            print(
                f"  DEBUG: Rejecting alts_20 local recover at {retrack['pos']} - "
                f"area={retrack['area']:.1f}px too large"
            )
            return None

        return retrack

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

    def _should_try_h10_recover(self, frame, predicted_point, allow_inactive):
        if allow_inactive or self.ball_center is None:
            return False
        frame_height = frame.shape[0]
        top_limit = max(260, int(frame_height * 0.16))
        current_y = self.ball_center[1]
        predicted_y = predicted_point[1] if predicted_point is not None else current_y
        last_dy = None if self.last_motion is None else self.last_motion.get('dy')
        current_size = float(self.ball_size) if self.ball_size is not None else 0.0
        return (
            current_y <= top_limit and
            predicted_y <= (top_limit + 18) and
            current_size <= 12.0 and
            (last_dy is None or last_dy <= 0.0)
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
                lower=lower, upper=upper, frame_gray=frame_gray,
                filter_key=self._normalize_false_point_filter_key(label)
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

    def _retrack_recent_return_tiny_hue(self, frame, frame_gray):
        """Try a tiny local hue recover only for recent offscreen-return bounce frames."""
        if self.ball_center is None or self.last_motion is None:
            return None
        if not self._recent_offscreen_return_hold_active(window_frames=8):
            return None

        prev_pos = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        last_distance = float(self.last_motion.get('distance', 0.0) or 0.0)
        ref_x, ref_y = prev_pos

        if prev_size > 18.0 or last_distance < 35.0 or last_distance > 75.0:
            return None
        if ref_y < 150 or ref_y > 260:
            return None

        x_radius = 40
        up_radius = 10
        down_radius = 55
        x1 = max(0, ref_x - x_radius)
        y1 = max(0, ref_y - up_radius)
        x2 = min(frame.shape[1], ref_x + x_radius + 1)
        y2 = min(frame.shape[0], ref_y + down_radius + 1)
        local_frame = frame[y1:y2, x1:x2]
        if local_frame.size == 0:
            return None

        hsv_local = cv2.cvtColor(local_frame, cv2.COLOR_BGR2HSV)
        hint_specs = [
            (
                "recent_tiny_center",
                np.array([148, 10, 80], dtype=np.uint8),
                np.array([160, 60, 255], dtype=np.uint8),
                6.0,
            ),
            (
                "recent_tiny_edge",
                np.array([118, 12, 55], dtype=np.uint8),
                np.array([122, 60, 245], dtype=np.uint8),
                0.0,
            ),
        ]
        expected_dy = max(18.0, min(36.0, last_distance * 0.55))
        best = None
        best_score = float('inf')

        for label, lower, upper, label_bonus in hint_specs:
            if lower is None or upper is None:
                continue
            mask = cv2.inRange(hsv_local, lower, upper)
            if cv2.countNonZero(mask) == 0:
                continue

            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for idx in range(1, count):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < 1 or area > 18:
                    continue

                bbox_w = int(stats[idx, cv2.CC_STAT_WIDTH])
                bbox_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
                comp_cx, comp_cy = centroids[idx]
                cx = int(round(comp_cx)) + x1
                cy = int(round(comp_cy)) + y1
                if label == "recent_tiny_center" and area >= 5:
                    cx = min(frame.shape[1] - 1, cx + min(6, max(0, bbox_w // 4)))
                    cy = max(0, cy - min(6, max(3, bbox_h + 3)))
                dx = cx - ref_x
                dy = cy - ref_y
                if dy < 8 or dy > 48:
                    continue
                if abs(dx) > 28:
                    continue

                motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                motion_mean = motion_metrics['mean'] if motion_metrics is not None else 0.0
                motion_max = motion_metrics['max'] if motion_metrics is not None else 0.0
                score = abs(dx) * 2.6
                score += abs(dy - expected_dy) * 1.5
                score -= min(12.0, area * 0.8)
                score -= min(8.0, motion_max * 0.12)
                score -= label_bonus

                if score < best_score:
                    best_score = score
                    best = {
                        'pos': (cx, cy),
                        'area': float(area),
                        'score': score,
                        'motion_mean': motion_mean,
                        'motion_max': motion_max,
                        'filter_key': label,
                    }

        if best is None:
            return None

        local_x = max(0, min(local_frame.shape[1] - 1, best['pos'][0] - x1))
        local_y = max(0, min(local_frame.shape[0] - 1, best['pos'][1] - y1))
        best['hsv'] = hsv_local[local_y, local_x]
        print(
            f"  DEBUG: recent tiny-hue recover selected contour at {best['pos']} "
            f"area={best['area']:.1f}px score={best['score']:.1f} "
            f"motion={best['motion_mean']:.1f}/{best['motion_max']:.1f} "
            f"filter={best['filter_key']}"
        )
        return best

    def _retrack_recent_return_bounce_continue(self, frame, frame_gray, predicted_point=None):
        """Recover the next bounce step after a recent-return rescue using a tiny local search."""
        if self.ball_center is None or self.last_motion is None:
            return None
        if not self._recent_return_bounce_recover_active(window_frames=2):
            return None

        prev_pos = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        last_distance = float(self.last_motion.get('distance', 0.0) or 0.0)
        ref_x, ref_y = prev_pos
        if prev_size > 24.0 or ref_y < 220 or ref_y > 320:
            return None

        x_radius = int(max(50, min(85, max(58.0, last_distance * 2.2))))
        up_radius = 12
        down_radius = 85
        x1 = max(0, ref_x - x_radius)
        y1 = max(0, ref_y - up_radius)
        x2 = min(frame.shape[1], ref_x + x_radius + 1)
        y2 = min(frame.shape[0], ref_y + down_radius + 1)
        local_frame = frame[y1:y2, x1:x2]
        if local_frame.size == 0:
            return None

        hsv_local = cv2.cvtColor(local_frame, cv2.COLOR_BGR2HSV)
        expected_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        expected_dy = max(14.0, min(42.0, float(self.last_motion.get('dy', 0.0) or 0.0) * 1.0))
        default_predicted = (
            int(round(ref_x + expected_dx)),
            int(round(ref_y + expected_dy)),
        )
        active_predicted = predicted_point if predicted_point is not None else default_predicted
        best = None
        best_score = float('inf')

        recover_specs = [
            (
                "recent_bounce_low_s",
                np.array([20, 10, 80], dtype=np.uint8),
                np.array([80, 110, 235], dtype=np.uint8),
                True,
                5.0,
            ),
            (
                "recent_bounce_h10",
                self.h10_hsv_lower,
                self.h10_hsv_upper,
                False,
                0.0,
            ),
            (
                "recent_bounce_s30",
                self.s30_hsv_lower,
                self.s30_hsv_upper,
                False,
                0.0,
            ),
        ]

        for label, lower, upper, use_close, label_bonus in recover_specs:
            if lower is None or upper is None:
                continue
            mask = cv2.inRange(hsv_local, lower, upper)
            if use_close:
                kernel = np.ones((2, 2), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
            if cv2.countNonZero(mask) == 0:
                continue

            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for idx in range(1, count):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < 1 or area > max(60.0, prev_size * 4.5):
                    continue

                comp_cx, comp_cy = centroids[idx]
                cx = int(round(comp_cx)) + x1
                cy = int(round(comp_cy)) + y1
                if self._find_ignored_tracking_position((cx, cy)) is not None:
                    continue

                dx = cx - ref_x
                dy = cy - ref_y
                distance = math.hypot(dx, dy)
                if dy < 6.0 or dy > 60.0:
                    continue
                if abs(dx) > 60.0:
                    continue
                if distance < 10.0 or distance > 90.0:
                    continue

                motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                motion_mean = motion_metrics['mean'] if motion_metrics is not None else 0.0
                motion_max = motion_metrics['max'] if motion_metrics is not None else 0.0
                frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                if frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0:
                    continue

                predicted_distance = math.hypot(cx - active_predicted[0], cy - active_predicted[1])
                score = predicted_distance * 1.0
                score += abs(dy - expected_dy) * 0.8
                score += abs(dx - expected_dx) * 0.9
                score += abs(dx) * 0.6
                score -= min(40.0, area * 0.7)
                score -= min(18.0, motion_mean * 0.7)
                score -= min(22.0, motion_max * 0.12)
                score -= label_bonus

                if score < best_score:
                    best_score = score
                    best = {
                        'pos': (cx, cy),
                        'area': float(area),
                        'score': score,
                        'motion_mean': motion_mean,
                        'motion_max': motion_max,
                        'filter_key': label,
                    }

        if best is None:
            return None

        local_x = max(0, min(local_frame.shape[1] - 1, best['pos'][0] - x1))
        local_y = max(0, min(local_frame.shape[0] - 1, best['pos'][1] - y1))
        best['hsv'] = hsv_local[local_y, local_x]
        print(
            f"  DEBUG: recent bounce-continue recover selected contour at {best['pos']} "
            f"area={best['area']:.1f}px score={best['score']:.1f} "
            f"motion={best['motion_mean']:.1f}/{best['motion_max']:.1f} "
            f"filter={best['filter_key']}"
        )
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
            ("alts_20", self.alts20_hsv_lower, self.alts20_hsv_upper),
            ("alts9_11", self.alts9_11_hsv_lower, self.alts9_11_hsv_upper),
        ]
        best = None
        best_distance = None
        for label, lower, upper in hsv_candidates:
            if lower is None or upper is None:
                continue
            retrack = self.retrack_with_alt2_hsv(
                search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, False,
                lower=lower, upper=upper, frame_gray=frame_gray, filter_key=label,
                sparse_mode=(label == "alts9_11")
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
        raw_mask = cv2.inRange(hsv_region, lower, upper)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        analysis_mask = raw_mask.copy()
        hidden_false_points = self._erase_diagnostic_false_points(analysis_mask, x1, y1)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        debug_text = f"H={lower[0]}-{upper[0]} S={lower[1]}-{upper[1]} V={lower[2]}-{upper[2]}"
        if hidden_false_points:
            debug_text += f" | Hidden false: {hidden_false_points}"
        cv2.putText(local_display, debug_text,
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
        if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.s30_hsv_lower, self.s30_hsv_upper,
                                        f"Predicted {frame_index} (s_30)")
        if self.h10_hsv_lower is not None and self.h10_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.h10_hsv_lower, self.h10_hsv_upper,
                                        f"Predicted {frame_index} (h_10)")
        if self.alts20_hsv_lower is not None and self.alts20_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alts20_hsv_lower, self.alts20_hsv_upper,
                                        f"Predicted {frame_index} (alts_20)")
        if self.alts9_11_hsv_lower is not None and self.alts9_11_hsv_upper is not None:
            self.show_alternative_debug(frame, point, self.alts9_11_hsv_lower, self.alts9_11_hsv_upper,
                                        f"Predicted {frame_index} (alts9_11)")

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
                
                # Newer configs may store the primary HSV set under regular_court,
                # with behind_net either present (dual mode) or omitted (single mode).
                if "regular_court" in config:
                    regular_cfg = config["regular_court"]
                    self.hsv_regular = {
                        'lower': np.array([regular_cfg["h_min"], regular_cfg["s_min"], regular_cfg["v_min"]], dtype=np.uint8),
                        'upper': np.array([regular_cfg["h_max"], regular_cfg["s_max"], regular_cfg["v_max"]], dtype=np.uint8)
                    }
                    self.pre_focus_hsv_regular = {
                        'lower': self.hsv_regular['lower'].copy(),
                        'upper': self.hsv_regular['upper'].copy()
                    }
                    self.pre_focus_hsv_regular['upper'][0] = min(self.pre_focus_hsv_regular['upper'][0], 73)

                    if "behind_net" in config:
                        behind_net_cfg = config["behind_net"]
                        self.hsv_behind_net = {
                            'lower': np.array([behind_net_cfg["h_min"], behind_net_cfg["s_min"], behind_net_cfg["v_min"]], dtype=np.uint8),
                            'upper': np.array([behind_net_cfg["h_max"], behind_net_cfg["s_max"], behind_net_cfg["v_max"]], dtype=np.uint8)
                        }
                        self.pre_focus_hsv_behind_net = {
                            'lower': self.hsv_behind_net['lower'].copy(),
                            'upper': self.hsv_behind_net['upper'].copy()
                        }
                        self.pre_focus_hsv_behind_net['upper'][0] = min(self.pre_focus_hsv_behind_net['upper'][0], 73)
                        self.net_area_y_min = config.get("net_area_y_min", 250)
                        self.net_area_y_max = config.get("net_area_y_max", 350)
                        self.net_area_points = config.get("net_area_points", [])
                        
                        # Load high/low net boundaries if available
                        if "high_net_y_min" in config:
                            self.high_net_y_min = config["high_net_y_min"]
                            self.high_net_y_max = config["high_net_y_max"]
                        if "low_net_y_min" in config:
                            self.low_net_y_min = config["low_net_y_min"]
                            self.low_net_y_max = config["low_net_y_max"]
                        
                        if self.disable_behind_net_mode:
                            print(f"Loaded REGULAR HSV config:")
                            print(f"  Regular court: H:{regular_cfg['h_min']}-{regular_cfg['h_max']}, S:{regular_cfg['s_min']}-{regular_cfg['s_max']}, V:{regular_cfg['v_min']}-{regular_cfg['v_max']}")
                            print(f"  Behind-net HSV disabled")
                        else:
                            print(f"Loaded DUAL HSV config:")
                            print(f"  Regular court: H:{regular_cfg['h_min']}-{regular_cfg['h_max']}, S:{regular_cfg['s_min']}-{regular_cfg['s_max']}, V:{regular_cfg['v_min']}-{regular_cfg['v_max']}")
                            print(f"  Behind net: H:{behind_net_cfg['h_min']}-{behind_net_cfg['h_max']}, S:{behind_net_cfg['s_min']}-{behind_net_cfg['s_max']}, V:{behind_net_cfg['v_min']}-{behind_net_cfg['v_max']}")
                            print(f"  Net area: Y={self.net_area_y_min}-{self.net_area_y_max}")
                            if hasattr(self, 'high_net_y_min') and hasattr(self, 'low_net_y_min'):
                                print(f"    HIGH NET: Y={self.high_net_y_min}-{self.high_net_y_max}")
                                print(f"    LOW NET: Y={self.low_net_y_min}-{self.low_net_y_max}")
                    else:
                        self.hsv_behind_net = None
                        self.pre_focus_hsv_behind_net = None
                        print(f"Loaded REGULAR HSV config:")
                        print(f"  Regular court: H:{regular_cfg['h_min']}-{regular_cfg['h_max']}, S:{regular_cfg['s_min']}-{regular_cfg['s_max']}, V:{regular_cfg['v_min']}-{regular_cfg['v_max']}")

                    # Set initial HSV to regular court
                    self.hsv_lower = self.hsv_regular['lower']
                    self.hsv_upper = self.hsv_regular['upper']

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
                    self.direction_change_min_degrees = float(
                        config.get('direction_change_min_degrees', self.direction_change_min_degrees)
                    )
                    print(f"  Serve direction: {self.serve_direction_label()}")
                    print(f"  Direction-change threshold: {self.direction_change_min_degrees:.1f} deg")
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
                if self.hsv_regular is not None:
                    self.s30_hsv_lower = self.hsv_regular['lower'].copy()
                    self.s30_hsv_upper = self.hsv_regular['upper'].copy()
                elif self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
                    self.s30_hsv_lower = self.primary_hsv_lower.copy()
                    self.s30_hsv_upper = self.primary_hsv_upper.copy()
                if self.s30_hsv_lower is not None:
                    self.s30_hsv_lower[1] = min(int(self.s30_hsv_lower[1]), 30)
                if "s_30" in config:
                    self.s30_hsv_lower = np.array([
                        config["s_30"]["h_min"], config["s_30"]["s_min"], config["s_30"]["v_min"]
                    ], dtype=np.uint8)
                    self.s30_hsv_upper = np.array([
                        config["s_30"]["h_max"], config["s_30"]["s_max"], config["s_30"]["v_max"]
                    ], dtype=np.uint8)
                self.h10_hsv_lower = np.array([11, 30, 130], dtype=np.uint8)
                self.h10_hsv_upper = np.array([72, 255, 255], dtype=np.uint8)
                if "h_10" in config:
                    self.h10_hsv_lower = np.array([
                        config["h_10"]["h_min"], config["h_10"]["s_min"], config["h_10"]["v_min"]
                    ], dtype=np.uint8)
                    self.h10_hsv_upper = np.array([
                        config["h_10"]["h_max"], config["h_10"]["s_max"], config["h_10"]["v_max"]
                    ], dtype=np.uint8)
                self.alts20_hsv_lower = np.array([50, 20, 130], dtype=np.uint8)
                self.alts20_hsv_upper = np.array([60, 30, 250], dtype=np.uint8)
                if "alts_20" in config:
                    self.alts20_hsv_lower = np.array([
                        config["alts_20"]["h_min"], config["alts_20"]["s_min"], config["alts_20"]["v_min"]
                    ], dtype=np.uint8)
                    self.alts20_hsv_upper = np.array([
                        config["alts_20"]["h_max"], config["alts_20"]["s_max"], config["alts_20"]["v_max"]
                    ], dtype=np.uint8)
                if "alts9_11" in config:
                    self.alts9_11_hsv_lower = np.array([
                        config["alts9_11"]["h_min"], config["alts9_11"]["s_min"], config["alts9_11"]["v_min"]
                    ], dtype=np.uint8)
                    self.alts9_11_hsv_upper = np.array([
                        config["alts9_11"]["h_max"], config["alts9_11"]["s_max"], config["alts9_11"]["v_max"]
                    ], dtype=np.uint8)
                # Use primary as active by default (full config range)
                self.hsv_lower = self.primary_hsv_lower
                self.hsv_upper = self.primary_hsv_upper
                self._build_frame0_background_hotspots()
                self._load_persistent_false_points()
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default HSV values")
            return None
    
    def select_hsv_for_position(self, y_position, at_edge=False):
        """Select appropriate HSV config based on ball Y position."""
        if self.using_alt6_hsv and self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
            return self.alt6_hsv_lower, self.alt6_hsv_upper, "alternative_6"
        if self.using_alt3_hsv and self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
            return self.alt3_hsv_lower, self.alt3_hsv_upper, "alternative_3"
        if self.using_alt4_hsv and self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
            return self.alt4_hsv_lower, self.alt4_hsv_upper, "alternative_4"
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
            self.net_area_points = [list(map(int, pt)) for pt in points]
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
            if getattr(self, "net_area_points", None):
                config["net_area_points"] = self.net_area_points
            
            # Save detailed high/low net boundaries if available
            if hasattr(self, 'high_net_y_min'):
                config["high_net_y_min"] = int(self.high_net_y_min)
                config["high_net_y_max"] = int(self.high_net_y_max)
            if hasattr(self, 'low_net_y_min'):
                config["low_net_y_min"] = int(self.low_net_y_min)
                config["low_net_y_max"] = int(self.low_net_y_max)
            
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"Net area boundaries saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving net area: {e}")

    def _draw_net_area_overlay(self, result, scale=1.0):
        """Draw the currently configured net area on the display frame.

        If the exact 6-point net polygon is available, draw that. Otherwise,
        fall back to the stored high/low Y bands so we can still see what the
        tracker currently considers the net region.
        """
        if scale <= 0:
            scale = 1.0

        display_height, display_width = result.shape[:2]
        orig_width = int(round(display_width / scale))

        left_x = int(getattr(self, 'serve_area_x_min', 0))
        right_x = int(getattr(self, 'serve_area_x_max', orig_width))
        left_x = max(0, min(orig_width - 1, left_x)) if orig_width > 0 else 0
        right_x = max(left_x + 1, min(orig_width, right_x)) if orig_width > 0 else display_width

        net_points = getattr(self, 'net_area_points', None) or []
        if len(net_points) >= 6:
            pts = np.array([
                net_points[0],
                net_points[1],
                net_points[2],
                net_points[3],
                net_points[4],
                net_points[5],
            ], dtype=np.int32)
            pts_scaled = np.array(
                [[int(round(px * scale)), int(round(py * scale))] for px, py in pts],
                dtype=np.int32
            )
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts_scaled], (0, 220, 255))
            result[:] = cv2.addWeighted(result, 0.82, overlay, 0.18, 0)
            cv2.polylines(result, [pts_scaled], True, (0, 220, 255), 2)

            cv2.polylines(result, [pts_scaled[:3]], False, (0, 255, 0), 2)
            cv2.polylines(result, [pts_scaled[3:]], False, (255, 180, 0), 2)

            label_x = int(min(pt[0] for pt in pts_scaled)) + 6
            label_y = int(min(pt[1] for pt in pts_scaled)) + 22
            cv2.putText(result, "NET AREA", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            return

        if hasattr(self, 'high_net_y_min') and hasattr(self, 'high_net_y_max'):
            hx1 = int(round(left_x * scale))
            hx2 = int(round(right_x * scale))
            hy1 = int(round(self.high_net_y_min * scale))
            hy2 = int(round(self.high_net_y_max * scale))
            if hy2 > hy1:
                overlay = result.copy()
                cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (0, 255, 0), -1)
                result[:] = cv2.addWeighted(result, 0.90, overlay, 0.10, 0)
                cv2.rectangle(result, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                cv2.putText(result, "HIGH NET BAND", (hx1 + 6, hy1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if hasattr(self, 'low_net_y_min') and hasattr(self, 'low_net_y_max'):
            lx1 = int(round(left_x * scale))
            lx2 = int(round(right_x * scale))
            ly1 = int(round(self.low_net_y_min * scale))
            ly2 = int(round(self.low_net_y_max * scale))
            if ly2 > ly1:
                overlay = result.copy()
                cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (255, 180, 0), -1)
                result[:] = cv2.addWeighted(result, 0.90, overlay, 0.10, 0)
                cv2.rectangle(result, (lx1, ly1), (lx2, ly2), (255, 180, 0), 2)
                cv2.putText(result, "LOW NET BAND", (lx1 + 6, ly2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            nx1 = int(round(left_x * scale))
            nx2 = int(round(right_x * scale))
            ny1 = int(round(self.net_area_y_min * scale))
            ny2 = int(round(self.net_area_y_max * scale))
            if ny2 > ny1:
                cv2.rectangle(result, (nx1, ny1), (nx2, ny2), (0, 220, 255), 1)
                cv2.putText(result, "NET AREA", (nx1 + 6, ny1 + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    def _build_white_line_mask_hsv(self, frame):
        """Detect painted white court features using an HSV white filter."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 170], dtype=np.uint8)
        upper = np.array([179, 70, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        height, width = mask.shape[:2]
        # Ignore the timestamp / camera overlay zone in the upper-left.
        mask[:int(height * 0.12), :int(width * 0.40)] = 0

        kernel_small = np.ones((2, 2), np.uint8)
        kernel_close = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        return mask

    def _build_white_line_visual_model(self, frame):
        """Build a display-only model of white court lines from HSV white filtering."""
        frame_shape = frame.shape[:2]
        if (self._white_line_visual_model is not None and
                self._white_line_visual_frame_shape == frame_shape):
            return self._white_line_visual_model

        height, width = frame_shape
        mask = self._build_white_line_mask_hsv(frame)
        min_line_length = max(120, int(width * 0.06))
        lines = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=min_line_length,
            maxLineGap=24,
        )
        if lines is None:
            self._white_line_visual_model = None
            self._white_line_visual_frame_shape = frame_shape
            return None

        segments = []
        horizontal_segments = []
        left_candidates = []
        right_candidates = []
        y_ref = int(height * 0.74)

        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_line_length:
                continue

            angle = math.degrees(math.atan2(dy, dx))
            seg = {
                'p1': (x1, y1),
                'p2': (x2, y2),
                'length': length,
                'angle': angle,
                'x_mid': (x1 + x2) / 2.0,
                'y_mid': (y1 + y2) / 2.0,
            }
            segments.append(seg)

            if abs(angle) <= 18.0:
                horizontal_segments.append(seg)

            if abs(dy) >= 20.0:
                a = dx / dy
                b = x1 - a * y1
                x_ref = a * y_ref + b
                if -75.0 < angle < -42.0:
                    entry = dict(seg)
                    entry.update({'a': a, 'b': b, 'x_ref': x_ref})
                    if width * 0.14 <= x_ref <= width * 0.32:
                        left_candidates.append(entry)
                elif 42.0 < angle < 75.0:
                    entry = dict(seg)
                    entry.update({'a': a, 'b': b, 'x_ref': x_ref})
                    if width * 0.68 <= x_ref <= width * 0.83:
                        right_candidates.append(entry)

        if not segments:
            self._white_line_visual_model = None
            self._white_line_visual_frame_shape = frame_shape
            return None

        segments = sorted(segments, key=lambda seg: seg['length'], reverse=True)[:80]
        left_sideline = max(left_candidates, key=lambda seg: seg['length']) if left_candidates else None
        right_sideline = max(right_candidates, key=lambda seg: seg['length']) if right_candidates else None

        net_y = None
        if hasattr(self, 'high_net_y_min') and hasattr(self, 'high_net_y_max'):
            net_y = float((self.high_net_y_min + self.high_net_y_max) / 2.0)
        elif hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            net_y = float((self.net_area_y_min + self.net_area_y_max) / 2.0)

        high_net_line = None
        if horizontal_segments and hasattr(self, 'high_net_y_min') and hasattr(self, 'high_net_y_max'):
            high_min = float(self.high_net_y_min) - 30.0
            high_max = float(self.high_net_y_max) + 30.0
            high_candidates = [
                seg for seg in horizontal_segments
                if high_min <= seg['y_mid'] <= high_max
            ]
            if high_candidates:
                high_net_line = max(high_candidates, key=lambda seg: seg['length'])

        service_line = None
        if horizontal_segments and net_y is not None:
            long_horiz = [seg for seg in horizontal_segments if seg['length'] >= max(250.0, width * 0.08)]
            if self.serve_direction_dy < 0:
                above_net = [
                    seg for seg in long_horiz
                    if height * 0.06 <= seg['y_mid'] <= (net_y - 40.0)
                ]
                if above_net:
                    service_line = max(above_net, key=lambda seg: (seg['y_mid'], seg['length']))
            elif self.serve_direction_dy > 0:
                below_net = [
                    seg for seg in long_horiz
                    if (net_y + 40.0) <= seg['y_mid'] <= height * 0.94
                ]
                if below_net:
                    service_line = min(below_net, key=lambda seg: (seg['y_mid'], -seg['length']))

        service_area_polygon = None
        if left_sideline is not None and right_sideline is not None and service_line is not None and net_y is not None:
            service_y = float(service_line['y_mid'])
            left_service_x = left_sideline['a'] * service_y + left_sideline['b']
            right_service_x = right_sideline['a'] * service_y + right_sideline['b']
            left_net_x = left_sideline['a'] * net_y + left_sideline['b']
            right_net_x = right_sideline['a'] * net_y + right_sideline['b']
            service_area_polygon = [
                (left_service_x, service_y),
                (right_service_x, service_y),
                (right_net_x, net_y),
                (left_net_x, net_y),
            ]

        self._white_line_visual_model = {
            'mask': mask,
            'segments': segments,
            'left_sideline': left_sideline,
            'right_sideline': right_sideline,
            'service_line': service_line,
            'high_net_line': high_net_line,
            'service_area_polygon': service_area_polygon,
            'white_hsv_lower': (0, 0, 170),
            'white_hsv_upper': (179, 70, 255),
        }
        self._white_line_visual_frame_shape = frame_shape
        print(
            f"  DEBUG: White-line HSV model built: segments={len(segments)} "
            f"sidelines={'yes' if left_sideline and right_sideline else 'no'} "
            f"service_line={'yes' if service_line else 'no'} "
            f"high_net={'yes' if high_net_line else 'no'}"
        )
        return self._white_line_visual_model

    def _draw_white_line_visual_overlay(self, result, scale=1.0):
        """Draw white-line HSV detection results for inspection."""
        model = getattr(self, '_white_line_visual_model', None)
        if model is None:
            return

        mask = model.get('mask')
        if mask is not None:
            if mask.shape[:2] != result.shape[:2]:
                mask = cv2.resize(mask, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask is not None and mask.shape[:2] == result.shape[:2]:
            mask_overlay = result.copy()
            mask_overlay[mask > 0] = (255, 255, 0)
            result[:] = cv2.addWeighted(result, 0.84, mask_overlay, 0.16, 0)

        overlay = result.copy()
        for seg in model.get('segments', []):
            p1 = (int(round(seg['p1'][0] * scale)), int(round(seg['p1'][1] * scale)))
            p2 = (int(round(seg['p2'][0] * scale)), int(round(seg['p2'][1] * scale)))
            cv2.line(overlay, p1, p2, (255, 255, 0), 1)
        result[:] = cv2.addWeighted(result, 0.88, overlay, 0.12, 0)

        for key, color, label in (
            ('left_sideline', (255, 0, 255), "LEFT SIDELINE"),
            ('right_sideline', (255, 0, 255), "RIGHT SIDELINE"),
            ('service_line', (0, 165, 255), "SERVICE LINE"),
            ('high_net_line', (0, 255, 255), "HIGH NET LINE"),
        ):
            seg = model.get(key)
            if seg is None:
                continue
            p1 = (int(round(seg['p1'][0] * scale)), int(round(seg['p1'][1] * scale)))
            p2 = (int(round(seg['p2'][0] * scale)), int(round(seg['p2'][1] * scale)))
            cv2.line(result, p1, p2, color, 3)
            label_x = int(round(((seg['p1'][0] + seg['p2'][0]) * 0.5) * scale))
            label_y = int(round(((seg['p1'][1] + seg['p2'][1]) * 0.5) * scale)) - 8
            cv2.putText(result, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        service_poly = model.get('service_area_polygon')
        if service_poly:
            pts = np.array(
                [[int(round(px * scale)), int(round(py * scale))] for px, py in service_poly],
                dtype=np.int32
            )
            poly_overlay = result.copy()
            cv2.fillPoly(poly_overlay, [pts], (0, 200, 120))
            result[:] = cv2.addWeighted(result, 0.86, poly_overlay, 0.14, 0)
            cv2.polylines(result, [pts], True, (0, 200, 120), 2)
            px_min = min(pt[0] for pt in pts)
            py_min = min(pt[1] for pt in pts)
            cv2.putText(result, "WHITE SERVICE AREA", (px_min + 6, py_min + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 120), 2)

        if hasattr(self, 'serve_area_x_min') and hasattr(self, 'serve_area_y_min'):
            sx1 = int(round(self.serve_area_x_min * scale))
            sy1 = int(round(self.serve_area_y_min * scale))
            sx2 = int(round(self.serve_area_x_max * scale))
            sy2 = int(round(self.serve_area_y_max * scale))
            cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 128, 255), 2)
            cv2.putText(result, "CONFIG SERVE AREA", (sx1 + 6, sy1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 2)

        net_points = getattr(self, 'net_area_points', None) or []
        if len(net_points) >= 6:
            pts = np.array(
                [[int(round(px * scale)), int(round(py * scale))] for px, py in net_points[:6]],
                dtype=np.int32
            )
            net_overlay = result.copy()
            cv2.fillPoly(net_overlay, [pts], (0, 64, 255))
            result[:] = cv2.addWeighted(result, 0.90, net_overlay, 0.10, 0)
            cv2.polylines(result, [pts], True, (0, 64, 255), 3)
            cv2.polylines(result, [pts[:3]], False, (0, 255, 255), 3)
            cv2.putText(result, "CONFIG NET AREA", (int(min(pt[0] for pt in pts)) + 6, int(min(pt[1] for pt in pts)) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 64, 255), 2)
            cv2.putText(result, "CONFIG HIGH NET", (int(min(pt[0] for pt in pts[:3])) + 6, int(min(pt[1] for pt in pts[:3])) + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        lower = model.get('white_hsv_lower', (0, 0, 170))
        upper = model.get('white_hsv_upper', (179, 70, 255))
        cv2.putText(
            result,
            f"WHITE HSV LINES H={lower[0]}-{upper[0]} S={lower[1]}-{upper[1]} V={lower[2]}-{upper[2]}",
            (10, max(210, int(210 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    def print_click_hsv_neighborhood(self, frame, point, display_point=None, scale=None):
        """Print the clicked frame position plus a 3x3 HSV neighborhood."""
        print("\n=== MOUSE CLICK HSV DEBUG ===")
        if frame is None:
            print("No current frame available for HSV sampling.")
            return

        x, y = int(point[0]), int(point[1])
        frame_height, frame_width = frame.shape[:2]
        if display_point is not None:
            scale_text = f", display_scale={scale:.4f}" if scale else ""
            print(f"Display click: ({display_point[0]}, {display_point[1]}) -> frame: ({x}, {y}){scale_text}")
        else:
            print(f"Frame click: ({x}, {y})")

        if x < 0 or y < 0 or x >= frame_width or y >= frame_height:
            print(f"Click is outside frame bounds: width={frame_width}, height={frame_height}")
            return

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        center_hsv = hsv_frame[y, x]
        print(f"Center HSV: H={int(center_hsv[0])}, S={int(center_hsv[1])}, V={int(center_hsv[2])}")
        print("3x3 HSV neighborhood (center marked with *):")

        for dy in (-1, 0, 1):
            row_values = []
            for dx in (-1, 0, 1):
                px = x + dx
                py = y + dy
                marker = "*" if dx == 0 and dy == 0 else " "
                if 0 <= px < frame_width and 0 <= py < frame_height:
                    hsv = hsv_frame[py, px]
                    row_values.append(
                        f"{marker}({px},{py})=H{int(hsv[0])} S{int(hsv[1])} V{int(hsv[2])}"
                    )
                else:
                    row_values.append(f"{marker}({px},{py})=OUT")
            print("  " + " | ".join(row_values))
    
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
        """Analyze a clicked ball position and open the HSV tuner windows."""
        x, y = point
        prev_pos = self.ball_center
        was_tracking = self.tracking and prev_pos is not None
        
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
        if self.hsv_lower is not None and self.hsv_upper is not None:
            mask = cv2.inRange(hsv_region, self.hsv_lower, self.hsv_upper)
        else:
            mask = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
        
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
        if was_tracking:
            print(f"\n{'='*70}")
            print(f"=== BALL TRACKING LOST - MANUAL RECOVERY NEEDED ===")
            print(f"{'='*70}")
            print(f"Previous ball position: {prev_pos}")
            print(f"New marked position: ({x}, {y})")
            print(f"Distance jumped: {np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2):.1f}px")
            print(f"\n>>> ANALYZING WHY BALL WAS LOST <<<")
            
            # Check if HSV of new ball is within our filter range
            print(f"\nHSV Analysis:")
            if self.hsv_lower is not None and self.hsv_upper is not None:
                print(f"  Current HSV filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
            else:
                print("  Current HSV filter: not initialized")
            print(f"  Ball HSV at new position: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]}")
            
            # Check each component
            if self.hsv_lower is not None and self.hsv_upper is not None:
                h_in_range = self.hsv_lower[0] <= hsv_values[0] <= self.hsv_upper[0]
                s_in_range = self.hsv_lower[1] <= hsv_values[1] <= self.hsv_upper[1]
                v_in_range = self.hsv_lower[2] <= hsv_values[2] <= self.hsv_upper[2]
            else:
                h_in_range = s_in_range = v_in_range = False
            
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
            
        else:
            print(f"\n=== BALL MARKED AT ({x}, {y}) ===")
        
        print(f"HSV at point: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]}")
        if self.hsv_lower is not None and self.hsv_upper is not None:
            print(f"HSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]} (Tennis ball optimized)")
        else:
            print("HSV Filter: not initialized")
        print(f"Bulb size: {bulb_size} pixels")
        print(f"Initial position saved: ({x}, {y})")
        print("Ball tracking enabled! Press 'Q' or ESC to close the HSV windows, then 'D' to advance to next frame")
        
        # Add to table
        hsv_range = None
        if self.hsv_lower is not None and self.hsv_upper is not None:
            hsv_range = [self.hsv_lower[0], self.hsv_upper[0], self.hsv_lower[1], self.hsv_upper[1], self.hsv_lower[2], self.hsv_upper[2]]
        self.hsv_table.append({
            'frame': self.frame_count,
            'position': (x, y),
            'hsv': hsv_values.tolist(),
            'bulb_size': bulb_size,
            'hsv_range': hsv_range
        })
        
        self._open_interactive_hsv_tuner(frame, point)
        
        return True

    def _get_click_hsv_tuner_specs(self):
        specs = []

        def _append(config_key, label, lower, upper):
            if lower is None or upper is None:
                return
            specs.append({
                'config_key': config_key,
                'label': label,
                'lower': lower.copy(),
                'upper': upper.copy(),
                'bulb_min': 3,
                'bulb_max': 80,
            })

        regular_lower = regular_upper = None
        if self.hsv_regular is not None:
            regular_lower = self.hsv_regular['lower']
            regular_upper = self.hsv_regular['upper']
        elif self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
            regular_lower = self.primary_hsv_lower
            regular_upper = self.primary_hsv_upper
        else:
            regular_lower = self.hsv_lower
            regular_upper = self.hsv_upper

        _append("regular_court", "REGULAR COURT", regular_lower, regular_upper)
        _append("alt1", "ALT1", self.alt_focus_hsv_lower, self.alt_focus_hsv_upper)
        _append("alt2", "ALT2", self.alt2_hsv_lower, self.alt2_hsv_upper)
        _append("alt3", "ALT3", self.alt3_hsv_lower, self.alt3_hsv_upper)
        _append("alt4", "ALT4", self.alt4_hsv_lower, self.alt4_hsv_upper)
        _append("alt5", "ALT5", self.alt5_hsv_lower, self.alt5_hsv_upper)
        _append("alt6", "ALT6", self.alt6_hsv_lower, self.alt6_hsv_upper)
        _append("s_30", "S_30", self.s30_hsv_lower, self.s30_hsv_upper)
        _append("h_10", "H_10", self.h10_hsv_lower, self.h10_hsv_upper)
        _append("alts_20", "ALTS_20", self.alts20_hsv_lower, self.alts20_hsv_upper)
        _append("alts9_11", "ALTS9_11", self.alts9_11_hsv_lower, self.alts9_11_hsv_upper)
        return specs

    def _apply_click_hsv_tuner_updates(self, specs):
        for spec in specs:
            lower = spec['final_lower'].copy()
            upper = spec['final_upper'].copy()
            key = spec['config_key']

            if key == "regular_court":
                if self.hsv_regular is not None:
                    self.hsv_regular['lower'] = lower.copy()
                    self.hsv_regular['upper'] = upper.copy()
                if self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
                    self.primary_hsv_lower = lower.copy()
                    self.primary_hsv_upper = upper.copy()
                if self.pre_focus_hsv_regular is not None:
                    self.pre_focus_hsv_regular = {
                        'lower': lower.copy(),
                        'upper': upper.copy(),
                    }
                    self.pre_focus_hsv_regular['upper'][0] = min(self.pre_focus_hsv_regular['upper'][0], 73)
                if self.alt_hsv_lower is not None and self.alt_hsv_upper is not None:
                    self.alt_hsv_lower = lower.copy()
                    self.alt_hsv_upper = upper.copy()
                    self.alt_hsv_upper[0] = min(self.alt_hsv_upper[0], 73)
                in_net_area = (
                    self.ball_center is not None and
                    hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max') and
                    self._behind_net_enabled() and
                    self.net_area_y_min <= self.ball_center[1] <= self.net_area_y_max
                )
                if (not in_net_area and not self.using_alt_hsv and not self.using_alt2_hsv
                        and not self.using_alt3_hsv and not self.using_alt6_hsv):
                    self.hsv_lower = lower.copy()
                    self.hsv_upper = upper.copy()
            elif key == "alt1":
                self.alt_focus_hsv_lower = lower.copy()
                self.alt_focus_hsv_upper = upper.copy()
                if self.using_alt_hsv:
                    self.hsv_lower = lower.copy()
                    self.hsv_upper = upper.copy()
            elif key == "alt2":
                self.alt2_hsv_lower = lower.copy()
                self.alt2_hsv_upper = upper.copy()
                if self.using_alt2_hsv:
                    self.hsv_lower = lower.copy()
                    self.hsv_upper = upper.copy()
            elif key == "alt3":
                self.alt3_hsv_lower = lower.copy()
                self.alt3_hsv_upper = upper.copy()
                if self.using_alt3_hsv:
                    self.hsv_lower = lower.copy()
                    self.hsv_upper = upper.copy()
            elif key == "alt6":
                self.alt6_hsv_lower = lower.copy()
                self.alt6_hsv_upper = upper.copy()
                if self.using_alt6_hsv:
                    self.hsv_lower = lower.copy()
                    self.hsv_upper = upper.copy()
            elif key == "s_30":
                self.s30_hsv_lower = lower.copy()
                self.s30_hsv_upper = upper.copy()
            elif key == "h_10":
                self.h10_hsv_lower = lower.copy()
                self.h10_hsv_upper = upper.copy()
            elif key == "alts_20":
                self.alts20_hsv_lower = lower.copy()
                self.alts20_hsv_upper = upper.copy()
            elif key == "alts9_11":
                self.alts9_11_hsv_lower = lower.copy()
                self.alts9_11_hsv_upper = upper.copy()
            elif key == "alt4":
                self.alt4_hsv_lower = lower.copy()
                self.alt4_hsv_upper = upper.copy()
            elif key == "alt5":
                self.alt5_hsv_lower = lower.copy()
                self.alt5_hsv_upper = upper.copy()

    def _save_click_hsv_tuner_config(self, specs):
        config_path = self.config_file or 'hsv_config.json'
        try:
            with open(config_path, 'r') as f:
                hsv_config = json.load(f)
        except Exception:
            hsv_config = {}

        for spec in specs:
            key = spec['config_key']
            lower = spec['final_lower']
            upper = spec['final_upper']
            entry = {
                'h_min': int(lower[0]),
                'h_max': int(upper[0]),
                's_min': int(lower[1]),
                's_max': int(upper[1]),
                'v_min': int(lower[2]),
                'v_max': int(upper[2]),
            }
            if key == "regular_court":
                if "regular_court" in hsv_config or self.hsv_regular is not None:
                    hsv_config.setdefault("regular_court", {})
                    hsv_config["regular_court"].update(entry)
                else:
                    hsv_config.update(entry)
            else:
                hsv_config.setdefault(key, {})
                hsv_config[key].update(entry)

        with open(config_path, 'w') as f:
            json.dump(hsv_config, f, indent=4)
        return config_path
    
    def _open_interactive_hsv_tuner(self, frame, search_center):
        """Open interactive HSV filter tuner windows for regular + alt filters."""
        print(f"\n>>> OPENING CLICK HSV FILTER TUNERS <<<")
        print(f"Showing search region around clicked position: {search_center}")

        specs = self._get_click_hsv_tuner_specs()
        if not specs:
            print("No HSV filters available for click tuning.")
            return

        print("Opening windows:")
        for idx, spec in enumerate(specs, start=1):
            print(f"  {idx}. {spec['label']} HSV")
        print("Press 'Q' or ESC to close and continue")

        search_radius = 120
        x_prev, y_prev = search_center
        x1 = max(0, x_prev - search_radius)
        y1 = max(0, y_prev - search_radius)
        x2 = min(frame.shape[1], x_prev + search_radius)
        y2 = min(frame.shape[0], y_prev + search_radius)
        search_frame = frame[y1:y2, x1:x2].copy()

        if search_frame.size == 0:
            print("Error: Cannot extract tuner region")
            return

        window_width = 800
        window_height = 600
        cascade_origin_x = 40
        cascade_origin_y = 40
        cascade_step_x = 55
        cascade_step_y = 40
        cascade_cycle = 8

        for idx, spec in enumerate(specs):
            window_name = f"{idx + 1}. {spec['label']} HSV"
            spec['window_name'] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_width, window_height)
            cascade_idx = idx % cascade_cycle
            window_x = cascade_origin_x + cascade_idx * cascade_step_x
            window_y = cascade_origin_y + cascade_idx * cascade_step_y
            cv2.moveWindow(window_name, window_x, window_y)
            cv2.createTrackbar("H Min", window_name, int(spec['lower'][0]), 179, lambda x: None)
            cv2.createTrackbar("H Max", window_name, int(spec['upper'][0]), 179, lambda x: None)
            cv2.createTrackbar("S Min", window_name, int(spec['lower'][1]), 255, lambda x: None)
            cv2.createTrackbar("S Max", window_name, int(spec['upper'][1]), 255, lambda x: None)
            cv2.createTrackbar("V Min", window_name, int(spec['lower'][2]), 255, lambda x: None)
            cv2.createTrackbar("V Max", window_name, int(spec['upper'][2]), 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", window_name, int(spec['bulb_min']), 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", window_name, int(spec['bulb_max']), 200, lambda x: None)
        cv2.waitKey(1)

        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        while True:
            for spec in specs:
                window_name = spec['window_name']
                h_min = cv2.getTrackbarPos("H Min", window_name)
                h_max = cv2.getTrackbarPos("H Max", window_name)
                s_min = cv2.getTrackbarPos("S Min", window_name)
                s_max = cv2.getTrackbarPos("S Max", window_name)
                v_min = cv2.getTrackbarPos("V Min", window_name)
                v_max = cv2.getTrackbarPos("V Max", window_name)
                bulb_min = cv2.getTrackbarPos("Bulb Min", window_name)
                bulb_max = cv2.getTrackbarPos("Bulb Max", window_name)

                hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
                hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
                raw_mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
                analysis_mask = raw_mask.copy()
                hidden_false_points = self._erase_diagnostic_false_points(analysis_mask, x1, y1)
                contours, _ = cv2.findContours(analysis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                display_frame = search_frame.copy()
                bulb_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < bulb_min or area > bulb_max:
                        continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_frame, (cx, cy), 8, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{area:.0f}", (cx + 12, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    bulb_count += 1

                mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([display_frame, mask_bgr])
                info_text = (
                    f"{spec['label']} | Bulbs: {bulb_count} | "
                    f"H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max}"
                )
                if hidden_false_points:
                    info_text += f" | Hidden false: {hidden_false_points}"
                cv2.putText(combined, info_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_name, combined)

                spec['final_lower'] = hsv_lower
                spec['final_upper'] = hsv_upper
                spec['final_bulb_min'] = bulb_min
                spec['final_bulb_max'] = bulb_max

            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break

        self._apply_click_hsv_tuner_updates(specs)
        config_path = self._save_click_hsv_tuner_config(specs)

        for spec in specs:
            try:
                cv2.destroyWindow(spec['window_name'])
            except Exception:
                pass
        cv2.waitKey(1)

        print("Tuner closed. Updated HSV ranges:")
        for spec in specs:
            lower = spec['final_lower']
            upper = spec['final_upper']
            print(
                f"  {spec['label']}: H={lower[0]}-{upper[0]}, "
                f"S={lower[1]}-{upper[1]}, V={lower[2]}-{upper[2]} | "
                f"Bulb={spec['final_bulb_min']}-{spec['final_bulb_max']}px"
            )
        print(f"HSV values saved to {config_path}\n")
    
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

            motion_reacq_grace_active = (
                getattr(self, '_serve_contact_grace_frames', 0) > 0 or
                getattr(self, '_rally_contact_grace_frames', 0) > 0 or
                getattr(self, '_ground_bounce_grace_frames', 0) > 0 or
                getattr(self, '_post_reacq_frames', 0) > 0 or
                self._serve_direction_search_active()
            )
            if (
                self.stuck_frame_count <= 1 and
                _recent_max_vel < 35.0 and
                self.last_motion is not None and
                not motion_reacq_grace_active
            ):
                last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
                last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
                expected_x = stuck_x + last_dx
                expected_y = stuck_y + last_dy
                predicted_jump = math.hypot(mx - expected_x, my - expected_y)
                max_slow_jump = max(80.0, _recent_max_vel * 6.0)
                max_slow_dist = max(110.0, _recent_max_vel * 9.0)
                if predicted_jump > max_slow_jump and dist_from_stuck > max_slow_dist:
                    print(
                        f"  DEBUG: [REACQ] SKIPPED implausible slow-path jump at ({mx},{my}) "
                        f"dist={dist_from_stuck:.0f}px pred_jump={predicted_jump:.1f}px "
                        f"recent_vel={_recent_max_vel:.1f}"
                    )
                    continue

            if (
                self.stuck_frame_count <= 1 and
                _recent_max_vel < 35 and
                self.last_motion is not None and
                stuck_y > max(120, int(frame_height * 0.06))
            ):
                last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
                last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
                if last_dy < -4.0:
                    expected_x = stuck_x + last_dx
                    expected_y = stuck_y + last_dy
                    predicted_jump = math.hypot(mx - expected_x, my - expected_y)
                    max_slow_jump = max(95.0, _recent_max_vel * 4.2)
                    upward_leap = stuck_y - my
                    max_upward_leap = max(120.0, abs(last_dy) * 5.0)
                    if upward_leap > max_upward_leap and predicted_jump > max_slow_jump:
                        print(
                            f"  DEBUG: [REACQ] SKIPPED implausible upper leap at ({mx},{my}) "
                            f"dy={-upward_leap:.0f}px pred_jump={predicted_jump:.1f}px "
                            f"recent_vel={_recent_max_vel:.1f}"
                        )
                        continue

            # Skip candidates far outside the expected travel range — prevents jumping to
            # unrelated corners or edges that happen to match ball colour.
            if self.last_motion is not None and stuck_y > int(frame_height * 0.35):
                last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
                upward_leap = float(stuck_y - my)
                max_upper_reacq_leap = max(260.0, frame_height * 0.14, last_dy * 5.5)
                if (
                    last_dy >= 18.0 and
                    my < int(frame_height * 0.25) and
                    upward_leap > max_upper_reacq_leap
                ):
                    print(
                        f"  DEBUG: [REACQ] SKIPPED falling-track upper jump at ({mx},{my}) "
                        f"dy={-upward_leap:.0f}px last_dy={last_dy:.1f}px "
                        f"limit={max_upper_reacq_leap:.1f}px"
                    )
                    continue

            if dist_from_stuck > _max_reacq_dist:
                continue

            if contact_bounds is not None:
                if (mx < contact_bounds['min_x'] or mx > contact_bounds['max_x'] or
                        my < contact_bounds['min_y'] or my > contact_bounds['max_y']):
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
        search_frame = frame
        search_radius = None
        x1, y1 = 0, 0
        x2, y2 = frame_width, frame_height
        search_anchor_y = frame_height // 2
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
        top_return_reentry_grace = getattr(self, '_top_return_reentry_grace_frames', 0) > 0
        back_return_reentry_grace = getattr(self, '_back_return_reentry_grace_frames', 0) > 0
        serve_contact_descending = (
            serve_contact_grace and (
                self._is_descending_serve_contact_motion() or
                self._is_descending_serve_contact_motion(self.prev_motion)
            )
        )
        serve_contact_launch = serve_contact_grace and self._is_forward_serve_launch_motion()
        contact_recovery_active = getattr(self, '_contact_recovery_frames', 0) > 0
        top_return_upper_wait = (
            self._top_return_wait_active() and
            getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket')
        )
        contact_reacquire_bounds = self._contact_reacquire_bounds(frame.shape, self.ball_center) if self.ball_center else None
        if top_return_upper_wait:
            contact_reacquire_bounds = None
        upper_contact_turn_commit = (
            not allow_inactive and
            contact_reacquire_bounds is not None and
            self.ball_center is not None and
            getattr(self, 'direction_change_streak', 0) >= 2 and
            self.ground_bounce_count > 0 and
            self.ball_size is not None and
            self.ball_size >= 20 and
            self.ball_size <= 35 and
            self.ball_center[1] >= max(190, int(frame.shape[0] * 0.085)) and
            self.ball_center[1] <= max(260, int(frame.shape[0] * 0.13))
        )
        near_side_large_hit_prep = False
        lower_contact_launch_context = None
        ground_bounce_context = None
        if serve_contact_grace:
            self._serve_contact_grace_frames -= 1
        if rally_contact_grace:
            self._rally_contact_grace_frames -= 1
            if getattr(self, '_rally_contact_large_ref_frames', 0) > 0:
                self._rally_contact_large_ref_frames -= 1
        if ground_bounce_grace:
            self._ground_bounce_grace_frames -= 1
        if top_return_reentry_grace:
            self._top_return_reentry_grace_frames -= 1
        if back_return_reentry_grace:
            self._back_return_reentry_grace_frames -= 1
        if contact_recovery_active:
            self._contact_recovery_frames -= 1

        # Store the last search position for when ball is lost
        if not hasattr(self, 'last_search_position'):
            self.last_search_position = None
        
        early_frames = self.frame_count <= (self.start_frame + 10)

        # Check if ball was near edge in previous frame
        self.edge_wait = False
        upper_exit_wait = getattr(self, '_upper_exit_wait_frames', 0) > 0
        top_return_wait = self._top_return_wait_active()
        back_return_wait = self._back_return_wait_active()
        top_return_expired_this_frame = False
        simple_top_edge_search = False
        custom_search_region = None

        if self.ball_center:
            x_prev, y_prev = self.ball_center
            self.last_search_position = (x_prev, y_prev)  # Remember this position
            edge_margin = 50  # pixels from edge (reduced from 100 to avoid false triggers)
            top_return_trigger_mode = self._should_start_top_return_wait(frame.shape)

            if upper_exit_wait:
                self._upper_exit_wait_frames -= 1
                print(f"\n  DEBUG: [UPPER-EXIT WAIT] holding near top edge from ({x_prev},{y_prev}), "
                      f"frames_left={self._upper_exit_wait_frames}")
                x, y = x_prev, min(40, max(10, y_prev))
                search_radius = max(220, self.max_ball_speed * 2)
                self.edge_wait = True
            elif top_return_wait:
                self._top_return_wait_frames -= 1
                top_return_expired_this_frame = self._top_return_wait_frames <= 0
                anchor_x, anchor_y = self._top_return_anchor
                elapsed = max(0, self.frame_count - self._top_return_origin_frame)
                print(f"\n  DEBUG: [TOP-RETURN WAIT] holding near top edge from ({anchor_x},{anchor_y}), "
                      f"frames_left={self._top_return_wait_frames}")
                x, y, x1_custom, y1_custom, x2_custom, y2_custom = self._build_top_return_search_region(frame.shape)
                custom_search_region = (x1_custom, y1_custom, x2_custom, y2_custom)
                search_radius = max((x2_custom - x1_custom) // 2, (y2_custom - y1_custom) // 2)
                self.edge_wait = True
            elif back_return_wait:
                self._back_return_wait_frames -= 1
                anchor_x, anchor_y = self._back_return_anchor
                elapsed = max(0, self.frame_count - self._back_return_origin_frame)
                print(f"\n  DEBUG: [BACK-RETURN WAIT] holding near back/bottom exit from ({anchor_x},{anchor_y}), "
                      f"frames_left={self._back_return_wait_frames}")
                x, y = anchor_x, anchor_y
                search_radius = 1500
                self.edge_wait = True
            elif self._should_start_back_return_wait(frame.shape):
                self._activate_back_return_wait()
                back_return_wait = True
                elapsed = 0
                anchor_x, anchor_y = self._back_return_anchor
                print(f"\n  DEBUG: Ball near back/bottom exit ({anchor_x},{anchor_y}), may have gone off-screen")
                print(f"  DEBUG: [BACK-RETURN WAIT] activated for delayed back-screen re-entry search")
                x, y = anchor_x, anchor_y
                search_radius = 1500
                self.edge_wait = True
            elif top_return_trigger_mode:
                self._activate_offscreen_return_wait("top", frame.shape, mode=top_return_trigger_mode)
                top_return_wait = True
                elapsed = 0
                anchor_x, anchor_y = self._top_return_anchor
                if top_return_trigger_mode in ("upper_side", "upper_racket"):
                    print(f"\n  DEBUG: Ball likely exited through upper side from ({anchor_x},{anchor_y})")
                    print(f"  DEBUG: [TOP-RETURN WAIT] activated for delayed upper-side re-entry search")
                else:
                    print(f"\n  DEBUG: Ball projected off TOP edge from ({anchor_x},{anchor_y})")
                    print(f"  DEBUG: [TOP-RETURN WAIT] activated for projected top-edge exit")
                x, y, x1_custom, y1_custom, x2_custom, y2_custom = self._build_top_return_search_region(frame.shape)
                custom_search_region = (x1_custom, y1_custom, x2_custom, y2_custom)
                search_radius = max((x2_custom - x1_custom) // 2, (y2_custom - y1_custom) // 2)
                self.edge_wait = True
            elif self._should_start_top_edge_return_wait(frame.shape):
                self._activate_offscreen_return_wait("top", frame.shape, mode="edge_clip")
                top_return_wait = True
                elapsed = 0
                anchor_x, anchor_y = self._top_return_anchor
                print(f"\n  DEBUG: Ball clipped/off TOP edge from ({anchor_x},{anchor_y})")
                print(f"  DEBUG: [TOP-RETURN WAIT] activated for shared top-edge re-entry search")
                x, y, x1_custom, y1_custom, x2_custom, y2_custom = self._build_top_return_search_region(frame.shape)
                custom_search_region = (x1_custom, y1_custom, x2_custom, y2_custom)
                search_radius = max((x2_custom - x1_custom) // 2, (y2_custom - y1_custom) // 2)
                self.edge_wait = True
            # Check if ball went off top edge
            elif y_prev < edge_margin:
                print(f"\n  DEBUG: Ball near TOP edge (y={y_prev}), may have gone off-screen")
                print(f"  DEBUG: Tracking visible top-edge ball before return wait...")
                # Search along top edge while the ball is still visible in-frame.
                x, y = x_prev, 10
                search_radius = 200
                self.edge_wait = True
                simple_top_edge_search = True
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
                if serve_contact_descending:
                    search_radius = max(search_radius, self.max_ball_speed)
                    print(f"  DEBUG: [SERVE-CONTACT] wide search radius={search_radius}px, frames_left={self._serve_contact_grace_frames}")
                if ground_bounce_grace:
                    search_radius = max(search_radius, max(self.max_ball_speed, 110))
                    print(f"  DEBUG: [GROUND-BOUNCE] wide search radius={search_radius}px, frames_left={self._ground_bounce_grace_frames}")
                if not allow_inactive and not serve_direction_search and not rally_contact_grace and not ground_bounce_grace:
                    near_side_large_hit_prep = self._near_side_large_racket_hit_prep_active(frame.shape)
                    if near_side_large_hit_prep:
                        search_radius = max(search_radius, self.max_ball_speed, 360)
                        print(f"  DEBUG: [NEAR-SIDE RACKET PREP] wide search radius={search_radius}px")
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
                                  f"origin={ground_bounce_context['origin']} "
                                  f"expected={ground_bounce_context['expected']} "
                                  f"incoming=({ground_bounce_context.get('incoming_dx', 0.0):.1f},"
                                  f"{ground_bounce_context.get('incoming_dy', 0.0):.1f},"
                                  f"{ground_bounce_context.get('incoming_dist', 0.0):.1f}) "
                                  f"count={self.ground_bounce_count}")
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
                        self._maybe_handle_reacquire_ground_bounce(reacq_pos, frame)
                        self.ball_velocity_history = []
                        self.last_motion = None
                        self.last_direction = None
                        self.direction_change_streak = 0
                        self.stuck_frame_count = 0
                        self._recent_max_ball_size = 0
                        self.ball_center = reacq_pos
                        self._last_motion_reacq_frame = self.frame_count
                        self._last_motion_reacq_pos = reacq_pos
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
        
        if search_radius is not None:
            search_anchor_y = y
        
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
                x1, y1, x2, y2 = self._build_serve_direction_region(
                    x, y, frame.shape, search_radius,
                    descending_contact=(serve_contact_descending and not serve_contact_launch)
                )
            elif custom_search_region is not None:
                x1, y1, x2, y2 = custom_search_region
            else:
                x1 = max(0, x - search_radius)
                y1 = max(0, y - search_radius)
                x2 = min(frame.shape[1], x + search_radius)
                y2 = min(frame.shape[0], y + search_radius)
            
            search_frame = frame[y1:y2, x1:x2]
            # Determine which HSV config will be used
            _, _, hsv_mode_check = self.select_hsv_for_position(search_anchor_y)
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
            if (self.net_area_y_min - net_approach_margin) <= search_anchor_y <= (self.net_area_y_max + net_approach_margin):
                should_check_both = True
        if is_dual_mode and contact_recovery_active:
            should_check_both = True
            print(f"  DEBUG: Contact recovery active - checking BOTH HSV filters")

        top_return_search_context = self._top_return_wait_active() or top_return_expired_this_frame
        back_return_search_context = self._back_return_wait_active()
        upper_wall_search_context = (
            not allow_inactive and
            self.ball_center is not None and
            self.ground_bounce_count > 0 and
            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 20 and
            search_anchor_y < max(145, int(frame.shape[0] * 0.08)) and
            (
                self.stuck_frame_count >= 2 or
                getattr(self, '_upper_exit_wait_frames', 0) > 0
            )
        )
        if should_check_both:
            # Search with BOTH filters and combine results
            hsv_mode = "dual_net"
            print(f"  DEBUG: Ball near net area (Y={search_anchor_y}, net Y={self.net_area_y_min}-{self.net_area_y_max})")
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
            hsv_lower_use, hsv_upper_use, hsv_mode = self.select_hsv_for_position(search_anchor_y)
            if (top_return_search_context or back_return_search_context) and self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
                hsv_lower_use = self.primary_hsv_lower.copy()
                hsv_upper_use = self.primary_hsv_upper.copy()
                hsv_mode = "top_return_regular" if top_return_search_context else "back_return_regular"
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
            if (upper_wall_search_context or top_return_search_context) and self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
                mask_alt4 = cv2.inRange(hsv_frame, self.alt4_hsv_lower, self.alt4_hsv_upper)
                mask_alt4 = cv2.morphologyEx(mask_alt4, cv2.MORPH_OPEN, kernel)
                mask_alt4 = cv2.morphologyEx(mask_alt4, cv2.MORPH_CLOSE, kernel)
            if (upper_wall_search_context or top_return_search_context) and self.alt5_hsv_lower is not None and self.alt5_hsv_upper is not None:
                mask_alt5 = cv2.inRange(hsv_frame, self.alt5_hsv_lower, self.alt5_hsv_upper)
                mask_alt5 = cv2.morphologyEx(mask_alt5, cv2.MORPH_OPEN, kernel)
                mask_alt5 = cv2.morphologyEx(mask_alt5, cv2.MORPH_CLOSE, kernel)
            if (upper_wall_search_context or top_return_search_context) and self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
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
            if search_radius is not None:
                print(f"  DEBUG:   - Moved faster than {search_radius}px/frame")
            else:
                print(f"  DEBUG:   - Fallen outside the current full-frame HSV search")
            print(f"  DEBUG:   - Be occluded by player/net")
            print(f"  DEBUG: Will continue searching in next frame at same position...")

            if back_return_search_context and self.ball_center:
                self.stuck_frame_count = min(self.stuck_frame_count, 4)
                print(f"[BALL_LOST] f{self.frame_count}: back-return wait holding pos={self.ball_center} stuck={self.stuck_frame_count}")
                return self.ball_center

            if upper_exit_wait and self.ball_center:
                self.stuck_frame_count += 1
                print(f"Frame {self.frame_count}: [UPPER-EXIT WAIT] no top-edge reentry, holding {self.ball_center}")
                return self.ball_center
            if top_return_search_context and self.ball_center:
                self.stuck_frame_count = min(self.stuck_frame_count, 4)
                print(f"Frame {self.frame_count}: [TOP-RETURN WAIT] no top-line reentry, holding {self.ball_center}")
                return self.ball_center

            if (not allow_inactive and self.ball_center and not getattr(self, 'edge_wait', False)
                    and self.stuck_frame_count < 5
                    and hasattr(self, '_prev_frame_gray') and self._prev_frame_gray is not None):
                saved_prev_gray = self._prev_frame_gray.copy()
                prev_reacq_center = self.ball_center
                reacq_pos = self._reacquire_ball_by_motion(frame)
                if reacq_pos is not None:
                    self.ball_center = reacq_pos
                    self.ball_size = None
                    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    self.ball_hsv = hsv_full[reacq_pos[1], reacq_pos[0]]
                    self.prev_motion = self.last_motion
                    self._update_recovered_motion(prev_reacq_center, reacq_pos)
                    self.direction_change_streak = 0
                    self.stuck_frame_count = 0
                    self._recent_max_ball_size = 0
                    self._post_reacq_frames = max(getattr(self, '_post_reacq_frames', 0), 3)
                    self._last_motion_reacq_frame = self.frame_count
                    self._last_motion_reacq_pos = reacq_pos
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
                upper_local_h10_retry = (
                    self.ball_center is not None and
                    search_radius is not None and search_radius <= 100 and
                    self.ball_center[1] <= 210 and
                    (predicted_point is None or predicted_point[1] <= 214) and
                    (self.ball_size is None or self.ball_size <= 8)
                )
                if (upper_local_h10_retry and
                        self.h10_hsv_lower is not None and self.h10_hsv_upper is not None and
                        self._should_try_h10_recover(frame, predicted_point, allow_inactive)):
                    preview_s30 = None
                    if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
                        preview_s30 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.s30_hsv_lower, upper=self.s30_hsv_upper, frame_gray=frame_gray,
                            filter_key="s_30", ignore_false_points=True, prefer_predicted_path=True
                        )
                    retrack_h10 = self.retrack_with_alt2_hsv(
                        search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.h10_hsv_lower, upper=self.h10_hsv_upper, frame_gray=frame_gray,
                            filter_key="h_10", ignore_false_points=True, prefer_predicted_path=True
                        )
                    local_cap = min(42.0, max(26.0, search_radius * 0.55))
                    early_local_candidates = []
                    for label, candidate, ignore_fp in (
                        ("s_30", preview_s30, False),
                        ("h_10", retrack_h10, True),
                    ):
                        if candidate is None:
                            print(
                                f"  DEBUG: [UPPER LOCAL PREVIEW] {label} candidate=None "
                                f"ignore_false_points={ignore_fp}"
                            )
                            continue
                        candidate_pos = candidate['pos']
                        prev_distance = math.hypot(
                            candidate_pos[0] - self.ball_center[0],
                            candidate_pos[1] - self.ball_center[1]
                        )
                        candidate_pred_distance = (
                            math.hypot(candidate_pos[0] - predicted_point[0], candidate_pos[1] - predicted_point[1])
                            if predicted_point is not None else prev_distance
                        )
                        print(
                            f"  DEBUG: [UPPER LOCAL PREVIEW] {label} pos={candidate_pos} "
                            f"area={candidate['area']:.1f}px score={candidate['score']:.1f} "
                            f"prev_dist={prev_distance:.1f}px pred_dist={candidate_pred_distance:.1f}px "
                            f"motion={candidate['motion_mean']:.1f}/{candidate['motion_max']:.1f} "
                            f"local_cap={local_cap:.1f} ignore_false_points={ignore_fp}"
                        )
                        if prev_distance <= local_cap and candidate_pred_distance <= (local_cap + 10.0):
                            trajectory_priority = (
                                candidate_pred_distance <= max(12.0, local_cap * 0.45) and
                                (candidate['motion_mean'] >= 4.0 or candidate['motion_max'] >= 18.0 or candidate['area'] >= 3.0)
                            )
                            early_local_candidates.append({
                                'label': label,
                                'candidate': candidate,
                                'prev_distance': prev_distance,
                                'pred_distance': candidate_pred_distance,
                                'trajectory_priority': trajectory_priority,
                            })
                    if early_local_candidates:
                        best_early_local = min(
                            early_local_candidates,
                            key=lambda item: (
                                0 if item['trajectory_priority'] else 1,
                                item['pred_distance'] if item['trajectory_priority'] else item['candidate']['score'],
                                -item['candidate']['motion_max'],
                                -item['candidate']['motion_mean'],
                                item['candidate']['score'],
                                item['prev_distance'],
                                item['pred_distance'],
                            )
                        )
                        best_label = best_early_local['label']
                        best_candidate = best_early_local['candidate']
                        best_pos = best_candidate['pos']
                        self.ball_center = best_pos
                        self.ball_hsv = best_candidate['hsv']
                        self.ball_size = best_candidate['area']
                        self._activate_regular_hsv()
                        self.stuck_frame_count = 0
                        print(f"Frame {self.frame_count}: [{best_label.upper()} EARLY LOCAL RECOVER] Ball at {best_pos}")
                        return self.ball_center
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

            if (not allow_inactive and self.ball_center is not None
                    and contact_reacquire_bounds is not None
                    and self.ball_center[1] <= 260
                    and (self.ball_size is None or self.ball_size <= 35)
                    and self.stuck_frame_count >= 2):
                regular_single = self._find_single_standard_candidate(
                    search_frame, x1, y1, self.ball_center, predicted_point, frame_gray
                )
                regular_search_radius = search_radius if search_radius is not None else max(x2 - x1, y2 - y1) / 2.0
                if self._should_use_single_regular_candidate(
                        regular_single, predicted_point, regular_search_radius):
                    new_pos = regular_single['pos']
                    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    self.ball_center = new_pos
                    self.ball_hsv = hsv_full[new_pos[1], new_pos[0]]
                    self.ball_size = regular_single['area']
                    self._activate_regular_hsv()
                    print(f"Frame {self.frame_count}: [REGULAR RECOVER] Ball at {new_pos}")
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
                    frame_gray=frame_gray, filter_key="alt1"
                )
                if retrack is not None:
                    new_pos = retrack['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack['hsv']
                    self.ball_size = retrack['area']
                    self.using_alt_hsv = True
                    self.using_alt2_hsv = False
                    self.using_alt3_hsv = False
                    self.using_alt4_hsv = False
                    self.using_alt6_hsv = False
                    if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                        self.hsv_lower = self.alt_focus_hsv_lower
                        self.hsv_upper = self.alt_focus_hsv_upper
                    print(f"Frame {self.frame_count}: [ALT HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center
            # Fallback: try alternative 2 HSV if alt1 failed
            if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
                retrack2 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    frame_gray=frame_gray, filter_key="alt2"
                )
                if retrack2 is not None:
                    if (not allow_inactive and self.ball_center is not None and
                            self.stuck_frame_count < 2):
                        alt2_jump = math.hypot(
                            retrack2['pos'][0] - self.ball_center[0],
                            retrack2['pos'][1] - self.ball_center[1],
                        )
                        if alt2_jump > 100.0:
                            print(
                                f"  DEBUG: Rejecting alt2 recover at {retrack2['pos']} - "
                                f"early recovery jump {alt2_jump:.1f}px > 100.0px"
                            )
                            retrack2 = None
                    if retrack2 is None:
                        pass
                    else:
                        new_pos = retrack2['pos']
                        prev_pos = self.ball_center
                        self.ball_center = new_pos
                        self.ball_hsv = retrack2['hsv']
                        self.ball_size = retrack2['area']
                        self.using_alt_hsv = False
                        self.using_alt2_hsv = True
                        self.using_alt3_hsv = False
                        self.using_alt4_hsv = False
                        self.using_alt6_hsv = False
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
                                    lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray,
                                    filter_key="alt3"
                                )
                                if retrack3 is not None:
                                    new_pos = retrack3['pos']
                                    self.ball_center = new_pos
                                    self.ball_hsv = retrack3['hsv']
                                    self.ball_size = retrack3['area']
                                    self.using_alt2_hsv = False
                                    self.using_alt3_hsv = True
                                    self.using_alt6_hsv = False
                                    self.hsv_lower = self.alt3_hsv_lower
                                    self.hsv_upper = self.alt3_hsv_upper
                                    print(f"Frame {self.frame_count}: [ALT3 HSV RECOVER] Ball at {new_pos}")
                        return self.ball_center
            if self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
                retrack4 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    lower=self.alt4_hsv_lower, upper=self.alt4_hsv_upper, frame_gray=frame_gray,
                    filter_key="alt4"
                )
                if retrack4 is not None:
                    new_pos = retrack4['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack4['hsv']
                    self.ball_size = retrack4['area']
                    self.using_alt_hsv = False
                    self.using_alt2_hsv = False
                    self.using_alt3_hsv = False
                    self.using_alt4_hsv = True
                    self.using_alt6_hsv = False
                    self.hsv_lower = self.alt4_hsv_lower
                    self.hsv_upper = self.alt4_hsv_upper
                    print(f"Frame {self.frame_count}: [ALT4 HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center
            if self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
                retrack6 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    lower=self.alt6_hsv_lower, upper=self.alt6_hsv_upper, frame_gray=frame_gray,
                    filter_key="alt6"
                )
                if retrack6 is not None:
                    new_pos = retrack6['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack6['hsv']
                    self.ball_size = retrack6['area']
                    self.using_alt_hsv = False
                    self.using_alt2_hsv = False
                    self.using_alt3_hsv = False
                    self.using_alt4_hsv = False
                    self.using_alt6_hsv = True
                    self.hsv_lower = self.alt6_hsv_lower
                    self.hsv_upper = self.alt6_hsv_upper
                    print(f"Frame {self.frame_count}: [ALT6 HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center
            if self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None:
                retrack3 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray,
                    filter_key="alt3"
                )
                if retrack3 is not None:
                    new_pos = retrack3['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack3['hsv']
                    self.ball_size = retrack3['area']
                    self.using_alt_hsv = False
                    self.using_alt2_hsv = False
                    self.using_alt3_hsv = True
                    self.using_alt6_hsv = False
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
                    self.using_alt6_hsv = False
                    self.hsv_lower = upper_exit_low_s['lower']
                    self.hsv_upper = upper_exit_low_s['upper']
                    self.stuck_frame_count = 0
                    print(f"Frame {self.frame_count}: [UPPER EXIT LOW-S RECOVER] Ball at {new_pos} via {upper_exit_low_s['label']}")
                    return self.ball_center

            if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
                retrack_s30 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    lower=self.s30_hsv_lower, upper=self.s30_hsv_upper, frame_gray=frame_gray,
                    filter_key="s_30"
                )
                if retrack_s30 is not None:
                    bottom_large_exit_false_s30 = (
                        not allow_inactive and
                        self.ball_center is not None and
                        self.ball_size is not None and
                        self.ball_size >= 700 and
                        self.ball_center[1] > frame.shape[0] - max(70, int(frame.shape[0] * 0.04)) and
                        retrack_s30['area'] < max(220, self.ball_size * 0.22) and
                        retrack_s30.get('motion_mean', 0.0) < 2.0 and
                        retrack_s30.get('motion_max', 0.0) < 10.0
                    )
                    recent_back_return_false_s30 = (
                        not allow_inactive and
                        (
                            self._back_return_wait_active() or
                            getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
                            self._recent_offscreen_return_hold_active(window_frames=24)
                        ) and
                        retrack_s30.get('motion_mean', 0.0) < 2.0 and
                        retrack_s30.get('motion_max', 0.0) < 10.0
                    )
                    if bottom_large_exit_false_s30 or recent_back_return_false_s30:
                        print(
                            f"  DEBUG: Rejecting s_30 recover at {retrack_s30['pos']} - "
                            f"back-screen return candidate is low-motion/static"
                        )
                    else:
                        new_pos = retrack_s30['pos']
                        self.ball_center = new_pos
                        self.ball_hsv = retrack_s30['hsv']
                        self.ball_size = retrack_s30['area']
                        self._activate_regular_hsv()
                        self.stuck_frame_count = 0
                        print(f"Frame {self.frame_count}: [S_30 HSV RECOVER] Ball at {new_pos}")
                        return self.ball_center

            if (self.h10_hsv_lower is not None and self.h10_hsv_upper is not None and
                    self._should_try_h10_recover(frame, predicted_point, allow_inactive)):
                retrack_h10 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                    lower=self.h10_hsv_lower, upper=self.h10_hsv_upper, frame_gray=frame_gray,
                    filter_key="h_10"
                )
                if retrack_h10 is not None:
                    new_pos = retrack_h10['pos']
                    self.ball_center = new_pos
                    self.ball_hsv = retrack_h10['hsv']
                    self.ball_size = retrack_h10['area']
                    self._activate_regular_hsv()
                    self.stuck_frame_count = 0
                    print(f"Frame {self.frame_count}: [H_10 HSV RECOVER] Ball at {new_pos}")
                    return self.ball_center

            lower_right_exit_reason = None if allow_inactive else self._lower_right_edge_exit_out_reason(frame.shape)
            if lower_right_exit_reason:
                self._pending_rally_end_reason = lower_right_exit_reason
                self._pending_rally_end_frame = self.frame_count
                print(f"Frame {self.frame_count}: [LOWER-RIGHT EXIT OUT] {lower_right_exit_reason} at {self.ball_center}")
                return self.ball_center

            if back_return_search_context and self.ball_center:
                self.stuck_frame_count = min(max(self.stuck_frame_count + 1, 1), 4)
                print(f"[BALL_LOST] f{self.frame_count}: back-return wait holding pos={self.ball_center} stuck={self.stuck_frame_count}")
                return self.ball_center
            if (
                not allow_inactive and self.ball_center and
                (
                    getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
                    self._recent_offscreen_return_hold_active(window_frames=24)
                )
            ):
                self.stuck_frame_count = min(max(self.stuck_frame_count + 1, 1), 4)
                print(f"[BALL_LOST] f{self.frame_count}: back-return re-entry grace holding pos={self.ball_center} stuck={self.stuck_frame_count}")
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
            if top_return_search_context:
                self.stuck_frame_count = min(self.stuck_frame_count, 4)
                print(f"[BALL_LOST] f{self.frame_count}: top-return wait holding pos={self.ball_center} stuck={self.stuck_frame_count}")
                return self.ball_center
            self.stuck_frame_count += 1
            return self.ball_center
        
        # Find the best ball candidate using weighted score (distance + size similarity)
        best_contour = None
        best_score = float('inf')
        best_source = None
        large_lower_launch_override = False
        near_camera_large_turn_override = False
        upper_far_player_escape_override = False
        upper_dynamic_departure_override = False
        large_lower_bounce_launch_override = False
        candidates = []
        candidate_meta = []
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
        large_ball_tracking_cap = 0
        large_ball_bg_threshold = 0
        if not allow_inactive and self.ball_size is not None and self.ball_size >= 200:
            if self.ball_size >= 700:
                large_ball_tracking_cap = min(
                    1800,
                    int(max(self.ball_size * 1.70, self.ball_size + 450))
                )
                large_ball_bg_threshold = int(max(self.ball_size * 1.65, self.ball_size + 450))
            else:
                large_ball_tracking_cap = min(
                    1100,
                    int(max(self.ball_size * 1.45, self.ball_size + 120))
                )
        ball_size_max_tracking = max(150, self.serve_ball_size_max, large_ball_tracking_cap)
        bg_threshold = max(500, int(self.serve_ball_size_max * 1.5), large_ball_bg_threshold)
        for i, (source, contour) in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Reject huge background regions first.
            # Upper limit scales with serve_ball_size_max so near-end large balls aren't
            # incorrectly classified as background (e.g. court 2 ball area ~280-360px²).
            if area > bg_threshold:
                if i < 3:  # Only print first few to avoid spam
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (background region)")
                continue

            # Size filter: tighter in inactive serve scan, looser when tracking.
            # Both limits are court-configurable via serve_ball_size_min/max so that
            # near-end servers (large ball) and far-end servers (tiny ball) both work.
            # When a close ball is already large, keep expanding the cap with it so
            # the next frame is not dropped by the old fixed 800/1100px ceilings.
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
                    not top_return_search_context and not top_return_reentry_grace and
                    not back_return_search_context and not back_return_reentry_grace and
                    self.ball_size and self.ball_size > 40 and
                    area < self._min_area_for_previous_ball_size(self.ball_size, self.ball_center, frame.shape)):
                min_area = self._min_area_for_previous_ball_size(self.ball_size, self.ball_center, frame.shape)
                print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px "
                      f"(too small relative to previous ball size {self.ball_size:.1f}px, min_area={min_area})")
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
            candidate_filter_key = self._candidate_false_point_filter_key(source, hsv_mode)
            
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
            predicted_distance = None
            
            # Calculate size difference (penalty for very different sizes)
            if self.ball_size and self.ball_size > 0:
                size_diff = abs(area - self.ball_size)
                size_ratio = size_diff / self.ball_size
            else:
                size_ratio = 0

            if serve_direction_search and self.ball_center:
                serve_direction_min_dx = self.serve_contact_min_dx
                serve_direction_min_dy = self.serve_contact_min_dy
                if serve_contact_descending and not serve_contact_launch:
                    serve_direction_min_dx = 0
                    serve_direction_min_dy = min(
                        serve_direction_min_dy,
                        -max(60, int(frame_height * 0.03))
                    )
                signed_dx = self._signed_serve_dx(cx - self.ball_center[0])
                if signed_dx < serve_direction_min_dx:
                    print(f"  DEBUG: Contour {i} REJECTED - serve-direction dx={signed_dx:.1f}px < {serve_direction_min_dx}")
                    continue
                if self.serve_direction_dy != 0:
                    signed_dy = self._signed_serve_dy(cy - self.ball_center[1])
                    if signed_dy < serve_direction_min_dy:
                        print(f"  DEBUG: Contour {i} REJECTED - serve-direction dy={signed_dy:.1f}px < {serve_direction_min_dy}")
                        continue

            pre_ignore_motion_metrics = None
            predicted_path_hotspot_override = False
            predicted_path_hotspot_debug = None
            if top_return_search_context and frame_gray is not None:
                pre_ignore_motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)

            if not allow_inactive:
                ignored_entry = self._find_ignored_tracking_position((cx, cy), filter_key=candidate_filter_key)
                if ignored_entry is not None:
                    top_return_motion_override = False
                    upper_post_miss_rebound_override = False
                    predicted_path_hotspot_override, predicted_path_hotspot_debug = (
                        self._predicted_path_hotspot_override(
                            (cx, cy), area, self.ball_center, predicted_point, frame_gray
                        )
                    )
                    if (
                        top_return_search_context and
                        getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
                        source in ('primary', 'regular', 'alt') and
                        pre_ignore_motion_metrics is not None
                    ):
                        motion_mean_for_override = pre_ignore_motion_metrics['mean']
                        motion_max_for_override = pre_ignore_motion_metrics['max']
                        top_return_ok, _ = self._top_return_reentry_ok(
                            (cx, cy),
                            area,
                            motion_mean_for_override,
                            motion_max_for_override,
                            frame.shape,
                        )
                        top_return_motion_override = (
                            top_return_ok and
                            motion_mean_for_override >= 8.0 and
                            motion_max_for_override >= 60.0
                        )
                    if (
                            rally_contact_grace and
                            self.ball_center is not None and
                            getattr(self, '_rally_contact_origin', None) is not None and
                            (self.frame_count - getattr(self, '_last_racket_contact_frame', -1000000)) <= 3):
                        origin_x, origin_y = self._rally_contact_origin
                        if origin_y <= max(360, int(frame_height * 0.18)):
                            rebound_motion = self._candidate_motion_metrics(frame_gray, cx, cy)
                            if rebound_motion is not None:
                                upper_post_miss_rebound_override = (
                                    cy <= self.ball_center[1] - 8 and
                                    abs(cx - self.ball_center[0]) <= max(48, int(frame_width * 0.014)) and
                                    area >= 1.0 and
                                    (rebound_motion['max'] >= 55.0 or rebound_motion['mean'] >= 8.0)
                                )
                                if upper_post_miss_rebound_override:
                                    pre_ignore_motion_metrics = rebound_motion
                    if (not top_return_motion_override and not predicted_path_hotspot_override and
                            not upper_post_miss_rebound_override):
                        print(f"  DEBUG: Contour {i} REJECTED - learned hotspot at ({cx},{cy}) "
                              f"reason={ignored_entry.get('reason', 'n/a')}")
                        continue
                    if top_return_motion_override:
                        print(
                            f"  DEBUG: Contour {i} ALLOWED - top-return motion overrides learned hotspot "
                            f"at ({cx},{cy}) motion={pre_ignore_motion_metrics['mean']:.1f}/"
                            f"{pre_ignore_motion_metrics['max']:.1f}"
                        )
                    elif upper_post_miss_rebound_override:
                        print(
                            f"  DEBUG: Contour {i} ALLOWED - upper post-miss rebound overrides learned hotspot "
                            f"at ({cx},{cy}) motion={pre_ignore_motion_metrics['mean']:.1f}/"
                            f"{pre_ignore_motion_metrics['max']:.1f} "
                            f"reason={ignored_entry.get('reason', 'n/a')}"
                        )
                    else:
                        print(
                            f"  DEBUG: Contour {i} ALLOWED - predicted path overrides learned hotspot "
                            f"at ({cx},{cy}) pred_dist={predicted_path_hotspot_debug['predicted_distance']:.1f}/"
                            f"{predicted_path_hotspot_debug['predicted_cap']:.1f} "
                            f"prev_dist={predicted_path_hotspot_debug['prev_distance']:.1f}/"
                            f"{predicted_path_hotspot_debug['prev_cap']:.1f} motion="
                            f"{predicted_path_hotspot_debug['motion_mean']:.1f}/"
                            f"{predicted_path_hotspot_debug['motion_max']:.1f} "
                            f"reason={ignored_entry.get('reason', 'n/a')}"
                        )
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
            elif serve_contact_grace:
                # During serve contact the ball can jump hundreds of pixels and reverse
                # direction instantly, so distance to the previous toss location is a
                # weak signal. Prefer contours that still look like the same ball size.
                _contact_ref_size = max(self.ball_size or area, 50)
                _contact_size_ratio = abs(area - _contact_ref_size) / _contact_ref_size
                score = _contact_size_ratio * 100 + distance * 0.1
            elif upper_contact_turn_commit:
                # On the last held upper-contact frame, prefer the launched ball that
                # clears upward/lateral space away from the contact blob.
                upward_progress = self.ball_center[1] - cy if self.ball_center is not None else 0.0
                lateral_shift = abs(cx - self.ball_center[0]) if self.ball_center is not None else 0.0
                _upper_turn_ref_size = max(8.0, min(float(self.ball_size or area), 18.0))
                _upper_turn_size_ratio = abs(area - _upper_turn_ref_size) / max(_upper_turn_ref_size, 1.0)
                score = _upper_turn_size_ratio * 35.0 + distance * 0.45
                if area < 4.0:
                    score += (4.0 - area) * 12.0
                if upward_progress < 18.0:
                    score += (18.0 - upward_progress) * 8.0
                else:
                    score -= min(85.0, upward_progress * 2.6)
                if lateral_shift < 45.0:
                    score += (45.0 - lateral_shift) * 2.5
                if area > max(_upper_turn_ref_size * 1.4, 24.0) and upward_progress < 24.0:
                    score += (area - _upper_turn_ref_size) * 2.0
                if predicted_point is not None:
                    _upper_turn_pred_dist = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    score += _upper_turn_pred_dist * 0.08
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
                    high_launch_active = (
                        self.frame_count <= getattr(self, '_rally_contact_high_launch_until_frame', -1) and
                        origin_y >= int(frame_height * 0.50) and
                        getattr(self, '_rally_contact_expected', None) is not None
                    )
                    if high_launch_active:
                        expected_x, expected_y = self._rally_contact_expected
                        high_clearance = expected_y - cy
                        radial_gain = radial_dist - getattr(self, '_rally_contact_progress', 0.0)
                        high_launch_motion = (
                            self._candidate_motion_metrics(frame_gray, cx, cy)
                            if frame_gray is not None else None
                        )
                        strong_motion = (
                            high_launch_motion is not None and
                            (high_launch_motion['max'] >= 90.0 or high_launch_motion['mean'] >= 24.0)
                        )
                        ball_sized = area >= max(18.0, _rally_ref_size * 0.30)
                        if (
                                high_clearance >= max(95.0, frame_height * 0.045) and
                                radial_gain >= max(55.0, frame_height * 0.025) and
                                strong_motion and
                                ball_sized):
                            score -= min(240.0, high_clearance * 0.9)
                            print(
                                f"  DEBUG: [RALLY-CONTACT HIGH-LAUNCH] bonus for ({cx},{cy}) "
                                f"clearance={high_clearance:.1f}px radial_gain={radial_gain:.1f}px"
                            )
                        elif cy >= expected_y - max(55.0, frame_height * 0.025):
                            score += 90.0
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
            motion_mean = 0.0
            motion_max = 0.0
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
            if (not allow_inactive and contact_reacquire_bounds is not None and
                    not top_return_search_context and not back_return_search_context):
                if upper_exit_transition_context:
                    if (cx < contact_reacquire_bounds['min_x'] or
                            cx > contact_reacquire_bounds['max_x'] or
                            cy > contact_reacquire_bounds['max_y']):
                        print(f"  DEBUG: Contour {i} REJECTED - upper-exit bounds "
                              f"x={contact_reacquire_bounds['min_x']}-{contact_reacquire_bounds['max_x']} "
                              f"max_y={contact_reacquire_bounds['max_y']}")
                        continue
                else:
                    relaxed_min_y = contact_reacquire_bounds['min_y']
                    if motion_metrics is not None:
                        relaxed_min_y = self._contact_reacquire_min_y(
                            contact_reacquire_bounds,
                            predicted_distance,
                            motion_metrics['mean'],
                            motion_metrics['max'],
                        )
                    if (cx < contact_reacquire_bounds['min_x'] or cx > contact_reacquire_bounds['max_x'] or
                            cy < relaxed_min_y):
                        print(f"  DEBUG: Contour {i} REJECTED - upper-contact bounds "
                              f"x={contact_reacquire_bounds['min_x']}-{contact_reacquire_bounds['max_x']} "
                              f"min_y={relaxed_min_y}")
                        continue
            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy)) if not allow_inactive else None
            if not allow_inactive and motion_metrics is not None:
                motion_mean = motion_metrics['mean']
                motion_max = motion_metrics['max']
                motion_note = f", motion_mean={motion_mean:.1f}, motion_max={motion_max:.1f}"
                if upper_exit_transition_context:
                    if motion_max >= 40.0:
                        score -= min(55.0, motion_max * 0.3)
                    elif motion_mean < 1.5 and motion_max < 12.0:
                        score += 220.0
                    elif motion_mean < 3.0 and motion_max < 20.0:
                        score += 100.0
                if (frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0 and
                        not predicted_path_hotspot_override):
                    score += 1800
                    print(f"  DEBUG: Contour {i} PENALIZED - frame0 hotspot at ({cx},{cy}) "
                          f"motion_mean={motion_mean:.1f} motion_max={motion_max:.1f}")
                static_hotspot = ((area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                if static_hotspot and not predicted_path_hotspot_override:
                    score += 1200
                if candidate_filter_key is not None:
                    steady_reason = None
                    if predicted_path_hotspot_override:
                        steady_reason = None
                    elif frame0_hotspot is not None and motion_mean < 8.0 and motion_max < 35.0:
                        steady_reason = "frame0-backed steady false point"
                    elif area <= 18.0 and motion_mean < 3.5 and motion_max < 12.0:
                        steady_reason = "steady repeated false point"
                    if steady_reason is not None and self._observe_steady_false_point(
                            (cx, cy), candidate_filter_key, area, motion_mean, motion_max, steady_reason):
                        print(f"  DEBUG: Contour {i} REJECTED - persistent false point at ({cx},{cy}) "
                              f"filter={candidate_filter_key}")
                        continue
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
            predicted_distance_meta = (
                math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                if predicted_point is not None else None
            )
            candidate_meta.append({
                'index': i,
                'source': source,
                'filter_key': candidate_filter_key,
                'pos': (cx, cy),
                'area': area,
                'distance': distance,
                'score': score,
                'motion_mean': motion_mean,
                'motion_max': motion_max,
                'predicted_distance': predicted_distance_meta,
                'predicted_path_hotspot_override': predicted_path_hotspot_override,
                'predicted_path_hotspot_debug': predicted_path_hotspot_debug,
                'contour': contour,
            })
            source_label = f"[{source}]" if source != 'single' else ""
            print(f"  DEBUG: Contour {i} {source_label} CANDIDATE - pos=({cx},{cy}), area={area:.1f}px, distance={distance:.1f}px, size_ratio={size_ratio:.2f}, score={score:.1f}{motion_note}")
            
            if score < best_score:
                best_score = score
                best_contour = contour
                best_source = source

        # Always persist candidates for C-key contour debug overlay (lightweight list).
        self._debug_contour_candidates = list(candidates)

        # During delayed top-return wait, the normal distance-heavy scorer tends to
        # cling to familiar tiny top-line specks near the exit point. If we already
        # have a candidate that looks like a real re-entry into the top band,
        # promote it before the generic selection/guard logic runs.
        if top_return_search_context and candidate_meta:
            top_return_candidates = []
            for meta in candidate_meta:
                if meta['source'] not in ('primary', 'regular', 'alt'):
                    continue
                ok, _ = self._top_return_reentry_ok(
                    meta['pos'],
                    meta['area'],
                    meta['motion_mean'],
                    meta['motion_max'],
                    frame.shape,
                )
                if not ok:
                    continue

                source = meta['source']
                source_bias = 0.0
                if source in ('primary', 'regular'):
                    source_bias -= 80.0
                elif source == 'alt':
                    source_bias -= 35.0
                elif source == 'behind_net':
                    source_bias += 40.0
                elif source == 'alt2':
                    source_bias += 80.0
                elif source == 'alt4':
                    source_bias += 120.0
                elif source == 'alt5':
                    source_bias += 160.0
                elif source == 'alt6':
                    source_bias += 220.0

                adjusted_score = (
                    meta['score'] +
                    source_bias -
                    min(900.0, meta['area'] * 6.0) -
                    min(220.0, meta['motion_max'] * 1.2) -
                    min(120.0, meta['motion_mean'] * 3.0)
                )
                top_return_candidates.append((adjusted_score, meta))

            if top_return_candidates:
                _, top_meta = min(top_return_candidates, key=lambda item: item[0])
                best_contour = top_meta['contour']
                best_source = top_meta['source']
                best_score = top_meta['score']
                print(
                    f"  DEBUG: [TOP-RETURN WAIT] prioritizing re-entry candidate at "
                    f"{top_meta['pos']} area={top_meta['area']:.1f}px "
                    f"score={top_meta['score']:.1f} motion="
                    f"{top_meta['motion_mean']:.1f}/{top_meta['motion_max']:.1f} "
                    f"source={top_meta['source']}"
                )

        if back_return_search_context and candidate_meta:
            back_return_candidates = []
            for meta in candidate_meta:
                if meta['source'] not in ('primary', 'regular', 'alt'):
                    continue
                ok, _ = self._back_return_reentry_ok(
                    meta['pos'],
                    meta['area'],
                    meta['motion_mean'],
                    meta['motion_max'],
                    frame.shape,
                )
                if not ok:
                    continue
                source_bias = -80.0 if meta['source'] in ('primary', 'regular') else -35.0
                adjusted_score = (
                    meta['score'] +
                    source_bias -
                    min(1100.0, meta['area'] * 4.0) -
                    min(220.0, meta['motion_max'] * 1.3) -
                    min(120.0, meta['motion_mean'] * 3.5)
                )
                back_return_candidates.append((adjusted_score, meta))

            if back_return_candidates:
                _, back_meta = min(back_return_candidates, key=lambda item: item[0])
                best_contour = back_meta['contour']
                best_source = back_meta['source']
                best_score = back_meta['score']
                print(
                    f"  DEBUG: [BACK-RETURN WAIT] prioritizing re-entry candidate at "
                    f"{back_meta['pos']} area={back_meta['area']:.1f}px "
                    f"score={back_meta['score']:.1f} motion="
                    f"{back_meta['motion_mean']:.1f}/{back_meta['motion_max']:.1f} "
                    f"source={back_meta['source']}"
                )

        if top_return_reentry_grace and candidate_meta and self.ball_center is not None:
            downward_return_meta = self._prefer_top_return_downward_continuation(
                candidate_meta, frame.shape
            )
            if downward_return_meta is not None:
                best_contour = downward_return_meta['contour']
                best_source = downward_return_meta['source']
                best_score = downward_return_meta['score']
                print(
                    f"  DEBUG: [TOP-RETURN DOWNWARD] prioritizing near vertical return at "
                    f"{downward_return_meta['pos']} area={downward_return_meta['area']:.1f}px "
                    f"score={downward_return_meta['score']:.1f} motion="
                    f"{downward_return_meta['motion_mean']:.1f}/{downward_return_meta['motion_max']:.1f} "
                    f"source={downward_return_meta['source']}"
                )
                continuing_candidates = []
            else:
                continuing_candidates = None
            motion = self.last_motion
            if motion is None or motion.get('distance', 0.0) <= 0:
                motion = getattr(self, 'last_nonzero_motion', None)
            if downward_return_meta is None and motion is not None:
                prev_x, prev_y = self.ball_center
                expected_x = prev_x + float(motion.get('dx', 0.0) or 0.0)
                expected_y = prev_y + float(motion.get('dy', 0.0) or 0.0)
                continuing_candidates = []
                for meta in candidate_meta:
                    cx_meta, cy_meta = meta['pos']
                    visible_ball = meta['area'] >= 4.0
                    moving_ball = (
                        meta['motion_max'] >= 70.0 or
                        (meta['motion_mean'] >= 18.0 and meta['motion_max'] >= 45.0)
                    )
                    if not visible_ball or not moving_ball:
                        continue
                    if motion.get('dy', 0.0) >= 6.0 and cy_meta < prev_y - 10:
                        continue
                    predicted_gap = math.hypot(cx_meta - expected_x, cy_meta - expected_y)
                    max_gap = max(70.0, min(150.0, float(motion.get('distance', 0.0) or 0.0) * 3.0))
                    if predicted_gap > max_gap:
                        continue
                    size_ratio_meta = abs(meta['area'] - float(self.ball_size or meta['area'])) / max(float(self.ball_size or meta['area']), 1.0)
                    adjusted_score = (
                        predicted_gap * 1.5 +
                        meta['distance'] * 0.12 +
                        size_ratio_meta * 18.0 -
                        min(80.0, meta['motion_max'] * 0.45) -
                        min(40.0, meta['motion_mean'] * 0.9)
                    )
                    continuing_candidates.append((adjusted_score, meta))
                if continuing_candidates:
                    _, cont_meta = min(continuing_candidates, key=lambda item: item[0])
                    best_contour = cont_meta['contour']
                    best_source = cont_meta['source']
                    best_score = cont_meta['score']
                    print(
                        f"  DEBUG: [TOP-RETURN GRACE] prioritizing continuation candidate at "
                        f"{cont_meta['pos']} area={cont_meta['area']:.1f}px "
                        f"score={cont_meta['score']:.1f} motion="
                        f"{cont_meta['motion_mean']:.1f}/{cont_meta['motion_max']:.1f} "
                        f"source={cont_meta['source']}"
                    )

        if (candidate_meta and
                not top_return_search_context and
                not back_return_search_context and
                not serve_direction_search):
            serve_launch_meta = None
            if serve_contact_grace:
                serve_launch_meta = self._prefer_serve_contact_launch_candidate(candidate_meta, frame.shape)
            if serve_launch_meta is not None:
                best_contour = serve_launch_meta['contour']
                best_source = serve_launch_meta['source']
                best_score = serve_launch_meta['score']
            else:
                continuation_meta = self._prefer_predicted_continuation_candidate(candidate_meta, predicted_point)
                if continuation_meta is not None:
                    best_contour = continuation_meta['contour']
                    best_source = continuation_meta['source']
                    best_score = continuation_meta['score']
            post_bounce_launch_meta = self._prefer_post_bounce_racket_launch_candidate(
                candidate_meta, frame.shape
            )
            if post_bounce_launch_meta is not None:
                best_contour = post_bounce_launch_meta['contour']
                best_source = post_bounce_launch_meta['source']
                best_score = post_bounce_launch_meta['score']
            upper_far_escape_meta = self._prefer_upper_far_player_escape_candidate(
                candidate_meta, best_contour, frame.shape
            )
            if upper_far_escape_meta is not None:
                best_contour = upper_far_escape_meta['contour']
                best_source = upper_far_escape_meta['source']
                best_score = upper_far_escape_meta['score']
                upper_far_player_escape_override = True
            upper_dynamic_departure_meta = self._prefer_upper_dynamic_departure_candidate(
                candidate_meta, best_contour, frame.shape
            )
            if upper_dynamic_departure_meta is not None:
                best_contour = upper_dynamic_departure_meta['contour']
                best_source = upper_dynamic_departure_meta['source']
                best_score = upper_dynamic_departure_meta['score']
                upper_dynamic_departure_override = True
            lower_contact_launch_candidates = []
            if lower_contact_launch_context is not None:
                lower_contact_launch_candidates = [
                    meta for meta in candidate_meta
                    if self._lower_contact_launch_candidate_ok(
                        meta['pos'],
                        meta['area'],
                        lower_contact_launch_context,
                        frame.shape,
                    )
                ]
            current_best_lower_launch_ok = False
            if lower_contact_launch_candidates and best_contour is not None:
                for meta in lower_contact_launch_candidates:
                    if meta['contour'] is best_contour:
                        current_best_lower_launch_ok = True
                        break
            if lower_contact_launch_candidates and not current_best_lower_launch_ok:
                launch_meta = max(
                    lower_contact_launch_candidates,
                    key=lambda meta: (meta['area'], meta['motion_max'], -meta['score'])
                )
                best_contour = launch_meta['contour']
                best_source = launch_meta['source']
                best_score = launch_meta['score']
                print(
                    f"  DEBUG: [LOWER-RACKET-LAUNCH] prioritizing candidate at "
                    f"{launch_meta['pos']} area={launch_meta['area']:.1f}px "
                    f"score={launch_meta['score']:.1f} motion="
                    f"{launch_meta['motion_mean']:.1f}/{launch_meta['motion_max']:.1f} "
                    f"source={launch_meta['source']}"
                )
            near_camera_turn_candidates = [
                meta for meta in candidate_meta
                if self._near_camera_large_racket_turn_candidate_ok(
                    meta['pos'],
                    meta['area'],
                    frame.shape,
                    motion_mean=meta['motion_mean'],
                    motion_max=meta['motion_max'],
                )
            ]
            if near_camera_turn_candidates:
                turn_meta = max(
                    near_camera_turn_candidates,
                    key=lambda meta: (
                        meta['area'],
                        self.ball_center[1] - meta['pos'][1] if self.ball_center is not None else 0,
                        meta['motion_max'],
                        -meta['score'],
                    )
                )
                selected_meta = None
                if best_contour is not None:
                    for meta in candidate_meta:
                        if meta['contour'] is best_contour:
                            selected_meta = meta
                            break
                selected_sep = (
                    math.hypot(
                        turn_meta['pos'][0] - selected_meta['pos'][0],
                        turn_meta['pos'][1] - selected_meta['pos'][1],
                    )
                    if selected_meta is not None else float('inf')
                )
                best_contour = turn_meta['contour']
                best_source = turn_meta['source']
                best_score = turn_meta['score']
                large_lower_launch_override = True
                near_camera_large_turn_override = True
                print(
                    f"  DEBUG: [NEAR-CAMERA RACKET TURN] prioritizing large candidate at "
                    f"{turn_meta['pos']} area={turn_meta['area']:.1f}px "
                    f"score={turn_meta['score']:.1f} motion="
                    f"{turn_meta['motion_mean']:.1f}/{turn_meta['motion_max']:.1f} "
                    f"source={turn_meta['source']} sep={selected_sep:.1f}"
                )
            large_launch_candidates = [
                meta for meta in candidate_meta
                if self._large_lower_launch_candidate_ok(meta['pos'], meta['area'], frame.shape)
            ]
            if large_launch_candidates:
                launch_meta = max(
                    large_launch_candidates,
                    key=lambda meta: (meta['area'], meta['motion_max'], -meta['score'])
                )
                best_contour = launch_meta['contour']
                best_source = launch_meta['source']
                best_score = launch_meta['score']
                large_lower_launch_override = True
                print(
                    f"  DEBUG: [LARGE-LOWER-LAUNCH] prioritizing candidate at "
                    f"{launch_meta['pos']} area={launch_meta['area']:.1f}px "
                    f"score={launch_meta['score']:.1f} motion="
                    f"{launch_meta['motion_mean']:.1f}/{launch_meta['motion_max']:.1f} "
                    f"source={launch_meta['source']}"
                )

        if (candidate_meta and
                not top_return_search_context and
                not back_return_search_context and
                not serve_direction_search):
            recent_return_meta = self._prefer_recent_return_dynamic_candidate(
                candidate_meta, best_contour, frame.shape
            )
            if recent_return_meta is not None:
                best_contour = recent_return_meta['contour']
                best_source = recent_return_meta['source']
                best_score = recent_return_meta['score']

            large_descent_meta = self._prefer_large_descending_continuation_candidate(
                candidate_meta, best_contour, frame.shape
            )
            if large_descent_meta is not None:
                best_contour = large_descent_meta['contour']
                best_source = large_descent_meta['source']
                best_score = large_descent_meta['score']

            large_lower_bounce_meta = self._prefer_large_lower_bounce_launch_candidate(
                candidate_meta, best_contour, frame.shape
            )
            if large_lower_bounce_meta is not None:
                best_contour = large_lower_bounce_meta['contour']
                best_source = large_lower_bounce_meta['source']
                best_score = large_lower_bounce_meta['score']
                large_lower_bounce_launch_override = True

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
            selected_meta_for_guard = None
            should_guard_selected = self.ball_center and not allow_inactive and (
                self.stuck_frame_count < 5 or
                top_return_search_context or
                back_return_search_context or
                contact_reacquire_bounds is not None or
                self._find_frame0_background_hotspot((cx, cy)) is not None
            )
            accepted_top_return_reentry = False
            accepted_back_return_reentry = False
            if should_guard_selected:
                x_prev, y_prev = self.ball_center
                actual_distance = np.sqrt((cx - x_prev)**2 + (cy - y_prev)**2)
                edge_threshold = 5  # pixels from edge
                jump_threshold = 50  # pixels - suspicious if ball "moves" more than this
                
                frame_height, frame_width = frame.shape[:2]
                at_edge = (y_prev < edge_threshold or y_prev > frame_height - edge_threshold or
                          x_prev < edge_threshold or x_prev > frame_width - edge_threshold)
                if candidate_meta:
                    for meta in candidate_meta:
                        if meta['contour'] is best_contour:
                            selected_meta_for_guard = meta
                            break
                
                if (at_edge and actual_distance > jump_threshold and
                        not top_return_search_context and
                        not back_return_search_context):
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
                selected_predicted_distance = (
                    math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    if predicted_point is not None else None
                )
                relaxed_contact_min_y = self._contact_reacquire_min_y(
                    contact_reacquire_bounds,
                    selected_predicted_distance,
                    motion_mean,
                    motion_max,
                ) if contact_reacquire_bounds is not None else None
                selected_area = cv2.contourArea(best_contour)
                if top_return_search_context:
                    if (
                        getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket') and
                        best_source not in ('primary', 'regular', 'alt')
                    ):
                        top_return_ok = False
                        top_return_reason = f"top-return source {best_source} not allowed for upper-side reentry"
                    else:
                        top_return_ok, top_return_reason = self._top_return_reentry_ok(
                            (cx, cy), selected_area, motion_mean, motion_max, frame.shape
                        )
                    if not top_return_ok:
                        motion_reentry = self._find_top_return_motion_reentry(frame, frame_gray)
                        if motion_reentry is not None:
                            return self._commit_top_return_reentry(
                                motion_reentry, "TOP-RETURN MOTION REENTRY"
                            )
                        self._record_rejected_contour_debug(
                            best_contour,
                            x1,
                            y1,
                            cx,
                            cy,
                            selected_area,
                            top_return_reason,
                            source=best_source,
                        )
                        self.stuck_frame_count = min(self.stuck_frame_count, 4)
                        print(f"Frame {self.frame_count}: [TOP-RETURN WAIT] ignoring non-reentry blob ({cx},{cy}) "
                              f"reason={top_return_reason}")
                        return self.ball_center
                    accepted_top_return_reentry = True
                    self._recent_offscreen_return_frame = self.frame_count
                    self._top_return_reentry_grace_frames = max(
                        getattr(self, '_top_return_reentry_grace_frames', 0), 4
                    )
                if back_return_search_context:
                    back_return_ok, back_return_reason = self._back_return_reentry_ok(
                        (cx, cy), selected_area, motion_mean, motion_max, frame.shape
                    )
                    if not back_return_ok:
                        self._record_rejected_contour_debug(
                            best_contour,
                            x1,
                            y1,
                            cx,
                            cy,
                            selected_area,
                            back_return_reason,
                            source=best_source,
                        )
                        self.stuck_frame_count = min(self.stuck_frame_count, 4)
                        print(f"Frame {self.frame_count}: [BACK-RETURN WAIT] ignoring non-reentry blob ({cx},{cy}) "
                              f"reason={back_return_reason}")
                        return self.ball_center
                    accepted_back_return_reentry = True
                    self._recent_offscreen_return_frame = self.frame_count
                    self._back_return_reentry_grace_frames = max(
                        getattr(self, '_back_return_reentry_grace_frames', 0), 8
                    )
                static_hotspot = ((selected_area <= 3 and motion_mean < 1.0 and motion_max < 5.0) or
                                  (cy < 100 and motion_mean < 2.5 and motion_max < 10.0))
                last_motion_distance = (
                    float(self.last_motion.get('distance', 0.0) or 0.0)
                    if self.last_motion is not None else 0.0
                )
                recent_return_static_hold = (
                    self._recent_offscreen_return_hold_active(window_frames=8) and
                    actual_distance >= max(
                        26.0,
                        last_motion_distance * 0.40 if self.last_motion is not None else 26.0
                    ) and
                    motion_mean < 4.0 and
                    motion_max < 18.0
                )
                recent_return_static_jump = (
                    self._recent_offscreen_return_hold_active(window_frames=36) and
                    actual_distance >= max(120.0, last_motion_distance * 2.7) and
                    motion_mean < 2.5 and
                    motion_max < 12.0
                )
                suspicious_upper_static_jump = (
                    actual_distance > max(180.0, frame_width * 0.045) and
                    cy < max(220, int(frame_height * 0.11)) and
                    motion_mean < 4.0 and
                    motion_max < 12.0
                )
                outside_contact_bounds = (
                    not accepted_top_return_reentry and
                    not accepted_back_return_reentry and
                    contact_reacquire_bounds is not None and (
                        cx < contact_reacquire_bounds['min_x'] or
                        cx > contact_reacquire_bounds['max_x'] or
                        cy < relaxed_contact_min_y
                    )
                )
                frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                frame0_background = (
                    frame0_hotspot is not None and (
                        selected_motion is None or (motion_mean < 8.0 and motion_max < 35.0)
                    )
                )
                selected_predicted_path_override = (
                    selected_meta_for_guard is not None and
                    bool(selected_meta_for_guard.get('predicted_path_hotspot_override'))
                )
                if selected_predicted_path_override and (static_hotspot or frame0_background):
                    debug = selected_meta_for_guard.get('predicted_path_hotspot_debug') or {}
                    print(
                        f"Frame {self.frame_count}: [PREDICTED-PATH HOTSPOT] accepting selected "
                        f"candidate at ({cx},{cy}) pred_dist="
                        f"{debug.get('predicted_distance', selected_predicted_distance or 0.0):.1f} "
                        f"motion={motion_mean:.1f}/{motion_max:.1f}"
                    )
                    static_hotspot = False
                    frame0_background = False
                upper_exit_non_reentry = (
                    not upper_exit_wait and
                    contact_reacquire_bounds is not None and
                    self.ground_bounce_count > 0 and
                    self.ball_size is not None and
                    self.ball_size >= 35 and
                    selected_area <= max(14.0, self.ball_size * 0.35) and
                    # This hold is only for true top-edge exits. Higher in-court
                    # flights can temporarily shrink while descending and should
                    # keep following the visible path.
                    y_prev <= max(170, int(frame_height * 0.08)) and
                    actual_distance <= max(90.0, frame_width * 0.025) and
                    cy >= (y_prev - 12)
                )
                if (static_hotspot or recent_return_static_hold or recent_return_static_jump or
                        suspicious_upper_static_jump or
                        outside_contact_bounds or frame0_background):
                    reason = f"static patch mean={motion_mean:.1f} max={motion_max:.1f}"
                    if outside_contact_bounds:
                        reason = (
                            f"upper-contact bounds x={contact_reacquire_bounds['min_x']}-"
                            f"{contact_reacquire_bounds['max_x']} min_y={relaxed_contact_min_y}"
                        )
                    elif recent_return_static_hold:
                        reason = (
                            f"recent-return static hold dist={actual_distance:.1f} "
                            f"mean={motion_mean:.1f} max={motion_max:.1f}"
                        )
                    elif recent_return_static_jump:
                        reason = (
                            f"recent-return static jump dist={actual_distance:.1f} "
                            f"mean={motion_mean:.1f} max={motion_max:.1f}"
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
                    if recent_return_static_hold:
                        tiny_hue_recover = self._retrack_recent_return_tiny_hue(frame, frame_gray)
                        if tiny_hue_recover is not None:
                            prev_pos = self.ball_center
                            new_pos = tiny_hue_recover['pos']
                            self.ball_center = new_pos
                            self.ball_hsv = tiny_hue_recover['hsv']
                            self.ball_size = tiny_hue_recover['area']
                            self._update_recovered_motion(prev_pos, new_pos)
                            self._activate_regular_hsv()
                            self._recent_return_bounce_recover_frame = self.frame_count
                            self.direction_change_streak = 0
                            self.stuck_frame_count = 0
                            print(f"Frame {self.frame_count}: [RECENT RETURN TINY-HUE RECOVER] Ball at {new_pos}")
                            return self.ball_center
                    self._learn_ignored_tracking_position((cx, cy), radius=80, ttl=200, reason=reason)
                    if (self.h10_hsv_lower is not None and self.h10_hsv_upper is not None and
                            self._should_try_h10_recover(frame, predicted_point, allow_inactive)):
                        retrack_h10 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.h10_hsv_lower, upper=self.h10_hsv_upper, frame_gray=frame_gray,
                            filter_key="h_10", ignore_false_points=True, prefer_predicted_path=True
                        )
                        if retrack_h10 is not None:
                            h10_pos = retrack_h10['pos']
                            h10_prev_distance = math.hypot(
                                h10_pos[0] - self.ball_center[0],
                                h10_pos[1] - self.ball_center[1]
                            )
                            h10_pred_distance = (
                                math.hypot(h10_pos[0] - predicted_point[0], h10_pos[1] - predicted_point[1])
                                if predicted_point is not None else h10_prev_distance
                            )
                            local_cap = max(55.0, min(105.0, actual_distance * 0.6))
                            very_local_cap = max(28.0, local_cap * 0.35)
                            trajectory_local_cap = max(16.0, very_local_cap)
                            trajectory_priority = (
                                predicted_point is not None and
                                h10_pred_distance <= trajectory_local_cap and
                                (retrack_h10['motion_mean'] >= 6.0 or retrack_h10['motion_max'] >= 24.0 or retrack_h10['area'] >= 3.0)
                            )
                            weak_off_path_h10 = (
                                retrack_h10['area'] <= 1.5 and
                                h10_pred_distance > trajectory_local_cap and
                                h10_prev_distance > very_local_cap
                            )
                            if (
                                not weak_off_path_h10 and (
                                    h10_prev_distance <= very_local_cap or
                                    trajectory_priority or
                                    (h10_prev_distance <= local_cap and h10_pred_distance <= (local_cap * 0.75))
                                )
                            ):
                                self.ball_center = h10_pos
                                self.ball_hsv = retrack_h10['hsv']
                                self.ball_size = retrack_h10['area']
                                self._activate_regular_hsv()
                                self.stuck_frame_count = 0
                                print(f"Frame {self.frame_count}: [H_10 FP RECOVER] Ball at {h10_pos}")
                                return self.ball_center
                            print(
                                f"  DEBUG: Rejecting h_10 false-point recover at {h10_pos} - "
                                f"off local/predicted path ({h10_prev_distance:.1f}px/{h10_pred_distance:.1f}px caps={very_local_cap:.1f}/{local_cap:.1f})"
                            )
                    if (contact_reacquire_bounds is not None and self.ball_center is not None and
                            self.ball_center[1] <= 260 and
                            (self.ball_size is None or self.ball_size <= 35)):
                        regular_single = self._find_single_standard_candidate(
                            search_frame, x1, y1, self.ball_center, predicted_point, frame_gray
                        )
                        if self._should_use_single_regular_candidate(
                                regular_single, predicted_point, search_radius):
                            new_pos = regular_single['pos']
                            hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            self.ball_center = new_pos
                            self.ball_hsv = hsv_full[new_pos[1], new_pos[0]]
                            self.ball_size = regular_single['area']
                            self._activate_regular_hsv()
                            self.direction_change_streak = 0
                            self.stuck_frame_count = 0
                            print(f"Frame {self.frame_count}: [REGULAR FP RECOVER] Ball at {new_pos}")
                            return self.ball_center
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

            selected_area_for_precedence = (
                selected_override['area'] if selected_override is not None else cv2.contourArea(best_contour)
            )
            current_filter_key = self._candidate_false_point_filter_key(best_source, hsv_mode)
            recent_regular_single = None
            recent_bounce_regular_precedence_context = (
                not allow_inactive and
                self.ball_center is not None and
                contact_reacquire_bounds is not None and
                self.ball_center[1] <= 260 and
                (self.ball_size is None or self.ball_size <= 35) and
                self._recent_return_bounce_recover_active(window_frames=2)
            )
            lower_bounce_regular_precedence_context = (
                not allow_inactive and
                self.ball_center is not None and
                ground_bounce_context is not None and
                self.ball_size is not None and
                self.ball_size >= 60.0 and
                self.last_motion is not None and
                self.last_motion.get('dy', 0.0) >= max(35.0, frame.shape[0] * 0.018) and
                self.ball_center[1] >= max(700, int(frame.shape[0] * 0.34)) and
                selected_area_for_precedence <= max(30.0, float(self.ball_size) * 0.40)
            )
            if (not allow_inactive and self.ball_center is not None and
                    ((contact_reacquire_bounds is not None and
                      self.ball_center[1] <= 260 and
                      (self.ball_size is None or self.ball_size <= 35) and
                      (best_source == "alt6" or recent_bounce_regular_precedence_context)) or
                     lower_bounce_regular_precedence_context)):
                regular_single = self._find_single_standard_candidate(
                    search_frame, x1, y1, self.ball_center, predicted_point, frame_gray
                )
                if regular_single is not None:
                    recent_regular_single = regular_single
                    current_distance = math.hypot(cx - self.ball_center[0], cy - self.ball_center[1])
                    current_predicted_distance = (
                        math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                        if predicted_point is not None else None
                    )
                    if self._should_use_single_regular_candidate(
                            regular_single, predicted_point, search_radius,
                            current_pos=(cx, cy),
                            current_area=selected_area_for_precedence,
                            current_score=best_score,
                            current_distance=current_distance,
                            current_predicted_distance=current_predicted_distance):
                        use_regular_precedence = True
                    else:
                        use_regular_precedence = False
                    if (not use_regular_precedence and recent_bounce_regular_precedence_context):
                        regular_predicted_distance = regular_single.get("predicted_distance")
                        regular_expected_distance = None
                        current_expected_distance = None
                        if ground_bounce_context is not None:
                            expected_x, expected_y = ground_bounce_context['expected']
                            regular_expected_distance = math.hypot(
                                regular_single["pos"][0] - expected_x,
                                regular_single["pos"][1] - expected_y,
                            )
                            current_expected_distance = math.hypot(
                                cx - expected_x,
                                cy - expected_y,
                            )
                        use_regular_precedence = (
                            regular_single["area"] >= max(4.0, selected_area_for_precedence * 2.0) and
                            regular_single["pos"][1] <= cy - 14 and
                            regular_single["distance"] <= max(
                                60.0,
                                (float(self.last_motion.get('distance', 0.0) or 0.0) * 1.8)
                                if self.last_motion is not None else 60.0
                            ) and
                            (
                                (regular_expected_distance is not None and current_expected_distance is not None and
                                 regular_expected_distance + 8.0 <= current_expected_distance) or
                                regular_predicted_distance is None or
                                current_predicted_distance is None or
                                regular_predicted_distance + 8.0 <= current_predicted_distance
                            )
                        )
                    if (not use_regular_precedence and lower_bounce_regular_precedence_context):
                        last_distance = 0.0
                        if self.last_motion is not None:
                            last_distance = float(self.last_motion.get('distance', 0.0) or 0.0)
                        regular_dx = regular_single["pos"][0] - self.ball_center[0]
                        regular_dy = regular_single["pos"][1] - self.ball_center[1]
                        current_dx = cx - self.ball_center[0]
                        current_dy = cy - self.ball_center[1]
                        use_regular_precedence = (
                            regular_single["area"] >= max(18.0, selected_area_for_precedence * 3.0) and
                            regular_dx <= min(-25.0, current_dx - 10.0) and
                            regular_dy >= max(40.0, current_dy + 10.0) and
                            regular_single["distance"] >= current_distance + 18.0 and
                            regular_single["distance"] <= max(150.0, last_distance * 2.2) and
                            regular_single["score"] <= best_score + 80.0
                        )
                    if use_regular_precedence:
                        if lower_bounce_regular_precedence_context:
                            precedence_label = "LOWER BOUNCE REGULAR PRECEDENCE"
                        else:
                            precedence_label = "REGULAR PRECEDENCE" if best_source == "alt6" else "RECENT BOUNCE REGULAR PRECEDENCE"
                        best_contour = regular_single["contour"]
                        best_source = "regular"
                        best_score = regular_single["score"]
                        large_lower_launch_override = False
                        cx, cy = regular_single["pos"]
                        self._activate_regular_hsv()
                        print(
                            f"Frame {self.frame_count}: [{precedence_label}] "
                            f"Using single regular candidate at ({cx}, {cy})"
                        )

            if (self._upper_slow_arc_active() and best_contour is not None and candidate_meta and
                    self.ball_size is not None and self.ball_size <= 90):
                selected_meta = None
                for meta in candidate_meta:
                    if meta['contour'] is best_contour:
                        selected_meta = meta
                        break
                if selected_meta is not None and selected_meta['area'] <= max(3.0, float(self.ball_size) * 0.18):
                    upper_visible_arc = (
                        self.ball_center is not None and
                        getattr(self, 'ground_bounce_count', 0) > 0 and
                        self.ball_center[1] <= max(285, int(frame.shape[0] * 0.14)) and
                        (
                            self._recent_offscreen_return_hold_active(window_frames=180) or
                            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 170
                        )
                    )
                    last_motion_dist = (
                        float(self.last_motion.get('distance', 0.0) or 0.0)
                        if self.last_motion is not None else 0.0
                    )
                    gap_cap = 8.0
                    distance_slack = 8.0
                    if upper_visible_arc:
                        gap_cap = max(gap_cap, min(46.0, max(24.0, last_motion_dist * 2.6)))
                        distance_slack = 42.0
                    nearby_larger = [
                        meta for meta in candidate_meta
                        if (
                            meta is not selected_meta and
                            math.hypot(
                                meta['pos'][0] - selected_meta['pos'][0],
                                meta['pos'][1] - selected_meta['pos'][1],
                            ) <= gap_cap and
                            meta['distance'] <= selected_meta['distance'] + distance_slack and
                            meta['area'] >= max(6.0, selected_meta['area'] * 4.0) and
                            (
                                math.hypot(
                                    meta['pos'][0] - selected_meta['pos'][0],
                                    meta['pos'][1] - selected_meta['pos'][1],
                                ) <= 8.0 or
                                (
                                    upper_visible_arc and
                                    (meta['motion_max'] >= 80.0 or meta['motion_mean'] >= 18.0)
                                )
                            )
                        )
                    ]
                    if nearby_larger:
                        larger_meta = min(
                            nearby_larger,
                            key=lambda meta: (
                                meta['score'] -
                                min(80.0, meta['area'] * 5.0) -
                                min(70.0, meta['motion_max'] * 0.35) -
                                min(35.0, meta['motion_mean'] * 1.2),
                                meta.get('predicted_distance') if meta.get('predicted_distance') is not None else 9999.0,
                            )
                        )
                        best_contour = larger_meta['contour']
                        best_source = larger_meta['source']
                        best_score = larger_meta['score']
                        cx, cy = larger_meta['pos']
                        print(
                            f"  DEBUG: [SAME-SPOT SIZE PREF] using larger candidate at ({cx},{cy}) "
                            f"area={larger_meta['area']:.1f}px over {selected_meta['area']:.1f}px fragment"
                        )

            # Get HSV values at new position
            hsv_values = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[cy, cx]
            bulb_size = selected_override['area'] if selected_override is not None else cv2.contourArea(best_contour)
            current_filter_key = self._candidate_false_point_filter_key(best_source, hsv_mode)
            if best_source == 'alt4' and self.alt4_hsv_lower is not None and self.alt4_hsv_upper is not None:
                self.using_alt_hsv = False
                self.using_alt2_hsv = False
                self.using_alt3_hsv = False
                self.using_alt4_hsv = True
                self.using_alt6_hsv = False
                self.hsv_lower = self.alt4_hsv_lower
                self.hsv_upper = self.alt4_hsv_upper
            elif best_source == 'alt5' and self.alt5_hsv_lower is not None and self.alt5_hsv_upper is not None:
                self.hsv_lower = self.alt5_hsv_lower
                self.hsv_upper = self.alt5_hsv_upper
            elif best_source == 'alt6' and self.alt6_hsv_lower is not None and self.alt6_hsv_upper is not None:
                self.using_alt_hsv = False
                self.using_alt2_hsv = False
                self.using_alt3_hsv = False
                self.using_alt4_hsv = False
                self.using_alt6_hsv = True
                self.hsv_lower = self.alt6_hsv_lower
                self.hsv_upper = self.alt6_hsv_upper
            
            # Calculate velocity (distance moved)
            prev_pos = self.ball_center if self.ball_center else None
            prev_ball_size = self.ball_size
            current_pos = (cx, cy)
            current_area = bulb_size
            dx = dy = 0
            direction_deg = None
            if prev_pos:
                dx = cx - prev_pos[0]
                dy = cy - prev_pos[1]
                velocity = math.hypot(dx, dy)
                direction_deg = math.degrees(math.atan2(dy, dx))
            else:
                velocity = 0
            hsv_override_applied = False

            # If direction/speed look wrong, try alternative HSV before committing
            # Skip this during full-frame scan recovery - ball direction changed after player hit.
            # Also skip during post-reacquire window: the serve contact can instantly reverse
            # the ball, so a direction / speed change here is expected and correct.
            skip_upper_wall_override = upper_wall_search_context and best_source in ('alt4', 'alt5', 'alt6')
            if (not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace and not serve_direction_search
                    and lower_contact_launch_context is None and not _in_post_reacq and self.last_motion
                    and not skip_upper_wall_override and not upper_contact_turn_commit
                    and not large_lower_launch_override
                    and not large_lower_bounce_launch_override
                    and not upper_far_player_escape_override
                    and not upper_dynamic_departure_override
                    and not accepted_top_return_reentry and not top_return_reentry_grace
                    and not accepted_back_return_reentry and not back_return_reentry_grace
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
                        frame_gray=frame_gray, filter_key="alt1"
                    )
                    override_applied = False
                    if retrack is not None and self._should_accept_hsv_override(
                            "alt1", retrack, current_pos, current_area, prev_pos, predicted_point,
                            frame_gray, current_filter_key=current_filter_key):
                        cx, cy = retrack['pos']
                        hsv_values = retrack['hsv']
                        bulb_size = retrack['area']
                        self.using_alt_hsv = True
                        self.using_alt2_hsv = False
                        self.using_alt3_hsv = False
                        self.using_alt4_hsv = False
                        self.using_alt6_hsv = False
                        if self.alt_focus_hsv_lower is not None and self.alt_focus_hsv_upper is not None:
                            self.hsv_lower = self.alt_focus_hsv_lower
                            self.hsv_upper = self.alt_focus_hsv_upper
                        print(f"Frame {self.frame_count}: [ALT HSV OVERRIDE] Ball at ({cx}, {cy})")
                        override_applied = True
                        hsv_override_applied = True
                    if (not override_applied and self.alt2_hsv_lower is not None and
                            self.alt2_hsv_upper is not None):
                        retrack2 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            frame_gray=frame_gray, filter_key="alt2"
                        )
                        if retrack2 is not None and self._should_accept_hsv_override(
                                "alt2", retrack2, current_pos, current_area, prev_pos, predicted_point,
                                frame_gray, current_filter_key=current_filter_key):
                            cx, cy = retrack2['pos']
                            hsv_values = retrack2['hsv']
                            bulb_size = retrack2['area']
                            self.using_alt_hsv = False
                            self.using_alt2_hsv = True
                            self.using_alt3_hsv = False
                            self.using_alt6_hsv = False
                            self.hsv_lower = self.alt2_hsv_lower
                            self.hsv_upper = self.alt2_hsv_upper
                            print(f"Frame {self.frame_count}: [ALT2 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            override_applied = True
                            hsv_override_applied = True
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
                                        lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray,
                                        filter_key="alt3"
                                    )
                                    if retrack3 is not None and self._should_accept_hsv_override(
                                            "alt3", retrack3, current_pos, current_area, prev_pos,
                                            predicted_point, frame_gray,
                                            current_filter_key=current_filter_key):
                                        cx, cy = retrack3['pos']
                                        hsv_values = retrack3['hsv']
                                        bulb_size = retrack3['area']
                                        self.using_alt2_hsv = False
                                        self.using_alt3_hsv = True
                                        self.using_alt6_hsv = False
                                        self.hsv_lower = self.alt3_hsv_lower
                                        self.hsv_upper = self.alt3_hsv_upper
                                        print(f"Frame {self.frame_count}: [ALT3 HSV OVERRIDE] Ball at ({cx}, {cy})")
                    if (not override_applied and self.alt4_hsv_lower is not None and
                            self.alt4_hsv_upper is not None):
                        retrack4 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.alt4_hsv_lower, upper=self.alt4_hsv_upper, frame_gray=frame_gray,
                            filter_key="alt4"
                        )
                        if retrack4 is not None and self._should_accept_hsv_override(
                                "alt4", retrack4, current_pos, current_area, prev_pos, predicted_point,
                                frame_gray, current_filter_key=current_filter_key):
                            cx, cy = retrack4['pos']
                            hsv_values = retrack4['hsv']
                            bulb_size = retrack4['area']
                            self.using_alt_hsv = False
                            self.using_alt2_hsv = False
                            self.using_alt3_hsv = False
                            self.using_alt4_hsv = True
                            self.using_alt6_hsv = False
                            self.hsv_lower = self.alt4_hsv_lower
                            self.hsv_upper = self.alt4_hsv_upper
                            print(f"Frame {self.frame_count}: [ALT4 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            override_applied = True
                            hsv_override_applied = True
                    if (not override_applied and self.alt6_hsv_lower is not None and
                            self.alt6_hsv_upper is not None):
                        retrack6 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.alt6_hsv_lower, upper=self.alt6_hsv_upper, frame_gray=frame_gray,
                            filter_key="alt6"
                        )
                        if retrack6 is not None and self._should_accept_hsv_override(
                                "alt6", retrack6, current_pos, current_area, prev_pos, predicted_point,
                                frame_gray, current_filter_key=current_filter_key):
                            cx, cy = retrack6['pos']
                            hsv_values = retrack6['hsv']
                            bulb_size = retrack6['area']
                            self.using_alt_hsv = False
                            self.using_alt2_hsv = False
                            self.using_alt3_hsv = False
                            self.using_alt6_hsv = True
                            self.hsv_lower = self.alt6_hsv_lower
                            self.hsv_upper = self.alt6_hsv_upper
                            print(f"Frame {self.frame_count}: [ALT6 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            override_applied = True
                            hsv_override_applied = True
                    if self._should_try_alts9_11_override(frame.shape, allow_inactive):
                        specialist_current_pos = (cx, cy)
                        specialist_current_area = bulb_size
                        retrack_s911 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.alts9_11_hsv_lower, upper=self.alts9_11_hsv_upper,
                            frame_gray=frame_gray, filter_key="alts9_11", sparse_mode=True
                        )
                        if retrack_s911 is not None and self._should_accept_hsv_override(
                                "alts9_11", retrack_s911, specialist_current_pos, specialist_current_area,
                                prev_pos, predicted_point,
                                frame_gray, current_filter_key=current_filter_key):
                            cx, cy = retrack_s911['pos']
                            hsv_values = retrack_s911['hsv']
                            bulb_size = retrack_s911['area']
                            print(f"Frame {self.frame_count}: [ALTS9_11 HSV OVERRIDE] Ball at ({cx}, {cy})")
                            override_applied = True
                            hsv_override_applied = True
                    if (not override_applied and
                            (self.alt6_hsv_lower is None or self.alt6_hsv_upper is None) and
                            self.alt3_hsv_lower is not None and self.alt3_hsv_upper is not None):
                        retrack3 = self.retrack_with_alt2_hsv(
                            search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                            lower=self.alt3_hsv_lower, upper=self.alt3_hsv_upper, frame_gray=frame_gray,
                            filter_key="alt3"
                        )
                        if retrack3 is not None and self._should_accept_hsv_override(
                                "alt3", retrack3, current_pos, current_area, prev_pos, predicted_point,
                                frame_gray, current_filter_key=current_filter_key):
                            cx, cy = retrack3['pos']
                            hsv_values = retrack3['hsv']
                            bulb_size = retrack3['area']
                            self.using_alt_hsv = False
                            self.using_alt2_hsv = False
                            self.using_alt3_hsv = True
                            self.using_alt6_hsv = False
                            self.hsv_lower = self.alt3_hsv_lower
                            self.hsv_upper = self.alt3_hsv_upper
                            print(f"Frame {self.frame_count}: [ALT3 HSV OVERRIDE] Ball at ({cx}, {cy})")

            # HSV overrides can replace a stale contour with the real ball during
            # top-screen return recovery. Keep that narrow path aligned with the
            # final candidate, or it may hold the old marker because it is still
            # scoring the pre-override fragment.
            recent_top_return_override = (
                hsv_override_applied and
                self._recent_offscreen_return_hold_active(window_frames=12)
            )
            if recent_top_return_override:
                current_pos = (cx, cy)
                current_area = bulb_size
                dx = dy = 0
                direction_deg = None
                if prev_pos:
                    dx = cx - prev_pos[0]
                    dy = cy - prev_pos[1]
                    velocity = math.hypot(dx, dy)
                    direction_deg = math.degrees(math.atan2(dy, dx)) if velocity > 0 else 0.0
                else:
                    velocity = 0
                if (
                        prev_pos is not None and
                        dy >= max(8.0, frame_height * 0.004) and
                        velocity <= max(70.0, float(self.last_motion.get('distance', 0.0) or 0.0) * 1.25)
                        and bulb_size <= max(14.0, float(prev_ball_size or 0.0) * 3.5)
                        and max(175, int(frame_height * 0.081)) <= cy <= max(270, int(frame_height * 0.13))):
                    existing_defer = int(getattr(self, '_top_far_out_defer_until_frame', -1000000))
                    if self.frame_count > existing_defer:
                        self._top_far_out_defer_until_frame = self.frame_count + 3
                        print(
                            f"Frame {self.frame_count}: [TOP-FAR-OUT DEFER] "
                            f"smooth HSV return continuation at ({cx},{cy})"
                        )

            # Detect sudden ball size drop (occlusion by player)
            # If ball was > 30px and now < 5px, it's being occluded — don't trust this detection
            small_upper_flight_candidate = (
                (prev_ball_size is not None and prev_ball_size <= 25) or
                bulb_size <= 25
            )
            soft_upper_contact_recover = (
                not allow_inactive and
                contact_reacquire_bounds is not None and
                self.ball_center is not None and
                self.last_motion is not None and
                self.last_motion.get('dy', 0.0) < -2.0 and
                self.ground_bounce_count > 0 and
                small_upper_flight_candidate and
                not accepted_top_return_reentry and not accepted_back_return_reentry
            )
            if soft_upper_contact_recover:
                soft_recover_applied = False
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
                    soft_recover_motion_mean = soft_recover.get('motion_mean')
                    soft_recover_motion_max = soft_recover.get('motion_max')
                    soft_recover_strong = (
                        soft_recover['area'] >= max(12.0, float(prev_ball_size or 0.0) * 2.0) or
                        (soft_recover_motion_mean is not None and soft_recover_motion_mean >= 18.0) or
                        (soft_recover_motion_max is not None and soft_recover_motion_max >= 100.0)
                    )
                    recent_return_upper_flight = self._recent_offscreen_return_hold_active(window_frames=80)
                    same_path_larger_recover = (
                        recent_return_upper_flight and
                        soft_recover_strong and
                        math.hypot(soft_recover['pos'][0] - cx, soft_recover['pos'][1] - cy) <= max(8.0, velocity * 0.35) and
                        soft_recover['area'] >= max(bulb_size * 1.6, bulb_size + 8.0) and
                        recovered_upward >= current_upward - 4.0
                    )
                    if (soft_recover_strong and (
                            recovered_upward >= current_upward + 12 or
                            recovered_predicted_distance + 10 < current_predicted_distance or
                            same_path_larger_recover)):
                        cx, cy = soft_recover['pos']
                        hsv_values = soft_recover['hsv']
                        bulb_size = soft_recover['area']
                        dx = cx - self.ball_center[0]
                        dy = cy - self.ball_center[1]
                        velocity = math.hypot(dx, dy)
                        direction_deg = math.degrees(math.atan2(dy, dx)) if velocity > 0 else 0.0
                        soft_recover_applied = True
                        recover_reason = "same-path larger mask" if same_path_larger_recover else "predicted/upward"
                        print(f"Frame {self.frame_count}: [UPPER CONTACT SOFT RECOVER] Ball at ({cx}, {cy}) "
                              f"from {soft_recover.get('label', 'n/a')} mode={soft_recover.get('mode', 'n/a')} "
                              f"reason={recover_reason}")
                current_dx = 0.0
                current_dy = 0.0
                current_velocity = 0.0
                if self.ball_center is not None:
                    current_dx = cx - self.ball_center[0]
                    current_dy = cy - self.ball_center[1]
                    current_velocity = math.hypot(current_dx, current_dy)
                current_candidate_tiny = (
                    bulb_size <= max(2.0, float(prev_ball_size or 0.0) * 0.7)
                )
                current_candidate_far = (
                    self.last_motion is not None and
                    current_velocity >= max(22.0, self.last_motion.get('distance', 0.0) * 1.9)
                )
                soft_recover_weak = (
                    soft_recover is not None and
                    soft_recover['area'] <= max(10.0, float(prev_ball_size or 0.0) * 3.0) and
                    ((soft_recover.get('motion_mean') is None or soft_recover.get('motion_mean') < 15.0)) and
                    ((soft_recover.get('motion_max') is None or soft_recover.get('motion_max') < 80.0))
                )
                if (not soft_recover_applied and current_candidate_tiny and current_candidate_far and soft_recover_weak):
                    print(f"Frame {self.frame_count}: [UPPER CONTACT HOLD] weak override/recover "
                          f"area={bulb_size:.1f}px vel={current_velocity:.1f}px "
                          f"soft_area={(soft_recover['area'] if soft_recover is not None else -1):.1f}")
                    self._deferred_motion_anchor = {
                        'frame': self.frame_count,
                        'pos': soft_recover['pos'],
                        'label': soft_recover.get('label', 'n/a'),
                    }
                    return self.ball_center

            upper_post_bounce_size_spike = (
                not allow_inactive and
                contact_reacquire_bounds is not None and
                self.ball_center is not None and
                self.ground_bounce_count > 0 and
                prev_ball_size is not None and prev_ball_size <= 15 and
                bulb_size > max(18, prev_ball_size * 4.0) and
                not accepted_top_return_reentry and not accepted_back_return_reentry
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

            deferred_motion_anchor = getattr(self, '_deferred_motion_anchor', None)
            if deferred_motion_anchor is not None:
                if (deferred_motion_anchor.get('frame') == self.frame_count - 1 and
                        self.ball_center is not None):
                    anchor_pos = deferred_motion_anchor.get('pos')
                    if anchor_pos is not None:
                        anchor_dx = cx - anchor_pos[0]
                        anchor_dy = cy - anchor_pos[1]
                        anchor_velocity = math.hypot(anchor_dx, anchor_dy)
                        if velocity > 0 and anchor_velocity + 8.0 < velocity and anchor_velocity <= velocity * 0.8:
                            dx = anchor_dx
                            dy = anchor_dy
                            velocity = anchor_velocity
                            direction_deg = math.degrees(math.atan2(dy, dx)) if velocity > 0 else 0.0
                            print(f"Frame {self.frame_count}: [DEFERRED MOTION ANCHOR] using "
                                  f"{deferred_motion_anchor.get('label', 'n/a')} anchor at {anchor_pos} "
                                  f"for motion ({velocity:.1f}px)")
                self._deferred_motion_anchor = None

            predicted_path_size_drop_ok = False
            if (
                    selected_meta_for_guard is not None and
                    selected_meta_for_guard.get('predicted_path_hotspot_override') and
                    selected_meta_for_guard.get('source') in ('primary', 'regular', 'alt', 'single') and
                    selected_meta_for_guard.get('pos') == (cx, cy) and
                    abs(float(selected_meta_for_guard.get('area', bulb_size)) - float(bulb_size)) <= 0.5
            ):
                debug = selected_meta_for_guard.get('predicted_path_hotspot_debug') or {}
                predicted_path_size_drop_ok = (
                    debug.get('motion_max', 0.0) >= 35.0 or
                    debug.get('motion_mean', 0.0) >= 8.0 or
                    debug.get('predicted_distance', float('inf')) <=
                    max(6.0, debug.get('predicted_cap', 0.0) * 0.75)
                )

            if (prev_ball_size and prev_ball_size > 30 and bulb_size < 5 and
                    not top_return_search_context and not accepted_top_return_reentry and
                    not top_return_reentry_grace and
                    not back_return_search_context and not accepted_back_return_reentry and
                    not back_return_reentry_grace and not predicted_path_size_drop_ok):
                print(f"Frame {self.frame_count}: Ball size dropped {prev_ball_size:.0f}->{bulb_size:.0f}px - likely occluded by player")
                self.stuck_frame_count += 2  # accelerate stuck detection
                return self.ball_center
            if (prev_ball_size and prev_ball_size > 30 and bulb_size < 5 and
                    predicted_path_size_drop_ok):
                debug = selected_meta_for_guard.get('predicted_path_hotspot_debug') or {}
                print(
                    f"Frame {self.frame_count}: [PREDICTED-PATH SIZE DROP] accepting "
                    f"{prev_ball_size:.0f}->{bulb_size:.0f}px at ({cx},{cy}) "
                    f"pred_dist={debug.get('predicted_distance', 0.0):.1f} "
                    f"motion={debug.get('motion_mean', 0.0):.1f}/"
                    f"{debug.get('motion_max', 0.0):.1f}"
                )

            if not allow_inactive and frame_gray is not None:
                final_motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                self._last_tracked_candidate_frame = self.frame_count
                self._last_tracked_candidate_motion_frame = self.frame_count
                self._last_tracked_candidate_motion_mean = (
                    final_motion_metrics['mean'] if final_motion_metrics is not None else 0.0
                )
                self._last_tracked_candidate_motion_max = (
                    final_motion_metrics['max'] if final_motion_metrics is not None else 0.0
                )

            # Gate large direction/velocity changes for a few frames
            # Skip this gate during full-frame scan recovery (ball changed direction after player hit).
            # Also skip during the post-reacquire window: serve contact reverses direction immediately.
            if (not serve_contact_grace and not rally_contact_grace and not ground_bounce_grace and not serve_direction_search
                    and not upper_exit_transition_context and not _in_post_reacq and self.last_motion
                    and self.ball_center and self.stuck_frame_count < 5
                    and not accepted_top_return_reentry and not top_return_reentry_grace
                    and not accepted_back_return_reentry and not back_return_reentry_grace):
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
                prev_dx = float(self.last_motion.get('dx', 0.0)) if self.last_motion is not None else 0.0
                prev_dy = float(self.last_motion.get('dy', 0.0)) if self.last_motion is not None else 0.0
                angle_change_threshold = max(0.0, float(getattr(self, 'direction_change_min_degrees', 20.0)))
                same_forward_x = abs(prev_dx) <= 3.0 or abs(dx) <= 3.0 or (prev_dx * dx) >= -6.0
                forward_vertical_reversal = (
                    prev_dy >= 4.0 and
                    dy <= -4.0 and
                    same_forward_x
                )
                hold_change_detected = (
                    (angle_jump > angle_threshold) or
                    (speed_ratio > 2.0) or
                    (speed_ratio < 0.5)
                )
                change_detected = (
                    hold_change_detected or
                    forward_vertical_reversal or
                    angle_jump >= angle_change_threshold
                )
                predicted_turn_candidate = False
                predicted_continuation_candidate = False
                small_ball_upper_flight = (
                    prev_ball_size is not None and prev_ball_size <= 12 and bulb_size <= 12
                )
                recent_upper_return_local_turn = (
                    self._recent_offscreen_return_hold_active(window_frames=36) and
                    small_ball_upper_flight and
                    220 <= cy < max(420, int(frame_height * 0.20)) and
                    velocity <= max(110.0, lm_dist * 1.25) and
                    dy >= -8 and
                    (motion_max >= 80.0 or motion_mean >= 18.0)
                )
                if (predicted_point is not None and small_ball_upper_flight and
                        240 <= cy < max(420, int(frame_height * 0.20))):
                    predicted_turn_distance = math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                    predicted_turn_cap = max(90, int(frame_width * 0.025))
                    if (
                            self._recent_offscreen_return_hold_active(window_frames=36) and
                            (motion_max >= 80.0 or motion_mean >= 18.0) and
                            velocity <= max(95.0, lm_dist * 1.15)):
                        predicted_turn_cap = max(predicted_turn_cap, 135.0)
                    predicted_turn_candidate = predicted_turn_distance <= predicted_turn_cap
                if recent_upper_return_local_turn:
                    predicted_turn_candidate = True
                lower_contact_launch_candidate = False
                if lower_contact_launch_context is not None and self.ball_center is not None:
                    lower_contact_launch_candidate = self._lower_contact_launch_candidate_ok(
                        (cx, cy),
                        bulb_size,
                        lower_contact_launch_context,
                        frame.shape,
                    )
                lower_racket_contact_turn_candidate = self._lower_racket_contact_turn_candidate_ok(
                    (cx, cy),
                    bulb_size,
                    frame.shape,
                    dx,
                    dy,
                    angle_jump,
                    motion_mean=motion_mean,
                    motion_max=motion_max,
                )
                immediate_lower_racket_contact_turn_candidate = self._immediate_lower_racket_contact_turn_candidate_ok(
                    (cx, cy),
                    bulb_size,
                    frame.shape,
                    dx,
                    dy,
                    angle_jump,
                    motion_mean=motion_mean,
                    motion_max=motion_max,
                )
                lower_racket_contact_turn_candidate = (
                    lower_racket_contact_turn_candidate or
                    immediate_lower_racket_contact_turn_candidate
                )
                upper_racket_contact_turn_candidate = self._upper_racket_contact_turn_candidate_ok(
                    (cx, cy),
                    bulb_size,
                    frame.shape,
                    dx,
                    dy,
                    angle_jump,
                    speed_ratio,
                    motion_mean=motion_mean,
                    motion_max=motion_max,
                )
                large_lower_launch_candidate = self._large_lower_launch_candidate_ok(
                    (cx, cy), bulb_size, frame.shape
                ) or large_lower_launch_override
                upper_slow_arc_candidate = self._upper_slow_arc_candidate_ok(
                    (cx, cy), bulb_size, velocity, predicted_point, frame.shape, frame_gray=frame_gray
                )
                ground_bounce_candidate = False
                ground_bounce_rebound_candidate = False
                upper_soft_ground_bounce_candidate = False
                if ground_bounce_context is not None and self.ball_center is not None:
                    origin_x, origin_y = ground_bounce_context['origin']
                    expected_x, expected_y = ground_bounce_context['expected']
                    bounce_dist = math.hypot(cx - origin_x, cy - origin_y)
                    upward_progress = origin_y - cy
                    expected_distance = math.hypot(cx - expected_x, cy - expected_y)
                    lateral_dx = cx - origin_x
                    incoming_dx = ground_bounce_context['incoming_dx']
                    incoming_dy = ground_bounce_context['incoming_dy']
                    same_direction_x = abs(incoming_dx) <= 2.0 or (lateral_dx * incoming_dx) >= -8.0
                    candidate_not_tiny = bulb_size >= max(1, min(int(ground_bounce_context['ref_size'] * 0.08), 4))
                    rebound_x_reverse = (lateral_dx * incoming_dx) < -8.0
                    strong_rebound_motion = (
                        motion_max >= 90.0 or
                        motion_mean >= 25.0 or
                        bulb_size >= max(45.0, ground_bounce_context['ref_size'] * 0.55)
                    )
                    ground_bounce_rebound_candidate = (
                        rebound_x_reverse and
                        angle_jump >= 115.0 and
                        origin_y <= max(1150, int(frame_height * 0.55)) and
                        abs(lateral_dx) <= max(95.0, frame_width * 0.035) and
                        upward_progress >= max(ground_bounce_context['min_upward'], frame_height * 0.020) and
                        expected_distance <= max(ground_bounce_context['expected_cap'], 95.0) and
                        bulb_size >= max(35.0, ground_bounce_context['ref_size'] * 0.50) and
                        strong_rebound_motion
                    )
                    ground_bounce_candidate = (
                        (same_direction_x or ground_bounce_rebound_candidate) and
                        candidate_not_tiny and
                        upward_progress >= ground_bounce_context['min_upward'] and
                        bounce_dist >= ground_bounce_context['min_launch_dist'] and
                        bounce_dist <= ground_bounce_context['max_launch_dist'] and
                        expected_distance <= ground_bounce_context['expected_cap']
                    )
                    if large_lower_bounce_launch_override:
                        ground_bounce_candidate = True
                    upper_soft_ground_bounce_candidate = (
                        not ground_bounce_candidate and
                        self._late_contact_prior_bounce_active() and
                        self._upper_slow_arc_active() and
                        origin_y <= max(360, int(frame_height * 0.18)) and
                        prev_dy >= 2.0 and
                        dy <= -3.0 and
                        candidate_not_tiny and
                        upward_progress >= max(3.0, ground_bounce_context['min_upward'] * 0.80) and
                        bounce_dist >= max(2.0, ground_bounce_context['min_launch_dist'] * 0.75) and
                        bounce_dist <= ground_bounce_context['max_launch_dist'] and
                        expected_distance <= max(ground_bounce_context['expected_cap'], 42.0)
                    )
                    post_miss_racket_debug = None
                    post_miss_racket_hint = False
                    if (
                            self.stuck_frame_count > 0 and
                            origin_y <= max(360, int(frame_height * 0.18)) and
                            incoming_dy >= max(6.0, frame_height * 0.0035)):
                        post_miss_racket_debug = self._racket_below_ball_motion_debug(
                            (origin_x, origin_y), frame
                        )
                        post_miss_racket_hint = bool(post_miss_racket_debug.get('object_hint'))
                    upper_post_miss_bounce_seed = (
                        not ground_bounce_candidate and
                        not upper_soft_ground_bounce_candidate and
                        self.stuck_frame_count > 0 and
                        origin_y <= max(360, int(frame_height * 0.18)) and
                        incoming_dy >= max(6.0, frame_height * 0.0035) and
                        post_miss_racket_hint and
                        dy <= 1.0 and
                        upward_progress >= -2.0 and
                        bounce_dist <= max(20.0, ground_bounce_context['min_launch_dist'] * 2.4) and
                        expected_distance <= max(32.0, ground_bounce_context['expected_cap']) and
                        candidate_not_tiny and
                        (motion_max >= 70.0 or motion_mean >= 12.0)
                    )
                    if upper_soft_ground_bounce_candidate or upper_post_miss_bounce_seed:
                        ground_bounce_candidate = True
                        change_detected = True
                        if upper_post_miss_bounce_seed:
                            racket_mean = (
                                post_miss_racket_debug.get('mean', 0.0)
                                if post_miss_racket_debug is not None else 0.0
                            )
                            racket_max = (
                                post_miss_racket_debug.get('max', 0.0)
                                if post_miss_racket_debug is not None else 0.0
                            )
                            print(
                                f"Frame {self.frame_count}: [UPPER BOUNCE SEED] accepting close post-miss "
                                f"candidate ({cx},{cy}) from origin=({origin_x},{origin_y}) "
                                f"motion={motion_mean:.1f}/{motion_max:.1f} "
                                f"racket_below={racket_mean:.1f}/{racket_max:.1f}"
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
                if upper_far_player_escape_override:
                    predicted_continuation_candidate = True
                recent_bounce_reversal_candidate = False
                recent_bounce_continue_needed = self._recent_return_bounce_recover_active(window_frames=2)
                if recent_bounce_continue_needed:
                    recent_bounce_pred_distance = (
                        math.hypot(cx - predicted_point[0], cy - predicted_point[1])
                        if predicted_point is not None else velocity
                    )
                    recent_bounce_regular_candidate = (
                        recent_regular_single is not None and
                        (best_source in ("regular", "primary") or current_filter_key in ("regular_court", "primary"))
                    )
                    recent_bounce_reversal_candidate = (
                        not recent_bounce_regular_candidate and
                        velocity >= 18.0 and
                        dy <= -18.0 and
                        recent_bounce_pred_distance <= max(90, int(frame_width * 0.03)) and
                        (motion_mean >= 10.0 or motion_max >= 60.0) and
                        bulb_size >= max(4.0, min(max(float(prev_ball_size or 0.0), 4.0) * 0.55, 18.0))
                    )
                    recent_bounce_continue_needed = (
                        not recent_bounce_reversal_candidate and
                        not recent_bounce_regular_candidate and
                        (velocity <= 16.0 or abs(dy) <= 12.0 or (motion_mean < 18.0 and motion_max < 90.0))
                    )
                else:
                    recent_bounce_regular_candidate = False

                serve_bounce_in_event = self._serve_bounce_in_from_turn_candidate(
                    (cx, cy), frame, dx, dy, angle_jump, velocity
                )
                serve_bounce_out_event = self._serve_bounce_out_from_turn_candidate(
                    (cx, cy), frame, dx, dy, angle_jump, velocity
                )
                impact_event = None
                if (hold_change_detected and not predicted_turn_candidate and not predicted_continuation_candidate
                        and not lower_contact_launch_candidate and not large_lower_launch_candidate
                        and not upper_racket_contact_turn_candidate
                        and not upper_slow_arc_candidate and not ground_bounce_candidate
                        and serve_bounce_in_event is None
                        and serve_bounce_out_event is None
                        and not recent_bounce_reversal_candidate and not recent_bounce_regular_candidate):
                    self.direction_change_streak += 1
                    max_hold = 1 if immediate_lower_racket_contact_turn_candidate else (
                        2 if (near_net or lower_racket_contact_turn_candidate) else 3
                    )
                    if self.direction_change_streak < max_hold:
                        if recent_bounce_continue_needed:
                            recent_bounce_continue = self._retrack_recent_return_bounce_continue(
                                frame, frame_gray, predicted_point=predicted_point
                            )
                            if recent_bounce_continue is not None:
                                self._register_ground_bounce_from_context(
                                    ground_bounce_context, frame, source_label="recent bounce continue"
                                )
                                prev_pos = self.ball_center
                                new_pos = recent_bounce_continue['pos']
                                self.ball_center = new_pos
                                self.ball_hsv = recent_bounce_continue['hsv']
                                self.ball_size = recent_bounce_continue['area']
                                self._update_recovered_motion(prev_pos, new_pos)
                                self._activate_regular_hsv()
                                self._recent_return_bounce_recover_frame = self.frame_count
                                self.direction_change_streak = 0
                                self.stuck_frame_count = 0
                                print(f"Frame {self.frame_count}: [RECENT RETURN BOUNCE CONTINUE] Ball at {new_pos}")
                                return self.ball_center
                        print(f"Frame {self.frame_count}: Direction change candidate (holding {self.direction_change_streak}/{max_hold}) angle_jump={angle_jump:.1f}")
                        self.stuck_frame_count += 1
                        return self.ball_center
                    if lower_racket_contact_turn_candidate:
                        if self.ground_bounce_count > 0:
                            print(f"Frame {self.frame_count}: Resetting bounce count after lower-racket contact turn")
                        self.ground_bounce_count = 0
                        self.last_ground_bounce_frame = -1000000
                        contact_origin = self.ball_center
                        rally_ref_cap = 340.0 if near_camera_large_turn_override else 140.0
                        contact_grace_frames = (
                            5 if immediate_lower_racket_contact_turn_candidate else 3
                        )
                        self._rally_contact_grace_frames = max(
                            getattr(self, '_rally_contact_grace_frames', 0),
                            contact_grace_frames,
                        )
                        if near_camera_large_turn_override:
                            self._rally_contact_large_ref_frames = max(
                                getattr(self, '_rally_contact_large_ref_frames', 0),
                                3,
                            )
                        self._rally_contact_ref_size = max(40.0, min(float(bulb_size), rally_ref_cap))
                        self._rally_contact_origin = contact_origin
                        self._rally_contact_expected = (cx, cy)
                        self._rally_contact_progress = math.hypot(
                            cx - contact_origin[0],
                            cy - contact_origin[1],
                        )
                        impact_event = {
                            'kind': 'racket_contact',
                            'point': contact_origin,
                            'label': 'racket contact',
                        }
                        print(f"Frame {self.frame_count}: Allowing lower-racket contact turn")
                        self._last_racket_contact_frame = self.frame_count
                        self._last_racket_contact_point = impact_event['point']
                        self._add_impact_marker(
                            impact_event['point'],
                            kind=impact_event['kind'],
                            ttl=7,
                            label=impact_event['label'],
                        )
                        self.direction_change_streak = 0
                else:
                    if change_detected and upper_racket_contact_turn_candidate:
                        if self.ground_bounce_count > 0:
                            print(f"Frame {self.frame_count}: Resetting bounce count after upper-racket contact turn")
                        self.ground_bounce_count = 0
                        self.last_ground_bounce_frame = -1000000
                        contact_origin = self.ball_center
                        self._rally_contact_grace_frames = max(
                            getattr(self, '_rally_contact_grace_frames', 0),
                            3,
                        )
                        self._rally_contact_ref_size = max(12.0, min(float(bulb_size), 60.0))
                        self._rally_contact_origin = contact_origin
                        self._rally_contact_expected = (cx, cy)
                        self._rally_contact_progress = math.hypot(
                            cx - contact_origin[0],
                            cy - contact_origin[1],
                        )
                        impact_event = {
                            'kind': 'racket_contact',
                            'point': contact_origin,
                            'label': 'upper racket contact',
                        }
                        print(f"Frame {self.frame_count}: Allowing upper-racket contact turn")
                    if (change_detected and not ground_bounce_candidate and not predicted_turn_candidate
                            and not predicted_continuation_candidate and not upper_slow_arc_candidate
                            and serve_bounce_out_event is None
                            and not recent_bounce_reversal_candidate and not recent_bounce_regular_candidate):
                        strong_x_reversal = (self.last_motion['dx'] * dx) < -12 if self.last_motion is not None else False
                        if (lower_contact_launch_candidate or large_lower_launch_candidate or
                                strong_x_reversal or angle_jump >= 120 or speed_ratio > 1.8):
                            if self.ground_bounce_count > 0:
                                if (
                                    large_lower_launch_candidate and
                                    self.ball_center is not None and
                                    self.ball_center[1] >= int(frame_height * 0.50)
                                ):
                                    self._late_contact_prior_bounce_count = max(
                                        int(getattr(self, '_late_contact_prior_bounce_count', 0)),
                                        int(self.ground_bounce_count),
                                    )
                                    self._late_contact_prior_bounce_until_frame = max(
                                        int(getattr(self, '_late_contact_prior_bounce_until_frame', -1000000)),
                                        self.frame_count + 90,
                                    )
                                    print(
                                        f"Frame {self.frame_count}: Preserving prior bounce count "
                                        f"{self._late_contact_prior_bounce_count} through late lower launch"
                                    )
                                print(f"Frame {self.frame_count}: Resetting bounce count after non-bounce shot change")
                            self.ground_bounce_count = 0
                            self.last_ground_bounce_frame = -1000000
                    if change_detected and serve_bounce_in_event is not None:
                        bounce_point = serve_bounce_in_event['point']
                        self.ground_bounce_count += 1
                        self.last_ground_bounce_frame = self.frame_count
                        print(f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} detected (serve box jump)")
                        self._handle_ground_bounce_event(
                            bounce_point,
                            frame,
                            accepted_in_reason="Serve bounce in",
                        )
                        print(
                            f"Frame {self.frame_count}: Allowing serve-bounce-in continuation "
                            f"at {bounce_point}"
                        )
                    elif change_detected and serve_bounce_out_event is not None:
                        bounce_point = serve_bounce_out_event['point']
                        reason = serve_bounce_out_event['reason']
                        self.ground_bounce_count += 1
                        self.last_ground_bounce_frame = self.frame_count
                        self._pending_rally_end_reason = reason
                        self._pending_rally_end_frame = self.frame_count
                        self._awaiting_serve_bounce = False
                        self._add_impact_marker(
                            bounce_point,
                            kind="serve_bounce",
                            color=(0, 0, 255),
                            label=reason,
                        )
                        print(f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} detected (serve box jump)")
                        print(f"Frame {self.frame_count}: [BOUNCE OUT] {reason} at {bounce_point}")
                        print(f"Frame {self.frame_count}: Allowing serve-bounce-out continuation")
                    elif change_detected and predicted_turn_candidate:
                        print(f"Frame {self.frame_count}: Allowing upper-flight turn near predicted path")
                    elif change_detected and predicted_continuation_candidate:
                        if upper_far_player_escape_override:
                            print(f"Frame {self.frame_count}: Allowing upper-far player ball correction")
                        else:
                            print(f"Frame {self.frame_count}: Allowing predicted-path continuation after speed drop")
                    elif change_detected and upper_slow_arc_candidate:
                        print(f"Frame {self.frame_count}: Allowing upper slow-arc continuation")
                    elif change_detected and recent_bounce_regular_candidate:
                        self._register_ground_bounce_from_context(
                            ground_bounce_context, frame, source_label="recent bounce regular"
                        )
                        self._recent_return_bounce_recover_frame = self.frame_count
                        print(f"Frame {self.frame_count}: Allowing recent bounce regular-candidate continuation")
                    elif change_detected and recent_bounce_reversal_candidate:
                        self._register_ground_bounce_from_context(
                            ground_bounce_context, frame, source_label="recent bounce rebound"
                        )
                        self._recent_return_bounce_recover_frame = self.frame_count
                        print(f"Frame {self.frame_count}: Allowing recent bounce rebound continuation")
                    elif change_detected and lower_contact_launch_candidate:
                        self._rally_contact_grace_frames = max(getattr(self, '_rally_contact_grace_frames', 0), 3)
                        self._rally_contact_ref_size = max(40.0, min(float(bulb_size), 140.0))
                        self._rally_contact_origin = lower_contact_launch_context['origin']
                        self._rally_contact_expected = lower_contact_launch_context['expected']
                        self._rally_contact_progress = math.hypot(
                            cx - self._rally_contact_origin[0],
                            cy - self._rally_contact_origin[1],
                        )
                        impact_event = {
                            'kind': 'racket_contact',
                            'point': lower_contact_launch_context['origin'],
                            'label': 'racket contact',
                        }
                        print(f"Frame {self.frame_count}: Allowing lower-racket contact launch")
                    elif change_detected and large_lower_launch_candidate:
                        launch_origin = self.ball_center
                        self._rally_contact_grace_frames = max(getattr(self, '_rally_contact_grace_frames', 0), 3)
                        rally_ref_cap = 340.0 if near_camera_large_turn_override else 140.0
                        if near_camera_large_turn_override:
                            self._rally_contact_large_ref_frames = max(
                                getattr(self, '_rally_contact_large_ref_frames', 0),
                                3,
                            )
                        self._rally_contact_ref_size = max(40.0, min(float(bulb_size), rally_ref_cap))
                        self._rally_contact_origin = launch_origin
                        self._rally_contact_expected = (cx, cy)
                        self._rally_contact_progress = math.hypot(
                            cx - launch_origin[0],
                            cy - launch_origin[1],
                        )
                        if launch_origin is not None and launch_origin[1] >= int(frame_height * 0.50):
                            self._rally_contact_high_launch_until_frame = max(
                                getattr(self, '_rally_contact_high_launch_until_frame', -1),
                                self.frame_count + 3,
                            )
                        impact_event = {
                            'kind': 'racket_contact',
                            'point': launch_origin,
                            'label': 'large lower-court launch',
                        }
                        self._upper_slow_arc_until_frame = max(
                            getattr(self, '_upper_slow_arc_until_frame', -1),
                            self.frame_count + 45,
                        )
                        print(f"Frame {self.frame_count}: Allowing large lower-court launch")
                    elif change_detected and ground_bounce_candidate and not upper_racket_contact_turn_candidate:
                        racket_below_contact, racket_below_debug = (
                            self._ground_bounce_has_racket_below_contact(
                                ground_bounce_context, frame
                            )
                        )
                        if racket_below_contact:
                            self._handle_racket_below_contact_instead_of_bounce(
                                ground_bounce_context,
                                frame,
                                (cx, cy),
                                bulb_size,
                                racket_below_debug,
                            )
                            print(f"Frame {self.frame_count}: Allowing racket-contact continuation")
                        else:
                            self._register_ground_bounce_from_context(
                                ground_bounce_context, frame,
                                source_label="racket rebound" if ground_bounce_rebound_candidate else "predicted launch"
                            )
                            if ground_bounce_rebound_candidate:
                                self._last_racket_contact_frame = self.frame_count
                                self._last_racket_contact_point = (cx, cy)
                            self._ground_bounce_grace_frames = max(getattr(self, '_ground_bounce_grace_frames', 0), 3)
                            self._ground_bounce_ref_size = max(8.0, min(max(float(bulb_size), ground_bounce_context['ref_size']), 90.0))
                            self._ground_bounce_origin = ground_bounce_context['origin']
                            self._ground_bounce_expected = ground_bounce_context['expected']
                            self._ground_bounce_progress = math.hypot(
                                cx - self._ground_bounce_origin[0],
                                cy - self._ground_bounce_origin[1],
                            )
                            print(f"Frame {self.frame_count}: Allowing ground-bounce continuation")
                    elif change_detected:
                        impact_event = self._classify_direction_change_impact(
                            (cx, cy),
                            frame.shape,
                            dx,
                            dy,
                            angle_jump,
                            speed_ratio,
                            serve_contact_grace=serve_contact_grace,
                            predicted_turn_candidate=predicted_turn_candidate,
                            predicted_continuation_candidate=predicted_continuation_candidate,
                            lower_contact_launch_candidate=lower_contact_launch_candidate,
                            lower_contact_launch_context=lower_contact_launch_context,
                            ground_bounce_candidate=ground_bounce_candidate,
                            ground_bounce_context=ground_bounce_context,
                        )
                    if impact_event is not None:
                        if impact_event.get('kind') == 'racket_contact':
                            self._last_racket_contact_frame = self.frame_count
                            self._last_racket_contact_point = impact_event.get('point')
                        self._add_impact_marker(
                            impact_event.get('point'),
                            kind=impact_event.get('kind', 'direction_change'),
                            ttl=7,
                            label=impact_event.get('label'),
                        )
                    self.direction_change_streak = 0

            # Update tracking data
            # If re-acquiring after full-frame scan, reset velocity/direction state
            if self.stuck_frame_count >= 5:
                print(f"Frame {self.frame_count}: [RE-ACQUIRED] Ball found at ({cx},{cy}) after {self.stuck_frame_count} stuck frames")
                self._maybe_handle_reacquire_ground_bounce((cx, cy), frame)
                self.ball_velocity_history = []
                self.last_motion = None
                self.last_direction = None
                self.direction_change_streak = 0
                self.stuck_frame_count = 0
                self._recent_max_ball_size = 0
                self.ball_center = (cx, cy)
                self.ball_hsv = hsv_values
                self.ball_size = bulb_size
                if not allow_inactive and frame_gray is not None:
                    final_motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                    self._last_tracked_candidate_motion_frame = self.frame_count
                    self._last_tracked_candidate_motion_mean = (
                        final_motion_metrics['mean'] if final_motion_metrics is not None else 0.0
                    )
                    self._last_tracked_candidate_motion_max = (
                        final_motion_metrics['max'] if final_motion_metrics is not None else 0.0
                    )
                self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Skip all correction mechanisms (focus loss, alt HSV) on re-acquisition
                return self.ball_center
            self.ball_center = (cx, cy)
            self.ball_hsv = hsv_values
            self.ball_size = bulb_size
            if not allow_inactive and frame_gray is not None:
                final_motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
                self._last_tracked_candidate_motion_frame = self.frame_count
                self._last_tracked_candidate_motion_mean = (
                    final_motion_metrics['mean'] if final_motion_metrics is not None else 0.0
                )
                self._last_tracked_candidate_motion_max = (
                    final_motion_metrics['max'] if final_motion_metrics is not None else 0.0
                )
                if top_return_search_context:
                    if (
                        accepted_top_return_reentry or
                        getattr(self, '_top_return_mode', 'edge') in ('upper_side', 'upper_racket')
                    ):
                        self._top_return_wait_frames = 0
                        self._top_return_anchor = None
                        self._top_return_origin_frame = -1
                        self._top_return_mode = None
                        self._top_return_exit_dx = 0.0
                    elif cy >= max(26, getattr(self, '_top_return_anchor', (cx, cy))[1] + 8):
                        self._top_return_wait_frames = 0
                        self._top_return_anchor = None
                        self._top_return_origin_frame = -1
                        self._top_return_mode = None
                        self._top_return_exit_dx = 0.0
            if back_return_search_context:
                anchor = getattr(self, '_back_return_anchor', None)
                if (anchor is None or
                        cy <= anchor[1] - 900 or
                        abs(cx - anchor[0]) >= 420):
                    self._back_return_wait_frames = 0
                    self._back_return_anchor = None
                    self._back_return_origin_frame = -1
                else:
                    self._back_return_wait_frames = max(self._back_return_wait_frames, 4)
            if serve_direction_search:
                # The first few frames after serve contact contain the biggest
                # legitimate direction/speed change of the point.
                self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 3)
            if rally_contact_grace:
                current_ref = getattr(self, '_rally_contact_ref_size', None)
                if current_ref is None:
                    current_ref = bulb_size
                rally_ref_cap = 340.0 if getattr(self, '_rally_contact_large_ref_frames', 0) > 0 else 140.0
                self._rally_contact_ref_size = max(
                    40.0,
                    min(max(float(current_ref), float(bulb_size)), rally_ref_cap),
                )
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
                focus_loss_triggered = self.log_motion_metrics(prev_pos, dx, dy, velocity, direction_deg)
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
                    frame_gray=frame_gray, filter_key="alt1"
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
                current_hold_pos = self.ball_center
                retrack2 = self.retrack_with_alt2_hsv(
                    search_frame, x1, y1, prev_pos, predicted_point, prev_ball_size, allow_inactive,
                    frame_gray=frame_gray, filter_key="alt2"
                )
                if retrack2 is not None:
                    new_pos = retrack2['pos']
                    new_area = retrack2['area']
                    new_hsv = retrack2['hsv']
                    reject_zero_motion_retrack = False
                    reference_motion = self.prev_motion
                    if (reference_motion is None or
                            float(reference_motion.get('distance', 0.0) or 0.0) < 20.0):
                        reference_motion = getattr(self, 'last_nonzero_motion', None)
                    if prev_pos and current_hold_pos is not None and reference_motion is not None:
                        prev_motion_distance = float(reference_motion.get('distance', 0.0) or 0.0)
                        if prev_motion_distance >= 20.0:
                            retrack_dx = new_pos[0] - prev_pos[0]
                            retrack_dy = new_pos[1] - prev_pos[1]
                            retrack_distance = math.hypot(retrack_dx, retrack_dy)
                            dot = (
                                float(reference_motion.get('dx', 0.0) or 0.0) * retrack_dx +
                                float(reference_motion.get('dy', 0.0) or 0.0) * retrack_dy
                            )
                            trusted_predicted_point = (
                                int(current_hold_pos[0] + float(reference_motion.get('dx', 0.0) or 0.0)),
                                int(current_hold_pos[1] + float(reference_motion.get('dy', 0.0) or 0.0)),
                            )
                            hold_predicted_distance = (
                                math.hypot(
                                    current_hold_pos[0] - trusted_predicted_point[0],
                                    current_hold_pos[1] - trusted_predicted_point[1]
                                )
                            )
                            retrack_predicted_distance = (
                                math.hypot(
                                    new_pos[0] - trusted_predicted_point[0],
                                    new_pos[1] - trusted_predicted_point[1]
                                )
                            )
                            predicted_path_worse = (
                                retrack_predicted_distance >= hold_predicted_distance + 28.0
                            )
                            if (retrack_distance >= max(35.0, prev_motion_distance * 0.40) and
                                    dot < 0.0 and
                                    predicted_path_worse):
                                reject_zero_motion_retrack = True
                                print(
                                    f"Frame {self.frame_count}: [RETRACK ALT2 HOLD] keeping {current_hold_pos} "
                                    f"instead of opposite-direction jump to {new_pos} "
                                    f"pred={retrack_predicted_distance:.1f}>{hold_predicted_distance:.1f}"
                                )

                    if not reject_zero_motion_retrack:
                        self.using_alt2_hsv = True
                        self.using_alt3_hsv = False
                        self.using_alt6_hsv = False
                        if self.alt2_hsv_lower is not None and self.alt2_hsv_upper is not None:
                            self.hsv_lower = self.alt2_hsv_lower
                            self.hsv_upper = self.alt2_hsv_upper
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
            tracking_size_cap = max(150, self.serve_ball_size_max)
            if 'ball_size_max_tracking' in locals():
                tracking_size_cap = ball_size_max_tracking
            size_cap = f"{self.serve_ball_size_min}-{self.serve_ball_size_max}px (serve scan)" if allow_inactive else f"1-{tracking_size_cap}px"
            print(f"  DEBUG: All {len(contours)} contours were rejected by size filter ({size_cap})")
            # Show the actual sizes that were rejected
            rejected_sizes = []
            for source, contour in contours[:5]:  # Show first 5
                rejected_sizes.append(f"{cv2.contourArea(contour):.1f}px")
            print(f"  DEBUG: Rejected sizes (first 5): {', '.join(rejected_sizes)}")
            print(f"  DEBUG: REASON: Ball size changed outside {size_cap}")
            print(f"  DEBUG:   - Ball may be too small (far away) or too large (very close)")
            print(f"  DEBUG:   - Consider adjusting size filter if ball is visible")
            if predicted_point:
                print(f"  DEBUG: Predicted point was {predicted_point}, consider widening search around it")
        if best_contour is None and top_return_search_context and self.ball_center is not None:
            self.stuck_frame_count = min(self.stuck_frame_count, 4)
            print(f"Frame {self.frame_count}: [TOP-RETURN WAIT] no valid re-entry candidate, holding {self.ball_center}")
            return self.ball_center
        if best_contour is None and back_return_search_context and self.ball_center is not None:
            if getattr(self, '_back_return_wait_frames', 0) <= 0:
                self._back_return_timed_out = True
                self._back_return_anchor = None
                self._back_return_origin_frame = -1
                print(f"Frame {self.frame_count}: [BACK-RETURN WAIT] timed out after "
                      f"{getattr(self, '_back_return_timeout_frames', 8)} frames with no re-entry")
                return None
            self.stuck_frame_count = min(self.stuck_frame_count, 4)
            print(f"Frame {self.frame_count}: [BACK-RETURN WAIT] no valid re-entry candidate, holding {self.ball_center}")
            return self.ball_center
        if (best_contour is None and not allow_inactive and self.ball_center is not None
                and contact_reacquire_bounds is not None
                and self.ball_center[1] <= 260
                and (self.ball_size is None or self.ball_size <= 35)
                and self.stuck_frame_count >= 2):
            regular_single = self._find_single_standard_candidate(
                search_frame, x1, y1, self.ball_center, predicted_point, frame_gray
            )
            regular_search_radius = search_radius if search_radius is not None else max(x2 - x1, y2 - y1) / 2.0
            if self._should_use_single_regular_candidate(
                    regular_single, predicted_point, regular_search_radius):
                new_pos = regular_single['pos']
                hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                self.ball_center = new_pos
                self.ball_hsv = hsv_full[new_pos[1], new_pos[0]]
                self.ball_size = regular_single['area']
                self._activate_regular_hsv()
                print(f"Frame {self.frame_count}: [REGULAR RECOVER] Ball at {new_pos}")
                return self.ball_center
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
                self.using_alt6_hsv = False
                self.hsv_lower = upper_exit_low_s['lower']
                self.hsv_upper = upper_exit_low_s['upper']
                self.stuck_frame_count = 0
                print(f"Frame {self.frame_count}: [UPPER EXIT LOW-S RECOVER] Ball at {new_pos} via {upper_exit_low_s['label']}")
                return self.ball_center
        h10_local_upper_y_cap = max(220, int(frame.shape[0] * 0.12))
        if (best_contour is None and self.ball_center is not None and
                self.ball_center[1] <= h10_local_upper_y_cap and
                (predicted_point is None or predicted_point[1] <= (h10_local_upper_y_cap + 16)) and
                search_radius is not None and search_radius <= 100 and
                self.h10_hsv_lower is not None and self.h10_hsv_upper is not None and
                self._should_try_h10_recover(frame, predicted_point, allow_inactive)):
            retrack_h10 = self.retrack_with_alt2_hsv(
                search_frame, x1, y1, self.ball_center, predicted_point, self.ball_size, allow_inactive,
                lower=self.h10_hsv_lower, upper=self.h10_hsv_upper, frame_gray=frame_gray,
                filter_key="h_10", ignore_false_points=True
            )
            if retrack_h10 is not None:
                h10_pos = retrack_h10['pos']
                prev_distance = math.hypot(
                    h10_pos[0] - self.ball_center[0],
                    h10_pos[1] - self.ball_center[1]
                )
                predicted_distance = (
                    math.hypot(h10_pos[0] - predicted_point[0], h10_pos[1] - predicted_point[1])
                    if predicted_point is not None else prev_distance
                )
                local_cap = min(45.0, max(28.0, (search_radius or 0) * 0.6)) if search_radius is not None else 35.0
                if prev_distance <= local_cap and predicted_distance <= (local_cap + 12.0):
                    self.ball_center = h10_pos
                    self.ball_hsv = retrack_h10['hsv']
                    self.ball_size = retrack_h10['area']
                    self._activate_regular_hsv()
                    self.stuck_frame_count = 0
                    print(f"Frame {self.frame_count}: [H_10 NO-VALID RECOVER] Ball at {h10_pos}")
                    return self.ball_center
                print(
                    f"  DEBUG: Rejecting h_10 no-valid recover at {h10_pos} - "
                    f"too far from local window ({prev_distance:.1f}px/{predicted_distance:.1f}px > {local_cap:.1f}px)"
                )
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

    def _draw_motion_debug_vectors(self, frame, scale=1.0):
        vectors = getattr(self, 'motion_debug_vectors', [])
        if not vectors:
            return frame

        result = frame
        recent_vectors = vectors[-max(1, int(getattr(self, 'motion_debug_vector_limit', 8))):]
        total = max(1, len(recent_vectors))
        for idx, entry in enumerate(recent_vectors):
            p0 = entry.get('from')
            p1 = entry.get('to')
            if p0 is None or p1 is None:
                continue
            fade = 0.35 + (0.65 * ((idx + 1) / float(total)))
            base_color = (0, 255, 255) if entry.get('changed') else (180, 180, 180)
            color = tuple(int(round(channel * fade)) for channel in base_color)
            x0 = int(p0[0] * scale)
            y0 = int(p0[1] * scale)
            x1 = int(p1[0] * scale)
            y1 = int(p1[1] * scale)
            cv2.line(result, (x0, y0), (x1, y1), color, 2)

            direction_deg = entry.get('direction_deg')
            angle_diff = entry.get('angle_diff')
            label = f"{direction_deg:+.0f}°" if direction_deg is not None else "N/A"
            if angle_diff is not None:
                label += f" / d{angle_diff:.0f}°"
            if entry.get('vertical_reversal'):
                label += " V"

            label_x = int(((p0[0] + p1[0]) * 0.5) * scale) + 4
            label_y = int(((p0[1] + p1[1]) * 0.5) * scale) - 4
            cv2.putText(
                result,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        return result

    def draw_analysis_info(self, frame, scale=1.0, show_paused_rejected=False, game_state=None):
        """Draw analysis information on the frame with proper scaling."""
        result = frame.copy()
        if game_state is None:
            game_state = getattr(self, 'current_game_state', None)

        point_end_visuals = self._active_point_end_visuals()
        use_point_end_visuals = point_end_visuals if self.ball_center is None else None
        visual_ball_center = self.ball_center
        visual_ball_hsv = self.ball_hsv
        visual_ball_size = self.ball_size
        visual_hsv_lower = self.hsv_lower
        visual_hsv_upper = self.hsv_upper
        if use_point_end_visuals is not None:
            visual_ball_center = use_point_end_visuals.get('ball_center')
            visual_ball_hsv = use_point_end_visuals.get('ball_hsv')
            visual_ball_size = use_point_end_visuals.get('ball_size')
            visual_hsv_lower = use_point_end_visuals.get('hsv_lower')
            visual_hsv_upper = use_point_end_visuals.get('hsv_upper')

        if visual_ball_center:
            # Scale ball coordinates for display
            x = int(visual_ball_center[0] * scale)
            y = int(visual_ball_center[1] * scale)
            
            # Draw green circle around the ball
            cv2.circle(result, (x, y), 10, (0, 255, 0), 2)  # Circle outline
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)  # Center dot
            
            # Draw info text at the top
            if visual_ball_hsv is not None and visual_ball_size is not None:
                # Show current HSV values at clicked point
                hsv_text = f"Ball HSV: H={visual_ball_hsv[0]}, S={visual_ball_hsv[1]}, V={visual_ball_hsv[2]}"
                cv2.putText(result, hsv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show HSV filter range
                if visual_hsv_lower is not None and visual_hsv_upper is not None:
                    filter_text = f"Filter: H={visual_hsv_lower[0]}-{visual_hsv_upper[0]}, S={visual_hsv_lower[1]}-{visual_hsv_upper[1]}, V={visual_hsv_lower[2]}-{visual_hsv_upper[2]}"
                    cv2.putText(result, filter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show bulb size
                size_text = f"Bulb Size: {visual_ball_size:.0f}px"
                cv2.putText(result, size_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show ball position
                pos_text = f"Ball Pos: ({visual_ball_center[0]}, {visual_ball_center[1]})"
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
            good_events = []
            problematic_events = []
            direction_events = getattr(self, 'direction_change_events', [])
            net_points = getattr(self, 'net_contact_points', [])
            if use_point_end_visuals is not None:
                direction_events = use_point_end_visuals.get('direction_change_events', [])
                net_points = use_point_end_visuals.get('net_contact_points', [])
            for event in direction_events:
                point = event.get('pos')
                if point is None:
                    continue
                if visual_ball_center is not None and math.hypot(
                    point[0] - visual_ball_center[0],
                    point[1] - visual_ball_center[1],
                ) <= 12:
                    continue
                is_good = event.get('status') == 'good'
                color = (255, 0, 0) if is_good else (0, 0, 255)
                px = int(point[0] * scale)
                py = int(point[1] * scale)
                cv2.circle(result, (px, py), 11, color, 2)
                cv2.circle(result, (px, py), 3, color, -1)
                frame_label = str(event.get('frame', '?'))
                cv2.putText(
                    result,
                    frame_label,
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                )
                if is_good:
                    good_events.append(event)
                else:
                    problematic_events.append(event)
            # Draw net contact points
            for point in net_points:
                px = int(point[0] * scale)
                py = int(point[1] * scale)
                cv2.circle(result, (px, py), 12, (0, 0, 255), 3)

            col_top = 240
            col_x1 = 10
            col_x2 = 190
            line_h = 16
            cv2.putText(result, "GOOD CHANGES", (col_x1, col_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(result, "SUSPECT CHANGES", (col_x2, col_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for idx, event in enumerate(good_events[-18:]):
                yy = col_top + 18 + idx * line_h
                label = f"f{event.get('frame', '?')} {event.get('reason', '')}"
                cv2.putText(result, label, (col_x1, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1)
            for idx, event in enumerate(problematic_events[-18:]):
                yy = col_top + 18 + idx * line_h
                label = f"f{event.get('frame', '?')} {event.get('reason', '')}"
                cv2.putText(result, label, (col_x2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

        self._prune_bounce_markers()
        bounce_markers = getattr(self, 'recent_bounce_markers', [])
        if use_point_end_visuals is not None:
            bounce_markers = use_point_end_visuals.get('recent_bounce_markers', [])
        for marker in bounce_markers:
            point = marker.get('pos')
            base_color = marker.get('color', (255, 0, 0))
            if point is None:
                continue
            start_frame = int(marker.get('start_frame', self.frame_count))
            ttl = max(1, int(marker.get('ttl', 7)))
            age = max(0, self.frame_count - start_frame)
            if ttl <= 1:
                fade = 1.0
            else:
                fade = max(0.0, 1.0 - (age / float(ttl - 1)))
            color = tuple(int(round(channel * fade)) for channel in base_color)
            px = int(point[0] * scale)
            py = int(point[1] * scale)
            cv2.circle(result, (px, py), 12, color, 3)
            cv2.circle(result, (px, py), 4, color, -1)

        result = self._draw_motion_debug_vectors(result, scale=scale)

        top_zone = self._active_top_return_overlay_region((frame.shape[0], frame.shape[1]))
        if top_zone is not None:
            tx1, ty1, tx2, ty2 = top_zone
            cv2.rectangle(
                result,
                (int(tx1 * scale), int(ty1 * scale)),
                (int(tx2 * scale), int(ty2 * scale)),
                (0, 215, 255),
                2,
            )
            cv2.putText(
                result,
                "TOP RETURN ZONE",
                (int(tx1 * scale) + 4, int(ty1 * scale) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 215, 255),
                2,
            )

        # Show motion metrics and focus-loss status
        if self.last_motion is not None:
            distance = self.last_motion['distance']
            direction_deg = self.last_motion['direction_deg']
            direction_text = f"{direction_deg:+.1f} deg" if direction_deg is not None else "N/A"
            motion_text = f"Ball Move: {distance:.1f}px | Dir: {direction_text}"
            cv2.putText(result, motion_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(
                result,
                f"Dir Change Threshold: {self.direction_change_min_degrees:.0f} deg",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )
        if self.focus_loss_active:
            cv2.putText(result, "FOCUS LOST", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.using_alt6_hsv:
            cv2.putText(result, "USING ALTERNATIVE 6 HSV", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.using_alt2_hsv:
            cv2.putText(result, "USING ALTERNATIVE 2 HSV", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.using_alt_hsv:
            cv2.putText(result, "USING ALTERNATIVE HSV", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
        if self.frame_count <= getattr(self, '_serve_scan_block_until_frame', -1):
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
        
        # DEBUG: Log serve detection attempts
        if 1170 <= self.frame_count <= 1200:
            print(f"[DETECT_SERVE] f{self.frame_count}: contours_found={len(contours)} "
                  f"hsv_range=H[{self.hsv_lower[0]}-{self.hsv_upper[0]}]S[{self.hsv_lower[1]}-{self.hsv_upper[1]}]V[{self.hsv_lower[2]}-{self.hsv_upper[2]}] "
                  f"size_range=[{self.serve_ball_size_min}-{self.serve_ball_size_max}]px")
        
        # Look for ball-sized contours. With a wider serve band, prefer blobs that
        # stay near the detected server horizontally while still rewarding size.
        best_contour = None
        best_score = float('inf')
        valid_count = 0

        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Tennis ball size range: configurable per court (small for far-end, large for near-end)
            if 1170 <= self.frame_count <= 1200:
                print(f"[CONTOUR_CHECK] f{self.frame_count}: area={area:.1f}px (need [{self.serve_ball_size_min}-{self.serve_ball_size_max}])")
            
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
                        if 1170 <= self.frame_count <= 1200:
                            print(f"[CONTOUR_REJECT] f{self.frame_count}: ({cx},{cy}) is in ignored list")
                        continue

                    if server_center_x is not None and server_x_gate is not None:
                        server_dx = abs(cx - server_center_x)
                        if server_dx > server_x_gate:
                            if 1170 <= self.frame_count <= 1200:
                                print(f"[CONTOUR_REJECT] f{self.frame_count}: ({cx},{cy}) outside server gate")
                            continue

                    frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
                    if frame0_hotspot is not None:
                        if 1170 <= self.frame_count <= 1200:
                            print(f"[CONTOUR_REJECT] f{self.frame_count}: ({cx},{cy}) is frame0 hotspot")
                    
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
                            if 1170 <= self.frame_count <= 1200:
                                print(f"[CONTOUR_REJECT] f{self.frame_count}: ({cx},{cy}) frame0_hotspot={frame0_hotspot} motion_delta={motion_delta} (rejected by hotspot filter)")
                            continue
                        if 1170 <= self.frame_count <= 1200:
                            print(f"[CONTOUR_ACCEPT] f{self.frame_count}: ({cx},{cy}) area={area:.1f}px score={score:.1f} motion_delta={motion_delta}")
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
        if 1170 <= self.frame_count <= 1200:
            if best_contour is None:
                print(f"[DETECT_RESULT] f{self.frame_count}: No valid contour found, returning None")
            else:
                print(f"[DETECT_RESULT] f{self.frame_count}: FOUND serve at {best_contour} score={best_score:.1f}")
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

    def _get_net_polygon(self):
        points = getattr(self, 'net_area_points', None) or []
        if len(points) < 6:
            return None
        return np.array(points[:6], dtype=np.float32)

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        px, py = float(point[0]), float(point[1])
        x1, y1 = float(seg_start[0]), float(seg_start[1])
        x2, y2 = float(seg_end[0]), float(seg_end[1])
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / max(1e-6, dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _net_contact_geometry(self, point):
        polygon = self._get_net_polygon()
        if polygon is None:
            return None
        signed_dist = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True)
        top_polyline = polygon[:3]
        top_dist = min(
            self._point_to_segment_distance(point, top_polyline[0], top_polyline[1]),
            self._point_to_segment_distance(point, top_polyline[1], top_polyline[2]),
        )
        return {
            'signed_dist': float(signed_dist),
            'inside': signed_dist >= 0.0,
            'near_polygon': signed_dist >= -18.0,
            'near_top_tape': top_dist <= 26.0,
            'top_dist': float(top_dist),
        }

    def _opponent_side_crossing_after_contact(self, contact_side, contact_frame, frame_shape):
        if contact_side not in ("near", "far"):
            return None
        if not hasattr(self, 'net_area_y_min') or not hasattr(self, 'net_area_y_max'):
            return None
        frame_height = frame_shape[0]
        margin = max(18.0, frame_height * 0.010)
        for entry in getattr(self, 'motion_history', []):
            if entry.get('frame', -1000000) <= contact_frame:
                continue
            pos = entry.get('pos')
            if pos is None:
                continue
            py = float(pos[1])
            if contact_side == "near" and py <= float(self.net_area_y_min) - margin:
                return entry
            if contact_side == "far" and py >= float(self.net_area_y_max) + margin:
                return entry
        return None

    def _same_side_pre_net_bounce_candidate(self, ball_position, frame):
        """Detect a post-stroke bounce on the hitter side before the ball crosses the net."""
        if self.prev_motion is None or self.last_motion is None:
            return False, None
        if getattr(self, '_awaiting_serve_bounce', False):
            serve_bounce_window_active = (
                self.point_start_frame_internal is not None and
                (self.frame_count - self.point_start_frame_internal) <= 45
            )
            if serve_bounce_window_active:
                return False, None
            self._awaiting_serve_bounce = False
        if (self.frame_count - getattr(self, '_recent_racket_rebound_bounce_frame', -1000000)) <= 5:
            return False, None
        if not hasattr(self, 'net_area_y_min') or not hasattr(self, 'net_area_y_max'):
            return False, None

        contact_frame = getattr(self, '_last_racket_contact_frame', -1000000)
        frames_since_contact = self.frame_count - contact_frame
        if frames_since_contact < 3 or frames_since_contact > 90:
            return False, None

        contact_point = getattr(self, '_last_racket_contact_point', None)
        if contact_point is None:
            return False, None

        height = frame.shape[0]
        net_min = float(self.net_area_y_min)
        net_max = float(self.net_area_y_max)
        contact_margin = max(55.0, height * 0.035)
        bounce_margin = max(18.0, height * 0.012)
        contact_y = float(contact_point[1])
        if contact_y >= net_max + contact_margin:
            contact_side = "near"
        elif contact_y <= net_min - contact_margin:
            contact_side = "far"
        else:
            return False, None

        curr_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        curr_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        prev_dx = float(self.prev_motion.get('dx', 0.0) or 0.0)
        prev_dy = float(self.prev_motion.get('dy', 0.0) or 0.0)
        curr_speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        prev_speed = float(self.prev_motion.get('distance', 0.0) or 0.0)
        if prev_speed < 18.0 or curr_speed < 10.0:
            return False, None

        x, y = ball_position
        bounce_point = (int(round(x - curr_dx)), int(round(y - curr_dy)))
        bounce_y = float(bounce_point[1])
        upper_exit_band = max(140.0, height * 0.075)
        if (
                contact_side == "far" and
                bounce_y <= upper_exit_band and
                float(y) <= upper_exit_band + max(18.0, height * 0.012) and
                curr_dy <= -8.0):
            print(
                f"Frame {self.frame_count}: [SAME-SIDE PRE-NET SUPPRESSED] "
                f"upper/top exit path bounce_point={bounce_point} "
                f"pos={ball_position} curr_motion=({curr_dx:.1f},{curr_dy:.1f}) "
                f"contact={contact_point}"
            )
            return False, None
        if contact_side == "near":
            if bounce_y < net_max - bounce_margin:
                return False, None
        elif bounce_y > net_min + bounce_margin:
            return False, None

        prev_dir = self.prev_motion.get('direction_deg')
        curr_dir = self.last_motion.get('direction_deg')
        angle_diff = 0.0
        if prev_dir is not None and curr_dir is not None:
            delta = abs(curr_dir - prev_dir) % 360
            angle_diff = min(delta, 360 - delta)

        vertical_reversal = prev_dy >= 18.0 and curr_dy <= -10.0
        sharp_turn = angle_diff >= 120.0 and prev_dy >= 12.0 and curr_dy <= -6.0
        if not (vertical_reversal or sharp_turn):
            return False, None

        outside, _, left_x, right_x = self._point_outside_singles_sidelines(bounce_point, frame)
        inside_width_known = left_x is not None and right_x is not None
        inside_or_unknown = (not outside) or not inside_width_known
        if not inside_or_unknown:
            return False, None

        crossed_entry = self._opponent_side_crossing_after_contact(
            contact_side, contact_frame, frame.shape
        )
        reason = "Ball bounced before crossing net on hitter side"
        crossing_text = ""
        if crossed_entry is not None:
            after_cross_margin = max(80.0, height * 0.040)
            after_cross_near_net = (
                (contact_side == "near" and bounce_y <= net_max + after_cross_margin) or
                (contact_side == "far" and bounce_y >= net_min - after_cross_margin)
            )
            if not after_cross_near_net:
                print(
                    f"Frame {self.frame_count}: [SAME-SIDE PRE-NET SUPPRESSED] "
                    f"ball already crossed net after contact at frame {crossed_entry.get('frame')} "
                    f"pos={crossed_entry.get('pos')} contact={contact_point} side={contact_side} "
                    f"bounce_point={bounce_point}"
                )
                return False, None
            reason = "Ball bounced on hitter side after crossing net"
            crossing_text = (
                f" crossed_after_contact_frame={crossed_entry.get('frame')} "
                f"crossed_pos={crossed_entry.get('pos')}"
            )
        print(
            f"Frame {self.frame_count}: [SAME-SIDE PRE-NET BOUNCE] "
            f"bounce_point={bounce_point} contact={contact_point} side={contact_side} "
            f"frames_since_contact={frames_since_contact} prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
            f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) angle_diff={angle_diff:.1f}"
            f"{crossing_text}"
        )
        return True, reason
    
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
        top_far_out, top_far_reason = self._top_far_baseline_fall_out_candidate(ball_position, frame)
        if top_far_out:
            return True, top_far_reason
        if (self._back_return_wait_active() or
                getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
                self._recent_offscreen_return_hold_active(window_frames=24)):
            return False, "Back-return wait"
        if getattr(self, 'ground_bounce_count', 0) >= 2:
            return True, "Ball bounced twice on court"
        
        # Check if ball is out of court bounds
        if x < 0 or x > width or y < 0 or y > height:
            return True, "Ball out of court bounds"

        if self._upper_fence_fall_end_candidate(ball_position, frame.shape):
            return True, "Ball hit upper fence and fell down"

        suppress_out_bounce = (
            self._back_return_wait_active() or
            getattr(self, '_back_return_reentry_grace_frames', 0) > 0
        )
        if not suppress_out_bounce:
            out_bounce_detected, out_bounce_reason = self._detect_out_of_court_bounce(ball_position, frame)
            if out_bounce_detected:
                return True, out_bounce_reason
        
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

    def detect_point_end(self, ball_position, frame):
        """Detect if a point has ended based on ball position and behavior."""
        height, width = frame.shape[:2]
        x, y = ball_position
        recent_ground_bounce = (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 2

        if self.frame_count <= (self.start_frame + 15):
            return False, "Early-serve grace"
        if getattr(self, 'edge_wait', False):
            return False, "Edge wait"
        top_far_out, top_far_reason = self._top_far_baseline_fall_out_candidate(ball_position, frame)
        if top_far_out:
            return True, top_far_reason
        if (self._back_return_wait_active() or
                getattr(self, '_back_return_reentry_grace_frames', 0) > 0 or
                self._recent_offscreen_return_hold_active(window_frames=24)):
            return False, "Back-return wait"
        if getattr(self, 'ground_bounce_count', 0) >= 2:
            return True, "Ball bounced twice on court"

        if x < 0 or x > width or y < 0 or y > height:
            return True, "Ball out of court bounds"

        if self._upper_fence_fall_end_candidate(ball_position, frame.shape):
            return True, "Ball hit upper fence and fell down"

        suppress_out_bounce = (
            self._back_return_wait_active() or
            getattr(self, '_back_return_reentry_grace_frames', 0) > 0
        )
        if not suppress_out_bounce:
            out_bounce_detected, out_bounce_reason = self._detect_out_of_court_bounce(ball_position, frame)
            if out_bounce_detected:
                return True, out_bounce_reason

        same_side_bounce, same_side_reason = self._same_side_pre_net_bounce_candidate(ball_position, frame)
        if same_side_bounce:
            return True, same_side_reason

        net_geometry = self._net_contact_geometry(ball_position)
        if net_geometry is not None and not recent_ground_bounce:
            net_zone_active = net_geometry['near_polygon'] or net_geometry['near_top_tape']
            core_net_zone = net_geometry['inside']
            if net_zone_active:
                ball_is_fast = self.last_motion and self.last_motion['distance'] > 15
                if core_net_zone and not ball_is_fast:
                    if not hasattr(self, 'net_area_frames'):
                        self.net_area_frames = 0
                    self.net_area_frames += 1
                else:
                    self.net_area_frames = 0

                net_linger_limit = 8 if ball_is_fast else 6
                if getattr(self, 'net_area_frames', 0) > net_linger_limit:
                    return True, "Ball hit the net"

                if self.last_motion and self.last_motion['distance'] < 2.5:
                    prev_was_slow = self.prev_motion and self.prev_motion['distance'] < 10
                    if prev_was_slow and core_net_zone:
                        return True, "Ball hit the net"

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
                        direct_contact_zone = core_net_zone or net_geometry['near_top_tape']
                        last_change_point = getattr(self, '_last_direction_change_point', None)
                        last_change_geometry = (
                            self._net_contact_geometry(last_change_point)
                            if last_change_point is not None else None
                        )
                        recent_net_turn = (
                            getattr(self, '_last_direction_change_frame', -1000000) == self.frame_count and
                            last_change_geometry is not None and
                            (last_change_geometry['inside'] or last_change_geometry['near_top_tape']) and
                            curr_dy >= 4.0 and
                            direct_contact_zone
                        )
                        if recent_net_turn:
                            return True, "Ball hit the net"
                        if direct_contact_zone and direction_reversed and slowed_after_impact:
                            return True, "Ball hit the net"
                        if direct_contact_zone and angle_diff >= 60 and slowed_after_impact:
                            return True, "Ball hit the net"
                        if direct_contact_zone and angle_diff >= 90 and speed_ratio < 0.6 and prev_dist < 30:
                            return True, "Ball hit the net"
            else:
                if hasattr(self, 'net_area_frames'):
                    self.net_area_frames = 0
        elif hasattr(self, 'net_area_frames'):
            self.net_area_frames = 0

        if hasattr(self, 'ball_velocity_history') and len(self.ball_velocity_history) > 10:
            recent_velocities = self.ball_velocity_history[-10:]
            avg_velocity = sum(recent_velocities) / len(recent_velocities)
            print(f"[BALL_STOPPED_CHECK] f{self.frame_count}: avg_vel={avg_velocity:.1f} hist={[round(v,1) for v in recent_velocities]}")
            if avg_velocity < 5:
                print(f"[BALL_STOPPED] f{self.frame_count}: avg_vel={avg_velocity:.1f} < 5 -> POINT ENDS")
                return True, "Ball stopped (possible double bounce)"

        edge_margin = 50
        if (x < edge_margin or x > width - edge_margin or
            y < edge_margin or y > height - edge_margin):
            if getattr(self, 'edge_wait', False) or getattr(self, 'near_edge', False):
                return False, "Edge return grace"
            self.edge_wait = True
            return False, "Edge return grace"

        return False, "Point continues"

    def _build_singles_sideline_model(self, frame):
        frame_shape = frame.shape[:2]
        if (self._singles_sideline_model is not None and
                self._singles_sideline_frame_shape == frame_shape):
            return self._singles_sideline_model

        height, width = frame_shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        min_line_length = max(250, int(width * 0.08))
        lines = cv2.HoughLinesP(
            white_mask,
            1,
            np.pi / 180,
            threshold=120,
            minLineLength=min_line_length,
            maxLineGap=20,
        )
        if lines is None:
            self._singles_sideline_model = None
            self._singles_sideline_frame_shape = frame_shape
            return None

        y_ref = int(height * 0.74)
        left_candidates = []
        right_candidates = []
        fallback_left = []
        fallback_right = []

        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            dy = y2 - y1
            if abs(dy) < 20.0:
                continue
            dx = x2 - x1
            angle = math.degrees(math.atan2(dy, dx))
            if not ((-75.0 < angle < -42.0) or (42.0 < angle < 75.0)):
                continue

            a = dx / dy
            b = x1 - a * y1
            x_ref = a * y_ref + b
            length = math.hypot(dx, dy)
            entry = {
                'a': a,
                'b': b,
                'x_ref': x_ref,
                'length': length,
                'angle': angle,
            }
            if angle < 0:
                if width * 0.14 <= x_ref <= width * 0.32:
                    left_candidates.append(entry)
                elif width * 0.06 <= x_ref <= width * 0.40:
                    fallback_left.append(entry)
            else:
                if width * 0.68 <= x_ref <= width * 0.83:
                    right_candidates.append(entry)
                elif width * 0.60 <= x_ref <= width * 0.90:
                    fallback_right.append(entry)

        if not left_candidates and fallback_left:
            left_candidates = sorted(fallback_left, key=lambda entry: entry['x_ref'], reverse=True)[:4]
        if not right_candidates and fallback_right:
            right_candidates = sorted(fallback_right, key=lambda entry: entry['x_ref'])[:4]

        if not left_candidates or not right_candidates:
            self._singles_sideline_model = None
            self._singles_sideline_frame_shape = frame_shape
            return None

        left_best = max(left_candidates, key=lambda entry: entry['length'])
        right_best = max(right_candidates, key=lambda entry: entry['length'])
        self._singles_sideline_model = {
            'left': left_best,
            'right': right_best,
            'margin': max(12.0, width * 0.0035),
        }
        self._singles_sideline_frame_shape = frame_shape
        print(
            f"  DEBUG: Singles sideline model built: "
            f"left_x@{y_ref}={left_best['x_ref']:.1f} angle={left_best['angle']:.1f}, "
            f"right_x@{y_ref}={right_best['x_ref']:.1f} angle={right_best['angle']:.1f}"
        )
        return self._singles_sideline_model

    def _point_outside_singles_sidelines(self, point, frame):
        model = self._build_singles_sideline_model(frame)
        if model is None:
            return False, None, None, None

        x, y = point
        height, _ = frame.shape[:2]
        if not (int(height * 0.12) <= y <= int(height * 0.92)):
            return False, None, None, None

        left_x = model['left']['a'] * y + model['left']['b']
        right_x = model['right']['a'] * y + model['right']['b']
        margin = model['margin']
        if x < left_x - margin:
            return True, 'left', left_x, right_x
        if x > right_x + margin:
            return True, 'right', left_x, right_x
        return False, None, left_x, right_x

    def _sideline_line_contact_info(self, point, side, left_x, right_x, frame, ball_size=None):
        if point is None or frame is None or side not in ("left", "right"):
            return None
        if left_x is None or right_x is None:
            return None

        frame_height, frame_width = frame.shape[:2]
        x, y = int(point[0]), int(point[1])
        if x < 0 or y < 0 or x >= frame_width or y >= frame_height:
            return None

        outside_px = (float(left_x) - float(x)) if side == "left" else (float(x) - float(right_x))
        if outside_px <= 0.0:
            return {
                'contact': True,
                'outside_px': outside_px,
                'tolerance': 0.0,
                'white_ratio': 0.0,
                'white_pixels': 0,
                'reason': 'inside sideline',
            }

        area = float(ball_size if ball_size is not None else (self.ball_size or 0.0))
        if area < 60.0:
            return {
                'contact': False,
                'outside_px': outside_px,
                'tolerance': max(10.0, frame_width * 0.0035),
                'white_ratio': 0.0,
                'white_pixels': 0,
                'reason': f'ball area {area:.1f}px too small for line-contact slack',
            }

        y_factor = max(0.0, min(1.0, (float(y) - frame_height * 0.34) / max(1.0, frame_height * 0.34)))
        tolerance = area * 0.70 + 10.0 + y_factor * max(18.0, frame_width * 0.006)
        tolerance = max(28.0, min(tolerance, max(90.0, frame_width * 0.036)))
        if outside_px > tolerance:
            return {
                'contact': False,
                'outside_px': outside_px,
                'tolerance': tolerance,
                'white_ratio': 0.0,
                'white_pixels': 0,
                'reason': f'{outside_px:.1f}px outside tolerance {tolerance:.1f}px',
            }

        half_w = int(max(44, min(150, outside_px + 52.0)))
        half_h = int(max(24, min(64, frame_height * 0.020)))
        x1 = max(0, x - half_w)
        x2 = min(frame_width, x + half_w + 1)
        y1 = max(0, y - half_h)
        y2 = min(frame_height, y + half_h + 1)
        if x2 <= x1 or y2 <= y1:
            return None

        patch = frame[y1:y2, x1:x2]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white_mask = (
            (hsv_patch[:, :, 2] >= 165) &
            (hsv_patch[:, :, 1] <= 95)
        )
        white_pixels = int(np.count_nonzero(white_mask))
        patch_area = int(white_mask.size)
        white_ratio = white_pixels / float(max(1, patch_area))
        min_white_pixels = max(18, int(patch_area * 0.006))
        nearest_white_distance = None
        if white_pixels > 0:
            white_y, white_x = np.nonzero(white_mask)
            center_x = x - x1
            center_y = y - y1
            nearest_white_distance = float(np.min(np.hypot(white_x - center_x, white_y - center_y)))
        contact_radius = max(16.0, min(42.0, math.sqrt(max(area, 1.0)) * 2.5))
        white_reaches_ball = (
            nearest_white_distance is not None and
            nearest_white_distance <= contact_radius
        )
        contact = white_pixels >= min_white_pixels and white_reaches_ball
        return {
            'contact': contact,
            'outside_px': outside_px,
            'tolerance': tolerance,
            'white_ratio': white_ratio,
            'white_pixels': white_pixels,
            'nearest_white_distance': nearest_white_distance,
            'contact_radius': contact_radius,
            'reason': (
                'visible sideline paint reaches bounce'
                if contact else
                (
                    f'white line nearest distance {nearest_white_distance:.1f}px '
                    f'> contact radius {contact_radius:.1f}px'
                    if white_pixels >= min_white_pixels and nearest_white_distance is not None else
                    f'white line pixels {white_pixels} < {min_white_pixels}'
                )
            ),
        }

    def _sideline_line_contact_override(self, point, side, left_x, right_x, frame, ball_size=None, emit_debug=False):
        info = self._sideline_line_contact_info(point, side, left_x, right_x, frame, ball_size)
        if info is None or not info.get('contact'):
            return False
        if emit_debug:
            nearest_white_text = (
                f"{info['nearest_white_distance']:.1f}px/{info['contact_radius']:.1f}px"
                if info.get('nearest_white_distance') is not None and info.get('contact_radius') is not None
                else "n/a"
            )
            print(
                f"Frame {self.frame_count}: [SIDELINE LINE CONTACT] point={point} side={side} "
                f"outside={info['outside_px']:.1f}px tolerance={info['tolerance']:.1f}px "
                f"white={info['white_pixels']} ({info['white_ratio'] * 100:.1f}%) "
                f"nearest_white={nearest_white_text} "
                f"reason={info['reason']}"
            )
        return True

    def _line_contact_bounce_context(self, bounce_point, post_bounce_point, prev_dx, prev_dy, prev_speed, frame):
        frame_height, frame_width = frame.shape[:2]
        bx, by = int(bounce_point[0]), int(bounce_point[1])
        px, py = int(post_bounce_point[0]), int(post_bounce_point[1])
        current_area = float(self.ball_size or 0.0)
        incoming_dist = max(float(prev_speed or 0.0), math.hypot(float(prev_dx), float(prev_dy)))
        return {
            'origin': (bx, by),
            'expected': (max(0, min(frame_width - 1, px)), max(0, min(frame_height - 1, py))),
            'incoming_dx': float(prev_dx),
            'incoming_dy': float(prev_dy),
            'incoming_dist': incoming_dist,
            'min_launch_dist': max(5.0, incoming_dist * 0.30),
            'max_launch_dist': max(55.0, incoming_dist * 3.2),
            'min_upward': max(5.0, abs(float(prev_dy)) * 0.25),
            'expected_cap': max(28.0, min(95.0, incoming_dist * 2.2)),
            'ref_size': max(8.0, min(max(current_area * 1.10, current_area + 6.0), 90.0)),
        }

    def _build_singles_court_model(self, frame):
        frame_shape = frame.shape[:2]
        if (self._singles_court_model is not None and
                self._singles_court_frame_shape == frame_shape):
            return self._singles_court_model

        sideline_model = self._build_singles_sideline_model(frame)
        if sideline_model is None:
            self._singles_court_model = None
            self._singles_court_frame_shape = frame_shape
            return None

        height, width = frame_shape
        adjusted_points_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adjusted_court_points.txt")
        if os.path.exists(adjusted_points_path):
            manual_points = {}
            in_full_video_section = False
            with open(adjusted_points_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line.startswith("Full Video Frame Coordinates"):
                        in_full_video_section = True
                        continue
                    if in_full_video_section and line.startswith("Right Court Region Coordinates"):
                        break
                    if not in_full_video_section or not line.startswith("Point"):
                        continue
                    try:
                        left_part, right_part = line.split(":", 1)
                        point_num = int(left_part.split()[1])
                        coords_text = right_part.strip().strip("()")
                        x_text, y_text = coords_text.split(",", 1)
                        manual_points[point_num] = (float(x_text), float(y_text))
                    except (ValueError, IndexError):
                        continue

            if all(point_num in manual_points for point_num in (5, 6, 7, 8)):
                top_left = manual_points[5]
                top_right = manual_points[6]
                bottom_left = manual_points[7]
                bottom_right = manual_points[8]
                max_x = max(top_left[0], top_right[0], bottom_left[0], bottom_right[0])
                max_y = max(top_left[1], top_right[1], bottom_left[1], bottom_right[1])
                if max_x <= width * 1.02 and max_y <= height * 1.02:
                    far_dx = top_right[0] - top_left[0]
                    near_dx = bottom_right[0] - bottom_left[0]
                    left_dy = bottom_left[1] - top_left[1]
                    right_dy = bottom_right[1] - top_right[1]
                    far_baseline = {
                        'a': ((top_right[1] - top_left[1]) / far_dx) if abs(far_dx) > 1e-6 else 0.0,
                        'b': top_left[1] - (((top_right[1] - top_left[1]) / far_dx) if abs(far_dx) > 1e-6 else 0.0) * top_left[0],
                        'length': math.hypot(top_right[0] - top_left[0], top_right[1] - top_left[1]),
                        'y_mid': (top_left[1] + top_right[1]) / 2.0,
                    }
                    near_baseline = {
                        'a': ((bottom_right[1] - bottom_left[1]) / near_dx) if abs(near_dx) > 1e-6 else 0.0,
                        'b': bottom_left[1] - (((bottom_right[1] - bottom_left[1]) / near_dx) if abs(near_dx) > 1e-6 else 0.0) * bottom_left[0],
                        'length': math.hypot(bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]),
                        'y_mid': (bottom_left[1] + bottom_right[1]) / 2.0,
                    }
                    left_sideline = {
                        'a': ((bottom_left[0] - top_left[0]) / left_dy) if abs(left_dy) > 1e-6 else 0.0,
                        'b': top_left[0] - (((bottom_left[0] - top_left[0]) / left_dy) if abs(left_dy) > 1e-6 else 0.0) * top_left[1],
                    }
                    right_sideline = {
                        'a': ((bottom_right[0] - top_right[0]) / right_dy) if abs(right_dy) > 1e-6 else 0.0,
                        'b': top_right[0] - (((bottom_right[0] - top_right[0]) / right_dy) if abs(right_dy) > 1e-6 else 0.0) * top_right[1],
                    }
                    self._singles_court_model = {
                        'sidelines': sideline_model,
                        'left_sideline_manual': left_sideline,
                        'right_sideline_manual': right_sideline,
                        'far_baseline': far_baseline,
                        'near_baseline': near_baseline,
                        'polygon': np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
                        'polygon_margin': max(12.0, width * 0.0035),
                        'baseline_margin': max(18.0, width * 0.005),
                    }
                    self._singles_court_frame_shape = frame_shape
                    print(
                        f"  DEBUG: Singles court model built from adjusted points: "
                        f"far_y={far_baseline['y_mid']:.1f} near_y={near_baseline['y_mid']:.1f}"
                    )
                    return self._singles_court_model

        if not hasattr(self, 'net_area_y_min') or not hasattr(self, 'net_area_y_max'):
            self._singles_court_model = None
            self._singles_court_frame_shape = frame_shape
            return None

        net_y = float((self.net_area_y_min + self.net_area_y_max) / 2.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        min_line_length = max(260, int(width * 0.12))
        lines = cv2.HoughLinesP(
            white_mask,
            1,
            np.pi / 180,
            threshold=120,
            minLineLength=min_line_length,
            maxLineGap=24,
        )
        if lines is None:
            self._singles_court_model = None
            self._singles_court_frame_shape = frame_shape
            return None

        upper_candidates = []
        lower_candidates = []
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < min_line_length or abs(dy) > 18.0:
                continue
            y_mid = (y1 + y2) / 2.0
            a = dy / dx if abs(dx) > 1e-6 else 0.0
            b = y1 - a * x1
            entry = {
                'a': a,
                'b': b,
                'length': math.hypot(dx, dy),
                'y_mid': y_mid,
            }
            if height * 0.08 <= y_mid <= net_y - 55.0:
                upper_candidates.append(entry)
            elif net_y + 55.0 <= y_mid <= height * 0.95:
                lower_candidates.append(entry)

        if not upper_candidates or not lower_candidates:
            self._singles_court_model = None
            self._singles_court_frame_shape = frame_shape
            return None

        upper_long = sorted(upper_candidates, key=lambda entry: entry['length'], reverse=True)[:12]
        lower_long = sorted(lower_candidates, key=lambda entry: entry['length'], reverse=True)[:12]
        far_baseline = min(upper_long, key=lambda entry: entry['y_mid'])
        near_baseline = max(lower_long, key=lambda entry: entry['y_mid'])

        self._singles_court_model = {
            'sidelines': sideline_model,
            'far_baseline': far_baseline,
            'near_baseline': near_baseline,
            'baseline_margin': max(18.0, width * 0.005),
        }
        self._singles_court_frame_shape = frame_shape
        print(
            f"  DEBUG: Singles court model built: "
            f"far_y={far_baseline['y_mid']:.1f} near_y={near_baseline['y_mid']:.1f}"
        )
        return self._singles_court_model

    def _point_outside_singles_court(self, point, frame):
        outside_side, side, left_x, right_x = self._point_outside_singles_sidelines(point, frame)
        if outside_side:
            return True, f"{side} sideline", left_x, right_x, None, None

        model = self._build_singles_court_model(frame)
        if model is None:
            return False, None, left_x, right_x, None, None

        x, y = point
        polygon = model.get('polygon')
        if polygon is not None:
            signed_dist = cv2.pointPolygonTest(polygon, (float(x), float(y)), True)
            far_y = model['far_baseline']['a'] * x + model['far_baseline']['b']
            near_y = model['near_baseline']['a'] * x + model['near_baseline']['b']
            left_manual = model.get('left_sideline_manual')
            right_manual = model.get('right_sideline_manual')
            if left_manual is not None:
                left_x = left_manual['a'] * y + left_manual['b']
            if right_manual is not None:
                right_x = right_manual['a'] * y + right_manual['b']
            if signed_dist >= -float(model.get('polygon_margin', 0.0)):
                return False, None, left_x, right_x, far_y, near_y

            edge_labels = (
                ("far baseline", tuple(polygon[0]), tuple(polygon[1])),
                ("right sideline", tuple(polygon[1]), tuple(polygon[2])),
                ("near baseline", tuple(polygon[2]), tuple(polygon[3])),
                ("left sideline", tuple(polygon[3]), tuple(polygon[0])),
            )
            boundary = min(
                edge_labels,
                key=lambda entry: self._point_to_segment_distance(point, entry[1], entry[2])
            )[0]
            return True, boundary, left_x, right_x, far_y, near_y

        far_y = model['far_baseline']['a'] * x + model['far_baseline']['b']
        near_y = model['near_baseline']['a'] * x + model['near_baseline']['b']
        margin = model['baseline_margin']
        if y < far_y - margin:
            return True, "far baseline", left_x, right_x, far_y, near_y
        if y > near_y + margin:
            return True, "near baseline", left_x, right_x, far_y, near_y
        return False, None, left_x, right_x, far_y, near_y

    def _point_outside_top_singles_baseline(self, point, frame):
        model = self._build_singles_court_model(frame)
        if model is None or model.get('polygon') is None:
            return False, None
        x, y = point
        far_y = model['far_baseline']['a'] * x + model['far_baseline']['b']
        margin = max(14.0, float(model.get('baseline_margin', 0.0)))
        return y < (far_y - margin), far_y

    def _build_service_box_model(self, frame):
        frame_shape = frame.shape[:2]
        if (self._service_box_model is not None and
                self._service_box_frame_shape == frame_shape):
            return self._service_box_model

        if not hasattr(self, 'net_area_y_min') or not hasattr(self, 'net_area_y_max'):
            self._service_box_model = None
            self._service_box_frame_shape = frame_shape
            return None

        height, width = frame_shape
        net_y = float((self.net_area_y_min + self.net_area_y_max) / 2.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        min_line_length = max(220, int(width * 0.10))
        lines = cv2.HoughLinesP(
            white_mask,
            1,
            np.pi / 180,
            threshold=120,
            minLineLength=min_line_length,
            maxLineGap=24,
        )
        if lines is None:
            self._service_box_model = None
            self._service_box_frame_shape = frame_shape
            return None

        horizontal_candidates = []
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in raw]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < min_line_length or abs(dy) > 18.0:
                continue
            y_mid = (y1 + y2) / 2.0
            if self.serve_direction_dy < 0:
                if not (height * 0.06 <= y_mid <= net_y - 40.0):
                    continue
            elif self.serve_direction_dy > 0:
                if not (net_y + 40.0 <= y_mid <= height * 0.94):
                    continue
            else:
                continue
            a = dy / dx if abs(dx) > 1e-6 else 0.0
            b = y1 - a * x1
            horizontal_candidates.append({
                'a': a,
                'b': b,
                'length': math.hypot(dx, dy),
                'y_mid': y_mid,
            })

        if not horizontal_candidates:
            self._service_box_model = None
            self._service_box_frame_shape = frame_shape
            return None

        top_long = sorted(horizontal_candidates, key=lambda entry: entry['length'], reverse=True)[:12]
        if self.serve_direction_dy < 0:
            service_line = max(top_long, key=lambda entry: entry['y_mid'])
        else:
            service_line = min(top_long, key=lambda entry: entry['y_mid'])

        self._service_box_model = {
            'net_y': net_y,
            'service_line': service_line,
            'service_margin': max(28.0, width * 0.0100),
            'net_margin': max(24.0, width * 0.0080),
        }
        self._service_box_frame_shape = frame_shape
        print(
            f"  DEBUG: Service box model built: net_y={net_y:.1f} "
            f"service_y_mid={service_line['y_mid']:.1f}"
        )
        return self._service_box_model

    def _point_in_target_service_box(self, point, frame, target_side):
        if target_side not in ("left", "right"):
            return None, None

        outside, side, left_x, right_x = self._point_outside_singles_sidelines(point, frame)
        if outside:
            return False, f"Serve bounce outside singles court ({side} sideline)"
        if left_x is None or right_x is None:
            return None, None

        model = self._build_service_box_model(frame)
        if model is None:
            return None, None

        x, y = point
        center_x = (left_x + right_x) / 2.0
        service_y = model['service_line']['a'] * x + model['service_line']['b']
        service_margin = model['service_margin']
        net_margin = model['net_margin']
        net_y = model['net_y']

        if self.serve_direction_dy < 0:
            in_vertical = (service_y - service_margin) <= y <= (net_y + net_margin)
        elif self.serve_direction_dy > 0:
            in_vertical = (net_y - net_margin) <= y <= (service_y + service_margin)
        else:
            return None, None

        if target_side == "left":
            in_half = x <= center_x
        else:
            in_half = x >= center_x

        if in_vertical and in_half:
            return True, None
        return False, f"Serve bounce outside {target_side} service box"

    def _evaluate_target_service_box_fresh(self, point, frame, target_side):
        """Evaluate a service bounce against freshly-read court lines for the current frame."""
        if target_side not in ("left", "right"):
            return None, None, {}

        saved_service_model = self._service_box_model
        saved_service_shape = self._service_box_frame_shape
        saved_sideline_model = self._singles_sideline_model
        saved_sideline_shape = self._singles_sideline_frame_shape
        try:
            self._service_box_model = None
            self._service_box_frame_shape = None
            self._singles_sideline_model = None
            self._singles_sideline_frame_shape = None

            outside, side, left_x, right_x = self._point_outside_singles_sidelines(point, frame)
            geometry = {
                'left_x': left_x,
                'right_x': right_x,
                'outside_side': side,
            }
            if outside:
                return False, f"Serve bounce outside singles court ({side} sideline)", geometry
            if left_x is None or right_x is None:
                return None, None, geometry

            model = self._build_service_box_model(frame)
            if model is None:
                return None, None, geometry

            x, y = point
            center_x = (left_x + right_x) / 2.0
            service_y = model['service_line']['a'] * x + model['service_line']['b']
            service_margin = model['service_margin']
            net_margin = model['net_margin']
            net_y = model['net_y']
            geometry.update({
                'center_x': center_x,
                'service_y': service_y,
                'service_margin': service_margin,
                'net_margin': net_margin,
                'net_y': net_y,
            })

            if self.serve_direction_dy < 0:
                in_vertical = (service_y - service_margin) <= y <= (net_y + net_margin)
            elif self.serve_direction_dy > 0:
                in_vertical = (net_y - net_margin) <= y <= (service_y + service_margin)
            else:
                return None, None, geometry

            in_half = x <= center_x if target_side == "left" else x >= center_x
            if in_vertical and in_half:
                return True, None, geometry
            return False, f"Serve bounce outside {target_side} service box", geometry
        finally:
            self._service_box_model = saved_service_model
            self._service_box_frame_shape = saved_service_shape
            self._singles_sideline_model = saved_sideline_model
            self._singles_sideline_frame_shape = saved_sideline_shape

    def _serve_bounce_out_from_turn_candidate(self, new_pos, frame, dx, dy, angle_jump, velocity):
        """Catch first-serve bounce jumps that miss the target service box."""
        if not getattr(self, '_awaiting_serve_bounce', False):
            return None
        if self.point_start_frame_internal is None:
            return None
        frames_since_start = self.frame_count - self.point_start_frame_internal
        if frames_since_start < 3 or frames_since_start > 45:
            return None
        if self.ball_center is None or self.last_motion is None:
            return None

        bounce_point = self.ball_center
        cx, cy = new_pos
        upward_progress = bounce_point[1] - cy
        incoming_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        if velocity < 10.0 or dy > -8.0 or upward_progress < 8.0:
            return None
        if angle_jump < 45.0 and not (incoming_dy >= 1.0 and dy <= -12.0):
            return None

        target_side = getattr(self, '_point_target_service_side', None)
        service_ok, service_reason, geometry = self._evaluate_target_service_box_fresh(
            bounce_point, frame, target_side
        )
        if service_ok is not False:
            return None

        left_x = geometry.get('left_x')
        right_x = geometry.get('right_x')
        center_x = geometry.get('center_x')
        if left_x is None or right_x is None or center_x is None:
            return None

        frame_width = frame.shape[1]
        boundary_margin = max(80.0, frame_width * 0.05)
        line_slack = max(45.0, frame_width * 0.012)
        bx = float(bounce_point[0])
        if target_side == "right":
            target_half = bx >= center_x
            near_target_boundary = (
                bx >= frame_width * 0.60 and
                (right_x - bx) <= boundary_margin and
                bx <= (right_x + line_slack)
            )
        else:
            target_half = bx <= center_x
            near_target_boundary = (
                bx <= frame_width * 0.40 and
                (bx - left_x) <= boundary_margin and
                bx >= (left_x - line_slack)
            )
        if not (target_half and near_target_boundary):
            return None

        return {
            'point': bounce_point,
            'reason': service_reason or f"Serve bounce outside {target_side} service box",
            'target_side': target_side,
            'geometry': geometry,
        }

    def _serve_bounce_in_from_turn_candidate(self, new_pos, frame, dx, dy, angle_jump, velocity):
        """Catch soft first-serve bounce turns that land inside the target service box."""
        if not getattr(self, '_awaiting_serve_bounce', False):
            return None
        if self.point_start_frame_internal is None:
            return None
        frames_since_start = self.frame_count - self.point_start_frame_internal
        if frames_since_start < 3 or frames_since_start > 45:
            return None
        if self.ball_center is None or self.last_motion is None:
            return None

        bounce_point = self.ball_center
        incoming_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        previous_dy = float(self.prev_motion.get('dy', 0.0) or 0.0) if self.prev_motion else 0.0
        upward_progress = float(bounce_point[1] - new_pos[1])
        recent_descent = incoming_dy >= 2.0 and (previous_dy >= 1.0 or incoming_dy >= 4.0)
        soft_vertical_reversal = recent_descent and dy <= -3.0 and upward_progress >= 3.0
        sharp_turn = angle_jump >= 45.0 and incoming_dy >= 1.0 and dy <= -3.0
        if velocity < 5.0 or not (soft_vertical_reversal or sharp_turn):
            return None

        target_side = getattr(self, '_point_target_service_side', None)
        service_ok, _, geometry = self._evaluate_target_service_box_fresh(
            bounce_point, frame, target_side
        )
        same_target_half = False
        center_x = geometry.get('center_x')
        if center_x is None and hasattr(self, 'serve_area_x_min') and hasattr(self, 'serve_area_x_max'):
            center_x = (self.serve_area_x_min + self.serve_area_x_max) / 2.0
        if center_x is not None:
            if target_side == "left":
                same_target_half = bounce_point[0] <= center_x
            elif target_side == "right":
                same_target_half = bounce_point[0] >= center_x
        net_y = geometry.get('net_y')
        net_margin = geometry.get('net_margin', 0.0)
        near_target_net_band = False
        if net_y is not None and same_target_half:
            extra_net_margin = max(60.0, frame.shape[0] * 0.035)
            if self.serve_direction_dy < 0:
                near_target_net_band = (net_y - extra_net_margin) <= bounce_point[1] <= (net_y + net_margin)
            elif self.serve_direction_dy > 0:
                near_target_net_band = (net_y - net_margin) <= bounce_point[1] <= (net_y + extra_net_margin)
        if not near_target_net_band and same_target_half:
            net_geometry = self._net_contact_geometry(bounce_point)
            near_target_net_band = (
                net_geometry is not None and
                (net_geometry['inside'] or net_geometry['near_top_tape'])
            )
        if service_ok is not True and not near_target_net_band:
            return None

        return {
            'point': bounce_point,
            'target_side': target_side,
            'geometry': geometry,
            'near_target_net_band': near_target_net_band,
            'soft_vertical_reversal': soft_vertical_reversal,
            'sharp_turn': sharp_turn,
        }

    def _detect_out_of_court_bounce(self, ball_position, frame):
        if self.prev_motion is None or self.last_motion is None:
            return False, None
        if not hasattr(self, 'net_area_y_min'):
            return False, None
        if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
            return False, None
        if (self.frame_count - getattr(self, '_last_racket_contact_frame', -1000000)) <= 1:
            return False, None
        if (
            getattr(self, '_awaiting_serve_bounce', False) and
            self.point_start_frame_internal is not None and
            (self.frame_count - self.point_start_frame_internal) <= 12 and
            getattr(self, '_serve_contact_grace_frames', 0) > 0
        ):
            return False, None

        x, y = ball_position
        curr_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        curr_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        prev_dx = float(self.prev_motion.get('dx', 0.0) or 0.0)
        prev_dy = float(self.prev_motion.get('dy', 0.0) or 0.0)
        curr_speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        prev_speed = float(self.prev_motion.get('distance', 0.0) or 0.0)
        if prev_speed < 20.0 or curr_speed < 12.0:
            return False, None

        prev_pos = (int(round(x - curr_dx)), int(round(y - curr_dy)))
        prev_pos_outside, prev_side, prev_left_x, prev_right_x = self._point_outside_singles_sidelines(prev_pos, frame)
        curr_pos_outside, curr_side, curr_left_x, curr_right_x = self._point_outside_singles_sidelines(ball_position, frame)
        if not prev_pos_outside and not curr_pos_outside:
            return False, None

        near_side_min_y = int(self.net_area_y_min + 60)
        if prev_pos[1] < near_side_min_y and y < near_side_min_y:
            return False, None

        prev_dir = self.prev_motion.get('direction_deg')
        curr_dir = self.last_motion.get('direction_deg')
        angle_diff = 0.0
        if prev_dir is not None and curr_dir is not None:
            delta = abs(curr_dir - prev_dir) % 360
            angle_diff = min(delta, 360 - delta)

        vertical_reversal = prev_dy >= 18.0 and curr_dy <= -12.0
        sharp_turn = angle_diff >= 95.0
        if not (vertical_reversal or sharp_turn):
            return False, None

        side = prev_side or curr_side
        bounce_point = prev_pos if prev_pos_outside else ball_position
        left_x = prev_left_x if prev_pos_outside else curr_left_x
        right_x = prev_right_x if prev_pos_outside else curr_right_x
        if left_x is None or right_x is None:
            return False, None

        if self._sideline_line_contact_override(
                bounce_point, side, left_x, right_x, frame, self.ball_size,
                emit_debug=False):
            line_context = self._line_contact_bounce_context(
                bounce_point, ball_position, prev_dx, prev_dy, prev_speed, frame
            )
            registered = self._register_ground_bounce_from_context(
                line_context, frame, source_label="sideline line contact"
            )
            if not registered:
                print(
                    f"Frame {self.frame_count}: [SIDELINE LINE CONTACT] point={bounce_point} side={side} "
                    f"treated as in-court bounce"
                )
            return False, None

        print(
            f"Frame {self.frame_count}: [OUT-BOUNCE] bounce_point={bounce_point} side={side} "
            f"court_x={left_x:.1f}-{right_x:.1f} prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
            f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) angle_diff={angle_diff:.1f}"
        )
        return True, f"Ball bounced out of court ({side} sideline)"
    
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
        
        if not auto_play and not self.headless:
            cv2.namedWindow("Tennis Game Tracker", cv2.WINDOW_NORMAL)
        
        # Store scale factor for mouse coordinate conversion
        scale_factor = 1.0
        current_frame = None
        
        def reset_tracking_state(hold_end_marker=False, end_position=None):
            if hold_end_marker:
                self._capture_point_end_visuals(end_position=end_position)
            self.tracking = False
            self.ball_stopped = False
            self.ball_center = None
            self.ball_size = None
            self.ball_hsv = None
            self.ball_velocity_history = []
            # Restore original HSV when leaving TRACKING_POINT
            if self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
                self.hsv_lower = self.primary_hsv_lower.copy()
                self.hsv_upper = self.primary_hsv_upper.copy()
            self.last_motion = None
            self.prev_motion = None
            self.last_direction = None
            self.direction_change_points = []
            self.direction_change_events = []
            self.recent_bounce_markers = []
            self.net_contact_points = []
            self.motion_debug_vectors = []
            self._last_impact_marker_frame = -1000000
            self._last_impact_marker_pos = None
            self._last_impact_marker_kind = None
            self.direction_change_streak = 0
            self.edge_wait = False
            self.near_edge = False
            self.using_alt_hsv = False
            self.using_alt2_hsv = False
            self.using_alt3_hsv = False
            self.using_alt6_hsv = False
            self.focus_loss_active = False
            self.stuck_frame_count = 0
            self.point_start_frame_internal = None
            self._serve_contact_grace_frames = 0
            self._rally_contact_grace_frames = 0
            self._rally_contact_large_ref_frames = 0
            self._rally_contact_ref_size = None
            self._rally_contact_origin = None
            self._rally_contact_expected = None
            self._rally_contact_progress = 0.0
            self._rally_contact_high_launch_until_frame = -1
            self._last_racket_contact_frame = -1000000
            self._last_racket_contact_point = None
            self._ground_bounce_grace_frames = 0
            self._ground_bounce_ref_size = None
            self._ground_bounce_origin = None
            self._ground_bounce_expected = None
            self._ground_bounce_progress = 0.0
            self._ground_bounce_debug_history = []
            self.ground_bounce_count = 0
            self.last_ground_bounce_frame = -1000000
            self._contact_recovery_frames = 0
            self._upper_exit_wait_frames = 0
            self._top_return_wait_frames = 0
            self._top_return_anchor = None
            self._top_return_origin_frame = -1
            self._top_return_reentry_grace_frames = 0
            self._top_return_mode = None
            self._top_return_exit_dx = 0.0
            self._recent_offscreen_return_frame = -1000000
            self._recent_return_bounce_recover_frame = -1000000
            self._recent_racket_rebound_bounce_frame = -1000000
            self._late_contact_prior_bounce_until_frame = -1000000
            self._late_contact_prior_bounce_count = 0
            self._back_return_wait_frames = 0
            self._back_return_anchor = None
            self._back_return_origin_frame = -1
            self._back_return_reentry_grace_frames = 0
            self._back_return_timed_out = False
            self._upper_slow_arc_until_frame = -1
            self._upper_fence_fall_frames = 0
            self._prev_serve_gray = None
            self._ignored_serve_positions = []
            self.waiting_serve_candidate = None
            self.waiting_serve_candidate_frame = -1
            self._last_detected_serve_candidate = None
            self._pending_rally_end_reason = None
            self._pending_rally_end_frame = -1
            self._awaiting_serve_bounce = False
            self._point_serve_start_side = None
            self._point_target_service_side = None
            self._last_direction_change_frame = -1000000
            self._last_direction_change_point = None
            self._last_direction_change_angle = 0.0
            self._last_direction_change_vertical_reversal = False
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

        def low_to_up_serve_toss_context(history):
            if len(history) < 4:
                return None
            if not hasattr(self, 'serve_area_y_min') or not hasattr(self, 'serve_area_y_max'):
                return None

            serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
            low_y_min = self.serve_area_y_min + int(serve_height * 0.45)
            min_rise = max(80.0, serve_height * 0.18)
            latest_index = len(history) - 1
            latest = history[-1]
            min_y = min(p[1] for p in history)
            min_index = min(range(len(history)), key=lambda idx: history[idx][1])

            # Serve start is only the live upward toss, not the later fall after apex.
            if latest_index - min_index > 1:
                return None
            if latest[1] > min_y + max(16.0, serve_height * 0.04):
                return None

            pre_apex = history[:min_index + 1]
            if len(pre_apex) < 4:
                return None
            low_index = max(range(len(pre_apex)), key=lambda idx: pre_apex[idx][1])
            low_pos = pre_apex[low_index]
            if low_index >= min_index:
                return None
            if low_pos[1] < low_y_min:
                return None

            rise = float(low_pos[1] - history[min_index][1])
            if rise < min_rise:
                return None

            up_steps = []
            for i in range(low_index + 1, min_index + 1):
                dy = history[i][1] - history[i - 1][1]
                up_amount = -float(dy)
                if up_amount >= max(6.0, serve_height * 0.015):
                    up_steps.append(up_amount)
            if len(up_steps) < 2:
                return None
            if len(up_steps) < 3 and rise < max(120.0, serve_height * 0.25):
                return None

            recent_dy = history[-1][1] - history[-2][1]
            if -float(recent_dy) < max(4.0, serve_height * 0.010):
                return None

            return {
                'low_pos': low_pos,
                'apex_pos': history[min_index],
                'rise': rise,
                'up_steps': len(up_steps),
                'latest_pos': latest,
                'low_index': low_index,
                'apex_index': min_index,
            }

        def log_tracking_start_position():
            if self.ball_center is None:
                return
            size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
            vel = self.last_motion['distance'] if self.last_motion else 0.0
            vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
            print(f"Frame {self.frame_count}: Ball tracked at {self.ball_center} - Size: {size_text}")
            print(f"[TRACK] f{self.frame_count}: pos={self.ball_center} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")
        
        # Modified mouse callback that converts resized coordinates to original
        def mouse_callback_with_scale(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert clicked coordinates from resized frame to original frame
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                self.print_click_hsv_neighborhood(
                    current_frame,
                    (orig_x, orig_y),
                    display_point=(x, y),
                    scale=scale_factor,
                )
                # Analyze ball at original coordinates
                self.analyze_ball_at_point(current_frame, (orig_x, orig_y))
        
        if not auto_play and not self.headless:
            cv2.setMouseCallback("Tennis Game Tracker", mouse_callback_with_scale)
        
        # Start from configured frame to wait for first serve
        print(f"\nStarting from frame {self.start_frame} to wait for first serve...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.frame_count = self.start_frame
        
        # Load HSV values from config file (or use defaults if not present)
        hsv_config = self.load_hsv_config()
        
        # Check if net area needs to be marked (first time setup)
        if (not self.disable_behind_net_mode) and hsv_config and "net_area_y_min" not in hsv_config:
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
            self.current_game_state = game_state
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
                self.using_alt3_hsv = False
                self.using_alt6_hsv = False
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
                        serve_position_history.append(candidate)
                        serve_candidate_details_history.append({'pos': candidate})
                        if len(serve_position_history) > 20:
                            serve_position_history = serve_position_history[-20:]
                        if len(serve_candidate_details_history) > 20:
                            serve_candidate_details_history = serve_candidate_details_history[-20:]
                        self.waiting_serve_candidate = candidate
                        self.waiting_serve_candidate_frame = self.frame_count
                        print(f"Frame {self.frame_count}: Serve candidate {candidate} (holding for confirmation)")
                else:
                    early_toss_context = low_to_up_serve_toss_context(serve_position_history)
                    if serve_tracking_frames >= 5 and last_serve_candidate is not None and early_toss_context is not None:
                        print(f"Frame {self.frame_count}: Serve exited area, entering TRACKING_POINT from {last_serve_candidate}")
                        game_state = "TRACKING_POINT"
                        point_start_frame = self.frame_count
                        self.point_start_frame_internal = self.frame_count
                        self.tracking = True
                        self.ball_center = last_serve_candidate
                        self._start_point_context(last_serve_candidate)
                        self.waiting_serve_candidate = None
                        self.waiting_serve_candidate_frame = -1
                        log_tracking_start_position()
                    else:
                        clear_waiting_serve_history()
            
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
                        has_upward_motion = False
                        has_valid_vertical = False
                        
                        for i in range(-3, 0):
                            pair_dx = scan_position_history[i][0] - scan_position_history[i-1][0]
                            pair_dy = scan_position_history[i][1] - scan_position_history[i-1][1]
                            signed_dx = self._signed_serve_dx(pair_dx)
                            
                            # Check for upward motion (negative dy means moving up in image coords)
                            if pair_dy < -3:  # Significant upward movement
                                has_upward_motion = True
                            
                            # Valid vertical: either upward OR downward (not staying flat)
                            if abs(pair_dy) > 3:
                                has_valid_vertical = True
                            
                            if signed_dx < 15:
                                all_forward = False
                                break
                            min_signed_dx = min(min_signed_dx, signed_dx)
                        
                        # Serve must have upward motion initially and valid vertical component
                        # (to avoid false serves that are just horizontal movement at a single height)
                        if (all_forward and min_signed_dx > 25 and has_upward_motion and has_valid_vertical and
                                low_to_up_serve_toss_context(scan_position_history) is not None):
                            print(f"\n{'='*70}")
                            print(f"SERVE DETECTED at frame {self.frame_count}!")
                            print(f"Ball position: {potential_serve}")
                            print(f"Starting to track the ball...")
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
                            self._start_point_context(potential_serve)
                            scan_position_history = []
                            game_state = "TRACKING_POINT"
                            log_tracking_start_position()
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
                    top_timeout_hold = (
                        self._top_return_wait_active() or
                        getattr(self, '_top_return_reentry_grace_frames', 0) > 0
                    )
                    back_timeout_hold = (
                        self._back_return_wait_active() or
                        getattr(self, '_back_return_reentry_grace_frames', 0) > 0
                    )
                    recent_return_hold = self._recent_offscreen_return_hold_active(window_frames=8)
                    recent_bounce_hold = self._recent_return_bounce_recover_active(window_frames=2)
                    frames_since_seen = (
                        self.frame_count - self.last_seen_frame
                        if self.last_seen_frame is not None else 1000000
                    )
                    last_motion_distance = (
                        float(self.last_motion.get('distance', 0.0) or 0.0)
                        if self.last_motion is not None else 0.0
                    )
                    last_candidate_motion_frame = int(
                        getattr(self, '_last_tracked_candidate_motion_frame', -1000000)
                    )
                    last_candidate_motion_mean = float(
                        getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
                    )
                    last_candidate_motion_max = float(
                        getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
                    )
                    last_candidate_frame = int(
                        getattr(self, '_last_tracked_candidate_frame', -1000000)
                    )
                    recent_visible_candidate_hold = (
                        self.tracking and
                        self.ball_center is not None and
                        frames_since_seen <= 3 and
                        self.stuck_frame_count < 5 and
                        last_candidate_motion_frame == self.last_seen_frame and
                        (self.ball_size is None or self.ball_size <= 80.0) and
                        last_candidate_motion_mean >= 10.0 and
                        last_candidate_motion_max >= 80.0
                    )
                    recent_selected_candidate_hold = (
                        self.tracking and
                        self.ball_center is not None and
                        frames_since_seen <= 3 and
                        self.stuck_frame_count < 5 and
                        last_candidate_frame == self.last_seen_frame and
                        (self.ball_size is None or self.ball_size <= 80.0) and
                        self.ball_center[1] <= max(500, int(frame.shape[0] * 0.25))
                    )
                    upper_visible_slow_flight_hold = (
                        self.tracking and
                        self.ball_center is not None and
                        frames_since_seen <= 3 and
                        self.stuck_frame_count < 5 and
                        self.ball_center[1] <= max(145, int(frame.shape[0] * 0.075)) and
                        (self.ball_size is None or self.ball_size <= 25.0) and
                        last_motion_distance >= 0.75 and
                        (
                            (self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)) <= 28 or
                            self._recent_offscreen_return_hold_active(window_frames=40)
                        )
                    )
                    active_tracking_hold = (
                        self.tracking and
                        self.ball_center is not None and
                        (
                            (
                                frames_since_seen <= 8 and
                                self.stuck_frame_count < 8 and
                                last_motion_distance >= 3.0
                            ) or
                            (
                                frames_since_seen <= 3 and
                                self.stuck_frame_count < 5 and
                                last_motion_distance >= 0.75
                            ) or
                            recent_visible_candidate_hold or
                            recent_selected_candidate_hold or
                            upper_visible_slow_flight_hold
                        )
                    )
                    hard_timeout = _max_point_frames + max(120, int(_max_point_frames * 0.5))
                    timeout_hold = (
                        top_timeout_hold or back_timeout_hold or
                        recent_return_hold or recent_bounce_hold
                    )
                    if active_tracking_hold or (timeout_hold and dur <= hard_timeout):
                        if top_timeout_hold or back_timeout_hold:
                            print(f"Frame {self.frame_count}: delaying point timeout while waiting for offscreen return")
                        elif recent_bounce_hold:
                            last_bounce_frame = getattr(self, '_recent_return_bounce_recover_frame', self.frame_count)
                            frames_since_bounce = max(0, self.frame_count - last_bounce_frame)
                            print(
                                f"Frame {self.frame_count}: delaying point timeout after recent bounce recover "
                                f"({frames_since_bounce}f ago)"
                            )
                        elif recent_return_hold:
                            last_return_frame = getattr(self, '_recent_offscreen_return_frame', self.frame_count)
                            frames_since_return = max(0, self.frame_count - last_return_frame)
                            print(
                                f"Frame {self.frame_count}: delaying point timeout after recent offscreen return "
                                f"re-entry ({frames_since_return}f ago)"
                            )
                        else:
                            print(
                                f"Frame {self.frame_count}: delaying point timeout while ball is still actively tracked "
                                f"(seen {frames_since_seen}f ago, motion={last_motion_distance:.1f}px)"
                            )
                    else:
                        print(f"[POINT_END] f{self.frame_count}: reason=POINT_TIMEOUT duration={dur}f — returning to serve detection")
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True)

                # Track ball through the point
                tracked_position = None
                if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
                    prev_ball_center = self.ball_center
                    prev_stuck = self.stuck_frame_count
                    prev_top_return_wait = self._top_return_wait_active()
                    prev_back_return_wait = self._back_return_wait_active()
                    tracked_position = self.track_ball_in_frame(frame)
                    # Reject any position that jumps impossibly far in one frame (false positive).
                    # When the tracker is in re-acquisition mode (stuck >= 5 before the call), allow
                    # a larger jump because the ball may have traveled far while lost.
                    if tracked_position and prev_ball_center:
                        jump = math.hypot(tracked_position[0] - prev_ball_center[0],
                                          tracked_position[1] - prev_ball_center[1])
                        motion_reacq_this_frame = (
                            getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count and
                            getattr(self, '_last_motion_reacq_pos', None) == tracked_position
                        )
                        # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
                        # a racket hit can send the ball 800+ px in one frame, so we use 1500px
                        # to let motion-based re-acquisition recover across the full court.
                        max_jump = 1500 if (
                            prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
                            motion_reacq_this_frame
                        ) else 400
                        if jump > max_jump:
                            print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
                            self.ball_center = prev_ball_center
                            self.stuck_frame_count = max(self.stuck_frame_count, prev_stuck + 1)
                            tracked_position = prev_ball_center
                if tracked_position:
                    vel = self.last_motion['distance'] if self.last_motion else 0
                    size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
                    vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
                    print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
                    print(f"[TRACK] f{self.frame_count}: pos={tracked_position} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")

                    pending_reason = getattr(self, '_pending_rally_end_reason', None)
                    if pending_reason:
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        print(f"Frame {self.frame_count}: POINT ENDED - {pending_reason}")
                        print(f"Point duration: {dur} frames")
                        print(f"[POINT_END] f{self.frame_count}: reason={pending_reason} duration={dur}f pos={tracked_position} vel={vel:.1f}px vel_hist={vel_hist_tail}")
                        if "bounced twice" in pending_reason.lower():
                            self._serve_scan_block_until_frame = max(
                                getattr(self, '_serve_scan_block_until_frame', -1),
                                self.frame_count + 60,
                            )
                            print(
                                f"Frame {self.frame_count}: [SERVE_SCAN_COOLDOWN] "
                                f"blocking serve detection until f{self._serve_scan_block_until_frame}"
                            )
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True, end_position=tracked_position)
                        continue

                    # Stuck-ball timeout: if ball hasn't moved for 15+ frames, end point
                    if self.stuck_frame_count >= 15 and not self._top_return_wait_active():
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        print(f"Frame {self.frame_count}: POINT ENDED - Ball stuck for {self.stuck_frame_count} frames")
                        print(f"Point duration: {dur} frames")
                        print(f"[POINT_END] f{self.frame_count}: reason=STUCK_TIMEOUT stuck={self.stuck_frame_count} duration={dur}f pos={tracked_position}")
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True, end_position=tracked_position)
                    elif self.stuck_frame_count >= 15 and self._top_return_wait_active():
                        self.stuck_frame_count = 4
                        print(f"Frame {self.frame_count}: [TOP-RETURN WAIT] suppressing stuck timeout while waiting for delayed re-entry")
                    else:
                        # Check if point has ended
                        point_ended, reason = self.detect_point_end(tracked_position, frame)
                        if point_ended:
                            point_end_frame = self.frame_count
                            dur = point_end_frame - point_start_frame if point_start_frame else 0
                            print(f"Frame {self.frame_count}: POINT ENDED - {reason}")
                            print(f"Point duration: {dur} frames")
                            print(f"[POINT_END] f{self.frame_count}: reason={reason} duration={dur}f pos={tracked_position} vel={vel:.1f}px vel_hist={vel_hist_tail}")
                            reason_lower = reason.lower()
                            if "bounced twice" in reason_lower:
                                self._serve_scan_block_until_frame = max(
                                    getattr(self, '_serve_scan_block_until_frame', -1),
                                    self.frame_count + 60,
                                )
                                print(
                                    f"Frame {self.frame_count}: [SERVE_SCAN_COOLDOWN] "
                                    f"blocking serve detection until f{self._serve_scan_block_until_frame}"
                                )
                            wait_for_next_serve = any(
                                token in reason_lower
                                for token in ("net", "bounce", "stopped", "lost", "fence")
                            )
                            if wait_for_next_serve:
                                if "net" in reason.lower():
                                    self.net_contact_points.append(tracked_position)
                                game_state = "WAITING_FOR_SERVE"
                            else:
                                game_state = "POINT_ENDED"
                            reset_tracking_state(hold_end_marker=True, end_position=tracked_position)
                        else:
                            print(f"Frame {self.frame_count}: Ball tracking continued")
                else:
                    # Ball lost - might be end of point
                    if getattr(self, '_back_return_timed_out', False):
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        print(f"Frame {self.frame_count}: POINT ENDED - Ball lost after back-return timeout")
                        print(f"Point duration: {dur} frames")
                        print(f"[POINT_END] f{self.frame_count}: reason=BACK_RETURN_TIMEOUT duration={dur}f")
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True)
                        continue
                    recent_return_hold = self._recent_offscreen_return_hold_active(window_frames=8)
                    recent_bounce_hold = self._recent_return_bounce_recover_active(window_frames=2)
                    grace_limit = 45 if point_start_frame and self.frame_count <= (self.start_frame + 45) else 30
                    if recent_return_hold:
                        last_return_frame = getattr(self, '_recent_offscreen_return_frame', self.frame_count)
                        frames_since_return = max(0, self.frame_count - last_return_frame)
                        print(
                            f"Frame {self.frame_count}: delaying lost-ball point end after recent offscreen return "
                            f"re-entry ({frames_since_return}f ago)"
                        )
                        continue
                    if recent_bounce_hold:
                        last_bounce_frame = getattr(self, '_recent_return_bounce_recover_frame', self.frame_count)
                        frames_since_bounce = max(0, self.frame_count - last_bounce_frame)
                        print(
                            f"Frame {self.frame_count}: delaying lost-ball point end after recent bounce recover "
                            f"({frames_since_bounce}f ago)"
                        )
                        continue
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
                        reset_tracking_state(hold_end_marker=True)
            
            elif game_state == "POINT_ENDED":
                # Wait a few frames then start scanning for next serve
                if self.frame_count - point_end_frame > 60:  # Wait 2 seconds (60 frames at 30fps)
                    print(f"Frame {self.frame_count}: Starting to scan for next serve...")
                    game_state = "SCANNING_FOR_SERVE"
                    point_start_frame = None
                    point_end_frame = None
                    # Restore original HSV when transitioning to SCANNING_FOR_SERVE
                    if self.primary_hsv_lower is not None and self.primary_hsv_upper is not None:
                        self.hsv_lower = self.primary_hsv_lower.copy()
                        self.hsv_upper = self.primary_hsv_upper.copy()
                # If we are still within early grace and see the ball, resume tracking
                elif (self.frame_count <= (self.start_frame + 15)) or getattr(self, 'edge_wait', False) or (self.last_seen_frame and self.frame_count - self.last_seen_frame <= 200):
                    candidate = self.track_ball_in_frame(frame, allow_inactive=True)
                    if candidate is not None:
                        print(f"Frame {self.frame_count}: Ball re-found during grace, resuming TRACKING_POINT at {candidate}")
                        game_state = "TRACKING_POINT"
                        point_start_frame = self.frame_count
                        self.tracking = True
                elif 1165 <= self.frame_count <= 1185:
                    print(f"[STATE_STUCK] f{self.frame_count}: Still in POINT_ENDED, waited {self.frame_count - point_end_frame}f (need 60f)")
        
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
                    serve_toss_context = low_to_up_serve_toss_context(serve_position_history)
                    if serve_toss_context is not None:
                        p1 = serve_position_history[-2]
                        p2 = serve_position_history[-1]
                        _dx = p2[0] - p1[0]
                        _dy = p2[1] - p1[1]
                        _dist = _math.hypot(_dx, _dy)
                        _dir = _math.degrees(_math.atan2(_dy, _dx))
                        self.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                        self.last_delta = (_dx, _dy)
                        self.ball_velocity_history = [_dist]
                        print(
                            f"[TRACKING_START] f{self.frame_count}: serve-toss-rise start={potential_serve} "
                            f"low={serve_toss_context['low_pos']} apex={serve_toss_context['apex_pos']} "
                            f"rise={serve_toss_context['rise']:.0f}px up_steps={serve_toss_context['up_steps']}"
                        )
                        self.ball_center = potential_serve
                        self.tracking = True
                        self.ball_stopped = False
                        self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                        self.initial_ball_position = serve_toss_context['low_pos']
                        self.ball_size = None
                        self.ball_hsv = None
                        seed_tracking_from_serve_history(potential_serve)
                        self.stuck_frame_count = 0
                        point_start_frame = self.frame_count
                        self.point_start_frame_internal = self.frame_count
                        self._start_point_context(potential_serve)
                        self.waiting_serve_candidate = None
                        self.waiting_serve_candidate_frame = -1
                        game_state = "TRACKING_POINT"
                        log_tracking_start_position()
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
                        serve_area_height = self.serve_area_y_max - self.serve_area_y_min
                        toss_drop_threshold = max(50, int(serve_area_height * 0.12))
                        toss_start_reason = "toss-in-flight"
                        recent_candidate_areas = [
                            float(detail.get('area', 0.0))
                            for detail in serve_candidate_details_history[-8:]
                            if isinstance(detail, dict)
                        ]
                        recent_candidate_area_max = max(recent_candidate_areas, default=0.0)
                        large_near_court_candidate = recent_candidate_area_max >= max(
                            220.0,
                            float(getattr(self, 'serve_ball_size_max', 800)) * 0.25,
                        )
                        if serve_tracking_frames >= 25 and large_near_court_candidate:
                            # A near-camera serve ball stays large and visible through a
                            # long toss. Wait for the deeper contact drop instead of
                            # starting on the first slow fall after the apex.
                            toss_drop_threshold = max(toss_drop_threshold, max(150, int(serve_area_height * 0.34)))
                            toss_start_reason = "near-court-contact-drop"
                        toss_speed_threshold = max(18, int(serve_area_height * 0.05))
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
                        pre_apex_rise_threshold = max(90, int(serve_area_height * 0.20))
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
                            print(
                                f"[SERVE_START_REJECT] f{self.frame_count}: {toss_start_reason} "
                                f"is after apex/drop; waiting for below-to-up toss start "
                                f"vertical_drop={vertical_drop:.0f}px recent_min_y={recent_min_y} "
                                f"candidate={potential_serve}"
                            )
                            clear_waiting_serve_history()
                    # Early serve start: if ball is moving fast in the configured serve direction within the serve area,
                    # start tracking immediately (don't wait for serve area exit)
                    # Require ALL consecutive pairs to move in the serve direction (not just first-to-last)
                    if len(serve_position_history) >= 4:
                        # Check last 3 consecutive pairs all move forward with signed dx > 15
                        recent_signed_dx = []
                        recent_signed_dy = []
                        for i in range(-3, 0):
                            pair_dx = serve_position_history[i][0] - serve_position_history[i-1][0]
                            pair_dy = serve_position_history[i][1] - serve_position_history[i-1][1]
                            recent_signed_dx.append(self._signed_serve_dx(pair_dx))
                            recent_signed_dy.append(self._signed_serve_dy(pair_dy))
                        all_forward = all(signed_dx >= 15 for signed_dx in recent_signed_dx)
                        min_signed_dx = min(recent_signed_dx) if recent_signed_dx else 0
                        avg_dx = min_signed_dx  # Use minimum signed dx as the threshold
                        serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
                        serve_width = max(1, self.serve_area_x_max - self.serve_area_x_min)
                        high_launch_y = self.serve_area_y_min + int(serve_height * 0.22)
                        total_signed_dx = sum(recent_signed_dx)
                        total_signed_dy = sum(recent_signed_dy)
                        serve_mid_x = (self.serve_area_x_min + self.serve_area_x_max) / 2.0
                        serve_side_dx_sign = 1 if serve_position_history[0][0] < serve_mid_x else -1
                        side_recent_signed_dx = []
                        for i in range(-3, 0):
                            pair_dx = serve_position_history[i][0] - serve_position_history[i-1][0]
                            side_recent_signed_dx.append(pair_dx * serve_side_dx_sign)
                        side_total_signed_dx = sum(side_recent_signed_dx)
                        contact_launch_y = self.serve_area_y_min + int(serve_height * 0.45)
                        recent_min_y = min(p[1] for p in serve_position_history)
                        pre_launch_pos = serve_position_history[-2]
                        pre_launch_descent = pre_launch_pos[1] - recent_min_y
                        post_toss_contact_launch = (
                            self.serve_direction_dy != 0 and
                            serve_tracking_frames <= 24 and
                            potential_serve[1] <= contact_launch_y and
                            side_recent_signed_dx[-1] >= max(80, int(serve_width * 0.035)) and
                            recent_signed_dy[-1] >= max(80, int(serve_height * 0.16)) and
                            recent_signed_dy[-2] <= -15 and
                            pre_launch_descent >= max(50, int(serve_height * 0.11)) and
                            abs(serve_position_history[-2][0] - serve_position_history[-3][0]) <= max(45, int(serve_width * 0.03))
                        )
                        # A classic pre-contact serve launch has a compact burst that is
                        # both forward and upward near the top of the serve area. This
                        # catches f3372/f3373 where the first forward step is slightly
                        # below the old fast-serve threshold, before a false high blob
                        # can reset the serve history.
                        precontact_toss_launch = (
                            all_forward and
                            self.serve_direction_dy != 0 and
                            serve_tracking_frames <= 10 and
                            potential_serve[1] <= high_launch_y and
                            total_signed_dx >= max(120, int(serve_width * 0.055)) and
                            total_signed_dy >= max(110, int(serve_height * 0.24)) and
                            recent_signed_dx[-1] >= 45 and
                            recent_signed_dx[-2] >= 45 and
                            recent_signed_dy[-1] >= 40 and
                            recent_signed_dy[-2] >= 40
                        )
                        fast_start_toss_context = low_to_up_serve_toss_context(serve_position_history)
                        if fast_start_toss_context is not None and (
                                (all_forward and (avg_dx >= 25 or precontact_toss_launch)) or
                                post_toss_contact_launch):
                            if post_toss_contact_launch:
                                serve_start_reason = "post-toss-contact-launch"
                                log_signed_dx = side_recent_signed_dx[-1]
                                log_total_dx = side_total_signed_dx
                            else:
                                serve_start_reason = (
                                    "precontact-toss-launch"
                                    if precontact_toss_launch and avg_dx < 25
                                    else "fast-serve"
                                )
                                log_signed_dx = avg_dx
                                log_total_dx = total_signed_dx
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
                            print(f"Ball position: {potential_serve}, signed dx={log_signed_dx:.1f}px/frame")
                            print(f"{'='*70}\n")
                            print(f"[TRACKING_START] f{self.frame_count}: {serve_start_reason} detected at {potential_serve} "
                                  f"signed_dx={log_signed_dx:.1f}px total_dx={log_total_dx:.1f}px "
                                  f"total_dy={total_signed_dy:.1f}px")
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
                            self._start_point_context(potential_serve)
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            log_tracking_start_position()
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
                        exit_start_toss_context = low_to_up_serve_toss_context(serve_position_history)
                        if exit_start_toss_context is not None:
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
                            print(f"[TRACKING_START] f{self.frame_count}: serve-toss-rise-exit detected at {last_serve_candidate} "
                                  f"low={exit_start_toss_context['low_pos']} rise={exit_start_toss_context['rise']:.0f}px "
                                  f"signed_total_dx={signed_total_dx:.0f}px signed_last_dx={signed_last_dx:.0f}px, predicted={predicted_pos}")
                            self.ball_center = predicted_pos
                            self.tracking = True
                            self.ball_stopped = False
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                            self.initial_ball_position = exit_start_toss_context['low_pos']
                            self.ball_size = None
                            self.ball_hsv = None
                            seed_tracking_from_serve_history(last_serve_candidate)
                            self.stuck_frame_count = 0
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self._start_point_context(last_serve_candidate)
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            log_tracking_start_position()
                        elif last_dy > 5 and last_dx >= -10 and serve_tracking_frames >= 5:
                            # Toss-complete: ball exiting serve area downward (falling after toss).
                            # Player is about to strike the ball just below the serve area.
                            # Start tracking from last known position — the 200px search radius
                            # will find the ball at the strike point.
                            print(f"[SERVE_START_REJECT] f{self.frame_count}: toss-complete is after apex/drop; "
                                  f"waiting for below-to-up toss start last={last_serve_candidate} last_dy={last_dy:.0f}px")
                            clear_waiting_serve_history()
                            continue
                            toss_complete_min_signed_total_dx = max(
                                25,
                                int((self.serve_area_x_max - self.serve_area_x_min) * 0.01),
                            )
                            if signed_total_dx < toss_complete_min_signed_total_dx or signed_last_dx < -2:
                                print(f"[SERVE_EXIT_REJECT] f{self.frame_count}: toss-complete blocked "
                                      f"signed_total_dx={signed_total_dx:.0f}px signed_last_dx={signed_last_dx:.0f}px "
                                      f"(need total>={toss_complete_min_signed_total_dx}px and last>=-2px)")
                                clear_waiting_serve_history()
                                continue
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
                            self._start_point_context(last_serve_candidate)
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            log_tracking_start_position()
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
                    game_state=game_state,
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

                if not self.headless:
                    cv2.imshow("Tennis Game Tracker", display_frame)

            prev_frame_for_debug = last_frame_for_debug
            prev_ball_center_for_debug = last_ball_center_for_debug
            prev_frame_index_for_debug = last_frame_index_for_debug
            last_frame_for_debug = frame.copy()
            last_ball_center_for_debug = self.ball_center
            last_frame_index_for_debug = self.frame_count

            if not getattr(self, 'auto_play', False) and not self.headless:
                if self.pause_requested:
                    play_mode = False
                key = cv2.waitKey(30 if play_mode else 0) & 0xFF
            else:
                key = 0xFF  # no-op key in headless/auto_play mode
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
    parser.add_argument("--court", choices=list(COURT_CONFIGS.keys()), default="2",
                        help="Which court/video to analyse (default: 2)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (no GUI windows)")
    parser.add_argument("--disable-false-points", dest="disable_false_points", action="store_true",
                        help="Disable learned false-point hiding in debug/tuner views (default)")
    parser.add_argument("--enable-false-points", dest="disable_false_points", action="store_false",
                        help="Enable learned false-point hiding in debug/tuner views")
    parser.set_defaults(disable_false_points=True)
    args = parser.parse_args()

    court = COURT_CONFIGS[args.court]
    print(f"[COURT] {court['label']}")
    if args.disable_false_points:
        print("[FALSE_POINT] Debug false-point masking disabled")
    analyzer = InteractiveBallAnalyzer(court["video"], start_frame=args.start_frame,
                                       config_file=court["config"], headless=args.headless,
                                       disable_false_points=args.disable_false_points)
    analyzer.process_video(auto_play=args.auto_play, max_frames=args.max_frames)
