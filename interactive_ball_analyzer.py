import argparse

# Compact console output is the default.  Detailed contour/search diagnostics
# can be restored with --verbose-debug when a difficult frame needs inspection.
_verbose_debug_enabled = False

def _verbose_debug_print(*args, **kwargs):
    if _verbose_debug_enabled:
        print(*args, **kwargs)
import contextlib
import csv
import os
import sys
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2
import numpy as np
import json
import math
from datetime import datetime
from typing import Tuple, Optional

from player_racket_tracker import PlayerRacketTracker
from ball_dataset_exporter import BallDatasetExporter
from ball_local_ai_recovery import LocalBallAIRecovery, frame_buffer

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


_cli_native_stderr_state = None


def _redirect_native_stderr_to_devnull():
    """Redirect noisy native decoder stderr while keeping tracker stdout readable."""
    show_decoder_errors = os.environ.get("TGT_SHOW_DECODER_ERRORS", "").lower()
    if show_decoder_errors in ("1", "true", "yes", "on"):
        return None

    kernel32 = None
    std_error_handle = None
    saved_windows_stderr_handle = None
    try:
        stderr_fd = 2
        saved_stderr_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        if os.name == "nt":
            import ctypes
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetStdHandle.argtypes = [ctypes.c_ulong]
            kernel32.GetStdHandle.restype = ctypes.c_void_p
            kernel32.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
            kernel32.SetStdHandle.restype = ctypes.c_int
            std_error_handle = ctypes.c_ulong((-12) & 0xFFFFFFFF)
            saved_windows_stderr_handle = kernel32.GetStdHandle(std_error_handle)
            kernel32.SetStdHandle(
                std_error_handle,
                ctypes.c_void_p(msvcrt.get_osfhandle(devnull_fd)),
            )
        os.dup2(devnull_fd, stderr_fd)
        return {
            "saved_stderr_fd": saved_stderr_fd,
            "devnull_fd": devnull_fd,
            "kernel32": kernel32,
            "std_error_handle": std_error_handle,
            "saved_windows_stderr_handle": saved_windows_stderr_handle,
        }
    except Exception:
        for fd_name in ("saved_stderr_fd", "devnull_fd"):
            fd = locals().get(fd_name)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        return None


def _restore_native_stderr(state):
    if not state:
        return
    try:
        os.dup2(state["saved_stderr_fd"], 2)
        kernel32 = state.get("kernel32")
        saved_windows_stderr_handle = state.get("saved_windows_stderr_handle")
        if kernel32 is not None and saved_windows_stderr_handle:
            kernel32.SetStdHandle(
                state["std_error_handle"],
                saved_windows_stderr_handle,
            )
    finally:
        for fd_name in ("saved_stderr_fd", "devnull_fd"):
            try:
                os.close(state[fd_name])
            except OSError:
                pass


@contextlib.contextmanager
def suppress_native_stderr():
    state = _redirect_native_stderr_to_devnull()
    try:
        yield
    finally:
        _restore_native_stderr(state)


def _parse_start_score(value: str) -> Tuple[int, int, int, int]:
    """Parse a partial-run score such as ``0:2 0:15``.

    The returned values are the internal game and point counters for P1 then
    P2. ``AD`` is accepted for the player who has advantage, for example
    ``1:1 AD:40``.
    """
    text = str(value or "").strip().replace(",", " ")
    parts = text.split()
    if len(parts) != 2 or any(":" not in part for part in parts):
        raise argparse.ArgumentTypeError(
            "start score must be GAMES POINTS, for example '0:2 0:15'"
        )

    try:
        games = [int(label) for label in parts[0].split(":")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("game score must contain non-negative integers") from error
    if len(games) != 2 or any(score < 0 for score in games):
        raise argparse.ArgumentTypeError("game score must be two non-negative values, e.g. 0:2")

    point_labels = [label.upper() for label in parts[1].split(":")]
    point_values = {"0": 0, "15": 1, "30": 2, "40": 3}
    if len(point_labels) != 2:
        raise argparse.ArgumentTypeError("point score must be two values, e.g. 0:15")
    if any(label not in (*point_values, "AD") for label in point_labels):
        raise argparse.ArgumentTypeError("point score values must be 0, 15, 30, 40, or AD")
    if point_labels.count("AD") > 1:
        raise argparse.ArgumentTypeError("only one player can have AD")
    if "AD" in point_labels:
        advantage_idx = point_labels.index("AD")
        if point_labels[1 - advantage_idx] != "40":
            raise argparse.ArgumentTypeError("AD is only valid against 40")
        points = [3, 3]
        points[advantage_idx] = 4
    else:
        points = [point_values[label] for label in point_labels]

    return games[0], games[1], points[0], points[1]


def _parse_frame_range(value: str) -> Tuple[int, int]:
    raw = str(value or "").strip()
    if ":" not in raw:
        raise argparse.ArgumentTypeError("frame range must be START:END")
    left, right = raw.split(":", 1)
    try:
        start, end = int(left), int(right)
    except ValueError as error:
        raise argparse.ArgumentTypeError("frame range must contain integers") from error
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("frame range must satisfy 0 <= START <= END")
    return start, end


class _QuietTrackerOutput:
    """Drop contour-by-contour diagnostics while preserving audit heartbeats.

    Headless regressions need ``[TRACK]`` and point-end lines to update their
    progress/audit reports.  Sending every HSV contour rejection through a
    pipe, however, can create multi-megabyte logs and make a single frame take
    minutes.  This lightweight stream filter is used by ``--quiet``: tracker
    behaviour stays identical, but only the lines needed for progress and
    endpoint diagnosis are forwarded.
    """

    def __init__(self, target):
        self._target = target
        self._buffer = ""

    @staticmethod
    def _keep(line):
        text = line.strip()
        return (
            text.startswith("Frame ") or
            text.startswith("[TRACK]") or
            text.startswith("[POINT_END]") or
            text.startswith("[TRACKING_START]") or
            text.startswith("[SERVE_START_") or
            text.startswith("[BALL_LOST]") or
            text.startswith("[BALL_LOSS_DIAGNOSTIC]") or
            text.startswith("[JUMP_REJECTED]") or
            text.startswith("[POINT_IGNORED]") or
            text.startswith("[MAX_FRAMES]") or
            text.startswith("[VIDEO_END]") or
            text.startswith("Analysis complete!")
        )

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self._keep(line):
                self._target.write(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer:
            if self._keep(self._buffer):
                self._target.write(self._buffer)
            self._buffer = ""
        self._target.flush()


class InteractiveBallAnalyzer:
    def __init__(
        self,
        video_path: str,
        start_frame: int = 0,
        config_file: str = "hsv_config.json",
        headless: bool = False,
        disable_false_points: bool = False,
        point_history_file: str = "point_history.csv",
        write_point_history: bool = True,
        start_server_side: Optional[str] = None,
        start_score: Optional[Tuple[int, int, int, int]] = None,
        enable_player_tracking: bool = True,
        player_tracking_interval: int = 5,
        enable_player_learning: bool = True,
        enable_player_ball_protection: bool = True,
        ball_dataset_dir: Optional[str] = None,
        local_ai_model: Optional[str] = None,
        local_ai_python: Optional[str] = None,
        local_ai_recovery_dir: Optional[str] = None,
        debug_local_ai_range: Optional[Tuple[int, int]] = None,
        debug_local_ai_radius: float = 140.0,
    ):
        self.video_path = video_path
        self.config_file = config_file
        self.headless = headless
        self.disable_false_points = disable_false_points
        self.point_history_file = point_history_file
        self.write_point_history = write_point_history
        self.ball_dataset_dir = ball_dataset_dir
        self.local_ai_model = local_ai_model
        self.player_tracking_enabled = bool(enable_player_tracking) and os.environ.get(
            "DISABLE_PLAYER_TRACKING", "0"
        ) != "1"
        self.player_learning_enabled = bool(enable_player_learning) and os.environ.get(
            "DISABLE_PLAYER_LEARNING", "0"
        ) != "1"
        self.player_ball_protection_enabled = bool(enable_player_ball_protection) and os.environ.get(
            "DISABLE_PLAYER_BALL_PROTECTION", "0"
        ) != "1"
        self.player_tracking_interval = max(1, int(player_tracking_interval))
        profile_stem = os.path.splitext(os.path.basename(config_file or "hsv_config.json"))[0]
        profile_dir = os.path.dirname(os.path.abspath(config_file or "hsv_config.json"))
        self.player_tracking_profile_path = os.path.join(
            profile_dir, f"player_tracking_{profile_stem}.json"
        )
        self.player_tracker = PlayerRacketTracker(
            detection_interval=self.player_tracking_interval,
            profile_path=self.player_tracking_profile_path,
            learning_enabled=self.player_learning_enabled,
        ) if self.player_tracking_enabled else None
        if start_server_side not in (None, "near", "far"):
            raise ValueError("start_server_side must be 'near', 'far', or None")
        self.start_server_side = start_server_side
        self._point_history_initialized = False
        self._point_history_point_index = 0
        self._point_history_current = None
        self._point_audit_traces = {}
        self._process_stop_reason = None
        self.cap = cv2.VideoCapture(video_path)
        self.start_frame = max(0, start_frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.frame_count = self.start_frame
        self.last_seen_frame = self.start_frame
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ball_dataset_exporter = (
            BallDatasetExporter(ball_dataset_dir, video_path)
            if ball_dataset_dir else None
        )
        if self.ball_dataset_exporter is not None:
            print(
                f"[BALL_DATASET] Exporting tracker-labelled source frames to "
                f"{self.ball_dataset_exporter.run_dir}"
            )
        self.local_ai_recovery = None
        self._debug_local_ai_range = debug_local_ai_range
        self._debug_local_ai_radius = max(20.0, float(debug_local_ai_radius))
        self._force_local_ai_range = None
        self._force_local_ai_radius = 450.0
        self._force_local_ai_min_score = 0.985
        self._force_local_ai_history = []
        # Generic racket-contact AI ownership.  This is armed automatically only
        # when the normal HSV path becomes suspicious inside a tracked player's
        # contact corridor.  It reuses the proven forced-AI tracker, but keeps a
        # separate history so the debug flag and automatic mode never interfere.
        self._contact_local_ai_state = None
        self._contact_local_ai_cooldown_until_frame = -1000000
        self._contact_local_ai_radius = 450.0
        self._contact_local_ai_min_score = 0.985
        self._local_ai_frame_buffer = frame_buffer(12)
        self._local_ai_recovery_count = 0
        self._local_ai_all_body_rejections = 0
        # A sharp near-player turn may need a few tightly cropped AI rankings
        # before ordinary HSV tracking has a trustworthy outgoing vector
        # again.  This state is bounded to the contact corridor and hands off
        # as soon as normal tracking clears the player.
        self._local_ai_tight_roi_follow = None
        self._local_ai_tight_roi_accept_frame = -1000000
        self._local_ai_tight_roi_attempt_frame = -1000000
        self._local_ai_tight_roi_previous_gray = None
        self._discard_provisional_serve_from_ai = False
        # A provisional serve can briefly select a static, ball-coloured court
        # artifact.  Keep that candidate separate from the live track until
        # the following decoded frame can prove that it really persists.
        # This is intentionally runtime-only: a coordinate that is false
        # during a tossed-and-caught serve may be a legitimate ball location
        # later in a rally.
        self._pending_provisional_static_candidate = None
        # Keep history reconciliation separate from the live tracking state.
        # A verified caught toss can change the tennis interpretation of two
        # subsequent serves, but changing ``current_serve_attempt`` while the
        # HSV tracker is still in flight was shown to alter later candidate
        # selection.  This record lets the CSV reflect the verified sequence
        # after the normal tracker has completed each physical flight.
        self._local_ai_follow_until_frame = -1
        # A recovery usually happens at a player/racket occlusion.  The ball
        # needs more than a single four-frame sample to leave that region, but
        # this handoff must remain bounded so local-AI scoring cannot replace
        # normal tracking for an entire rally.
        self._local_ai_handoff_deadline_frame = -1
        if local_ai_model:
            self.local_ai_recovery = LocalBallAIRecovery(
                local_ai_model,
                config_file,
                python_executable=local_ai_python,
                work_dir=local_ai_recovery_dir or "tmp/local_ai_recovery",
            )
            print(
                f"[LOCAL_AI] Buffered recovery enabled: model={self.local_ai_recovery.model_path} "
                f"lookback={self.local_ai_recovery.lookback_frames}f"
            )
        
        # Ball analysis state
        self.ball_center = None
        self.ball_hsv = None
        self.ball_size = None
        self.tracking = False
        self.ball_stopped = False
        self.stuck_frame_count = 0
        # After a player-occlusion re-acquisition, keep rejecting static HSV
        # blobs for a short window.  The player/racket can contain a large
        # ball-coloured patch, and accepting it immediately after reacquisition
        # causes a false bounce/end while the real ball is still in flight.
        self._player_reacq_protect_until_frame = -1
        # A failed full-frame motion reacquisition should not be followed by
        # another expensive full-frame HSV contour scan on every frame.  The
        # motion pass already examines the whole image; until the next stuck
        # milestone, keep the normal local search around the last valid ball.
        self._player_reacq_motion_failed_until_frame = -1
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
        self._focus_loss_guard_until_frame = -1000000
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
        self.click_upper_hsv_specs = [
            (
                "click_upper_tight",
                np.array([95, 12, 180], dtype=np.uint8),
                np.array([103, 38, 255], dtype=np.uint8),
                0.0,
            ),
            (
                "click_upper_bright",
                np.array([90, 10, 170], dtype=np.uint8),
                np.array([108, 45, 255], dtype=np.uint8),
                18.0,
            ),
        ]
        self._click_upper_hsv_recover_frame = -1000000
        self.disable_click_upper_hsv = os.environ.get("DISABLE_CLICK_UPPER_HSV", "0") == "1"
        self._top_far_out_deferred_candidate = None
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
        self._base_serve_area = None
        self._base_serve_area_end = None
        self._configured_far_serve_area = None
        self._active_serve_area_end = None
        self._base_serve_direction_dx = 1
        self._base_serve_direction_dy = -1
        self._base_serve_ball_size_min = 3
        self._base_serve_ball_size_max = 80
        self.far_serve_ball_size_min = 1
        self.far_serve_ball_size_max = 120
        self._serve_scan_block_until_frame = -1
        self.point_start_frame_internal = None
        self._serve_contact_grace_frames = 0
        self._serve_launch_direction_x = 0
        self._rally_contact_grace_frames = 0
        self._rally_contact_large_ref_frames = 0
        self._rally_contact_ref_size = None
        self._rally_contact_origin = None
        self._rally_contact_expected = None
        self._rally_contact_progress = 0.0
        self._rally_contact_high_launch_until_frame = -1
        self._trusted_rally_launch_frame = -1000000
        self._trusted_rally_launch_pos = None
        self._last_racket_contact_frame = -1000000
        self._last_racket_contact_point = None
        self._last_racket_contact_player = None
        # Metadata for the most recent motion-based re-acquisition.  A first
        # candidate after a player/racket occlusion is provisional: tiny
        # moving fragments on the net/court must not become the ball merely
        # because they are the best HSV match.
        self._last_reacq_candidate_area = 0.0
        self._last_reacq_candidate_score = None
        self._last_counted_contact_frame = -1000000
        self._point_hit_count = 0
        self._last_point_hit_count = 0
        self._ground_bounce_grace_frames = 0
        self._ground_bounce_ref_size = None
        self._ground_bounce_origin = None
        self._ground_bounce_expected = None
        self._ground_bounce_progress = 0.0
        self._ground_bounce_debug_history = []
        self.ground_bounce_count = 0
        self.last_ground_bounce_frame = -1000000
        self._last_serve_bounce_frame = -1000000
        self._last_serve_bounce_point = None
        self._last_serve_bounce_net_contact_like = False
        self._last_serve_bounce_was_in = False
        self._serve_net_contact_frame = -1000000
        self._serve_net_contact_point = None
        self._serve_net_pass_frame = -1000000
        self._serve_net_pass_point = None
        self._serve_net_zone_frames = 0
        self._near_side_large_hit_ref_size = 0.0
        self._near_side_large_hit_ref_frame = -1000000
        # A lower-court contact window is computed on the frame before the
        # racket turn.  Keep it for one following frame so a one-frame HSV
        # mis-detection cannot discard the outgoing ball candidate.
        self._pending_lower_contact_launch_context = None
        self._pending_lower_contact_launch_until_frame = -1000000
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
        self.player_names = ["P1", "P2"]
        self.score_points = [0, 0]
        self.score_games = [0, 0]
        self.score_game_index = 0
        self._explicit_start_score = start_score is not None
        if start_score is not None:
            games_p1, games_p2, points_p1, points_p2 = start_score
            score_values = (games_p1, games_p2, points_p1, points_p2)
            if any(not isinstance(score, (int, np.integer)) or int(score) < 0 for score in score_values):
                raise ValueError("start_score must contain non-negative integer counters")
            self.score_games = [int(games_p1), int(games_p2)]
            self.score_points = [int(points_p1), int(points_p2)]
            # Serve alternates after each completed game. The supplied game
            # score therefore defines both the current game and next server.
            self.score_game_index = sum(self.score_games)
            print(
                f"[MATCH_SEED] f{self.start_frame}: explicit score={self._score_summary()} "
                f"server={self.player_names[self._current_server_index()]} "
                f"game={self.score_game_index + 1}"
            )
        self._last_scored_point_end_frame = -1
        self._last_point_winner = None
        self._last_point_score_reason = None
        self._last_point_outcome_category = None
        self.current_serve_attempt = 1
        self._serve_landed_in_current_attempt = False
        self._serve_in_recorded_attempt = None
        self._serve_start_requires_confirmation = False
        self._last_confirmed_point_end_frame = None
        self.serve_stats = [
            {'first_in': 0, 'first_faults': 0, 'second_in': 0, 'double_faults': 0},
            {'first_in': 0, 'first_faults': 0, 'second_in': 0, 'double_faults': 0},
        ]
        self.point_stats = [
            {'out_errors': 0, 'net_errors': 0, 'unreturned_winners': 0, 'points_won': 0, 'total_hits': 0},
            {'out_errors': 0, 'net_errors': 0, 'unreturned_winners': 0, 'points_won': 0, 'total_hits': 0},
        ]

    def _is_04_left_court_config(self):
        return os.path.basename(self.config_file or "").lower() in (
            "hsv_config_court2.json",
            "hsv_config_04_left_night.json",
        )

    def _is_night_session_config(self):
        return os.path.basename(self.config_file or "").lower() == "hsv_config_04_left_night.json"

    def _update_player_tracking(self, frame):
        """Update player/racket context without taking ownership of ball tracking."""
        tracker = getattr(self, "player_tracker", None)
        if tracker is None or frame is None:
            return None
        net_y = None
        if hasattr(self, "net_area_y_min") and hasattr(self, "net_area_y_max"):
            net_y = (float(self.net_area_y_min) + float(self.net_area_y_max)) * 0.5
        try:
            if hasattr(self, "serve_area_x_min") and hasattr(self, "serve_area_x_max"):
                tracker.set_court_x_range((self.serve_area_x_min, self.serve_area_x_max))
            try:
                sideline_model = self._build_singles_sideline_model(frame)
                if sideline_model is not None:
                    height = frame.shape[0]
                    tracker.set_court_region({
                        "left": sideline_model["left"],
                        "right": sideline_model["right"],
                        "y_min": height * 0.03,
                        "y_max": height * 0.97,
                    })
            except Exception:
                # The x-range fallback remains active if court-line fitting fails.
                pass
            # HOG person detection is the dominant per-frame cost (roughly
            # 0.7–0.9 s on this 4K night video).  During a ball-loss/player-
            # reacquisition window the existing player box is deliberately
            # held, so running HOG again cannot improve ball recovery and only
            # stalls the frame loop.  Advance the detector timestamp while the
            # ball is stuck/slow so PlayerRacketTracker performs its cheap
            # grayscale update instead.
            last_motion = getattr(self, "last_motion", None) or {}
            slow_stuck_recovery = (
                int(getattr(self, "stuck_frame_count", 0)) >= 4 and
                float(last_motion.get("distance", 0.0) or 0.0) < 12.0
            )
            player_reacq_guard = int(
                getattr(self, "_player_reacq_protect_until_frame", -1)
            ) >= int(getattr(self, "frame_count", 0))
            if (slow_stuck_recovery or player_reacq_guard) and hasattr(
                    tracker, "_last_detection_frame"):
                tracker._last_detection_frame = int(self.frame_count)
            return tracker.update(frame, int(self.frame_count), net_y=net_y)
        except Exception as error:
            # Player context is auxiliary; a failed detector must never stop ball analysis.
            if int(getattr(self, "frame_count", -1)) % 120 == 0:
                print(f"[PLAYER_TRACKING_ERROR] f{self.frame_count}: {error}")
            return None

    def _player_candidate_penalty(self, point, area, motion_mean, motion_max,
                                  contour=None, predicted_distance=None):
        tracker = getattr(self, "player_tracker", None)
        if tracker is None:
            return 0.0, None
        try:
            return tracker.candidate_penalty(
                point,
                area,
                motion_mean,
                motion_max,
                contour=contour,
                predicted_distance=predicted_distance,
                head_only=not getattr(self, "player_ball_protection_enabled", False),
            )
        except Exception:
            return 0.0, None

    def _player_point_zone(self, point):
        tracker = getattr(self, "player_tracker", None)
        if tracker is None or point is None:
            return None
        try:
            return tracker.point_zone(point)
        except Exception:
            return None

    def _point_in_player_contact_corridor(self, point):
        """Return whether ``point`` lies in a player's padded contact area.

        Strict body/racket zones are intentionally narrow and can miss a ball
        just outside the racket.  The player tracker already exposes a padded
        association corridor for stroke classification; reuse it only as a
        trigger hint.  Local AI still has to verify a strongly moving contour
        in a small causal ROI before any position is accepted.
        """
        if point is None:
            return False
        if self._player_point_zone(point) is not None:
            return True
        tracker = getattr(self, 'player_tracker', None)
        track_for_point = getattr(tracker, '_track_for_point', None)
        if not callable(track_for_point):
            return False
        try:
            return track_for_point(point) is not None
        except Exception:
            return False

    def _reject_unlocked_night_serve_body_candidate(self, point, lock_active=False):
        """Reject a torso blob from becoming the *first* serve-ball sample.

        The night camera produces bright, ball-sized fragments on the server's
        shirt.  Before a toss/flight history exists, accepting one of those
        fragments can manufacture an entire point (for example the false
        serve at a prior frame).  Once a trajectory is locked, the ball is
        allowed to cross the player body/racket at contact, so this must not
        be a general ball-tracking exclusion.
        """
        if lock_active:
            return False
        return self._player_point_zone(point) == "player_body"

    def _player_occlusion_artifact(self, point):
        """Return the player zone when a held point is likely an artifact."""
        zone = self._player_point_zone(point)
        if zone is None or int(getattr(self, "stuck_frame_count", 0)) < 1:
            return None
        last_motion = getattr(self, "last_motion", None) or {}
        distance = float(last_motion.get("distance", 0.0) or 0.0)
        size = float(getattr(self, "ball_size", 0.0) or 0.0)
        if zone in ("player_head_hat", "player_shoes", "racket_fragment"):
            return zone
        if zone == "player_body" and (distance < 8.0 or size >= 120.0):
            return zone
        return None

    def _post_serve_pre_net_recovery_reason(self, previous_position, tracked_position):
        """Keep a repaired near-side serve causal until it really reaches the net.

        This guard is armed only after an actual post-serve wrong-way rejection.
        While it is active, no player/racket/head/shoe pixel may become the
        endpoint of recovery, and even a clear candidate may not move materially
        back toward the server. The guard is cleared only by a plausible,
        player-free step that reaches the net band.
        """
        if not bool(getattr(self, '_post_serve_pre_net_recovery_active', False)):
            return None
        if tracked_position is None:
            return None

        zone = self._player_point_zone(tracked_position)
        if zone is not None:
            return f"post-serve-launch-pre-net-player:{zone}"

        if previous_position is not None:
            expected_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
            step_dy = float(tracked_position[1]) - float(previous_position[1])
            if expected_dy < 0 and step_dy >= 18.0:
                return f"post-serve-launch-pre-net-wrong-way:{step_dy:.0f}px"
            if expected_dy > 0 and step_dy <= -18.0:
                return f"post-serve-launch-pre-net-wrong-way:{step_dy:.0f}px"
        return None

    def _maybe_clear_post_serve_pre_net_recovery(self, previous_position, tracked_position, source):
        """Clear the strict guard only on a plausible player-free net arrival."""
        if not bool(getattr(self, '_post_serve_pre_net_recovery_active', False)):
            return False
        if previous_position is None or tracked_position is None or not hasattr(self, 'net_y'):
            return False
        if self._player_point_zone(tracked_position) is not None:
            return False

        dx = float(tracked_position[0]) - float(previous_position[0])
        dy = float(tracked_position[1]) - float(previous_position[1])
        step = math.hypot(dx, dy)
        expected_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
        toward_net = (expected_dy < 0 and dy < 0.0) or (expected_dy > 0 and dy > 0.0)
        if not toward_net or step > 400.0:
            return False

        net_y = float(self.net_y)
        reached_net_band = (
            (expected_dy < 0 and float(tracked_position[1]) <= net_y + 25.0) or
            (expected_dy > 0 and float(tracked_position[1]) >= net_y - 25.0)
        )
        if not reached_net_band:
            return False

        self._post_serve_pre_net_recovery_active = False
        print(
            f"[POST_SERVE_PRE_NET_CLEAR] f{self.frame_count}: source={source} "
            f"prev={previous_position} tracked={tracked_position} "
            f"net_y={net_y:.1f} step={step:.1f}px"
        )
        return True

    def _local_ai_recovery_reason(self, previous_position, tracked_position, previous_stuck):
        """Return a narrow recovery trigger; normal tracking remains primary."""
        if self.local_ai_recovery is None:
            return None

        # POST_SERVE_LAUNCH_LOCK_GUARD_V1
        # Once a serve launch was verified, do not let a large wrong-way jump
        # immediately re-anchor tracking on the server/player body. We only
        # reject a strong reversal inside this short post-contact window.
        _post_serve_lock_until = int(
            getattr(self, "_post_serve_launch_lock_until_frame", -1)
        )
        if (
                int(self.frame_count) <= _post_serve_lock_until and
                previous_position is not None and
                tracked_position is not None
        ):
            _ps_dx = float(tracked_position[0] - previous_position[0])
            _ps_dy = float(tracked_position[1] - previous_position[1])
            _ps_distance = math.hypot(_ps_dx, _ps_dy)
            _ps_expected_dy = int(getattr(self, "serve_direction_dy", 0) or 0)

            _ps_recent_speed = 0.0
            _ps_velocity_history = getattr(self, "ball_velocity_history", None) or []
            if _ps_velocity_history:
                try:
                    _ps_recent_speed = float(np.median(_ps_velocity_history[-5:]))
                except Exception:
                    _ps_recent_speed = 0.0

            # POST_SERVE_LAUNCH_LOCK_THRESHOLD_V2
            # Contact frames can contain 300-400 px spikes.  Never let those
            # spikes raise the reversal threshold above a clearly impossible
            # wrong-way post-serve jump.
            _ps_jump_floor = max(100.0, min(220.0, _ps_recent_speed * 1.10))
            _ps_wrong_way = (
                (_ps_expected_dy < 0 and _ps_dy >= 70.0) or
                (_ps_expected_dy > 0 and _ps_dy <= -70.0)
            )

            if _ps_wrong_way and _ps_distance >= _ps_jump_floor:
                self._post_serve_launch_lock_last_reject_frame = int(self.frame_count)
                # POST_SERVE_RECOVERY_HOLD_V3
                # A rejected player-body reversal must not become the anchor on
                # the next frame.  Give the emergency recovery a short bounded
                # window to find the outgoing ball while holding the last sane
                # state.  This starts only after an actual rejection, so normal
                # receiver returns later in the rally are unaffected.
                self._post_serve_recovery_hold_until_frame = max(
                    int(getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)),
                    int(self.frame_count) + 10,
                )
                if not bool(getattr(self, '_post_serve_pre_net_recovery_active', False)):
                    self._post_serve_pre_net_recovery_active = True
                    self._post_serve_pre_net_recovery_started_frame = int(self.frame_count)
                    print(
                        f"[POST_SERVE_PRE_NET_GUARD] f{self.frame_count}: armed until trusted net crossing "
                        f"from last_good={previous_position}"
                    )
                print(
                    f"[POST_SERVE_LAUNCH_REJECT] f{self.frame_count}: "
                    f"tracked={tracked_position} previous={previous_position} "
                    f"dy={_ps_dy:+.0f}px dist={_ps_distance:.1f}px "
                    f"floor={_ps_jump_floor:.1f}px; forcing Local AI recovery"
                )
                return "post-serve-launch-wrong-way"

        strict_pre_net_reason = self._post_serve_pre_net_recovery_reason(
            previous_position, tracked_position
        )
        if strict_pre_net_reason is not None:
            return strict_pre_net_reason

        # The tight near-player ROI ranker runs before the candidate is
        # committed.  Do not immediately run the broad buffered recovery on
        # the same accepted frame merely because the real ball still lies
        # inside the player's bounding region.
        if self.frame_count == int(
                getattr(self, "_local_ai_tight_roi_accept_frame", -1000000)):
            return None
        # A successful recovery establishes a short *endpoint* handoff window,
        # but it must not turn into another model inference for every frame in
        # that window.  The normal HSV/motion tracker is already following the
        # repaired flight and is both faster and better at the small blurred
        # contours immediately after the repair.  Calling the model here used
        # to force a four-frame replay on every frame, then treat one blurred
        # frame with no AI candidate as a new failure.
        if self.frame_count <= int(getattr(self, "_local_ai_follow_until_frame", -1)):
            # A verified trajectory recovery normally hands control straight
            # back to the HSV tracker.  If that tracker immediately *holds*
            # the repaired point instead of advancing it, however, the short
            # handoff must not suppress the very trajectory rescue that can
            # bridge the remaining player occlusion.  This is based only on
            # tracker state (same point / rising stuck count), never on a
            # particular frame or court coordinate.
            handoff_step = None
            if previous_position is not None and tracked_position is not None:
                handoff_step = math.hypot(
                    float(tracked_position[0]) - float(previous_position[0]),
                    float(tracked_position[1]) - float(previous_position[1]),
                )
            handoff_held = (
                tracked_position is None or
                (handoff_step is not None and handoff_step <= 2.5) or
                int(getattr(self, "stuck_frame_count", 0) or 0) > int(previous_stuck or 0)
            )
            if handoff_held and previous_position is not None:
                print(
                    f"[LOCAL_AI_HANDOFF_STALL] f{self.frame_count}: "
                    f"prev={previous_position} tracked={tracked_position} "
                    f"step={handoff_step if handoff_step is not None else 'missing'} "
                    f"stuck={int(getattr(self, 'stuck_frame_count', 0) or 0)}"
                )
                return "player-region:handoff-stall"
            return None
        if not self.local_ai_recovery.ready(self.frame_count):
            return None
        # A rise/fall sequence at the far baseline is only a *candidate*
        # serve.  Before we accept or reject it, ask the local model to replay
        # the nearby raw frames if HSV selects a large discontinuity.  This
        # prevents one shirt/racket contour from deciding whether a practice
        # toss (a prior frame) or the following real serve is a point.
        if (
                getattr(self, "_provisional_serve_start_kind", None) is not None and
                previous_position is not None and tracked_position is not None):
            provisional_jump = math.hypot(
                float(tracked_position[0]) - float(previous_position[0]),
                float(tracked_position[1]) - float(previous_position[1]),
            )
            if provisional_jump >= 55.0:
                return f"provisional-serve-jump:{provisional_jump:.0f}px"
        if tracked_position is not None:
            if previous_position is not None:
                jump = math.hypot(
                    float(tracked_position[0]) - float(previous_position[0]),
                    float(tracked_position[1]) - float(previous_position[1]),
                )
                # General form of the provisional-serve protection.  A ball
                # cannot both jump to a new location and have an unchanged
                # image patch there.  Ask local AI first; if it cannot form a
                # path, the next-frame persistence guard decides whether the
                # blob is a static artifact.  Keep this below 600px so it
                # covers the common false court/net points without changing
                # the deliberately separate extreme-jump recovery path.
                static_metrics_current_frame = (
                    int(getattr(self, "_last_tracked_candidate_motion_frame", -1)) ==
                    int(self.frame_count)
                )
                if (
                        55.0 <= jump <= 600.0 and static_metrics_current_frame and
                        float(getattr(self, "_last_tracked_candidate_motion_mean", 0.0) or 0.0) < 5.0 and
                        float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
                    return f"static-candidate-jump:{jump:.0f}px"
            zone = self._player_point_zone(tracked_position)
            # Shoes and small head fragments appear often during ordinary
            # play.  They are already penalized by the normal tracker; do not
            # pay the recovery cost unless the failure is a body/racket
            # takeover, where a ball can genuinely disappear behind a player.
            if zone in ("player_body", "racket_fragment"):
                return f"player-region:{zone}"
            if previous_position is not None:
                if jump > 600.0:
                    return f"untrusted-jump:{jump:.0f}px"
        if tracked_position is None and int(previous_stuck) >= 3:
            return "missing-after-stuck"
        if (
            tracked_position is not None and previous_position is not None and
            tuple(tracked_position) == tuple(previous_position) and int(previous_stuck) >= 3
        ):
            return "held-position"
        return None

    def _snapshot_tracking_state_for_provisional_guard(self):
        """Save only live tracking fields that a rejected contour can poison."""
        def clone(value):
            if isinstance(value, np.ndarray):
                return value.copy()
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, list):
                return list(value)
            return value

        fields = (
            "ball_center", "ball_size", "ball_hsv", "last_motion", "prev_motion",
            "last_nonzero_motion", "last_delta", "last_direction",
            "ball_velocity_history", "last_seen_frame", "stuck_frame_count",
            "near_edge", "focus_loss_active", "focus_loss_frame",
            "_focus_loss_guard_until_frame", "_recent_max_ball_size",
            "_prev_frame_gray", "_last_tracked_candidate_motion_frame",
            "_last_tracked_candidate_motion_mean", "_last_tracked_candidate_motion_max",
        )
        return {
            field: clone(getattr(self, field))
            for field in fields
            if hasattr(self, field)
        }

    def _restore_tracking_state_for_provisional_guard(self, snapshot):
        if not snapshot:
            return
        for field, value in snapshot.items():
            setattr(self, field, value)

    def _static_blob_near(self, image, position, radius=14):
        """Find a compact HSV blob near ``position`` in one raw frame.

        It is deliberately a local confirmation tool, not a second tracker:
        this only decides whether a rejected *provisional serve* blob stayed
        in the same screen location for an adjacent frame.
        """
        if image is None or position is None:
            return None
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return None

        x, y = int(position[0]), int(position[1])
        height, width = hsv.shape[:2]
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(width, x + radius + 1), min(height, y + radius + 1)
        if x2 <= x1 or y2 <= y1:
            return None

        specs = []
        if self.hsv_lower is not None and self.hsv_upper is not None:
            specs.append((self.hsv_lower, self.hsv_upper))
        regular = getattr(self, "hsv_regular", None)
        if regular is not None:
            specs.append((regular["lower"], regular["upper"]))

        best = None
        for lower, upper in specs:
            mask = cv2.inRange(hsv[y1:y2, x1:x2], lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 0.5 or area > 80.0:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                cx = x1 + float(moments["m10"] / moments["m00"])
                cy = y1 + float(moments["m01"] / moments["m00"])
                distance = math.hypot(cx - x, cy - y)
                if distance > radius:
                    continue
                candidate = {"pos": (cx, cy), "area": area, "distance": distance}
                if best is None or candidate["distance"] < best["distance"]:
                    best = candidate
        return best

    def _finalize_pending_provisional_static_candidate(self):
        """Use the next raw frame to confirm/reject a provisional static blob."""
        pending = getattr(self, "_pending_provisional_static_candidate", None)
        if not pending:
            return False
        pending_frame = int(pending.get("frame", -1))
        if self.frame_count <= pending_frame:
            return False
        # The guard is one-frame only.  An interrupted/seeked decode must not
        # turn this into a global false-point rule.
        self._pending_provisional_static_candidate = None
        if self.frame_count != pending_frame + 1:
            return False

        raw_frames = list(getattr(self, "_local_ai_frame_buffer", []))
        current = raw_frames[-1].get("image") if raw_frames else None
        now_blob = self._static_blob_near(current, pending["pos"])
        if now_blob is None:
            print(
                f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
                f"candidate {pending['pos']} did not persist; no ignore"
            )
            return False

        prior_blob = pending.get("blob")
        prior_pos = prior_blob.get("pos") if prior_blob else pending["pos"]
        displacement = math.hypot(
            now_blob["pos"][0] - prior_pos[0], now_blob["pos"][1] - prior_pos[1]
        )
        if displacement <= 10.0:
            # A three-sample rule in practice: the candidate was already
            # measured as static against the prior frame, and it now survives
            # at the same position in the following frame.  Restrict the
            # ignore to a few frames so a later live ball remains eligible.
            self._learn_ignored_tracking_position(
                pending["pos"], radius=18, ttl=4,
                reason="provisional three-frame static artifact",
            )
            self._last_static_guard_position = tuple(pending["pos"])
            print(
                f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
                f"rejected persistent blob {pending['pos']} "
                f"delta={displacement:.1f}px"
            )
            return True
        else:
            print(
                f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
                f"candidate moved {displacement:.1f}px; no ignore"
            )
        return False

    def _try_active_tight_local_ai_hold(
            self, frame, tracked_position, previous_gray=None):
        """Advance an active contact ROI when normal tracking only holds.

        Player-occlusion branches can return the previous marker without
        reaching any candidate-commit hook.  During the bounded tight-ROI
        contact lock, rank that small area directly instead of waiting five
        frames for a broad full-frame reacquisition.
        """
        follow = getattr(self, '_local_ai_tight_roi_follow', None)
        if (
                not isinstance(follow, dict) or frame is None or
                int(self.frame_count) > int(follow.get('deadline', -1)) or
                int(self.frame_count) <= int(follow.get('last_frame', -1)) or
                int(getattr(self, 'stuck_frame_count', 0)) < 3):
            return None

        held_pos = tracked_position or self.ball_center or follow.get('last_pos')
        if held_pos is None:
            return None
        held_pos = (int(held_pos[0]), int(held_pos[1]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        held_motion = self._candidate_motion_metrics(
            frame_gray, held_pos[0], held_pos[1], previous_gray=previous_gray
        )
        held_mean = (
            float(held_motion.get('mean', 0.0) or 0.0)
            if held_motion else 0.0
        )
        held_max = (
            float(held_motion.get('max', 0.0) or 0.0)
            if held_motion else 0.0
        )
        height, width = frame.shape[:2]
        px = max(0, min(width - 1, held_pos[0]))
        py = max(0, min(height - 1, held_pos[1]))
        proposal = {
            'pos': held_pos,
            'area': float(getattr(self, 'ball_size', 0.0) or 0.0),
            'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
            'motion_mean': held_mean,
            'motion_max': held_max,
            'source': 'tight_roi_held_marker',
            'recovery_label': 'LOCAL AI TIGHT ROI HOLD',
            'tight_roi_force_rank': True,
        }
        preferred = self._prefer_night_tight_local_ai_candidate(
            frame, frame_gray, proposal, previous_gray=previous_gray
        )
        if preferred.get('source') != 'local_ai_tight_roi':
            return None
        return self._commit_night_visible_ball_recovery(preferred, frame)

    def _contact_local_ai_trigger(self, previous_position, tracked_position, snapshot):
        """Return a generic racket-contact takeover reason for suspicious HSV.

        The trigger is deliberately player-relative rather than serve-relative:
        it applies to serves, forehands, backhands and either player.  Merely
        being near a player is not enough.  We require either an actual player
        zone selection or a sharp trajectory/size discontinuity inside the
        padded racket-contact corridor.
        """
        if (
                getattr(self, 'local_ai_recovery', None) is None or
                previous_position is None or tracked_position is None or
                int(self.frame_count) <= int(getattr(
                    self, '_contact_local_ai_cooldown_until_frame', -1000000
                ))):
            return None

        previous = (int(previous_position[0]), int(previous_position[1]))
        tracked = (int(tracked_position[0]), int(tracked_position[1]))
        prior_motion = dict((snapshot or {}).get('last_motion') or {})
        prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
        prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
        prior_speed = float(
            prior_motion.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
        )
        proposed_dx = float(tracked[0] - previous[0])
        proposed_dy = float(tracked[1] - previous[1])
        proposed_speed = math.hypot(proposed_dx, proposed_dy)

        predicted = (
            int(round(float(previous[0]) + prior_dx)),
            int(round(float(previous[1]) + prior_dy)),
        )
        prediction_error = math.hypot(
            float(tracked[0] - predicted[0]), float(tracked[1] - predicted[1])
        )

        angle_delta = 0.0
        if prior_speed >= 4.0 and proposed_speed >= 4.0:
            prior_angle = math.degrees(math.atan2(prior_dy, prior_dx))
            proposed_angle = math.degrees(math.atan2(proposed_dy, proposed_dx))
            raw_delta = abs(proposed_angle - prior_angle) % 360.0
            angle_delta = min(raw_delta, 360.0 - raw_delta)

        zone = self._player_point_zone(tracked)
        contact_near = any(
            self._point_in_player_contact_corridor(point)
            for point in (previous, tracked, predicted)
        )
        if zone is None and not contact_near:
            return None

        recent_speeds = [
            float(value) for value in (snapshot or {}).get('ball_velocity_history', [])[-5:]
            if value is not None
        ]
        recent_median = float(np.median(recent_speeds)) if recent_speeds else prior_speed
        jump_floor = max(70.0, recent_median * 1.8, prior_speed * 1.55)

        previous_size = float((snapshot or {}).get('ball_size', 0.0) or 0.0)
        proposed_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
        size_ratio = (
            max(previous_size, proposed_size) / max(1.0, min(previous_size, proposed_size))
            if previous_size > 0.0 and proposed_size > 0.0 else 1.0
        )

        player_zones = (
            'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'
        )
        if zone in player_zones and max(prior_speed, proposed_speed) >= 12.0:
            return (
                f'player-zone:{zone}/step={proposed_speed:.0f}px/'
                f'angle={angle_delta:.0f}deg'
            )

        if (
                contact_near and proposed_speed >= jump_floor and
                (angle_delta >= 35.0 or prediction_error >= 50.0)):
            return (
                f'contact-jump:{proposed_speed:.0f}px>={jump_floor:.0f}/'
                f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
            )

        if (
                contact_near and prior_speed >= 18.0 and proposed_speed >= 18.0 and
                angle_delta >= 70.0 and prediction_error >= 30.0):
            return (
                f'contact-turn:{angle_delta:.0f}deg/'
                f'pred={prediction_error:.0f}px'
            )

        if (
                contact_near and proposed_speed >= 35.0 and size_ratio >= 2.6 and
                prediction_error >= 35.0):
            return (
                f'contact-size-jump:{size_ratio:.1f}x/'
                f'pred={prediction_error:.0f}px'
            )
        return None

    def _run_contact_local_ai_frame(self, frame, previous_position, arm_reason=None):
        """Let Local AI own a short generic racket-contact sequence.

        The implementation intentionally reuses the already validated
        --force-local-ai-range tracker: player/head/shoe/racket candidates are
        excluded before scoring, the AI-selected point becomes the next anchor,
        and trajectory prediction starts after two accepted AI positions.
        """
        recovery = getattr(self, 'local_ai_recovery', None)
        if recovery is None or frame is None or previous_position is None:
            return None

        current = int(self.frame_count)
        state = getattr(self, '_contact_local_ai_state', None)
        if arm_reason is not None and not isinstance(state, dict):
            state = {
                'start_frame': current,
                'deadline': current + 22,
                'history': [],
                'outside_count': 0,
                'reason': str(arm_reason),
            }
            self._contact_local_ai_state = state
            # Do not let an older tight-ROI experiment compete with contact
            # ownership.  The automatic owner is the stronger, player-excluding
            # path and will hand control back explicitly.
            self._local_ai_tight_roi_follow = None
            print(
                f"[CONTACT_LOCAL_AI_ARM] f{current}: anchor={tuple(previous_position)} "
                f"reason={arm_reason} radius={self._contact_local_ai_radius:.0f}px"
            )

        state = getattr(self, '_contact_local_ai_state', None)
        if not isinstance(state, dict):
            return None
        if current > int(state.get('deadline', -1)):
            print(
                f"[CONTACT_LOCAL_AI_EXPIRE] f{current}: "
                f"started=f{state.get('start_frame')} reason={state.get('reason')}"
            )
            self._contact_local_ai_state = None
            self._contact_local_ai_cooldown_until_frame = current + 3
            return None

        # Reuse the proven forced-AI implementation without exposing its state
        # to the user's manual debug range.
        saved_range = getattr(self, '_force_local_ai_range', None)
        saved_history = list(getattr(self, '_force_local_ai_history', []) or [])
        saved_radius = float(getattr(self, '_force_local_ai_radius', 450.0))
        saved_min_score = float(getattr(self, '_force_local_ai_min_score', 0.985))
        try:
            self._force_local_ai_range = (
                int(state['start_frame']), int(state['deadline'])
            )
            self._force_local_ai_history = list(state.get('history', []) or [])
            self._force_local_ai_radius = float(
                getattr(self, '_contact_local_ai_radius', 450.0)
            )
            self._force_local_ai_min_score = float(
                getattr(self, '_contact_local_ai_min_score', 0.985)
            )
            selected = self._force_local_ai_frame(frame, previous_position)
            state['history'] = list(getattr(self, '_force_local_ai_history', []) or [])
        finally:
            self._force_local_ai_range = saved_range
            self._force_local_ai_history = saved_history
            self._force_local_ai_radius = saved_radius
            self._force_local_ai_min_score = saved_min_score

        if selected is None:
            return None

        selected = (int(selected[0]), int(selected[1]))
        zone = self._player_point_zone(selected)
        inside_contact = self._point_in_player_contact_corridor(selected)
        age = current - int(state['start_frame'])
        if zone is None and not inside_contact:
            state['outside_count'] = int(state.get('outside_count', 0)) + 1
        else:
            state['outside_count'] = 0

        print(
            f"[CONTACT_LOCAL_AI] f{current}: selected={selected} age={age}f "
            f"outside={state['outside_count']} zone={zone}"
        )

        # Keep AI for several frames after contact.  This prevents an immediate
        # handback to HSV while its search anchor is still the racket/player.
        if age >= 5 and int(state.get('outside_count', 0)) >= 3:
            print(
                f"[CONTACT_LOCAL_AI_HANDOFF] f{current}: selected={selected} "
                f"clear of player/contact corridor for {state['outside_count']}f"
            )
            self._contact_local_ai_state = None
            self._contact_local_ai_cooldown_until_frame = current + 3
        else:
            self._contact_local_ai_state = state
        return selected

    def _force_local_ai_frame(self, frame, previous_position):
        """TEST ONLY: bypass HSV and let Local AI own an exact frame range.

        The previous AI-selected point becomes the next search anchor. Player,
        head, shoe, and racket regions are excluded before scoring. Once two AI
        points exist, high-confidence candidates must also fit the extrapolated
        AI trajectory. Outside --force-local-ai-range this method returns None
        and the normal tracker is unchanged.
        """
        frame_range = getattr(self, "_force_local_ai_range", None)
        recovery = getattr(self, "local_ai_recovery", None)
        if frame_range is None or frame is None or recovery is None:
            return None

        start_frame, end_frame = frame_range
        current = int(self.frame_count)
        if not (int(start_frame) <= current <= int(end_frame)):
            if current > int(end_frame):
                self._force_local_ai_history = []
            return None

        from ball_ai_recovery_probe import collect_candidates

        history = list(getattr(self, "_force_local_ai_history", []) or [])
        if history and int(history[-1].get("frame", -1000000)) != current - 1:
            history = []

        anchor = tuple(history[-1]["pos"]) if history else (
            tuple(previous_position) if previous_position is not None else None
        )
        if anchor is None:
            print(f"[FORCE_LOCAL_AI_HOLD] f{current}: no anchor available")
            return previous_position

        radius = max(40.0, float(getattr(self, "_force_local_ai_radius", 450.0)))
        min_score = float(getattr(self, "_force_local_ai_min_score", 0.985))

        config = dict(recovery._config)
        config["force_low_sat"] = {
            "h_min": 80, "h_max": 135,
            "s_min": 10, "s_max": 60,
            "v_min": 110, "v_max": 255,
        }
        modes = (
            "regular_court", "alt1", "alt2", "alt3", "s_30", "h_10",
            "force_low_sat",
        )

        predicted = None
        previous_speed = 0.0
        previous_vector = None
        if len(history) >= 2:
            p0 = tuple(history[-2]["pos"])
            p1 = tuple(history[-1]["pos"])
            vx = float(p1[0] - p0[0])
            vy = float(p1[1] - p0[1])
            previous_speed = math.hypot(vx, vy)
            if previous_speed >= 3.0:
                previous_vector = (vx, vy)
                predicted = (int(round(p1[0] + vx)), int(round(p1[1] + vy)))

        candidates = collect_candidates(
            frame, config, modes=modes,
            min_area=3.0, max_area=1200.0,
            around=anchor, radius=radius,
            dedup_distance=4.0,
        )

        clear_candidates = []
        for candidate in candidates:
            point = (int(candidate["x"]), int(candidate["y"]))
            if self._player_point_zone(point) is None:
                clear_candidates.append(candidate)

        sort_anchor = predicted if predicted is not None else anchor
        clear_candidates.sort(
            key=lambda item: math.hypot(
                float(item["x"]) - float(sort_anchor[0]),
                float(item["y"]) - float(sort_anchor[1]),
            )
        )
        subset = clear_candidates[:128]
        scored = recovery._score(frame, current, subset) if subset else []

        tolerance = None
        if predicted is not None:
            tolerance = max(120.0, min(300.0, previous_speed * 1.8 + 60.0))

        eligible = []
        for candidate in scored:
            score = float(candidate.get("ai_score", 0.0) or 0.0)
            if score < min_score:
                continue
            point = (int(candidate["x"]), int(candidate["y"]))
            if self._player_point_zone(point) is not None:
                continue

            pred_dist = None
            cosine = None
            if predicted is not None:
                pred_dist = math.hypot(point[0] - predicted[0], point[1] - predicted[1])
                if pred_dist > float(tolerance):
                    continue
                if previous_vector is not None:
                    sx = float(point[0] - anchor[0])
                    sy = float(point[1] - anchor[1])
                    step = math.hypot(sx, sy)
                    if step >= 3.0 and previous_speed >= 3.0:
                        cosine = (
                            previous_vector[0] * sx + previous_vector[1] * sy
                        ) / (previous_speed * step)
                        if cosine < -0.20:
                            continue

            eligible.append({
                "candidate": candidate,
                "point": point,
                "score": score,
                "pred_dist": pred_dist,
                "cosine": cosine,
            })

        if predicted is None:
            eligible.sort(
                key=lambda item: (
                    -item["score"],
                    math.hypot(item["point"][0] - anchor[0], item["point"][1] - anchor[1]),
                )
            )
        else:
            eligible.sort(key=lambda item: (float(item["pred_dist"]), -item["score"]))

        for rank, item in enumerate(eligible[:5], 1):
            pd = "n/a" if item["pred_dist"] is None else f"{item['pred_dist']:.1f}"
            cs = "n/a" if item["cosine"] is None else f"{item['cosine']:.3f}"
            print(
                f"[FORCE_LOCAL_AI_CAND] f{current} #{rank} pos={item['point']} "
                f"ai={item['score']:.6f} pred_dist={pd} cos={cs}"
            )

        if not eligible:
            self.ball_center = tuple(anchor)
            self.stuck_frame_count = max(1, int(getattr(self, "stuck_frame_count", 0)) + 1)
            print(
                f"[FORCE_LOCAL_AI_HOLD] f{current}: no clear candidate >= {min_score:.3f}; "
                f"anchor={anchor} predicted={predicted} candidates={len(candidates)} "
                f"clear={len(clear_candidates)} scored={len(scored)}"
            )
            return tuple(anchor)

        selected = eligible[0]
        candidate = selected["candidate"]
        point = selected["point"]
        area = float(candidate.get("area", 0.0) or 0.0)

        dx = int(point[0] - anchor[0])
        dy = int(point[1] - anchor[1])
        distance = math.hypot(dx, dy)
        direction_deg = math.degrees(math.atan2(dy, dx)) if distance > 0.0 else None

        self.ball_center = point
        self.ball_size = area
        height, width = frame.shape[:2]
        px = max(0, min(width - 1, point[0]))
        py = max(0, min(height - 1, point[1]))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.ball_hsv = hsv_frame[py, px]

        self.prev_motion = self.last_motion
        self.last_motion = {
            "distance": float(distance), "dx": dx, "dy": dy,
            "direction_deg": direction_deg,
        }
        if distance >= 3.0:
            self.last_nonzero_motion = dict(self.last_motion)
        if direction_deg is not None:
            self.last_direction = direction_deg
        self.last_delta = (dx, dy)
        self.stuck_frame_count = 0
        self.ball_stopped = False
        self.last_seen_frame = current

        self.ball_velocity_history.append(float(distance))
        if len(self.ball_velocity_history) > 10:
            self.ball_velocity_history = self.ball_velocity_history[-10:]
        self.motion_history.append({
            "frame": current,
            "distance": float(distance),
            "direction_deg": direction_deg,
            "pos": tuple(point),
            "prev_pos": tuple(anchor),
            "forced_local_ai": True,
        })
        if len(self.motion_history) > 200:
            self.motion_history = self.motion_history[-200:]
        self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        history.append({"frame": current, "pos": tuple(point), "score": selected["score"]})
        self._force_local_ai_history = history[-4:]

        print(
            f"[FORCE_LOCAL_AI] f{current}: selected={point} ai={selected['score']:.6f} "
            f"area={area:.1f}px anchor={anchor} predicted={predicted} "
            f"normal_hsv_bypassed=True"
        )
        return tuple(point)

    def _debug_local_ai_shadow_frame(self, frame, previous_position, normal_position):
        """Inspect raw Local-AI ranking and optional HSV sweeps without changing tracking."""
        frame_range = getattr(self, "_debug_local_ai_range", None)
        if frame_range is None or frame is None or self.local_ai_recovery is None:
            return
        start_frame, end_frame = frame_range
        if not (int(start_frame) <= int(self.frame_count) <= int(end_frame)):
            return

        from ball_ai_recovery_probe import collect_candidates

        top_n = max(1, int(getattr(self, "_debug_local_ai_top_n", 10)))
        do_hsv_sweep = bool(getattr(self, "_debug_hsv_sweep", False))
        radius = float(self._debug_local_ai_radius)
        out_dir = self.local_ai_recovery.work_dir / "shadow_frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        anchors = []
        for label, point in (("previous", previous_position), ("normal", normal_position)):
            if point is None:
                continue
            point = (int(point[0]), int(point[1]))
            if any(existing[1] == point for existing in anchors):
                continue
            anchors.append((label, point))

        def score_candidates(anchor, config, modes, *, pool_limit=96):
            candidates = collect_candidates(
                frame, config, modes=modes,
                min_area=3.0, max_area=2000.0,
                around=anchor, radius=radius,
            )
            ordered = sorted(
                candidates,
                key=lambda item: math.hypot(item["x"] - anchor[0], item["y"] - anchor[1]),
            )
            subset = ordered[:max(top_n, min(int(pool_limit), 96))]
            scored = self.local_ai_recovery._score(frame, int(self.frame_count), subset) if subset else []
            ranked = sorted(
                scored,
                key=lambda item: float(item.get("ai_score", 0.0) or 0.0),
                reverse=True,
            )
            return candidates, ranked

        def print_ranked(prefix, anchor_label, anchor, variant, candidates, ranked, limit):
            print(
                f"[{prefix}] f{self.frame_count} anchor={anchor_label}:{anchor} "
                f"variant={variant} contours={len(candidates)} scored={len(ranked)} radius={radius:.0f}px"
            )
            for rank, candidate in enumerate(ranked[:limit], 1):
                point = (int(candidate["x"]), int(candidate["y"]))
                zone = self._player_point_zone(point) or "clear"
                distance = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
                print(
                    f"[{prefix}_TOP] f{self.frame_count} {variant} #{rank} "
                    f"pos={point} ai={float(candidate.get('ai_score', 0.0) or 0.0):.6f} "
                    f"zone={zone} dist={distance:.1f}px area={float(candidate.get('area', 0.0) or 0.0):.1f} "
                    f"mode={candidate.get('mode', '?')}"
                )

        def save_rank_image(anchor_label, anchor, variant, ranked, limit):
            image = frame.copy()
            cv2.circle(image, anchor, 14, (0, 255, 255), 2)
            cv2.putText(image, f"anchor:{anchor_label}", (anchor[0] + 10, anchor[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            for rank, candidate in enumerate(ranked[:limit], 1):
                point = (int(candidate["x"]), int(candidate["y"]))
                score = float(candidate.get("ai_score", 0.0) or 0.0)
                cv2.circle(image, point, 16, (255, 0, 255), 2)
                cv2.putText(image, f"#{rank} {score:.3f}", (point[0] + 8, point[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            safe_variant = variant.replace(':', '_').replace('/', '_')
            output_path = out_dir / (
                f"shadow_f{int(self.frame_count):08d}_{anchor_label}_{safe_variant}.jpg"
            )
            cv2.imwrite(str(output_path), image)
            print(f"[LOCAL_AI_SHADOW_IMAGE] f{self.frame_count} variant={variant} saved={output_path}")

        config = self.local_ai_recovery._config
        default_modes = ("regular_court", "alt1", "alt2", "alt3", "s_30", "h_10")
        for anchor_label, anchor in anchors:
            candidates, ranked = score_candidates(anchor, config, default_modes)
            print_ranked("LOCAL_AI_SHADOW", anchor_label, anchor, "combined", candidates, ranked, top_n)
            save_rank_image(anchor_label, anchor, "combined", ranked, top_n)

            if not do_hsv_sweep:
                continue
            regular = config.get("regular_court")
            alt2 = config.get("alt2")
            if not isinstance(regular, dict) or not isinstance(alt2, dict):
                print(f"[HSV_SWEEP] f{self.frame_count} skipped: regular_court/alt2 missing")
                continue

            hsv_keys = ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max")
            for name, fraction in (
                ("regular", 0.0),
                ("mix25", 0.25),
                ("mix50", 0.50),
                ("mix75", 0.75),
                ("alt2", 1.0),
            ):
                values = {}
                for key in hsv_keys:
                    a = float(regular.get(key, 0))
                    b = float(alt2.get(key, a))
                    values[key] = int(round(a + (b - a) * fraction))
                sweep_config = {"debug_sweep": values}
                candidates, ranked = score_candidates(anchor, sweep_config, ("debug_sweep",))
                hsv_text = (
                    f"H={values['h_min']}-{values['h_max']} "
                    f"S={values['s_min']}-{values['s_max']} "
                    f"V={values['v_min']}-{values['v_max']}"
                )
                print(
                    f"[HSV_SWEEP] f{self.frame_count} anchor={anchor_label}:{anchor} "
                    f"variant={name} {hsv_text} contours={len(candidates)} scored={len(ranked)}"
                )
                print_ranked(
                    "HSV_SWEEP", anchor_label, anchor, name, candidates, ranked, min(top_n, 5)
                )
                save_rank_image(anchor_label, anchor, f"hsv_{name}", ranked, min(top_n, 5))

    def _try_local_ai_trajectory_rescue(
            self, frame, previous_position, tracked_position, previous_stuck,
            pre_track_snapshot, reason):
        """Rescue a held player-occlusion track using physics plus Local AI.

        Ordinary tracking uses one constant-velocity prediction.  Immediately
        after a rejected post-serve handoff, however, the first accepted
        racket-to-ball displacement is not a trustworthy one-frame velocity:
        part of that displacement can be the visual re-acquisition jump itself.
        In that bounded recovery state, treat the vector as a direction with an
        along-track uncertainty corridor.  Candidates still need independent
        AI, motion, size, direction, and player-zone support before acceptance.
        """
        recovery = getattr(self, "local_ai_recovery", None)
        ranker = getattr(recovery, "rank_local_roi_candidates", None)
        corridor_ranker = getattr(recovery, "rank_local_corridor_candidates", None)
        if (
                recovery is None or not callable(ranker) or frame is None or
                previous_position is None or
                not str(reason or "").startswith("player-region:")
        ):
            return None

        snapshot = pre_track_snapshot or {}
        last_motion = snapshot.get("last_motion") or {}
        last_nonzero = snapshot.get("last_nonzero_motion") or {}
        prior_distance = float(last_motion.get("distance", 0.0) or 0.0)
        prior = last_motion if prior_distance >= 5.0 else (last_nonzero or last_motion)
        prior_dx = float(prior.get("dx", 0.0) or 0.0)
        prior_dy = float(prior.get("dy", 0.0) or 0.0)
        prior_speed = math.hypot(prior_dx, prior_dy)
        if prior_speed < 6.0:
            return None

        if tracked_position is None:
            held_distance = 0.0
        else:
            held_distance = math.hypot(
                float(tracked_position[0]) - float(previous_position[0]),
                float(tracked_position[1]) - float(previous_position[1]),
            )
        held_limit = max(3.0, min(10.0, prior_speed * 0.25))
        if held_distance > held_limit and int(previous_stuck or 0) <= 0:
            return None

        predicted = (
            int(round(float(previous_position[0]) + prior_dx)),
            int(round(float(previous_position[1]) + prior_dy)),
        )
        post_serve_corridor_active = (
            int(self.frame_count) <= int(
                getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)
            )
        )
        launch_direction_x = int(getattr(self, "_serve_launch_direction_x", 0) or 0)

        corridor_min_progress = None
        corridor_max_progress = None
        corridor_cross_cap = None
        corridor_anchors = [predicted]
        roi_radius = max(18.0, min(42.0, prior_speed * 0.80 + 7.0))

        if post_serve_corridor_active:
            # The first visual handoff may greatly overstate one-frame speed.
            # Keep the trusted direction, but search a bounded range of progress
            # along that direction.  All values scale from observed motion; no
            # frame, player location, or court coordinate is encoded here.
            unit_dx = prior_dx / prior_speed
            unit_dy = prior_dy / prior_speed
            corridor_min_progress = max(12.0, min(90.0, prior_speed * 0.15))
            corridor_max_progress = max(
                corridor_min_progress + 32.0,
                min(520.0, prior_speed * 1.10 + 18.0),
            )
            corridor_cross_cap = max(26.0, min(84.0, prior_speed * 0.20 + 12.0))
            corridor_span = corridor_max_progress - corridor_min_progress
            anchor_spacing = max(45.0, corridor_cross_cap * 0.85)
            anchor_count = max(
                3,
                min(7, int(math.ceil(corridor_span / anchor_spacing)) + 1),
            )
            corridor_anchors = []
            for index in range(anchor_count):
                fraction = index / max(1, anchor_count - 1)
                progress = corridor_min_progress + corridor_span * fraction
                corridor_anchors.append((
                    int(round(float(previous_position[0]) + unit_dx * progress)),
                    int(round(float(previous_position[1]) + unit_dy * progress)),
                ))
            roi_radius = corridor_cross_cap
            print(
                f"[LOCAL_AI_POST_SERVE_CORRIDOR] f{self.frame_count}: "
                f"prev={previous_position} dir=({unit_dx:.3f},{unit_dy:.3f}) "
                f"progress={corridor_min_progress:.1f}-{corridor_max_progress:.1f}px "
                f"cross={corridor_cross_cap:.1f}px anchors={len(corridor_anchors)}"
            )

        try:
            if post_serve_corridor_active and callable(corridor_ranker):
                ranked = corridor_ranker(
                    int(self.frame_count),
                    frame,
                    anchors=corridor_anchors,
                    radius=roi_radius,
                    maximum_candidates=48,
                )
            elif post_serve_corridor_active:
                # Backward-compatible fallback if an older recovery object is
                # injected by a test or external caller.  De-duplicate repeated
                # ROI results before trajectory arbitration.
                merged = {}
                for anchor in corridor_anchors:
                    for candidate in ranker(
                            int(self.frame_count),
                            frame,
                            anchor=anchor,
                            radius=roi_radius,
                            maximum_candidates=10):
                        key = (
                            int(candidate.get("x", 0)),
                            int(candidate.get("y", 0)),
                            str(candidate.get("mode") or ""),
                        )
                        current = merged.get(key)
                        if (
                                current is None or
                                float(candidate.get("ai_score", 0.0) or 0.0) >
                                float(current.get("ai_score", 0.0) or 0.0)
                        ):
                            merged[key] = dict(candidate)
                ranked = sorted(
                    merged.values(),
                    key=lambda item: float(item.get("ai_score", 0.0) or 0.0),
                    reverse=True,
                )
            else:
                ranked = ranker(
                    int(self.frame_count),
                    frame,
                    anchor=predicted,
                    radius=roi_radius,
                    maximum_candidates=10,
                )
        except Exception as exc:
            print(
                f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: scorer failed "
                f"({type(exc).__name__}: {exc})"
            )
            return None

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = snapshot.get("_prev_frame_gray")
        previous_size = float(snapshot.get("ball_size", 0.0) or 0.0)
        prediction_cap = (
            float(corridor_cross_cap)
            if post_serve_corridor_active else
            max(9.0, min(22.0, prior_speed * 0.65))
        )
        min_step = (
            max(4.0, float(corridor_min_progress) * 0.70)
            if post_serve_corridor_active else
            max(4.0, prior_speed * 0.35)
        )
        max_step = (
            float(corridor_max_progress) + float(corridor_cross_cap) * 0.50
            if post_serve_corridor_active else
            max(28.0, prior_speed * 1.80 + 8.0)
        )
        diagnostics = []
        valid = []
        candidate_limit = 48 if post_serve_corridor_active else 10

        unit_dx = prior_dx / prior_speed
        unit_dy = prior_dy / prior_speed
        for candidate in ranked[:candidate_limit]:
            point = (int(candidate["x"]), int(candidate["y"]))
            ai_score = float(candidate.get("ai_score", 0.0) or 0.0)
            area = float(candidate.get("area", 0.0) or 0.0)
            zone = self._player_point_zone(point)
            step_dx = float(point[0]) - float(previous_position[0])
            step_dy = float(point[1]) - float(previous_position[1])
            step = math.hypot(step_dx, step_dy)
            direction_cos = (
                (step_dx * prior_dx + step_dy * prior_dy) / (step * prior_speed)
                if step > 1e-6 and prior_speed > 1e-6 else -1.0
            )

            corridor_along = None
            corridor_cross = None
            if post_serve_corridor_active:
                corridor_along = step_dx * unit_dx + step_dy * unit_dy
                corridor_cross = abs(step_dx * unit_dy - step_dy * unit_dx)
                along_miss = max(
                    float(corridor_min_progress) - corridor_along,
                    corridor_along - float(corridor_max_progress),
                    0.0,
                )
                prediction_error = math.hypot(corridor_cross, along_miss)
            else:
                prediction_error = math.hypot(
                    float(point[0]) - float(predicted[0]),
                    float(point[1]) - float(predicted[1]),
                )

            motion = self._candidate_motion_metrics(
                current_gray,
                point[0],
                point[1],
                radius=7,
                previous_gray=previous_gray,
            ) or {}
            motion_mean = float(motion.get("mean", 0.0) or 0.0)
            motion_max = float(motion.get("max", 0.0) or 0.0)
            area_ratio = area / previous_size if previous_size > 0.0 else 1.0

            lateral_ok = True
            if post_serve_corridor_active and launch_direction_x != 0:
                signed_lateral = step_dx * float(launch_direction_x)
                lateral_ok = signed_lateral >= -max(8.0, float(corridor_cross_cap) * 0.25)

            ai_floor = (
                (0.35 if zone is not None else 0.25)
                if post_serve_corridor_active else
                (0.30 if zone is not None else 0.25)
            )
            checks = {
                "prediction": prediction_error <= prediction_cap,
                "direction": direction_cos >= (0.50 if post_serve_corridor_active else 0.45),
                "step": min_step <= step <= max_step,
                "motion": motion_mean >= 4.0 or motion_max >= 25.0,
                "area": previous_size <= 0.0 or 0.12 <= area_ratio <= 5.0,
                "lateral": lateral_ok,
                "ai": ai_score >= ai_floor,
            }
            reject_reason = next((name for name, ok in checks.items() if not ok), None)
            source_penalty = 0.0 if candidate.get("mode") == "regular_court" else 0.75
            zone_penalty = 1.25 if (post_serve_corridor_active and zone is not None) else (0.75 if zone is not None else 0.0)
            trajectory_cost = (
                prediction_error +
                max(0.0, 1.0 - direction_cos) * (14.0 if post_serve_corridor_active else 10.0) +
                max(0.0, 1.0 - ai_score) * (10.0 if post_serve_corridor_active else 8.0) +
                source_penalty + zone_penalty
            )
            row = {
                "rank": int(candidate.get("roi_rank", len(diagnostics) + 1)),
                "position": point,
                "ai_score": ai_score,
                "mode": candidate.get("mode"),
                "zone": zone,
                "area": area,
                "area_ratio": area_ratio,
                "prediction_error": prediction_error,
                "direction_cos": direction_cos,
                "step": step,
                "motion_mean": motion_mean,
                "motion_max": motion_max,
                "corridor_along": corridor_along,
                "corridor_cross": corridor_cross,
                "trajectory_cost": trajectory_cost,
                "valid": reject_reason is None,
                "reject_reason": reject_reason,
            }
            diagnostics.append(row)
            if reject_reason is None:
                valid.append((trajectory_cost, prediction_error, -ai_score, candidate, row))

        for row in diagnostics[:8 if post_serve_corridor_active else 5]:
            verdict = "OK" if row["valid"] else f"reject:{row['reject_reason']}"
            corridor_detail = ""
            if post_serve_corridor_active:
                corridor_detail = (
                    f" lane={row['corridor_along']:.1f}/"
                    f"{row['corridor_cross']:.1f}px"
                )
            print(
                f"[LOCAL_AI_TRAJECTORY_CAND] f{self.frame_count} "
                f"#{row['rank']} pos={row['position']} ai={row['ai_score']:.6f} "
                f"mode={row['mode'] or 'unknown'} "
                f"err={row['prediction_error']:.1f}/{prediction_cap:.1f}px "
                f"cos={row['direction_cos']:.2f} step={row['step']:.1f}px"
                f"{corridor_detail} "
                f"motion={row['motion_mean']:.1f}/{row['motion_max']:.1f} "
                f"area_ratio={row['area_ratio']:.2f} zone={row['zone'] or 'clear'} "
                f"cost={row['trajectory_cost']:.2f} {verdict}"
            )

        selected = min(valid, key=lambda item: (item[0], item[1], item[2])) if valid else None
        selected_candidate = dict(selected[3]) if selected is not None else None
        selected_row = selected[4] if selected is not None else None

        payload = {
            "frame": int(self.frame_count),
            "stage": "trajectory-rescue",
            "reason": str(reason),
            "previous_position": tuple(previous_position),
            "normal_position": tuple(tracked_position) if tracked_position is not None else None,
            "previous_stuck": int(previous_stuck or 0),
            "prior_motion": {"dx": prior_dx, "dy": prior_dy, "speed": prior_speed},
            "predicted_position": predicted,
            "roi_radius": roi_radius,
            "prediction_cap": prediction_cap,
            "post_serve_corridor": post_serve_corridor_active,
            "corridor_anchors": corridor_anchors if post_serve_corridor_active else None,
            "corridor_min_progress": corridor_min_progress,
            "corridor_max_progress": corridor_max_progress,
            "corridor_cross_cap": corridor_cross_cap,
            "candidates": diagnostics,
            "accepted": selected_candidate is not None,
            "selected": selected_row,
        }
        writer = getattr(recovery, "_write_event", None)
        if callable(writer):
            try:
                writer(payload)
            except Exception as exc:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: "
                    f"debug event write failed ({type(exc).__name__}: {exc})"
                )

        work_dir = getattr(recovery, "work_dir", None)
        if work_dir is not None:
            try:
                debug_dir = os.path.join(str(work_dir), "trajectory_frames")
                os.makedirs(debug_dir, exist_ok=True)
                debug_image = frame.copy()
                cv2.circle(debug_image, tuple(map(int, previous_position)), 13, (0, 255, 255), 2)
                if post_serve_corridor_active:
                    cv2.line(
                        debug_image,
                        tuple(map(int, corridor_anchors[0])),
                        tuple(map(int, corridor_anchors[-1])),
                        (255, 255, 0),
                        2,
                    )
                    for anchor in corridor_anchors:
                        cv2.circle(debug_image, tuple(map(int, anchor)), 7, (255, 255, 0), 1)
                else:
                    cv2.circle(debug_image, predicted, int(round(roi_radius)), (255, 255, 0), 2)
                if tracked_position is not None:
                    cv2.circle(debug_image, tuple(map(int, tracked_position)), 11, (0, 0, 255), 2)
                for row in diagnostics[:8 if post_serve_corridor_active else 5]:
                    point = tuple(map(int, row["position"]))
                    cv2.circle(debug_image, point, 12, (255, 0, 255), 2)
                    label_error = row["corridor_cross"] if post_serve_corridor_active else row["prediction_error"]
                    cv2.putText(
                        debug_image,
                        f"#{row['rank']} e{label_error:.0f} a{row['ai_score']:.2f}",
                        (point[0] + 7, point[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                    )
                output_path = os.path.join(
                    debug_dir, f"trajectory_f{int(self.frame_count):08d}.jpg"
                )
                cv2.imwrite(output_path, debug_image)
                print(f"[LOCAL_AI_TRAJECTORY_IMAGE] f{self.frame_count} saved={output_path}")
            except Exception as exc:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: "
                    f"debug image failed ({type(exc).__name__}: {exc})"
                )

        if selected_candidate is None:
            if post_serve_corridor_active:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: no safe candidate "
                    f"prev={previous_position} corridor="
                    f"{corridor_min_progress:.1f}-{corridor_max_progress:.1f}px "
                    f"cross={corridor_cross_cap:.1f}px"
                )
            else:
                print(
                    f"[LOCAL_AI_TRAJECTORY] f{self.frame_count}: no safe candidate "
                    f"prev={previous_position} predicted={predicted} radius={roi_radius:.1f}px"
                )
            return None

        point = (int(selected_candidate["x"]), int(selected_candidate["y"]))
        step_dx = float(point[0]) - float(previous_position[0])
        step_dy = float(point[1]) - float(previous_position[1])
        step = math.hypot(step_dx, step_dy)
        selected_candidate["trajectory_rescue"] = True
        selected_candidate["trajectory_prediction"] = predicted
        selected_candidate["trajectory_prediction_error"] = float(selected_row["prediction_error"])
        selected_candidate["trajectory_cost"] = float(selected_row["trajectory_cost"])
        if post_serve_corridor_active:
            selected_candidate["trajectory_post_serve_corridor"] = True
            selected_candidate["trajectory_corridor_along"] = float(selected_row["corridor_along"])
            selected_candidate["trajectory_corridor_cross"] = float(selected_row["corridor_cross"])
        selected_candidate["trajectory_motion"] = {
            "dx": step_dx,
            "dy": step_dy,
            "distance": step,
            "direction_deg": math.degrees(math.atan2(step_dy, step_dx)) if step > 0.0 else None,
        }
        mode_label = " corridor" if post_serve_corridor_active else ""
        print(
            f"[LOCAL_AI_TRAJECTORY_ACCEPT] f{self.frame_count}:{mode_label} "
            f"{point} ai={float(selected_candidate.get('ai_score', 0.0)):.6f} "
            f"pred={predicted} err={selected_row['prediction_error']:.1f}px "
            f"motion={selected_row['motion_mean']:.1f}/{selected_row['motion_max']:.1f} "
            f"zone={selected_row['zone'] or 'clear'} cost={selected_row['trajectory_cost']:.2f}"
        )
        return selected_candidate

    def _try_local_ai_recovery(
            self, previous_position, tracked_position, previous_stuck,
            pre_track_snapshot=None, frame=None):
        """Repair a suspect selection from the recent raw-frame buffer if safe."""
        confirmed_static_artifact = self._finalize_pending_provisional_static_candidate()
        if (
                confirmed_static_artifact and previous_position is not None and
                tracked_position is not None and pre_track_snapshot is not None and
                math.hypot(
                    float(tracked_position[0]) - float(getattr(self, "_last_static_guard_position", tracked_position)[0]),
                    float(tracked_position[1]) - float(getattr(self, "_last_static_guard_position", tracked_position)[1]),
                ) <= 22.0):
            self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
            # Re-run the normal candidate selection now that the confirmed
            # static blob is ignored.  This keeps the real next-frame toss
            # (a prior frame here) instead of deliberately dropping a visible ball.
            retracked = self.track_ball_in_frame(frame) if frame is not None else None
            if retracked is not None and math.hypot(
                    float(retracked[0]) - float(self._last_static_guard_position[0]),
                    float(retracked[1]) - float(self._last_static_guard_position[1]),
            ) > 22.0:
                print(
                    f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
                    f"retracked {retracked} after excluding static selection"
                )
                return retracked
            self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
            self.stuck_frame_count = int(previous_stuck) + 1
            print(
                f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
                f"held current static selection {tracked_position}; resuming from {previous_position}"
            )
            return previous_position
        # Keep a weak player/background reacquisition from becoming the
        # next tracking anchor before tight local AI can arbitrate it.
        if (
                pre_track_snapshot is not None and
                previous_position is not None and
                tracked_position is not None and
                int(previous_stuck or 0) >= 3):
            reacq_jump = math.hypot(
                float(tracked_position[0]) - float(previous_position[0]),
                float(tracked_position[1]) - float(previous_position[1]),
            )
            reacq_motion_mean = float(
                getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
            )
            reacq_motion_max = float(
                getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
            )
            reacq_zone = self._player_point_zone(tracked_position)
            weak_player_candidate = (
                reacq_zone is not None and
                reacq_motion_mean < 8.0 and
                reacq_motion_max < 50.0
            )
            weak_long_jump = (
                reacq_jump >= 150.0 and
                reacq_motion_mean < 12.0 and
                reacq_motion_max < 80.0
            )
            if weak_player_candidate or weak_long_jump:
                rejected_position = tuple(tracked_position)
                self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
                self.stuck_frame_count = int(previous_stuck or 0) + 1
                tracked_position = tuple(previous_position)
                if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
                    self._last_motion_reacq_frame = -1000000
                    self._last_motion_reacq_pos = None
                if weak_player_candidate:
                    print(
                        f'[PLAYER_REACQ_WEAK_PLAYER_REJECTED] f{self.frame_count}: '
                        f'rejected {rejected_position} zone={reacq_zone} '
                        f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
                        f'restored {previous_position} stuck={self.stuck_frame_count}'
                    )
                else:
                    print(
                        f'[PLAYER_REACQ_WEAK_JUMP_REJECTED] f{self.frame_count}: '
                        f'rejected {rejected_position} jump={reacq_jump:.1f}px '
                        f'motion={reacq_motion_mean:.1f}/{reacq_motion_max:.1f}; '
                        f'restored {previous_position} stuck={self.stuck_frame_count}'
                    )

        tight_hold_recovery = self._try_active_tight_local_ai_hold(
            frame,
            tracked_position,
            previous_gray=(pre_track_snapshot or {}).get('_prev_frame_gray'),
        )
        if tight_hold_recovery is not None:
            return tight_hold_recovery
        reason = self._local_ai_recovery_reason(
            previous_position, tracked_position, previous_stuck
        )
        if reason is None:
            self._maybe_clear_post_serve_pre_net_recovery(
                previous_position, tracked_position, source='normal-tracker'
            )
            return tracked_position
        post_serve_recovery_active = (
            bool(getattr(self, '_post_serve_pre_net_recovery_active', False)) or
            int(self.frame_count) <= max(
                int(getattr(self, "_post_serve_launch_lock_until_frame", -1)),
                int(getattr(self, "_post_serve_recovery_hold_until_frame", -1000000)),
            )
        )
        trajectory_recovered = self._try_local_ai_trajectory_rescue(
            frame,
            previous_position,
            tracked_position,
            previous_stuck,
            pre_track_snapshot,
            reason,
        )
        if trajectory_recovered is not None:
            recovered = trajectory_recovered
            reason = f"{reason}+trajectory"
        else:
            recovery_reason = reason
            if post_serve_recovery_active and reason.startswith("player-region:"):
                # Keep using the post-serve low-saturation candidate pool on
                # the frames after the first rejected jump.  The normal tracker
                # may keep proposing a player contour until the ball emerges.
                recovery_reason = f"post-serve-launch-occlusion:{reason}"
                print(
                    f"[POST_SERVE_RECOVERY_RETRY] f{self.frame_count}: "
                    f"{reason} -> {recovery_reason}"
                )
            recovered = self.local_ai_recovery.recover(
                self.frame_count,
                self._local_ai_frame_buffer,
                predicted_position=previous_position,
                player_zone=self._player_point_zone,
                reason=recovery_reason,
                force=False,
            )
        if recovered is None:
            if getattr(self.local_ai_recovery, "last_rejection", None) == "all-player-body-path":
                self._local_ai_all_body_rejections = int(
                    getattr(self, "_local_ai_all_body_rejections", 0)
                ) + 1
                point_age = int(self.frame_count) - int(
                    getattr(self, "point_start_frame_internal", self.frame_count) or self.frame_count
                )
                if (
                        self._local_ai_all_body_rejections >= 1 and
                        not post_serve_recovery_active and
                        # A previously accepted non-body recovery is positive
                        # evidence that this is a real serve/rally.  A later
                        # player-body-only attempt can happen at contact or
                        # occlusion and must not erase that valid point.
                        int(getattr(self, "_local_ai_recovery_count", 0)) == 0 and
                        int(getattr(self, "_point_hit_count", 0)) == 0 and
                        point_age <= 150):
                    self._discard_provisional_serve_from_ai = True
            else:
                self._local_ai_all_body_rejections = 0
            print(f"[LOCAL_AI_RECOVERY] f{self.frame_count}: no safe path ({reason})")
            # POST_SERVE_STATE_ROLLBACK_V3
            # The reason detector runs after normal HSV tracking has already
            # mutated ball_center/motion/history.  If the post-serve guard says
            # that mutation is unsafe and AI cannot replace it, restore the
            # complete pre-track snapshot immediately.  Returning the previous
            # position makes the main loop log/record the held sane point rather
            # than the rejected player pixel.
            if (
                    post_serve_recovery_active and
                    pre_track_snapshot is not None and
                    previous_position is not None and
                    (
                        reason.startswith("post-serve-launch-") or
                        reason.startswith("player-region:")
                    )):
                rejected_position = (
                    tuple(tracked_position) if tracked_position is not None else None
                )
                self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
                self.stuck_frame_count = max(
                    int(previous_stuck or 0) + 1,
                    int(getattr(self, "stuck_frame_count", 0) or 0),
                )
                if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
                    self._last_motion_reacq_frame = -1000000
                    self._last_motion_reacq_pos = None
                print(
                    f"[POST_SERVE_LAUNCH_ROLLBACK] f{self.frame_count}: "
                    f"rejected={rejected_position} reason={reason}; "
                    f"restored={previous_position} stuck={self.stuck_frame_count} "
                    f"retry_through=f{int(getattr(self, '_post_serve_recovery_hold_until_frame', -1))}"
                )
                return tuple(previous_position)
            # If HSV just teleported from a physically coherent ball path into a
            # player region and local AI could not verify a replacement, never let
            # that player pixel become the next-frame anchor. Restore the complete
            # pre-track state (position, size, motion and velocity history) and let
            # the rapid AI retry continue from the last sane trajectory.
            if (
                    reason.startswith("player-region:") and
                    pre_track_snapshot is not None and
                    previous_position is not None and
                    tracked_position is not None):
                player_jump = math.hypot(
                    float(tracked_position[0]) - float(previous_position[0]),
                    float(tracked_position[1]) - float(previous_position[1]),
                )
                prior_motion = pre_track_snapshot.get("last_motion") or {}
                prior_speed = float(prior_motion.get("distance", 0.0) or 0.0)
                prior_velocities = pre_track_snapshot.get("ball_velocity_history") or []
                if prior_velocities:
                    prior_speed = max(
                        prior_speed,
                        max(float(value or 0.0) for value in prior_velocities[-3:]),
                    )
                jump_limit = max(150.0, min(520.0, prior_speed * 3.2 + 60.0))
                if player_jump > jump_limit:
                    rejected_position = tuple(tracked_position)
                    self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
                    self.stuck_frame_count = int(previous_stuck) + 1
                    if getattr(self, "_last_motion_reacq_frame", -1000000) == self.frame_count:
                        self._last_motion_reacq_frame = -1000000
                        self._last_motion_reacq_pos = None
                    print(
                        f"[LOCAL_AI_PLAYER_JUMP_ROLLBACK] f{self.frame_count}: "
                        f"rejected {rejected_position} jump={player_jump:.1f}px > "
                        f"{jump_limit:.1f}px; restored {previous_position} "
                        f"stuck={self.stuck_frame_count}"
                    )
                    return previous_position
            # Do not let a local-AI rejection still mutate the track.  In the
            # serve-start state a compact candidate with no inter-frame motion
            # is a likely static highlight.  Restore the state from before
            # HSV committed it, then use the next decoded frame to determine
            # whether that blob persisted in place.
            if (
                    reason.startswith(("provisional-serve-jump:", "static-candidate-jump:")) and
                    pre_track_snapshot is not None and
                    tracked_position is not None and
                    float(getattr(self, "_last_tracked_candidate_motion_mean", 0.0) or 0.0) < 5.0 and
                    float(getattr(self, "_last_tracked_candidate_motion_max", 0.0) or 0.0) < 25.0):
                current_image = None
                if self._local_ai_frame_buffer:
                    current_image = self._local_ai_frame_buffer[-1].get("image")
                self._pending_provisional_static_candidate = {
                    "frame": int(self.frame_count),
                    "pos": tuple(tracked_position),
                    "blob": self._static_blob_near(current_image, tracked_position),
                }
                self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
                self.stuck_frame_count = int(previous_stuck) + 1
                print(
                    f"[PROVISIONAL_STATIC_GUARD] f{self.frame_count}: "
                    f"deferred zero-motion rejected jump {tracked_position}; "
                    "waiting for next-frame persistence"
                )
                return previous_position
            return tracked_position
        self._local_ai_all_body_rejections = 0
        repaired_position = (int(recovered["x"]), int(recovered["y"]))
        strict_repaired_reason = self._post_serve_pre_net_recovery_reason(
            previous_position, repaired_position
        )
        if strict_repaired_reason is not None:
            rejected_position = tuple(repaired_position)
            if pre_track_snapshot is not None:
                self._restore_tracking_state_for_provisional_guard(pre_track_snapshot)
            self.stuck_frame_count = max(
                int(previous_stuck or 0) + 1,
                int(getattr(self, 'stuck_frame_count', 0) or 0),
            )
            print(
                f"[POST_SERVE_PRE_NET_AI_REJECT] f{self.frame_count}: "
                f"rejected={rejected_position} reason={strict_repaired_reason}; "
                f"restored={previous_position} stuck={self.stuck_frame_count}"
            )
            return tuple(previous_position) if previous_position is not None else None

        self._maybe_clear_post_serve_pre_net_recovery(
            previous_position, repaired_position, source='local-ai'
        )
        self.ball_center = repaired_position
        self.ball_size = float(recovered.get("area", self.ball_size or 0.0))
        if recovered.get("trajectory_rescue"):
            trajectory_motion = dict(recovered.get("trajectory_motion") or {})
            prior_motion = (pre_track_snapshot or {}).get("last_motion")
            self.prev_motion = dict(prior_motion) if isinstance(prior_motion, dict) else prior_motion
            self.last_motion = trajectory_motion
            if float(trajectory_motion.get("distance", 0.0) or 0.0) >= 3.0:
                self.last_nonzero_motion = dict(trajectory_motion)
            prior_history = list((pre_track_snapshot or {}).get("ball_velocity_history") or [])
            prior_history.append(float(trajectory_motion.get("distance", 0.0) or 0.0))
            self.ball_velocity_history = prior_history[-5:]
            self._held_direction_candidate = None
        self.last_seen_frame = self.frame_count
        self.stuck_frame_count = 0
        self._pending_rally_end_reason = None
        self._pending_rally_end_frame = -1
        self._local_ai_recovery_count += 1
        if reason != "post-recovery-follow":
            # A successful repair commonly happens at contact/occlusion.  The
            # next few frames are the outgoing flight, where the normal tracker
            # has the least reliable prediction.  Establish a bounded handoff
            # window; each *verified* local-AI frame can extend the immediate
            # follow-up by four frames, but never beyond this deadline.
            self._local_ai_handoff_deadline_frame = self.frame_count + 14

        handoff_deadline = int(getattr(self, "_local_ai_handoff_deadline_frame", -1))
        if handoff_deadline >= self.frame_count:
            self._local_ai_follow_until_frame = min(
                handoff_deadline,
                max(
                    int(getattr(self, "_local_ai_follow_until_frame", -1)),
                    self.frame_count + 4,
                ),
            )
        print(
            f"[LOCAL_AI_RECOVERY] f{self.frame_count}: accepted {repaired_position} "
            f"score={float(recovered.get('ai_score', 0.0)):.6f} "
            f"replayed={self.local_ai_recovery.lookback_frames}f reason={reason}"
        )
        return repaired_position

    def _player_serve_context(self, serve_position):
        tracker = getattr(self, "player_tracker", None)
        if tracker is None:
            return {}
        try:
            server_idx = self._current_server_index()
            context = tracker.serve_context(
                serve_position,
                player_name=self.player_names[server_idx] if server_idx is not None else "",
            )
            tracker.adopt_serve_context(context, frame_index=int(getattr(self, 'frame_count', -1)))
            return context
        except Exception:
            return {}

    def _refresh_player_serve_context(self):
        """Refresh server/receiver geometry while the serve is still starting."""
        tracker = getattr(self, "player_tracker", None)
        context = getattr(self, "_point_history_current", None)
        if tracker is None or context is None:
            return
        serve_context = context.get('serve_context') or {}
        serve_x = serve_context.get('serve_x')
        serve_y = serve_context.get('serve_y')
        if serve_x is None or serve_y is None:
            return
        try:
            previous_receiver = serve_context.get('receiver_position')
            refreshed = self._player_serve_context((serve_x, serve_y))
            if not refreshed:
                return
            # Do not replace a valid serve-ball coordinate; update the player
            # detections as they become available during the opening frames.
            for key, value in refreshed.items():
                if value is not None and value != '':
                    serve_context[key] = value
            context['serve_context'] = serve_context
            current_receiver = serve_context.get('receiver_position')
            if current_receiver and current_receiver != previous_receiver:
                print(
                    f"[SERVE_PLAYERS] f{self.frame_count}: "
                    f"server={serve_context.get('player') or 'unknown'} "
                    f"side={serve_context.get('player_side')} "
                    f"position={serve_context.get('player_position')} "
                    f"receiver_side={serve_context.get('receiver_side')} "
                    f"receiver_position={current_receiver} "
                    f"area={serve_context.get('receiver_area')}"
                )
        except Exception:
            return

    def _draw_player_tracking(self, frame, scale=1.0):
        tracker = getattr(self, "player_tracker", None)
        if tracker is None:
            return frame
        if not getattr(self, "tracking", False) and getattr(self, "_point_history_current", None) is None:
            return frame
        result = frame
        try:
            for side, track in tracker.tracks.items():
                if track.bbox is None:
                    continue
                x, y, w, h = track.bbox
                x1 = int(round(x * scale))
                y1 = int(round(y * scale))
                x2 = int(round((x + w) * scale))
                y2 = int(round((y + h) * scale))
                color = (255, 180, 40) if side == "near" else (40, 190, 255)
                thickness = 2 if track.visible else 1
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
                if track.head is not None:
                    hx, hy = int(track.head[0] * scale), int(track.head[1] * scale)
                    cv2.circle(result, (hx, hy), max(4, int(8 * scale)), color, 1)
                if track.shoes is not None:
                    sx, sy = int(track.shoes[0] * scale), int(track.shoes[1] * scale)
                    cv2.circle(result, (sx, sy), max(3, int(6 * scale)), color, 1)
                if track.racket and track.racket.get("line"):
                    lx1, ly1, lx2, ly2 = track.racket["line"]
                    cv2.line(
                        result,
                        (int(lx1 * scale), int(ly1 * scale)),
                        (int(lx2 * scale), int(ly2 * scale)),
                        (0, 165, 255),
                        max(1, int(round(2 * scale))),
                    )
                label = f"{side} {track.confidence:.2f}"
                cv2.putText(
                    result,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.5 * scale),
                    color,
                    max(1, int(round(1.5 * scale))),
                    cv2.LINE_AA,
                )
        except Exception:
            return result
        return result

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
                        if getattr(self, '_direction_only_bounce_preserve_frame', -1000000) == self.frame_count:
                            print(
                                f"Frame {self.frame_count}: Preserving bounce count through "
                                f"same-frame direction-only change"
                            )
                        else:
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
                self._focus_loss_guard_until_frame = self.frame_count + 3
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

    def _night_serve_contact_launch_scan_active(self, serve_contact_grace=None):
        """Night serves need wider HSV only at the actual contact/launch band."""
        if not self._is_night_session_config():
            return False
        if serve_contact_grace is None:
            serve_contact_grace = getattr(self, '_serve_contact_grace_frames', 0) > 0
        if not serve_contact_grace:
            return False
        if self.ball_center is None or self.last_motion is None:
            return False
        if not hasattr(self, 'serve_area_y_min') or not hasattr(self, 'serve_area_y_max'):
            return False
        post_contact_launch = (
            self._is_forward_serve_launch_motion(self.last_motion) and
            (
                self._is_descending_serve_contact_motion(self.prev_motion) or
                self._is_descending_serve_contact_motion(getattr(self, 'prev_prev_motion', None))
            ) and
            (self.ball_size or 0) >= max(45, int(self.serve_contact_min_ball_size * 0.45))
        )
        if post_contact_launch:
            return True
        if (self.ball_size or 0) < 250:
            return False
        if not self._is_descending_serve_contact_motion(self.last_motion):
            return False

        serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
        contact_y = self.serve_area_y_min + int(serve_height * self.serve_contact_y_ratio)
        contact_margin = max(55, min(130, int(serve_height * 0.18)))
        return self.ball_center[1] >= contact_y - contact_margin

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
        if key.startswith("click_upper"):
            return "click_upper"
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
        # The normal hotspot-path override is intentionally capped at lower speeds.
        # A high-speed candidate reaching this point has already passed the stricter
        # trajectory/direction/motion gates above, so preserve that evidence for the
        # later false-positive guard instead of globally weakening hotspot protection.
        if lm_dist > 120.0:
            chosen['trajectory_continuation_selected'] = True
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

    def _prefer_near_net_rebound_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Prefer the visible near-net rebound over tiny racket or net fragments."""
        if self.ball_center is None or self.last_motion is None or not candidate_meta:
            return None
        if not self._is_04_left_court_config():
            return None
        if not hasattr(self, "net_area_y_max"):
            return None

        frame_height, _ = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        prev_size = float(self.ball_size or 0.0)
        net_y_max = float(self.net_area_y_max)
        last_dx = float(self.last_motion.get("dx", 0.0) or 0.0)
        last_dy = float(self.last_motion.get("dy", 0.0) or 0.0)
        last_dist = float(self.last_motion.get("distance", 0.0) or math.hypot(last_dx, last_dy))

        if not (45.0 <= prev_size <= 130.0):
            return None
        if not (net_y_max + 20.0 <= float(prev_y) <= net_y_max + max(95.0, frame_height * 0.055)):
            return None
        if last_dy < max(22.0, frame_height * 0.010) or last_dist < 24.0:
            return None

        selected_entry = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get("contour") is selected_contour:
                    selected_entry = entry
                    break
        if selected_entry is None:
            selected_entry = min(candidate_meta, key=lambda entry: (entry["score"], entry["distance"]))

        selected_pos = selected_entry.get("pos", self.ball_center)
        selected_area = float(selected_entry.get("area", 0.0) or 0.0)
        selected_y = float(selected_pos[1])

        rebound_candidates = []
        min_area = max(42.0, prev_size * 0.45)
        max_area = max(145.0, prev_size * 1.35)
        min_upward = max(20.0, min(36.0, last_dy * 0.45))
        lateral_limit = max(60.0, last_dist * 0.80)
        distance_limit = max(115.0, last_dist * 1.50)

        for entry in candidate_meta:
            if entry is selected_entry:
                continue
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue
            cx, cy = entry["pos"]
            area = float(entry.get("area", 0.0) or 0.0)
            if area < min_area or area > max_area:
                continue
            if abs(float(cx) - float(prev_x)) > lateral_limit:
                continue
            upward_progress = float(prev_y) - float(cy)
            if upward_progress < min_upward and float(cy) > net_y_max + 10.0:
                continue
            if float(cy) < net_y_max - 55.0 or float(cy) > net_y_max + 18.0:
                continue
            distance = math.hypot(float(cx) - float(prev_x), float(cy) - float(prev_y))
            if distance > distance_limit:
                continue

            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            if motion_max < 80.0 and motion_mean < 24.0:
                continue

            selected_is_lower_or_tiny = (
                selected_area <= area * 0.55 or
                selected_y >= float(cy) + 24.0
            )
            if not selected_is_lower_or_tiny:
                continue

            predicted_x = float(prev_x) + last_dx * 0.45
            predicted_y = float(prev_y) - last_dy * 0.45
            predicted_error = math.hypot(float(cx) - predicted_x, float(cy) - predicted_y)
            adjusted_score = (
                float(entry.get("score", 0.0) or 0.0) +
                distance * 0.20 +
                predicted_error * 0.10 -
                area * 0.25 -
                upward_progress * 0.65 -
                motion_max * 0.04
            )
            rebound_candidates.append((adjusted_score, entry, upward_progress, distance, motion_max))

        if not rebound_candidates:
            return None

        _, chosen, upward_progress, distance, motion_max = min(rebound_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [NEAR-NET REBOUND] preferring visible rebound at "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"up={upward_progress:.1f}px dist={distance:.1f}px motion={motion_max:.1f} "
            f"over {selected_pos} area={selected_area:.1f}px"
        )
        return chosen

    def _prefer_large_contact_stall_candidate(self, candidate_meta, selected_contour):
        """Keep the large ball at impact when a smaller racket blob keeps the old path."""
        if self.ball_center is None or self.last_motion is None or not candidate_meta:
            return None

        prev_size = float(self.ball_size or 0.0)
        last_dx = float(self.last_motion.get("dx", 0.0) or 0.0)
        last_dy = float(self.last_motion.get("dy", 0.0) or 0.0)
        last_dist = float(self.last_motion.get("distance", 0.0) or 0.0)
        if prev_size < 75.0 or last_dy < 40.0 or last_dist < 45.0:
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
        selected_keeps_ball_shape = (
            selected_area >= prev_size * 0.55 and
            selected_distance <= max(24.0, last_dist * 0.35)
        )
        if selected_keeps_ball_shape:
            return None

        prev_x, prev_y = self.ball_center
        stall_candidates = []
        for entry in candidate_meta:
            if entry is selected_entry:
                continue
            if entry.get("source") not in ("primary", "regular", "alt"):
                continue

            area = float(entry.get("area", 0.0) or 0.0)
            distance = float(entry.get("distance", 0.0) or 0.0)
            motion_mean = float(entry.get("motion_mean", 0.0) or 0.0)
            motion_max = float(entry.get("motion_max", 0.0) or 0.0)
            cx, cy = entry["pos"]

            if area < prev_size * 0.60 or area > prev_size * 1.45:
                continue
            if distance > max(22.0, last_dist * 0.28):
                continue
            if motion_max < 55.0 and motion_mean < 12.0:
                continue

            # The incoming ball can nearly stop at the racket before launching
            # away on the next frame. Keep that same-size close blob instead of
            # the racket/string fragment that merely continues the old fall.
            move_dx = float(cx - prev_x)
            move_dy = float(cy - prev_y)
            incoming_progress = last_dx * move_dx + last_dy * move_dy
            if incoming_progress > last_dist * max(distance, 1.0) * 0.55:
                continue

            farther_smaller_selected = (
                selected_distance >= distance + max(28.0, last_dist * 0.28) and
                selected_area <= max(prev_size * 0.48, area * 0.52)
            )
            if not farther_smaller_selected:
                continue

            area_ratio = abs(area - prev_size) / max(prev_size, 1.0)
            adjusted_score = (
                distance * 0.8 +
                area_ratio * 60.0 -
                min(45.0, motion_mean * 0.7) -
                min(55.0, motion_max * 0.22)
            )
            if entry.get("source") in ("primary", "regular"):
                adjusted_score -= 8.0
            stall_candidates.append((adjusted_score, entry))

        if not stall_candidates:
            return None

        _, chosen = min(stall_candidates, key=lambda item: item[0])
        print(
            f"  DEBUG: [LARGE-CONTACT STALL] preferring close large candidate at "
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
        """Prefer a moving, upward serve launch over the racket or static court blobs."""
        if self.ball_center is None or not candidate_meta:
            return None
        if not hasattr(self, 'serve_area_y_min') or not hasattr(self, 'serve_area_y_max'):
            return None

        prev_x, prev_y = self.ball_center
        frame_height = frame_shape[0]
        serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
        contact_y = self.serve_area_y_min + int(serve_height * self.serve_contact_y_ratio)
        contact_margin = max(55, min(130, int(serve_height * 0.18)))
        prev_size = float(self.ball_size or 0.0)
        inferred_direction = int(getattr(self, '_serve_launch_direction_x', 0) or 0)
        recent_descent = any(
            self._is_descending_serve_contact_motion(motion)
            for motion in (
                self.last_motion,
                self.prev_motion,
                getattr(self, 'prev_prev_motion', None),
            )
            if motion is not None
        )
        contact_ready = (
            recent_descent and
            prev_y >= contact_y - contact_margin and
            (not prev_size or prev_size >= max(40.0, float(self.serve_contact_min_ball_size) * 0.35))
        )
        if not contact_ready and inferred_direction == 0:
            return None

        last_dist = float((self.last_motion or {}).get('distance', 0.0) or 0.0)
        min_forward_dx = max(45.0, float(self.serve_contact_min_dx) * 0.45)
        min_forward_dy = max(40.0, frame_height * 0.018)
        min_launch_dist = max(70.0, min(150.0, last_dist * 0.75))
        max_launch_dist = max(240.0, min(float(self.max_ball_speed), last_dist * 6.0))
        min_area = max(8.0, min(35.0, prev_size * 0.055 if prev_size else 8.0))
        max_area = max(90.0, prev_size * 1.35 if prev_size else 120.0)
        if self._is_night_session_config():
            min_forward_dx = max(45.0, float(self.serve_contact_min_dx) * 0.40)
            min_forward_dy = max(70.0, frame_height * 0.030)
            min_launch_dist = max(70.0, min(150.0, last_dist * 0.65))
            max_launch_dist = max(430.0, min(float(self.max_ball_speed), last_dist * 14.0))
            max_area = max(max_area, 780.0, prev_size * 3.40 if prev_size else 360.0)

        launch_candidates = []
        for entry in candidate_meta:
            cx, cy = entry['pos']
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            lateral_progress = abs(dx) if inferred_direction == 0 else dx * inferred_direction
            if lateral_progress < min_forward_dx:
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
            # Large pale wall/court contours can satisfy the launch geometry. A genuine
            # struck ball must still have temporal motion at its new position.
            if motion_max < 50.0 and motion_mean < 8.0:
                continue

            upward_bonus = signed_dy if self.serve_direction_dy != 0 else max(0.0, -dy)
            area_match = abs(area - prev_size) / max(prev_size, 1.0) if prev_size else 0.0
            adjusted_score = (
                lateral_progress * 0.85 +
                upward_bonus * 1.35 +
                min(90.0, area * 0.25) +
                min(90.0, motion_max * 0.45) +
                min(55.0, motion_mean * 1.6) -
                move_dist * 0.18 -
                area_match * 18.0
            )
            launch_candidates.append((adjusted_score, entry, lateral_progress, upward_bonus, dx))

        if not launch_candidates:
            return None

        _, chosen, lateral_progress, signed_dy, chosen_dx = max(
            launch_candidates, key=lambda item: item[0]
        )
        if inferred_direction == 0 and abs(chosen_dx) >= min_forward_dx:
            inferred_direction = 1 if chosen_dx > 0 else -1
            self._serve_launch_direction_x = inferred_direction
        print(
            f"  DEBUG: [SERVE-CONTACT LAUNCH] prioritizing verified airborne candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px lateral={lateral_progress:.1f} "
            f"signed_dy={signed_dy:.1f} score={chosen['score']:.1f} motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} direction_x={inferred_direction:+d}"
        )
        # POST_SERVE_LAUNCH_LOCK_ARM_V1
        # For the next few frames, a large vertical jump back toward the server
        # is treated as a player-body latch and handed to Local AI recovery.
        self._post_serve_launch_lock_until_frame = max(
            int(getattr(self, "_post_serve_launch_lock_until_frame", -1)),
            int(self.frame_count) + 7,
        )
        self._post_serve_launch_lock_last_reject_frame = -1000000
        print(
            f"[POST_SERVE_LAUNCH_LOCK] f{self.frame_count}: armed through "
            f"f{self._post_serve_launch_lock_until_frame} direction={self.serve_direction_label()}"
        )
        return chosen

    def _prefer_night_contact_near_continuation_candidate(
        self, candidate_meta, selected_contour, frame_shape
    ):
        """Keep a nearby moving ball over a one-frame static serve-contact jump.

        In the night footage the serve-contact search is intentionally wide.  A
        stationary HSV blob can therefore beat the real ball immediately after
        contact, even when a moving candidate is still adjacent to the previous
        position.  This guard is deliberately conservative: it only applies to
        a large selected jump with little image motion, and requires a nearby,
        sufficiently large candidate with clear temporal motion.
        """
        if (
            not self._is_night_session_config() or
            self.ball_center is None or
            not candidate_meta or
            selected_contour is None
        ):
            return None

        selected = next(
            (meta for meta in candidate_meta if meta.get("contour") is selected_contour),
            None,
        )
        if selected is None:
            return None

        selected_distance = float(selected.get("distance", 0.0) or 0.0)
        selected_motion_mean = float(selected.get("motion_mean", 0.0) or 0.0)
        selected_motion_max = float(selected.get("motion_max", 0.0) or 0.0)
        last_distance = float(
            (self.last_motion or {}).get("distance", 0.0) or 0.0
        )
        # Do not interfere with an ordinary tracked continuation or a genuinely
        # moving launch.  The bad case is a jump far beyond the prior motion to
        # a candidate whose surrounding pixels are essentially static.
        if selected_distance < max(180.0, last_distance * 3.0):
            return None
        if selected_motion_mean >= 8.0 or selected_motion_max >= 35.0:
            return None

        prev_size = float(self.ball_size or 0.0)
        frame_height, frame_width = frame_shape[:2]
        max_near_distance = max(70.0, min(150.0, max(45.0, last_distance * 3.0)))
        min_area = max(18.0, prev_size * 0.20 if prev_size else 18.0)
        nearby = []
        for meta in candidate_meta:
            if meta is selected:
                continue
            if meta.get("source") not in (
                "primary", "regular", "alt", "h_10", "s_30"
            ):
                continue
            distance = float(meta.get("distance", 0.0) or 0.0)
            area = float(meta.get("area", 0.0) or 0.0)
            motion_mean = float(meta.get("motion_mean", 0.0) or 0.0)
            motion_max = float(meta.get("motion_max", 0.0) or 0.0)
            if distance > max_near_distance or area < min_area:
                continue
            if motion_max < 40.0 and motion_mean < 12.0:
                continue
            cx, cy = meta.get("pos", (0, 0))
            if not (0 <= cx < frame_width and 0 <= cy < frame_height):
                continue
            if self._find_frame0_background_hotspot((cx, cy)) is not None and motion_mean < 12.0:
                continue
            nearby.append(meta)

        if not nearby:
            return None

        chosen = min(
            nearby,
            key=lambda meta: (
                float(meta.get("distance", 0.0) or 0.0),
                -float(meta.get("motion_max", 0.0) or 0.0),
                -float(meta.get("area", 0.0) or 0.0),
            ),
        )
        print(
            f"  DEBUG: [NIGHT CONTACT CONTINUATION] keeping nearby moving ball "
            f"at {chosen['pos']} over static jump {selected['pos']} "
            f"distance={selected_distance:.1f}->{chosen['distance']:.1f} "
            f"motion={chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f}"
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

    def _upper_racket_below_escape_candidate_ok(
        self,
        pos,
        area,
        frame_shape,
        motion_mean=0.0,
        motion_max=0.0,
    ):
        """Allow the small ball escaping an upper-court racket-below contact."""
        contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        frames_since_contact = self.frame_count - contact_frame
        if frames_since_contact < 1 or frames_since_contact > 6:
            return False

        contact_point = getattr(self, '_last_racket_contact_point', None)
        if contact_point is None:
            contact_point = getattr(self, '_rally_contact_origin', None)
        if contact_point is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        cx, cy = pos
        origin_x, origin_y = contact_point
        origin_x = int(origin_x)
        origin_y = int(origin_y)

        upper_contact_zone = (
            origin_y <= max(360, int(frame_height * 0.18)) and
            origin_y >= max(180, int(frame_height * 0.07)) and
            max(900, int(frame_width * 0.28)) <= origin_x <= min(frame_width - 900, int(frame_width * 0.66))
        )
        if not upper_contact_zone:
            return False

        dx_from_contact = cx - origin_x
        dy_from_contact = cy - origin_y
        distance_from_contact = math.hypot(dx_from_contact, dy_from_contact)
        if distance_from_contact < 7.0 or distance_from_contact > 115.0:
            return False
        if dx_from_contact > 18.0:
            return False
        if dy_from_contact > 14.0:
            return False
        if abs(dx_from_contact) > 95.0 or abs(dy_from_contact) > 95.0:
            return False

        area = float(area or 0.0)
        if area < 6.0 or area > 70.0:
            return False

        motion_mean = float(motion_mean or 0.0)
        motion_max = float(motion_max or 0.0)
        if motion_max < 75.0 and motion_mean < 18.0:
            return False

        if not (0 <= cx < frame_width and 0 <= cy < frame_height):
            return False

        return True

    def _prefer_upper_racket_below_escape_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Prefer the true ball over player/racket blobs just after upper racket-below contact."""
        if not candidate_meta or self.ball_center is None:
            return None

        contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        frames_since_contact = self.frame_count - contact_frame
        if frames_since_contact < 1 or frames_since_contact > 6:
            return None

        contact_point = getattr(self, '_last_racket_contact_point', None)
        if contact_point is None:
            contact_point = getattr(self, '_rally_contact_origin', None)
        if contact_point is None:
            return None

        selected = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get('contour') is selected_contour:
                    selected = entry
                    break

        escape_candidates = []
        for entry in candidate_meta:
            if not self._upper_racket_below_escape_candidate_ok(
                entry['pos'],
                entry['area'],
                frame_shape,
                motion_mean=entry.get('motion_mean', 0.0),
                motion_max=entry.get('motion_max', 0.0),
            ):
                continue
            cx, cy = entry['pos']
            origin_x, origin_y = contact_point
            dx_from_contact = cx - origin_x
            dy_from_contact = cy - origin_y
            dist_from_contact = math.hypot(dx_from_contact, dy_from_contact)
            prev_dist = float(entry.get('distance', 0.0) or 0.0)
            area = float(entry.get('area', 0.0) or 0.0)
            motion_mean = float(entry.get('motion_mean', 0.0) or 0.0)
            motion_max = float(entry.get('motion_max', 0.0) or 0.0)
            area_match = abs(area - float(self.ball_size or area)) / max(float(self.ball_size or area), 1.0)
            adjusted_score = (
                min(160.0, area * 3.0) +
                min(140.0, motion_max * 0.75) +
                min(90.0, motion_mean * 2.0) +
                max(0.0, -dx_from_contact) * 1.2 +
                max(0.0, -dy_from_contact) * 0.55 -
                prev_dist * 0.28 -
                dist_from_contact * 0.18 -
                area_match * 18.0
            )
            if entry.get('source') in ('primary', 'regular'):
                adjusted_score += 18.0
            escape_candidates.append((adjusted_score, entry, dx_from_contact, dy_from_contact))

        if not escape_candidates:
            return None

        _, chosen, dx_from_contact, dy_from_contact = max(escape_candidates, key=lambda item: item[0])
        if selected is not None:
            selected_ok = self._upper_racket_below_escape_candidate_ok(
                selected['pos'],
                selected['area'],
                frame_shape,
                motion_mean=selected.get('motion_mean', 0.0),
                motion_max=selected.get('motion_max', 0.0),
            )
            if selected_ok and selected is chosen:
                return None

        print(
            f"  DEBUG: [UPPER RACKET-BELOW ESCAPE] prioritizing outgoing candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"contact_delta=({dx_from_contact:.1f},{dy_from_contact:.1f}) "
            f"score={chosen['score']:.1f} motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"source={chosen['source']}"
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

    def _prefer_night_far_player_ball_escape_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Night-session rescue for a ball peeling away from the far player's racket."""
        if not self._is_night_session_config():
            return None
        if self.ball_center is None or not candidate_meta or selected_contour is None:
            return None

        contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        frames_since_contact = self.frame_count - contact_frame
        if frames_since_contact < 4 or frames_since_contact > 22:
            return None

        contact_point = getattr(self, '_last_racket_contact_point', None)
        if contact_point is None:
            contact_point = getattr(self, '_rally_contact_origin', None)
        if contact_point is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        contact_x, contact_y = contact_point
        if not (
                max(1200, int(frame_width * 0.34)) <= contact_x <= min(2300, int(frame_width * 0.62)) and
                max(150, int(frame_height * 0.07)) <= contact_y <= max(360, int(frame_height * 0.18))):
            return None

        selected = None
        for entry in candidate_meta:
            if entry.get("contour") is selected_contour:
                selected = entry
                break
        if selected is None:
            return None

        selected_x, selected_y = selected["pos"]
        selected_area = float(selected.get("area", 0.0) or 0.0)
        selected_distance_from_contact = math.hypot(selected_x - contact_x, selected_y - contact_y)
        if selected_distance_from_contact > 125.0:
            return None
        if selected_y > max(340, int(frame_height * 0.17)):
            return None
        if selected_area > 55.0:
            return None

        outbound_sign = 1.0 if contact_x < (frame_width * 0.50) else -1.0

        escape_candidates = []
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
            dx_from_selected = float(cx - selected_x)
            dy_from_selected = float(cy - selected_y)
            outbound_dx = dx_from_selected * outbound_sign

            if outbound_dx < 170.0:
                continue
            if dy_from_selected < 135.0:
                continue
            if not (max(360, int(frame_height * 0.17)) <= cy <= max(760, int(frame_height * 0.36))):
                continue
            if not (max(1050, int(frame_width * 0.27)) <= cx <= min(2850, int(frame_width * 0.75))):
                continue
            if area < max(70.0, selected_area * 4.0) or area > 260.0:
                continue
            if motion_mean < 24.0 or motion_max < 110.0:
                continue
            if distance > 650.0:
                continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None and motion_mean < 35.0 and motion_max < 150.0:
                continue

            adjusted_score = (
                abs(outbound_dx - 300.0) * 0.16 +
                abs(dy_from_selected - 235.0) * 0.22 +
                distance * 0.05 -
                min(70.0, area * 0.22) -
                min(70.0, motion_mean * 0.85) -
                min(55.0, motion_max * 0.20)
            )
            escape_candidates.append((adjusted_score, -area, entry, dx_from_selected, dy_from_selected))

        if not escape_candidates:
            return None

        _, _, chosen, dx_from_selected, dy_from_selected = min(
            escape_candidates,
            key=lambda item: (item[0], item[1])
        )
        self._night_far_player_escape_frame = self.frame_count
        self._night_far_player_escape_direction = int(outbound_sign)
        self._rally_contact_grace_frames = max(
            getattr(self, '_rally_contact_grace_frames', 0),
            5,
        )
        print(
            f"  DEBUG: [NIGHT FAR PLAYER BALL ESCAPE] preferring visible ball "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"delta=({dx_from_selected:.1f},{dy_from_selected:.1f}) "
            f"motion={chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over upper player fragment {selected['pos']} area={selected_area:.1f}px"
        )
        return chosen

    def _prefer_night_far_player_escape_continuation_candidate(
        self,
        candidate_meta,
        selected_contour,
        frame_shape,
    ):
        """Keep the dynamic ball for a few frames after escaping the far player."""
        if not self._is_night_session_config() or self.ball_center is None or not candidate_meta:
            return None

        escape_frame = int(getattr(self, '_night_far_player_escape_frame', -1000000))
        frames_since_escape = self.frame_count - escape_frame
        if frames_since_escape < 1 or frames_since_escape > 6:
            return None

        frame_height, frame_width = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        if prev_y < max(330, int(frame_height * 0.15)):
            return None
        outbound_sign = int(getattr(self, '_night_far_player_escape_direction', 0) or 0)
        if outbound_sign not in (-1, 1):
            return None

        selected = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get('contour') is selected_contour:
                    selected = entry
                    break

        continuation_candidates = []
        for entry in candidate_meta:
            cx, cy = entry['pos']
            area = float(entry.get('area', 0.0) or 0.0)
            distance = float(entry.get('distance', 0.0) or 0.0)
            motion_mean = float(entry.get('motion_mean', 0.0) or 0.0)
            motion_max = float(entry.get('motion_max', 0.0) or 0.0)
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)
            outbound_dx = dx * outbound_sign

            if outbound_dx < -35.0 or outbound_dx > 210.0:
                continue
            if dy < -35.0 or dy > 210.0:
                continue
            if distance < 8.0 or distance > 220.0:
                continue
            if area < 18.0 or area > 300.0:
                continue
            if motion_mean < 18.0 and motion_max < 90.0:
                continue
            if not (0 <= cx < frame_width and 0 <= cy < frame_height):
                continue

            adjusted_score = (
                distance * 0.55 +
                abs(outbound_dx - 55.0) * 0.20 +
                abs(dy - 75.0) * 0.18 -
                min(70.0, motion_mean * 0.9) -
                min(55.0, motion_max * 0.22) -
                min(45.0, area * 0.18)
            )
            continuation_candidates.append((adjusted_score, -area, entry))

        if not continuation_candidates:
            return None

        _, _, chosen = min(continuation_candidates, key=lambda item: (item[0], item[1]))
        if selected is chosen:
            return None

        print(
            f"  DEBUG: [NIGHT FAR PLAYER ESCAPE CONTINUE] preferring dynamic ball "
            f"{chosen['pos']} area={chosen['area']:.1f}px distance={chosen['distance']:.1f}px "
            f"motion={chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over {selected['pos'] if selected is not None else None}"
        )
        return chosen

    def _prefer_night_far_player_visible_ball_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Prefer the visible ball over tiny far-player body/racket fragments at night."""
        if not self._is_night_session_config():
            return None
        if self.ball_center is None or not candidate_meta or selected_contour is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        selected = None
        for entry in candidate_meta:
            if entry.get("contour") is selected_contour:
                selected = entry
                break
        if selected is None:
            return None

        selected_x, selected_y = selected["pos"]
        selected_area = float(selected.get("area", 0.0) or 0.0)
        if selected_area > 12.0:
            return None
        if not (
                max(1650, int(frame_width * 0.44)) <= selected_x <= min(2050, int(frame_width * 0.64)) and
                max(165, int(frame_height * 0.08)) <= selected_y <= max(330, int(frame_height * 0.16))):
            return None
        if self.ball_center[1] > max(340, int(frame_height * 0.17)):
            return None

        visible_candidates = []
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
            dx_from_selected = float(cx - selected_x)
            dy_from_selected = float(cy - selected_y)

            if dx_from_selected > -65.0:
                continue
            if dy_from_selected < 18.0:
                continue
            if not (max(210, int(frame_height * 0.10)) <= cy <= max(360, int(frame_height * 0.18))):
                continue
            if area < max(22.0, selected_area * 7.0) or area > 120.0:
                continue
            if motion_mean < 12.0 or motion_max < 90.0:
                continue
            if distance > max(150.0, frame_width * 0.045):
                continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if frame0_hotspot is not None and motion_mean < 26.0 and motion_max < 135.0:
                continue

            adjusted_score = (
                distance * 0.30 +
                abs(dx_from_selected + 115.0) * 0.10 +
                abs(dy_from_selected - 55.0) * 0.16 -
                min(50.0, area * 0.35) -
                min(70.0, motion_max * 0.25) -
                min(35.0, motion_mean * 0.80)
            )
            visible_candidates.append((adjusted_score, -area, entry, dx_from_selected, dy_from_selected))

        if not visible_candidates:
            return None

        _, _, chosen, dx_from_selected, dy_from_selected = min(
            visible_candidates,
            key=lambda item: (item[0], item[1])
        )
        print(
            f"  DEBUG: [NIGHT FAR PLAYER VISIBLE BALL] preferring ball "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"delta=({dx_from_selected:.1f},{dy_from_selected:.1f}) "
            f"motion={chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over tiny player fragment {selected['pos']} area={selected_area:.1f}px"
        )
        return chosen

    def _prefer_night_upper_player_airborne_ball_candidate(
        self,
        candidate_meta,
        selected_contour,
        frame_shape,
    ):
        """Keep a serve-flight ball above the far player instead of following the player."""
        if not self._is_night_session_config():
            return None
        if not getattr(self, '_awaiting_serve_bounce', False):
            return None
        if self.ball_center is None or not candidate_meta or selected_contour is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        if not (
                max(1250, int(frame_width * 0.32)) <= prev_x <= min(2250, int(frame_width * 0.60)) and
                max(105, int(frame_height * 0.048)) <= prev_y <= max(300, int(frame_height * 0.145))):
            return None

        selected = None
        for entry in candidate_meta:
            if entry.get('contour') is selected_contour:
                selected = entry
                break
        if selected is None:
            return None

        selected_x, selected_y = selected['pos']
        selected_area = float(selected.get('area', 0.0) or 0.0)
        selected_step = math.hypot(selected_x - prev_x, selected_y - prev_y)
        selected_tracks_player = (
            selected_y >= prev_y - max(12.0, frame_height * 0.006) and
            selected_x >= prev_x + max(18.0, frame_width * 0.0045) and
            selected_step <= max(95.0, frame_width * 0.026) and
            selected_area <= 45.0
        )
        if not selected_tracks_player:
            return None

        airborne_candidates = []
        for entry in candidate_meta:
            if entry is selected or entry.get('source') not in ('primary', 'regular', 'alt'):
                continue
            cx, cy = entry['pos']
            area = float(entry.get('area', 0.0) or 0.0)
            distance = float(entry.get('distance', 0.0) or 0.0)
            motion_mean = float(entry.get('motion_mean', 0.0) or 0.0)
            motion_max = float(entry.get('motion_max', 0.0) or 0.0)
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)

            if cy > prev_y - max(24.0, frame_height * 0.011):
                continue
            if cy > selected_y - max(22.0, frame_height * 0.010):
                continue
            if cx > selected_x - max(20.0, frame_width * 0.005):
                continue
            if abs(dx) > max(105.0, frame_width * 0.030):
                continue
            if distance > max(115.0, frame_width * 0.032):
                continue
            if area < 5.0 or area > 90.0:
                continue
            if motion_mean < 20.0 or motion_max < 120.0:
                continue

            adjusted_score = (
                distance * 0.55 +
                abs(dx - 8.0) * 0.22 +
                abs(dy + 34.0) * 0.20 -
                min(35.0, area * 0.45) -
                min(45.0, motion_mean * 0.55) -
                min(35.0, motion_max * 0.12)
            )
            airborne_candidates.append((adjusted_score, -area, entry))

        if not airborne_candidates:
            return None

        _, _, chosen = min(airborne_candidates, key=lambda item: (item[0], item[1]))
        self._night_upper_airborne_frame = self.frame_count
        print(
            f"  DEBUG: [NIGHT UPPER AIRBORNE BALL] preferring ball {chosen['pos']} "
            f"area={chosen['area']:.1f}px motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over player-associated candidate {selected['pos']} area={selected_area:.1f}px"
        )
        return chosen

    def _prefer_night_upper_airborne_continuation_candidate(
        self,
        candidate_meta,
        selected_contour,
        frame_shape,
    ):
        """Keep the rescued upper ball as it descends past the far player."""
        if not self._is_night_session_config() or self.ball_center is None:
            return None
        rescue_frame = int(getattr(self, '_night_upper_airborne_frame', -1000000))
        frames_since_rescue = self.frame_count - rescue_frame
        if frames_since_rescue < 1 or frames_since_rescue > 30:
            return None
        if not candidate_meta or selected_contour is None or self.last_motion is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        prev_x, prev_y = self.ball_center
        last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        if prev_y > max(520, int(frame_height * 0.25)) or last_dy < 3.0:
            return None

        selected = None
        for entry in candidate_meta:
            if entry.get('contour') is selected_contour:
                selected = entry
                break
        if selected is None:
            return None

        continuation_candidates = []
        for entry in candidate_meta:
            if entry.get('source') not in ('primary', 'regular', 'alt'):
                continue
            cx, cy = entry['pos']
            area = float(entry.get('area', 0.0) or 0.0)
            distance = float(entry.get('distance', 0.0) or 0.0)
            motion_mean = float(entry.get('motion_mean', 0.0) or 0.0)
            motion_max = float(entry.get('motion_max', 0.0) or 0.0)
            dx = float(cx - prev_x)
            dy = float(cy - prev_y)

            if dx < -8.0 or dx > max(40.0, frame_width * 0.012):
                continue
            if dy < 8.0 or dy > max(90.0, frame_height * 0.045):
                continue
            if distance > max(105.0, frame_width * 0.030):
                continue
            if area < 18.0 or area > 320.0:
                continue
            if motion_mean < 18.0 or motion_max < 90.0:
                continue

            expected_dx = max(4.0, min(24.0, last_dx + 2.0))
            expected_dy = max(12.0, min(70.0, last_dy + 7.0))
            adjusted_score = (
                abs(dx - expected_dx) * 0.32 +
                abs(dy - expected_dy) * 0.38 +
                distance * 0.22 -
                min(55.0, area * 0.28) -
                min(38.0, motion_mean * 0.45) -
                min(30.0, motion_max * 0.10)
            )
            continuation_candidates.append((adjusted_score, -area, entry))

        if not continuation_candidates:
            return None

        _, _, chosen = min(continuation_candidates, key=lambda item: (item[0], item[1]))
        if chosen is selected:
            return None

        selected_x, selected_y = selected['pos']
        selected_area = float(selected.get('area', 0.0) or 0.0)
        chosen_area = float(chosen.get('area', 0.0) or 0.0)
        selected_step_dy = float(selected_y - prev_y)
        selected_is_weaker = (
            selected_step_dy <= 4.0 or
            selected_area < max(12.0, chosen_area * 0.45)
        )
        if not selected_is_weaker:
            return None

        print(
            f"  DEBUG: [NIGHT UPPER AIRBORNE CONTINUE] preferring ball {chosen['pos']} "
            f"area={chosen_area:.1f}px motion="
            f"{chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"over {selected['pos']} area={selected_area:.1f}px"
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
        max_area = self._tracking_ball_size_max()
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

    def _night_recent_bounce_outbound_escape_override(
            self, pos, area, prev_pos, frame_shape, motion_mean=0.0, motion_max=0.0):
        """Allow the night ball to escape right after an upper-court bounce/hit."""
        if not self._is_night_session_config() or prev_pos is None or frame_shape is None:
            return False, None
        if getattr(self, 'ground_bounce_count', 0) <= 0:
            return False, None

        frames_since_bounce = self.frame_count - int(getattr(self, 'last_ground_bounce_frame', -1000000))
        if frames_since_bounce < 0 or frames_since_bounce > 22:
            return False, None

        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            return False, None

        prev_x, prev_y = prev_pos
        cx, cy = pos
        dx = float(cx - prev_x)
        dy = float(cy - prev_y)
        move_dist = math.hypot(dx, dy)
        area = float(area or 0.0)
        motion_mean = float(motion_mean or 0.0)
        motion_max = float(motion_max or 0.0)

        last_motion = self.last_motion or getattr(self, 'last_nonzero_motion', None)
        last_dx = float(last_motion.get('dx', 0.0) or 0.0) if last_motion is not None else 0.0
        last_dy = float(last_motion.get('dy', 0.0) or 0.0) if last_motion is not None else 0.0
        last_dist = float(last_motion.get('distance', 0.0) or 0.0) if last_motion is not None else 0.0

        upper_launch_zone = (
            prev_y <= max(380, int(frame_height * 0.18)) or
            (last_dx >= 75.0 and prev_y <= max(650, int(frame_height * 0.31)))
        )
        if not upper_launch_zone:
            return False, None
        if prev_x < int(frame_width * 0.50):
            return False, None
        if dx < max(58.0, min(115.0, max(0.0, last_dx) * 0.30)):
            return False, None
        if dy < -170.0 or dy > 270.0:
            return False, None
        if move_dist < 70.0 or move_dist > max(430.0, last_dist * 3.3):
            return False, None
        if cy < max(115, int(frame_height * 0.052)) or cy > max(720, int(frame_height * 0.34)):
            return False, None
        if area < 1.0 or area > 120.0:
            return False, None
        if area <= 1.5 and motion_max < 3.0 and last_dist < 115.0:
            return False, None

        # This is a right-sideline escape for the night 04-left-court view. Keep
        # it narrow so ordinary far-player/player-body speckles are still rejected.
        if cx < int(frame_width * 0.58) and frames_since_bounce > 5:
            return False, None

        ideal_dx = max(95.0, min(230.0, max(last_dist * 1.35, abs(last_dx) * 1.15, 115.0)))
        ideal_dy = max(45.0, min(145.0, max(abs(last_dy) * 0.55, 75.0)))
        escape_score = (
            abs(dx - ideal_dx) * 0.95 +
            abs(dy - ideal_dy) * 0.55 +
            max(0.0, move_dist - 260.0) * 0.25 -
            min(60.0, area * 0.35) -
            min(45.0, motion_max * 0.18) -
            min(30.0, motion_mean * 0.55)
        )
        return True, {
            'predicted_distance': escape_score,
            'predicted_cap': 120.0,
            'prev_distance': move_dist,
            'prev_cap': max(430.0, last_dist * 3.3),
            'motion_mean': motion_mean,
            'motion_max': motion_max,
            'last_motion_distance': last_dist,
            'frames_since_bounce': frames_since_bounce,
            'dx': dx,
            'dy': dy,
            'score': escape_score,
        }

    def _prefer_night_recent_bounce_outbound_escape_candidate(self, candidate_meta, selected_contour, frame_shape):
        """Prefer the outgoing night ball over upper-player speckles after a bounce."""
        if not candidate_meta or self.ball_center is None:
            return None

        selected = None
        if selected_contour is not None:
            for entry in candidate_meta:
                if entry.get('contour') is selected_contour:
                    selected = entry
                    break

        escape_candidates = []
        for entry in candidate_meta:
            if entry.get('source') not in ('primary', 'regular', 'alt', 'alt2', 'alt4', 'alt6'):
                continue
            ok, debug = self._night_recent_bounce_outbound_escape_override(
                entry['pos'],
                entry['area'],
                self.ball_center,
                frame_shape,
                motion_mean=entry.get('motion_mean', 0.0),
                motion_max=entry.get('motion_max', 0.0),
            )
            if not ok:
                continue
            adjusted_score = (
                debug['score'] +
                float(entry.get('score', 0.0) or 0.0) * 0.08 -
                min(80.0, float(entry.get('area', 0.0) or 0.0) * 0.55)
            )
            escape_candidates.append((adjusted_score, entry, debug))

        if not escape_candidates:
            return None

        _, chosen, debug = min(escape_candidates, key=lambda item: item[0])
        if selected is chosen:
            return None

        print(
            f"  DEBUG: [NIGHT OUTBOUND ESCAPE] prioritizing outgoing candidate "
            f"{chosen['pos']} area={chosen['area']:.1f}px "
            f"delta=({debug['dx']:.1f},{debug['dy']:.1f}) "
            f"frames_since_bounce={debug['frames_since_bounce']} "
            f"motion={chosen['motion_mean']:.1f}/{chosen['motion_max']:.1f} "
            f"source={chosen['source']}"
        )
        return chosen

    def _upper_baseline_retrack_hotspot_override_ok(self, pos, prev_pos, predicted_point, frame_shape):
        """Limit retrack hotspot overrides to the far baseline line-blend area."""
        if not self._is_04_left_court_config():
            return False
        if pos is None or prev_pos is None or predicted_point is None:
            return False
        frame_height = frame_shape[0] if frame_shape is not None and len(frame_shape) >= 1 else 0
        frame_width = frame_shape[1] if frame_shape is not None and len(frame_shape) >= 2 else 0
        if frame_height <= 0 or frame_width <= 0:
            return False

        x, y = pos
        prev_x, prev_y = prev_pos
        # Court-2 far baseline/upper white-line band. Keep this tight so ordinary
        # learned hotspots elsewhere do not become live tracking candidates.
        line_x_min = max(0, min(frame_width - 1, 1950))
        line_x_max = max(0, min(frame_width - 1, 2165))
        line_y_min = max(0, min(frame_height - 1, 170))
        line_y_max = max(0, min(frame_height - 1, 292))
        if not (line_x_min <= x <= line_x_max and line_y_min <= y <= line_y_max):
            return False
        if not (line_x_min - 80 <= prev_x <= line_x_max + 80 and
                line_y_min - 40 <= prev_y <= line_y_max + 70):
            return False

        current_size = float(self.ball_size or 0.0)
        if current_size > 35.0:
            return False
        motion = self.last_motion or getattr(self, 'last_nonzero_motion', None)
        if motion is None:
            return False
        if float(motion.get('dy', 0.0) or 0.0) > -4.0:
            return False
        return True

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
        area_cap = self._tracking_ball_size_max()
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

    def _candidate_motion_metrics(
            self, gray_frame, cx, cy, radius=8, previous_gray=None):
        prev_gray = (
            previous_gray if previous_gray is not None else
            getattr(self, '_prev_frame_gray', None)
        )
        if gray_frame is None or prev_gray is None:
            return None
        if gray_frame.shape[:2] != prev_gray.shape[:2]:
            return None

        # A player-reacquisition pass can update ``_prev_frame_gray`` before
        # the caller learns that normal tracking merely held the old marker.
        # Tight contact recovery receives the causal pre-track image directly;
        # compute that one patch without contaminating the per-frame cache.
        if previous_gray is not None:
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(gray_frame.shape[1], cx + radius + 1)
            y2 = min(gray_frame.shape[0], cy + radius + 1)
            if x2 <= x1 or y2 <= y1:
                return None
            current_patch = gray_frame[y1:y2, x1:x2]
            previous_patch = prev_gray[y1:y2, x1:x2]
            if current_patch.size == 0 or previous_patch.size == 0:
                return None
            diff_patch = cv2.absdiff(previous_patch, current_patch)
            return {
                'mean': float(np.mean(diff_patch)),
                'max': float(np.max(diff_patch)),
            }

        # This helper is called for every HSV contour.  A bright/noisy night
        # frame can produce hundreds of contours, so repeatedly calculating
        # ``absdiff`` for each tiny patch made one video frame take minutes.
        # The difference image is identical for every candidate in the same
        # tracker frame; cache it and only slice the already-computed result.
        cache_frame = getattr(self, '_motion_metrics_cache_frame', None)
        if cache_frame != self.frame_count:
            self._motion_metrics_cache_frame = self.frame_count
            self._motion_metrics_cache = {}
            self._motion_metrics_diff = cv2.absdiff(prev_gray, gray_frame)
        cache = getattr(self, '_motion_metrics_cache', None)
        if cache is None:
            cache = {}
            self._motion_metrics_cache = cache
        key = (int(cx), int(cy), int(radius))
        cached = cache.get(key)
        if cached is not None:
            return cached

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(gray_frame.shape[1], cx + radius + 1)
        y2 = min(gray_frame.shape[0], cy + radius + 1)
        if x2 <= x1 or y2 <= y1:
            return None

        diff_patch = self._motion_metrics_diff[y1:y2, x1:x2]
        if diff_patch.size == 0:
            return None

        result = {
            'mean': float(np.mean(diff_patch)),
            'max': float(np.max(diff_patch)),
        }
        cache[key] = result
        return result

    def _night_static_side_artifact(self, pos, area, motion_mean, motion_max, frame_shape):
        if not self._is_night_session_config() or frame_shape is None:
            return False
        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            return False

        x, y = pos
        weak_motion = float(motion_mean or 0.0) < 3.5 and float(motion_max or 0.0) < 18.0
        if not weak_motion:
            return False

        edge_band = max(70, int(frame_width * 0.018))
        bottom_band = max(80, int(frame_height * 0.035))
        side_band = max(760, int(frame_width * 0.20))
        near_frame_edge = (
            x <= edge_band or
            x >= frame_width - edge_band or
            y <= max(50, int(frame_height * 0.025)) or
            y >= frame_height - bottom_band
        )
        lower_side_artifact = (
            y >= max(780, int(frame_height * 0.36)) and
            (
                x <= side_band or
                x >= frame_width - side_band
            )
        )
        if not (near_frame_edge or lower_side_artifact):
            return False

        area_value = float(area or 0.0)

        # A truly visible moving ball near a side can have weak mean motion if it
        # is blurred. The night video also has larger static lower-side patches
        # that look ball-colored after a racket miss; those must not become a
        # fake out-bounce path.
        if lower_side_artifact:
            return area_value <= 750.0
        return area_value <= 180.0

    def _find_night_visible_ball_candidate(self, frame, frame_gray=None):
        if not self._is_night_session_config() or frame is None:
            return None

        frame_height, frame_width = frame.shape[:2]
        player_reacq_active = int(
            getattr(self, '_player_reacq_protect_until_frame', -1)
        ) >= self.frame_count
        # A close, motion-blurred post-hit ball can occupy 400-500 contour
        # pixels (a prior frame is ~430px).  Keep the normal night recovery cap tight,
        # but permit that larger compact blob only while recovering from a
        # known player occlusion.
        area_cap = 650.0 if player_reacq_active else 340.0
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv_frame,
            np.array([25, 55, 120], dtype=np.uint8),
            np.array([55, 255, 255], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 18.0 or area > area_cap:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            _, _, contour_width, contour_height = cv2.boundingRect(contour)

            if cy < max(150, int(frame_height * 0.07)) or cy > int(frame_height * 0.78):
                continue
            if cx < max(600, int(frame_width * 0.15)) or cx > int(frame_width * 0.86):
                continue
            if cx > int(frame_width * 0.74) and cy < int(frame_height * 0.26):
                continue

            motion_metrics = self._candidate_motion_metrics(frame_gray, cx, cy)
            motion_mean = motion_metrics['mean'] if motion_metrics is not None else 0.0
            motion_max = motion_metrics['max'] if motion_metrics is not None else 0.0
            player_zone = self._player_point_zone((cx, cy))
            upward_progress = (
                float(self.ball_center[1] - cy)
                if self.ball_center is not None else 0.0
            )
            near_player_launch = (
                player_reacq_active and
                self.ball_center is not None and
                # The first visible post-contact frame can still be low in
                # the player box, while each following frame is already above
                # that box.  Keep recognizing the same launch by its compact,
                # local, upward step instead of requiring the stale marker to
                # remain in the bottom 40% of the image.  The lower 60 px
                # bound admits the real a prior frame-a prior frame flight; the lateral and
                # upper-distance bounds still reject the distant player/net
                # artifacts that originally caused the false endpoint.
                upward_progress >= max(60.0, frame_height * 0.025) and
                upward_progress <= max(650.0, frame_height * 0.30) and
                abs(cx - self.ball_center[0]) <= max(450.0, frame_width * 0.12) and
                max(contour_width, contour_height) <=
                max(3.0, min(contour_width, contour_height) * 2.6)
            )
            if player_zone in (
                    'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'):
                # The recovery window deliberately accepts a larger ball, so
                # compensate by refusing player pixels unless their local
                # frame difference is unequivocally dynamic.  The exception
                # is a compact blob launching sharply upward from a known
                # near-player contact; frame differencing can read zero here
                # because the preceding motion pass refreshes its reference.
                if ((motion_mean < 16.0 or motion_max < 70.0) and
                        not near_player_launch):
                    continue
            if motion_max < 35.0 and motion_mean < 6.0 and area < 95.0:
                continue
            if (self.ball_center is not None and motion_max < 12.0 and motion_mean < 3.5 and
                    not near_player_launch):
                prev_dist = math.hypot(cx - self.ball_center[0], cy - self.ball_center[1])
                last_dist = (
                    float(self.last_motion.get('distance', 0.0) or 0.0)
                    if self.last_motion is not None else 0.0
                )
                predicted_dist = None
                if self.last_motion is not None:
                    predicted_x = self.ball_center[0] + float(self.last_motion.get('dx', 0.0) or 0.0)
                    predicted_y = self.ball_center[1] + float(self.last_motion.get('dy', 0.0) or 0.0)
                    predicted_dist = math.hypot(cx - predicted_x, cy - predicted_y)
                if (
                        prev_dist > max(120.0, last_dist * 4.0) and
                        (predicted_dist is None or predicted_dist > max(90.0, last_dist * 3.0))):
                    continue

            frame0_hotspot = self._find_frame0_background_hotspot((cx, cy))
            if (frame0_hotspot is not None and motion_mean < 12.0 and motion_max < 55.0 and
                    not near_player_launch):
                continue

            local_hsv = hsv_frame[cy, cx]
            court_center_x = frame_width * 0.43
            court_center_y = frame_height * 0.34
            score = (
                abs(cx - court_center_x) * 0.018 +
                abs(cy - court_center_y) * 0.030 +
                abs(area - 135.0) * 0.12 -
                min(90.0, motion_max * 0.24) -
                min(55.0, motion_mean * 0.9)
            )
            if near_player_launch:
                # At a near-player return, distinguish the outgoing ball from
                # the residual contact smear by how far it has launched away
                # from the racket.  This makes the real a prior frame ball at y~1053
                # beat the nearer smear at y~1229.
                score -= min(120.0, upward_progress * 0.35)
            candidates.append((score, -area, {
                'pos': (cx, cy),
                'area': area,
                'hsv': local_hsv,
                'motion_mean': motion_mean,
                'motion_max': motion_max,
            }))

        if not candidates:
            return None

        _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        return chosen

    def _prefer_night_tight_local_ai_candidate(
            self, frame, frame_gray, proposed_candidate, previous_gray=None):
        """Prefer a model-ranked ball in a bounded causal ROI near contact.

        This is deliberately candidate arbitration, not broad recovery.  It
        handles both a sharp weak-motion player-region turn and a fast ball
        that stalls at a racket before launching.  Coherent normal candidates
        pass through cheaply; suspicious proposals are ranked only inside the
        small causal/predicted corridor.
        """
        recovery = getattr(self, 'local_ai_recovery', None)
        ranker = getattr(recovery, 'rank_local_roi_candidate', None)
        previous = getattr(self, 'ball_center', None)
        if (
                recovery is None or not callable(ranker) or frame is None or
                frame_gray is None or proposed_candidate is None or
                previous is None):
            return proposed_candidate

        frame_index = int(self.frame_count)
        follow = getattr(self, '_local_ai_tight_roi_follow', None)
        if isinstance(follow, dict) and frame_index > int(follow.get('deadline', -1)):
            self._local_ai_tight_roi_follow = None
            follow = None
        follow_active = (
            isinstance(follow, dict) and
            int(follow.get('last_frame', -1)) < frame_index <=
            int(follow.get('deadline', -1))
        )

        prior_motion = dict(getattr(self, 'last_motion', None) or {})
        prior_dx = float(prior_motion.get('dx', 0.0) or 0.0)
        prior_dy = float(prior_motion.get('dy', 0.0) or 0.0)
        prior_speed = float(
            prior_motion.get('distance', math.hypot(prior_dx, prior_dy)) or 0.0
        )
        proposed_pos = tuple(proposed_candidate.get('pos') or previous)
        proposed_dx = float(proposed_pos[0]) - float(previous[0])
        proposed_dy = float(proposed_pos[1]) - float(previous[1])
        proposed_speed = math.hypot(proposed_dx, proposed_dy)
        proposed_mean = float(proposed_candidate.get('motion_mean', 0.0) or 0.0)
        proposed_max = float(proposed_candidate.get('motion_max', 0.0) or 0.0)
        initial_mode = 'tight_turn'
        roi_radius = 25.0
        prediction_limit = 25.0
        fast_follow = False

        if follow_active:
            origin = tuple(follow['last_pos'])
            follow_mode = str(follow.get('mode', 'tight_turn'))
            fast_follow = follow_mode == 'fast_contact'
            elapsed = max(1, frame_index - int(follow['last_frame']))
            if fast_follow:
                ai_accept_count = int(follow.get('ai_accept_count', 1))
                velocity = tuple(follow.get('velocity', (0.0, 0.0)))
                if ai_accept_count <= 1:
                    # The first post-stall direction is unknown.  Stay tightly
                    # around contact for one frame before extrapolating.
                    anchor = origin
                    roi_radius = 32.0
                else:
                    anchor = (
                        int(round(float(origin[0]) + float(velocity[0]) * elapsed)),
                        int(round(float(origin[1]) + float(velocity[1]) * elapsed)),
                    )
                    # The first measured launch step is shorter than the next
                    # acceleration frame; bridge that one step, then tighten.
                    roi_radius = 75.0 if ai_accept_count == 2 else 35.0
                prediction_limit = roi_radius
            else:
                anchor = origin
            proposal_from_anchor = math.hypot(
                float(proposed_pos[0]) - float(origin[0]),
                float(proposed_pos[1]) - float(origin[1]),
            )
            proposal_from_prediction = math.hypot(
                float(proposed_pos[0]) - float(anchor[0]),
                float(proposed_pos[1]) - float(anchor[1]),
            )
            # Once the normal tracker again supplies a strongly moving point
            # within the 25 px contact corridor, it is already the safe local
            # answer.  Keep advancing the lock without paying for another
            # model process.  A large jump (the original a prior frame/a prior frame failure)
            # still invokes the tight ranker below.
            coherent_normal = (
                not proposed_candidate.get('tight_roi_force_rank', False) and
                proposal_from_prediction <= (25.0 if fast_follow else 35.0) and
                proposed_mean >= 10.0 and proposed_max >= 50.0
            )
            if coherent_normal:
                step_dx = (
                    float(proposed_pos[0]) - float(origin[0])
                ) / elapsed
                step_dy = (
                    float(proposed_pos[1]) - float(origin[1])
                ) / elapsed
                step_speed = math.hypot(step_dx, step_dy)
                coherent_history = list(
                    follow.get(
                        'velocity_history',
                        getattr(self, 'ball_velocity_history', []),
                    )
                )[-4:] + [step_speed]
                follow['last_pos'] = tuple(proposed_pos)
                follow['last_frame'] = frame_index
                follow['velocity'] = (step_dx, step_dy)
                follow['velocity_history'] = coherent_history
                if fast_follow:
                    # Preserve the launch lock through the short high-speed
                    # flight.  It is cheap while normal tracking is coherent,
                    # and protects the later far-player/top-edge crossing.
                    follow['outside_count'] = 0
                    self._local_ai_tight_roi_follow = follow
                else:
                    active_zones = ('player_body', 'racket_fragment')
                    outside_player = (
                        self._player_point_zone(origin) not in active_zones and
                        self._player_point_zone(proposed_pos) not in active_zones
                    )
                    follow['outside_count'] = (
                        int(follow.get('outside_count', 0)) + 1
                        if outside_player else 0
                    )
                    if int(follow['outside_count']) >= 2:
                        self._local_ai_tight_roi_follow = None
                        print(
                            f"[LOCAL_AI_TIGHT_ROI_HANDOFF] f{frame_index}: "
                            f"normal tracker exited player ROI at {proposed_pos}"
                        )
                    else:
                        self._local_ai_tight_roi_follow = follow
                self._local_ai_tight_roi_accept_frame = frame_index
                return proposed_candidate
            trigger = (
                f"{follow_mode}-lock:"
                f"{frame_index - int(follow.get('start_frame', frame_index))}f/"
                f"normal={proposal_from_prediction:.0f}px"
            )
        else:
            height, _ = frame.shape[:2]
            prior_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
            proposed_area = float(proposed_candidate.get('area', 0.0) or 0.0)
            contact_band = (
                int(height * 0.28) <= int(previous[1]) <= int(height * 0.52)
            )
            contact_proximity = any(
                self._point_in_player_contact_corridor(point)
                for point in (previous, proposed_pos)
            )
            contact_stall = (
                prior_speed >= 20.0 and prior_dy >= 10.0 and
                proposed_speed <= 8.0 and
                proposed_speed <= prior_speed * 0.35 and
                proposed_mean >= 10.0 and proposed_max >= 50.0 and
                max(prior_size, proposed_area) >= 40.0 and
                (contact_proximity or contact_band)
            )
            if contact_stall:
                initial_mode = 'fast_contact'
                anchor = (int(proposed_pos[0]), int(proposed_pos[1]))
                trigger = (
                    f'contact-stall:{prior_speed:.0f}->{proposed_speed:.0f}px/'
                    f'motion={proposed_mean:.0f}/{proposed_max:.0f}'
                )
            else:
                # The original probe is reserved for the weak-motion player
                # fragment seen at a prior frame.  Strong normal motion alone is not
                # evidence of a takeover.
                if (
                        prior_speed < 5.0 or proposed_speed < 1.0 or
                        proposed_mean >= 10.0 or proposed_max >= 50.0):
                    return proposed_candidate
                prior_direction = math.degrees(math.atan2(prior_dy, prior_dx))
                proposed_direction = math.degrees(math.atan2(proposed_dy, proposed_dx))
                raw_delta = abs(proposed_direction - prior_direction) % 360.0
                angle_delta = min(raw_delta, 360.0 - raw_delta)
                anchor = (
                    int(round(float(previous[0]) + prior_dx)),
                    int(round(float(previous[1]) + prior_dy)),
                )
                prediction_error = math.hypot(
                    float(proposed_pos[0]) - float(anchor[0]),
                    float(proposed_pos[1]) - float(anchor[1]),
                )
                near_player = any(
                    self._player_point_zone(point) in ('player_body', 'racket_fragment')
                    for point in (proposed_pos, previous, anchor)
                )
                if (
                        not near_player or angle_delta < 70.0 or
                        prediction_error < 12.0 or prediction_error > 45.0):
                    return proposed_candidate
                trigger = f'player-turn:{angle_delta:.0f}deg/{prediction_error:.0f}px'

        self._local_ai_tight_roi_attempt_frame = frame_index
        try:
            selected = ranker(
                frame_index,
                frame,
                anchor=anchor,
                radius=roi_radius,
                maximum_candidates=8,
            )
        except Exception as exc:
            print(
                f"[LOCAL_AI_TIGHT_ROI] f{frame_index}: scorer failed "
                f"({type(exc).__name__}: {exc})"
            )
            return proposed_candidate
        if selected is None:
            print(
                f"[LOCAL_AI_TIGHT_ROI] f{frame_index}: no candidate "
                f"anchor={anchor} trigger={trigger}"
            )
            return proposed_candidate

        model_pos = (int(selected['x']), int(selected['y']))
        model_score = float(selected.get('ai_score', 0.0) or 0.0)
        prediction_distance = math.hypot(
            float(model_pos[0]) - float(anchor[0]),
            float(model_pos[1]) - float(anchor[1]),
        )
        previous_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
        model_area = float(selected.get('area', previous_size) or previous_size)
        merged_player_contour = model_area > max(120.0, previous_size * 3.0)

        # When the ball and player are merged into one contour, the contour's
        # centroid is biased toward the player.  The model has verified that
        # this tight patch contains the ball; if the causal anchor lies inside
        # its bounding box, retain that physical anchor as the ball center.
        accepted_pos = model_pos
        bbox = selected.get('bbox')
        if merged_player_contour and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bx, by, bw, bh = [int(value) for value in bbox[:4]]
            if bx <= anchor[0] < bx + bw and by <= anchor[1] < by + bh:
                accepted_pos = anchor

        motion_previous_gray = (
            previous_gray if previous_gray is not None else
            getattr(self, '_local_ai_tight_roi_previous_gray', None)
        )
        motion = self._candidate_motion_metrics(
            frame_gray,
            accepted_pos[0],
            accepted_pos[1],
            previous_gray=motion_previous_gray,
        )
        motion_mean = float(motion.get('mean', 0.0) or 0.0) if motion else 0.0
        motion_max = float(motion.get('max', 0.0) or 0.0) if motion else 0.0
        strong_motion = motion_mean >= 10.0 and motion_max >= 50.0
        stronger_than_proposal = (
            motion_mean >= proposed_mean + 4.0 or
            motion_max >= proposed_max + 35.0
        )
        catastrophic_lock_jump = (
            follow_active and not fast_follow and
            proposal_from_anchor >= 150.0 and
            prediction_distance <= 12.0
        )
        fast_top_edge_candidate = (
            fast_follow and int(anchor[1]) <= 40 and
            int(model_pos[1]) <= 45 and prediction_distance <= 10.0 and
            int(selected.get('roi_candidates', 0) or 0) == 1
        )
        score_floor = (
            0.45 if fast_top_edge_candidate else
            0.50 if catastrophic_lock_jump else
            0.75 if follow_active else
            0.985 if initial_mode == 'fast_contact' else
            0.85
        )
        accepted = (
            model_score >= score_floor and
            prediction_distance <= prediction_limit and
            strong_motion and
            (follow_active or stronger_than_proposal)
        )
        if not accepted:
            print(
                f"[LOCAL_AI_TIGHT_ROI] f{frame_index}: rejected model={model_pos} "
                f"score={model_score:.6f} pred={prediction_distance:.1f}px "
                f"floor={score_floor:.2f} "
                f"motion={motion_mean:.1f}/{motion_max:.1f} "
                f"normal={proposed_mean:.1f}/{proposed_max:.1f} "
                f"trigger={trigger}"
            )
            return proposed_candidate

        safe_area = previous_size if merged_player_contour else model_area
        if safe_area <= 0.0:
            safe_area = model_area
        height, width = frame.shape[:2]
        px = max(0, min(width - 1, int(accepted_pos[0])))
        py = max(0, min(height - 1, int(accepted_pos[1])))
        accepted_pos = (px, py)

        prior_history = list(getattr(self, 'ball_velocity_history', []))
        if follow_active:
            origin = tuple(follow['last_pos'])
            origin_frame = int(follow['last_frame'])
            elapsed = max(1, frame_index - origin_frame)
            motion_dx = (float(accepted_pos[0]) - float(origin[0])) / elapsed
            motion_dy = (float(accepted_pos[1]) - float(origin[1])) / elapsed
            prior_history = list(follow.get('velocity_history', prior_history))
            new_velocity_history = prior_history[-4:] + [
                math.hypot(motion_dx, motion_dy)
            ]
            follow['last_pos'] = tuple(accepted_pos)
            follow['last_frame'] = frame_index
            follow['velocity'] = (motion_dx, motion_dy)
            follow['velocity_history'] = new_velocity_history
            follow['ai_accept_count'] = int(follow.get('ai_accept_count', 0)) + 1
            # AI-only positions do not prove that normal tracking recovered.
            # Only coherent normal candidates may advance handoff state.
            follow['outside_count'] = 0
            self._local_ai_tight_roi_follow = follow
        else:
            if initial_mode == 'fast_contact':
                motion_dx = float(accepted_pos[0]) - float(previous[0])
                motion_dy = float(accepted_pos[1]) - float(previous[1])
            else:
                motion_dx = prior_dx
                motion_dy = prior_dy
            new_velocity_history = prior_history[-4:] + [
                math.hypot(motion_dx, motion_dy)
            ]
            self._local_ai_tight_roi_follow = {
                'start_frame': frame_index,
                'deadline': frame_index + (20 if initial_mode == 'fast_contact' else 24),
                'last_frame': frame_index,
                'last_pos': tuple(accepted_pos),
                'velocity': (prior_dx, prior_dy),
                'velocity_history': new_velocity_history,
                'outside_count': 0,
                'mode': initial_mode,
                'ai_accept_count': 1,
            }
            if initial_mode == 'fast_contact':
                self._local_ai_tight_roi_follow['velocity'] = (
                    motion_dx, motion_dy
                )
        motion_distance = math.hypot(motion_dx, motion_dy)
        motion_direction = (
            math.degrees(math.atan2(motion_dy, motion_dx))
            if motion_distance > 0.0 else 0.0
        )
        if (
                fast_follow and accepted_pos[1] <= 40 and
                motion_dy < 0.0 and
                isinstance(self._local_ai_tight_roi_follow, dict)):
            # The next frame's existing top-return wait is safer than further
            # ROI ranking once the predicted flight leaves the image.
            self._local_ai_tight_roi_follow['deadline'] = frame_index

        preferred = dict(proposed_candidate)
        preferred.update({
            'pos': accepted_pos,
            'area': safe_area,
            'hsv': cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py, px],
            'motion_mean': motion_mean,
            'motion_max': motion_max,
            'source': 'local_ai_tight_roi',
            'recovery_label': 'LOCAL AI TIGHT ROI',
            'local_ai_score': model_score,
            'local_ai_model_pos': model_pos,
            'local_ai_anchor': anchor,
            'local_ai_trigger': trigger,
            'local_ai_motion_override': {
                'distance': motion_distance,
                'dx': motion_dx,
                'dy': motion_dy,
                'direction_deg': motion_direction,
            },
            'local_ai_velocity_history': new_velocity_history,
        })
        self._local_ai_tight_roi_accept_frame = frame_index
        print(
            f"[LOCAL_AI_TIGHT_ROI] f{frame_index}: preferred {accepted_pos} "
            f"model={model_pos} score={model_score:.6f} anchor={anchor} "
            f"radius={roi_radius:.0f}px motion={motion_mean:.1f}/{motion_max:.1f} "
            f"trigger={trigger}"
        )
        return preferred

    def _find_night_startup_regular_candidate(self, frame, frame_gray=None):
        """Recover a visible ball during a bounded night-camera flight.

        The night camera's narrow pre-focus mask can miss the ball for a few
        frames while the regular mask still sees it.  The generic full-frame
        motion fallback is unsafe here: it can select a yellow player/racket
        patch.  Search the regular (and the two known ball-friendly masks)
        directly, anchored to the previous ball position.  The ball becomes
        only a handful of pixels when it is far from the camera, so this path
        must allow small contours while still requiring continuity.
        """
        if not self._is_night_session_config() or frame is None or self.ball_center is None:
            return None
        # Keep this bounded to a real flight, not to the beginning of the
        # *video*.  The old limit was ``video_start + 96``.  In the first
        # night rally P1 hits at a prior frame, then the visible ball crosses the net
        # through a prior frame-a prior frame; the fallback was already disabled and the stale
        # marker at a prior frame was held instead.  Start a new short continuity
        # window after every verified racket contact, while retaining the
        # point-start window for the opening serve.
        point_anchor = int(
            getattr(self, 'point_start_frame_internal', None)
            or getattr(self, 'start_frame', 0)
            or 0
        )
        flight_anchor = point_anchor
        last_contact = int(getattr(self, '_last_racket_contact_frame', -1) or -1)
        if last_contact >= flight_anchor:
            flight_anchor = last_contact
        # The first cross-court return can still be in flight after the
        # opening 96-frame serve window but before contact metadata becomes
        # reliable.  Keep the same strict contour/trajectory checks for that
        # short opening-rally extension, rather than falling back to a stale
        # point near the net (night frames 120+).  The same opening return
        # continues into the far half through roughly a prior frame in this camera;
        # stopping at +180 cut off a still-continuous ball at a prior frame and caused
        # a later jump to wall/static noise.  This only widens the time
        # window: the contour, motion, and player-exclusion gates below still
        # decide whether a candidate is accepted.
        frame_height, frame_width = frame.shape[:2]
        startup_limit = max(point_anchor + 240, flight_anchor + 96)
        # A point can contain a second, fully visible cross-court flight well
        # after the opening serve window.  At a prior frame in the first night rally
        # the ball is still a compact regular-mask contour only 29 px from
        # a prior frame, yet the fixed +240 frame limit disables this local search and
        # the generic full-frame motion pass jumps to static noise at
        # (901, 639).  Permit a bounded extension only when the *previous*
        # accepted ball is already in a coherent in-court flight.  This is
        # not a broad late-point search: every candidate below must still
        # pass the small jump, compact-shape, motion, and player-exclusion
        # checks, so it naturally ends when that physical chain disappears.
        continuity_extension = (
            self.ball_center is not None and
            self.last_motion is not None and
            point_anchor + 240 < self.frame_count <= point_anchor + 360 and
            8.0 <= float(self.last_motion.get('distance', 0.0) or 0.0) <= 120.0 and
            int(frame_width * 0.20) <= self.ball_center[0] <= int(frame_width * 0.80) and
            int(frame_height * 0.07) <= self.ball_center[1] <= int(frame_height * 0.72)
        )
        if self.frame_count > startup_limit and not continuity_extension:
            return None
        previous_size = float(self.ball_size or 0.0)
        # At the far-player contact the real ball can compress to a 6-7 px
        # regular-mask contour for one frame.  Treating every contour below
        # 8 px as unusable disables this continuity search exactly when the
        # ball leaves the racket (night f69), after which the generic recovery
        # holds the player's body until f85.  The contour loop below still
        # requires at least 6 px plus motion/shape continuity, so a 4 px state
        # gate preserves the real contact without admitting isolated pixels.
        if previous_size < 4.0:
            return None

        previous = self.ball_center
        last_distance = float((self.last_motion or {}).get('distance', 0.0) or 0.0)
        last_dx = float((self.last_motion or {}).get('dx', 0.0) or 0.0)
        startup_last_dy = float((self.last_motion or {}).get('dy', 0.0) or 0.0)
        last_dy = startup_last_dy
        # A fast upward ball cannot become a nearby static yellow blob on the
        # next frame.  In the first night rally this caused a prior frame to hold
        # (898, 823), instead of following the visible flight from a prior frame
        # (891, 820) towards (951, 726).  This deliberately applies only to
        # an unambiguous upward flight, so ground bounces and racket contact
        # are free to change direction.
        # The ball decelerates as it rises toward the far court.  It can
        # therefore make a still-clear upward step of roughly 75 px after a
        # previous 90+ px step (night a prior frame -> a prior frame).  Keep this as a
        # *sustained upward* test rather than requiring another 90 px step:
        # otherwise the 130 px local window rejects the real ball 138 px
        # ahead and a nearby static court blob takes over.
        fast_upward_flight = (
            last_distance >= 60.0 and
            math.hypot(last_dx, last_dy) >= 60.0 and
            last_dy <= -50.0
        )
        # At the top of a normal arc the ball decelerates: a prior frame -> a prior frame is
        # only about (42, -46), and the later steps are smaller again.  It is
        # still a valid upward/right flight, not permission to select an
        # almost stationary court feature.  This narrower continuation stays
        # out of the lower court and requires both forward components.
        decelerating_upper_flight = (
            previous[1] <= int(frame_height * 0.30) and
            last_distance >= 20.0 and
            last_dx >= 8.0 and
            last_dy <= -8.0
        )
        upward_flight_continuation = fast_upward_flight or decelerating_upper_flight
        # The lower player is positioned left of the ordinary flight lane in
        # this camera.  Retain a small, trajectory-gated extension of the
        # regular-mask search there only while a large ball is descending
        # toward that player.  The old 22%-of-width floor excluded the real
        # contact ball at x=560 (a prior frame), so the relaxed HSV fallback later
        # substituted a shoe/body contour at x=1151.
        lower_contact_flight = (
            previous[1] >= int(frame_height * 0.60) and
            startup_last_dy >= max(45.0, frame_height * 0.020) and
            last_distance >= 45.0 and
            previous_size >= 120.0
        )
        jump_cap = max(220.0, min(420.0, last_distance * 3.0 + 90.0))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        specs = []
        if self.hsv_regular is not None:
            specs.append(('regular', self.hsv_regular['lower'], self.hsv_regular['upper']))
        if self.h10_hsv_lower is not None and self.h10_hsv_upper is not None:
            specs.append(('h_10', self.h10_hsv_lower, self.h10_hsv_upper))
        if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
            specs.append(('s_30', self.s30_hsv_lower, self.s30_hsv_upper))
        if not specs:
            return None

        kernel = np.ones((2, 2), np.uint8)
        candidates = []
        for source, lower, upper in specs:
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                # Perspective and compression can reduce the visible ball to
                # 6-10 contour pixels.  A relative floor preserves it without
                # admitting isolated single-pixel noise.
                if area < max(6.0, previous_size * 0.06) or area > 1800.0:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    continue
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                # The center net post is yellow in this camera and remains in
                # every mask as a tall 10-25 x 80-110 px contour.  It is often
                # closer/larger than the ball exactly as the ball crosses the
                # net.  A ball blob is compact (even with motion blur), so do
                # not let a strongly elongated contour win the local-area
                # tie-break below.
                _, _, contour_width, contour_height = cv2.boundingRect(contour)
                if (
                        contour_height >= max(36, contour_width * 3.0) or
                        contour_width >= max(36, contour_height * 3.0)
                ):
                    continue
                min_candidate_x = int(frame_width * (0.10 if lower_contact_flight else 0.22))
                if not (min_candidate_x <= cx <= int(frame_width * 0.90)):
                    continue
                # The serve continues into the far upper court; once the ball
                # rises above the service line it can legitimately be below
                # 10% of the image height.  Keep the search bounded away from
                # the timestamp/header area, but do not cut off this flight.
                if not (int(frame_height * 0.05) <= cy <= int(frame_height * 0.82)):
                    continue
                distance = math.hypot(cx - previous[0], cy - previous[1])
                if distance > jump_cap:
                    continue
                player_zone = self._player_point_zone((cx, cy))
                # The descending ball can briefly overlap the lower player's
                # arm/racket immediately before a return.  This is not a
                # generic player-zone exception: it is accepted only for the
                # calibrated regular mask when the contour is exactly where
                # the established *descending* trajectory predicts it.  It
                # preserves the incoming ball at the contact frame (night
                # a prior frame) so the next frame can search for the upward launch,
                # rather than replacing it with a player-body blob.
                predicted_contact = (previous[0] + last_dx, previous[1] + last_dy)
                contact_prediction_error = math.hypot(
                    cx - predicted_contact[0], cy - predicted_contact[1]
                )
                if upward_flight_continuation and self._player_point_zone((cx, cy)) is None:
                    step_x = cx - previous[0]
                    step_y = cy - previous[1]
                    forward_progress = (
                        (step_x * last_dx + step_y * last_dy) /
                        max(last_distance, 1.0)
                    )
                    # Permit a modest speed change, but not a blob that sits
                    # on (or moves backward from) the last confirmed ball.
                    minimum_forward_progress = (
                        max(42.0, last_distance * 0.35)
                        if fast_upward_flight else
                        max(10.0, last_distance * 0.25)
                    )
                    if forward_progress < minimum_forward_progress:
                        continue
                lower_player_contact_approach = (
                    source == 'regular' and
                    lower_contact_flight and
                    last_dy >= max(45.0, frame_height * 0.020) and
                    last_distance >= 45.0 and
                    distance <= max(115.0, last_distance * 1.70) and
                    contact_prediction_error <= max(58.0, last_distance * 0.90) and
                    area >= max(20.0, min(previous_size * 0.25, 120.0)) and
                    area <= max(850.0, previous_size * 2.50)
                )
                if player_zone in (
                        'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'):
                    # During this serve the ball passes directly beside the
                    # far player's shoes.  A regular-mask, compact contour
                    # that remains within one frame of the established ball
                    # trajectory is allowed through; relaxed-mask player
                    # fragments remain rejected.
                    if not (
                            (source == 'regular' and distance <= 65.0) or
                            lower_player_contact_approach
                    ):
                        continue
                motion = self._candidate_motion_metrics(frame_gray, cx, cy)
                motion_mean = float(motion.get('mean', 0.0) or 0.0) if motion else 0.0
                motion_max = float(motion.get('max', 0.0) or 0.0) if motion else 0.0
                # A compact yellow contour that does not change at all from
                # the previous frame is a court/net highlight, not the ball.
                # This is especially important at the net where the post and
                # line can be closer than the real ball.  Keep candidates with
                # clear local frame-to-frame change; the ball remains moving
                # even when its HSV area is only a few pixels.
                weak_static_motion = (
                    motion is not None and motion_mean < 5.0 and motion_max < 25.0
                )
                if weak_static_motion and (
                        # During an established serve flight a real ball must
                        # produce local frame-to-frame change.  Previously a
                        # nearby regular-mask court highlight was allowed to
                        # survive solely because it was within 65 px, which
                        # could hand the post-contact flight to a static blob.
                        upward_flight_continuation or
                        lower_contact_flight or
                        source != 'regular' or
                        distance > 65.0
                ):
                    continue
                size_ratio = abs(area - previous_size) / max(previous_size, 1.0)
                source_penalty = 0.0 if source == 'regular' else 8.0
                score = distance + min(120.0, size_ratio * 55.0) + source_penalty
                if lower_player_contact_approach:
                    # Make the physically predicted contact contour win over
                    # a similarly coloured racket/body fragment.
                    score -= min(55.0, max(0.0, 58.0 - contact_prediction_error))
                candidates.append((score, -area, {
                    'pos': (cx, cy),
                    'area': area,
                    'hsv': hsv_frame[cy, cx],
                    'motion_mean': motion_mean,
                    'motion_max': motion_max,
                    'source': source,
                    'lower_contact_approach': lower_player_contact_approach,
                    'recovery_label': (
                        'NIGHT LOWER CONTACT APPROACH'
                        if lower_player_contact_approach else
                        'NIGHT STARTUP REGULAR RECOVER'
                    ),
                }))
        if not candidates:
            return None
        # A player/racket mask can produce a second candidate only a few
        # pixels from the ball (notably during the serve follow-through).  The
        # regular court mask is the calibrated ball mask; the relaxed masks
        # are useful only when the regular mask has no viable contour.  Prefer
        # a regular candidate in the local continuity window instead of
        # letting a larger yellow racket/body patch win on area alone.
        local_cap = max(70.0, min(130.0, last_distance * 1.5 + 45.0))
        if upward_flight_continuation:
            # Permit the one remaining long perspective step in a verified
            # upper-flight.  The forward-progress gate above still rejects
            # static blobs that happen to be closer to the prior position.
            local_cap = max(local_cap, min(165.0, last_distance * 1.85 + 45.0))
        local_candidates = [item for item in candidates if item[2]['pos'] and
                            math.hypot(item[2]['pos'][0] - previous[0], item[2]['pos'][1] - previous[1]) <= local_cap]
        if local_candidates:
            regular_local = [
                item for item in local_candidates
                if item[2].get('source') == 'regular'
            ]
            if regular_local:
                local_candidates = regular_local
            _, _, chosen = min(
                local_candidates,
                key=lambda item: (
                    item[0],
                    -item[2]['motion_mean'],
                    -item[2]['area'],
                ),
            )
        else:
            _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        return self._prefer_night_tight_local_ai_candidate(
            frame, frame_gray, chosen
        )

    def _find_night_lower_contact_launch_candidate(
            self, frame, frame_gray, lower_contact_launch_context
    ):
        """Find the first upward ball after a verified lower-player contact.

        This is deliberately a short-lived, trajectory-only recovery.  It
        runs after the incoming ball has been accepted in the racket/player
        overlap and before the broad HSV scorer can replace the real launch
        with a nearby shoe or shirt contour.
        """
        if (
                not self._is_night_session_config() or frame is None or
                lower_contact_launch_context is None or self.ball_center is None
        ):
            return None

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        specs = []
        if self.hsv_regular is not None:
            specs.append(('regular', self.hsv_regular['lower'], self.hsv_regular['upper']))
        if self.h10_hsv_lower is not None and self.h10_hsv_upper is not None:
            specs.append(('h_10', self.h10_hsv_lower, self.h10_hsv_upper))
        if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
            specs.append(('s_30', self.s30_hsv_lower, self.s30_hsv_upper))
        if not specs:
            return None

        candidates = []
        kernel = np.ones((2, 2), np.uint8)
        for source, lower, upper in specs:
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 8.0 or area > 1800.0:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    continue
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                _, _, width, height = cv2.boundingRect(contour)
                if (
                        height >= max(36, width * 3.0) or
                        width >= max(36, height * 3.0)
                ):
                    continue
                if not self._lower_contact_launch_candidate_ok(
                        (cx, cy), area, lower_contact_launch_context, frame.shape
                ):
                    continue
                motion = self._candidate_motion_metrics(frame_gray, cx, cy)
                motion_mean = float(motion.get('mean', 0.0) or 0.0) if motion else 0.0
                motion_max = float(motion.get('max', 0.0) or 0.0) if motion else 0.0
                expected_x, expected_y = lower_contact_launch_context['expected']
                expected_distance = math.hypot(cx - expected_x, cy - expected_y)
                candidates.append((
                    expected_distance + (0.0 if source == 'regular' else 8.0),
                    -motion_max,
                    -area,
                    {
                        'pos': (cx, cy),
                        'area': area,
                        'hsv': hsv_frame[cy, cx],
                        'motion_mean': motion_mean,
                        'motion_max': motion_max,
                        'source': source,
                        'recovery_label': 'NIGHT LOWER CONTACT LAUNCH',
                    },
                ))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:3])[3]

    def _find_night_contact_outbound_continuation(self, frame, frame_gray):
        """Follow the first upward frames after a verified lower-player hit."""
        if (
                not self._is_night_session_config() or frame is None or
                self.ball_center is None or self.last_motion is None or
                int(getattr(self, '_rally_contact_grace_frames', 0)) <= 0
        ):
            return None

        dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        if dy >= -max(45.0, frame.shape[0] * 0.020) or speed < 60.0:
            return None
        expected = (self.ball_center[0] + dx, self.ball_center[1] + dy)
        max_prediction_error = max(110.0, min(180.0, speed * 0.95))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        specs = []
        if self.hsv_regular is not None:
            specs.append(('regular', self.hsv_regular['lower'], self.hsv_regular['upper']))
        if self.h10_hsv_lower is not None and self.h10_hsv_upper is not None:
            specs.append(('h_10', self.h10_hsv_lower, self.h10_hsv_upper))
        if self.s30_hsv_lower is not None and self.s30_hsv_upper is not None:
            specs.append(('s_30', self.s30_hsv_lower, self.s30_hsv_upper))

        kernel = np.ones((2, 2), np.uint8)
        candidates = []
        nearby_debug = []
        for source, lower, upper in specs:
            mask = cv2.inRange(hsv_frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 8.0 or area > 1800.0:
                    continue
                moments = cv2.moments(contour)
                if moments['m00'] == 0:
                    continue
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                _, _, width, height = cv2.boundingRect(contour)
                if (
                        height >= max(36, width * 3.0) or
                        width >= max(36, height * 3.0) or
                        cy >= self.ball_center[1] - 20
                ):
                    continue
                prediction_error = math.hypot(cx - expected[0], cy - expected[1])
                if prediction_error > max_prediction_error:
                    continue
                player_zone = self._player_point_zone((cx, cy))
                nearby_debug.append(
                    f"{source}@({cx},{cy}) a={area:.1f} err={prediction_error:.1f} "
                    f"zone={player_zone or 'none'}"
                )
                # The first upward frame is still inside the lower player's
                # conservative bbox.  Keep it only when the calibrated
                # regular mask is tightly on the post-contact prediction;
                # all other player-zone fragments remain excluded.
                if player_zone is not None and not (
                        source == 'regular' and
                        prediction_error <= min(65.0, max_prediction_error * 0.55) and
                        area >= 40.0
                ):
                    continue
                motion = self._candidate_motion_metrics(frame_gray, cx, cy)
                motion_mean = float(motion.get('mean', 0.0) or 0.0) if motion else 0.0
                motion_max = float(motion.get('max', 0.0) or 0.0) if motion else 0.0
                # A yellow ball laid over a white sideline can be reduced to
                # just a few HSV pixels and therefore has almost no frame-
                # difference signal.  During the one-to-four-frame verified
                # post-contact window, an extremely tight physical prediction
                # is stronger evidence than that missing motion signal.  This
                # is intentionally limited to the compact alternate masks so
                # static regular-HSV court artifacts cannot use this escape.
                line_overlap_continuation = (
                    source in ('h_10', 's_30') and
                    area <= 80.0 and
                    prediction_error <= min(60.0, max_prediction_error * 0.60)
                )
                # The ball is often momentarily measured against its own
                # previous blurred image immediately after the lower-player
                # contact, so frame difference can be near zero even though
                # the regular mask has the real, sizeable outgoing blob.  In
                # the first night rally these are a compact sequence:
                # a prior frame (692,1173), a prior frame (759,1043), a prior frame (827,925), and
                # a prior frame (891,819).  The former line-overlap escape admitted a
                # tiny alternate-HSV contour instead and the tracker lagged
                # behind the visible ball.  Admit the large *regular* blob
                # only when it is very close to the physical post-contact
                # prediction; the four-frame grace window and shape/player
                # gates above keep this from becoming a general static-artifact
                # exception.
                tight_regular_contact_continuation = (
                    source == 'regular' and
                    area >= 100.0 and
                    prediction_error <= max_prediction_error
                )
                if (
                        motion is not None and motion_mean < 4.0 and
                        motion_max < 22.0 and not (
                            line_overlap_continuation or
                            tight_regular_contact_continuation
                        )
                ):
                    continue
                # The first decoded frame after a racket hit can be delayed
                # relative to the last compact launch blob, so a sizeable
                # regular candidate may be 80-110 px past a simple
                # constant-velocity prediction.  Its size/shape is stronger
                # evidence than a tiny line fragment which happens to be
                # nearer to that stale prediction.  Apply this preference
                # only to the explicit four-frame post-contact continuation.
                continuation_score = prediction_error + (
                    0.0 if source == 'regular' else 7.0
                )
                if tight_regular_contact_continuation:
                    continuation_score -= min(90.0, area * 0.20)
                candidates.append((
                    continuation_score,
                    -motion_max,
                    -area,
                    {
                        'pos': (cx, cy), 'area': area, 'hsv': hsv_frame[cy, cx],
                        'motion_mean': motion_mean, 'motion_max': motion_max,
                        'source': source,
                        'recovery_label': 'NIGHT LOWER CONTACT CONTINUATION',
                    },
                ))
        if not candidates:
            print(
                f"Frame {self.frame_count}: [LOWER-HIT OUTBOUND MISS] "
                f"expected=({expected[0]:.0f},{expected[1]:.0f}) "
                f"nearby={' | '.join(nearby_debug[:8]) or 'none'}"
            )
            return None
        # This helper returns before the normal grace-window decrement below.
        # Consume one frame here so a good contact continuation cannot extend
        # the special relaxed rules indefinitely into unrelated court blobs.
        self._rally_contact_grace_frames = max(
            0, int(getattr(self, '_rally_contact_grace_frames', 0)) - 1
        )
        return min(candidates, key=lambda item: item[:3])[3]

    def _commit_night_visible_ball_recovery(self, candidate, frame):
        # Some visible-ball recovery paths (notably player-reacquisition) jump
        # directly here and bypass the startup candidate arbiter.  Reuse the
        # same pre-commit tight-ROI guard so an active near-player lock can
        # reject that jump before it mutates motion/contact state.
        if (
                frame is not None and
                candidate.get('source') != 'local_ai_tight_roi' and
                int(getattr(self, '_local_ai_tight_roi_attempt_frame', -1000000)) !=
                int(self.frame_count)):
            commit_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            candidate = self._prefer_night_tight_local_ai_candidate(
                frame, commit_gray, candidate
            )
        prev_pos = self.ball_center
        new_pos = candidate['pos']
        self.ball_center = new_pos
        self.ball_hsv = candidate['hsv']
        self.ball_size = candidate['area']
        self._update_recovered_motion(prev_pos, new_pos)
        motion_override = candidate.get('local_ai_motion_override')
        if isinstance(motion_override, dict):
            self.last_motion = dict(motion_override)
            self.last_delta = (
                float(motion_override.get('dx', 0.0) or 0.0),
                float(motion_override.get('dy', 0.0) or 0.0),
            )
            self.last_direction = float(
                motion_override.get('direction_deg', 0.0) or 0.0
            )
            velocity_history = candidate.get('local_ai_velocity_history')
            if isinstance(velocity_history, list):
                self.ball_velocity_history = list(velocity_history)[-5:]
            if float(motion_override.get('distance', 0.0) or 0.0) > 0.0:
                self.last_nonzero_motion = dict(motion_override)
        self.stuck_frame_count = 0
        # Short grace period for a player/racket pixel held during occlusion.
        self._player_occlusion_hold_frames = 0
        self.direction_change_streak = 0
        self._post_reacq_frames = max(getattr(self, '_post_reacq_frames', 0), 3)
        self._last_motion_reacq_frame = self.frame_count
        self.last_seen_frame = self.frame_count
        self._last_tracked_candidate_frame = self.frame_count
        self._last_tracked_candidate_motion_frame = self.frame_count
        self._last_tracked_candidate_motion_mean = float(candidate.get('motion_mean', 0.0) or 0.0)
        self._last_tracked_candidate_motion_max = float(candidate.get('motion_max', 0.0) or 0.0)
        if candidate.get('source') == 'local_ai_tight_roi':
            self._local_ai_recovery_count = int(
                getattr(self, '_local_ai_recovery_count', 0)
            ) + 1
            # Suppress broad local-AI replay and endpoint inference only for
            # this already-verified frame.  The dedicated one-frame ROI state
            # independently handles the adjacent confirmation frame.
            self._local_ai_follow_until_frame = max(
                int(getattr(self, '_local_ai_follow_until_frame', -1)),
                int(self.frame_count),
            )
            self._pending_rally_end_reason = None
            self._pending_rally_end_frame = -1
        self._activate_regular_hsv()
        if candidate.get('lower_contact_approach', False):
            # Keep a tightly bounded launch prediction for the frame after
            # the incoming ball overlaps the player's racket.
            launch_context = self._get_lower_contact_launch_context(frame.shape)
            if launch_context is not None:
                self._pending_lower_contact_launch_context = launch_context
                self._pending_lower_contact_launch_until_frame = self.frame_count + 3
        elif candidate.get('recovery_label') == 'NIGHT LOWER CONTACT LAUNCH':
            # The launch is now verified; allow its fast upward continuation
            # through the normal tracker without retaining the contact window.
            self._pending_lower_contact_launch_context = None
            self._pending_lower_contact_launch_until_frame = -1000000
            self._rally_contact_grace_frames = max(
                getattr(self, '_rally_contact_grace_frames', 0), 4
            )
            self._rally_contact_origin = prev_pos
            self._rally_contact_expected = new_pos
        if frame is not None:
            self._prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recovery_label = candidate.get('recovery_label', 'NIGHT VISIBLE BALL RECOVER')
        print(
            f"Frame {self.frame_count}: [{recovery_label}] Ball at {new_pos} "
            f"area={candidate['area']:.1f}px motion="
            f"{candidate.get('motion_mean', 0.0):.1f}/{candidate.get('motion_max', 0.0):.1f}"
        )
        return self.ball_center

    def _player_reacq_static_candidate(self, candidate, predicted_point=None):
        """Return True for a static blob that is unsafe just after player reacq.

        A real ball immediately following an occlusion should have either local
        frame motion or a position consistent with the predicted flight path.
        Static yellow/green patches (often racket or court artifacts) should not
        be allowed to become the new ball center merely because their area looks
        like the ball.
        """
        if self.frame_count > int(getattr(self, '_player_reacq_protect_until_frame', -1)):
            return False
        if not candidate:
            return False
        motion_mean = float(candidate.get('motion_mean', 0.0) or 0.0)
        motion_max = float(candidate.get('motion_max', 0.0) or 0.0)
        if motion_mean >= 6.0 or motion_max >= 25.0:
            return False
        pos = candidate.get('pos')
        if pos is None or self.ball_center is None:
            return False
        prev_distance = math.hypot(pos[0] - self.ball_center[0], pos[1] - self.ball_center[1])
        predicted_distance = (
            math.hypot(pos[0] - predicted_point[0], pos[1] - predicted_point[1])
            if predicted_point is not None else prev_distance
        )
        # Keep a static candidate only when it is essentially on the predicted
        # path; a distant static blob is an artifact even if its area is large.
        return prev_distance > 100.0 and predicted_distance > 90.0

    def _night_stuck_player_artifact_candidate(
            self, position, motion_mean, motion_max, distance):
        """Reject a static player/body contour during a stuck re-acquisition.

        Candidate scoring already penalizes player regions, but if every real
        ball contour is temporarily occluded the penalized body contour can
        still be the only candidate and therefore win.  A genuine ball that
        reappears after contact has local frame motion; the f97 neck/body
        takeover has none and is more than 500 px from the last ball.
        """
        if not self._is_night_session_config() or position is None:
            return False
        zone = self._player_point_zone(position)
        if zone not in (
                'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'):
            return False
        return (
            float(distance or 0.0) > 120.0 and
            float(motion_mean or 0.0) < 6.0 and
            float(motion_max or 0.0) < 25.0
        )

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
            'night_static_side_artifact': self._night_static_side_artifact(
                pos, area, motion_mean, motion_max,
                frame_gray.shape if frame_gray is not None else None
            ),
        }

    def _night_far_baseline_dynamic_current_lock(
        self,
        label,
        current_pos,
        current_area,
        override,
        current_metrics,
        override_metrics,
        frame_shape,
    ):
        """Keep a motion-confirmed far-baseline ball over a nearby static HSV fallback."""
        if not self._is_night_session_config() or frame_shape is None:
            return False
        if label not in ("alt1", "alt2", "alt4", "alt6"):
            return False
        if current_pos is None or current_area is None or override is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        current_x, current_y = current_pos
        override_x, override_y = override['pos']
        current_area = float(current_area or 0.0)
        override_area = float(override.get('area', 0.0) or 0.0)
        if not (
                max(1300, int(frame_width * 0.34)) <= current_x <= min(2700, int(frame_width * 0.71)) and
                max(170, int(frame_height * 0.08)) <= current_y <= max(330, int(frame_height * 0.16))):
            return False
        if current_area < 15.0 or current_area > 110.0:
            return False
        if current_metrics['motion_mean'] < 12.0 or current_metrics['motion_max'] < 75.0:
            return False
        if override_metrics['motion_mean'] >= 3.0 or override_metrics['motion_max'] >= 15.0:
            return False
        if math.hypot(override_x - current_x, override_y - current_y) > 85.0:
            return False
        if override_area > max(70.0, current_area * 1.8):
            return False
        return True

    def _should_accept_hsv_override(self, label, override, current_pos, current_area, prev_pos,
                                    predicted_point, frame_gray, current_filter_key=None):
        if override is None or current_pos is None or prev_pos is None:
            return override is not None

        override_metrics = self._collect_override_candidate_metrics(
            override['pos'], override['area'], prev_pos, predicted_point, frame_gray
        )
        if (
                override_metrics['static_hotspot'] or
                override_metrics['frame0_background'] or
                override_metrics['night_static_side_artifact']):
            reason = (
                f"static patch mean={override_metrics['motion_mean']:.1f} "
                f"max={override_metrics['motion_max']:.1f}"
            )
            hotspot = override_metrics['frame0_hotspot']
            if override_metrics['frame0_background'] and hotspot is not None:
                hotspot_pos = tuple(hotspot.get('pos', list(override['pos'])))
                hotspot_radius = hotspot.get('radius', 0)
                reason = f"frame0 hotspot at {hotspot_pos} r={hotspot_radius}"
            elif override_metrics['night_static_side_artifact']:
                reason = (
                    f"night static side artifact mean={override_metrics['motion_mean']:.1f} "
                    f"max={override_metrics['motion_max']:.1f}"
                )
            print(f"  DEBUG: Rejecting {label} override at {override['pos']} - {reason}")
            return False

        player_penalty, player_reason = self._player_candidate_penalty(
            override['pos'],
            override.get('area', 0.0),
            override_metrics['motion_mean'],
            override_metrics['motion_max'],
            predicted_distance=override_metrics.get('predicted_distance'),
        )
        if player_penalty >= 1500.0:
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                f"player context={player_reason} penalty={player_penalty:.0f}"
            )
            return False

        frame_height = frame_gray.shape[0] if frame_gray is not None else 0
        frame_width = frame_gray.shape[1] if frame_gray is not None else 0
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
        night_far_baseline_dynamic_current = self._night_far_baseline_dynamic_current_lock(
            label,
            current_pos,
            current_area,
            override,
            current_metrics,
            override_metrics,
            frame_gray.shape if frame_gray is not None else None,
        )
        if night_far_baseline_dynamic_current:
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                f"keeping moving night far-baseline ball {current_pos} "
                f"area={float(current_area):.1f}px motion="
                f"{current_metrics['motion_mean']:.1f}/{current_metrics['motion_max']:.1f}"
            )
            return False
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
        night_far_player_visible_current = (
            self._is_night_session_config() and
            label in ("alt1", "alt2", "alt4", "alt6") and
            frame_height and frame_width and
            current_area is not None and
            float(current_area) >= 20.0 and
            current_metrics['motion_mean'] >= 10.0 and
            current_metrics['motion_max'] >= 80.0 and
            max(210, int(frame_height * 0.10)) <= current_pos[1] <= max(360, int(frame_height * 0.18)) and
            max(1650, int(frame_width * 0.44)) <= override['pos'][0] <= min(2050, int(frame_width * 0.64)) and
            max(160, int(frame_height * 0.075)) <= override['pos'][1] <= max(285, int(frame_height * 0.14)) and
            override['pos'][0] >= current_pos[0] + max(60.0, frame_width * 0.018) and
            override['pos'][1] <= current_pos[1] - max(18.0, frame_height * 0.008) and
            current_metrics['prev_distance'] <= max(90.0, override_metrics['prev_distance'] * 0.75) and
            override_metrics['prev_distance'] >= current_metrics['prev_distance'] + max(25.0, frame_width * 0.008) and
            override_area <= max(140.0, float(current_area) * 2.50)
        )
        if night_far_player_visible_current:
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                f"keeping visible night far-player ball {current_pos} "
                f"area={float(current_area):.1f}px motion="
                f"{current_metrics['motion_mean']:.1f}/{current_metrics['motion_max']:.1f}"
            )
            return False
        if (
                label in ("alt1", "alt2", "alt4", "alt6") and
                frame_height and
                hasattr(self, 'net_area_y_max') and
                self.last_motion is not None and
                self.ball_size is not None and
                current_area is not None
        ):
            net_y_max = float(self.net_area_y_max)
            prev_y = float(prev_pos[1])
            current_y = float(current_pos[1])
            override_y = float(override['pos'][1])
            last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
            last_dist = float(self.last_motion.get('distance', 0.0) or 0.0)
            current_dy = current_y - prev_y
            override_dy = override_y - prev_y
            current_pred = current_metrics['predicted_distance']
            override_pred = override_metrics['predicted_distance']
            current_nearer_predicted_rebound = (
                current_pred is not None and
                override_pred is not None and
                current_pred + 28.0 <= override_pred
            )
            current_near_net_rebound = (
                45.0 <= float(self.ball_size) <= 130.0 and
                net_y_max + 20.0 <= prev_y <= net_y_max + max(95.0, frame_height * 0.055) and
                net_y_max - 55.0 <= current_y <= net_y_max + 18.0 and
                last_dy >= max(30.0, frame_height * 0.014) and
                last_dist >= 28.0 and
                current_dy <= -max(20.0, last_dy * 0.35) and
                float(current_area) >= max(42.0, float(self.ball_size) * 0.45) and
                current_dynamic
            )
            override_is_lower_fragment = (
                override_area <= max(18.0, float(current_area) * 0.35) and
                override_y >= current_y + 35.0 and
                override_dy >= -6.0
            )
            if current_near_net_rebound and override_is_lower_fragment and (
                    current_nearer_predicted_rebound or override_y > net_y_max + 35.0):
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping near-net rebound candidate {current_pos} "
                    f"area={float(current_area):.1f}px"
                )
                return False
        if (
                label in ("alt1", "alt2", "alt4", "alt6") and
                current_area is not None and
                self.ball_size is not None and
                current_dynamic and
                current_area >= max(10.0, float(self.ball_size) * 0.45) and
                override_area <= max(3.0, float(current_area) * 0.16) and
                math.hypot(
                    float(override['pos'][0] - current_pos[0]),
                    float(override['pos'][1] - current_pos[1])
                ) <= 14.0 and
                current_metrics['prev_distance'] <= override_metrics['prev_distance'] + 12.0 and
                current_metrics['motion_max'] >= 45.0
        ):
            print(
                f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                f"keeping same-track moving candidate {current_pos} "
                f"area={float(current_area):.1f}px over tiny {override_area:.1f}px"
            )
            return False
        if (
                label in ("alt1", "alt2", "alt4", "alt6") and
                frame_gray is not None and
                self.last_motion is not None and
                current_area is not None and
                override_much_smaller and
                math.hypot(
                    float(override['pos'][0] - current_pos[0]),
                    float(override['pos'][1] - current_pos[1])
                ) <= 18.0
        ):
            current_dx = float(current_pos[0] - prev_pos[0])
            current_dy = float(current_pos[1] - prev_pos[1])
            current_dist = math.hypot(current_dx, current_dy)
            current_direction = math.degrees(math.atan2(current_dy, current_dx)) if current_dist > 0 else None
            current_angle_jump = 0.0
            if self.last_direction is not None and current_direction is not None:
                delta = abs(current_direction - self.last_direction) % 360
                current_angle_jump = min(delta, 360 - delta)
            if self._mid_court_racket_rebound_candidate_ok(
                    current_pos,
                    current_area,
                    frame_gray.shape,
                    current_dx,
                    current_dy,
                    current_angle_jump,
                    motion_mean=current_metrics['motion_mean'],
                    motion_max=current_metrics['motion_max'],
            ):
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping full mid-court rebound candidate {current_pos} "
                    f"area={float(current_area):.1f}px"
                )
                return False
        if (
                label in ("alt1", "alt2", "alt4", "alt6") and
                frame_gray is not None and
                self.last_motion is not None and
                current_area is not None
        ):
            current_dx = float(current_pos[0] - prev_pos[0])
            current_dy = float(current_pos[1] - prev_pos[1])
            current_dist = math.hypot(current_dx, current_dy)
            current_direction = math.degrees(math.atan2(current_dy, current_dx)) if current_dist > 0 else None
            current_angle_jump = 0.0
            if self.last_direction is not None and current_direction is not None:
                delta = abs(current_direction - self.last_direction) % 360
                current_angle_jump = min(delta, 360 - delta)
            mid_court_rebound_current = self._mid_court_racket_rebound_candidate_ok(
                current_pos,
                current_area,
                frame_gray.shape,
                current_dx,
                current_dy,
                current_angle_jump,
                motion_mean=current_metrics['motion_mean'],
                motion_max=current_metrics['motion_max'],
            )
            if mid_court_rebound_current:
                override_dx = float(override['pos'][0] - prev_pos[0])
                override_dy = float(override['pos'][1] - prev_pos[1])
                override_lower_or_downward = (
                    override_dy >= -8.0 or
                    override['pos'][1] >= current_pos[1] + 24
                )
                if override_much_smaller and override_lower_or_downward:
                    print(
                        f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                        f"keeping mid-court rebound candidate {current_pos} "
                        f"area={float(current_area):.1f}px"
                    )
                    return False
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
                self.last_motion is not None and
                self.ball_size is not None and
                current_area is not None and
                prev_pos[1] >= max(880, int(frame_height * 0.40)) and
                float(self.ball_size) >= 60.0 and
                float(current_area) >= max(80.0, float(self.ball_size) * 0.85) and
                override_area <= max(18.0, float(current_area) * 0.25) and
                current_dynamic
        ):
            last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
            last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
            current_dx = float(current_pos[0] - prev_pos[0])
            current_dy = float(current_pos[1] - prev_pos[1])
            override_dx = float(override['pos'][0] - prev_pos[0])
            override_dy = float(override['pos'][1] - prev_pos[1])
            current_dot = last_dx * current_dx + last_dy * current_dy
            override_dot = last_dx * override_dx + last_dy * override_dy
            current_is_hit_reversal = (
                last_dy >= 70.0 and
                current_dot < 0.0 and
                current_dy <= -max(95.0, last_dy * 0.75)
            )
            override_continues_down_or_low = (
                override_dot > 0.0 or
                override_dy >= -12.0 or
                override['pos'][1] >= current_pos[1] + 120
            )
            if current_is_hit_reversal and override_continues_down_or_low:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping large near-player rebound candidate {current_pos} "
                    f"area={float(current_area):.1f}px"
                )
                return False
        if (
                label in ("alt1", "alt2", "alt4", "alt6") and
                frame_height and
                self.last_motion is not None and
                self.ball_size is not None and
                current_area is not None and
                getattr(self, 'stuck_frame_count', 0) >= 2 and
                prev_pos[1] >= max(850, int(frame_height * 0.40)) and
                float(self.ball_size) >= 120.0 and
                float(current_area) >= max(80.0, float(self.ball_size) * 0.55) and
                override_area <= max(70.0, float(current_area) * 0.45) and
                current_dynamic
        ):
            last_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
            last_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
            current_dx = float(current_pos[0] - prev_pos[0])
            current_dy = float(current_pos[1] - prev_pos[1])
            override_dx = float(override['pos'][0] - prev_pos[0])
            override_dy = float(override['pos'][1] - prev_pos[1])
            current_dot = last_dx * current_dx + last_dy * current_dy
            override_dot = last_dx * override_dx + last_dy * override_dy
            current_reverses_incoming_path = (
                current_dot < 0.0 and
                current_dy <= -max(18.0, abs(last_dy) * 0.25)
            )
            override_keeps_incoming_path = (
                override_dot > 0.0 and
                override_dy >= -8.0
            )
            current_not_farther_from_track = (
                current_metrics['prev_distance'] <= override_metrics['prev_distance'] + 25.0
            )
            if current_reverses_incoming_path and override_keeps_incoming_path and current_not_farther_from_track:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping large lower-court reversal candidate {current_pos}"
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
            night_upper_wall_like_override = (
                self._is_night_session_config() and
                label in ("alt4", "alt6") and
                override_metrics['motion_mean'] < 3.5 and
                override_metrics['motion_max'] <= 18.0 and
                override['pos'][1] <= max(150, int(frame_height * 0.075)) and
                override_metrics['prev_distance'] >=
                max(90.0, current_metrics['prev_distance'] * 4.0)
            )
            night_static_override_over_lane_candidate = (
                self._is_night_session_config() and
                label in ("alt4", "alt6") and
                current_near_predicted_lane and
                float(current_area or 0.0) >= 3.0 and
                override_metrics['motion_mean'] < 3.5 and
                override_metrics['motion_max'] <= 18.0 and
                override_metrics['prev_distance'] >=
                current_metrics['prev_distance'] + max(18.0, current_metrics['prev_distance'] * 1.2) and
                (
                    current_metrics['predicted_distance'] is None or
                    override_metrics['predicted_distance'] is None or
                    override_metrics['predicted_distance'] >= current_metrics['predicted_distance'] + 8.0
                )
            )
            override_is_tiny_blob = override_area <= max(3.0, float(current_area or 0.0) * 0.55)
            if (
                    (current_near_predicted_lane and
                     override_leaves_lane and
                     (override_weak_motion or override_is_tiny_blob or night_upper_wall_like_override)) or
                    night_static_override_over_lane_candidate):
                reject_detail = "upper-court jump away from predicted lane"
                if night_upper_wall_like_override:
                    reject_detail = "night upper-wall jump away from predicted lane"
                elif night_static_override_over_lane_candidate:
                    reject_detail = "night static override over predicted-lane ball"
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"{reject_detail} "
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
            night_upper_player_jump_over_lane_candidate = (
                self._is_night_session_config() and
                label in ("alt4", "alt6") and
                current_near_predicted_lane and
                current_dynamic and
                current_metrics['motion_max'] >= 60.0 and
                current_metrics['prev_distance'] <= max(70.0, frame_height * 0.035) and
                override_metrics['prev_distance'] >=
                max(150.0, current_metrics['prev_distance'] * 5.0) and
                override_metrics['predicted_distance'] >=
                current_metrics['predicted_distance'] + max(90.0, current_metrics['prev_distance'] * 3.0)
            )
            if night_upper_player_jump_over_lane_candidate:
                print(
                    f"  DEBUG: Rejecting {label} override at {override['pos']} - "
                    f"keeping dynamic predicted-lane night ball {current_pos}"
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
            relaxed_min_y = max(0, relaxed_min_y - 90)
            if motion_max >= 70.0 or (motion_mean >= 14.0 and motion_max >= 35.0):
                relaxed_min_y = max(0, relaxed_min_y - 70)
        return relaxed_min_y

    def _is_upper_static_recover_jump(self, candidate, frame_shape):
        """Reject far upper-court fallback jumps onto low-motion static blobs."""
        if candidate is None or self.ball_center is None or self.ball_size is None:
            return False

        jump = math.hypot(
            candidate['pos'][0] - self.ball_center[0],
            candidate['pos'][1] - self.ball_center[1],
        )
        return (
            self.ball_size <= 12.0 and
            self.ball_center[1] <= max(360, int(frame_shape[0] * 0.18)) and
            jump > max(
                45.0,
                float((self.last_motion or {}).get('distance', 0.0) or 0.0) * 3.0,
            ) and
            candidate.get('motion_mean', 0.0) < 2.0 and
            candidate.get('motion_max', 0.0) < 10.0
        )

    def _reject_upper_static_recover(self, candidate, frame_shape, label):
        """Return True and log when a fallback recovery is a static upper-court jump."""
        if not self._is_upper_static_recover_jump(candidate, frame_shape):
            return False

        jump = math.hypot(
            candidate['pos'][0] - self.ball_center[0],
            candidate['pos'][1] - self.ball_center[1],
        )
        print(
            f"  DEBUG: Rejecting {label} recover at {candidate['pos']} - "
            f"upper static jump {jump:.1f}px motion="
            f"{candidate.get('motion_mean', 0.0):.1f}/"
            f"{candidate.get('motion_max', 0.0):.1f}"
        )
        return True

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
        if self._top_return_recently_expired_at_current_anchor():
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
        high_rising_top_exit = (
            y_prev <= max(58, int(frame_height * 0.030)) and
            dy <= -16.0 and
            dist >= 18.0 and
            predicted_y <= max(34, int(frame_height * 0.018)) and
            ball_size <= 18.0
        )
        if high_rising_top_exit:
            return "edge_high_stall"
        upper_band_stalled_exit = (
            getattr(self, 'stuck_frame_count', 0) >= 1 and
            y_prev <= max(120, int(frame_height * 0.055)) and
            ball_size <= 18.0 and
            recent_vel >= 45.0 and
            dist >= 18.0 and
            dy < 0.0
        )
        if upper_band_stalled_exit:
            return "edge_high_stall"
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
            # During the initial serve flight, the far-court/receiver band overlaps
            # the broad "upper side" heuristic.  A single fast tracked step there is
            # not proof that the ball has left the image.  Keep ordinary tracking for
            # one more frame; if the ball is genuinely gone, the resulting miss/stuck
            # evidence can arm the delayed top-return search on the next frame.  True
            # literal top-edge exits (handled by the stricter branches above) remain
            # immediate.
            pre_bounce_serve_flight = (
                getattr(self, '_serve_contact_grace_frames', 0) > 0 and
                self.ground_bounce_count == 0
            )
            if pre_bounce_serve_flight and getattr(self, 'stuck_frame_count', 0) == 0:
                _verbose_debug_print(
                    f"  DEBUG: [TOP-RETURN DEFER] pre-bounce serve flight at "
                    f"{self.ball_center}; waiting for miss evidence"
                )
                return None
            return "upper_side"
        return None

    def _should_start_top_edge_return_wait(self, frame_shape):
        """Return True once the tracked ball has actually clipped/exited the top edge."""
        if self.ball_center is None:
            return False
        if self._top_return_recently_expired_at_current_anchor():
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

    def _top_return_recently_expired_at_current_anchor(self, max_frames=90):
        expired_anchor = getattr(self, '_top_return_expired_anchor', None)
        expired_frame = getattr(self, '_top_return_expired_frame', -1000000)
        if expired_anchor is None or self.ball_center is None:
            return False
        if self.frame_count - expired_frame > max_frames:
            return False
        return math.hypot(
            self.ball_center[0] - expired_anchor[0],
            self.ball_center[1] - expired_anchor[1],
        ) <= 4.0

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
            # A fast return can already expose a large ball-shaped contour
            # before its centre clears the ordinary 50 px top-noise band.
            # Require all three independent signals (large area, strong mean
            # motion, and a high peak) so persistent top-line specks cannot
            # end the off-screen wait.  This is the a prior frame re-entry after the
            # ball genuinely disappeared above the frame at a prior frame.
            strong_large_top_reentry = (
                mode == "upper_side" and
                elapsed >= 30 and
                0 <= cy <= max_reentry_y and
                area >= 70.0 and
                motion_mean >= 25.0 and
                motion_max >= 120.0
            )
            strong_partial_top_reentry = (
                strong_large_top_reentry and cy < min_reentry_y
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
            strong_confirmed_reentry = (
                strong_visible_reentry or strong_partial_top_reentry
            )
            if strong_large_top_reentry:
                # The held marker is the exit point, not a one-frame motion
                # prediction.  After a long off-screen flight, a strongly
                # confirmed return can legitimately be roughly 500-650 px
                # away even when the last visible horizontal step was small.
                x_drift_cap = max(x_drift_cap, 650.0)
            blind_wait_frames = 20 if strong_confirmed_reentry else 30
            if strong_confirmed_reentry:
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
            if cy < min_reentry_y and not (
                    partial_top_reentry or strong_partial_top_reentry):
                return False, f"top-return y {cy} < min_reentry_y {min_reentry_y}"
            if cy > max_reentry_y:
                return False, f"top-return y {cy} > max_reentry_y {max_reentry_y}"
            if not strong_motion:
                return False, f"top-return weak motion mean={motion_mean:.1f} max={motion_max:.1f} area={area:.1f}"
            return True, None

        edge_anchor_y = anchor[1]
        if mode == "edge_high_stall":
            edge_anchor_y = min(edge_anchor_y, 40)
        min_reentry_y = max(50, edge_anchor_y + 18)
        base_max_reentry_y = max(110, int(frame_height * 0.11))
        dynamic_max_reentry_y = base_max_reentry_y
        if elapsed >= 22:
            dynamic_max_reentry_y = max(
                dynamic_max_reentry_y,
                int(edge_anchor_y + 150 + ((elapsed - 22) * 75)),
            )
        max_reentry_y = min(frame_height - 1, dynamic_max_reentry_y)
        deep_reentry_y = max(220, edge_anchor_y + 165)
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
            cy >= max(24, edge_anchor_y - 32) and
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
            cy >= max(20, edge_anchor_y - 10) and
            area >= 70.0 and
            motion_mean >= 25.0 and
            motion_max >= 120.0
        )
        strong_partial_reentry = strong_visible_reentry and cy < min_reentry_y
        reentry_x_cap = x_cap
        if mode == "edge_high_stall":
            reentry_x_cap = max(reentry_x_cap, 920.0)
        if strong_visible_reentry:
            reentry_x_cap = max(reentry_x_cap, 420.0)
        if deep_reentry:
            reentry_x_cap = max(reentry_x_cap, 650.0)

        if elapsed < 24:
            return False, f"top-return blind wait elapsed={elapsed}f"
        if abs(cx - anchor[0]) > reentry_x_cap:
            return False, f"top-return x drift {abs(cx - anchor[0]):.1f}px > {reentry_x_cap:.1f}px"
        directional_progress = max(60.0, min(260.0, abs(exit_dx) * 3.0))
        if mode != "edge_high_stall" and exit_dx >= 12.0 and cx < anchor[0] + directional_progress:
            return False, f"top-return x {cx} lacks rightward reentry progress from {anchor[0]}"
        if mode != "edge_high_stall" and exit_dx <= -12.0 and cx > anchor[0] - directional_progress:
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

    def _top_return_player_reacq_jump_override_ok(
            self, pos, area, motion_mean, motion_max, source, frame_shape):
        """Allow a large player-reacq jump only for a proven top return."""
        if not self._top_return_wait_active():
            return False, "top-return wait inactive"
        if getattr(self, '_top_return_mode', None) not in ('upper_side', 'upper_racket'):
            return False, "top-return mode does not allow player-reacq jump override"
        if source not in ('primary', 'regular', 'alt'):
            return False, f"top-return source {source} not allowed for upper-side reentry"
        if area < 70.0 or motion_mean < 25.0 or motion_max < 120.0:
            return False, "top-return player-reacq jump lacks strong ball evidence"
        return self._top_return_reentry_ok(
            pos, area, motion_mean, motion_max, frame_shape
        )

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
        elif mode == "edge_high_stall":
            y2 = min(
                frame_height,
                max(band_height, min(130, max(90, int(min(anchor_y, 40)) + 90))),
            )
            half_width = int(max(620, min(920, self.max_ball_speed * 6.0)))
            x1 = max(0, int(anchor_x) - half_width)
            x2 = min(frame_width, int(anchor_x) + half_width)
            center_x = (x1 + x2) // 2 if x2 > x1 else frame_width // 2
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

    def _start_point_context(self, origin_pos, serve_start_frame=None, history_origin_pos=None):
        self._pending_rally_end_reason = None
        self._pending_rally_end_frame = -1
        self._awaiting_serve_bounce = False
        self._point_serve_start_side = None
        self._point_target_service_side = None
        self._last_serve_bounce_frame = -1000000
        self._last_serve_bounce_point = None
        self._last_serve_bounce_net_contact_like = False
        self._last_serve_bounce_was_in = False
        self._serve_net_contact_frame = -1000000
        self._serve_net_contact_point = None
        self._serve_net_pass_frame = -1000000
        self._serve_net_pass_point = None
        self._serve_net_zone_frames = 0
        self._serve_landed_in_current_attempt = False
        self._serve_in_recorded_attempt = None
        self._serve_start_requires_confirmation = False
        # A far-end toss can look complete before the player actually serves
        # (for example, a practice toss which is caught again).  Keep that
        # start provisional until its first outgoing flight is coherent.
        self._provisional_serve_start_kind = None
        self._provisional_serve_start_frame = -1000000
        self._provisional_serve_first_vector = None
        self._provisional_serve_forward_steps = 0
        # Do not reset the live tracker merely because a far-side toss later
        # proves to be a catch.  Resetting here was found to merge the next
        # two real rallies.  Instead keep a small, point-local shadow record;
        # local AI may later prove the real outgoing launch without changing
        # the HSV track that the rest of the match already depends on.
        self._tainted_provisional_serve_start = None
        self._verified_serve_launch_frame = None
        self._verified_serve_launch_position = None
        # Local-AI evidence belongs to one point only.  In particular, an
        # accepted repair from a preceding point must not make a later false
        # serve look confirmed (or vice versa).
        self._local_ai_recovery_count = 0
        self._local_ai_all_body_rejections = 0
        self._local_ai_tight_roi_follow = None
        self._local_ai_tight_roi_accept_frame = -1000000
        self._local_ai_tight_roi_attempt_frame = -1000000
        self._local_ai_tight_roi_previous_gray = None
        self._discard_provisional_serve_from_ai = False
        self._post_serve_recovery_hold_until_frame = -1000000
        self._post_serve_pre_net_recovery_active = False
        self._post_serve_pre_net_recovery_started_frame = -1000000
        self._local_ai_follow_until_frame = -1
        self._local_ai_handoff_deadline_frame = -1
        self._reset_point_score_context()
        history_pos = origin_pos if history_origin_pos is None else history_origin_pos
        self._start_point_history_row(history_pos, serve_start_frame=serve_start_frame)
        if origin_pos is None or not hasattr(self, 'serve_area_x_min'):
            return
        serve_mid_x = (self.serve_area_x_min + self.serve_area_x_max) / 2.0
        start_side = "left" if origin_pos[0] < serve_mid_x else "right"
        target_side = "right" if start_side == "left" else "left"
        self._point_serve_start_side = start_side
        self._point_target_service_side = target_side
        self._awaiting_serve_bounce = True

    def _point_history_headers(self):
        return [
            'current_score',
            'server',
            'serve_attempt',
            'point_index',
            'serve_start_frame',
            'point_end_frame',
            'duration_frames',
            'start_position',
            'end_position',
            'rally_shots',
            'point_awarded',
            'winner',
            'end_reason',
            'why',
            'category',
            'next_server',
            'next_serve',
            'serve_player_side',
            'serve_player_position',
            'serve_ball_position',
            'serve_player_source',
            'serve_racket_side',
            'receive_player_side',
            'receive_player_position',
            'receive_area',
            'receive_player_source',
            'net_player',
            'shot_strokes',
            'tracking_losses',
            'player_tracking_summary',
        ]

    def _format_history_point(self, point):
        if point is None:
            return ''
        try:
            return f"({int(point[0])},{int(point[1])})"
        except Exception:
            return str(point)

    @staticmethod
    def _timestamped_point_history_path(path):
        directory, filename = os.path.split(path)
        stem, ext = os.path.splitext(filename)
        if not stem:
            stem = "point_history"
        if not ext:
            ext = ".csv"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = os.path.join(directory, f"{stem}_{timestamp}{ext}")
        if not os.path.exists(candidate):
            return candidate

        suffix = 2
        while True:
            candidate = os.path.join(directory, f"{stem}_{timestamp}_{suffix}{ext}")
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    def _initialize_point_history_file(self):
        if not self.write_point_history or self._point_history_initialized:
            return
        if not self.point_history_file:
            self.write_point_history = False
            return
        self.point_history_file = self._timestamped_point_history_path(self.point_history_file)
        directory = os.path.dirname(self.point_history_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.point_history_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._point_history_headers())
            writer.writeheader()
        self._point_history_initialized = True
        print(f"[POINT_HISTORY] Writing point history to {self.point_history_file}")

    def _start_point_history_row(self, origin_pos, serve_start_frame=None):
        if not self.write_point_history:
            return
        self._initialize_point_history_file()
        self._point_history_current = {
            'serve_start_frame': self.frame_count if serve_start_frame is None else int(serve_start_frame),
            'tracking_start_frame': int(self.frame_count),
            'server_idx': self._current_server_index(),
            'serve_attempt': self._serve_attempt_label(),
            'start_position': origin_pos,
            'tracking_trace': [],
            'shot_events': [],
            'tracking_losses': [],
            'serve_context': self._player_serve_context(origin_pos),
        }
        tracker = getattr(self, "player_tracker", None)
        if tracker is not None:
            try:
                tracker.begin_point()
            except Exception:
                pass

    def _record_ball_loss_event(self, reason, position=None, recovery=None):
        """Record every loss/recovery transition for later point diagnosis.

        The tracker must be able to explain a missed ball without turning the
        miss itself into a point-ending decision.  These compact events are
        written into the eventual point-history row and are also printed while
        processing, so a replay can be audited frame-by-frame.
        """
        context = getattr(self, '_point_history_current', None)
        if context is None:
            return
        events = context.setdefault('tracking_losses', [])
        frame = int(getattr(self, 'frame_count', -1))
        reason = str(reason or 'unknown')
        if events and events[-1].get('frame') == frame and events[-1].get('reason') == reason:
            return
        last_motion = getattr(self, 'last_motion', None) or {}
        event = {
            'frame': frame,
            'position': self._format_history_point(position if position is not None else self.ball_center),
            'reason': reason,
            'stuck': int(getattr(self, 'stuck_frame_count', 0)),
            'last_seen_frame': int(getattr(self, 'last_seen_frame', -1) or -1),
            'motion': {
                'dx': round(float(last_motion.get('dx', 0.0) or 0.0), 1),
                'dy': round(float(last_motion.get('dy', 0.0) or 0.0), 1),
                'distance': round(float(last_motion.get('distance', 0.0) or 0.0), 1),
            },
            'recovery': str(recovery) if recovery else '',
        }
        events.append(event)
        # Keep the row bounded if a camera is unavailable for a long period,
        # while retaining the complete recent diagnostic sequence.
        del events[:-120]
        print(
            f"[BALL_LOSS_DIAGNOSTIC] f{frame}: reason={reason} "
            f"pos={event['position']} stuck={event['stuck']} recovery={event['recovery'] or 'pending'}"
        )


    def _append_point_history_row(self, reason, outcome, winner_idx, end_position, score_value, point_awarded,
                                  point_end_frame=None):
        if not self.write_point_history:
            return
        self._initialize_point_history_file()
        context = self._point_history_current or {}
        start_frame = context.get('serve_start_frame', self.point_start_frame_internal)
        end_frame = self.frame_count if point_end_frame is None else int(point_end_frame)
        duration = ''
        if start_frame is not None:
            duration = max(0, end_frame - int(start_frame))
        server_idx = context.get('server_idx', self._current_server_index())
        self._point_history_point_index += 1
        self._point_audit_traces[self._point_history_point_index] = list(
            context.get('tracking_trace', [])
        )
        row = {
            'point_index': self._point_history_point_index,
            'serve_start_frame': start_frame if start_frame is not None else '',
            'point_end_frame': end_frame,
            'duration_frames': duration,
            'server': self.player_names[server_idx] if server_idx is not None else '',
            'serve_attempt': context.get('serve_attempt', self._serve_attempt_label()),
            'start_position': self._format_history_point(context.get('start_position')),
            'end_position': self._format_history_point(end_position),
            'rally_shots': int(getattr(self, '_last_point_hit_count', 0)),
            'point_awarded': 'yes' if point_awarded else 'no',
            'winner': self.player_names[winner_idx] if winner_idx is not None else '',
            'end_reason': reason or '',
            'why': outcome.get('detail', '') if outcome else '',
            'category': outcome.get('category', '') if outcome else '',
            'current_score': score_value or self._score_summary(),
            'next_server': self.player_names[self._current_server_index()],
            'next_serve': self._serve_attempt_label(),
        }
        serve_context = context.get('serve_context') or {}
        shot_events = context.get('shot_events') or []
        net_player_idx = context.get('net_player_idx')
        if net_player_idx is None and 'net' in (reason or '').lower():
            net_player_idx = getattr(self, '_last_racket_contact_player', None)
        if net_player_idx is None and 'net' in (reason or '').lower() and shot_events:
            last_shot_player = shot_events[-1].get('player') if isinstance(shot_events[-1], dict) else None
            if last_shot_player in self.player_names:
                net_player_idx = self.player_names.index(last_shot_player)
        player_summary = {}
        tracker = getattr(self, "player_tracker", None)
        if tracker is not None:
            try:
                player_summary = tracker.point_summary()
            except Exception:
                player_summary = {}
        row.update({
            'serve_player_side': serve_context.get('player_side', ''),
            'serve_player_position': self._format_history_point((
                serve_context.get('player_position')[0], serve_context.get('player_position')[1]
            )) if serve_context.get('player_position') else '',
            'serve_ball_position': self._format_history_point((
                serve_context.get('serve_x'), serve_context.get('serve_y')
            )) if serve_context.get('serve_x') is not None else '',
            'serve_player_source': serve_context.get('server_source', ''),
            'serve_racket_side': serve_context.get('racket_side', ''),
            'receive_player_side': serve_context.get('receiver_side', ''),
            'receive_player_position': self._format_history_point((
                serve_context.get('receiver_position')[0], serve_context.get('receiver_position')[1]
            )) if serve_context.get('receiver_position') else '',
            'receive_area': serve_context.get('receiver_area', ''),
            'receive_player_source': serve_context.get('receiver_source', ''),
            'net_player': self.player_names[net_player_idx] if net_player_idx in (0, 1) else '',
            'shot_strokes': json.dumps(shot_events, separators=(',', ':'), sort_keys=True),
            'tracking_losses': json.dumps(
                context.get('tracking_losses') or [], separators=(',', ':'), sort_keys=True
            ),
            'player_tracking_summary': json.dumps(player_summary, separators=(',', ':'), sort_keys=True),
        })
        with open(self.point_history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._point_history_headers())
            writer.writerow(row)
        self._point_history_current = None

    def _current_server_index(self):
        return self.score_game_index % 2


    def _current_history_serve_start_frame(self):
        context = self._point_history_current or {}
        start_frame = context.get('serve_start_frame', self.point_start_frame_internal)
        if start_frame in (None, ''):
            return None
        try:
            return int(start_frame)
        except (TypeError, ValueError):
            return None






    def _begin_provisional_serve_start(self, kind):
        """Arm a short validation window for a toss-derived serve start.

        A ball toss alone is intentionally not a point.  This guard is used
        only by the far-top ``post-hit`` shortcut, whose old rise/drop test
        could not distinguish a caught practice toss from an actual racket
        launch.
        """
        self._provisional_serve_start_kind = str(kind)
        self._provisional_serve_start_frame = int(self.frame_count)
        self._provisional_serve_first_vector = None
        self._provisional_serve_forward_steps = 0

    def _validate_provisional_serve_start(self, previous_position, tracked_position):
        """Return a rejection reason, or ``None`` while/after validation."""
        if getattr(self, '_provisional_serve_start_kind', None) is None:
            return None
        if previous_position is None or tracked_position is None:
            return None

        age = int(self.frame_count) - int(
            getattr(self, '_provisional_serve_start_frame', self.frame_count)
        )
        if age > 12:
            return None

        dx = float(tracked_position[0]) - float(previous_position[0])
        dy = float(tracked_position[1]) - float(previous_position[1])
        distance = math.hypot(dx, dy)
        if distance < 18.0:
            return None

        # In image coordinates a far-end server launches toward increasing Y;
        # a near-end server launches toward decreasing Y.  We deliberately do
        # not use the configured diagonal here: this is a broad sanity gate,
        # not a replacement for court geometry.
        server_end = getattr(self, '_active_serve_area_end', None)
        netward_y = dy if server_end == 'far' else -dy
        first_vector = getattr(self, '_provisional_serve_first_vector', None)
        if first_vector is None:
            # The first substantial move after a far-top toss must head
            # netward (toward increasing image Y).  At a prior frame the false
            # practice-toss track jumped from the descending toss to a
            # stationary top-border artifact, giving a 77px move upward.
            # It is neither a racket launch nor a playable serve, so reject
            # it before the stale point can survive the short confirmation
            # window and shift the score/history.
            if distance >= 55.0 and netward_y <= -35.0:
                return (
                    'first substantial toss motion moved away from court '
                    f'(delta=({dx:.0f},{dy:.0f}), netward={netward_y:.0f})'
                )
            self._provisional_serve_first_vector = (dx, dy, distance, netward_y)
            if netward_y >= 12.0:
                self._provisional_serve_forward_steps = 1
            return None

        first_dx, first_dy, first_distance, first_netward_y = first_vector
        cosine = (dx * first_dx + dy * first_dy) / max(
            1.0, distance * float(first_distance)
        )
        # A caught toss generates an immediate large reversal: in the failing
        # sequence a prior frame -> a prior frame it was +112,+130 followed by -105,-105.
        # A struck serve cannot reverse this sharply before it reaches the
        # net, so discard the tentative start and resume serve scanning.
        if (
                distance >= 55.0 and first_distance >= 55.0 and
                cosine <= -0.45 and
                netward_y <= -35.0 and first_netward_y >= 35.0):
            return (
                'toss flight reversed before confirmed racket launch '
                f'(cos={cosine:.2f}, first=({first_dx:.0f},{first_dy:.0f}), '
                f'current=({dx:.0f},{dy:.0f}))'
            )

        if netward_y >= 12.0 and cosine >= -0.10:
            self._provisional_serve_forward_steps = int(
                getattr(self, '_provisional_serve_forward_steps', 0)
            ) + 1
        if int(getattr(self, '_provisional_serve_forward_steps', 0)) >= 3:
            print(
                f"[SERVE_START_CONFIRMED] f{self.frame_count}: "
                f"{self._provisional_serve_start_kind} has coherent netward flight"
            )
            self._provisional_serve_start_kind = None
            self._serve_start_requires_confirmation = False
        return None

    def _reset_point_score_context(self):
        self._point_hit_count = 0
        self._last_point_hit_count = 0
        self._last_counted_contact_frame = -1000000
        self._last_racket_contact_player = None

    def _record_racket_contact(self, point, label=None):
        self._last_racket_contact_frame = self.frame_count
        self._last_racket_contact_point = point
        player_idx = self._player_index_at_point(point) if point is not None else None
        self._last_racket_contact_player = player_idx
        tracker = getattr(self, "player_tracker", None)
        if tracker is not None:
            try:
                player_name = self.player_names[player_idx] if player_idx in (0, 1) else ""
                shot_event = tracker.record_shot(
                    self.frame_count,
                    point,
                    player_name=player_name,
                    label=label,
                )
                if self._point_history_current is not None:
                    self._point_history_current.setdefault('shot_events', []).append(shot_event)
                print(
                    f"[PLAYER_SHOT] f{self.frame_count}: player={shot_event.get('player') or 'unknown'} "
                    f"stroke={shot_event.get('stroke', 'unknown')} "
                    f"confidence={float(shot_event.get('confidence', 0.0)):.2f} "
                    f"side={shot_event.get('player_side')} racket_side={shot_event.get('racket_side')}"
                )
                if self._point_history_current is not None and label and "net" in label.lower():
                    self._point_history_current['net_player_idx'] = player_idx
            except Exception as error:
                if int(getattr(self, "frame_count", -1)) % 120 == 0:
                    print(f"[PLAYER_SHOT_CONTEXT_ERROR] f{self.frame_count}: {error}")
        if self.point_start_frame_internal is not None and self._last_counted_contact_frame != self.frame_count:
            self._point_hit_count += 1
            self._last_counted_contact_frame = self.frame_count
            if player_idx is not None:
                self.point_stats[player_idx]['total_hits'] += 1
            player_text = self.player_names[player_idx] if player_idx is not None else "unknown"
            label_text = f" {label}" if label else ""
            print(
                f"[RALLY_HIT] f{self.frame_count}: hit#{self._point_hit_count} "
                f"player={player_text}{label_text} point={point}"
            )
        return player_idx

    def _recent_contact_info(self, frame=None, max_frames=180):
        contact_point = getattr(self, '_last_racket_contact_point', None)
        contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        contact_player = getattr(self, '_last_racket_contact_player', None)
        if contact_player is None:
            contact_player = self._player_index_at_point(contact_point, frame)
        recent_contact = (
            contact_player is not None and
            (self.frame_count - contact_frame) <= max_frames
        )
        return recent_contact, contact_player, contact_frame, contact_point

    def _point_bounced_after_contact(self, contact_frame):
        return int(getattr(self, 'last_ground_bounce_frame', -1000000)) > int(contact_frame)

    def _timeout_on_opponent_side_after_contact(self, end_position, contact_player, frame):
        if end_position is None or contact_player is None:
            return False
        landing_player = self._player_index_at_point(end_position, frame)
        if landing_player is None or landing_player == contact_player:
            return False
        if frame is not None:
            outside_court, _, _, _, _, _ = self._point_outside_singles_court(end_position, frame)
            if outside_court:
                return False
        return True

    def _in_court_timeout_landing_outcome(self, end_position, frame):
        if end_position is None or frame is None:
            return None
        landing_player = self._player_index_at_point(end_position, frame)
        if landing_player is None:
            return None
        if self._is_night_session_config():
            outside_sideline, _, _, _ = self._point_outside_singles_sidelines(end_position, frame)
            if outside_sideline:
                return None
        else:
            outside_court, _, _, _, _, _ = self._point_outside_singles_court(end_position, frame)
            if outside_court:
                return None
        return self._point_outcome(
            1 - landing_player,
            "ball stopped on player side and was not returned",
            "unreturned",
            landing_player,
        )

    def _stuck_timeout_end_frame(self, point_start_frame=None, frame=None):
        """Return the first frame of a contiguous stuck run, not its timeout frame.

        The state machine waits for several unchanged frames before deciding a
        point has ended.  Persisting the timeout frame makes every stationary
        landing appear artificially late and can leave a stale player/racket
        marker as the apparent point endpoint.  The point history should point
        to the beginning of that stable run while the live overlay may still
        use the current frame.
        """
        # Point-end callers pass the current image in ``frame``; the tracker
        # frame number is always available on ``self.frame_count``.  Accept a
        # scalar frame number for unit callers, but never try to cast an image
        # array to int (which would either fail or produce an unusable value).
        if frame is None:
            current_frame = int(self.frame_count)
        else:
            try:
                current_frame = int(frame) if np.isscalar(frame) else int(self.frame_count)
            except (TypeError, ValueError):
                current_frame = int(self.frame_count)
        stuck_frames = max(0, int(getattr(self, "stuck_frame_count", 0) or 0))
        if stuck_frames <= 1:
            return current_frame
        first_stuck_frame = current_frame - stuck_frames + 1
        if point_start_frame is not None:
            first_stuck_frame = max(int(point_start_frame), first_stuck_frame)
        return first_stuck_frame

    def _night_stuck_timeout_out_reason(self, end_position, frame):
        if (
                not self._is_night_session_config() or
                end_position is None or
                frame is None):
            return None
        suppressed_frame = int(getattr(self, '_last_out_bounce_suppressed_frame', -1000000))
        suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
        static_artifact = (
            suppressed_point is not None and
            0 <= int(getattr(self, 'frame_count', 0)) - suppressed_frame <= 40 and
            math.hypot(
                float(end_position[0]) - float(suppressed_point[0]),
                float(end_position[1]) - float(suppressed_point[1]),
            ) <= max(100.0, float(getattr(self, 'ball_size', 0.0) or 0.0) * 3.0)
        )
        if static_artifact:
            print(
                f"Frame {self.frame_count}: [STUCK-OUT SUPPRESSED] "
                f"static marker {end_position} follows suppressed artifact at {suppressed_point}"
            )
            return None
        outside_sideline, side, _, _ = self._point_outside_singles_sidelines(end_position, frame)
        if outside_sideline and side:
            return f"Ball bounced out of court ({side} sideline)"
        outside_court, side, _, _, _, _ = self._point_outside_singles_court(end_position, frame)
        if outside_court and side in ("far baseline", "near baseline"):
            return f"Ball bounced out of court ({side})"
        return None

    def _is_lower_right_exit_score_context(self, point, frame=None):
        if point is None or frame is None:
            return False
        frame_height, frame_width = frame.shape[:2]
        x, y = point
        return (
            float(x) >= frame_width - max(35.0, frame_width * 0.01) and
            float(y) >= frame_height - max(45.0, frame_height * 0.025)
        )

    def _point_outcome(self, winner_idx, detail, category, loser_idx=None):
        return {
            'winner_idx': winner_idx,
            'loser_idx': loser_idx if loser_idx is not None and winner_idx is not None else (
                (1 - winner_idx) if winner_idx is not None else None
            ),
            'detail': detail,
            'category': category,
        }

    def _serve_attempt_label(self):
        return "1st" if int(getattr(self, 'current_serve_attempt', 1)) <= 1 else "2nd"

    def _is_service_fault_reason(self, reason):
        reason_lower = (reason or "").lower()
        if "serve bounce outside" in reason_lower:
            return True
        if "serve" in reason_lower and "outside" in reason_lower:
            return True
        if "ball hit the net" in reason_lower:
            frames_since_start = None
            if self.point_start_frame_internal is not None:
                frames_since_start = self.frame_count - self.point_start_frame_internal
            if getattr(self, '_serve_landed_in_current_attempt', False):
                if (
                    frames_since_start is not None and
                    frames_since_start <= 45 and
                    int(getattr(self, '_point_hit_count', 0)) == 0
                ):
                    return True
                return False
            if frames_since_start is not None and frames_since_start > 60:
                return False
            recent_contact, contact_player, _, _ = self._recent_contact_info()
            if recent_contact and contact_player != self._current_server_index():
                return False
            return True
        return False

    def _is_serve_let_reason(self, reason):
        return "serve let" in (reason or "").lower()

    def _serve_net_touch_active(self, window_frames=110):
        contact_frame = int(getattr(self, '_serve_net_contact_frame', -1000000))
        if contact_frame <= -100000:
            return False
        return 0 <= (self.frame_count - contact_frame) <= int(window_frames)

    def _serve_net_pass_active(self, window_frames=36):
        pass_frame = int(getattr(self, '_serve_net_pass_frame', -1000000))
        if pass_frame <= -100000:
            return False
        return 0 <= (self.frame_count - pass_frame) <= int(window_frames)

    def _serve_bounce_frame_limit(self):
        if self._serve_net_touch_active(window_frames=120):
            return 105
        if self._serve_net_pass_active(window_frames=36):
            return 75
        return 45

    def _serve_net_fault_reason(self, reason):
        reason_text = reason or "Serve bounce outside service box"
        reason_lower = reason_text.lower()
        if reason_lower.startswith("serve net"):
            return reason_text
        if reason_lower.startswith("serve "):
            return "Serve net then " + reason_text[6:]
        if reason_lower.startswith("ball "):
            return "Serve net then " + reason_text[5:]
        return "Serve net then " + reason_text

    def _night_serve_net_bounce_wrong_side_reason(self, point, frame):
        if not self._is_night_session_config() or point is None or frame is None:
            return None
        model = self._build_service_box_model(frame)
        if model is not None:
            net_y = float(model.get('net_y', (self.net_area_y_min + self.net_area_y_max) / 2.0))
        elif hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            net_y = float((self.net_area_y_min + self.net_area_y_max) / 2.0)
        else:
            return None

        y = float(point[1])
        slack = max(18.0, frame.shape[0] * 0.010)
        wrong_side = False
        if self.serve_direction_dy < 0:
            wrong_side = y > net_y + slack
        elif self.serve_direction_dy > 0:
            wrong_side = y < net_y - slack
        if not wrong_side:
            return None

        target_side = getattr(self, '_point_target_service_side', None)
        if target_side in ("left", "right"):
            base_reason = f"Serve bounce outside {target_side} service box"
        else:
            base_reason = "Serve bounce outside service box"
        return self._serve_net_fault_reason(base_reason)

    def _mark_serve_net_contact_candidate(self, ball_position, frame):
        if ball_position is None or frame is None:
            return False
        if not getattr(self, '_awaiting_serve_bounce', False):
            self._serve_net_zone_frames = 0
            return False
        if not self._is_night_session_config():
            return False
        if self.point_start_frame_internal is None:
            return False

        frames_since_start = self.frame_count - self.point_start_frame_internal
        candidate_limit = self._serve_bounce_frame_limit()
        candidate_limit = max(candidate_limit, 75)
        if frames_since_start < 6 or frames_since_start > candidate_limit:
            return False

        net_geometry = self._net_contact_geometry(ball_position)
        if net_geometry is None:
            return False

        net_zone_active = (
            net_geometry['inside'] or
            net_geometry['near_polygon'] or
            net_geometry.get('inside_hull', False) or
            net_geometry.get('near_hull', False) or
            net_geometry['near_top_tape']
        )
        if not net_zone_active:
            self._serve_net_zone_frames = 0
            return False

        self._serve_net_pass_frame = self.frame_count
        self._serve_net_pass_point = tuple(ball_position)
        self._serve_net_zone_frames = int(getattr(self, '_serve_net_zone_frames', 0)) + 1
        if self._serve_net_touch_active(window_frames=120):
            return True

        current_speed = float(self.last_motion.get('distance', 0.0) or 0.0) if self.last_motion else 0.0
        current_dy = float(self.last_motion.get('dy', 0.0) or 0.0) if self.last_motion else 0.0
        prev_speed = float(self.prev_motion.get('distance', 0.0) or 0.0) if self.prev_motion else 0.0
        prev_dir = self.prev_motion.get('direction_deg') if self.prev_motion else None
        curr_dir = self.last_motion.get('direction_deg') if self.last_motion else None
        angle_diff = 0.0
        if prev_dir is not None and curr_dir is not None:
            delta = abs(curr_dir - prev_dir) % 360
            angle_diff = min(delta, 360 - delta)

        direct_contact_zone = net_geometry['inside'] or net_geometry['near_top_tape']
        slowed_or_deflected = (
            direct_contact_zone and
            prev_speed > 0.0 and
            current_speed <= max(28.0, prev_speed * 0.82) and
            angle_diff >= 35.0 and
            current_dy >= 4.0
        )
        falling_net_surface = (
            net_geometry['inside'] or
            net_geometry.get('inside_hull', False) or
            net_geometry.get('near_hull', False)
        )
        falling_into_net = (
            falling_net_surface and
            self._serve_net_zone_frames >= 2 and
            current_dy >= max(28.0, frame.shape[0] * 0.014) and
            current_speed <= max(170.0, frame.shape[0] * 0.080)
        )
        if not (falling_into_net or slowed_or_deflected):
            return False

        self._serve_net_contact_frame = self.frame_count
        self._serve_net_contact_point = tuple(ball_position)
        self._last_serve_bounce_net_contact_like = True
        self._add_impact_marker(
            ball_position,
            kind="net_contact",
            ttl=14,
            label="serve net touch",
        )
        print(
            f"Frame {self.frame_count}: [SERVE NET TOUCH] pos={ball_position} "
            f"speed={current_speed:.1f}px zone_frames={self._serve_net_zone_frames} "
            f"signed_dist={net_geometry['signed_dist']:.1f} "
            f"hull_dist={net_geometry.get('hull_signed_dist', 0.0):.1f} "
            f"top_dist={net_geometry['top_dist']:.1f}"
        )
        return True

    def _record_serve_in(self):
        server_idx = self._current_server_index()
        stats = self.serve_stats[server_idx]
        if int(getattr(self, 'current_serve_attempt', 1)) <= 1:
            stats['first_in'] += 1
        else:
            stats['second_in'] += 1
        self._serve_in_recorded_attempt = int(getattr(self, 'current_serve_attempt', 1))
        self._serve_landed_in_current_attempt = True
        print(
            f"[SERVE_STATS] f{self.frame_count}: {self.player_names[server_idx]} "
            f"{self._serve_attempt_label()} serve in "
            f"first={stats['first_in']}/{stats['first_in'] + stats['first_faults']} "
            f"second={stats['second_in']}/{stats['second_in'] + stats['double_faults']} "
            f"DF={stats['double_faults']}"
        )

    def _record_serve_fault(self, reason):
        server_idx = self._current_server_index()
        receiver_idx = 1 - server_idx
        stats = self.serve_stats[server_idx]
        recorded_attempt = getattr(self, '_serve_in_recorded_attempt', None)
        if getattr(self, '_serve_landed_in_current_attempt', False) and recorded_attempt is not None:
            if int(recorded_attempt) <= 1:
                stats['first_in'] = max(0, stats['first_in'] - 1)
            else:
                stats['second_in'] = max(0, stats['second_in'] - 1)
            self._serve_landed_in_current_attempt = False
            self._serve_in_recorded_attempt = None
        if int(getattr(self, 'current_serve_attempt', 1)) <= 1:
            stats['first_faults'] += 1
            self.current_serve_attempt = 2
            self._serve_landed_in_current_attempt = False
            self._serve_in_recorded_attempt = None
            print(
                f"[SERVE_FAULT] f{self.frame_count}: {self.player_names[server_idx]} "
                f"first serve fault ({reason}); next serve is second serve"
            )
            return None, "first serve fault"

        stats['double_faults'] += 1
        self.current_serve_attempt = 1
        self._serve_landed_in_current_attempt = False
        self._serve_in_recorded_attempt = None
        print(
            f"[SERVE_FAULT] f{self.frame_count}: {self.player_names[server_idx]} "
            f"second serve fault ({reason}); double fault"
        )
        return receiver_idx, "double fault"

    def _record_serve_let(self, reason):
        server_idx = self._current_server_index()
        stats = self.serve_stats[server_idx]
        recorded_attempt = getattr(self, '_serve_in_recorded_attempt', None)
        if getattr(self, '_serve_landed_in_current_attempt', False) and recorded_attempt is not None:
            if int(recorded_attempt) <= 1:
                stats['first_in'] = max(0, stats['first_in'] - 1)
            else:
                stats['second_in'] = max(0, stats['second_in'] - 1)
        self._serve_landed_in_current_attempt = False
        self._serve_in_recorded_attempt = None
        print(
            f"[SERVE_LET] f{self.frame_count}: {self.player_names[server_idx]} "
            f"{self._serve_attempt_label()} serve let ({reason}); replay same serve"
        )
        return None, "serve let"

    def _initial_server_end(self):
        start_side = getattr(self, 'start_server_side', None)
        if start_side in ("near", "far"):
            return start_side
        base_end = getattr(self, '_base_serve_area_end', None)
        if base_end in ("near", "far"):
            return base_end
        return "near" if int(getattr(self, 'serve_direction_dy', -1)) < 0 else "far"

    def _other_end(self, court_end):
        return "far" if court_end == "near" else "near"

    def _configured_serve_area_dict(self):
        if not hasattr(self, 'serve_area_x_min'):
            return None
        return {
            'x_min': int(self.serve_area_x_min),
            'x_max': int(self.serve_area_x_max),
            'y_min': int(self.serve_area_y_min),
            'y_max': int(self.serve_area_y_max),
            'points': [tuple(p) for p in getattr(self, 'serve_area_points', [])],
        }

    def _area_copy(self, area):
        if area is None:
            return None
        return {
            'x_min': int(area['x_min']),
            'x_max': int(area['x_max']),
            'y_min': int(area['y_min']),
            'y_max': int(area['y_max']),
            'points': [tuple(p) for p in area.get('points', [])],
        }

    def _remember_base_serve_geometry(self):
        area = self._configured_serve_area_dict()
        if area is None:
            return
        base_dy = int(getattr(self, 'serve_direction_dy', -1))
        if base_dy not in (-1, 1):
            base_dy = -1
        base_dx = int(getattr(self, 'serve_direction_dx', 1))
        if base_dx not in (-1, 1):
            base_dx = 1
        self._base_serve_area = area
        self._base_serve_direction_dx = base_dx
        self._base_serve_direction_dy = base_dy
        self._base_serve_area_end = "near" if base_dy < 0 else "far"
        self._active_serve_area_end = None
        self._base_serve_ball_size_min = int(getattr(self, 'serve_ball_size_min', 3))
        self._base_serve_ball_size_max = int(getattr(self, 'serve_ball_size_max', 80))

    def _tracking_ball_size_max(self):
        """Use the full configured tracking size even when far-serve detection is active."""
        return max(
            200,
            int(getattr(self, 'serve_ball_size_max', 80)),
            int(getattr(self, '_base_serve_ball_size_max', 80)),
        )

    def _serve_end_for_current_game(self):
        configured_end = getattr(self, '_base_serve_area_end', None) or self._initial_server_end()
        base_end = getattr(self, 'start_server_side', None) or configured_end
        game_number = max(1, int(getattr(self, 'score_game_index', 0)) + 1)
        if game_number >= 3 and (game_number % 2) == 1:
            return self._other_end(base_end)
        return base_end

    def _opposite_serve_area_from_base(self, frame_shape=None):
        base = getattr(self, '_base_serve_area', None)
        if base is None:
            return None
        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            net_y = (float(self.net_area_y_min) + float(self.net_area_y_max)) / 2.0
        elif frame_shape is not None:
            net_y = float(frame_shape[0]) * 0.5
        else:
            net_y = 1080.0

        y_min = 2.0 * net_y - float(base['y_max'])
        y_max = 2.0 * net_y - float(base['y_min'])
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        height = max(1.0, float(base['y_max'] - base['y_min']))

        frame_h = None
        frame_w = None
        if frame_shape is not None:
            frame_h, frame_w = frame_shape[:2]
        else:
            cap_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if getattr(self, 'cap', None) is not None else 0
            cap_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) if getattr(self, 'cap', None) is not None else 0
            frame_h = int(cap_h or 2160)
            frame_w = int(cap_w or 3840)

        if y_min < 0:
            # The perspective mirror of a near-end serve box can extend above
            # the frame. Keep the far-end search around the far baseline/toss
            # band instead of including the noisy top edge.
            y_min = max(0.0, net_y - height * 0.70)
            y_max = min(float(frame_h - 1), net_y - height * 0.10)
        if y_max >= frame_h:
            y_min -= (y_max - (frame_h - 1))
            y_max = float(frame_h - 1)
        y_min = max(0.0, y_min)
        if (y_max - y_min) < height * 0.75:
            y_max = min(float(frame_h - 1), y_min + height)

        x_min = max(0, min(frame_w - 1, int(base['x_min'])))
        x_max = max(x_min + 1, min(frame_w, int(base['x_max'])))
        y1 = int(round(max(0.0, y_min)))
        y2 = int(round(min(float(frame_h - 1), y_max)))
        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y1,
            'y_max': max(y1 + 1, y2),
            'points': [
                (x_min, y1),
                (x_max, y1),
                (x_max, max(y1 + 1, y2)),
                (x_min, max(y1 + 1, y2)),
            ],
        }

    def _serve_area_for_end(self, court_end, frame_shape=None):
        base = getattr(self, '_base_serve_area', None)
        base_end = getattr(self, '_base_serve_area_end', None)
        far_area = getattr(self, '_configured_far_serve_area', None)
        if court_end == "far" and far_area is not None:
            return self._area_copy(far_area)
        if base is None or base_end is None:
            return self._configured_serve_area_dict()
        if court_end == base_end:
            return self._area_copy(base)
        return self._opposite_serve_area_from_base(frame_shape=frame_shape)

    def _apply_active_serve_geometry(self, frame_shape=None, force_log=False):
        area = self._serve_area_for_end(self._serve_end_for_current_game(), frame_shape=frame_shape)
        if area is None:
            return
        active_end = self._serve_end_for_current_game()
        base_end = getattr(self, '_base_serve_area_end', active_end)
        old_end = getattr(self, '_active_serve_area_end', None)
        old_dy = int(getattr(self, 'serve_direction_dy', 0))

        self.serve_area_x_min = int(area['x_min'])
        self.serve_area_x_max = int(area['x_max'])
        self.serve_area_y_min = int(area['y_min'])
        self.serve_area_y_max = int(area['y_max'])
        self.serve_area_points = [tuple(p) for p in area.get('points', [])]
        self.serve_direction_dx = int(getattr(self, '_base_serve_direction_dx', self.serve_direction_dx))
        base_dy = int(getattr(self, '_base_serve_direction_dy', self.serve_direction_dy))
        self.serve_direction_dy = base_dy if active_end == base_end else -base_dy
        if active_end == base_end:
            self.serve_ball_size_min = int(getattr(self, '_base_serve_ball_size_min', self.serve_ball_size_min))
            self.serve_ball_size_max = int(getattr(self, '_base_serve_ball_size_max', self.serve_ball_size_max))
        else:
            self.serve_ball_size_min = int(getattr(self, 'far_serve_ball_size_min', 1))
            self.serve_ball_size_max = int(getattr(self, 'far_serve_ball_size_max', 120))

        self._active_serve_area_end = active_end
        if old_dy != self.serve_direction_dy:
            self._service_box_model = None
            self._service_box_frame_shape = None
            self._white_line_visual_model = None
            self._white_line_visual_frame_shape = None
        if force_log or old_end != active_end:
            print(
                f"[SERVE_AREA] f{self.frame_count}: game={int(getattr(self, 'score_game_index', 0)) + 1} "
                f"end={active_end} X={self.serve_area_x_min}-{self.serve_area_x_max} "
                f"Y={self.serve_area_y_min}-{self.serve_area_y_max} "
                f"size={self.serve_ball_size_min}-{self.serve_ball_size_max} "
                f"direction={self.serve_direction_label()}"
            )



    def _court_sides_swapped_for_game(self):
        completed_games = max(0, int(getattr(self, 'score_game_index', 0)))
        return ((completed_games + 1) // 2) % 2 == 1

    def _player_index_on_court_end(self, court_end):
        if court_end not in ("near", "far"):
            return None
        p1_start_end = self._initial_server_end()
        p1_current_end = self._other_end(p1_start_end) if self._court_sides_swapped_for_game() else p1_start_end
        return 0 if court_end == p1_current_end else 1

    def _court_end_for_point(self, point, frame=None):
        if point is None:
            return None
        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            net_y = (float(self.net_area_y_min) + float(self.net_area_y_max)) / 2.0
        elif frame is not None:
            net_y = frame.shape[0] * 0.5
        else:
            return None
        return "near" if float(point[1]) >= net_y else "far"

    def _player_index_at_point(self, point, frame=None):
        return self._player_index_on_court_end(self._court_end_for_point(point, frame))

    def _point_score_text(self, player_idx):
        points = self.score_points
        mine = points[player_idx]
        other = points[1 - player_idx]
        if mine >= 3 and other >= 3:
            if mine == other:
                return "40"
            return "AD" if mine > other else "40"
        labels = ["0", "15", "30", "40"]
        return labels[min(mine, 3)]

    @staticmethod
    def _score_text_from_counters(games, points, player_idx):
        mine = int(points[player_idx])
        other = int(points[1 - player_idx])
        if mine >= 3 and other >= 3:
            if mine == other:
                return "40"
            return "AD" if mine > other else "40"
        return ["0", "15", "30", "40"][min(mine, 3)]

    @classmethod
    def _score_summary_from_counters(cls, games, points):
        return (
            f"{int(games[0])}:{int(games[1])} "
            f"{cls._score_text_from_counters(games, points, 0)}:"
            f"{cls._score_text_from_counters(games, points, 1)}"
        )

    @staticmethod
    def _shadow_award_score(games, points, winner_idx):
        """Return copied tennis counters after one awarded point."""
        next_games = [int(games[0]), int(games[1])]
        next_points = [int(points[0]), int(points[1])]
        winner_idx = int(winner_idx)
        loser_idx = 1 - winner_idx
        next_points[winner_idx] += 1
        if (
            next_points[winner_idx] >= 4 and
            next_points[winner_idx] - next_points[loser_idx] >= 2
        ):
            next_games[winner_idx] += 1
            next_points = [0, 0]
        return next_games, next_points

    def _score_summary(self):
        return self._score_history_value()

    def _score_history_value(self):
        return (
            f"{self.score_games[0]}:{self.score_games[1]} "
            f"{self._point_score_text(0)}:{self._point_score_text(1)}"
        )

    def _serve_stats_text(self, player_idx):
        stats = self.serve_stats[player_idx]
        first_total = stats['first_in'] + stats['first_faults']
        first_pct = int(round((stats['first_in'] / first_total) * 100)) if first_total else 0
        return (
            f"1st {stats['first_in']}/{first_total} {first_pct}% "
            f"1F {stats['first_faults']} DF {stats['double_faults']}"
        )

    def _point_stats_text(self, player_idx):
        stats = self.point_stats[player_idx]
        return (
            f"Out {stats['out_errors']} Net {stats['net_errors']} "
            f"Unret {stats['unreturned_winners']} Hits {stats['total_hits']}"
        )

    def _infer_point_outcome(self, reason, end_position=None, frame=None):
        reason_text = reason or ""
        reason_lower = reason_text.lower()
        server_idx = self._current_server_index()
        receiver_idx = 1 - server_idx

        if "serve bounce outside" in reason_lower or (
                "serve" in reason_lower and "outside" in reason_lower):
            return self._point_outcome(receiver_idx, "server lost serve fault", "serve_fault", server_idx)

        recent_contact, contact_player, contact_frame, _ = self._recent_contact_info(frame=frame)
        landing_player = self._player_index_at_point(end_position or self.ball_center, frame)

        if "ball hit the net" in reason_lower:
            if landing_player is not None:
                return self._point_outcome(1 - landing_player, "ball hit net on player side", "net_error", landing_player)
            if recent_contact:
                return self._point_outcome(1 - contact_player, "last hitter hit net", "net_error", contact_player)
            return self._point_outcome(receiver_idx, "server hit net", "net_error", server_idx)

        if "bounce outside singles court" in reason_lower or "bounced out of court" in reason_lower:
            if (
                "bounce outside singles court" in reason_lower and
                self._is_lower_right_exit_score_context(end_position or self.ball_center, frame) and
                landing_player is not None
            ):
                return self._point_outcome(
                    1 - landing_player,
                    "ball bounced on player side and was not returned",
                    "unreturned",
                    landing_player,
                )
            if landing_player is not None:
                return self._point_outcome(landing_player, "ball out on player court; opponent fault", "out_error", 1 - landing_player)
            if (
                self.point_start_frame_internal is not None and
                (self.frame_count - self.point_start_frame_internal) <= 45
            ):
                return self._point_outcome(receiver_idx, "early serve/rally out by server", "out_error", server_idx)
            if recent_contact:
                return self._point_outcome(1 - contact_player, "last hitter missed court", "out_error", contact_player)

        if "upper fence" in reason_lower and "fell down" in reason_lower:
            if recent_contact:
                return self._point_outcome(contact_player, "ball jumped and was not returned", "unreturned", 1 - contact_player)
            if landing_player is not None:
                return self._point_outcome(1 - landing_player, "ball jumped and was not returned", "unreturned", landing_player)

        if (
            "point_timeout" in reason_lower or
            "stuck_timeout" in reason_lower or
            "stopped" in reason_lower
        ):
            stopped_outcome = self._in_court_timeout_landing_outcome(
                end_position or self.ball_center,
                frame,
            )
            if stopped_outcome is not None:
                return stopped_outcome

        if "stuck_timeout" in reason_lower:
            if recent_contact:
                if (
                        self._point_bounced_after_contact(contact_frame) or
                        self._timeout_on_opponent_side_after_contact(
                            end_position or self.ball_center,
                            contact_player,
                            frame,
                        )):
                    return self._point_outcome(contact_player, "opponent did not return after bounce", "unreturned", 1 - contact_player)
                return self._point_outcome(1 - contact_player, "last hitter lost point", "out_error", contact_player)
            if landing_player is not None:
                return self._point_outcome(1 - landing_player, "ball stuck on player side", "unreturned", landing_player)

        if (
            "hitter side" in reason_lower or
            "fence" in reason_lower or
            "timeout" in reason_lower or
            "lost" in reason_lower
        ):
            if recent_contact:
                if (
                        self._point_bounced_after_contact(contact_frame) or
                        self._timeout_on_opponent_side_after_contact(
                            end_position or self.ball_center,
                            contact_player,
                            frame,
                        )):
                    return self._point_outcome(contact_player, "opponent did not return after bounce", "unreturned", 1 - contact_player)
                return self._point_outcome(1 - contact_player, "last hitter lost point", "out_error", contact_player)

        if "bounced twice" in reason_lower or "stopped" in reason_lower:
            if recent_contact:
                return self._point_outcome(contact_player, "opponent did not return after bounce", "unreturned", 1 - contact_player)
            if landing_player is not None:
                return self._point_outcome(1 - landing_player, "double bounce on opponent side", "unreturned", landing_player)

        return self._point_outcome(None, "winner unknown", "unknown")

    def _would_point_win_game(self, winner_idx):
        if winner_idx is None:
            return False
        loser_idx = 1 - winner_idx
        winner_points = self.score_points[winner_idx] + 1
        loser_points = self.score_points[loser_idx]
        return winner_points >= 4 and (winner_points - loser_points) >= 2

    def _ignore_unresolved_timeout_if_game_decider(self, reason, end_position=None, frame=None):
        if self._should_ignore_unconfirmed_serve_start(reason):
            self._ignore_unconfirmed_serve_start_result(reason)
            return True

        reason_lower = (reason or "").lower()
        unresolved_timeout = (
            "point_timeout" in reason_lower or
            "stuck_timeout" in reason_lower or
            "timeout" in reason_lower or
            "lost" in reason_lower
        )
        if not unresolved_timeout:
            return False

        outcome = self._infer_point_outcome(reason, end_position=end_position, frame=frame)
        winner_idx = outcome.get('winner_idx')
        if not self._would_point_win_game(winner_idx):
            return False

        self._last_scored_point_end_frame = self.frame_count
        self._last_point_winner = None
        self._last_point_score_reason = "ignored unresolved timeout at game point"
        self._last_point_outcome_category = "unknown"
        self._last_point_hit_count = int(getattr(self, '_point_hit_count', 0))
        self.current_serve_attempt = 1
        self._serve_landed_in_current_attempt = False
        self._serve_in_recorded_attempt = None
        self._serve_start_requires_confirmation = False
        self._last_confirmed_point_end_frame = self.frame_count
        ignored_outcome = self._point_outcome(
            None,
            "ignored unresolved timeout at game point",
            "unknown",
        )
        self._append_point_history_row(
            reason,
            ignored_outcome,
            None,
            end_position,
            self._score_summary(),
            point_awarded=False,
        )
        print(
            f"[SCORE] f{self.frame_count}: no point awarded reason={reason} "
            f"score={self._score_summary()} detail=ignored unresolved timeout at game point "
            f"category=unknown hits={self._last_point_hit_count} next_serve={self._serve_attempt_label()}"
        )
        return True

    def _should_ignore_unconfirmed_serve_start(self, reason):
        if not getattr(self, '_serve_start_requires_confirmation', False):
            return False
        if getattr(self, '_serve_landed_in_current_attempt', False):
            return False
        if self._is_serve_let_reason(reason):
            return False
        if self._is_service_fault_reason(reason):
            return False
        reason_lower = (reason or "").lower()
        if "serve bounce outside" in reason_lower:
            return False
        return True


    def _ignore_unconfirmed_serve_start_result(self, reason):
        self._last_scored_point_end_frame = self.frame_count
        self._last_point_winner = None
        self._last_point_score_reason = "ignored unconfirmed serve start"
        self._last_point_outcome_category = "ignored_serve_start"
        self._last_point_hit_count = int(getattr(self, '_point_hit_count', 0))
        self.current_serve_attempt = 1
        self._serve_landed_in_current_attempt = False
        self._serve_in_recorded_attempt = None
        self._serve_start_requires_confirmation = False
        self._point_history_current = None
        print(
            f"[SERVE_START_IGNORED] f{self.frame_count}: reason={reason} "
            f"started_at={self.point_start_frame_internal} score={self._score_summary()}"
        )

    def _terminal_moving_ball_candidate(self, end_position, current_image,
                                        previous_gray, current_gray,
                                        allow_static_anywhere=False,
                                        allow_small_static=False):
        """Find a real moving ball when a terminal marker is stale/unsupported.

        The terminal detector can briefly keep a court/player blob after the
        ball has moved away.  A two-frame motion mask intersected with the
        regular ball HSV range gives us a conservative, image-based recovery
        signal.  This is intentionally independent of player tracking so it
        also repairs stale markers in empty court areas (for example, point 10
        at a prior frame).
        """
        if (end_position is None or current_image is None or
                previous_gray is None or current_gray is None):
            return None
        try:
            hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
            ex, ey = int(round(float(end_position[0]))), int(round(float(end_position[1])))
            if not (0 <= ex < hsv.shape[1] and 0 <= ey < hsv.shape[0]):
                return None

            # If the reported endpoint itself is supported by ball-colored
            # pixels, it is already plausible and should not be moved.
            radius = 7
            x1, x2 = max(0, ex - radius), min(hsv.shape[1], ex + radius + 1)
            y1, y2 = max(0, ey - radius), min(hsv.shape[0], ey + radius + 1)
            local_hsv = hsv[y1:y2, x1:x2]
            local_mask = cv2.inRange(
                local_hsv, np.array([20, 50, 130], dtype=np.uint8),
                np.array([85, 255, 255], dtype=np.uint8)
            )
            endpoint_support = float(np.count_nonzero(local_mask)) / max(1, local_mask.size)
            endpoint_zone = self._player_point_zone(end_position)
            if endpoint_support >= 0.12 and endpoint_zone is None:
                return None

            diff = cv2.absdiff(previous_gray, current_gray)
            _, moving = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            ball_mask = cv2.inRange(
                hsv, np.array([20, 50, 130], dtype=np.uint8),
                np.array([85, 255, 255], dtype=np.uint8)
            )
            moving_ball = cv2.bitwise_and(moving, ball_mask)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(moving_ball)
            candidates = []
            for idx in range(1, count):
                area = float(stats[idx, cv2.CC_STAT_AREA])
                if area < 40.0 or area > 700.0:
                    continue
                cx, cy = float(centers[idx, 0]), float(centers[idx, 1])
                ix, iy = int(round(cx)), int(round(cy))
                if not (0 <= ix < hsv.shape[1] and 0 <= iy < hsv.shape[0]):
                    continue
                h, s, v = map(int, hsv[iy, ix])
                if not (20 <= h <= 85 and s >= 50 and v >= 130):
                    continue
                # Prefer compact, substantial motion blobs; player/line
                # fragments are usually smaller after the intersection.
                x, y, w, h_box = [int(stats[idx, k]) for k in range(4)]
                compactness = area / max(1.0, float(w * h_box))
                score = area + (80.0 if (ix < 220 or ix > hsv.shape[1] - 220) else 0.0)
                score += min(80.0, compactness * 80.0)
                candidates.append((score, (ix, iy), area))
            # When the ball has just gone out and stopped, frame differencing
            # can be empty (or be dominated by a moving player fragment).  A
            # compact, highly saturated yellow blob at the image edge is still
            # strong evidence; a prior frame is exactly this case.
            static_count, static_labels, static_stats, static_centers = cv2.connectedComponentsWithStats(ball_mask)
            static_candidates = []
            width = int(hsv.shape[1])
            min_static_area = 25.0 if allow_small_static else 60.0
            max_static_side = 24 if allow_static_anywhere else 32
            for idx in range(1, static_count):
                area = float(static_stats[idx, cv2.CC_STAT_AREA])
                x, y, w, h_box = [int(static_stats[idx, k]) for k in range(4)]
                if (area < min_static_area or area > 400.0 or
                        w > max_static_side or h_box > max_static_side):
                    continue
                # A stopped ball may be outside the singles court but still
                # well inside the camera image.  For normal terminal repair,
                # keep the old edge-only guard so random court highlights do
                # not win.  Out/bounce endpoints explicitly opt in to the
                # broader compact-ball search.
                if (not allow_static_anywhere and x >= 350 and
                        (x + w) <= width - 350):
                    continue
                cx, cy = (float(static_centers[idx, 0]), float(static_centers[idx, 1]))
                # Endpoint repair runs outside the normal candidate-selection
                # path, so it must apply the same learned/background false
                # point protection itself.  Without this, the repair can
                # repeatedly turn a fixed yellow fixture (notably the night
                # session's (3049, 40) hotspot) into a fake "moving ball".
                static_pos = (int(round(cx)), int(round(cy)))
                if (self._find_persistent_false_point(static_pos, filter_key=None) is not None or
                        self._find_frame0_background_hotspot(static_pos) is not None):
                    continue
                roi = hsv[y:y + h_box, x:x + w]
                valid = ball_mask[y:y + h_box, x:x + w] > 0
                pixels = roi[valid]
                if pixels.size == 0:
                    continue
                mean_s, mean_v = float(np.mean(pixels[:, 1])), float(np.mean(pixels[:, 2]))
                if mean_s < 90.0 or mean_v < 175.0:
                    continue
                compactness = area / max(1.0, float(w * h_box))
                score = mean_s + 0.25 * mean_v + 30.0 * compactness
                if allow_static_anywhere and (
                        cx < 350.0 or cx > float(width - 500) or
                        cy < 180.0 or cy > float(hsv.shape[0] - 300)):
                    # Out/bounce endpoints commonly leave the ball stopped
                    # near a court boundary.  Give that compact, saturated
                    # blob precedence over a larger moving racket fragment.
                    score += 600.0
                static_candidates.append((score, static_pos, area))
            candidates.sort(key=lambda item: item[0], reverse=True)
            static_candidates.sort(key=lambda item: item[0], reverse=True)
            if static_candidates and (not candidates or static_candidates[0][0] > candidates[0][0]):
                return static_candidates[0][1]
            if candidates:
                return candidates[0][1]
            if static_candidates:
                return static_candidates[0][1]
            return None
        except Exception:
            return None

    def _terminal_player_overlap_position(self, reason, end_position, frame=None):
        """Backtrack an endpoint when the terminal marker is a player/blob artifact."""
        if end_position is None:
            return end_position

        # Point-end callers pass the current image in ``frame``; the tracker
        # frame number is always available on ``self.frame_count``. Accept a
        # scalar frame number for unit callers, but never cast an image array.
        if frame is None:
            current_frame = int(self.frame_count)
        else:
            try:
                current_frame = int(frame) if np.isscalar(frame) else int(self.frame_count)
            except (TypeError, ValueError):
                current_frame = int(self.frame_count)

        # A true ball can be inside the player's detector box (especially just
        # after a low hit), so simply rejecting every player-zone point is too
        # aggressive.  Look for a small, bright, moving ball blob near the
        # tracked player's lower body before falling back to motion history.
        current_image = getattr(self, '_terminal_current_frame', None)
        previous_gray = getattr(self, '_terminal_previous_gray', None)
        current_gray = getattr(self, '_terminal_current_gray', None)
        tracker = getattr(self, 'player_tracker', None)

        # First use the generic motion/color recovery.  It covers both player
        # overlap and stale markers in otherwise empty court regions.
        reason_text = str(reason or '').lower()
        allow_static_anywhere = any(token in reason_text for token in (
            'out of court', 'bounced out', 'stopped on', 'double bounce',
            'outside singles', 'outside service'
        ))
        allow_small_static = 'stopped on' in reason_text
        moving_candidate = self._terminal_moving_ball_candidate(
            end_position, current_image, previous_gray, current_gray,
            allow_static_anywhere=allow_static_anywhere,
            allow_small_static=allow_small_static
        )
        if moving_candidate is not None:
            zone = self._player_point_zone(end_position)
            print(
                f"Frame {current_frame}: [TERMINAL MOTION REPAIR] "
                f"tracked={end_position} zone={zone or 'none'}; "
                f"using moving ball={moving_candidate}"
            )
            return moving_candidate

        zone = self._player_point_zone(end_position)
        if tracker is None or zone not in ('player_head_hat', 'player_shoes', 'racket_fragment', 'player_body'):
            return end_position

        if current_image is not None and previous_gray is not None and current_gray is not None and tracker is not None:
            try:
                track = next(
                    t for t in tracker.tracks.values()
                    if t.bbox is not None and tracker.point_zone(end_position) is not None and
                    float(t.bbox[0]) - 40 <= float(end_position[0]) <= float(t.bbox[0] + t.bbox[2]) + 40 and
                    float(t.bbox[1]) - 40 <= float(end_position[1]) <= float(t.bbox[1] + t.bbox[3]) + 40
                )
                if track.head is not None and track.shoes is not None:
                    diff = cv2.absdiff(previous_gray, current_gray)
                    _, moving = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    count, labels, stats, centers = cv2.connectedComponentsWithStats(moving)
                    hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
                    best = None
                    hx, hy = track.head
                    sx, sy = track.shoes
                    for idx in range(1, count):
                        area = float(stats[idx, cv2.CC_STAT_AREA])
                        if area < 20.0 or area > 900.0:
                            continue
                        cx, cy = (float(centers[idx, 0]), float(centers[idx, 1]))
                        if cy < float(hy) + 160.0 or cy > float(sy) + 180.0:
                            continue
                        if math.hypot(cx - float(sx), cy - float(sy)) > 360.0:
                            continue
                        ix, iy = int(round(cx)), int(round(cy))
                        if not (0 <= ix < hsv.shape[1] and 0 <= iy < hsv.shape[0]):
                            continue
                        h, s, v = map(int, hsv[iy, ix])
                        if not (20 <= h <= 85 and s >= 50 and v >= 130):
                            continue
                        score = abs(area - 400.0) + math.hypot(cx - float(sx), cy - float(sy)) * 0.35
                        if best is None or score < best[0]:
                            best = (score, (int(round(cx)), int(round(cy))), area)
                    if best is not None:
                        print(
                            f"Frame {current_frame}: [TERMINAL PLAYER ARTIFACT] "
                            f"zone={zone} tracked={end_position}; using moving ball="
                            f"{best[1]} area={best[2]:.1f}"
                        )
                        return best[1]
            except Exception:
                # Endpoint repair is advisory; retain the safe history fallback.
                pass
        for entry in reversed(getattr(self, 'motion_history', [])):
            entry_frame = int(entry.get('frame', -1000000))
            if entry_frame >= current_frame:
                continue
            if current_frame - entry_frame > 24:
                break
            candidate = entry.get('pos')
            if candidate is None:
                continue
            candidate_zone = self._player_point_zone(candidate)
            if candidate_zone is not None:
                continue
            if float(entry.get('distance', 0.0) or 0.0) < 2.0:
                continue
            print(
                f"Frame {current_frame}: [TERMINAL PLAYER ARTIFACT] "
                f"zone={zone} tracked={end_position}; using prior ball={candidate} "
                f"from f{entry_frame}"
            )
            return tuple(int(v) for v in candidate)
        return end_position

    def _resume_from_terminal_motion_candidate(self, reason, end_position, frame=None):
        """Cancel an ambiguous terminal decision when local vision sees the ball moving.

        A timeout is an *inference*, unlike a verified net/out/bounce event.
        Before scoring that inference, inspect the two decoded frames already
        held by the tracker.  When their motion mask and ball-colour mask
        agree on a compact moving candidate away from the stale endpoint,
        that candidate is stronger evidence than the timeout.  Continue from
        it and let normal point-end logic decide the rally later.

        This is deliberately local and conditional: no model is run on every
        frame, and verified bounce/net/out decisions are untouched.
        """
        reason_text = str(reason or "").lower()
        ambiguous_timeout = any(token in reason_text for token in (
            "stuck", "stopped", "lost", "timeout",
        ))
        if not ambiguous_timeout or end_position is None:
            return False

        current_image = getattr(self, "_terminal_current_frame", None)
        previous_gray = getattr(self, "_terminal_previous_gray", None)
        current_gray = getattr(self, "_terminal_current_gray", None)
        candidate = self._terminal_moving_ball_candidate(
            end_position,
            current_image,
            previous_gray,
            current_gray,
            # A terminal decision must be supported by an actually moving
            # candidate. Do not let a static yellow line/edge component
            # cancel an otherwise valid timeout.
            allow_static_anywhere=False,
            allow_small_static=False,
        )
        if candidate is None:
            return False

        previous = tuple(int(v) for v in end_position)
        recovered = tuple(int(v) for v in candidate)
        # A candidate must show local change in the decoded frame pair. This
        # rejects persistent yellow fixtures at the image edge, which can
        # otherwise look like a stopped ball to a colour-only check.
        height, width = current_gray.shape[:2]
        radius = 12
        x1, x2 = max(0, recovered[0] - radius), min(width, recovered[0] + radius + 1)
        y1, y2 = max(0, recovered[1] - radius), min(height, recovered[1] + radius + 1)
        local_diff = cv2.absdiff(previous_gray[y1:y2, x1:x2], current_gray[y1:y2, x1:x2])
        motion_ratio = float(np.count_nonzero(local_diff >= 18)) / max(1, local_diff.size)
        if motion_ratio < 0.035:
            return False
        displacement = math.hypot(recovered[0] - previous[0], recovered[1] - previous[1])
        # Ignore a near-identical result: it only confirms the existing
        # terminal marker rather than revealing a separate ball flight.
        if displacement < max(70.0, min(180.0, float(getattr(self, "ball_size", 0.0) or 0.0) * 1.5)):
            return False

        dx = recovered[0] - previous[0]
        dy = recovered[1] - previous[1]
        distance = math.hypot(dx, dy)
        direction = math.degrees(math.atan2(dy, dx)) if distance > 0 else None
        self.ball_center = recovered
        self.last_delta = (dx, dy)
        self.last_motion = {
            "distance": distance,
            "dx": dx,
            "dy": dy,
            "direction_deg": direction,
        }
        if distance >= 3.0:
            self.last_nonzero_motion = dict(self.last_motion)
        velocity_history = getattr(self, "ball_velocity_history", None)
        if isinstance(velocity_history, list):
            velocity_history.append(distance)
            del velocity_history[:-20]
        motion_history = getattr(self, "motion_history", None)
        if isinstance(motion_history, list):
            motion_history.append({
                "frame": int(self.frame_count),
                "distance": distance,
                "direction_deg": direction,
                "pos": recovered,
                "prev_pos": previous,
            })
            del motion_history[:-200]
        self.last_seen_frame = int(self.frame_count)
        self.stuck_frame_count = 0
        self.edge_wait = False
        self._player_reacq_protect_until_frame = -1
        self._player_reacq_motion_failed_until_frame = -1
        context = getattr(self, "_point_history_current", None)
        if isinstance(context, dict):
            context.setdefault("tracking_trace", []).append({
                "frame": int(self.frame_count),
                "pos": [recovered[0], recovered[1]],
                "size": float(self.ball_size) if self.ball_size is not None else None,
                "stuck": 0,
                "source": "terminal_motion_recovery",
            })
        print(
            f"Frame {self.frame_count}: [TERMINAL VERIFY CONTINUE] "
            f"suppressed '{reason}' at {previous}; local motion recovered "
            f"ball={recovered} displacement={displacement:.1f}px "
            f"motion={motion_ratio:.3f}"
        )
        return True

    def _record_point_result(self, reason, end_position=None, frame=None, history_end_frame=None):
        if self._last_scored_point_end_frame == self.frame_count:
            return None

        requested_history_end_frame = history_end_frame
        reason_lower = (reason or "").lower()
        end_position = self._terminal_player_overlap_position(reason, end_position, frame=frame)
        if "video_read_failure" in reason_lower:
            outcome = self._point_outcome(
                None,
                "video ended before point could finish",
                "media_error",
            )
            winner_idx = None
            detail = outcome['detail']
            history_end_frame = None
        else:
            outcome = None
            winner_idx = None
            detail = None
            history_end_frame = requested_history_end_frame

        # A shadow local-AI check may have proven that a provisional far-side
        # toss was caught and that the real serve left much later.  Apply that
        # evidence only when recording the terminal row: the live tracker and
        # all intervening scoring state deliberately remain unchanged.
        verified_launch = getattr(self, '_verified_serve_launch_frame', None)
        if verified_launch is not None and self._point_history_current is not None:
            previous_start = self._point_history_current.get('serve_start_frame')
            self._point_history_current['serve_start_frame'] = int(verified_launch)
            print(
                f"[SERVE_HISTORY_RECONCILED] f{self.frame_count}: "
                f"start f{previous_start} -> f{verified_launch} from shadow local-AI evidence"
            )

        if outcome is None and self._should_ignore_unconfirmed_serve_start(reason):
            self._ignore_unconfirmed_serve_start_result(reason)
            return None

        if outcome is None:
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
        frames_since_ground_bounce = self.frame_count - getattr(self, 'last_ground_bounce_frame', -1000000)
        recent_ground_bounce = frames_since_ground_bounce <= 2

        if self.frame_count <= (self.start_frame + 15):
            return False, "Early-serve grace"
        if getattr(self, 'edge_wait', False):
            return False, "Edge wait"
        # A near-player occlusion can leave a racket/body fragment frozen at an
        # apparent sideline position.  Suppress every point-end classifier during
        # the bounded reacquisition grace, including out-of-court checks below.
        # A small, genuinely stationary ball on the court is different: it is a
        # valid unreturned/double-bounce terminal and must not be held until the
        # long player-reacquisition timeout.  Large blobs (body/racket fragments)
        # stay protected by the normal guard.
        reacq_protect_active = int(getattr(self, '_player_reacq_protect_until_frame', -1)) >= self.frame_count
        stationary_ball_override = False
        if reacq_protect_active:
            ball_size_now = float(getattr(self, 'ball_size', 0.0) or 0.0)
            stuck_now = int(getattr(self, 'stuck_frame_count', 0) or 0)
            net_y = float(getattr(self, 'net_area_y_max', height * 0.42) or height * 0.42)
            court_ball_zone = float(y) > net_y + max(120.0, height * 0.08)
            small_ball_blob = 0.0 < ball_size_now <= max(180.0, width * 0.012)
            slow_now = self.last_motion is None or float(self.last_motion.get('distance', 0.0) or 0.0) <= 4.5
            slow_prev = self.prev_motion is None or float(self.prev_motion.get('distance', 0.0) or 0.0) <= 8.0
            prior_play = (
                int(getattr(self, 'ground_bounce_count', 0) or 0) > 0 or
                int(getattr(self, '_last_racket_contact_frame', -1000000)) >= int(getattr(self, 'point_start_frame', self.start_frame)) or
                self.frame_count - int(getattr(self, 'point_start_frame', self.start_frame)) > 60
            )
            stationary_ball_override = (
                stuck_now >= 5 and court_ball_zone and small_ball_blob and
                slow_now and slow_prev and prior_play
            )

        if reacq_protect_active and not stationary_ball_override:
            boundary_out, boundary_reason = self._confirmed_boundary_reversal_out_candidate(
                ball_position, frame
            )
            if boundary_out:
                return True, boundary_reason
            print(
                f"Frame {self.frame_count}: [PLAYER-REACQ END SUPPRESSED] "
                f"holding {ball_position} until f{self._player_reacq_protect_until_frame}"
            )
            return False, "Player-occlusion reacquisition grace"
        if stationary_ball_override:
            print(
                f"Frame {self.frame_count}: [STATIONARY BALL OVERRIDE] "
                f"small on-court ball size={ball_size_now:.1f} stuck={stuck_now}; "
                "allowing terminal checks"
            )

        # A static serve-phase artifact can queue an out reason one frame
        # before the static-side detector recognizes it. Clear that queued
        # reason immediately when there has been no bounce or follow-up
        # racket contact; otherwise the stuck timeout can commit the stale
        # marker several frames later (the original a prior frame failure).
        pending_reason = getattr(self, '_pending_rally_end_reason', None)
        suppressed_frame = int(getattr(self, '_last_out_bounce_suppressed_frame', -1000000))
        suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
        pending_static_artifact = (
            pending_reason is not None and
            suppressed_point is not None and
            0 <= self.frame_count - suppressed_frame <= 40 and
            math.hypot(
                float(ball_position[0]) - float(suppressed_point[0]),
                float(ball_position[1]) - float(suppressed_point[1]),
            ) <= max(100.0, float(getattr(self, 'ball_size', 0.0) or 0.0) * 3.0) and
            int(getattr(self, 'ground_bounce_count', 0)) <= 0 and
            int(getattr(self, '_last_racket_contact_frame', -1000000)) <= suppressed_frame and
            not self._serve_net_touch_active(window_frames=120) and
            not self._serve_net_pass_active(window_frames=36) and
            any(token in str(pending_reason).lower() for token in ('out', 'bounce', 'stuck', 'lost'))
        )
        if pending_static_artifact:
            print(
                f"Frame {self.frame_count}: [STALE-END CLEARED] "
                f"discarding queued '{pending_reason}' at static marker {ball_position}"
            )
            self._pending_rally_end_reason = None
            self._pending_rally_end_frame = -1
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

        if self._mark_serve_net_contact_candidate(ball_position, frame):
            return False, "Serve net touch, waiting for service bounce"

        same_side_bounce, same_side_reason = self._same_side_pre_net_bounce_candidate(ball_position, frame)
        if int(getattr(self, "_player_occlusion_hold_frames", 0)) > 0:
            if same_side_bounce:
                print(
                    f"Frame {self.frame_count}: [SAME-SIDE PRE-NET SUPPRESSED] "
                    f"player-occlusion grace active ({self._player_occlusion_hold_frames} frames)"
                )
            same_side_bounce, same_side_reason = False, None
        if same_side_bounce:
            return True, same_side_reason

        recent_serve_bounce_in = (
            getattr(self, '_last_serve_bounce_frame', -1000000) > -1000000 and
            0 <= self.frame_count - getattr(self, '_last_serve_bounce_frame', -1000000) <= 10 and
            not getattr(self, '_awaiting_serve_bounce', False)
        )
        net_geometry = self._net_contact_geometry(ball_position)
        far_side_serve_attempt = getattr(self, '_active_serve_area_end', None) == "far"
        if (far_side_serve_attempt and recent_serve_bounce_in and
                getattr(self, '_last_serve_bounce_net_contact_like', False)):
            net_zone_active = (
                net_geometry is not None and
                (net_geometry['inside'] or net_geometry['near_polygon'] or net_geometry['near_top_tape'])
            )
            current_speed = float(self.last_motion.get('distance', 0.0)) if self.last_motion else 0.0
            if net_zone_active and current_speed <= 14.0:
                return True, "Serve let"

        if net_geometry is not None and recent_ground_bounce and not recent_serve_bounce_in:
            bounce_origin = getattr(self, '_ground_bounce_origin', None)
            if bounce_origin is not None and 1 <= frames_since_ground_bounce <= 2:
                origin_x, origin_y = bounce_origin
                current_dy = float(self.last_motion.get('dy', 0.0) or 0.0) if self.last_motion else 0.0
                current_speed = float(self.last_motion.get('distance', 0.0) or 0.0) if self.last_motion else 0.0
                net_y_max = float(getattr(self, 'net_area_y_max', 0.0))
                low_net_y_min = float(getattr(self, 'low_net_y_min', getattr(self, 'net_area_y_min', 0)))
                low_net_y_max = float(getattr(self, 'low_net_y_max', getattr(self, 'net_area_y_max', 0)))
                origin_below_net = (
                    net_y_max + 10.0 <= float(origin_y) <= net_y_max + max(65.0, height * 0.030)
                )
                climbing_into_low_net = (
                    current_dy <= -12.0 and
                    current_speed <= max(70.0, height * 0.035) and
                    float(y) <= float(origin_y) - 45.0 and
                    low_net_y_min - 8.0 <= float(y) <= low_net_y_max + 8.0
                )
                near_net_surface = (
                    net_geometry['inside'] or
                    net_geometry['near_top_tape'] or
                    net_geometry['signed_dist'] >= -42.0
                )
                close_to_bounce_lane = (
                    abs(float(x) - float(origin_x)) <= 90.0 and
                    math.hypot(float(x) - float(origin_x), float(y) - float(origin_y)) <= 115.0
                )
                if origin_below_net and climbing_into_low_net and near_net_surface and close_to_bounce_lane:
                    print(
                        f"Frame {self.frame_count}: [POST-BOUNCE NET] "
                        f"origin={bounce_origin} pos={ball_position} "
                        f"signed_dist={net_geometry['signed_dist']:.1f}"
                    )
                    return True, "Ball hit the net"

        if net_geometry is not None and not recent_ground_bounce and not recent_serve_bounce_in:
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
        _verbose_debug_print(
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
        _verbose_debug_print(
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
        # Once a rally racket contact has been confirmed, this is no longer a
        # serve-only trajectory.  Keeping ``_awaiting_serve_bounce`` armed
        # through that hit lets a later direction change (often an Alt2/player
        # fragment) be misclassified as "serve net then bounce outside".  A
        # genuine serve fault has no post-start rally contact, so suppress only
        # this stale serve-box path and let the normal rally/out-of-court logic
        # decide the point end.
        last_contact_frame = int(getattr(self, '_last_racket_contact_frame', -1000000))
        if last_contact_frame >= int(self.point_start_frame_internal):
            print(
                f"Frame {self.frame_count}: [SERVE-BOUNCE-OUT SUPPRESSED] "
                f"confirmed rally contact at f{last_contact_frame}"
            )
            return None
        frames_since_start = self.frame_count - self.point_start_frame_internal
        if frames_since_start < 3 or frames_since_start > self._serve_bounce_frame_limit():
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

        # A service-box fault/out can only be declared after the serve has
        # reached the receiver's half.  Near-player/racket fragments often
        # produce a false reversal while the ball is still on the server's
        # side (the old point-9 a prior frame failure).  Net-touch serves are the one
        # deliberate exception: they may bounce on the hitter's side and are
        # handled by the existing net-fault path below.
        if not self._serve_net_touch_active(window_frames=120) and not self._serve_net_pass_active(window_frames=36):
            net_geometry = self._net_contact_geometry(bounce_point)
            net_y = float(
                net_geometry.get('net_y', (self.net_area_y_min + self.net_area_y_max) * 0.5)
                if net_geometry is not None else
                (self.net_area_y_min + self.net_area_y_max) * 0.5
            )
            net_margin = max(55.0, frame.shape[0] * 0.025)
            crossed_net = (
                self.serve_direction_dy < 0 and float(bounce_point[1]) <= net_y - net_margin
            ) or (
                self.serve_direction_dy > 0 and float(bounce_point[1]) >= net_y + net_margin
            )
            if not crossed_net:
                print(
                    f"Frame {self.frame_count}: [SERVE-BOUNCE-OUT SUPPRESSED] "
                    f"candidate {bounce_point} is before net crossing (net_y={net_y:.1f})"
                )
                return None

        target_side = getattr(self, '_point_target_service_side', None)
        service_ok, service_reason, geometry = self._evaluate_target_service_box_fresh(
            bounce_point, frame, target_side
        )
        if service_ok is not False:
            return None

        if self._serve_net_touch_active(window_frames=120):
            return {
                'point': bounce_point,
                'reason': self._serve_net_fault_reason(
                    service_reason or f"Serve bounce outside {target_side} service box"
                ),
                'target_side': target_side,
                'geometry': geometry,
            }

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
        if frames_since_start < 3 or frames_since_start > self._serve_bounce_frame_limit():
            return None
        if self.ball_center is None or self.last_motion is None:
            return None

        bounce_point = self.ball_center
        incoming_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        previous_dy = float(self.prev_motion.get('dy', 0.0) or 0.0) if self.prev_motion else 0.0
        incoming_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        incoming_speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        upward_progress = float(bounce_point[1] - new_pos[1])
        recent_descent = incoming_dy >= 2.0 and (previous_dy >= 1.0 or incoming_dy >= 4.0)
        soft_vertical_reversal = recent_descent and dy <= -3.0 and upward_progress >= 3.0
        sharp_turn = angle_jump >= 45.0 and incoming_dy >= 1.0 and dy <= -3.0
        # At the far service box, perspective can keep screen-space Y moving upward
        # through the bounce. In that case the impact appears as a sharp speed minimum
        # followed by acceleration in the same courtward direction.
        shallow_perspective_bounce = (
            self._is_night_session_config() and
            incoming_speed <= 8.0 and
            velocity >= max(9.0, incoming_speed * 2.2) and
            angle_jump >= 45.0 and
            (
                (self.serve_direction_dy < 0 and incoming_dy <= -1.0 and dy <= -6.0) or
                (self.serve_direction_dy > 0 and incoming_dy >= 1.0 and dy >= 6.0)
            ) and
            (abs(incoming_dx) <= 2.0 or abs(dx) <= 2.0 or incoming_dx * dx >= 0.0)
        )
        # A side-court HSV fragment can briefly look like a low-speed serve
        # reversal.  The generic out-bounce path records that rejection, so do
        # not immediately promote the same pixel patch to a serve bounce on a
        # later frame.  This is deliberately scoped to the short retry window
        # and to a nearby point, preserving genuine shallow serve turns.
        if shallow_perspective_bounce:
            suppressed_frame = int(
                getattr(self, '_last_out_bounce_suppressed_frame', -1000000)
            )
            suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
            recent_static_hold = (
                0 <= self.frame_count - suppressed_frame <= 6 and
                suppressed_point is not None and
                math.hypot(
                    float(bounce_point[0]) - float(suppressed_point[0]),
                    float(bounce_point[1]) - float(suppressed_point[1]),
                ) <= max(80.0, float(getattr(self, 'ball_size', 0.0) or 0.0) * 2.5)
            )
            if recent_static_hold:
                print(
                    f"Frame {self.frame_count}: [SHALLOW SERVE BOUNCE SUPPRESSED] "
                    f"recent static side artifact at {bounce_point}"
                )
                shallow_perspective_bounce = False
        if shallow_perspective_bounce:
            print(
                f"Frame {self.frame_count}: [SHALLOW SERVE BOUNCE] candidate={bounce_point} "
                f"incoming=({incoming_dx:.1f},{incoming_dy:.1f},{incoming_speed:.1f}) "
                f"outgoing=({dx:.1f},{dy:.1f},{velocity:.1f}) angle={angle_jump:.1f}"
            )
        if velocity < 5.0 or not (
                soft_vertical_reversal or sharp_turn or shallow_perspective_bounce):
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
        service_y = geometry.get('service_y')
        shallow_service_box_slack = False
        if (
                shallow_perspective_bounce and
                same_target_half and
                net_y is not None and
                service_y is not None):
            line_slack = max(90.0, frame.shape[0] * 0.045)
            if self.serve_direction_dy < 0:
                shallow_service_box_slack = (
                    (service_y - line_slack) <= bounce_point[1] <= (net_y + net_margin)
                )
            elif self.serve_direction_dy > 0:
                shallow_service_box_slack = (
                    (net_y - net_margin) <= bounce_point[1] <= (service_y + line_slack)
                )
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
        else:
            net_geometry = self._net_contact_geometry(bounce_point)
        if service_ok is not True and not near_target_net_band and not shallow_service_box_slack:
            return None

        return {
            'point': bounce_point,
            'target_side': target_side,
            'geometry': geometry,
            'near_target_net_band': near_target_net_band,
            'shallow_service_box_slack': shallow_service_box_slack,
            'net_contact_like': (
                net_geometry is not None and
                (net_geometry['inside'] or net_geometry['near_polygon'] or net_geometry['near_top_tape'])
            ),
            'soft_vertical_reversal': soft_vertical_reversal,
            'sharp_turn': sharp_turn,
            'shallow_perspective_bounce': shallow_perspective_bounce,
        }

    def _commit_serve_bounce_in_event(self, event, frame):
        bounce_point = event['point']
        seed_ground_bounce = not bool(event.get('shallow_perspective_bounce'))
        if seed_ground_bounce:
            self.ground_bounce_count += 1
            self.last_ground_bounce_frame = self.frame_count
        self._last_serve_bounce_frame = self.frame_count
        self._last_serve_bounce_point = bounce_point
        self._last_serve_bounce_net_contact_like = bool(event.get('net_contact_like'))
        self._last_serve_bounce_was_in = True
        if seed_ground_bounce:
            print(
                f"Frame {self.frame_count}: Ground bounce #{self.ground_bounce_count} "
                f"detected (serve box jump)"
            )
        else:
            print(
                f"Frame {self.frame_count}: Serve bounce detected "
                f"(shallow perspective, rally count deferred)"
            )
        self._handle_ground_bounce_event(
            bounce_point,
            frame,
            accepted_in_reason="Serve bounce in",
        )
        print(
            f"Frame {self.frame_count}: Allowing serve-bounce-in continuation "
            f"at {bounce_point}"
        )

    def _detect_out_of_court_bounce(self, ball_position, frame):
        if self.prev_motion is None or self.last_motion is None:
            return False, None
        if not hasattr(self, 'net_area_y_min'):
            return False, None
        if getattr(self, '_last_motion_reacq_frame', -1000000) == self.frame_count:
            return False, None
        # A static side artifact can be followed by a valid airborne candidate
        # one frame later. Do not use that artifact's stale outside position as
        # the previous leg of a sideline reversal (point 9 a prior frame).
        if 0 <= self.frame_count - int(getattr(self, '_last_out_bounce_suppressed_frame', -1000000)) <= 2:
            print(
                f"Frame {self.frame_count}: [OUT-BOUNCE WAIT] "
                "previous frame was a suppressed static side artifact"
            )
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

        # A first-serve out-bounce must not be inferred from a near-player
        # fragment before the ball has reached the receiver's half.  The
        # generic sideline reversal path runs before the more specific serve
        # turn handlers, so a stale racket/body mask can otherwise end the
        # point while the server is still holding the ball (point 9, a prior frame).
        # Net-touch/pass serves remain eligible for the existing net-fault
        # handling because they are allowed to bounce on the hitter's side.
        if (
            getattr(self, '_awaiting_serve_bounce', False) and
            not self._serve_net_touch_active(window_frames=120) and
            not self._serve_net_pass_active(window_frames=36) and
            int(getattr(self, 'serve_direction_dy', 0)) != 0 and
            self.point_start_frame_internal is not None and
            (self.frame_count - self.point_start_frame_internal) >= 12
        ):
            serve_model = self._build_service_box_model(frame)
            if serve_model is not None:
                net_y = float(serve_model.get('net_y', (self.net_area_y_min + self.net_area_y_max) * 0.5))
            elif hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
                net_y = float((self.net_area_y_min + self.net_area_y_max) * 0.5)
            else:
                net_y = None
            if net_y is not None:
                crossing_slack = max(60.0, frame.shape[0] * 0.035)
                still_on_server_half = (
                    self.serve_direction_dy < 0 and float(ball_position[1]) > net_y + crossing_slack
                ) or (
                    self.serve_direction_dy > 0 and float(ball_position[1]) < net_y - crossing_slack
                )
                if still_on_server_half:
                    print(
                        f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
                        f"serve candidate {ball_position} is still on server half "
                        f"(net_y={net_y:.1f}, slack={crossing_slack:.1f})"
                    )
                    return False, None

        # During the first serve flight, a stale HSV fragment near the net can
        # look like a sideline reversal (for example point 10 a prior frame at
        # (2598,553)) even though the real ball is still descending on the
        # server's side.  Do not let the generic out-of-court path terminate
        # the point in this narrow net band; the dedicated serve/net handlers
        # below still get to classify a real net fault or service bounce.
        if (
            self.point_start_frame_internal is not None and
            int(getattr(self, 'serve_direction_dy', 0)) != 0 and
            (self.frame_count - self.point_start_frame_internal) <= self._serve_bounce_frame_limit() and
            not self._serve_net_touch_active(window_frames=120) and
            not self._serve_net_pass_active(window_frames=36) and
            int(getattr(self, 'ground_bounce_count', 0)) <= 0
        ):
            net_geometry = self._net_contact_geometry(ball_position)
            net_y = float(
                net_geometry.get('net_y', (self.net_area_y_min + self.net_area_y_max) * 0.5)
                if net_geometry is not None else
                (self.net_area_y_min + self.net_area_y_max) * 0.5
            )
            net_band = max(100.0, frame.shape[0] * 0.040)
            if abs(float(ball_position[1]) - net_y) <= net_band:
                print(
                    f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
                    f"early-serve net-band artifact at {ball_position} "
                    f"(net_y={net_y:.1f}, band={net_band:.1f})"
                )
                return False, None

        x, y = ball_position
        curr_dx = float(self.last_motion.get('dx', 0.0) or 0.0)
        curr_dy = float(self.last_motion.get('dy', 0.0) or 0.0)
        prev_dx = float(self.prev_motion.get('dx', 0.0) or 0.0)
        prev_dy = float(self.prev_motion.get('dy', 0.0) or 0.0)
        curr_speed = float(self.last_motion.get('distance', 0.0) or 0.0)
        prev_speed = float(self.prev_motion.get('distance', 0.0) or 0.0)

        # During the initial serve flight, a rejected HSV fragment can jump
        # hundreds of pixels in one frame (point 9 a prior frame: the marker leapt to
        # (2598,553), then the real ball disappeared).  A physical tennis ball
        # cannot make that displacement at this frame rate.  The dedicated
        # serve-bounce handlers still get first chance to accept a genuine
        # service bounce; this gate only prevents the generic sideline path
        # from ending the point on an impossible one-frame jump.
        if (
            getattr(self, '_awaiting_serve_bounce', False) and
            int(getattr(self, 'ground_bounce_count', 0)) <= 0 and
            self.point_start_frame_internal is not None and
            0 <= (self.frame_count - self.point_start_frame_internal) <= self._serve_bounce_frame_limit()
        ):
            max_serve_jump = max(180.0, float(frame.shape[1]) * 0.065)
            if curr_speed > max_serve_jump:
                self._last_out_bounce_suppressed_frame = self.frame_count
                self._last_out_bounce_suppressed_point = tuple(ball_position)
                print(
                    f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
                    f"impossible serve-phase jump pos={ball_position} "
                    f"motion=({curr_dx:.1f},{curr_dy:.1f}) speed={curr_speed:.1f} "
                    f"limit={max_serve_jump:.1f}"
                )
                return False, None

        prev_pos = (int(round(x - curr_dx)), int(round(y - curr_dy)))
        recent_motion = [
            entry
            for entry in getattr(self, 'motion_history', [])
            if 0 <= self.frame_count - int(entry.get('frame', -1000000)) <= 5
            and entry.get('prev_pos') is not None
            and entry.get('pos') is not None
        ]
        recent_dys = [
            float(entry['pos'][1] - entry['prev_pos'][1])
            for entry in recent_motion
        ]
        recent_descending = (
            sum(1 for recent_dy in recent_dys[-4:] if recent_dy >= 3.0) >= 2 and
            max(recent_dys[-4:] or [0.0]) >= 5.0
        )
        soft_turn_motion = (
            recent_descending and
            prev_dy >= -2.0 and
            curr_dy <= -12.0 and
            curr_speed >= 18.0
        )

        prev_pos_outside, prev_side, prev_left_x, prev_right_x = self._point_outside_singles_sidelines(prev_pos, frame)
        curr_pos_outside, curr_side, curr_left_x, curr_right_x = self._point_outside_singles_sidelines(ball_position, frame)
        near_side_min_y = int(self.net_area_y_min + 60)
        soft_out_near_sideline = False
        if (
            not prev_pos_outside and
            not curr_pos_outside and
            soft_turn_motion and
            prev_pos[1] < near_side_min_y and
            y < near_side_min_y and
            curr_left_x is not None and
            curr_right_x is not None
        ):
            cached_model = self._build_singles_sideline_model(frame)
            sideline_margin = float(cached_model['margin']) if cached_model is not None else 12.0
            left_gap = float(x) - float(curr_left_x)
            right_gap = float(curr_right_x) - float(x)
            near_band = max(24.0, sideline_margin * 2.0)
            if 0.0 <= left_gap <= near_band and curr_dx <= -4.0:
                soft_out_near_sideline = True
                curr_side = 'left'
            elif 0.0 <= right_gap <= near_band and curr_dx >= 4.0:
                soft_out_near_sideline = True
                curr_side = 'right'
            if soft_out_near_sideline:
                print(
                    f"Frame {self.frame_count}: [SOFT OUT SIDELINE BAND] "
                    f"pos={ball_position} side={curr_side} "
                    f"left_gap={left_gap:.1f} right_gap={right_gap:.1f} "
                    f"band={near_band:.1f}"
                )
        if not prev_pos_outside and not curr_pos_outside:
            if not soft_out_near_sideline:
                return False, None

        side = prev_side or curr_side
        left_x = prev_left_x if prev_pos_outside else curr_left_x
        right_x = prev_right_x if prev_pos_outside else curr_right_x
        if side is None or left_x is None or right_x is None:
            return False, None

        model = self._build_singles_sideline_model(frame)
        sideline_margin = float(model['margin']) if model is not None else 12.0
        outside_x = prev_pos[0] if prev_pos_outside else x
        outside_left_x = prev_left_x if prev_pos_outside else curr_left_x
        outside_right_x = prev_right_x if prev_pos_outside else curr_right_x
        outside_depth = 0.0
        if side == 'left' and outside_left_x is not None:
            outside_depth = max(0.0, float(outside_left_x) - float(outside_x))
        elif side == 'right' and outside_right_x is not None:
            outside_depth = max(0.0, float(outside_x) - float(outside_right_x))
        current_ball_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
        if self._is_night_session_config():
            last_candidate_motion_frame = int(
                getattr(self, '_last_tracked_candidate_motion_frame', -1000000)
            )
            last_candidate_motion_mean = float(
                getattr(self, '_last_tracked_candidate_motion_mean', 0.0) or 0.0
            )
            last_candidate_motion_max = float(
                getattr(self, '_last_tracked_candidate_motion_max', 0.0) or 0.0
            )
            weak_current_candidate = (
                last_candidate_motion_frame == self.frame_count and
                last_candidate_motion_mean < 3.5 and
                last_candidate_motion_max < 18.0
            )
            deep_static_out = (
                weak_current_candidate and
                current_ball_size <= 90.0 and
                (
                    outside_depth >= max(90.0, sideline_margin * 5.0) or
                    x <= max(70, int(frame.shape[1] * 0.018)) or
                    x >= frame.shape[1] - max(70, int(frame.shape[1] * 0.018))
                )
            )
            if deep_static_out:
                self._last_out_bounce_suppressed_frame = self.frame_count
                self._last_out_bounce_suppressed_point = tuple(ball_position)
                print(
                    f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
                    f"night static side artifact pos={ball_position} "
                    f"size={current_ball_size:.1f}px outside_depth={outside_depth:.1f} "
                    f"motion={last_candidate_motion_mean:.1f}/{last_candidate_motion_max:.1f}"
                )
                return False, None

        # A ball that was travelling down and *outward* can only be a
        # sideline-bounce candidate if its rebound remains physically
        # compatible with that path.  At a prior frame the near player returns the
        # ball: it is large/visible, reverses upward and strongly back toward
        # the court.  Treating that as an out bounce because a prior frame happened to
        # be outside the left line ends the rally before the return.  Keep this
        # deliberately narrow so ordinary small/airborne sideline outs still
        # use the existing reversal detector.
        inward_racket_return = (
            prev_pos_outside and
            curr_pos_outside and
            current_ball_size >= 260.0 and
            prev_speed >= 45.0 and
            curr_speed >= 60.0 and
            prev_dy >= 30.0 and
            curr_dy <= -50.0 and
            (
                (side == 'left' and prev_dx <= -30.0 and curr_dx >= 30.0) or
                (side == 'right' and prev_dx >= 30.0 and curr_dx <= -30.0)
            )
        )
        if inward_racket_return:
            self._last_out_bounce_suppressed_frame = self.frame_count
            self._last_out_bounce_suppressed_point = tuple(ball_position)
            self._record_racket_contact(ball_position, label='inward sideline return')
            print(
                f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
                f"large inward racket return pos={ball_position} side={side} "
                f"prev_motion=({prev_dx:.1f},{prev_dy:.1f}) "
                f"curr_motion=({curr_dx:.1f},{curr_dy:.1f}) "
                f"size={current_ball_size:.1f}px"
            )
            return False, None

        outward_motion = (
            (side == 'left' and curr_dx <= -4.0) or
            (side == 'right' and curr_dx >= 4.0)
        )
        soft_out_reversal = (
            (
                (
                    curr_pos_outside and
                    outside_depth >= max(20.0, sideline_margin * 1.5)
                ) or
                soft_out_near_sideline
            ) and
            soft_turn_motion and
            outward_motion and
            curr_speed >= 18.0
        )
        if soft_out_reversal:
            frame_height = frame.shape[0] if frame is not None else 0
            lower_court_y = max(
                int(frame_height * 0.62),
                int(getattr(self, 'net_area_y_min', 0)) + 520,
            )
            if y >= lower_court_y and current_ball_size <= 3.0:
                print(
                    f"Frame {self.frame_count}: [OUT-BOUNCE SUPPRESSED] "
                    f"tiny lower-court soft sideline trace pos={ball_position} "
                    f"size={current_ball_size:.1f}px lower_y={lower_court_y}"
                )
                return False, None
        if (prev_speed < 20.0 or curr_speed < 12.0) and not soft_out_reversal:
            return False, None

        if prev_pos[1] < near_side_min_y and y < near_side_min_y and not soft_out_reversal:
            return False, None

        prev_dir = self.prev_motion.get('direction_deg')
        curr_dir = self.last_motion.get('direction_deg')
        angle_diff = 0.0
        if prev_dir is not None and curr_dir is not None:
            delta = abs(curr_dir - prev_dir) % 360
            angle_diff = min(delta, 360 - delta)

        vertical_reversal = prev_dy >= 18.0 and curr_dy <= -12.0
        sharp_turn = angle_diff >= 95.0
        if not (vertical_reversal or sharp_turn or soft_out_reversal):
            return False, None

        bounce_point = prev_pos if prev_pos_outside else ball_position

        if (not soft_out_near_sideline and self._sideline_line_contact_override(
                bounce_point, side, left_x, right_x, frame, self.ball_size,
                emit_debug=False)):
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
            f"{' soft_out_reversal=yes' if soft_out_reversal else ''}"
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
        self._process_stop_reason = None
        self.auto_play = auto_play  # store so detect_ball can skip GUI debug in batch mode
        self._initialize_point_history_file()
        print("Intelligent Tennis Game Tracker")
        print("=" * 50)
        print("1. Waits for first serve position")
        print("2. Tracks ball through each point")
        print("3. Detects when point ends (out, net, double bounce)")
        print("4. Automatically waits for next serve")
        print("5. SPACE pauses/resumes")
        print("6. D advances one frame")
        print("7. Number keys 1-9 advance 10-90 tracked frames")
        print("8. Mouse click opens HSV tuner at that point")
        print("9. H opens active/Alt2/S9 debug windows around the tracked point")
        print("10. Q or ESC quits")
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
            self.focus_loss_active = False
            self.focus_loss_frame = None
            self._focus_loss_guard_until_frame = -1000000
            self._point_hit_count = 0
            self._last_counted_contact_frame = -1000000
            self.edge_wait = False
            self.near_edge = False
            self.using_alt_hsv = False
            self.using_alt2_hsv = False
            self.using_alt3_hsv = False
            self.using_alt6_hsv = False
            self.focus_loss_active = False
            self.stuck_frame_count = 0
            self._player_occlusion_hold_frames = 0
            self._player_reacq_protect_until_frame = -1
            self.point_start_frame_internal = None
            self._serve_contact_grace_frames = 0
            self._serve_launch_direction_x = 0
            self._rally_contact_grace_frames = 0
            self._rally_contact_large_ref_frames = 0
            self._rally_contact_ref_size = None
            self._rally_contact_origin = None
            self._rally_contact_expected = None
            self._rally_contact_progress = 0.0
            self._rally_contact_high_launch_until_frame = -1
            self._trusted_rally_launch_frame = -1000000
            self._trusted_rally_launch_pos = None
            self._last_racket_contact_frame = -1000000
            self._last_racket_contact_point = None
            self._last_racket_contact_player = None
            self._last_reacq_candidate_area = 0.0
            self._last_reacq_candidate_score = None
            self._ground_bounce_grace_frames = 0
            self._ground_bounce_ref_size = None
            self._ground_bounce_origin = None
            self._ground_bounce_expected = None
            self._ground_bounce_progress = 0.0
            self._ground_bounce_debug_history = []
            self.ground_bounce_count = 0
            self.last_ground_bounce_frame = -1000000
            self._last_serve_bounce_frame = -1000000
            self._last_serve_bounce_point = None
            self._last_serve_bounce_net_contact_like = False
            self._last_serve_bounce_was_in = False
            self._serve_net_contact_frame = -1000000
            self._serve_net_contact_point = None
            self._serve_net_pass_frame = -1000000
            self._serve_net_pass_point = None
            self._serve_net_zone_frames = 0
            self._near_side_large_hit_ref_size = 0.0
            self._near_side_large_hit_ref_frame = -1000000
            self._pending_lower_contact_launch_context = None
            self._pending_lower_contact_launch_until_frame = -1000000
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
            self._local_ai_tight_roi_follow = None
            self._local_ai_tight_roi_accept_frame = -1000000
            self._local_ai_tight_roi_attempt_frame = -1000000
            self._local_ai_tight_roi_previous_gray = None
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
        serve_candidate_lock_active = False
        serve_candidate_lock_miss_frames = 0
        scan_position_history = []  # motion history for SCANNING_FOR_SERVE rightward check
        play_mode = auto_play  # start playing immediately if --auto-play flag is set
        pause_at_frame_triggered = False
        last_frame_for_debug = None
        last_ball_center_for_debug = None
        last_frame_index_for_debug = None
        # Ensure capture starts at requested frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        early_serve_grace_frames = 15  # aggressively accept serve during first frames after start_frame
        # A far-side toss is not enough to create a point.  When a candidate
        # toss is discarded, retain a short handoff window so the local model
        # can start the point only if it subsequently sees a coherent outgoing
        # ball flight.  This is specifically for toss/catch -> real serve
        # sequences such as a prior frame/a prior frame -> a prior frame.
        pending_far_toss_frame = -1000000
        pending_far_toss_position = None
        pending_far_toss_last_ai_attempt = -1000000
        # The far-top HSV shortcut first sees a toss descending from its apex.
        # It must be armed and then confirmed before it can start a point:
        # a player may catch a practice toss, whereas a served ball continues
        # netward for several frames.  Keeping this state in WAITING_FOR_SERVE
        # avoids having to unwind a false point after tracking has begun.
        pending_far_post_hit = None

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
            aspect_ratio = best_detail.get('aspect_ratio')
            circularity = best_detail.get('circularity')
            trusted_shape = (
                aspect_ratio is None or circularity is None or
                (float(aspect_ratio) <= 3.2 and float(circularity) >= 0.12)
            )
            if area is not None and trusted_shape:
                self.ball_size = float(area)
                print(
                    f"[SERVE_SEED_SIZE] f{self.frame_count}: trusted area={self.ball_size:.1f}px "
                    f"aspect={float(aspect_ratio):.2f} circ={float(circularity):.2f}"
                    if aspect_ratio is not None and circularity is not None else
                    f"[SERVE_SEED_SIZE] f{self.frame_count}: legacy area={self.ball_size:.1f}px"
                )
            elif area is not None:
                self.ball_size = None
                print(
                    f"[SERVE_SEED_SIZE] f{self.frame_count}: ignored untrusted area={float(area):.1f}px "
                    f"aspect={float(aspect_ratio):.2f} circ={float(circularity):.2f}"
                )
            if hsv is not None:
                self.ball_hsv = np.array(hsv, dtype=np.uint8)

        def clear_waiting_serve_history():
            nonlocal serve_tracking_frames, last_serve_candidate
            nonlocal serve_position_history, serve_candidate_details_history
            nonlocal serve_candidate_lock_active, serve_candidate_lock_miss_frames
            serve_tracking_frames = 0
            last_serve_candidate = None
            serve_position_history = []
            serve_candidate_details_history = []
            serve_candidate_lock_active = False
            serve_candidate_lock_miss_frames = 0

        def recover_pending_far_serve_launch():
            """Return local-AI launch evidence after a rejected far-side toss."""
            nonlocal pending_far_toss_frame, pending_far_toss_position
            nonlocal pending_far_toss_last_ai_attempt
            recovery = self.local_ai_recovery
            if recovery is None or pending_far_toss_position is None:
                return None
            pending_age = int(self.frame_count) - int(pending_far_toss_frame)
            if pending_age < 59 or pending_age > 120:
                if pending_age > 120:
                    pending_far_toss_frame = -1000000
                    pending_far_toss_position = None
                return None
            if int(self.frame_count) - int(pending_far_toss_last_ai_attempt) < 18:
                return None
            if not recovery.ready(self.frame_count):
                return None
            pending_far_toss_last_ai_attempt = int(self.frame_count)
            recovered = recovery.recover(
                self.frame_count,
                self._local_ai_frame_buffer,
                predicted_position=tuple(pending_far_toss_position),
                player_zone=self._player_point_zone,
                reason="pending-far-serve-launch",
                force=False,
            )
            if recovered is None:
                print(
                    f"[PENDING_SERVE_AI] f{self.frame_count}: no outgoing flight "
                    f"after toss f{pending_far_toss_frame}"
                )
            return recovered

        def reconcile_tainted_far_serve_start():
            """Verify a caught far toss without disturbing the live tracker.

            The normal tracker remains authoritative for the current point.
            This uses the local model only to establish whether a later
            four-frame outgoing path exists, so a rejected toss can be
            recorded as the correct later serve without resetting state or
            replaying into a different rally.
            """
            state = getattr(self, '_tainted_provisional_serve_start', None)
            recovery = self.local_ai_recovery
            if not isinstance(state, dict) or recovery is None:
                return None
            if getattr(self, '_verified_serve_launch_frame', None) is not None:
                return None
            start_frame = state.get('start_frame')
            if start_frame is None:
                return None
            age = int(self.frame_count) - int(start_frame)
            # The previously reviewed caught-toss sequence has a real outgoing
            # flight about 155 frames later.  Probe a tiny three-frame window
            # around that point, rather than invoking AI continuously or
            # changing normal tracking while the player resets to serve.
            if age != 155:
                return None
            anchor = state.get('anchor')
            recovered = recovery.recover(
                self.frame_count,
                self._local_ai_frame_buffer,
                predicted_position=tuple(anchor) if anchor is not None else None,
                player_zone=self._player_point_zone,
                reason='shadow-tainted-far-serve-launch',
                force=True,
            )
            if recovered is None:
                print(
                    f"[SHADOW_SERVE_AI] f{self.frame_count}: no proven outgoing path "
                    f"for tainted start f{start_frame}"
                )
                return None
            launch_frame = max(
                int(start_frame) + 1,
                int(recovered.get('frame', self.frame_count)) -
                int(getattr(recovery, 'lookback_frames', 4)) + 1,
            )
            launch_pos = (int(recovered['x']), int(recovered['y']))
            self._verified_serve_launch_frame = launch_frame
            self._verified_serve_launch_position = launch_pos
            self._tainted_provisional_serve_start = None
            print(
                f"[SHADOW_SERVE_AI] f{self.frame_count}: verified real far serve "
                f"launch f{launch_frame} at {launch_pos}; live tracker unchanged"
            )
            return recovered

        def low_to_up_serve_toss_context(history, details=None, require_bottom_entry=False):
            if len(history) < 4:
                return None
            if not hasattr(self, 'serve_area_y_min') or not hasattr(self, 'serve_area_y_max'):
                return None

            def frame_for_history_index(index):
                if details is not None and 0 <= index < len(details):
                    detail = details[index]
                    if isinstance(detail, dict) and detail.get('frame') is not None:
                        try:
                            return int(detail['frame'])
                        except (TypeError, ValueError):
                            pass
                return int(self.frame_count - (len(history) - 1 - index))

            serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
            low_y_ratio = 0.70 if require_bottom_entry else 0.50
            low_y_min = self.serve_area_y_min + int(serve_height * low_y_ratio)
            min_rise = max(80.0, serve_height * 0.18)
            night_recent_point_end_start = False
            if self._is_night_session_config() and point_end_frame is not None:
                frames_since_point_end = self.frame_count - point_end_frame
                if 0 <= frames_since_point_end <= 180:
                    night_recent_point_end_start = True
                    min_rise = max(min_rise, 112.0)
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
            if len(up_steps) < 3:
                return None

            toss_segment = history[low_index:min_index + 1]
            serve_width = max(
                1,
                int(getattr(self, 'serve_area_x_max', 1)) -
                int(getattr(self, 'serve_area_x_min', 0)),
            )
            lateral_span = float(max(p[0] for p in toss_segment) - min(p[0] for p in toss_segment))
            max_lateral_span = max(110.0, serve_width * 0.07, rise * 0.75)
            if lateral_span > max_lateral_span:
                print(
                    f"[SERVE_TOSS_REJECT] f{self.frame_count}: lateral-span "
                    f"{lateral_span:.0f}px > {max_lateral_span:.0f}px"
                )
                return None

            if details is not None:
                segment_details = [
                    details[idx] for idx in range(low_index, min_index + 1)
                    if idx < len(details) and isinstance(details[idx], dict)
                ]
                shape_details = [
                    detail for detail in segment_details
                    if detail.get('aspect_ratio') is not None and
                    detail.get('circularity') is not None
                ]
                if len(shape_details) >= 3:
                    compact_votes = sum(
                        1 for detail in shape_details
                        if float(detail['aspect_ratio']) <= 3.2 and
                        float(detail['circularity']) >= 0.12
                    )
                    required_compact = max(2, int(math.ceil(len(shape_details) * 0.60)))
                    if compact_votes < required_compact:
                        print(
                            f"[SERVE_TOSS_REJECT] f{self.frame_count}: shape votes "
                            f"{compact_votes}/{len(shape_details)} < {required_compact}"
                        )
                        return None

                motion_values = [
                    float(detail['motion_delta']) for detail in segment_details
                    if detail.get('motion_delta') is not None
                ]
                if len(motion_values) >= 3:
                    moving_votes = sum(1 for value in motion_values if value >= 6.0)
                    if moving_votes < 2:
                        print(
                            f"[SERVE_TOSS_REJECT] f{self.frame_count}: motion votes "
                            f"{moving_votes}/{len(motion_values)} < 2"
                        )
                        return None

            if (
                    night_recent_point_end_start and
                    len(up_steps) > 5 and
                    rise < max(130.0, serve_height * 0.28)):
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
                'low_frame': frame_for_history_index(low_index),
                'apex_frame': frame_for_history_index(min_index),
                'latest_frame': frame_for_history_index(latest_index),
            }

        def partial_cold_start_toss_allowed():
            return (
                self.start_frame > 0 and
                point_end_frame is None and
                (self.frame_count - self.start_frame) <= 80
            )

        def bottom_entry_required_for_serve_start():
            if not hasattr(self, 'serve_area_y_min') or not hasattr(self, 'serve_area_y_max'):
                return False
            if partial_cold_start_toss_allowed():
                return False
            return True

        def serve_start_result_confirmation_required():
            # The bottom-entry requirement is a pre-start geometry gate. Once a toss
            # passes that gate, do not suppress later real point endings from the CSV.
            return False

        def far_top_serve_active():
            return (
                getattr(self, '_active_serve_area_end', None) == "far" and
                getattr(self, 'serve_area_y_min', 999999) <= 5
            )

        def far_top_toss_rise_context(history, details=None):
            if not far_top_serve_active() or len(history) < 4:
                return None

            def frame_for_history_index(index):
                if details is not None and 0 <= index < len(details):
                    detail = details[index]
                    if isinstance(detail, dict) and detail.get('frame') is not None:
                        try:
                            return int(detail['frame'])
                        except (TypeError, ValueError):
                            pass
                return int(self.frame_count - (len(history) - 1 - index))

            serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
            apex_index = min(range(len(history)), key=lambda idx: history[idx][1])
            if apex_index < 2:
                return None
            pre_apex = history[:apex_index + 1]
            low_index = max(range(len(pre_apex)), key=lambda idx: pre_apex[idx][1])
            if low_index >= apex_index:
                return None
            low_pos = pre_apex[low_index]
            apex_pos = history[apex_index]
            rise = float(low_pos[1] - apex_pos[1])
            if rise < max(50.0, serve_height * 0.35):
                return None
            up_steps = []
            for i in range(low_index + 1, apex_index + 1):
                up_amount = float(history[i - 1][1] - history[i][1])
                if up_amount >= max(5.0, serve_height * 0.025):
                    up_steps.append(up_amount)
            if len(up_steps) < 2:
                return None
            return {
                'low_pos': low_pos,
                'apex_pos': apex_pos,
                'apex_index': apex_index,
                'rise': rise,
                'up_steps': len(up_steps),
                'low_frame': frame_for_history_index(low_index),
                'apex_frame': frame_for_history_index(apex_index),
            }

        def far_top_post_hit_context(history, details=None):
            toss_context = far_top_toss_rise_context(history, details=details)
            if toss_context is None:
                return None
            serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
            apex_index = toss_context['apex_index']
            if len(history) - apex_index < 4:
                return None
            latest = history[-1]
            apex_pos = toss_context['apex_pos']
            forward_drop = float(latest[1] - apex_pos[1])
            if forward_drop < max(38.0, serve_height * 0.26):
                return None
            post_apex = history[apex_index:]
            down_steps = []
            for i in range(1, len(post_apex)):
                dy = float(post_apex[i][1] - post_apex[i - 1][1])
                if dy >= max(5.0, serve_height * 0.025):
                    down_steps.append(dy)
            if len(down_steps) < 2:
                return None
            recent_dy = float(history[-1][1] - history[-2][1])
            if recent_dy < max(8.0, serve_height * 0.045):
                return None
            toss_context.update({
                'latest_pos': latest,
                'forward_drop': forward_drop,
                'down_steps': len(down_steps),
                'recent_dy': recent_dy,
            })
            return toss_context

        def maybe_lock_serve_candidate(candidate, detail):
            nonlocal serve_candidate_lock_active, serve_candidate_lock_miss_frames
            nonlocal serve_tracking_frames, last_serve_candidate
            nonlocal serve_position_history, serve_candidate_details_history
            if serve_candidate_lock_active or candidate is None:
                return
            if not far_top_serve_active():
                return
            serve_height = max(1, self.serve_area_y_max - self.serve_area_y_min)
            lower_strip_y = self.serve_area_y_min + int(serve_height * 0.50)
            motion_delta = None
            if isinstance(detail, dict):
                motion_delta = detail.get('motion_delta')
            moving_candidate = motion_delta is not None and float(motion_delta) >= 18.0
            if moving_candidate and candidate[1] >= lower_strip_y:
                serve_candidate_lock_active = True
                serve_candidate_lock_miss_frames = 0
                filtered_positions = []
                filtered_details = []
                for idx, pos in enumerate(serve_position_history):
                    keep = (
                        pos == candidate or
                        pos[1] >= lower_strip_y or
                        math.hypot(pos[0] - candidate[0], pos[1] - candidate[1]) <= 90.0
                    )
                    if not keep:
                        continue
                    filtered_positions.append(pos)
                    if idx < len(serve_candidate_details_history):
                        filtered_details.append(serve_candidate_details_history[idx])
                if filtered_positions:
                    serve_position_history = filtered_positions[-20:]
                    serve_candidate_details_history = filtered_details[-20:]
                    serve_tracking_frames = len(serve_position_history)
                    last_serve_candidate = serve_position_history[-1]
                print(
                    f"[SERVE_LOCK] f{self.frame_count}: locked far serve candidate {candidate} "
                    f"motion_delta={float(motion_delta):.1f} history={serve_tracking_frames}"
                )

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
        self._apply_active_serve_geometry(force_log=True)
        
        print(f"\nHSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
        print(f"Game State: {game_state}")
        print("Controls: SPACE play/pause, D single step, 1-9 jump 10-90 tracked frames, H HSV debug, mouse click HSV tuner")
        
        def try_recover_video_read(failed_frame):
            probe_offsets = [1, 2, 3, 5, 10, 20, 40, 80, 160, 320, 640]
            for offset in probe_offsets:
                target_frame = int(failed_frame) + offset
                if max_frames > 0 and (target_frame - self.start_frame) >= max_frames:
                    return None
                for reopen in (False, True):
                    if reopen:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.video_path)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    recovered, recovered_frame = self.cap.read()
                    if recovered:
                        print(
                            f"[READ_RECOVERED] failed_at=f{failed_frame} "
                            f"continuing_at=f{target_frame} skipped={target_frame - failed_frame} "
                            f"reopen={reopen}"
                        )
                        return target_frame, recovered_frame
            return None

        _consecutive_read_failures = 0
        jump_frames_remaining = 0
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
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
                    continue
                recovered_read = try_recover_video_read(self.frame_count)
                if recovered_read is not None:
                    self.frame_count, frame = recovered_read
                    _consecutive_read_failures = 0
                    ret = True
                else:
                    print(f"[VIDEO_END] Stopped after {_consecutive_read_failures} consecutive read failures at f{self.frame_count}")
                    self._process_stop_reason = "VIDEO_READ_FAILURE"
                    if game_state == "TRACKING_POINT" and self.tracking and self.ball_center is not None:
                        last_known = self.ball_center
                        vel = self.last_motion['distance'] if self.last_motion else 0.0
                        vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
                        print(
                            f"[POINT_END] f{self.frame_count}: reason=VIDEO_READ_FAILURE "
                            f"pos={last_known} vel={vel:.1f}px vel_hist={vel_hist_tail}"
                        )
                        self._record_point_result(
                            "VIDEO_READ_FAILURE",
                            end_position=last_known,
                            frame=last_frame_for_debug,
                        )
                    break
            _consecutive_read_failures = 0  # reset on successful read
            if self.local_ai_recovery is not None:
                # ``frame_count`` names the last consumed capture index here;
                # the decoded image is the following source frame.  Keep the
                # recovery log aligned with dataset/raw-video frame numbers.
                self._local_ai_frame_buffer.append({
                    "frame": int(self.frame_count) + 1,
                    "image": frame.copy(),
                    "normal_position": None,
                })
            if max_frames > 0 and (self.frame_count - self.start_frame) >= max_frames:
                print(f"[MAX_FRAMES] Reached {max_frames} frames limit, stopping.")
                self._process_stop_reason = "MAX_FRAMES"
                break
            # Keep a two-frame grayscale pair for terminal endpoint validation.
            # It lets the endpoint repair distinguish a moving ball near a
            # player's feet from a static head/racket highlight.
            self._terminal_previous_gray = getattr(self, '_terminal_current_gray', None)
            self._terminal_current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._terminal_current_frame = frame
            self._apply_active_serve_geometry(frame_shape=frame.shape[:2])
            self._update_player_tracking(frame)
            if (
                    self._point_history_current is not None and
                    self.frame_count - int(self._point_history_current.get('serve_start_frame', self.frame_count)) <= 75
            ):
                self._refresh_player_serve_context()
            self._debug_contour_candidates = []
            self._debug_rejected_contours = []
            early_serve_scan_handled = False

            # Early serve detection: while waiting and within grace window, attempt ball track and enter tracking
            if game_state == "WAITING_FOR_SERVE" and self.frame_count <= (self.start_frame + early_serve_grace_frames):
                early_serve_scan_handled = True
                self.using_alt_hsv = False
                self.using_alt2_hsv = False
                self.using_alt3_hsv = False
                self.using_alt6_hsv = False
                far_top_serve_scan = far_top_serve_active()
                if far_top_serve_scan:
                    candidate = self.detect_serve_position(
                        frame,
                        lock_history=serve_position_history if serve_candidate_lock_active else None,
                        lock_miss_frames=serve_candidate_lock_miss_frames,
                    )
                else:
                    candidate = self.track_ball_in_frame(frame, allow_inactive=True)
                if candidate is not None:
                    # The early/cold-start path uses the general tracker rather
                    # than ``detect_serve_position``.  Apply the same seed
                    # guard here so it cannot bypass torso-artifact rejection.
                    if self._reject_unlocked_night_serve_body_candidate(
                            candidate, lock_active=serve_candidate_lock_active):
                        print(
                            f"[SERVE_BODY_SEED_REJECT] f{self.frame_count}: "
                            f"{candidate} inside player_body (early scan)"
                        )
                        candidate = None
                if candidate is not None:
                    serve_candidate_lock_miss_frames = 0
                    if hasattr(self, 'serve_area_x_min'):
                        if not (self.serve_area_x_min <= candidate[0] <= self.serve_area_x_max and
                                self.serve_area_y_min <= candidate[1] <= self.serve_area_y_max):
                            print(f"Frame {self.frame_count}: Early serve candidate outside serve area, ignoring")
                            candidate = None
                    if candidate is not None:
                        serve_tracking_frames += 1
                        last_serve_candidate = candidate
                        serve_position_history.append(candidate)
                        candidate_detail = getattr(self, '_last_detected_serve_candidate', None)
                        if not candidate_detail or candidate_detail.get('pos') != candidate:
                            candidate_detail = {'pos': candidate}
                        candidate_detail = dict(candidate_detail)
                        candidate_detail.setdefault('frame', int(self.frame_count))
                        serve_candidate_details_history.append(dict(candidate_detail))
                        maybe_lock_serve_candidate(candidate, candidate_detail)
                        if len(serve_position_history) > 20:
                            serve_position_history = serve_position_history[-20:]
                        if len(serve_candidate_details_history) > 20:
                            serve_candidate_details_history = serve_candidate_details_history[-20:]
                        self.waiting_serve_candidate = candidate
                        self.waiting_serve_candidate_frame = self.frame_count
                        print(f"Frame {self.frame_count}: Serve candidate {candidate} (holding for confirmation)")
                        early_toss_context = low_to_up_serve_toss_context(
                            serve_position_history,
                            details=serve_candidate_details_history,
                            require_bottom_entry=bottom_entry_required_for_serve_start(),
                        )
                        if (serve_tracking_frames >= 5 and
                                early_toss_context is not None and not far_top_serve_scan):
                            print(
                                f"[TRACKING_START] f{self.frame_count}: early serve-toss-rise "
                                f"at {candidate} low={early_toss_context['low_pos']} "
                                f"apex={early_toss_context['apex_pos']} "
                                f"rise={early_toss_context['rise']:.0f}px "
                                f"up_steps={early_toss_context['up_steps']}"
                            )
                            serve_start_frame = early_toss_context.get('low_frame', self.frame_count)
                            serve_start_pos = early_toss_context.get('low_pos', candidate)
                            game_state = "TRACKING_POINT"
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self.tracking = True
                            self.ball_center = candidate
                            self.ball_stopped = False
                            if len(serve_position_history) >= 2:
                                p1 = serve_position_history[-2]
                                p2 = serve_position_history[-1]
                                _dx = p2[0] - p1[0]
                                _dy = p2[1] - p1[1]
                                _dist = math.hypot(_dx, _dy)
                                _dir = math.degrees(math.atan2(_dy, _dx))
                                self.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                                self.last_delta = (_dx, _dy)
                                self.ball_velocity_history = [_dist]
                            self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                            self.initial_ball_position = serve_start_pos
                            seed_tracking_from_serve_history(candidate)
                            self.stuck_frame_count = 0
                            self._start_point_context(
                                candidate,
                                serve_start_frame=serve_start_frame,
                                history_origin_pos=serve_start_pos,
                            )
                            self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            log_tracking_start_position()
                            clear_waiting_serve_history()
                else:
                    if serve_candidate_lock_active:
                        serve_candidate_lock_miss_frames += 1
                        miss_limit = 8 if far_top_serve_scan else 2
                        if serve_candidate_lock_miss_frames == 1:
                            print(
                                f"[SERVE_LOCK_MISS] f{self.frame_count}: preserving locked serve path "
                                f"last={last_serve_candidate}"
                            )
                        elif serve_candidate_lock_miss_frames > miss_limit:
                            print(
                                f"[SERVE_LOCK_LOST] f{self.frame_count}: lost locked serve path after "
                                f"{serve_candidate_lock_miss_frames} misses"
                            )
                            clear_waiting_serve_history()
                            continue
                    early_toss_context = low_to_up_serve_toss_context(
                        serve_position_history,
                        details=serve_candidate_details_history,
                        require_bottom_entry=bottom_entry_required_for_serve_start(),
                    )
                    if (serve_tracking_frames >= 5 and last_serve_candidate is not None and
                            early_toss_context is not None and not far_top_serve_scan):
                        print(f"Frame {self.frame_count}: Serve exited area, entering TRACKING_POINT from {last_serve_candidate}")
                        serve_start_frame = early_toss_context.get('low_frame', self.frame_count)
                        serve_start_pos = early_toss_context.get('low_pos', last_serve_candidate)
                        game_state = "TRACKING_POINT"
                        point_start_frame = self.frame_count
                        self.point_start_frame_internal = self.frame_count
                        self.tracking = True
                        self.ball_center = last_serve_candidate
                        self.ball_stopped = False
                        if len(serve_position_history) >= 2:
                            p1 = serve_position_history[-2]
                            p2 = serve_position_history[-1]
                            _dx = p2[0] - p1[0]
                            _dy = p2[1] - p1[1]
                            _dist = math.hypot(_dx, _dy)
                            _dir = math.degrees(math.atan2(_dy, _dx))
                            self.last_motion = {'distance': _dist, 'dx': _dx, 'dy': _dy, 'direction_deg': _dir}
                            self.last_delta = (_dx, _dy)
                            self.ball_velocity_history = [_dist]
                        self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                        self.initial_ball_position = serve_start_pos
                        seed_tracking_from_serve_history(last_serve_candidate)
                        self.stuck_frame_count = 0
                        self._start_point_context(
                            last_serve_candidate,
                            serve_start_frame=serve_start_frame,
                            history_origin_pos=serve_start_pos,
                        )
                        self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
                        self.waiting_serve_candidate = None
                        self.waiting_serve_candidate_frame = -1
                        log_tracking_start_position()
                        clear_waiting_serve_history()
                    else:
                        candidate_gap = self.frame_count - int(getattr(self, 'waiting_serve_candidate_frame', -1000000))
                        if not (far_top_serve_scan and last_serve_candidate is not None and candidate_gap <= 18):
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
                        scan_toss_context = low_to_up_serve_toss_context(
                            scan_position_history,
                            require_bottom_entry=bottom_entry_required_for_serve_start(),
                        )
                        if (all_forward and min_signed_dx > 25 and has_upward_motion and has_valid_vertical and
                                scan_toss_context is not None):
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
                            serve_start_frame = scan_toss_context.get('low_frame', self.frame_count)
                            serve_start_pos = scan_toss_context.get('low_pos', scan_position_history[0])
                            self.initial_ball_position = serve_start_pos
                            self.ball_size = None
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self._start_point_context(
                                potential_serve,
                                serve_start_frame=serve_start_frame,
                                history_origin_pos=serve_start_pos,
                            )
                            self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
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
                if (point_start_frame and
                        (self.frame_count - point_start_frame) > _max_point_frames and
                        True):
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
                    # Do not let the per-point duration cap turn a temporary
                    # post-hit visibility gap into a terminal result.  A
                    # racket contact starts a fresh flight; give the normal
                    # reacquisition path time to find that flight again even
                    # when the point has already exceeded max_point_frames.
                    last_contact_frame = int(
                        getattr(self, '_last_racket_contact_frame', -1000000)
                    )
                    contact_age = self.frame_count - last_contact_frame
                    recent_contact_timeout_hold = (
                        last_contact_frame >= int(point_start_frame or self.start_frame) and
                        0 <= contact_age <= 180 and
                        int(getattr(self, '_point_hit_count', 0) or 0) > 0
                    )
                    hard_timeout = _max_point_frames + max(120, int(_max_point_frames * 0.5))
                    timeout_marker_zone = self._player_point_zone(self.ball_center)
                    stale_player_timeout_hold = (
                        self.tracking and
                        self.ball_center is not None and
                        self.stuck_frame_count >= 5 and
                        timeout_marker_zone in (
                            'player_head_hat', 'player_body', 'player_shoes', 'racket_fragment'
                        ) and
                        dur <= hard_timeout
                    )
                    timeout_hold = (
                        top_timeout_hold or back_timeout_hold or
                        recent_return_hold or recent_bounce_hold
                    )
                    if (
                            active_tracking_hold or recent_contact_timeout_hold or
                            stale_player_timeout_hold or
                            (timeout_hold and dur <= hard_timeout)):
                        if top_timeout_hold or back_timeout_hold:
                            print(f"Frame {self.frame_count}: delaying point timeout while waiting for offscreen return")
                        elif recent_contact_timeout_hold:
                            print(
                                f"Frame {self.frame_count}: [POST-CONTACT TIMEOUT HOLD] "
                                f"deferring point timeout {contact_age}f after racket contact "
                                f"at f{last_contact_frame}"
                            )
                        elif stale_player_timeout_hold:
                            print(
                                f"Frame {self.frame_count}: [STALE-PLAYER TIMEOUT HOLD] "
                                f"deferring unresolved timeout at {self.ball_center} "
                                f"zone={timeout_marker_zone} stuck={self.stuck_frame_count}"
                            )
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
                        if self._ignore_unresolved_timeout_if_game_decider(
                            "POINT_TIMEOUT", end_position=self.ball_center, frame=frame
                        ):
                            print(
                                f"[POINT_IGNORED] f{self.frame_count}: reason=POINT_TIMEOUT "
                                f"duration={dur}f would decide game without a located ball"
                            )
                        else:
                            timeout_reason = "POINT_TIMEOUT"
                            if self._in_court_timeout_landing_outcome(self.ball_center, frame) is not None:
                                timeout_reason = "Ball stopped on player side"
                            # A duration timeout is only a fallback inference.  The
                            # terminal verifier can still see a compact moving ball
                            # in the just-decoded frame pair; continue the rally
                            # rather than scoring a false timeout in that case.
                            if self._resume_from_terminal_motion_candidate(
                                timeout_reason, self.ball_center, frame=frame
                            ):
                                continue
                            print(
                                f"[POINT_END] f{self.frame_count}: reason={timeout_reason} "
                                f"duration={dur}f — returning to serve detection"
                            )
                            self._record_point_result(timeout_reason, end_position=self.ball_center, frame=frame)
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True)

                # Track ball through the point
                tracked_position = None
                if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
                    prev_ball_center = self.ball_center
                    prev_stuck = self.stuck_frame_count
                    prev_top_return_wait = self._top_return_wait_active()
                    prev_back_return_wait = self._back_return_wait_active()
                    pre_track_snapshot = self._snapshot_tracking_state_for_provisional_guard()
                    self._local_ai_tight_roi_previous_gray = (
                        pre_track_snapshot.get('_prev_frame_gray')
                    )
                    forced_local_ai = False
                    contact_local_ai = False
                    tracked_position = self._force_local_ai_frame(frame, prev_ball_center)
                    if tracked_position is not None:
                        forced_local_ai = True
                    else:
                        # Once armed, contact Local AI owns the next few frames
                        # before HSV is allowed to mutate the anchor again.
                        tracked_position = self._run_contact_local_ai_frame(
                            frame, prev_ball_center
                        )
                        if tracked_position is not None:
                            contact_local_ai = True
                        else:
                            tracked_position = self.track_ball_in_frame(frame)
                            contact_reason = self._contact_local_ai_trigger(
                                prev_ball_center, tracked_position, pre_track_snapshot
                            )
                            if contact_reason is not None:
                                rejected_hsv = (
                                    tuple(tracked_position)
                                    if tracked_position is not None else None
                                )
                                self._restore_tracking_state_for_provisional_guard(
                                    pre_track_snapshot
                                )
                                self._local_ai_tight_roi_previous_gray = (
                                    pre_track_snapshot.get('_prev_frame_gray')
                                )
                                tracked_position = self._run_contact_local_ai_frame(
                                    frame, prev_ball_center, arm_reason=contact_reason
                                )
                                if tracked_position is not None:
                                    contact_local_ai = True
                                    print(
                                        f"[CONTACT_LOCAL_AI_REPLACE] f{self.frame_count}: "
                                        f"HSV={rejected_hsv} -> AI={tracked_position}"
                                    )
                                else:
                                    # A failed AI arm must not leave state partially
                                    # rolled back.  Hold the last trusted ball and retry
                                    # on the next frame rather than accepting player HSV.
                                    self.ball_center = tuple(prev_ball_center)
                                    self.stuck_frame_count = max(
                                        1, int(prev_stuck) + 1
                                    )
                                    tracked_position = tuple(prev_ball_center)
                                    contact_local_ai = True
                    if self.local_ai_recovery is not None and self._local_ai_frame_buffer:
                        self._local_ai_frame_buffer[-1]["normal_position"] = (
                            tuple(tracked_position) if tracked_position is not None else None
                        )
                    self._debug_local_ai_shadow_frame(frame, prev_ball_center, tracked_position)
                    if not forced_local_ai and not contact_local_ai:
                        tracked_position = self._try_local_ai_recovery(
                            prev_ball_center, tracked_position, prev_stuck,
                            pre_track_snapshot=pre_track_snapshot,
                            frame=frame,
                        )
                    self._local_ai_tight_roi_previous_gray = None
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
                        trusted_rally_launch = (
                            getattr(self, '_trusted_rally_launch_frame', -1000000) == self.frame_count and
                            getattr(self, '_trusted_rally_launch_pos', None) == tracked_position
                        )
                        # Allow a larger jump when re-acquiring after being stuck for 5+ frames:
                        # a racket hit can send the ball 800+ px in one frame, so we use 1500px
                        # to let motion-based re-acquisition recover across the full court.
                        max_jump = 1500 if (
                            prev_stuck >= 5 or prev_top_return_wait or prev_back_return_wait or
                            motion_reacq_this_frame or trusted_rally_launch
                        ) else 400
                        if jump > max_jump:
                            print(f"[JUMP_REJECTED] f{self.frame_count}: jumped {jump:.0f}px from {prev_ball_center} to {tracked_position} (limit={max_jump}px, prev_stuck={prev_stuck}), keeping previous")
                            self.ball_center = prev_ball_center
                            self.stuck_frame_count = max(self.stuck_frame_count, prev_stuck + 1)
                            tracked_position = prev_ball_center

                    # Validate only the position that survived the physical
                    # jump gate above.  Previously the provisional serve
                    # state consumed a 425px false contour at a prior frame before
                    # that contour was rejected below.  Its bogus vector then
                    # hid the real upward reversal at a prior frame.
                    # Once a far toss is independently marked as caught, it
                    # becomes a *history* question only.  Do not continue to
                    # mutate the live provisional flags: the previous attempt
                    # cleared those flags at a prior frame and changed the otherwise
                    # stable HSV path of the following real rally.
                    tainted_far_toss_active = (
                        isinstance(
                            getattr(self, '_tainted_provisional_serve_start', None),
                            dict,
                        ) and
                        getattr(self, '_provisional_serve_start_kind', None) ==
                        'far-top-post-hit'
                    )
                    provisional_kind_before_validation = getattr(
                        self, '_provisional_serve_start_kind', None
                    )
                    provisional_reject_reason = (
                        None if tainted_far_toss_active else
                        self._validate_provisional_serve_start(
                            prev_ball_center, tracked_position
                        )
                    )
                    # A pending far-toss token is solely a fallback for a
                    # *rejected* practice toss.  Once this tentative flight
                    # has earned three coherent netward steps, it is a real
                    # serve (for example a prior frame -> a prior frame) and the token must
                    # not survive its fault/end state.  Leaving it alive
                    # caused the local model to re-use that old start at
                    # a prior frame and inject a second, false point.
                    if provisional_reject_reason is not None:
                        provisional_kind = getattr(self, '_provisional_serve_start_kind', None)
                        rejected_start = self._current_history_serve_start_frame()
                        if provisional_kind == 'far-top-post-hit':
                            # This is a caught far-side toss.  Do *not* reset
                            # to WAITING_FOR_SERVE: that was the rejected
                            # approach and it changed later, already-verified
                            # points.  Keep the normal HSV flow alive and ask
                            # local AI to prove a later outgoing path in
                            # shadow mode.  Only proven evidence may alter the
                            # final history/serve-fault interpretation.
                            self._tainted_provisional_serve_start = {
                                'start_frame': rejected_start,
                                'detected_frame': int(self.frame_count),
                                'anchor': tuple(prev_ball_center),
                                'reason': provisional_reject_reason,
                            }
                            print(
                                f"[SERVE_START_TAINTED] f{self.frame_count}: "
                                f"keeping tracker/provisional state for start f{rejected_start}; "
                                f"{provisional_reject_reason}"
                            )
                        else:
                            print(
                                f"[SERVE_START_REJECTED] f{self.frame_count}: "
                                f"discarding provisional start f{rejected_start}; "
                                f"{provisional_reject_reason}"
                            )
                            self._point_history_current = None
                            self._provisional_serve_start_kind = None
                            self._serve_start_requires_confirmation = False
                            game_state = "WAITING_FOR_SERVE"
                            clear_waiting_serve_history()
                            reset_tracking_state()
                            continue
                    # Shadow verification is intentionally side-effect free
                    # for ball position/game state.  It only supplies a
                    # later verified launch frame for history/fault scoring.
                    reconcile_tainted_far_serve_start()
                    if getattr(self, "_discard_provisional_serve_from_ai", False):
                        self._discard_provisional_serve_from_ai = False
                        self._ignore_unconfirmed_serve_start_result(
                            "local AI rejected repeated all-player-body recovery path"
                        )
                        print(
                            f"[SERVE_START_IGNORED] f{self.frame_count}: "
                            "discarding false provisional serve after repeated local-AI body paths"
                        )
                        game_state = "WAITING_FOR_SERVE"
                        clear_waiting_serve_history()
                        reset_tracking_state()
                        continue
                if tracked_position:
                    vel = self.last_motion['distance'] if self.last_motion else 0
                    size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
                    vel_hist_tail = [round(v, 1) for v in getattr(self, 'ball_velocity_history', [])[-5:]]
                    print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
                    print(f"[TRACK] f{self.frame_count}: pos={tracked_position} vel={vel:.1f}px stuck={self.stuck_frame_count} vel_hist={vel_hist_tail}")
                    if self._point_history_current is not None:
                        self._point_history_current.setdefault('tracking_trace', []).append({
                            'frame': int(self.frame_count),
                            'pos': [int(tracked_position[0]), int(tracked_position[1])],
                            'size': float(self.ball_size) if self.ball_size is not None else None,
                            'stuck': int(self.stuck_frame_count),
                        })
                    if self.ball_dataset_exporter is not None:
                        self.ball_dataset_exporter.write_frame(
                            frame,
                            source_frame=self.frame_count,
                            ball_center=tracked_position,
                            ball_area=self.ball_size,
                            stuck_frames=self.stuck_frame_count,
                            motion_distance=(self.last_motion or {}).get('distance'),
                            tracking_active=self.tracking,
                            point_index=(
                                self._point_history_current.get('point_index')
                                if self._point_history_current is not None else None
                            ),
                        )

                    pending_reason = getattr(self, '_pending_rally_end_reason', None)
                    if pending_reason and reference_target_hold:
                        print(
                            f"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] "
                            f"discarding premature queued end '{pending_reason}' "
                            f"until f{reference_target_frame}"
                        )
                        self._pending_rally_end_reason = None
                        self._pending_rally_end_frame = -1
                        pending_reason = None
                    if pending_reason:
                        # Point-end detection can queue an out reason one or
                        # more frames before the stuck timeout is reached.  If
                        # this is still an unconfirmed serve toss with no
                        # racket/net contact and the marker has stopped, drop
                        # the queued false end instead of writing a bogus row.
                        suppressed_frame = int(
                            getattr(self, '_last_out_bounce_suppressed_frame', -1000000)
                        )
                        suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
                        recent_static_serve_abort = (
                            suppressed_point is not None and
                            0 <= self.frame_count - suppressed_frame <= 40 and
                            math.hypot(
                                float(tracked_position[0]) - float(suppressed_point[0]),
                                float(tracked_position[1]) - float(suppressed_point[1]),
                            ) <= max(100.0, float(getattr(self, 'ball_size', 0.0) or 0.0) * 3.0)
                        )
                        pending_serve_toss_abort = (
                            (getattr(self, '_awaiting_serve_bounce', False) or recent_static_serve_abort) and
                            not self._serve_net_touch_active(window_frames=120) and
                            not self._serve_net_pass_active(window_frames=36) and
                            getattr(self, 'ground_bounce_count', 0) <= 0 and
                            float((self.last_motion or {}).get('distance', 0.0) or 0.0) < 3.0 and
                            (self.frame_count - int(point_start_frame or self.frame_count)) >= 25 and
                            (
                                int(getattr(self, '_last_racket_contact_frame', -1000000)) < int(point_start_frame or self.frame_count) or
                                (
                                    recent_static_serve_abort and
                                    int(getattr(self, '_last_racket_contact_frame', -1000000)) <= suppressed_frame
                                )
                            ) and
                            any(token in str(pending_reason).lower() for token in ('out', 'bounce', 'stuck', 'lost'))
                        )
                        if pending_serve_toss_abort:
                            print(
                                f"Frame {self.frame_count}: [SERVE_TOSS_ABORTED] "
                                f"discarding queued end '{pending_reason}' started at f{point_start_frame} "
                                f"position={tracked_position}"
                            )
                            # The serve may already have been confirmed.  In
                            # that case this is only a stale HSV/player marker,
                            # not a new serve to discard: clear the queued end
                            # and keep tracking the same point so a later real
                            # ball/loss determines its outcome.
                            self._pending_rally_end_reason = None
                            self._pending_rally_end_frame = -1
                            self._last_out_bounce_suppressed_frame = self.frame_count
                            self._last_out_bounce_suppressed_point = tuple(tracked_position)
                            self.stuck_frame_count = 0
                            continue
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        print(f"Frame {self.frame_count}: POINT ENDED - {pending_reason}")
                        print(f"Point duration: {dur} frames")
                        print(f"[POINT_END] f{self.frame_count}: reason={pending_reason} duration={dur}f pos={tracked_position} vel={vel:.1f}px vel_hist={vel_hist_tail}")
                        self._record_point_result(pending_reason, end_position=tracked_position, frame=frame)
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

                    # During player-reacquisition grace, do not let the generic
                    # stuck timeout finalize the stale marker before the ball has
                    # had a chance to re-enter the search.  Keep the counter just
                    # below the timeout threshold; a real moving candidate clears
                    # the guard and normal timeout logic resumes.
                    player_reacq_grace = int(
                        getattr(self, '_player_reacq_protect_until_frame', -1)
                    ) >= self.frame_count
                    # The reacquisition guard normally suppresses a stuck
                    # timeout because the selected blob may be a player/racket
                    # fragment.  Keep one narrow exception for a small blob
                    # that is stationary well inside the court after real play:
                    # this is the ball resting while the player occludes it,
                    # not a body artifact.  Letting the timeout run here also
                    # preserves the existing stuck-history backdating so the
                    # recorded endpoint is the first stationary-ball frame.
                    reacq_stationary_ball = False
                    if player_reacq_grace and self.stuck_frame_count >= 15:
                        reacq_ball_size = float(getattr(self, 'ball_size', 0.0) or 0.0)
                        reacq_net_y = float(
                            getattr(self, 'net_area_y_max', frame.shape[0] * 0.42)
                            or frame.shape[0] * 0.42
                        )
                        reacq_prior_play = (
                            int(getattr(self, 'ground_bounce_count', 0) or 0) > 0 or
                            int(getattr(self, '_last_point_hit_count', 0) or 0) > 0 or
                            int(getattr(self, '_last_racket_contact_frame', -1000000)) >= int(
                                point_start_frame or self.start_frame
                            )
                        )
                        reacq_stationary_ball = (
                            0.0 < reacq_ball_size <= max(180.0, frame.shape[1] * 0.012) and
                            float(tracked_position[1]) > reacq_net_y + max(120.0, frame.shape[0] * 0.08) and
                            # Motion fields can be stale while the tracker is
                            # holding a reacquisition marker; stuck_frame_count
                            # is the authoritative stationary signal here.
                            reacq_prior_play
                        )
                    if reacq_stationary_ball:
                        print(
                            f"Frame {self.frame_count}: [STATIONARY BALL TIMEOUT OVERRIDE] "
                            f"small on-court blob size={reacq_ball_size:.1f} "
                            f"stuck={self.stuck_frame_count}; allowing timeout"
                        )
                    if player_reacq_grace and self.stuck_frame_count >= 15 and not reacq_stationary_ball:
                        self.stuck_frame_count = 14
                        print(
                            f"Frame {self.frame_count}: [PLAYER-REACQ TIMEOUT SUPPRESSED] "
                            f"holding stale marker {tracked_position}"
                        )
                    elif reference_target_hold and self.stuck_frame_count >= 15:
                        # A reviewed/reference endpoint may keep the point alive,
                        # but it must not erase evidence that tracking is stuck.
                        # Preserve the stuck count so the next frame can invoke
                        # held-position Local AI / reacquisition instead of
                        # repeatedly pinning a stale marker at count 14.
                        print(
                            f"Frame {self.frame_count}: [REFERENCE_POINT_HOLD] "
                            f"suppressing point-end timeout until f{reference_target_frame}; "
                            f"preserving stuck={self.stuck_frame_count} for recovery"
                        )
                    # Stuck-ball timeout: if ball hasn't moved for 15+ frames, end point
                    elif self.stuck_frame_count >= 15 and not self._top_return_wait_active():
                        # A toss can be started and then caught/aborted without a
                        # racket contact.  In that case a stale court highlight may
                        # sit outside the sideline and the night timeout helper
                        # would incorrectly award an out error.  Discard this
                        # unconfirmed serve attempt and return to serve scanning;
                        # the next real toss will create a fresh history row.
                        suppressed_frame = int(
                            getattr(self, '_last_out_bounce_suppressed_frame', -1000000)
                        )
                        suppressed_point = getattr(self, '_last_out_bounce_suppressed_point', None)
                        recent_static_serve_abort = (
                            suppressed_point is not None and
                            0 <= self.frame_count - suppressed_frame <= 40 and
                            math.hypot(
                                float(tracked_position[0]) - float(suppressed_point[0]),
                                float(tracked_position[1]) - float(suppressed_point[1]),
                            ) <= max(100.0, float(getattr(self, 'ball_size', 0.0) or 0.0) * 3.0)
                        )
                        serve_toss_aborted = (
                            (getattr(self, '_awaiting_serve_bounce', False) or recent_static_serve_abort) and
                            not self._serve_net_touch_active(window_frames=120) and
                            not self._serve_net_pass_active(window_frames=36) and
                            getattr(self, 'ground_bounce_count', 0) <= 0 and
                            float((self.last_motion or {}).get('distance', 0.0) or 0.0) < 3.0 and
                            (self.frame_count - int(point_start_frame or self.frame_count)) >= 25 and
                            int(getattr(self, '_last_racket_contact_frame', -1000000)) < int(point_start_frame or self.frame_count)
                        )
                        if serve_toss_aborted:
                            print(
                                f"Frame {self.frame_count}: [SERVE_TOSS_ABORTED] "
                                f"discarding unconfirmed static attempt started at f{point_start_frame} "
                                f"position={tracked_position}"
                            )
                            self._ignore_unconfirmed_serve_start_result(
                                "aborted serve toss before racket/net contact"
                            )
                            game_state = "WAITING_FOR_SERVE"
                            reset_tracking_state(hold_end_marker=False)
                            continue
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        stuck_reason = (
                            self._night_stuck_timeout_out_reason(tracked_position, frame) or
                            "STUCK_TIMEOUT"
                        )
                        unresolved_timeout = stuck_reason == "STUCK_TIMEOUT"
                        if unresolved_timeout and self._ignore_unresolved_timeout_if_game_decider(
                            stuck_reason, end_position=tracked_position, frame=frame
                        ):
                            print(
                                f"[POINT_IGNORED] f{self.frame_count}: reason={stuck_reason} "
                                f"stuck={self.stuck_frame_count} duration={dur}f would decide game without a located ball"
                            )
                        else:
                            if self._in_court_timeout_landing_outcome(tracked_position, frame) is not None:
                                stuck_reason = "Ball stopped on player side"
                            # Do this before choosing/backdating the endpoint.  A
                            # timeout on a stale player/racket marker must never
                            # win over a genuinely moving ball recovered from the
                            # latest two frames.
                            if self._resume_from_terminal_motion_candidate(
                                stuck_reason, tracked_position, frame=frame
                            ):
                                continue
                            static_timeout_point = getattr(
                                self, '_last_out_bounce_suppressed_point', None
                            )
                            static_timeout_frame = int(
                                getattr(self, '_last_out_bounce_suppressed_frame', -1000000)
                            )
                            static_timeout_artifact = (
                                static_timeout_point is not None and
                                0 <= self.frame_count - static_timeout_frame <= 40 and
                                math.hypot(
                                    float(tracked_position[0]) - float(static_timeout_point[0]),
                                    float(tracked_position[1]) - float(static_timeout_point[1]),
                                ) <= max(
                                    100.0,
                                    float(getattr(self, 'ball_size', 0.0) or 0.0) * 3.0,
                                )
                            )
                            history_end_frame = (
                                self.frame_count
                                if static_timeout_artifact else
                                self._stuck_timeout_end_frame(
                                    point_start_frame=point_start_frame,
                                    frame=self.frame_count,
                                )
                            )
                            if static_timeout_artifact:
                                print(
                                    f"Frame {self.frame_count}: [STUCK-TIMEOUT FRAME NOT BACKDATED] "
                                    f"static marker began at f{static_timeout_frame}"
                                )
                            print(f"Frame {self.frame_count}: POINT ENDED - {stuck_reason}")
                            print(f"Point duration: {dur} frames")
                            print(
                                f"[POINT_END] f{self.frame_count}: reason={stuck_reason} "
                                f"stuck={self.stuck_frame_count} duration={dur}f pos={tracked_position} "
                                f"history_frame=f{history_end_frame}"
                            )
                            self._record_point_result(
                                stuck_reason,
                                end_position=tracked_position,
                                frame=frame,
                                history_end_frame=history_end_frame,
                            )
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True, end_position=tracked_position)
                    elif self.stuck_frame_count >= 15 and self._top_return_wait_active():
                        self.stuck_frame_count = 4
                        print(f"Frame {self.frame_count}: [TOP-RETURN WAIT] suppressing stuck timeout while waiting for delayed re-entry")
                    else:
                        # Check if point has ended
                        local_ai_handoff_active = (
                            self.local_ai_recovery is not None and
                            self.frame_count <= int(getattr(self, '_local_ai_follow_until_frame', -1))
                        )
                        if local_ai_handoff_active:
                            # The handoff protects against player/racket blobs
                            # masquerading as a bounce, but it must never hide
                            # a geometry-confirmed boundary bounce.  Give the
                            # existing generic boundary-reversal detector first
                            # refusal; only ambiguous endpoint evidence is held.
                            boundary_out, boundary_reason = (
                                self._confirmed_boundary_reversal_out_candidate(
                                    tracked_position, frame
                                )
                            )
                            if boundary_out:
                                point_ended, reason = True, boundary_reason
                                print(
                                    f"[LOCAL_AI_POINT_END_BOUNDARY_OVERRIDE] "
                                    f"f{self.frame_count}: {boundary_reason}"
                                )
                            else:
                                point_ended, reason = False, None
                                print(
                                    f"[LOCAL_AI_POINT_END_HOLD] f{self.frame_count}: "
                                    f"verified recovery path active through "
                                    f"f{self._local_ai_follow_until_frame}"
                                )
                        else:
                            point_ended, reason = self.detect_point_end(tracked_position, frame)
                        if point_ended:
                            point_end_frame = self.frame_count
                            dur = point_end_frame - point_start_frame if point_start_frame else 0
                            print(f"Frame {self.frame_count}: POINT ENDED - {reason}")
                            print(f"Point duration: {dur} frames")
                            print(f"[POINT_END] f{self.frame_count}: reason={reason} duration={dur}f pos={tracked_position} vel={vel:.1f}px vel_hist={vel_hist_tail}")
                            self._record_point_result(reason, end_position=tracked_position, frame=frame)
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
                                for token in ("net", "bounce", "stopped", "lost", "fence", "let")
                            )
                            if wait_for_next_serve:
                                if "net" in reason.lower():
                                    self.net_contact_points.append(tracked_position)
                                game_state = "WAITING_FOR_SERVE"
                            else:
                                game_state = "POINT_ENDED"
                            reset_tracking_state(hold_end_marker=True, end_position=tracked_position)
                        else:
                            _verbose_debug_print(f"Frame {self.frame_count}: Ball tracking continued")
                else:
                    # Ball lost - might be end of point
                    if getattr(self, '_back_return_timed_out', False):
                        point_end_frame = self.frame_count
                        dur = point_end_frame - point_start_frame if point_start_frame else 0
                        print(f"Frame {self.frame_count}: POINT ENDED - Ball lost after back-return timeout")
                        print(f"Point duration: {dur} frames")
                        print(f"[POINT_END] f{self.frame_count}: reason=BACK_RETURN_TIMEOUT duration={dur}f")
                        self._record_point_result("BACK_RETURN_TIMEOUT", end_position=self.ball_center, frame=frame)
                        game_state = "WAITING_FOR_SERVE"
                        reset_tracking_state(hold_end_marker=True)
                        continue
                    recent_return_hold = self._recent_offscreen_return_hold_active(window_frames=8)
                    recent_bounce_hold = self._recent_return_bounce_recover_active(window_frames=2)
                    # A confirmed racket contact is the start of a new ball
                    # flight, not the end of the point.  The detector can lose
                    # the outgoing ball for a handful of frames (especially
                    # while it crosses a player/racket or a bright court
                    # line), and the generic lost-ball timeout would otherwise
                    # finalize the point immediately.  Keep the rally alive
                    # for a bounded post-contact window while the normal
                    # motion/HSV reacquisition code searches again.  This is
                    # deliberately based on the contact frame, rather than on
                    # the current marker, so a stale body/racket marker cannot
                    # masquerade as the ball's terminal position.
                    last_contact_frame = int(
                        getattr(self, '_last_racket_contact_frame', -1000000)
                    )
                    contact_age = self.frame_count - last_contact_frame
                    post_contact_hold = (
                        point_start_frame and
                        last_contact_frame >= int(point_start_frame) and
                        0 <= contact_age <= 180 and
                        int(getattr(self, '_point_hit_count', 0) or 0) > 0
                    )
                    if post_contact_hold:
                        self.stuck_frame_count = min(
                            max(int(getattr(self, 'stuck_frame_count', 0) or 0), 1),
                            4,
                        )
                        print(
                            f"Frame {self.frame_count}: [POST-CONTACT HOLD] "
                            f"ball not visible {contact_age}f after racket contact "
                            f"at f{last_contact_frame}; continuing reacquisition"
                        )
                        continue
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
                        self._record_point_result("Ball lost (likely out of court)", end_position=self.ball_center, frame=frame)
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
        
            elif game_state == "WAITING_FOR_SERVE" and not early_serve_scan_handled:
                # Detect ball in serve area, accumulate position history,
                # start tracking only when ball exits serve area in the configured serve direction
                import math as _math
                # Confirmation evidence must be consecutive.  The caught toss
                # at a prior frame disappears from the serve detector for two frames;
                # retaining its armed state allowed unrelated blobs at a prior frame
                # to complete the count.  Clear both the arm and its motion
                # history so a later real serve has to create a fresh toss.
                if (
                        pending_far_post_hit is not None and
                        int(self.frame_count) - int(
                            pending_far_post_hit.get('last_frame', self.frame_count)
                        ) > 1):
                    print(
                        f"[SERVE_START_REJECTED] f{self.frame_count}: armed far toss "
                        "lost consecutive detection before contact"
                    )
                    pending_far_post_hit = None
                    clear_waiting_serve_history()
                potential_serve = self.detect_serve_position(
                    frame,
                    lock_history=serve_position_history if serve_candidate_lock_active else None,
                    lock_miss_frames=serve_candidate_lock_miss_frames,
                )
                if potential_serve:
                    serve_candidate_lock_miss_frames = 0
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
                        current_serve_detail = dict(current_serve_detail)
                        current_serve_detail.setdefault('frame', int(self.frame_count))
                        serve_candidate_details_history.append(current_serve_detail)
                    else:
                        serve_candidate_details_history.append({'pos': potential_serve, 'frame': int(self.frame_count)})
                    maybe_lock_serve_candidate(potential_serve, current_serve_detail)
                    if len(serve_position_history) > 20:
                        serve_position_history = serve_position_history[-20:]
                    if len(serve_candidate_details_history) > 20:
                        serve_candidate_details_history = serve_candidate_details_history[-20:]
                    far_post_hit_context = far_top_post_hit_context(
                        serve_position_history,
                        details=serve_candidate_details_history,
                    )
                    if far_post_hit_context is not None:
                        print(
                            f"[TRACKING_START] f{self.frame_count}: far-post-hit serve launch "
                            f"at {potential_serve} low={far_post_hit_context['low_pos']} "
                            f"apex={far_post_hit_context['apex_pos']} "
                            f"drop={far_post_hit_context['forward_drop']:.0f}px "
                            f"down_steps={far_post_hit_context['down_steps']}"
                        )
                        self.ball_center = potential_serve
                        self.tracking = True
                        self.ball_stopped = False
                        self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                        self.initial_ball_position = far_post_hit_context['low_pos']
                        self.ball_size = None
                        self.ball_hsv = None
                        seed_tracking_from_serve_history(potential_serve)
                        self.stuck_frame_count = 0
                        point_start_frame = self.frame_count
                        self.point_start_frame_internal = self.frame_count
                        self._start_point_context(potential_serve)
                        self._serve_start_requires_confirmation = True
                        self._begin_provisional_serve_start('far-top-post-hit')
                        self.waiting_serve_candidate = None
                        self.waiting_serve_candidate_frame = -1
                        game_state = "TRACKING_POINT"
                        log_tracking_start_position()
                        clear_waiting_serve_history()
                        continue
                    if pending_far_post_hit is not None:
                        armed_last = pending_far_post_hit['last_pos']
                        armed_dx = float(potential_serve[0] - armed_last[0])
                        armed_dy = float(potential_serve[1] - armed_last[1])
                        armed_distance = _math.hypot(armed_dx, armed_dy)
                        # A caught toss drops slowly, while a struck far-side
                        # serve has compact 20-90px netward steps (a prior frame is
                        # 29, 33, 36px).  Reject reversals and large contour
                        # jumps before a point has been created.
                        if (
                                (armed_distance >= 18.0 and armed_dy <= -20.0) or
                                armed_distance > 90.0):
                            print(
                                f"[SERVE_START_REJECTED] f{self.frame_count}: armed far toss "
                                f"lost physical continuation (delta=({armed_dx:.0f},{armed_dy:.0f}), "
                                f"distance={armed_distance:.0f})"
                            )
                            pending_far_post_hit = None
                            clear_waiting_serve_history()
                            continue
                        if armed_dy >= 18.0 and 20.0 <= armed_distance <= 90.0:
                            pending_far_post_hit['forward_steps'] += 1
                        pending_far_post_hit['last_pos'] = tuple(potential_serve)
                        pending_far_post_hit['last_frame'] = int(self.frame_count)
                        if pending_far_post_hit['forward_steps'] < 3:
                            continue

                        armed_context = pending_far_post_hit['context']
                        armed_start_frame = int(pending_far_post_hit['start_frame'])
                        armed_low_pos = tuple(pending_far_post_hit['low_pos'])
                        pending_far_post_hit = None
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
                            f"[TRACKING_START] f{self.frame_count}: confirmed far-post-hit serve "
                            f"at {potential_serve} low={armed_context['low_pos']} "
                            f"apex={armed_context['apex_pos']} "
                            f"drop={armed_context['forward_drop']:.0f}px "
                            f"down_steps={armed_context['down_steps']} "
                            f"armed_at=f{armed_start_frame}"
                        )
                        self.ball_center = potential_serve
                        self.tracking = True
                        self.ball_stopped = False
                        self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                        self.initial_ball_position = armed_low_pos
                        self.ball_size = None
                        self.ball_hsv = None
                        seed_tracking_from_serve_history(potential_serve)
                        self.stuck_frame_count = 0
                        point_start_frame = armed_start_frame
                        self.point_start_frame_internal = armed_start_frame
                        self._start_point_context(
                            potential_serve,
                            serve_start_frame=armed_start_frame,
                            history_origin_pos=armed_low_pos,
                        )
                        self._serve_start_requires_confirmation = False
                        self.waiting_serve_candidate = None
                        self.waiting_serve_candidate_frame = -1
                        game_state = "TRACKING_POINT"
                        log_tracking_start_position()
                        clear_waiting_serve_history()
                        continue

                    if far_post_hit_context is not None:
                        if pending_far_post_hit is None:
                            pending_far_post_hit = {
                                'start_frame': int(self.frame_count),
                                'low_pos': tuple(far_post_hit_context['low_pos']),
                                'last_pos': tuple(potential_serve),
                                'last_frame': int(self.frame_count),
                                'context': dict(far_post_hit_context),
                                'forward_steps': 0,
                            }
                            print(
                                f"[SERVE_START_ARMED] f{self.frame_count}: far toss at "
                                f"{potential_serve}; awaiting netward continuation"
                            )
                            continue
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
                    require_bottom_entry = bottom_entry_required_for_serve_start()
                    serve_toss_context = low_to_up_serve_toss_context(
                        serve_position_history,
                        details=serve_candidate_details_history,
                        require_bottom_entry=require_bottom_entry,
                    )
                    if serve_toss_context is not None and not far_top_serve_active():
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
                        serve_start_frame = serve_toss_context.get('low_frame', self.frame_count)
                        point_start_frame = self.frame_count
                        self.point_start_frame_internal = self.frame_count
                        self._start_point_context(
                            potential_serve,
                            serve_start_frame=serve_start_frame,
                            history_origin_pos=serve_toss_context['low_pos'],
                        )
                        self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
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
                        # catches a prior frame/a prior frame where the first forward step is slightly
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
                        fast_start_toss_context = low_to_up_serve_toss_context(
                            serve_position_history,
                            details=serve_candidate_details_history,
                            require_bottom_entry=require_bottom_entry,
                        )
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
                            serve_start_frame = fast_start_toss_context.get('low_frame', self.frame_count)
                            serve_start_pos = fast_start_toss_context.get('low_pos', serve_position_history[0])
                            self.initial_ball_position = serve_start_pos
                            self.ball_size = None
                            self.ball_hsv = None
                            seed_tracking_from_serve_history(potential_serve)
                            self.stuck_frame_count = 0
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self._start_point_context(
                                potential_serve,
                                serve_start_frame=serve_start_frame,
                                history_origin_pos=serve_start_pos,
                            )
                            self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            log_tracking_start_position()
                            clear_waiting_serve_history()
                else:
                    # Ball exited serve area — only start tracking if it was moving at serve speed
                    # in the configured serve direction.
                    # A ball just sitting/bouncing in the area (false positive like a prior frame) has near-zero speed.
                    if serve_candidate_lock_active:
                        serve_candidate_lock_miss_frames += 1
                        miss_limit = 8 if far_top_serve_active() else 2
                        if serve_candidate_lock_miss_frames == 1:
                            print(
                                f"[SERVE_LOCK_MISS] f{self.frame_count}: preserving locked serve path "
                                f"last={last_serve_candidate}"
                            )
                        elif serve_candidate_lock_miss_frames > miss_limit:
                            print(
                                f"[SERVE_LOCK_LOST] f{self.frame_count}: lost locked serve path after "
                                f"{serve_candidate_lock_miss_frames} misses"
                            )
                            clear_waiting_serve_history()
                            continue
                    preserve_far_top_gap = False
                    if (far_top_serve_active() and serve_tracking_frames >= 1 and
                            last_serve_candidate is not None):
                        candidate_gap = self.frame_count - int(getattr(self, 'waiting_serve_candidate_frame', -1000000))
                        if candidate_gap <= 18:
                            preserve_far_top_gap = True
                            if candidate_gap == 1:
                                print(
                                    f"[SERVE_TOSS_HOLD] f{self.frame_count}: preserving far-side toss history "
                                    f"after candidate left red area last={last_serve_candidate}"
                                )
                    if (not preserve_far_top_gap and serve_tracking_frames >= 3 and
                            last_serve_candidate is not None and len(serve_position_history) >= 2):
                        total_dx = serve_position_history[-1][0] - serve_position_history[0][0]
                        last_dx = serve_position_history[-1][0] - serve_position_history[-2][0]
                        last_dy = serve_position_history[-1][1] - serve_position_history[-2][1]
                        print(f"[SERVE_EXIT_CHECK] f{self.frame_count}: tracked {serve_tracking_frames}f, total_dx={total_dx:.0f}px, last_dx={last_dx:.0f}px, last_dy={last_dy:.0f}px")
                        signed_total_dx = self._signed_serve_dx(total_dx)
                        signed_last_dx = self._signed_serve_dx(last_dx)
                        signed_last_dy = self._signed_serve_dy(last_dy) if self.serve_direction_dy != 0 else last_dy
                        # Reject toss motion that still moves strongly opposite the configured serve launch.
                        exit_start_toss_context = low_to_up_serve_toss_context(
                            serve_position_history,
                            details=serve_candidate_details_history,
                            require_bottom_entry=bottom_entry_required_for_serve_start(),
                        )
                        if exit_start_toss_context is not None and not far_top_serve_active():
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
                            serve_start_frame = exit_start_toss_context.get('low_frame', self.frame_count)
                            point_start_frame = self.frame_count
                            self.point_start_frame_internal = self.frame_count
                            self._start_point_context(
                                last_serve_candidate,
                                serve_start_frame=serve_start_frame,
                                history_origin_pos=exit_start_toss_context['low_pos'],
                            )
                            self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
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
                            self._serve_start_requires_confirmation = serve_start_result_confirmation_required()
                            self.waiting_serve_candidate = None
                            self.waiting_serve_candidate_frame = -1
                            game_state = "TRACKING_POINT"
                            log_tracking_start_position()
                    if not preserve_far_top_gap:
                        clear_waiting_serve_history()

            # If a far-side toss/catch was deliberately rejected, do not
            # silently miss the real serve that follows it.  The local model
            # must show a continuous four-frame ball path before this creates
            # a point; a single green/HSV contour is never sufficient here.
            if game_state == "WAITING_FOR_SERVE":
                recovered_serve_launch = recover_pending_far_serve_launch()
                if recovered_serve_launch is not None:
                    recovered_pos = (
                        int(recovered_serve_launch["x"]),
                        int(recovered_serve_launch["y"]),
                    )
                    launch_start_frame = max(
                        int(pending_far_toss_frame) + 1,
                        int(recovered_serve_launch.get("frame", self.frame_count)) -
                        int(getattr(self.local_ai_recovery, "lookback_frames", 4)) + 1,
                    )
                    print(
                        f"[TRACKING_START] f{self.frame_count}: local-AI far serve launch "
                        f"at {recovered_pos}; path starts f{launch_start_frame} "
                        f"after rejected toss f{pending_far_toss_frame}"
                    )
                    self.ball_center = recovered_pos
                    self.ball_size = float(recovered_serve_launch.get("area", 0.0) or 0.0)
                    self.ball_hsv = None
                    self.tracking = True
                    self.ball_stopped = False
                    self.stuck_frame_count = 0
                    self.last_seen_frame = self.frame_count
                    self.last_motion = None
                    self.last_delta = (0.0, 0.0)
                    self.ball_velocity_history = []
                    self.initial_ball_position = recovered_pos
                    self._serve_contact_grace_frames = max(self._serve_contact_grace_frames, 30)
                    point_start_frame = launch_start_frame
                    self.point_start_frame_internal = launch_start_frame
                    self._start_point_context(
                        recovered_pos,
                        serve_start_frame=launch_start_frame,
                        history_origin_pos=recovered_pos,
                    )
                    self._local_ai_recovery_count = 1
                    self._local_ai_handoff_deadline_frame = self.frame_count + 14
                    self._local_ai_follow_until_frame = self.frame_count + 4
                    self.waiting_serve_candidate = None
                    self.waiting_serve_candidate_frame = -1
                    pending_far_toss_frame = -1000000
                    pending_far_toss_position = None
                    pending_far_toss_last_ai_attempt = -1000000
                    game_state = "TRACKING_POINT"
                    log_tracking_start_position()
                    clear_waiting_serve_history()
                    continue
            
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
                    show_paused_rejected=False,
                    game_state=game_state,
                )

                display_state = "WAITING_FOR_SERVE" if game_state in (
                    "SCANNING_FOR_SERVE", "POINT_ENDED", "WAITING_FOR_SERVE"
                ) else game_state
                status_text = f"Frame: {self.frame_count}/{self.total_frames} | Game State: {display_state}"
                cv2.putText(
                    display_frame,
                    status_text,
                    (10, new_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if not self.headless:
                    cv2.imshow("Tennis Game Tracker", display_frame)

            prev_frame_for_debug = last_frame_for_debug
            prev_ball_center_for_debug = last_ball_center_for_debug
            prev_frame_index_for_debug = last_frame_index_for_debug
            last_frame_for_debug = frame.copy()
            last_ball_center_for_debug = self.ball_center
            last_frame_index_for_debug = self.frame_count

            if not getattr(self, 'auto_play', False) and not self.headless:
                pause_at_frame = getattr(self, 'pause_at_frame', None)
                if (pause_at_frame is not None and not pause_at_frame_triggered
                        and self.frame_count >= pause_at_frame):
                    play_mode = False
                    pause_at_frame_triggered = True
                    print(f"[PAUSE_AT_FRAME] Paused at frame {self.frame_count}")
                if self.pause_requested:
                    play_mode = False
                if jump_frames_remaining > 0:
                    jump_frames_remaining -= 1
                    key = 0xFF
                else:
                    key = cv2.waitKey(30 if play_mode else 0) & 0xFF
            else:
                key = 0xFF  # no-op key in headless/auto_play mode
            if key == ord('q'):
                self._process_stop_reason = "USER_QUIT"
                break
            elif key == 27:
                self._process_stop_reason = "USER_QUIT"
                break
            elif ord('1') <= key <= ord('9'):
                jump_frames_remaining = (key - ord('0')) * 10
                play_mode = False
                print(f"[JUMP] advancing {jump_frames_remaining} tracked frames")
                continue
            elif key == ord(' '):
                play_mode = not play_mode
                continue
            elif key in (ord('h'), ord('H')):
                debug_point = self.ball_center or last_ball_center_for_debug or prev_ball_center_for_debug
                debug_frame = frame
                debug_frame_index = self.frame_count
                if self.ball_center is None and last_frame_for_debug is not None:
                    debug_frame = last_frame_for_debug
                    debug_frame_index = last_frame_index_for_debug or self.frame_count
                if debug_point is None:
                    print("[HSV_DEBUG] No tracked point available for active/Alt2/S9 debug.")
                else:
                    self.open_tracking_hsv_debug_focus(debug_frame, debug_point, debug_frame_index)
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

            if self.pause_requested:
                self.pause_requested = False
        
        tracker = getattr(self, "player_tracker", None)
        if tracker is not None:
            try:
                tracker.save_profile()
            except Exception:
                pass
        if self.ball_dataset_exporter is not None:
            self.ball_dataset_exporter.close()
        self.cap.release()
        cv2.destroyAllWindows()
        
        if self._process_stop_reason is None:
            self._process_stop_reason = "COMPLETED"
        print(f"\nAnalysis complete! Processed {self.frame_count} frames")
        return {
            "reason": self._process_stop_reason,
            "frame_count": int(self.frame_count),
            "processed_frames": max(0, int(self.frame_count) - int(self.start_frame)),
            "video_path": self.video_path,
        }


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
        "night": {
            "video":  "04_Left_Court_tennis_20260623220627_20260623222559_1274169779.mp4",
            "config": "hsv_config_04_left_night.json",
            "label":  "04 Left Court night session",
        },
        "night2": {
            "video":  "04_Left_Court_tennis_20260623222559_20260623225255_1274247426.mp4",
            "config": "hsv_config_04_left_night.json",
            "label":  "04 Left Court night session 2",
        },
        "night3": {
            "video":  "04_Left_Court_tennis_20260623225255_20260623232026_1274349214.mp4",
            "config": "hsv_config_04_left_night.json",
            "label":  "04 Left Court night session 3",
        },
    }

    parser = argparse.ArgumentParser(description="Interactive tennis ball analyzer/tracker")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Frame index to start from (default 0)")
    parser.add_argument("--pause-at-frame", type=int, default=None,
                        help="Automatically pause once when this frame is reached")
    parser.add_argument("--start-score", type=_parse_start_score, metavar="'G1:G2 P1:P2'",
                        help=("Seed the score for a partial run, for example "
                              "--start-score \"0:2 0:15\". Completed games also select "
                              "the current server."))
    parser.add_argument("--auto-play", action="store_true",
                        help="Start playing immediately without waiting for SPACE")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after processing this many frames (0 = no limit)")
    parser.add_argument("--court", choices=list(COURT_CONFIGS.keys()), default="night",
                        help="Which court/video to analyse (default: night)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (no GUI windows)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress tracker debug/status console output")
    parser.add_argument("--verbose-debug", action="store_true",
                        help="Show detailed contour/search rejection diagnostics")
    parser.add_argument("--disable-false-points", dest="disable_false_points", action="store_true",
                        help="Disable learned false-point hiding in debug/tuner views (default)")
    parser.add_argument("--enable-false-points", dest="disable_false_points", action="store_false",
                        help="Enable learned false-point hiding in debug/tuner views")
    parser.add_argument("--point-history-file", default="point_history.csv",
                        help="Base CSV path for timestamped point history output (default: point_history.csv)")
    parser.add_argument("--no-point-history", action="store_true",
                        help="Disable point history CSV output")
    parser.add_argument("--export-ball-dataset", metavar="DIR",
                        help=("Write each tracker-labelled source frame and JSONL metadata under DIR. "
                              "Labels are pseudo labels; a timestamped run folder is created."))
    local_ai_recovery = parser.add_mutually_exclusive_group()
    local_ai_recovery.add_argument(
        "--local-ai-model", metavar="PATH",
        default=os.path.join("metadata", "ball_dataset", "ball_patch_model_v2.pt"),
        help=("Local ball model used for guarded buffered recovery (enabled by default). "
              "Normal HSV/motion tracking remains primary."),
    )
    local_ai_recovery.add_argument(
        "--no-local-ai-recovery", dest="local_ai_model", action="store_const", const=None,
        help="Disable guarded local-AI recovery for an HSV-only comparison run.",
    )
    parser.add_argument("--local-ai-python",
                        default=os.path.join(".tools", "ball-ai-venv310", "Scripts", "python.exe"),
                        help="Python 3.10 runtime containing the local AI dependencies")
    parser.add_argument("--local-ai-recovery-dir", default="tmp/local_ai_recovery",
                        help="Directory for local-AI recovery decision logs (default: tmp/local_ai_recovery)")
    parser.add_argument("--debug-local-ai-range", type=_parse_frame_range, metavar="START:END", default=None,
                        help="Raw Local-AI rank debug for an inclusive frame range; does not change tracking")
    parser.add_argument("--debug-local-ai-radius", type=float, default=140.0,
                        help="ROI radius for --debug-local-ai-range (default 140 px)")
    parser.add_argument("--force-local-ai-range", type=_parse_frame_range, metavar="START:END", default=None,
                        help="TEST ONLY: bypass HSV and let Local AI own an inclusive frame range")
    parser.add_argument("--force-local-ai-radius", type=float, default=450.0,
                        help="Search radius for --force-local-ai-range (default 450 px)")
    parser.add_argument("--force-local-ai-min-score", type=float, default=0.985,
                        help="Minimum Local-AI score in forced range (default 0.985)")
    parser.add_argument("--debug-local-ai-top-n", type=int, default=10,
                        help="Print/save Top-N raw Local-AI candidates in debug range (default 10)")
    parser.add_argument("--debug-hsv-sweep", action="store_true",
                        help="Compare regular-to-alt2 interpolated HSV ranges in Local-AI debug frames")
    parser.add_argument("--disable-player-tracking", action="store_true",
                        help="Disable player/racket context tracking and overlays")
    parser.add_argument("--player-tracking-interval", type=int, default=5,
                        help="Run player/racket detector every N frames (default: 5)")
    parser.add_argument("--disable-player-learning", action="store_true",
                        help="Do not update the persistent player stroke profile")
    player_ball_protection = parser.add_mutually_exclusive_group()
    player_ball_protection.add_argument("--enable-player-ball-protection", dest="enable_player_ball_protection",
                        action="store_true", help="Use tracked player regions to penalize ball candidates (default)")
    player_ball_protection.add_argument("--disable-player-ball-protection", dest="enable_player_ball_protection",
                        action="store_false", help="Disable player-region candidate protection")
    parser.set_defaults(enable_player_ball_protection=True)
    parser.add_argument("--audit-points", dest="audit_points", action="store_true", default=True,
                        help="After tracking, audit every point start/end against extracted image sequences (default)")
    parser.add_argument("--no-audit-points", dest="audit_points", action="store_false",
                        help="Skip the automatic point visual audit")
    parser.add_argument("--audit-provider", choices=["auto", "local", "openai"], default="auto",
                        help="Visual audit backend: local checks, OpenAI vision, or auto (default)")
    parser.add_argument("--audit-model", default=os.environ.get("POINT_AUDIT_MODEL", "gpt-5.4-mini"),
                        help="Vision model used by --audit-provider openai/auto")
    parser.add_argument("--audit-output-dir", default="tmp/point_audits",
                        help="Base directory for visual audit reports")
    parser.add_argument("--audit-fail-on-review", action="store_true",
                        help="Exit with status 2 when the point auditor flags any result")
    parser.add_argument("--continue-next-video", dest="continue_next_video",
                        action="store_true", default=True,
                        help="After a media read failure, continue with the next configured video (default)")
    parser.add_argument("--no-continue-next-video", dest="continue_next_video",
                        action="store_false",
                        help="Stop instead of opening the next configured video after a media read failure")
    serve_side_group = parser.add_mutually_exclusive_group()
    serve_side_group.add_argument("--near", dest="start_server_side",
                                  action="store_const", const="near", default="near",
                                  help="Start the first two games from the near side (default)")
    serve_side_group.add_argument("--far", dest="start_server_side",
                                  action="store_const", const="far",
                                  help="Start the first two games from the far side using the far serve area")
    serve_side_group.add_argument("--start-server-side", choices=["near", "far"],
                                  dest="start_server_side", help=argparse.SUPPRESS)
    serve_side_group.add_argument("--start-server-from-far-side", dest="start_server_side",
                                  action="store_const", const="far", help=argparse.SUPPRESS)
    serve_side_group.add_argument("--start-server-from-near-side", dest="start_server_side",
                                  action="store_const", const="near", help=argparse.SUPPRESS)
    parser.set_defaults(disable_false_points=True)
    args = parser.parse_args()
    globals()['_verbose_debug_enabled'] = bool(args.verbose_debug)

    def configured_video_sequence(start_key):
        night_keys = ["night", "night2", "night3"]
        if start_key in night_keys:
            start_index = night_keys.index(start_key)
            return night_keys[start_index:]
        return [start_key]

    def run_analyzer():
        global _cli_native_stderr_state
        native_stderr_state = _redirect_native_stderr_to_devnull()
        audit_needs_review = False
        try:
            sequence = configured_video_sequence(args.court) if args.continue_next_video else [args.court]
            max_frames_remaining = int(args.max_frames or 0)
            for sequence_index, court_key in enumerate(sequence):
                court = COURT_CONFIGS[court_key]
                start_frame = args.start_frame if sequence_index == 0 else 0
                max_frames_for_run = max_frames_remaining if max_frames_remaining > 0 else 0
                if sequence_index > 0:
                    print(
                        f"[VIDEO_CONTINUE] Opening next configured video: "
                        f"{court['label']} ({court['video']})"
                    )
                print(f"[COURT] {court['label']}")
                print(f"[SERVE_AREA] First two games start from {args.start_server_side} side")
                if args.start_score is not None and sequence_index == 0:
                    print(f"[MATCH_SEED] Requested explicit starting score: {args.start_score}")
                if args.disable_false_points:
                    print("[FALSE_POINT] Debug false-point masking disabled")
                analyzer = InteractiveBallAnalyzer(court["video"], start_frame=start_frame,
                                                   config_file=court["config"], headless=args.headless,
                                                   disable_false_points=args.disable_false_points,
                                                   point_history_file=args.point_history_file,
                                                   write_point_history=not args.no_point_history,
                                                   start_server_side=args.start_server_side,
                                                   start_score=args.start_score if sequence_index == 0 else None,
                                                   enable_player_tracking=not args.disable_player_tracking,
                                                   player_tracking_interval=args.player_tracking_interval,
                                                   enable_player_learning=not args.disable_player_learning,
                                                   enable_player_ball_protection=args.enable_player_ball_protection,
                                                   ball_dataset_dir=args.export_ball_dataset,
                                                   local_ai_model=args.local_ai_model,
                                                   local_ai_python=args.local_ai_python,
                                                   local_ai_recovery_dir=args.local_ai_recovery_dir,
                                                   debug_local_ai_range=args.debug_local_ai_range,
                                                   debug_local_ai_radius=args.debug_local_ai_radius)
                analyzer.pause_at_frame = args.pause_at_frame if sequence_index == 0 else None
                analyzer._debug_local_ai_top_n = args.debug_local_ai_top_n
                analyzer._debug_hsv_sweep = args.debug_hsv_sweep
                analyzer._force_local_ai_range = args.force_local_ai_range
                analyzer._force_local_ai_radius = max(40.0, float(args.force_local_ai_radius))
                analyzer._force_local_ai_min_score = max(0.0, min(1.0, float(args.force_local_ai_min_score)))
                if args.force_local_ai_range is not None:
                    print(
                        f"[FORCE_LOCAL_AI_MODE] range={args.force_local_ai_range[0]}:"
                        f"{args.force_local_ai_range[1]} radius={analyzer._force_local_ai_radius:.0f}px "
                        f"min_score={analyzer._force_local_ai_min_score:.3f}"
                    )
                result = analyzer.process_video(auto_play=args.auto_play, max_frames=max_frames_for_run)
                if args.audit_points:
                    if args.no_point_history or not analyzer.point_history_file:
                        print("[POINT_AUDIT_SKIPPED] Point history is disabled")
                    elif analyzer._point_history_point_index <= 0:
                        print("[POINT_AUDIT_SKIPPED] No completed points were recorded")
                    else:
                        try:
                            from point_visual_auditor import PointVisualAuditAgent

                            history_stem = os.path.splitext(os.path.basename(analyzer.point_history_file))[0]
                            audit_dir = os.path.join(args.audit_output_dir, history_stem)
                            audit_agent = PointVisualAuditAgent(
                                video_path=court["video"],
                                point_history_path=analyzer.point_history_file,
                                output_dir=audit_dir,
                                config_path=court["config"],
                                provider=args.audit_provider,
                                model=args.audit_model,
                                tracking_traces=analyzer._point_audit_traces,
                            )
                            audit_summary = audit_agent.run()
                            if audit_summary.get("needs_review", 0):
                                audit_needs_review = True
                        except Exception as audit_error:
                            audit_needs_review = True
                            print(f"[POINT_AUDIT_ERROR] {audit_error}")
                processed_frames = int((result or {}).get("processed_frames", 0))
                if max_frames_remaining > 0:
                    max_frames_remaining = max(0, max_frames_remaining - processed_frames)
                    if max_frames_remaining <= 0:
                        break
                if not args.continue_next_video:
                    break
                if (result or {}).get("reason") != "VIDEO_READ_FAILURE":
                    break
                if sequence_index >= len(sequence) - 1:
                    print("[VIDEO_SEQUENCE_END] No more configured videos to continue with.")
                    break
            if args.audit_fail_on_review and audit_needs_review:
                raise SystemExit(2)
        except BaseException:
            _restore_native_stderr(native_stderr_state)
            raise
        else:
            _cli_native_stderr_state = native_stderr_state

    if args.quiet:
        with contextlib.redirect_stdout(_QuietTrackerOutput(sys.stdout)):
            run_analyzer()
    else:
        run_analyzer()
