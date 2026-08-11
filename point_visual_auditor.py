import argparse
import base64
import csv
import html
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np


# Follow the serve far enough past toss/contact to expose a racket or player-body
# takeover during the first airborne handoff.
START_OFFSETS = (-6, -2, 0, 4, 8, 14, 22, 32)
END_OFFSETS = (-14, -8, -4, -1, 0, 2, 6, 12)
VISION_VERDICTS = (
    "correct",
    "wrong_start",
    "wrong_end",
    "wrong_reason",
    "uncertain",
)


def parse_point(value):
    if value is None:
        return None
    match = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def slugify(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "audit"


class PointVisualAuditAgent:
    def __init__(
        self,
        video_path,
        point_history_path,
        output_dir=None,
        config_path=None,
        provider="auto",
        model="gpt-5.4-mini",
        from_frame=None,
        to_frame=None,
        point_indices=None,
        max_points=0,
        tracking_traces=None,
    ):
        self.video_path = Path(video_path)
        self.point_history_path = Path(point_history_path)
        self.config_path = Path(config_path) if config_path else None
        self.provider = provider
        self.model = model
        self.from_frame = from_frame
        self.to_frame = to_frame
        self.point_indices = set(point_indices or [])
        self.max_points = max(0, int(max_points or 0))
        self.tracking_traces = self._normalize_tracking_traces(tracking_traces)
        default_output = Path("tmp") / "point_audits" / self.point_history_path.stem
        self.output_dir = Path(output_dir) if output_dir else default_output
        self.images_dir = self.output_dir / "images"
        self.config = self._load_config()
        self.hsv_filters = self._load_hsv_filters()
        self.frames = {}
        self.frame_width = 0
        self.frame_height = 0
        # Keeping every requested 4K source frame makes a whole-video audit
        # consume many GB of RAM.  Review sheets are only 640px wide, so keep
        # an audit-resolution copy and map reported full-resolution points to
        # it for image operations.  Geometry checks continue to use the
        # original coordinate system.
        self.frame_scale = 1.0
        # 640px is sufficient for the generated contact-sheet tiles and keeps
        # a 36-point, 4K audit below roughly 0.5GB of decoded-frame memory.
        self.max_cached_dimension = 640
        self.fps = 0.0

    @staticmethod
    def _normalize_tracking_traces(tracking_traces):
        normalized = {}
        for point_index, samples in (tracking_traces or {}).items():
            by_frame = {}
            for sample in samples or []:
                frame_number = parse_int(sample.get("frame"), -1)
                pos = sample.get("pos")
                if frame_number < 0 or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                    continue
                by_frame[frame_number] = {
                    "pos": (int(pos[0]), int(pos[1])),
                    "size": sample.get("size"),
                    "stuck": parse_int(sample.get("stuck"), 0),
                }
            normalized[parse_int(point_index, 0)] = by_frame
        return normalized

    def _load_config(self):
        if self.config_path is None or not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_hsv_filters(self):
        filters = []
        for name, value in self.config.items():
            if not isinstance(value, dict):
                continue
            required = {"h_min", "h_max", "s_min", "s_max", "v_min", "v_max"}
            if not required.issubset(value):
                continue
            lower = np.array([value["h_min"], value["s_min"], value["v_min"]], dtype=np.uint8)
            upper = np.array([value["h_max"], value["s_max"], value["v_max"]], dtype=np.uint8)
            filters.append((name, lower, upper))
        return filters

    def _read_rows(self):
        if not self.point_history_path.exists():
            raise FileNotFoundError(f"Point history not found: {self.point_history_path}")
        with self.point_history_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))

        selected = []
        for row_number, row in enumerate(rows, start=1):
            point_index = parse_int(row.get("point_index"), row_number)
            start_frame = parse_int(row.get("serve_start_frame"))
            end_frame = parse_int(row.get("point_end_frame"))
            if start_frame is None or end_frame is None:
                continue
            if self.point_indices and point_index not in self.point_indices:
                continue
            if self.from_frame is not None and end_frame < self.from_frame:
                continue
            if self.to_frame is not None and start_frame > self.to_frame:
                continue
            normalized = dict(row)
            normalized["_point_index"] = point_index
            normalized["_start_frame"] = start_frame
            normalized["_end_frame"] = end_frame
            selected.append(normalized)
            if self.max_points and len(selected) >= self.max_points:
                break
        return selected

    @staticmethod
    def _phase_frames(center_frame, offsets):
        return [max(0, center_frame + offset) for offset in offsets]

    def _decode_frames(self, required_frames):
        required = sorted(set(int(frame) for frame in required_frames if frame is not None and frame >= 0))
        if not required:
            return
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Some of the night-session HEVC recordings become unreliable after a
        # long sequential decode from an arbitrary seek point.  Endpoint
        # auditing needs only small temporal windows, so split widely-spaced
        # requests and reopen the decoder for each window.
        windows = []
        current = [required[0]]
        for frame_number in required[1:]:
            if frame_number - current[-1] > 64:
                windows.append(current)
                current = [frame_number]
            else:
                current.append(frame_number)
        windows.append(current)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        largest_dimension = max(self.frame_width, self.frame_height, 1)
        self.frame_scale = min(1.0, float(self.max_cached_dimension) / float(largest_dimension))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()

        for window in windows:
            first_frame = window[0]
            last_frame = window[-1]
            required_set = set(window)
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
            frame_index = first_frame
            consecutive_failures = 0
            while frame_index <= last_frame:
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= 12:
                        break
                    frame_index += 1
                    continue
                consecutive_failures = 0
                if frame_index in required_set:
                    if self.frame_scale < 1.0:
                        frame = cv2.resize(
                            frame,
                            None,
                            fx=self.frame_scale,
                            fy=self.frame_scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    self.frames[frame_index] = frame.copy()
                frame_index += 1
            cap.release()

    def _frame_point(self, point):
        """Map an original-video point into the cached audit frame."""
        if point is None:
            return None
        return (
            int(round(float(point[0]) * self.frame_scale)),
            int(round(float(point[1]) * self.frame_scale)),
        )

    @staticmethod
    def _draw_marker(frame, point, color=(0, 0, 255)):
        if point is None:
            return
        x, y = point
        radius = max(18, int(min(frame.shape[:2]) * 0.018))
        thickness = max(2, radius // 8)
        cv2.circle(frame, (x, y), radius, color, thickness)
        cv2.line(frame, (x - radius - 10, y), (x + radius + 10, y), color, thickness)
        cv2.line(frame, (x, y - radius - 10), (x, y + radius + 10), color, thickness)

    @staticmethod
    def _resize_cover(frame, width, height):
        scale = max(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        x0 = max(0, (resized.shape[1] - width) // 2)
        y0 = max(0, (resized.shape[0] - height) // 2)
        return resized[y0:y0 + height, x0:x0 + width]

    def _render_sheet(
        self, frame_numbers, center_frame, reported_point, phase, output_path, tracked_points=None
    ):
        tile_width = 640
        tile_height = 390
        image_height = 350
        tiles = []
        for frame_number in frame_numbers:
            source = self.frames.get(frame_number)
            if source is None:
                tile = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
                cv2.putText(
                    tile,
                    f"f{frame_number} unavailable",
                    (24, tile_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
                continue

            display = source.copy()
            is_center = frame_number == center_frame
            tracked_point = self._frame_point(
                (tracked_points or {}).get(frame_number, {}).get("pos")
            )
            display_point = self._frame_point(reported_point)
            if tracked_point is not None:
                self._draw_marker(display, tracked_point, color=(0, 255, 0))
            if is_center:
                self._draw_marker(display, display_point)
            image = self._resize_cover(display, tile_width, image_height)

            crop_point = tracked_point or display_point
            if crop_point is not None:
                px, py = crop_point
                crop_radius = max(100, int(min(source.shape[:2]) * 0.08))
                x1 = max(0, px - crop_radius)
                y1 = max(0, py - crop_radius)
                x2 = min(source.shape[1], px + crop_radius)
                y2 = min(source.shape[0], py + crop_radius)
                crop = source[y1:y2, x1:x2].copy()
                if crop.size:
                    marker_color = (0, 255, 0) if tracked_point is not None else (255, 0, 255)
                    self._draw_marker(crop, (px - x1, py - y1), color=marker_color)
                    zoom = self._resize_cover(crop, 190, 126)
                    zx = tile_width - zoom.shape[1] - 8
                    zy = image_height - zoom.shape[0] - 8
                    cv2.rectangle(image, (zx - 3, zy - 3), (tile_width - 5, image_height - 5), (255, 255, 255), 2)
                    image[zy:zy + zoom.shape[0], zx:zx + zoom.shape[1]] = zoom

            tile = np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
            tile[:image_height] = image
            marker_text = " TRACK MARKER" if tracked_point is not None else ""
            center_text = " + REPORTED CENTER" if is_center else ""
            label = f"{phase.upper()}  f{frame_number}{marker_text}{center_text}"
            cv2.putText(
                tile,
                label,
                (12, tile_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (90, 210, 255) if is_center else (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        while len(tiles) < 8:
            tiles.append(np.full((tile_height, tile_width, 3), 18, dtype=np.uint8))
        sheet = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:8])))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91])

    def _clip_patch(self, frame, point, radius):
        if frame is None or point is None:
            return None
        x, y = self._frame_point(point)
        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            return None
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(frame.shape[1], x + radius + 1)
        y2 = min(frame.shape[0], y + radius + 1)
        patch = frame[y1:y2, x1:x2]
        return patch if patch.size else None

    def _hsv_match_fraction(self, patch):
        if patch is None or not self.hsv_filters:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for _, lower, upper in self.hsv_filters:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
        return float(np.count_nonzero(combined)) / float(combined.size)

    def _phase_metrics(self, center_frame, point, offsets):
        center = self.frames.get(center_frame)
        metrics = {
            "frame_available": center is not None,
            "position_available": point is not None,
            "position_in_frame": False,
            "motion_mean": None,
            "motion_max": None,
            "hsv_match_fraction": None,
            "center_hsv": None,
        }
        if center is None or point is None:
            return metrics
        # Decoded audit frames may be downscaled to control memory use, while
        # point-history coordinates are always in the original video space.
        # Render/crop helpers already convert through _frame_point(); metrics
        # must do the same before checking bounds or sampling HSV.
        x, y = self._frame_point(point)
        metrics["position_in_frame"] = bool(0 <= x < center.shape[1] and 0 <= y < center.shape[0])
        if not metrics["position_in_frame"]:
            return metrics

        radius = max(24, int(min(center.shape[:2]) * 0.018))
        center_patch = self._clip_patch(center, point, radius)
        metrics["hsv_match_fraction"] = self._hsv_match_fraction(center_patch)
        center_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)[y, x]
        metrics["center_hsv"] = [int(value) for value in center_hsv]

        diffs = []
        for offset in offsets:
            if offset == 0:
                continue
            comparison = self.frames.get(max(0, center_frame + offset))
            comparison_patch = self._clip_patch(comparison, point, radius)
            if comparison_patch is None or center_patch is None:
                continue
            if comparison_patch.shape != center_patch.shape:
                continue
            gray_center = cv2.cvtColor(center_patch, cv2.COLOR_BGR2GRAY)
            gray_comparison = cv2.cvtColor(comparison_patch, cv2.COLOR_BGR2GRAY)
            diffs.append(cv2.absdiff(gray_center, gray_comparison))
        if diffs:
            stacked = np.stack(diffs)
            metrics["motion_mean"] = round(float(np.mean(stacked)), 2)
            metrics["motion_max"] = int(np.max(stacked))
        return metrics

    def _point_in_polygon(self, point, key):
        values = self.config.get(key)
        if point is None or not isinstance(values, list) or len(values) < 3:
            return None
        polygon = np.asarray(values, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def _net_distance(self, point):
        values = self.config.get("net_area_points")
        if point is None or not isinstance(values, list) or len(values) < 3:
            return None
        polygon = np.asarray(values, dtype=np.float32)
        return abs(float(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True)))

    def _serve_handoff_metrics(self, start_frame, start_point, tracked_points):
        metrics = {
            "available": False,
            "sample_count": 0,
            "contact_frame": None,
            "contact_y": None,
            "post_contact_min_y": None,
            "launch_progress": None,
            "required_progress": max(80.0, self.frame_height * 0.04),
        }
        if start_point is None or self.frame_height <= 0 or start_point[1] < self.frame_height * 0.35:
            return metrics

        samples = sorted(
            (frame_number, value["pos"])
            for frame_number, value in (tracked_points or {}).items()
            if start_frame <= frame_number <= start_frame + 32
        )
        metrics["sample_count"] = len(samples)
        if len(samples) < 6:
            return metrics

        contact_window = [sample for sample in samples if sample[0] <= start_frame + 20]
        if len(contact_window) < 4:
            return metrics
        apex_index = min(range(len(contact_window)), key=lambda index: contact_window[index][1][1])
        after_apex = contact_window[apex_index:]
        contact_frame, contact_pos = max(after_apex, key=lambda sample: sample[1][1])
        apex_y = contact_window[apex_index][1][1]
        if contact_pos[1] - apex_y < 30:
            return metrics

        post_contact = [sample for sample in samples if sample[0] > contact_frame]
        if len(post_contact) < 2:
            return metrics
        post_min_y = min(sample[1][1] for sample in post_contact)
        metrics.update({
            "available": True,
            "contact_frame": contact_frame,
            "contact_y": contact_pos[1],
            "post_contact_min_y": post_min_y,
            "launch_progress": float(contact_pos[1] - post_min_y),
        })
        return metrics

    def _local_review(self, row, start_metrics, end_metrics, handoff_metrics=None):
        flags = []
        start_point = parse_point(row.get("start_position"))
        end_point = parse_point(row.get("end_position"))
        reason = (row.get("end_reason") or "").strip()
        reason_lower = reason.lower()
        duration = parse_int(row.get("duration_frames"), 0)

        for phase, point, metrics in (
            ("start", start_point, start_metrics),
            ("end", end_point, end_metrics),
        ):
            if point is None:
                flags.append(f"{phase} position is missing")
            elif not metrics["frame_available"]:
                flags.append(f"{phase} frame could not be decoded")
            elif not metrics["position_in_frame"]:
                flags.append(f"{phase} position is outside the image")

        serve_in_near = self._point_in_polygon(start_point, "serve_area_points")
        serve_in_far = self._point_in_polygon(start_point, "far_serve_area_points")
        if serve_in_near is False and serve_in_far is False:
            flags.append("reported start is outside both configured serve areas")

        # A serve that clips the net can legitimately end within a handful of
        # frames.  Short duration is useful as a review hint for unknown/rally
        # points, but is not evidence of an error when the reason explicitly
        # says the ball hit the net.
        if duration and duration < 12 and "net" not in reason_lower:
            flags.append(f"very short point duration ({duration} frames)")
        if handoff_metrics and handoff_metrics.get("available"):
            progress = float(handoff_metrics.get("launch_progress") or 0.0)
            required = float(handoff_metrics.get("required_progress") or 0.0)
            if progress < required:
                flags.append(
                    f"serve handoff did not launch toward court after contact "
                    f"({progress:.0f}px < {required:.0f}px)"
                )
        if any(token in reason_lower for token in ("timeout", "read_failure", "media_error")):
            flags.append(f"unresolved end reason: {reason}")

        # The temporal window is intentionally sampled *after* the reported
        # end.  Once a point is over, players commonly stop or reset, so low
        # image motion is expected for explicit terminal outcomes.  Keep this
        # warning for unresolved/unknown endings where a stale marker is still
        # a meaningful failure signal.
        explicit_terminal_reason = any(token in reason_lower for token in (
            "hit the net", "bounced", "bounce", "stopped", "out of court",
            "lost", "double fault", "first serve fault", "let",
        ))
        end_motion_mean = end_metrics.get("motion_mean")
        end_motion_max = end_metrics.get("motion_max")
        if (not explicit_terminal_reason and
                end_motion_mean is not None and end_motion_max is not None and
                end_motion_mean < 1.2 and end_motion_max < 18):
            flags.append("reported end marker has almost no temporal image change")

        compound_net_landing = (
            "net" in reason_lower and
            any(token in reason_lower for token in ("bounce", "outside", "out of court"))
        )
        if "net" in reason_lower and not compound_net_landing:
            net_distance = self._net_distance(end_point)
            if net_distance is not None:
                tolerance = max(85.0, self.frame_height * 0.05)
                if net_distance > tolerance:
                    flags.append(
                        f"net reason conflicts with geometry (marker is {net_distance:.0f}px from net)"
                    )

        hsv_fraction = end_metrics.get("hsv_match_fraction")
        # HSV can disappear immediately after a net/bounce/out event because
        # the ball is occluded by the net, leaves the calibrated court, or is
        # picked up.  Treat that as expected for explicit terminal reasons;
        # unresolved endings still require review.
        if (not explicit_terminal_reason and
                hsv_fraction is not None and hsv_fraction < 0.002):
            flags.append("almost no configured ball-color pixels near the end marker")

        verdict = "likely_correct" if not flags else "needs_review"
        return verdict, flags

    @staticmethod
    def _image_data_url(path):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _vision_client(self):
        if self.provider == "local":
            return None
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        if not has_key:
            if self.provider == "openai":
                raise RuntimeError("OPENAI_API_KEY is required for --provider openai")
            return None
        try:
            from openai import OpenAI
        except ImportError:
            if self.provider == "openai":
                raise RuntimeError(
                    "The optional OpenAI package is required. Install requirements-vision.txt."
                )
            return None
        return OpenAI()

    def _vision_review(self, client, row, start_sheet, end_sheet, local_flags):
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": list(VISION_VERDICTS)},
                "start_valid": {"type": "boolean"},
                "end_valid": {"type": "boolean"},
                "reason_valid": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "observed_start_frame": {"type": ["integer", "null"]},
                "observed_end_frame": {"type": ["integer", "null"]},
                "observed_end_reason": {"type": "string"},
                "summary": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "verdict",
                "start_valid",
                "end_valid",
                "reason_valid",
                "confidence",
                "observed_start_frame",
                "observed_end_frame",
                "observed_end_reason",
                "summary",
                "evidence",
            ],
            "additionalProperties": False,
        }
        metadata = {
            key: row.get(key, "")
            for key in (
                "point_index",
                "serve_start_frame",
                "point_end_frame",
                "duration_frames",
                "start_position",
                "end_position",
                "serve_attempt",
                "rally_shots",
                "point_awarded",
                "winner",
                "end_reason",
                "why",
                "category",
                "current_score",
            )
        }
        prompt = (
            "You are auditing a fixed-camera tennis tracking result. Compare the script metadata "
            "with the two temporal contact sheets. A green cross marks the script's tracked position "
            "where trace data is available; the red cross marks the reported CENTER position. The "
            "inset follows the green tracked position when present. "
            "First decide whether START shows a genuine serve toss/contact that begins a point, not "
            "walking, ball retrieval, a player body, or a static artifact. Verify that the green path "
            "continues on the visible airborne ball after racket contact and does not switch to the "
            "racket, hand, player body, or a fixed court point. Then decide whether END "
            "shows the true last ball event and whether the stated reason (net, out, double bounce, "
            "serve fault, timeout, etc.) agrees with the visible sequence. Follow the small tennis "
            "ball across adjacent frames; do not assume the red marker is correct. If the ball is too "
            "small or occluded, return uncertain. Do not invent a winner when images do not establish it.\n\n"
            f"SCRIPT RESULT:\n{json.dumps(metadata, indent=2)}\n\n"
            f"LOCAL FLAGS:\n{json.dumps(local_flags, indent=2)}"
        )
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": self._image_data_url(start_sheet),
                            "detail": "high",
                        },
                        {
                            "type": "input_image",
                            "image_url": self._image_data_url(end_sheet),
                            "detail": "high",
                        },
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "tennis_point_visual_audit",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        return json.loads(response.output_text)

    def _write_json(self, results, summary):
        path = self.output_dir / "audit_results.json"
        payload = {
            "video": str(self.video_path),
            "point_history": str(self.point_history_path),
            "config": str(self.config_path) if self.config_path else None,
            "provider": summary["provider"],
            "model": self.model if summary["provider"] == "openai" else None,
            "summary": summary,
            "points": results,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_csv(self, results):
        path = self.output_dir / "audit_results.csv"
        headers = [
            "point_index",
            "serve_start_frame",
            "point_end_frame",
            "script_end_reason",
            "local_verdict",
            "final_verdict",
            "vision_confidence",
            "observed_end_frame",
            "observed_end_reason",
            "flags",
            "summary",
            "start_sheet",
            "end_sheet",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for result in results:
                vision = result.get("vision") or {}
                writer.writerow({
                    "point_index": result["point_index"],
                    "serve_start_frame": result["serve_start_frame"],
                    "point_end_frame": result["point_end_frame"],
                    "script_end_reason": result["script_end_reason"],
                    "local_verdict": result["local_verdict"],
                    "final_verdict": result["final_verdict"],
                    "vision_confidence": vision.get("confidence", ""),
                    "observed_end_frame": vision.get("observed_end_frame", ""),
                    "observed_end_reason": vision.get("observed_end_reason", ""),
                    "flags": " | ".join(result.get("flags", [])),
                    "summary": vision.get("summary", result.get("summary", "")),
                    "start_sheet": result["start_sheet"],
                    "end_sheet": result["end_sheet"],
                })
        return path

    def _write_markdown(self, results, summary):
        path = self.output_dir / "audit_report.md"
        lines = [
            "# Point Visual Audit",
            "",
            f"- Video: `{self.video_path}`",
            f"- Point history: `{self.point_history_path}`",
            f"- Provider: `{summary['provider']}`",
            f"- Points checked: `{summary['points_checked']}`",
            f"- Needs review: `{summary['needs_review']}`",
            "",
        ]
        for result in results:
            lines.extend([
                f"## Point {result['point_index']}: {result['final_verdict']}",
                "",
                f"Script: f{result['serve_start_frame']} -> f{result['point_end_frame']} "
                f"`{result['script_end_reason']}`",
                "",
                f"![Start]({result['start_sheet']})",
                "",
                f"![End]({result['end_sheet']})",
                "",
            ])
            if result.get("flags"):
                lines.append("Local flags:")
                lines.extend(f"- {flag}" for flag in result["flags"])
                lines.append("")
            vision = result.get("vision")
            if vision:
                lines.extend([
                    f"Vision summary: {vision.get('summary', '')}",
                    "",
                    f"Observed end: f{vision.get('observed_end_frame')} "
                    f"`{vision.get('observed_end_reason', '')}`",
                    "",
                ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_html(self, results, summary):
        path = self.output_dir / "index.html"
        rows = []
        for result in results:
            vision = result.get("vision") or {}
            flags = "".join(f"<li>{html.escape(flag)}</li>" for flag in result.get("flags", []))
            evidence = "".join(
                f"<li>{html.escape(str(item))}</li>" for item in vision.get("evidence", [])
            )
            rows.append(f"""
            <section class="point">
              <header>
                <h2>Point {result['point_index']}</h2>
                <strong class="verdict {html.escape(result['final_verdict'])}">{html.escape(result['final_verdict'])}</strong>
                <span>f{result['serve_start_frame']} to f{result['point_end_frame']}</span>
              </header>
              <p><b>Script:</b> {html.escape(result['script_end_reason'])}</p>
              <div class="sheets">
                <figure><img src="{html.escape(result['start_sheet'])}" alt="Start frames"><figcaption>Start sequence</figcaption></figure>
                <figure><img src="{html.escape(result['end_sheet'])}" alt="End frames"><figcaption>End sequence</figcaption></figure>
              </div>
              <div class="notes">
                <div><h3>Local checks</h3><ul>{flags or '<li>No local geometry/motion flags</li>'}</ul></div>
                <div><h3>Vision review</h3><p>{html.escape(vision.get('summary', result.get('summary', 'Not run')))}</p><ul>{evidence}</ul></div>
              </div>
            </section>
            """)
        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Point Visual Audit</title>
<style>
body {{ margin: 0; font: 15px Arial, sans-serif; color: #17202a; background: #f4f6f7; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
.summary {{ display: flex; gap: 28px; padding: 14px 0 24px; border-bottom: 2px solid #aeb6bf; }}
.point {{ padding: 24px 0; border-bottom: 1px solid #ccd1d1; }}
.point header {{ display: flex; align-items: baseline; gap: 18px; margin-bottom: 10px; }}
.verdict {{ text-transform: uppercase; }}
.correct, .likely_correct {{ color: #117864; }}
.needs_review, .wrong_start, .wrong_end, .wrong_reason {{ color: #b03a2e; }}
.uncertain, .insufficient_images {{ color: #9a7d0a; }}
.sheets {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
figure {{ margin: 0; min-width: 0; }}
img {{ width: 100%; height: auto; display: block; border: 1px solid #85929e; }}
figcaption {{ margin-top: 6px; color: #566573; }}
.notes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-top: 16px; }}
ul {{ margin: 6px 0 0; padding-left: 20px; }}
@media (max-width: 900px) {{ .sheets, .notes {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body><main>
<h1>Point Visual Audit</h1>
<div class="summary"><span>Provider: <b>{html.escape(summary['provider'])}</b></span><span>Checked: <b>{summary['points_checked']}</b></span><span>Review: <b>{summary['needs_review']}</b></span></div>
{''.join(rows)}
</main></body></html>"""
        path.write_text(document, encoding="utf-8")
        return path

    def run(self):
        rows = self._read_rows()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise RuntimeError("No point-history rows matched the requested frame/point filters")

        required_frames = []
        for row in rows:
            required_frames.extend(self._phase_frames(row["_start_frame"], START_OFFSETS))
            required_frames.extend(self._phase_frames(row["_end_frame"], END_OFFSETS))
        self._decode_frames(required_frames)
        client = self._vision_client()
        active_provider = "openai" if client is not None else "local"

        results = []
        for row in rows:
            point_index = row["_point_index"]
            start_frame = row["_start_frame"]
            end_frame = row["_end_frame"]
            start_point = parse_point(row.get("start_position"))
            end_point = parse_point(row.get("end_position"))
            start_numbers = self._phase_frames(start_frame, START_OFFSETS)
            end_numbers = self._phase_frames(end_frame, END_OFFSETS)
            start_sheet_path = self.images_dir / f"point_{point_index:03d}_start.jpg"
            end_sheet_path = self.images_dir / f"point_{point_index:03d}_end.jpg"
            tracked_points = self.tracking_traces.get(point_index, {})
            self._render_sheet(
                start_numbers, start_frame, start_point, "start", start_sheet_path, tracked_points
            )
            self._render_sheet(
                end_numbers, end_frame, end_point, "end", end_sheet_path, tracked_points
            )

            start_metrics = self._phase_metrics(start_frame, start_point, START_OFFSETS)
            end_metrics = self._phase_metrics(end_frame, end_point, END_OFFSETS)
            handoff_metrics = self._serve_handoff_metrics(start_frame, start_point, tracked_points)
            local_verdict, flags = self._local_review(
                row, start_metrics, end_metrics, handoff_metrics
            )
            relative_start = start_sheet_path.relative_to(self.output_dir).as_posix()
            relative_end = end_sheet_path.relative_to(self.output_dir).as_posix()
            result = {
                "point_index": point_index,
                "serve_start_frame": start_frame,
                "point_end_frame": end_frame,
                "script_end_reason": row.get("end_reason", ""),
                "script_winner": row.get("winner", ""),
                "script_score": row.get("current_score", ""),
                "local_verdict": local_verdict,
                "final_verdict": local_verdict,
                "flags": flags,
                "start_metrics": start_metrics,
                "end_metrics": end_metrics,
                "serve_handoff_metrics": handoff_metrics,
                "start_sheet": relative_start,
                "end_sheet": relative_end,
                "vision": None,
                "summary": "Local image, motion, HSV, and geometry checks only.",
            }
            if client is not None:
                try:
                    vision = self._vision_review(client, row, start_sheet_path, end_sheet_path, flags)
                    result["vision"] = vision
                    result["final_verdict"] = vision.get("verdict", "uncertain")
                except Exception as error:
                    result["flags"].append(f"vision review failed: {error}")
                    result["final_verdict"] = "needs_review"
                    result["summary"] = "Vision review failed; local evidence is preserved."
            results.append(result)
            print(
                f"[POINT_AUDIT] point={point_index} start={start_frame} end={end_frame} "
                f"verdict={result['final_verdict']} reason={row.get('end_reason', '')}"
            )

        verdict_counts = Counter(result["final_verdict"] for result in results)
        review_verdicts = {"needs_review", "wrong_start", "wrong_end", "wrong_reason", "uncertain", "insufficient_images"}
        summary = {
            "provider": active_provider,
            "points_checked": len(results),
            "needs_review": sum(result["final_verdict"] in review_verdicts for result in results),
            "verdict_counts": dict(verdict_counts),
            "decoded_frames": len(self.frames),
        }
        json_path = self._write_json(results, summary)
        csv_path = self._write_csv(results)
        markdown_path = self._write_markdown(results, summary)
        html_path = self._write_html(results, summary)
        summary.update({
            "json_report": str(json_path),
            "csv_report": str(csv_path),
            "markdown_report": str(markdown_path),
            "html_report": str(html_path),
        })
        print(
            f"[POINT_AUDIT_SUMMARY] checked={summary['points_checked']} "
            f"needs_review={summary['needs_review']} provider={active_provider} "
            f"report={html_path}"
        )
        return summary


def parse_point_indices(value):
    if not value:
        return []
    indices = []
    for item in value.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Audit tennis point starts/ends against temporal image contact sheets"
    )
    parser.add_argument("--video", required=True, help="Source tennis video")
    parser.add_argument("--history", required=True, help="Point-history CSV produced by the tracker")
    parser.add_argument("--config", help="Tracker HSV/court geometry JSON")
    parser.add_argument("--output-dir", help="Report directory (default: tmp/point_audits/<history>)")
    parser.add_argument("--provider", choices=("auto", "local", "openai"), default="auto")
    parser.add_argument("--model", default=os.environ.get("POINT_AUDIT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--from-frame", type=int)
    parser.add_argument("--to-frame", type=int)
    parser.add_argument("--points", help="Comma-separated point indices")
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--fail-on-review", action="store_true")
    args = parser.parse_args()

    agent = PointVisualAuditAgent(
        video_path=args.video,
        point_history_path=args.history,
        output_dir=args.output_dir,
        config_path=args.config,
        provider=args.provider,
        model=args.model,
        from_frame=args.from_frame,
        to_frame=args.to_frame,
        point_indices=parse_point_indices(args.points),
        max_points=args.max_points,
    )
    summary = agent.run()
    if args.fail_on_review and summary["needs_review"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
