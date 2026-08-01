"""Export tracker-labelled tennis-ball frames for local model training.

The current tracker is used as a pseudo-labeller: it decides a ball centre and
this module saves the undecorated source frame plus that decision.  Labels are
intentionally marked as pseudo labels.  A later visual-review pass should
promote only verified examples to a training split.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np


class BallDatasetExporter:
    """Write ball-labelled image frames and JSONL metadata for one tracker run."""

    SCHEMA_VERSION = 1

    def __init__(self, output_root: str, video_path: str) -> None:
        root = Path(output_root).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_path).stem).strip("._")
        self.video_id = video_stem or "video"
        self.run_id = f"{timestamp}_{self.video_id}"
        self.run_dir = root / self.run_id
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.run_dir / "manifest.jsonl"
        self.progress_path = self.run_dir / "progress.json"
        self.summary_path = self.run_dir / "dataset_summary.json"
        self.video_path = str(Path(video_path).resolve())
        self._manifest = self.manifest_path.open("w", encoding="utf-8", buffering=1)
        self._written_frames: set[int] = set()
        self.saved_count = 0
        self.held_count = 0
        self.failed_count = 0
        self.first_frame: Optional[int] = None
        self.last_frame: Optional[int] = None
        self._closed = False
        self._write_progress(status="running")

    @staticmethod
    def _radius_from_area(area: Optional[float]) -> Optional[float]:
        """Approximate a circle radius only when the HSV contour area is usable."""
        if area is None:
            return None
        try:
            numeric_area = float(area)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric_area) or numeric_area <= 0:
            return None
        # The tracker occasionally retains a large blob area.  Keep the hint
        # bounded; the centre remains the authoritative pseudo label.
        return round(float(np.clip(math.sqrt(numeric_area / math.pi), 2.0, 96.0)), 3)

    def _write_progress(self, *, status: str, current_frame: Optional[int] = None) -> None:
        progress = {
            "schema_version": self.SCHEMA_VERSION,
            "status": status,
            "run_id": self.run_id,
            "video": self.video_path,
            "frames_saved": self.saved_count,
            "held_position_frames": self.held_count,
            "image_write_failures": self.failed_count,
            "first_source_frame": self.first_frame,
            "last_source_frame": self.last_frame,
            "current_source_frame": current_frame if current_frame is not None else self.last_frame,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def write_frame(
        self,
        frame: np.ndarray,
        *,
        source_frame: int,
        ball_center: Sequence[float],
        ball_area: Optional[float],
        stuck_frames: int,
        motion_distance: Optional[float],
        tracking_active: bool,
        point_index: Optional[int],
    ) -> bool:
        """Save one undecorated source frame and its tracker pseudo label.

        ``stuck_frames`` is recorded rather than discarded: it lets training
        code include only ``label_status == 'tracked'`` examples initially.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed BallDatasetExporter")
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.failed_count += 1
            self._write_progress(status="running", current_frame=int(source_frame))
            return False

        frame_number = int(source_frame)
        if frame_number in self._written_frames:
            return False
        if len(ball_center) != 2:
            raise ValueError("ball_center must contain exactly two coordinates")

        x = int(round(float(ball_center[0])))
        y = int(round(float(ball_center[1])))
        height, width = frame.shape[:2]
        filename = f"{self.video_id}_f{frame_number:08d}.jpg"
        image_path = self.images_dir / filename
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            self.failed_count += 1
            self._write_progress(status="running", current_frame=frame_number)
            return False

        radius = self._radius_from_area(ball_area)
        label_status = "tracked" if int(stuck_frames) == 0 else "held_previous_position"
        if label_status != "tracked":
            self.held_count += 1
        ball: dict[str, Any] = {
            "center": [x, y],
            "area": float(ball_area) if ball_area is not None else None,
            "radius_hint": radius,
        }
        if radius is not None:
            ball["bbox_xyxy_hint"] = [
                max(0, int(round(x - radius))),
                max(0, int(round(y - radius))),
                min(width - 1, int(round(x + radius))),
                min(height - 1, int(round(y + radius))),
            ]
        row = {
            "schema_version": self.SCHEMA_VERSION,
            "image": image_path.relative_to(self.run_dir).as_posix(),
            "video": self.video_path,
            "video_id": self.video_id,
            "source_frame": frame_number,
            "image_width": int(width),
            "image_height": int(height),
            "ball": ball,
            "label_source": "tracker_accepted_position",
            "label_status": label_status,
            "tracking_active": bool(tracking_active),
            "stuck_frames": int(stuck_frames),
            "motion_distance_px": float(motion_distance) if motion_distance is not None else None,
            "point_index": int(point_index) if point_index is not None else None,
        }
        self._manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._written_frames.add(frame_number)
        self.saved_count += 1
        self.first_frame = frame_number if self.first_frame is None else min(self.first_frame, frame_number)
        self.last_frame = frame_number if self.last_frame is None else max(self.last_frame, frame_number)
        if self.saved_count % 25 == 0:
            self._write_progress(status="running", current_frame=frame_number)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._manifest.close()
        self._closed = True
        self._write_progress(status="completed")
        summary = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "video": self.video_path,
            "manifest": self.manifest_path.name,
            "images_dir": self.images_dir.relative_to(self.run_dir).as_posix(),
            "frames_saved": self.saved_count,
            "held_position_frames": self.held_count,
            "image_write_failures": self.failed_count,
            "first_source_frame": self.first_frame,
            "last_source_frame": self.last_frame,
            "label_note": (
                "All labels are tracker pseudo labels. Start local-model training with "
                "label_status=tracked only, then add visually verified examples."
            ),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
