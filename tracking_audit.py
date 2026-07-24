"""Automatic tracking-event audit for TennisGameTracker.

This is deliberately separate from the tracker.  It runs a bounded headless
regression, parses the track/loss diagnostics already emitted by
``interactive_ball_analyzer.py``, and produces small visual windows only for
frames that look suspicious.  The resulting JSON/HTML report is suitable for
local review or optional OpenAI vision review.
"""

import argparse
import base64
import csv
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np


COURTS = {
    "1": ("20251011124747503_FV3553362380_FV3553362.mp4", "hsv_config.json"),
    "2": ("20260322115648717_FV3553362360_FV3553362.mp4", "hsv_config_court2.json"),
    "night": (
        "04_Left_Court_tennis_20260623220627_20260623222559_1274169779.mp4",
        "hsv_config_04_left_night.json",
    ),
    "night2": (
        "04_Left_Court_tennis_20260623222559_20260623225255_1274247426.mp4",
        "hsv_config_04_left_night.json",
    ),
    "night3": (
        "04_Left_Court_tennis_20260623225255_20260623232026_1274349214.mp4",
        "hsv_config_04_left_night.json",
    ),
}

TRACK_RE = re.compile(
    r"\[TRACK\]\s+f(?P<frame>\d+):\s+pos=\((?P<x>-?\d+),\s*(?P<y>-?\d+)\)\s+"
    r"vel=(?P<vel>[\d.]+)px\s+stuck=(?P<stuck>\d+)"
)
EVENT_FRAME_PATTERNS = {
    "ball_loss": re.compile(r"\[BALL_LOSS_DIAGNOSTIC\]\s+f(?P<frame>\d+):"),
    "jump_rejected": re.compile(r"\[JUMP_REJECTED\]\s+f(?P<frame>\d+):"),
    "reacquired": re.compile(r"Frame\s+(?P<frame>\d+):\s+\[RE-ACQUIRED\]"),
    "player_occlusion": re.compile(r"Frame\s+(?P<frame>\d+):\s+\[PLAYER-OCCLUSION\]"),
    "player_reacq": re.compile(r"Frame\s+(?P<frame>\d+):\s+\[PLAYER-REACQ[^]]*\]"),
}

PROGRESS_FRAME_PATTERNS = (
    re.compile(r"\[TRACK\]\s+f(?P<frame>\d+):"),
    re.compile(r"\[BALL_LOSS_DIAGNOSTIC\]\s+f(?P<frame>\d+):"),
    re.compile(r"\[SKIP_BAD_FRAME\]\s+f(?P<frame>\d+):"),
    re.compile(r"\bFrame\s+(?P<frame>\d+):"),
    re.compile(r"\[VIDEO_END\].*\bf(?P<frame>\d+)"),
)


def _progress_bar(percent, width=36):
    percent = max(0.0, min(100.0, float(percent or 0.0)))
    filled = int(round(width * percent / 100.0))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {percent:5.1f}%"


class AuditProgress:
    """Write a small live status file that is safe to keep open in an editor."""

    def __init__(self, output_dir, json_path=None):
        self.output_dir = Path(output_dir)
        self.json_path = Path(json_path) if json_path else self.output_dir / "progress.json"
        self.text_path = self.json_path.with_suffix(".txt")
        self.state = {}

    def update(self, **changes):
        self.state.update(changes)
        self.state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        percent = float(self.state.get("overall_percent", 0.0) or 0.0)
        self.state["overall_percent"] = max(0.0, min(100.0, percent))
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

        current_frame = self.state.get("current_frame")
        end_frame = self.state.get("end_frame")
        frame_line = ""
        if current_frame is not None:
            frame_line = f"\nFrame: {current_frame}"
            if end_frame is not None:
                frame_line += f" / {end_frame}"
        event_line = ""
        if self.state.get("events_total") is not None:
            event_line = (
                f"\nVisual events: {self.state.get('events_completed', 0)}"
                f" / {self.state.get('events_total', 0)}"
            )
        text = (
            f"Tennis tracking audit: {str(self.state.get('status', 'unknown')).upper()}\n"
            f"Stage: {self.state.get('stage', 'starting')}\n"
            f"Progress: {_progress_bar(self.state['overall_percent'])}"
            f"{frame_line}{event_line}\n"
            f"Message: {self.state.get('message', '')}\n"
            f"Updated: {self.state['updated_at']}\n"
        )
        self.text_path.write_text(text, encoding="utf-8")


def _progress_frame_from_line(line):
    for pattern in PROGRESS_FRAME_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group("frame"))
    processed = re.search(r"Analysis complete! Processed\s+(?P<count>\d+)\s+frames", line)
    if processed:
        return int(processed.group("count")) - 1
    return None


def _video_end_frame(video_path, start_frame, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if cap.isOpened() else 0
    cap.release()
    if max_frames and max_frames > 0:
        requested_end = int(start_frame) + int(max_frames) - 1
        return min(requested_end, frame_count - 1) if frame_count > 0 else requested_end
    return frame_count - 1 if frame_count > 0 else None


def parse_track_log(text):
    """Return selected positions and explicit diagnostics from tracker output."""
    tracks = {}
    events = []
    for line in text.splitlines():
        match = TRACK_RE.search(line)
        if match:
            frame = int(match.group("frame"))
            tracks[frame] = {
                "frame": frame,
                "pos": [int(match.group("x")), int(match.group("y"))],
                "velocity": float(match.group("vel")),
                "stuck": int(match.group("stuck")),
            }
        for event_type, pattern in EVENT_FRAME_PATTERNS.items():
            match = pattern.search(line)
            if match:
                events.append({"frame": int(match.group("frame")), "type": event_type, "line": line.strip()})
    return tracks, events


def find_suspicious_events(tracks, explicit_events, frame_width=3840, jump_limit=420):
    """Flag temporal failures without making a point-end decision."""
    events = list(explicit_events)
    ordered = [tracks[key] for key in sorted(tracks)]
    for previous, current in zip(ordered, ordered[1:]):
        gap = current["frame"] - previous["frame"]
        dx = current["pos"][0] - previous["pos"][0]
        dy = current["pos"][1] - previous["pos"][1]
        jump = math.hypot(dx, dy)
        if gap > 1:
            events.append({
                "frame": current["frame"],
                "type": "trace_gap",
                "line": f"missing {gap - 1} tracked frames before f{current['frame']}",
            })
        if jump > max(jump_limit, previous["velocity"] * 5.0 + 120.0):
            events.append({
                "frame": current["frame"],
                "type": "large_jump",
                "line": f"jump={jump:.1f}px from {previous['pos']} to {current['pos']}",
            })
        if current["stuck"] >= 3:
            events.append({
                "frame": current["frame"],
                "type": "stuck",
                "line": f"stuck={current['stuck']} at {current['pos']}",
            })
    # Collapse the per-frame spam produced while a loss is being held.  Keep
    # the first diagnostic in each short interval; the contact sheet around it
    # still includes the following frames, while the report remains readable.
    events.sort(key=lambda item: (item["frame"], item["type"]))
    filtered = []
    last_by_type = {}
    repeat_spacing = {
        "ball_loss": 40,
        "stuck": 40,
        "player_reacq": 20,
        "reacquired": 20,
    }
    for event in events:
        previous_frame = last_by_type.get(event["type"])
        minimum_spacing = repeat_spacing.get(event["type"])
        if minimum_spacing is not None and previous_frame is not None and event["frame"] - previous_frame < minimum_spacing:
            continue
        filtered.append(event)
        last_by_type[event["type"]] = event["frame"]
    events = filtered
    merged = []
    for event in events:
        if (
                merged and
                event["frame"] - merged[-1]["end_frame"] <= 6 and
                event["frame"] - merged[-1]["start_frame"] <= 24
        ):
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], event["frame"])
            merged[-1]["types"] = sorted(set(merged[-1]["types"] + [event["type"]]))
            merged[-1]["evidence"].append(event["line"])
        else:
            merged.append({
                "start_frame": event["frame"],
                "end_frame": event["frame"],
                "center_frame": event["frame"],
                "types": [event["type"]],
                "evidence": [event["line"]],
            })
    for index, event in enumerate(merged, start=1):
        event["event_index"] = index
        event["severity"] = (
            "high" if any(t in event["types"] for t in ("player_occlusion", "player_reacq", "large_jump", "jump_rejected"))
            else "medium"
        )
    return merged


def _timestamped_history(base_path):
    base = Path(base_path)
    candidates = sorted(base.parent.glob(f"{base.stem}_*.csv"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else base


def _point_for_frame(history_rows, frame):
    for row in history_rows:
        try:
            if int(row.get("serve_start_frame", -1)) <= frame <= int(row.get("point_end_frame", -1)):
                return row.get("point_index", "")
        except (TypeError, ValueError):
            pass
    return ""


def _parse_position(value):
    """Parse the ``(x,y)`` positions stored in point-history CSV files."""
    if not value:
        return None
    match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", str(value))
    if not match:
        return None
    return [int(round(float(match.group(1)))), int(round(float(match.group(2))))]


def point_endpoints(history_rows):
    """Return usable point ends, preserving their CSV ordering.

    The tracker numbers decoded frames from one while OpenCV's frame index is
    zero based.  A history endpoint ``N`` is therefore rendered from raw video
    frame ``N - 1``.  Keeping both values in the audit output makes the offset
    explicit and prevents accidental direct seeking at an ambiguous frame.
    """
    endpoints = []
    for row_number, row in enumerate(history_rows, start=1):
        try:
            end_frame = int(row.get("point_end_frame", ""))
        except (TypeError, ValueError):
            continue
        if end_frame < 1:
            continue
        endpoints.append({
            "audit_index": len(endpoints) + 1,
            "row_number": row_number,
            "point_index": str(row.get("point_index", "")),
            "serve_start_frame": int(row.get("serve_start_frame", 0) or 0),
            "end_frame": end_frame,
            "raw_end_frame": end_frame - 1,
            "end_position": _parse_position(row.get("end_position")),
            "end_reason": str(row.get("end_reason", "")),
            "category": str(row.get("category", "")),
            "winner": str(row.get("winner", "")),
        })
    return endpoints


def _endpoint_contact_sheet(frames, endpoint, selected, before, after, output_path):
    """Render a clean sequentially-decoded endpoint window.

    Green is the live tracker marker; red is the point-history endpoint.  The
    latter may be a reviewed/reference endpoint, so a disagreement is evidence
    to inspect, not a reason to silently overwrite history.
    """
    first_raw = max(0, endpoint["raw_end_frame"] - before)
    last_raw = endpoint["raw_end_frame"] + after
    numbers = list(range(first_raw, last_raw + 1))
    tiles = []
    tile_w, image_h, tile_h = 480, 275, 340
    for raw_frame in numbers:
        source = frames.get(raw_frame)
        tile = np.full((tile_h, tile_w, 3), 20, dtype=np.uint8)
        tracker_frame = raw_frame + 1
        if source is not None:
            display = source.copy()
            _marker(display, (selected.get(tracker_frame) or {}).get("pos"), (0, 255, 0))
            if tracker_frame == endpoint["end_frame"]:
                _marker(display, endpoint.get("end_position"), (0, 0, 255))
            scale = max(tile_w / display.shape[1], image_h / display.shape[0])
            resized = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            x0 = max(0, (resized.shape[1] - tile_w) // 2)
            y0 = max(0, (resized.shape[0] - image_h) // 2)
            tile[:image_h] = resized[y0:y0 + image_h, x0:x0 + tile_w]
        label = f"tracker f{tracker_frame} / raw f{raw_frame}"
        cv2.putText(tile, label, (10, image_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        if tracker_frame == endpoint["end_frame"]:
            cv2.putText(tile, "POINT END  green=track red=history", (10, image_h + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 180, 255), 2)
        elif selected.get(tracker_frame, {}).get("stuck", 0) >= 3:
            cv2.putText(tile, f"stuck={selected[tracker_frame]['stuck']}", (10, image_h + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
        tiles.append(tile)
    columns = 3
    rows = int(math.ceil(len(tiles) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 12, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * tile_h
        x = (index % columns) * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = tile
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _endpoint_zoom_sheet(frames, endpoint, selected, before, after, output_path, crop_size=720):
    """Write a high-resolution crop around the stored endpoint location."""
    first_raw = max(0, endpoint["raw_end_frame"] - before)
    last_raw = endpoint["raw_end_frame"] + after
    focus = endpoint.get("end_position")
    if focus is None:
        focus = (selected.get(endpoint["end_frame"]) or {}).get("pos")
    tiles = []
    tile_w, image_h, tile_h = 480, 480, 545
    for raw_frame in range(first_raw, last_raw + 1):
        source = frames.get(raw_frame)
        tile = np.full((tile_h, tile_w, 3), 20, dtype=np.uint8)
        tracker_frame = raw_frame + 1
        if source is not None:
            point = focus or [source.shape[1] // 2, source.shape[0] // 2]
            left = max(0, min(source.shape[1] - 1, int(point[0] - crop_size // 2)))
            top = max(0, min(source.shape[0] - 1, int(point[1] - crop_size // 2)))
            right = min(source.shape[1], left + crop_size)
            bottom = min(source.shape[0], top + crop_size)
            left = max(0, right - crop_size)
            top = max(0, bottom - crop_size)
            crop = source[top:bottom, left:right].copy()
            track_point = (selected.get(tracker_frame) or {}).get("pos")
            if track_point is not None:
                _marker(crop, [track_point[0] - left, track_point[1] - top], (0, 255, 0))
            if tracker_frame == endpoint["end_frame"] and endpoint.get("end_position") is not None:
                end_point = endpoint["end_position"]
                _marker(crop, [end_point[0] - left, end_point[1] - top], (0, 0, 255))
            tile[:image_h] = cv2.resize(crop, (tile_w, image_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"tracker f{tracker_frame} / raw f{raw_frame}", (10, image_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
        if tracker_frame == endpoint["end_frame"]:
            cv2.putText(tile, "POINT END", (10, image_h + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        tiles.append(tile)
    columns = 3
    rows = int(math.ceil(len(tiles) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 12, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * tile_h
        x = (index % columns) * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = tile
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def sequential_endpoint_audit(video_path, endpoints, selected, output_dir, before, after, progress=None):
    """Decode all point-end windows in one forward pass.

    HEVC random seeking can decode the wrong reference picture.  This routine
    intentionally opens the source once and reads forward from frame zero.
    """
    endpoint_dir = Path(output_dir) / "endpoints"
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    requests = {}
    for endpoint in endpoints:
        for raw_frame in range(max(0, endpoint["raw_end_frame"] - before), endpoint["raw_end_frame"] + after + 1):
            requests.setdefault(raw_frame, []).append(endpoint["audit_index"])
    if not requests:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    captured = {endpoint["audit_index"]: {} for endpoint in endpoints}
    last_requested = max(requests)
    if progress is not None:
        progress.update(
            stage="sequential endpoint decode",
            overall_percent=80.0,
            current_frame=0,
            start_frame=0,
            end_frame=last_requested,
            events_completed=0,
            events_total=len(endpoints),
            message=f"Sequentially decoding raw frames 0 through {last_requested}",
        )
    raw_frame = 0
    while raw_frame <= last_requested:
        ok, frame = cap.read()
        if not ok:
            break
        for audit_index in requests.get(raw_frame, ()):
            captured[audit_index][raw_frame] = frame.copy()
        raw_frame += 1
        if progress is not None and (raw_frame % 100 == 0 or raw_frame > last_requested):
            progress.update(
                stage="sequential endpoint decode",
                overall_percent=80.0 + 19.0 * raw_frame / max(1, last_requested + 1),
                current_frame=raw_frame,
                events_completed=sum(bool(captured[item["audit_index"]]) for item in endpoints),
                events_total=len(endpoints),
                message=f"Sequentially decoded raw frame {raw_frame}/{last_requested}",
            )
    cap.release()
    report_rows = []
    for completed, endpoint in enumerate(endpoints, start=1):
        sheet_path = endpoint_dir / f"point_{endpoint['audit_index']:04d}_end_f{endpoint['end_frame']}.jpg"
        zoom_path = endpoint_dir / f"point_{endpoint['audit_index']:04d}_end_f{endpoint['end_frame']}_zoom.jpg"
        _endpoint_contact_sheet(captured[endpoint["audit_index"]], endpoint, selected, before, after, sheet_path)
        _endpoint_zoom_sheet(captured[endpoint["audit_index"]], endpoint, selected, before, after, zoom_path)
        report = dict(endpoint)
        report["sheet"] = str(sheet_path.relative_to(output_dir)).replace("\\", "/")
        report["zoom_sheet"] = str(zoom_path.relative_to(output_dir)).replace("\\", "/")
        report["decoded_frames"] = sorted(captured[endpoint["audit_index"]])
        report_rows.append(report)
        if progress is not None:
            progress.update(
                stage="writing endpoint sheets",
                overall_percent=99.0,
                events_completed=completed,
                events_total=len(endpoints),
                message=f"Wrote clean endpoint sheet {completed}/{len(endpoints)}",
            )
    return report_rows


def _write_endpoint_html(output_dir, endpoint_rows):
    rows = []
    for endpoint in endpoint_rows:
        rows.append(
            f"<section><h2>Point {html.escape(endpoint['point_index']) or endpoint['audit_index']} "
            f"· endpoint f{endpoint['end_frame']}</h2>"
            f"<p><b>Start:</b> f{endpoint['serve_start_frame']} · <b>Reason:</b> {html.escape(endpoint['end_reason'])} "
            f"· <b>Category:</b> {html.escape(endpoint['category'])}</p>"
            f"<p>Green: live tracker · Red: stored point endpoint · labels show tracker frame / raw video frame.</p>"
            f"<img src='{html.escape(endpoint['sheet'])}'><h3>Endpoint-centred crop</h3>"
            f"<img src='{html.escape(endpoint['zoom_sheet'])}'></section>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Point-end audit</title>
<style>body{{font:15px Arial;max-width:1500px;margin:auto;padding:24px;background:#f4f6f7}}section{{background:#fff;padding:18px;margin:18px 0}}img{{max-width:100%;border:1px solid #999}}</style></head>
<body><h1>Sequential point-end visual audit</h1><p>{len(endpoint_rows)} point ends. Source frames were decoded forward once; no random HEVC seeks were used.</p>{''.join(rows)}</body></html>"""
    path = Path(output_dir) / "endpoint_index.html"
    path.write_text(document, encoding="utf-8")
    return path


def _decode_window(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    frames = {}
    index = max(0, int(start_frame))
    while index <= int(end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        frames[index] = frame.copy()
        index += 1
    cap.release()
    return frames


def _marker(frame, point, color):
    if point is None:
        return
    x, y = int(point[0]), int(point[1])
    radius = max(12, int(min(frame.shape[:2]) * 0.012))
    cv2.circle(frame, (x, y), radius, color, max(2, radius // 6))
    cv2.line(frame, (x - radius - 5, y), (x + radius + 5, y), color, 2)
    cv2.line(frame, (x, y - radius - 5), (x, y + radius + 5), color, 2)


def _contact_sheet(frames, selected, start_frame, end_frame, output_path):
    frame_numbers = list(range(start_frame, end_frame + 1))
    if len(frame_numbers) > 15:
        stride = max(1, len(frame_numbers) // 15)
        frame_numbers = frame_numbers[::stride]
        if frame_numbers[-1] != end_frame:
            frame_numbers.append(end_frame)
    tiles = []
    tile_w, image_h, tile_h = 480, 275, 310
    for number in frame_numbers:
        source = frames.get(number)
        tile = np.full((tile_h, tile_w, 3), 20, dtype=np.uint8)
        if source is not None:
            display = source.copy()
            selected_point = (selected.get(number) or {}).get("pos")
            _marker(display, selected_point, (0, 255, 0))
            scale = max(tile_w / display.shape[1], image_h / display.shape[0])
            resized = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            x0 = max(0, (resized.shape[1] - tile_w) // 2)
            y0 = max(0, (resized.shape[0] - image_h) // 2)
            tile[:image_h] = resized[y0:y0 + image_h, x0:x0 + tile_w]
        cv2.putText(tile, f"f{number}", (10, image_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        if selected.get(number, {}).get("stuck", 0) >= 3:
            cv2.putText(tile, f"stuck={selected[number]['stuck']}", (110, image_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
        tiles.append(tile)
    columns = 3
    rows = int(math.ceil(len(tiles) / columns))
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 12, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * tile_h
        x = (index % columns) * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = tile
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _image_data_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _vision_review(client, model, sheet_path, event):
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": (
                    "Audit this fixed-camera tennis tracking contact sheet. Green markers are the "
                    "tracker's selected positions. Follow the yellow ball across adjacent frames. "
                    "Decide whether the tracker follows the ball or switches to a player/racket/court "
                    "artifact. Return JSON only with ball_visible, tracker_correct, player_takeover, "
                    "observed_ball_position, confidence, summary, and recommended_action.\n\n"
                    f"EVENT: {json.dumps(event)}"
                )},
                {"type": "input_image", "image_url": _image_data_url(sheet_path), "detail": "high"},
            ],
        }],
        text={"format": {"type": "json_schema", "name": "tracking_event_audit", "strict": True, "schema": {
            "type": "object",
            "properties": {
                "ball_visible": {"type": "boolean"},
                "tracker_correct": {"type": "boolean"},
                "player_takeover": {"type": "boolean"},
                "observed_ball_position": {"type": ["array", "null"], "items": {"type": "number"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "summary": {"type": "string"},
                "recommended_action": {"type": "string"},
            },
            "required": ["ball_visible", "tracker_correct", "player_takeover", "observed_ball_position", "confidence", "summary", "recommended_action"],
            "additionalProperties": False,
        }}},
    )
    return json.loads(response.output_text)


def _write_html(output_dir, summary, events):
    rows = []
    for event in events:
        vision = event.get("vision") or {}
        rows.append(
            f"<section><h2>Event {event['event_index']} · f{event['center_frame']} · "
            f"{html.escape(event['severity'])}</h2>"
            f"<p><b>Types:</b> {html.escape(', '.join(event['types']))}</p>"
            f"<p><b>Evidence:</b> {html.escape(' | '.join(event['evidence']))}</p>"
            f"<img src='{html.escape(event['sheet'])}'><pre>{html.escape(json.dumps(vision, indent=2))}</pre></section>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Tracking audit</title>
<style>body{{font:15px Arial;max-width:1500px;margin:auto;padding:24px;background:#f4f6f7}}section{{background:#fff;padding:18px;margin:18px 0}}img{{max-width:100%;border:1px solid #999}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>Tracking event audit</h1><p>Events: {summary['events']} · High severity: {summary['high_severity']} · Provider: {html.escape(summary['provider'])}</p>{''.join(rows)}</body></html>"""
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find and visually audit suspicious ball-tracking windows")
    parser.add_argument("--court", choices=sorted(COURTS), default="night")
    parser.add_argument("--video")
    parser.add_argument("--config")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument(
        "--player-tracking-interval", type=int, default=5,
        help="Run the auxiliary player/racket detector every N frames during a regression",
    )
    parser.add_argument("--output-dir", default="tmp/tracking_audit")
    parser.add_argument("--provider", choices=("local", "auto", "openai"), default="local")
    parser.add_argument("--model", default=os.environ.get("TRACKING_AUDIT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--endpoint-audit", action="store_true",
        help="Create clean sequentially-decoded sheets for every stored point end",
    )
    parser.add_argument(
        "--endpoint-only", action="store_true",
        help="Skip suspicious-event rendering and only create the point-end audit",
    )
    parser.add_argument("--endpoint-before", type=int, default=4)
    parser.add_argument("--endpoint-after", type=int, default=5)
    parser.add_argument(
        "--endpoint-indices",
        help="Comma-separated point-end audit indices for a focused sequential review",
    )
    parser.add_argument("--skip-run", action="store_true", help="Audit an existing --log and --history")
    parser.add_argument("--log")
    parser.add_argument("--history")
    parser.add_argument(
        "--progress-file",
        help="Live JSON progress path (a matching .txt file is also written)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    video = Path(args.video or COURTS[args.court][0])
    config = Path(args.config or COURTS[args.court][1])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else output_dir / "tracker.log"
    history_base = Path(args.history) if args.history else output_dir / "point_history.csv"
    progress = AuditProgress(output_dir, args.progress_file)
    end_frame = _video_end_frame(video, args.start_frame, args.max_frames)
    progress.update(
        status="running",
        stage="starting",
        overall_percent=0.0,
        current_frame=args.start_frame,
        start_frame=args.start_frame,
        end_frame=end_frame,
        events_completed=0,
        events_total=None,
        message="Preparing tracker regression",
    )

    if not args.skip_run:
        command = [
            sys.executable, str(root / "interactive_ball_analyzer.py"),
            "--court", args.court, "--start-frame", str(args.start_frame),
            "--max-frames", str(args.max_frames), "--auto-play", "--headless",
            "--quiet", "--no-audit-points", "--no-continue-next-video", "--disable-player-learning",
            "--player-tracking-interval", str(max(1, int(args.player_tracking_interval))),
            "--point-history-file", str(history_base),
        ]
        progress.update(
            stage="tracker regression",
            overall_percent=0.0,
            message="Running headless ball tracker",
        )
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
            )
            last_progress_frame = args.start_frame - 1
            last_progress_time = 0.0
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                frame = _progress_frame_from_line(line)
                if frame is None or frame < last_progress_frame:
                    continue
                now = time.monotonic()
                if frame - last_progress_frame < 10 and now - last_progress_time < 1.0:
                    continue
                last_progress_frame = frame
                last_progress_time = now
                handle.flush()
                if end_frame is not None and end_frame >= args.start_frame:
                    frame_percent = (
                        (frame - args.start_frame + 1) /
                        max(1, end_frame - args.start_frame + 1)
                    ) * 100.0
                else:
                    frame_percent = 0.0
                progress.update(
                    current_frame=frame,
                    overall_percent=min(80.0, frame_percent * 0.80),
                    message=f"Tracking frame {frame}",
                )
            return_code = process.wait()
        (output_dir / "tracker_exit_code.txt").write_text(str(return_code), encoding="utf-8")
        progress.update(
            stage="analyzing tracker log",
            overall_percent=80.0,
            current_frame=end_frame if end_frame is not None else last_progress_frame,
            message=f"Tracker finished with exit code {return_code}; finding suspicious windows",
        )
    else:
        progress.update(
            stage="analyzing existing tracker log",
            overall_percent=80.0,
            message="Using existing log and point history",
        )

    if not log_path.exists():
        raise FileNotFoundError(f"Tracker log not found: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    tracks, explicit_events = parse_track_log(log_text)
    events = find_suspicious_events(tracks, explicit_events)
    if args.max_events and len(events) > args.max_events:
        events = events[:args.max_events]
    history_path = Path(args.history) if args.history else _timestamped_history(history_base)
    history_rows = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
            history_rows = list(csv.DictReader(handle))

    endpoint_rows = []
    endpoint_report = None
    if args.endpoint_audit or args.endpoint_only:
        endpoints = point_endpoints(history_rows)
        if args.endpoint_indices:
            requested_indices = {
                int(value.strip()) for value in args.endpoint_indices.split(",") if value.strip()
            }
            endpoints = [item for item in endpoints if item["audit_index"] in requested_indices]
        progress.update(
            stage="sequential endpoint decode",
            overall_percent=80.0,
            events_completed=0,
            events_total=len(endpoints),
            message=f"Preparing clean sequential review for {len(endpoints)} point ends",
        )
        endpoint_rows = sequential_endpoint_audit(
            video, endpoints, tracks, output_dir,
            max(0, args.endpoint_before), max(0, args.endpoint_after), progress,
        )
        (output_dir / "endpoint_audit.json").write_text(
            json.dumps({
                "video": str(video),
                "history": str(history_path),
                "frame_mapping": "tracker frame N = raw decoded video frame N-1",
                "endpoints": endpoint_rows,
            }, indent=2),
            encoding="utf-8",
        )
        endpoint_report = _write_endpoint_html(output_dir, endpoint_rows)
        if args.endpoint_only:
            summary = {
                "video": str(video), "history": str(history_path), "log": str(log_path),
                "tracked_frames": len(tracks), "events": 0, "high_severity": 0,
                "provider": "local", "endpoint_audit": len(endpoint_rows),
            }
            progress.update(
                status="complete", stage="complete", overall_percent=100.0,
                events_completed=len(endpoint_rows), events_total=len(endpoint_rows),
                message=f"Sequential endpoint audit complete; report: {endpoint_report}",
            )
            print(json.dumps({"summary": summary, "report": str(endpoint_report)}, indent=2))
            return 0

    images_dir = output_dir / "events"
    images_dir.mkdir(parents=True, exist_ok=True)
    client = None
    provider = "local"
    if args.provider != "local" and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            provider = "openai"
        except ImportError:
            if args.provider == "openai":
                raise RuntimeError("Install requirements-vision.txt for --provider openai")
    elif args.provider == "openai":
        raise RuntimeError("OPENAI_API_KEY is required for --provider openai")

    progress.update(
        stage="rendering visual event windows",
        overall_percent=80.0,
        events_completed=0,
        events_total=len(events),
        message=f"Rendering {len(events)} suspicious windows",
    )
    for completed_count, event in enumerate(events, start=1):
        start = max(0, event["start_frame"] - args.window)
        end = event["end_frame"] + args.window
        frames = _decode_window(video, start, end)
        sheet_path = images_dir / f"event_{event['event_index']:04d}_f{event['center_frame']}.jpg"
        _contact_sheet(frames, tracks, start, end, sheet_path)
        event["sheet"] = str(sheet_path.relative_to(output_dir)).replace("\\", "/")
        event["point_index"] = _point_for_frame(history_rows, event["center_frame"])
        if client is not None:
            try:
                event["vision"] = _vision_review(client, args.model, sheet_path, event)
            except Exception as error:
                event["vision_error"] = str(error)
        progress.update(
            overall_percent=80.0 + 19.0 * completed_count / max(1, len(events)),
            events_completed=completed_count,
            message=(
                f"Rendered event {completed_count}/{len(events)} "
                f"around frame {event['center_frame']}"
            ),
        )

    summary = {
        "video": str(video),
        "history": str(history_path),
        "log": str(log_path),
        "tracked_frames": len(tracks),
        "events": len(events),
        "high_severity": sum(event["severity"] == "high" for event in events),
        "provider": provider,
        "endpoint_audit": len(endpoint_rows),
    }
    (output_dir / "tracking_audit.json").write_text(
        json.dumps({"summary": summary, "events": events}, indent=2), encoding="utf-8"
    )
    (output_dir / "tracking_trace.jsonl").write_text(
        "".join(json.dumps(tracks[frame]) + "\n" for frame in sorted(tracks)), encoding="utf-8"
    )
    report = _write_html(output_dir, summary, events)
    progress.update(
        status="complete",
        stage="complete",
        overall_percent=100.0,
        events_completed=len(events),
        events_total=len(events),
        message=f"Audit complete; report: {report}",
    )
    print(json.dumps({"summary": summary, "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
