"""Serve-start player stance validation.

A ball-like toss is not enough to start a tennis point. When player tracking
is available, require evidence that the player on the serving end is standing
behind the physical baseline and that the serve candidate is horizontally
consistent with that player.

The rule is frame-independent. Court geometry is read from the calibrated
``adjusted_court_points.txt`` file. In that layout points 1-2 are the far
baseline endpoints and points 3-4 are the near baseline endpoints. The
player's feet/bottom-of-box are compared with the appropriate perspective
line.
"""
from __future__ import annotations

import math
import os
import re
from typing import Dict, Optional, Tuple


_POINT_RE = re.compile(r"Point\s+(\d+)\s*:\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_GEOMETRY_CACHE: Dict[str, Tuple[Optional[float], Optional[Dict[int, Tuple[float, float]]]]] = {}


def _load_full_frame_points(path: str) -> Optional[Dict[int, Tuple[float, float]]]:
    """Load calibrated full-video court points, with a tiny mtime cache."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None

    cached = _GEOMETRY_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    points: Dict[int, Tuple[float, float]] = {}
    in_full = False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("Full Video Frame Coordinates"):
                    in_full = True
                    continue
                if line.startswith("Right Court Region Coordinates"):
                    break
                if not in_full:
                    continue
                match = _POINT_RE.search(line)
                if match:
                    points[int(match.group(1))] = (
                        float(match.group(2)),
                        float(match.group(3)),
                    )
    except OSError:
        points = {}

    result = points if all(idx in points for idx in (1, 2, 3, 4)) else None
    _GEOMETRY_CACHE[path] = (mtime, result)
    return result


def _line_y(p0: Tuple[float, float], p1: Tuple[float, float], x: float) -> Optional[float]:
    dx = float(p1[0]) - float(p0[0])
    if abs(dx) < 1e-6:
        return None
    t = (float(x) - float(p0[0])) / dx
    return float(p0[1]) + t * (float(p1[1]) - float(p0[1]))


def _net_y(analyzer, frame_height: float) -> float:
    values = []
    for name in ("net_area_y_min", "net_area_y_max", "high_net_y_max", "low_net_y_min"):
        value = getattr(analyzer, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if values:
        return sum(values) / len(values)
    return frame_height * 0.5


def _horizontal_alignment_limit(frame_width: float, side: str, bbox) -> float:
    """Return a perspective-aware max X gap between server and toss candidate.

    A near-end 4K player can be several hundred pixels wide during a serve,
    so the gate must allow the real toss to sit beside the body. At the same
    time it must reject the persistent court-colour blobs hundreds/thousands
    of pixels away that originally caused the stance guard regression.
    """
    width = max(1.0, float(frame_width))
    bbox_width = 0.0
    if bbox is not None:
        try:
            bbox_width = max(0.0, float(bbox[2]))
        except (TypeError, ValueError, IndexError):
            bbox_width = 0.0

    if side == "near":
        minimum = max(160.0, width * 0.070)
        maximum = max(minimum, width * 0.090)
        box_based = bbox_width * 0.55 + 90.0
    else:
        minimum = max(90.0, width * 0.028)
        maximum = max(minimum, width * 0.055)
        box_based = bbox_width * 0.70 + 50.0

    return float(min(maximum, max(minimum, box_based)))


def _horizontal_alignment(serve_position, side: str, center, bbox, frame_width: float):
    if serve_position is None or center is None:
        return None, None, None
    try:
        dx = abs(float(serve_position[0]) - float(center[0]))
    except (TypeError, ValueError, IndexError):
        return None, None, None
    limit = _horizontal_alignment_limit(frame_width, side, bbox)
    return dx <= limit, float(dx), float(limit)


def _remember_valid_stance(analyzer, frame_index: int, side: str, result: Dict) -> None:
    analyzer._serve_stance_last_valid = {
        "frame": int(frame_index),
        "side": str(side),
        "center": result.get("center"),
        "bbox": result.get("bbox"),
        "feet": result.get("feet"),
        "baseline_y": result.get("baseline_y"),
        "delta": result.get("delta"),
        "confidence": result.get("confidence"),
    }


def _recent_valid_stance(analyzer, frame_index: int, side: str) -> Optional[Dict]:
    recent = getattr(analyzer, "_serve_stance_last_valid", None)
    if not isinstance(recent, dict) or recent.get("side") != side:
        return None
    try:
        age = int(frame_index) - int(recent.get("frame", -1000000))
    except (TypeError, ValueError):
        return None
    interval = max(1, int(getattr(analyzer, "player_tracking_interval", 5) or 5))
    # A legal server can stand behind the baseline for a few seconds before
    # the toss. Keep that verified stance long enough to cover temporary HOG
    # misses, but still bounded so it cannot survive an ordinary between-point
    # pause. Horizontal ball/player alignment is checked again before reuse.
    if 0 <= age <= max(120, interval * 6):
        return recent
    return None


def _apply_recent_stance(result: Dict, recent: Dict, serve_position, frame_width: float) -> Dict:
    side = str(result.get("side") or recent.get("side") or "")
    center = recent.get("center")
    bbox = recent.get("bbox")
    aligned, horizontal_dx, horizontal_limit = _horizontal_alignment(
        serve_position, side, center, bbox, frame_width
    )
    result.update(
        center=center,
        bbox=bbox,
        feet=recent.get("feet"),
        baseline_y=recent.get("baseline_y"),
        delta=recent.get("delta"),
        confidence=recent.get("confidence"),
        horizontal_dx=horizontal_dx,
        horizontal_limit=horizontal_limit,
    )
    if aligned is False:
        result.update(
            decision="reject",
            reason=(
                f"serve candidate is {horizontal_dx:.0f}px horizontally from recent "
                f"{side} server (limit={horizontal_limit:.0f}px)"
            ),
        )
    else:
        result.update(decision="allow", reason="recent verified legal server stance + aligned toss")
    return result


def evaluate_serve_stance(analyzer, serve_position, frame) -> Dict:
    """Return allow/hold/reject/bypass for a potential serve toss.

    ``hold`` means player tracking is enabled but no recent reliable server
    detection has established a legal serving stance. ``reject`` means either
    a fresh player is clearly inside the court or the ball-like candidate is
    not horizontally associated with that server.

    A verified behind-baseline stance is cached for a short pre-serve window.
    Reusing that cache still requires the current candidate to be horizontally
    aligned with the cached server, which prevents static court blobs from
    polluting the toss history. If player tracking or calibrated baseline
    geometry is unavailable, ``bypass`` preserves the previous tracker
    behaviour.
    """
    result = {
        "decision": "bypass",
        "reason": "guard unavailable",
        "side": None,
        "center": None,
        "bbox": None,
        "feet": None,
        "baseline_y": None,
        "delta": None,
        "confidence": None,
        "horizontal_dx": None,
        "horizontal_limit": None,
        "fresh": False,
    }

    if frame is None or serve_position is None:
        result["reason"] = "missing frame/serve position"
        return result

    tracker = getattr(analyzer, "player_tracker", None)
    if tracker is None or not bool(getattr(analyzer, "player_tracking_enabled", False)):
        result["reason"] = "player tracking disabled"
        return result

    frame_index = int(getattr(analyzer, "frame_count", -1))
    height = float(frame.shape[0])
    width = float(frame.shape[1])
    sy = float(serve_position[1])
    side = "near" if sy > _net_y(analyzer, height) else "far"
    result["side"] = side

    tracks = getattr(tracker, "tracks", None)
    track = tracks.get(side) if isinstance(tracks, dict) else None
    if track is None or getattr(track, "bbox", None) is None:
        recent = _recent_valid_stance(analyzer, frame_index, side)
        if recent is not None:
            return _apply_recent_stance(result, recent, serve_position, width)
        result.update(decision="hold", reason=f"no {side} player track")
        return result

    result["center"] = tuple(track.center) if getattr(track, "center", None) is not None else None
    result["bbox"] = tuple(track.bbox) if getattr(track, "bbox", None) is not None else None
    result["confidence"] = float(getattr(track, "confidence", 0.0) or 0.0)

    if not bool(getattr(track, "visible", False)):
        recent = _recent_valid_stance(analyzer, frame_index, side)
        if recent is not None:
            return _apply_recent_stance(result, recent, serve_position, width)
        result.update(decision="hold", reason=f"{side} player track is not freshly detected")
        return result

    result["fresh"] = True
    if result["confidence"] < 0.35:
        result.update(decision="hold", reason=f"{side} player confidence too low")
        return result

    feet = getattr(track, "shoes", None)
    if feet is None:
        x, y, w, h = [float(v) for v in track.bbox]
        feet = (x + 0.5 * w, y + h)
    feet = (float(feet[0]), float(feet[1]))
    result["feet"] = (int(round(feet[0])), int(round(feet[1])))

    geometry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adjusted_court_points.txt")
    points = _load_full_frame_points(geometry_path)
    if points is None:
        result.update(decision="bypass", reason="calibrated baseline points unavailable")
        return result

    baseline = (points[3], points[4]) if side == "near" else (points[1], points[2])
    baseline_y = _line_y(baseline[0], baseline[1], feet[0])
    if baseline_y is None:
        result.update(decision="bypass", reason="invalid calibrated baseline")
        return result

    tolerance = max(24.0, height * 0.012)
    result["baseline_y"] = float(baseline_y)

    if side == "near":
        # Image Y grows toward the near camera. A legal near-end server is on
        # the camera side of (at or below) the near baseline.
        delta = feet[1] - baseline_y
        behind = delta >= -tolerance
    else:
        # At the far end, behind the baseline is toward the top of the image.
        delta = baseline_y - feet[1]
        behind = delta >= -tolerance

    result["delta"] = float(delta)
    if not behind:
        # A fresh contradictory detection invalidates any short cached legal
        # stance from earlier frames.
        analyzer._serve_stance_last_valid = None
        result.update(
            decision="reject",
            reason=(
                f"{side} player feet are inside court by "
                f"{abs(delta):.0f}px (tolerance={tolerance:.0f}px)"
            ),
        )
        return result

    # Cache the legal player stance independently of the current ball-like
    # candidate. A static false candidate must not erase useful knowledge that
    # the real server is already standing legally behind the baseline.
    _remember_valid_stance(analyzer, frame_index, side, result)

    aligned, horizontal_dx, horizontal_limit = _horizontal_alignment(
        serve_position, side, result.get("center"), result.get("bbox"), width
    )
    result["horizontal_dx"] = horizontal_dx
    result["horizontal_limit"] = horizontal_limit
    if aligned is False:
        result.update(
            decision="reject",
            reason=(
                f"serve candidate is {horizontal_dx:.0f}px horizontally from fresh "
                f"{side} server (limit={horizontal_limit:.0f}px)"
            ),
        )
        return result

    result.update(decision="allow", reason=f"fresh {side} player behind baseline + aligned toss")
    return result


def format_serve_stance_debug(frame_index: int, result: Dict) -> str:
    decision = str(result.get("decision") or "unknown").upper()
    baseline = result.get("baseline_y")
    delta = result.get("delta")
    confidence = result.get("confidence")
    horizontal_dx = result.get("horizontal_dx")
    horizontal_limit = result.get("horizontal_limit")
    return (
        f"[SERVE_STANCE_{decision}] f{int(frame_index)}: "
        f"side={result.get('side') or '?'} "
        f"fresh={bool(result.get('fresh'))} "
        f"center={result.get('center')} feet={result.get('feet')} "
        f"baseline_y={f'{baseline:.1f}' if isinstance(baseline, (int, float)) else 'n/a'} "
        f"delta={f'{delta:.1f}' if isinstance(delta, (int, float)) else 'n/a'} "
        f"dx={f'{horizontal_dx:.1f}' if isinstance(horizontal_dx, (int, float)) else 'n/a'} "
        f"x_limit={f'{horizontal_limit:.1f}' if isinstance(horizontal_limit, (int, float)) else 'n/a'} "
        f"conf={f'{confidence:.3f}' if isinstance(confidence, (int, float)) else 'n/a'} "
        f"reason={result.get('reason') or ''}"
    )
