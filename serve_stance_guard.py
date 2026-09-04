"""Serve-start player stance validation.

A ball-like toss is not enough to start a tennis point.  When player tracking
is available, require fresh evidence that the player on the serving end is
standing behind the physical baseline before a toss may be confirmed.

This module is deliberately independent of frame numbers.  Court geometry is
read from the calibrated ``adjusted_court_points.txt`` file.  In that layout
points 1-2 are the far baseline endpoints and points 3-4 are the near baseline
endpoints.  The player's feet/bottom-of-box are compared with the appropriate
perspective line.
"""
from __future__ import annotations

import math
import os
import re
from typing import Dict, Optional, Tuple


_POINT_RE = re.compile(r"Point\s+(\d+)\s*:\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_GEOMETRY_CACHE: Dict[str, Tuple[Optional[float], Optional[Dict[int, Tuple[float, float]]]]] = {}


def _load_full_frame_points(path: str) -> Optional[Dict[int, Tuple[float, float]]]:
    """Load the calibrated full-video court points, with a tiny mtime cache."""
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


def evaluate_serve_stance(analyzer, serve_position, frame) -> Dict:
    """Return allow/hold/reject for a potential serve toss.

    ``hold`` means player tracking is enabled but there is no *fresh* reliable
    server detection on this frame.  The caller should wait rather than turn a
    ball-coloured artifact into a point.  ``reject`` means the fresh player's
    feet are clearly inside the court instead of behind the baseline.

    If player tracking or calibrated baseline geometry is unavailable, the
    guard deliberately returns ``bypass`` so existing non-player-tracking
    workflows keep working.
    """
    result = {
        "decision": "bypass",
        "reason": "guard unavailable",
        "side": None,
        "center": None,
        "feet": None,
        "baseline_y": None,
        "delta": None,
        "confidence": None,
    }

    if frame is None or serve_position is None:
        result["reason"] = "missing frame/serve position"
        return result

    tracker = getattr(analyzer, "player_tracker", None)
    if tracker is None or not bool(getattr(analyzer, "player_tracking_enabled", False)):
        result["reason"] = "player tracking disabled"
        return result

    height = float(frame.shape[0])
    sy = float(serve_position[1])
    side = "near" if sy > _net_y(analyzer, height) else "far"
    result["side"] = side

    tracks = getattr(tracker, "tracks", None)
    track = tracks.get(side) if isinstance(tracks, dict) else None
    if track is None or getattr(track, "bbox", None) is None:
        result.update(decision="hold", reason=f"no {side} player track")
        return result

    result["center"] = tuple(track.center) if getattr(track, "center", None) is not None else None
    result["confidence"] = float(getattr(track, "confidence", 0.0) or 0.0)

    # ``visible`` is true only on an actual detector update.  On the intervening
    # propagation frames PlayerRacketTracker intentionally sets it false.  This
    # prevents an old player box from authorising a new serve minutes later.
    if not bool(getattr(track, "visible", False)):
        result.update(decision="hold", reason=f"{side} player track is not freshly detected")
        return result
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
        # Image Y grows toward the near camera.  A legal near-end server is on
        # the camera side of (at or below) the near baseline.
        delta = feet[1] - baseline_y
        behind = delta >= -tolerance
    else:
        # At the far end, behind the baseline is toward the top of the image.
        delta = baseline_y - feet[1]
        behind = delta >= -tolerance

    result["delta"] = float(delta)
    if not behind:
        result.update(
            decision="reject",
            reason=(
                f"{side} player feet are inside court by "
                f"{abs(delta):.0f}px (tolerance={tolerance:.0f}px)"
            ),
        )
        return result

    result.update(decision="allow", reason=f"fresh {side} player is behind baseline")
    return result


def format_serve_stance_debug(frame_index: int, result: Dict) -> str:
    decision = str(result.get("decision") or "unknown").upper()
    return (
        f"[SERVE_STANCE_{decision}] f{int(frame_index)}: "
        f"side={result.get('side') or '?'} "
        f"center={result.get('center')} feet={result.get('feet')} "
        f"baseline_y={result.get('baseline_y') if result.get('baseline_y') is not None else 'n/a'} "
        f"delta={result.get('delta') if result.get('delta') is not None else 'n/a'} "
        f"conf={result.get('confidence') if result.get('confidence') is not None else 'n/a'} "
        f"reason={result.get('reason') or ''}"
    )
