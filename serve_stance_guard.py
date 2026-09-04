"""Serve-start player stance validation.

A ball-like toss is not enough to start a tennis point. When player tracking
is available, require fresh evidence that the player on the serving end is
standing behind the physical baseline before a toss may be confirmed.

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


def _remember_valid_stance(analyzer, frame_index: int, side: str, result: Dict) -> None:
    analyzer._serve_stance_last_valid = {
        "frame": int(frame_index),
        "side": str(side),
        "center": result.get("center"),
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
    # PlayerRacketTracker only runs its detector every N frames. Keep one fresh
    # legal stance alive across the propagation frames, but never long enough
    # for a player box from a previous point to authorise a new serve.
    if 0 <= age <= max(6, interval * 2 + 2):
        return recent
    return None


def evaluate_serve_stance(analyzer, serve_position, frame) -> Dict:
    """Return allow/hold/reject/bypass for a potential serve toss.

    ``hold`` means player tracking is enabled but no recent reliable server
    detection has established a legal serving stance. ``reject`` means a fresh
    player is clearly inside the court instead of behind the baseline.

    A fresh valid detection is cached only across the short detector interval,
    so normal 5-frame player-detection cadence does not repeatedly reset a real
    toss. If player tracking or calibrated baseline geometry is unavailable,
    ``bypass`` preserves the previous tracker behaviour.
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
    sy = float(serve_position[1])
    side = "near" if sy > _net_y(analyzer, height) else "far"
    result["side"] = side

    tracks = getattr(tracker, "tracks", None)
    track = tracks.get(side) if isinstance(tracks, dict) else None
    if track is None or getattr(track, "bbox", None) is None:
        recent = _recent_valid_stance(analyzer, frame_index, side)
        if recent is not None:
            result.update(
                decision="allow",
                reason="recent fresh legal player stance",
                center=recent.get("center"),
                feet=recent.get("feet"),
                baseline_y=recent.get("baseline_y"),
                delta=recent.get("delta"),
                confidence=recent.get("confidence"),
            )
            return result
        result.update(decision="hold", reason=f"no {side} player track")
        return result

    result["center"] = tuple(track.center) if getattr(track, "center", None) is not None else None
    result["confidence"] = float(getattr(track, "confidence", 0.0) or 0.0)

    if not bool(getattr(track, "visible", False)):
        recent = _recent_valid_stance(analyzer, frame_index, side)
        if recent is not None:
            result.update(
                decision="allow",
                reason="recent fresh legal player stance",
                center=recent.get("center"),
                feet=recent.get("feet"),
                baseline_y=recent.get("baseline_y"),
                delta=recent.get("delta"),
                confidence=recent.get("confidence"),
            )
            return result
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

    result.update(decision="allow", reason=f"fresh {side} player is behind baseline")
    _remember_valid_stance(analyzer, frame_index, side, result)
    return result


def format_serve_stance_debug(frame_index: int, result: Dict) -> str:
    decision = str(result.get("decision") or "unknown").upper()
    baseline = result.get("baseline_y")
    delta = result.get("delta")
    confidence = result.get("confidence")
    return (
        f"[SERVE_STANCE_{decision}] f{int(frame_index)}: "
        f"side={result.get('side') or '?'} "
        f"fresh={bool(result.get('fresh'))} "
        f"center={result.get('center')} feet={result.get('feet')} "
        f"baseline_y={f'{baseline:.1f}' if isinstance(baseline, (int, float)) else 'n/a'} "
        f"delta={f'{delta:.1f}' if isinstance(delta, (int, float)) else 'n/a'} "
        f"conf={f'{confidence:.3f}' if isinstance(confidence, (int, float)) else 'n/a'} "
        f"reason={result.get('reason') or ''}"
    )
