#!/usr/bin/env python3
"""Generic tennis serve-side consistency guard and point-history auditor.

Tennis invariant:
- Even point parity in a game -> deuce/right service court.
- Odd point parity -> ad/left service court.
- A second serve is from the same side as the first serve for that point.

The CSV auditor infers the observed side from two spatial clusters of serve-ball
x positions, separately for each server/physical end. It never uses frame
numbers and never changes the score; it only reports consistency problems.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
_STANDARD_POINTS = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4, "ADV": 4}
_POS_RE = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?")


def _parse_position(value: str) -> Optional[Point]:
    if not value:
        return None
    match = _POS_RE.fullmatch(value.strip())
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _split_score(score: str) -> Tuple[str, str]:
    """Return (games_token, points_token) from strings like '0:1 30:15'."""
    parts = (score or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return parts[-2], parts[-1]


def _point_value(token: str) -> Optional[int]:
    normalized = token.strip().upper()
    if normalized in _STANDARD_POINTS:
        return _STANDARD_POINTS[normalized]
    try:
        # Also supports numeric tiebreak scores.
        return int(normalized)
    except ValueError:
        return None


def point_parity(score: str) -> Optional[int]:
    """Return 0 for even point parity, 1 for odd, None if unparseable."""
    _games, points = _split_score(score)
    if ":" not in points:
        return None
    left, right = points.split(":", 1)
    a, b = _point_value(left), _point_value(right)
    if a is None or b is None:
        return None
    return (a + b) & 1


def expected_serve_side(score: str) -> Optional[str]:
    parity = point_parity(score)
    if parity is None:
        return None
    return "right" if parity == 0 else "left"


def _kmeans_1d(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    if len(values) < 4:
        return None
    low, high = min(values), max(values)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return None
    c0, c1 = low, high
    for _ in range(32):
        group0: List[float] = []
        group1: List[float] = []
        for value in values:
            if abs(value - c0) <= abs(value - c1):
                group0.append(value)
            else:
                group1.append(value)
        if not group0 or not group1:
            return None
        new0 = sum(group0) / len(group0)
        new1 = sum(group1) / len(group1)
        if abs(new0 - c0) < 0.01 and abs(new1 - c1) < 0.01:
            c0, c1 = new0, new1
            break
        c0, c1 = new0, new1
    return min(c0, c1), max(c0, c1)


@dataclass
class ServeCheck:
    row_number: int
    point_index: str
    frame: str
    server: str
    attempt: str
    score_before: str
    expected: Optional[str]
    observed: Optional[str]
    confidence: float
    x: Optional[float]
    status: str
    detail: str


class ServeSideConsistencyGuard:
    """Generic spatial serve-side validator.

    With the current court-camera convention, a near-side server's right/deuce
    side is the high-x half of the image. A far-side server faces the opposite
    direction, so right/deuce is the low-x half.

    classify_from_centerline() is intended for direct integration with the live
    tracker when a projected court centerline is available. audit_rows() is the
    independent CSV consistency audit and learns the two serve-position clusters
    without using frame numbers.
    """

    def __init__(self, min_cluster_separation_px: float = 160.0, min_confidence: float = 0.25):
        self.min_cluster_separation_px = float(min_cluster_separation_px)
        self.min_confidence = float(min_confidence)

    def classify_from_centerline(
        self,
        *,
        serve_x: float,
        center_x: float,
        physical_side: str,
        scale_px: float = 120.0,
    ) -> Tuple[Optional[str], float]:
        physical = (physical_side or "").strip().lower()
        if physical not in {"near", "far"}:
            return None, 0.0
        delta = float(serve_x) - float(center_x)
        if physical == "near":
            observed = "right" if delta >= 0 else "left"
        else:
            observed = "right" if delta <= 0 else "left"
        confidence = min(1.0, abs(delta) / max(1.0, float(scale_px)))
        return observed, confidence

    def audit_rows(self, rows: Sequence[dict], initial_score: str = "0:0 0:0") -> List[ServeCheck]:
        enriched: List[dict] = []
        previous_score = initial_score
        for row_number, raw in enumerate(rows, start=2):
            row = dict(raw)
            score_before = (row.get("score_before") or previous_score or initial_score).strip()
            row["_score_before"] = score_before
            row["_row_number"] = row_number
            row["_serve_pos"] = _parse_position(row.get("serve_ball_position", ""))
            enriched.append(row)
            current = (row.get("current_score") or "").strip()
            if current:
                previous_score = current

        # Learn two x clusters per server and physical end. First serves only,
        # so a first-serve fault does not overweight one side.
        groups: Dict[Tuple[str, str], List[float]] = {}
        for row in enriched:
            if (row.get("serve_attempt") or "").strip().lower() not in {"1st", "first", "1"}:
                continue
            position = row["_serve_pos"]
            if position is None:
                continue
            key = (
                (row.get("server") or "").strip(),
                (row.get("serve_player_side") or "").strip().lower(),
            )
            groups.setdefault(key, []).append(position[0])

        models: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        for key, xs in groups.items():
            centers = _kmeans_1d(xs)
            if centers is None:
                continue
            low, high = centers
            separation = high - low
            if separation < self.min_cluster_separation_px:
                continue
            models[key] = (low, high, (low + high) / 2.0)

        checks: List[ServeCheck] = []
        last_first_side_by_game: Dict[Tuple[str, str, str], str] = {}
        first_attempt_by_point: Dict[Tuple[str, str, str], Tuple[str, float]] = {}

        for row in enriched:
            score_before = row["_score_before"]
            expected = expected_serve_side(score_before)
            attempt = (row.get("serve_attempt") or "").strip()
            server = (row.get("server") or "").strip()
            physical = (row.get("serve_player_side") or "").strip().lower()
            point_index = (row.get("point_index") or "").strip()
            frame = (row.get("serve_start_frame") or "").strip()
            position = row["_serve_pos"]
            x = position[0] if position else None
            observed: Optional[str] = None
            confidence = 0.0
            details: List[str] = []

            model = models.get((server, physical))
            if x is not None and model is not None and physical in {"near", "far"}:
                low, high, midpoint = model
                if physical == "near":
                    observed = "right" if x >= midpoint else "left"
                else:
                    observed = "right" if x <= midpoint else "left"
                half_separation = max(1.0, (high - low) / 2.0)
                confidence = min(1.0, abs(x - midpoint) / half_separation)
                details.append(f"clusters={low:.0f}/{high:.0f}px midpoint={midpoint:.0f}px")

            status = "UNKNOWN"
            if expected and observed and confidence >= self.min_confidence:
                status = "OK" if expected == observed else "MISMATCH"
            elif expected and observed:
                status = "LOW_CONFIDENCE"

            games_token, _points_token = _split_score(score_before)
            is_first = attempt.lower() in {"1st", "first", "1"}
            is_second = attempt.lower() in {"2nd", "second", "2"}

            # Independent invariant: consecutive new points in one game must
            # alternate service sides. This can expose missing/wrong points even
            # if the score state is already wrong.
            if is_first and observed and confidence >= self.min_confidence:
                game_key = (games_token, server, physical)
                previous_observed = last_first_side_by_game.get(game_key)
                if previous_observed == observed:
                    status = "MISMATCH"
                    details.append("consecutive first serves stayed on same side")
                last_first_side_by_game[game_key] = observed

            # First-serve fault does not change point parity: second serve must
            # stay on the same service side.
            point_key = (server, physical, score_before)
            if is_first and observed:
                first_attempt_by_point[point_key] = (observed, confidence)
            elif is_second and observed:
                first_info = first_attempt_by_point.get(point_key)
                if first_info and min(first_info[1], confidence) >= self.min_confidence:
                    if first_info[0] != observed:
                        status = "MISMATCH"
                        details.append(f"second serve changed side ({first_info[0]} -> {observed})")

            if status == "MISMATCH" and expected and observed:
                details.append(f"score expects {expected}, observed {observed}")

            checks.append(ServeCheck(
                row_number=row["_row_number"],
                point_index=point_index,
                frame=frame,
                server=server,
                attempt=attempt,
                score_before=score_before,
                expected=expected,
                observed=observed,
                confidence=confidence,
                x=x,
                status=status,
                detail="; ".join(details),
            ))
        return checks


def read_point_history(path: Path) -> List[dict]:
    # A live CSV may end in a partially written final line. Rows without a
    # point_index are ignored until the next watch poll.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row and (row.get("point_index") or "").strip():
                rows.append(row)
        return rows


def print_report(checks: Sequence[ServeCheck], only_problems: bool = False) -> int:
    problems = 0
    for check in checks:
        if only_problems and check.status in {"OK", "UNKNOWN", "LOW_CONFIDENCE"}:
            continue
        if check.status == "MISMATCH":
            problems += 1
        x_text = "?" if check.x is None else f"{check.x:.0f}"
        expected = check.expected or "?"
        observed = check.observed or "?"
        print(
            f"[SERVE_SIDE_{check.status}] point={check.point_index} f={check.frame} "
            f"server={check.server} attempt={check.attempt} score_before={check.score_before} "
            f"expected={expected} observed={observed} x={x_text} conf={check.confidence:.2f}"
            + (f" | {check.detail}" if check.detail else "")
        )
    print(f"[SERVE_SIDE_SUMMARY] checked={len(checks)} mismatches={problems}")
    return problems


def audit_file(path: Path, only_problems: bool = False) -> int:
    rows = read_point_history(path)
    guard = ServeSideConsistencyGuard()
    return print_report(guard.audit_rows(rows), only_problems=only_problems)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit tennis serve-side parity/alternation in point_history CSV")
    parser.add_argument("csv", type=Path, help="point_history_*.csv")
    parser.add_argument("--watch", action="store_true", help="recheck a growing CSV while tracker is still running")
    parser.add_argument("--interval", type=float, default=2.0, help="watch polling interval in seconds")
    parser.add_argument("--problems-only", action="store_true", help="show only mismatches plus summary")
    args = parser.parse_args()

    if not args.watch:
        return 1 if audit_file(args.csv, args.problems_only) else 0

    last_signature = None
    try:
        while True:
            try:
                stat = args.csv.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                if signature != last_signature:
                    print("\n" + "=" * 80)
                    print(f"[SERVE_SIDE_WATCH] {args.csv} size={stat.st_size}")
                    audit_file(args.csv, args.problems_only)
                    last_signature = signature
            except FileNotFoundError:
                print(f"[SERVE_SIDE_WATCH] waiting for {args.csv}")
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        print("\n[SERVE_SIDE_WATCH] stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
