#!/usr/bin/env python3
"""Generic tennis serve-side consistency guard and point-history auditor.

Tennis invariants:
- Even point parity in a game -> deuce/right service court.
- Odd point parity -> ad/left service court.
- A second serve is from the same side as the immediately preceding first serve
  for that tennis point.
- Consecutive *new points* in one game alternate service sides.
- A serve let (net touch + correct service-box bounce) replays the SAME serve:
  score, serve attempt and service side do not change.

The CSV auditor uses the SERVER position as the primary spatial observation.
The toss/ball position is only a fallback when server position is unavailable.
It never uses hard-coded frame numbers and never changes the score; it only
reports consistency problems.
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
_FIRST = {"1st", "first", "1"}
_SECOND = {"2nd", "second", "2"}


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
        return int(normalized)  # tiebreak score
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


def _is_serve_let_row(row: dict) -> bool:
    """Recognize a tracker-recorded let without relying on any frame number."""
    category = (row.get("category") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if category == "serve_let":
        return True
    for field in ("end_reason", "why"):
        if "serve let" in (row.get(field) or "").strip().lower():
            return True
    return False


def _same_server_score(a: Optional[dict], b: dict) -> bool:
    if a is None:
        return False
    return (
        (a.get("server") or "").strip() == (b.get("server") or "").strip()
        and a.get("_score_before") == b.get("_score_before")
    )


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
            (group0 if abs(value - c0) <= abs(value - c1) else group1).append(value)
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
    position_source: str
    is_let: bool
    replay_after_let: bool
    score_side_mismatch: bool
    first_second_mismatch: bool
    alternation_mismatch: bool
    let_replay_mismatch: bool
    status: str
    detail: str


class ServeSideConsistencyGuard:
    """Generic spatial serve-side validator.

    Camera convention used by the tracker:
    - near-side server: right/deuce side is high-x
    - far-side server:  right/deuce side is low-x

    The offline audit learns two server-position x clusters independently for
    each server/physical end. Low-confidence or unknown observations BREAK the
    alternation chain instead of being silently skipped. A serve let is treated
    as a replay of the same tennis serve, so it must keep the score, attempt and
    side unchanged and must NOT advance the first-serve alternation chain.
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

    @staticmethod
    def _row_observation(row: dict) -> Tuple[Optional[Point], str]:
        player_pos = _parse_position(row.get("serve_player_position", ""))
        if player_pos is not None:
            return player_pos, "server"
        ball_pos = _parse_position(row.get("serve_ball_position", ""))
        if ball_pos is not None:
            return ball_pos, "ball-fallback"
        return None, "none"

    @staticmethod
    def _classify_position(
        position: Optional[Point],
        physical: str,
        model: Optional[Tuple[float, float, float]],
    ) -> Tuple[Optional[str], float]:
        if position is None or model is None or physical not in {"near", "far"}:
            return None, 0.0
        low, high, midpoint = model
        x = position[0]
        if physical == "near":
            observed = "right" if x >= midpoint else "left"
        else:
            observed = "right" if x <= midpoint else "left"
        half_separation = max(1.0, (high - low) / 2.0)
        confidence = min(1.0, abs(x - midpoint) / half_separation)
        return observed, confidence

    def audit_rows(self, rows: Sequence[dict], initial_score: str = "0:0 0:0") -> List[ServeCheck]:
        enriched: List[dict] = []
        previous_score = initial_score
        for row_number, raw in enumerate(rows, start=2):
            row = dict(raw)
            score_before = (row.get("score_before") or previous_score or initial_score).strip()
            row["_score_before"] = score_before
            row["_row_number"] = row_number
            row["_serve_pos"], row["_position_source"] = self._row_observation(row)
            row["_is_let"] = _is_serve_let_row(row)
            enriched.append(row)
            current = (row.get("current_score") or "").strip()
            if current:
                previous_score = current

        # Learn two x clusters per server/physical end from FIRST serves only.
        # A repeated first serve after a let is excluded so one let does not
        # overweight one side of the cluster model.
        groups: Dict[Tuple[str, str], List[float]] = {}
        previous_row: Optional[dict] = None
        for row in enriched:
            attempt_norm = (row.get("serve_attempt") or "").strip().lower()
            replay_after_let = bool(
                previous_row is not None
                and previous_row.get("_is_let")
                and _same_server_score(previous_row, row)
                and (previous_row.get("serve_attempt") or "").strip().lower() == attempt_norm
            )
            if attempt_norm in _FIRST and not replay_after_let:
                position = row["_serve_pos"]
                if position is not None:
                    key = (
                        (row.get("server") or "").strip(),
                        (row.get("serve_player_side") or "").strip().lower(),
                    )
                    groups.setdefault(key, []).append(position[0])
            previous_row = row

        models: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        for key, xs in groups.items():
            centers = _kmeans_1d(xs)
            if centers is None:
                continue
            low, high = centers
            if high - low < self.min_cluster_separation_px:
                continue
            models[key] = (low, high, (low + high) / 2.0)

        checks: List[ServeCheck] = []
        last_first_side_by_game: Dict[Tuple[str, str, str], Optional[str]] = {}
        previous_row = None

        for row in enriched:
            score_before = row["_score_before"]
            expected = expected_serve_side(score_before)
            attempt = (row.get("serve_attempt") or "").strip()
            attempt_norm = attempt.lower()
            server = (row.get("server") or "").strip()
            physical = (row.get("serve_player_side") or "").strip().lower()
            point_index = (row.get("point_index") or "").strip()
            frame = (row.get("serve_start_frame") or "").strip()
            position = row["_serve_pos"]
            source = row["_position_source"]
            x = position[0] if position else None
            is_let = bool(row.get("_is_let"))
            details: List[str] = []

            model = models.get((server, physical))
            observed, confidence = self._classify_position(position, physical, model)
            if model is not None and position is not None and physical in {"near", "far"}:
                low, high, midpoint = model
                details.append(
                    f"source={source} clusters={low:.0f}/{high:.0f}px midpoint={midpoint:.0f}px"
                )
            elif source != "none":
                details.append(f"source={source}; no reliable two-cluster model")

            reliable = bool(observed and confidence >= self.min_confidence)
            score_side_mismatch = bool(reliable and expected and observed != expected)
            alternation_mismatch = False
            first_second_mismatch = False
            let_replay_mismatch = False

            is_first = attempt_norm in _FIRST
            is_second = attempt_norm in _SECOND
            games_token, _points_token = _split_score(score_before)

            # A let means the NEXT serve is a replay of this exact serve.
            replay_after_let = bool(
                previous_row is not None
                and previous_row.get("_is_let")
                and _same_server_score(previous_row, row)
            )
            if replay_after_let:
                prev_attempt = (previous_row.get("serve_attempt") or "").strip().lower()
                prev_physical = (previous_row.get("serve_player_side") or "").strip().lower()
                prev_model = models.get((server, prev_physical))
                prev_obs, prev_conf = self._classify_position(
                    previous_row.get("_serve_pos"), prev_physical, prev_model
                )
                if attempt_norm != prev_attempt:
                    let_replay_mismatch = True
                    details.append(
                        f"serve let must replay same attempt ({prev_attempt or '?'} -> {attempt_norm or '?'})"
                    )
                if prev_physical and physical and prev_physical != physical:
                    let_replay_mismatch = True
                    details.append(
                        f"serve let changed physical server end ({prev_physical} -> {physical})"
                    )
                if (
                    prev_obs is not None
                    and observed is not None
                    and min(prev_conf, confidence) >= self.min_confidence
                ):
                    if prev_obs != observed:
                        let_replay_mismatch = True
                        details.append(
                            f"serve let replay changed side ({prev_obs} -> {observed})"
                        )
                    else:
                        details.append(f"replay after serve let stayed on {observed}")
                else:
                    details.append("replay after serve let; side comparison uncertain")

            # New FIRST serves alternate. A repeated FIRST serve after a let is
            # explicitly NOT a new point and therefore must stay on the same side.
            if is_first:
                game_key = (games_token, server, physical)
                if replay_after_let:
                    details.append("first-serve let replay; alternation not advanced")
                else:
                    previous_observed = last_first_side_by_game.get(game_key)
                    if reliable:
                        if previous_observed is not None and previous_observed == observed:
                            alternation_mismatch = True
                            details.append("consecutive reliable new-point first serves stayed on same side")
                        last_first_side_by_game[game_key] = observed
                    else:
                        last_first_side_by_game[game_key] = None
                        details.append("alternation chain reset by uncertain first serve")

            # A normal second serve is paired only with the immediately
            # preceding FIRST-serve row for the same score state. If the previous
            # row was a let, replay rules above are authoritative instead.
            if is_second and previous_row is not None and not replay_after_let:
                prev_attempt = (previous_row.get("serve_attempt") or "").strip().lower()
                same_point_state = (
                    prev_attempt in _FIRST
                    and not previous_row.get("_is_let")
                    and (previous_row.get("server") or "").strip() == server
                    and (previous_row.get("serve_player_side") or "").strip().lower() == physical
                    and previous_row.get("_score_before") == score_before
                )
                if same_point_state:
                    prev_obs, prev_conf = self._classify_position(
                        previous_row.get("_serve_pos"), physical, model
                    )
                    prev_source = previous_row.get("_position_source", "none")
                    if (
                        prev_obs is not None
                        and observed is not None
                        and min(prev_conf, confidence) >= self.min_confidence
                        and prev_obs != observed
                    ):
                        first_second_mismatch = True
                        details.append(
                            f"immediate first/second serve changed side "
                            f"({prev_obs} -> {observed}; first source={prev_source})"
                        )

            if is_let:
                details.append("serve let: no point, same attempt and side must be replayed")
            if score_side_mismatch:
                details.append(f"score expects {expected}, observed {observed}")

            if let_replay_mismatch:
                status = "LET_REPLAY_MISMATCH"
            elif first_second_mismatch:
                status = "FIRST_SECOND_MISMATCH"
            elif score_side_mismatch:
                status = "SCORE_SIDE_MISMATCH"
            elif alternation_mismatch:
                status = "ALTERNATION_MISMATCH"
            elif expected and observed and confidence < self.min_confidence:
                status = "LOW_CONFIDENCE"
            elif not expected or not observed:
                status = "UNKNOWN"
            else:
                status = "OK"

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
                position_source=source,
                is_let=is_let,
                replay_after_let=replay_after_let,
                score_side_mismatch=score_side_mismatch,
                first_second_mismatch=first_second_mismatch,
                alternation_mismatch=alternation_mismatch,
                let_replay_mismatch=let_replay_mismatch,
                status=status,
                detail="; ".join(details),
            ))
            previous_row = row

        return checks


def read_point_history(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row and (row.get("point_index") or "").strip()
        ]


def print_report(checks: Sequence[ServeCheck], only_problems: bool = False) -> int:
    score_mismatches = sum(c.score_side_mismatch for c in checks)
    first_second_mismatches = sum(c.first_second_mismatch for c in checks)
    alternation_mismatches = sum(c.alternation_mismatch for c in checks)
    let_replay_mismatches = sum(c.let_replay_mismatch for c in checks)
    lets = sum(c.is_let for c in checks)
    let_replays = sum(c.replay_after_let for c in checks)
    low_confidence = sum(c.status == "LOW_CONFIDENCE" for c in checks)
    unknown = sum(c.status == "UNKNOWN" for c in checks)
    problem_serves = sum(
        c.score_side_mismatch
        or c.first_second_mismatch
        or c.alternation_mismatch
        or c.let_replay_mismatch
        for c in checks
    )

    for check in checks:
        is_problem = (
            check.score_side_mismatch
            or check.first_second_mismatch
            or check.alternation_mismatch
            or check.let_replay_mismatch
        )
        if only_problems and not is_problem:
            continue
        x_text = "?" if check.x is None else f"{check.x:.0f}"
        expected = check.expected or "?"
        observed = check.observed or "?"
        flags = []
        if check.is_let:
            flags.append("let=yes")
        if check.replay_after_let:
            flags.append("let_replay=yes")
        flag_text = (" " + " ".join(flags)) if flags else ""
        print(
            f"[SERVE_SIDE_{check.status}] point={check.point_index} f={check.frame} "
            f"server={check.server} attempt={check.attempt} score_before={check.score_before} "
            f"expected={expected} observed={observed} x={x_text} conf={check.confidence:.2f} "
            f"source={check.position_source}{flag_text}"
            + (f" | {check.detail}" if check.detail else "")
        )

    print(
        "[SERVE_SIDE_SUMMARY] "
        f"checked={len(checks)} problem_serves={problem_serves} "
        f"score_side_mismatches={score_mismatches} "
        f"first_second_mismatches={first_second_mismatches} "
        f"alternation_mismatches={alternation_mismatches} "
        f"let_replay_mismatches={let_replay_mismatches} "
        f"lets={lets} let_replays={let_replays} "
        f"low_confidence={low_confidence} unknown={unknown}"
    )
    return problem_serves


def audit_file(path: Path, only_problems: bool = False) -> int:
    rows = read_point_history(path)
    guard = ServeSideConsistencyGuard()
    return print_report(guard.audit_rows(rows), only_problems=only_problems)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit tennis serve-side parity/alternation in point_history CSV")
    parser.add_argument("csv", type=Path, help="point_history_*.csv")
    parser.add_argument("--watch", action="store_true", help="recheck a growing CSV while tracker is still running")
    parser.add_argument("--interval", type=float, default=2.0, help="watch polling interval in seconds")
    parser.add_argument("--problems-only", action="store_true", help="show only true mismatch rows plus summary")
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
