#!/usr/bin/env python3
"""Generic tennis serve-side consistency guard and point-history auditor.

Tennis invariants:
- Even point parity in a game -> deuce/right service court.
- Odd point parity -> ad/left service court.
- A second serve is from the same side as the immediately preceding first serve.
- Consecutive NEW points in one game alternate service sides.
- A let replays the SAME serve: score, serve attempt and service side do not change.

Video let inference is deliberately retrospective. A tiny net-cord touch may be
impossible to see reliably, so an unlabelled let candidate is inferred only
from a sequence:
  1. the previous serve has evidence that it was IN,
  2. the rally is very short (0 or 1 return/hit),
  3. the immediately following serve is by the same server,
  4. it starts from the same physical/service side.

The tracker serve-attempt label is NOT a prerequisite for inferred lets. If the
tracker mistakenly called a first-serve let a fault, it may label the replay as
2nd; that attempt change is itself evidence of tracker-state error after the
same-side replay has confirmed the let. A direct tracker "serve let" label is
accepted, but visible net contact is not required for retrospective inference.
No frame numbers are hard-coded.
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
_TRUE = {"1", "true", "yes", "y", "in"}
_SERVICE_FAULT_CATEGORIES = {"first_serve_fault", "double_fault", "serve_fault"}


def _parse_position(value: str) -> Optional[Point]:
    if not value:
        return None
    match = _POS_RE.fullmatch(value.strip())
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _split_score(score: str) -> Tuple[str, str]:
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
        return int(normalized)
    except ValueError:
        return None


def point_parity(score: str) -> Optional[int]:
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


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _is_serve_let_row(row: dict) -> bool:
    category = _norm(row.get("category")).replace("-", "_").replace(" ", "_")
    if category == "serve_let":
        return True
    return any("serve let" in _norm(row.get(field)) for field in ("end_reason", "why"))


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value or "").strip()))
    except (TypeError, ValueError):
        return default


def _score_changed(row: dict) -> bool:
    before = _norm(row.get("_score_before"))
    after = _norm(row.get("current_score"))
    return bool(before and after and before != after)


def _service_fault_row(row: dict) -> bool:
    category = _norm(row.get("category")).replace("-", "_").replace(" ", "_")
    text = " ".join(_norm(row.get(k)) for k in ("end_reason", "why", "category"))
    if category in _SERVICE_FAULT_CATEGORIES:
        return True
    fault_tokens = (
        "first serve fault",
        "double fault",
        "serve bounce outside",
        "serve net then bounce outside",
        "serve net then bounce before",
    )
    return any(token in text for token in fault_tokens)


def _serve_in_evidence(row: dict) -> Tuple[bool, str]:
    """Return whether the previous serve can reasonably be treated as in.

    Future CSVs may expose a direct serve-in field; use it first. For older
    histories, a receiver return is strong evidence. With zero returns, a short
    non-fault sequence that the tracker treated as live is weaker behavioural
    evidence and is used only together with immediate same-side replay.
    """
    if _is_serve_let_row(row):
        return True, "explicit-let"

    for field in ("serve_in", "serve_landed_in", "serve_was_in"):
        raw = _norm(row.get(field))
        if raw:
            return raw in _TRUE, f"{field}={raw}"

    if _service_fault_row(row):
        return False, "service-fault"

    rally_shots = _safe_int(row.get("rally_shots"), 0)
    if rally_shots >= 1:
        return True, "receiver-returned-serve"

    awarded = _norm(row.get("point_awarded")) in _TRUE
    if rally_shots == 0 and awarded:
        return True, "tracker-treated-short-serve-as-live"

    if rally_shots == 0 and not _service_fault_row(row):
        return True, "short-nonfault-serve"

    return False, "no-serve-in-evidence"


def _kmeans_1d(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    if len(values) < 4:
        return None
    low, high = min(values), max(values)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return None
    c0, c1 = low, high
    for _ in range(32):
        g0: List[float] = []
        g1: List[float] = []
        for value in values:
            (g0 if abs(value - c0) <= abs(value - c1) else g1).append(value)
        if not g0 or not g1:
            return None
        n0, n1 = sum(g0) / len(g0), sum(g1) / len(g1)
        if abs(n0 - c0) < 0.01 and abs(n1 - c1) < 0.01:
            c0, c1 = n0, n1
            break
        c0, c1 = n0, n1
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
    inferred_let_replay: bool
    missed_let_candidate: bool
    replay_after_let: bool
    score_side_mismatch: bool
    first_second_mismatch: bool
    alternation_mismatch: bool
    let_replay_mismatch: bool
    status: str
    detail: str


class ServeSideConsistencyGuard:
    def __init__(self, min_cluster_separation_px: float = 160.0, min_confidence: float = 0.25):
        self.min_cluster_separation_px = float(min_cluster_separation_px)
        self.min_confidence = float(min_confidence)

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
    def _classify_position(position, physical, model):
        if position is None or model is None or physical not in {"near", "far"}:
            return None, 0.0
        low, high, midpoint = model
        x = position[0]
        if physical == "near":
            observed = "right" if x >= midpoint else "left"
        else:
            observed = "right" if x <= midpoint else "left"
        half = max(1.0, (high - low) / 2.0)
        return observed, min(1.0, abs(x - midpoint) / half)

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

        groups: Dict[Tuple[str, str], List[float]] = {}
        for row in enriched:
            if _norm(row.get("serve_attempt")) not in _FIRST:
                continue
            pos = row["_serve_pos"]
            if pos is None:
                continue
            key = (_norm(row.get("server")), _norm(row.get("serve_player_side")))
            groups.setdefault(key, []).append(pos[0])

        models: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        for key, xs in groups.items():
            centers = _kmeans_1d(xs)
            if centers is None:
                continue
            low, high = centers
            if high - low >= self.min_cluster_separation_px:
                models[key] = (low, high, (low + high) / 2.0)

        for row in enriched:
            server = _norm(row.get("server"))
            physical = _norm(row.get("serve_player_side"))
            model = models.get((server, physical))
            obs, conf = self._classify_position(row["_serve_pos"], physical, model)
            row["_observed"] = obs
            row["_confidence"] = conf
            row["_model"] = model

        replay_info: Dict[int, dict] = {}
        for i in range(1, len(enriched)):
            prev, cur = enriched[i - 1], enriched[i]
            same_server = _norm(prev.get("server")) == _norm(cur.get("server"))
            same_physical = _norm(prev.get("serve_player_side")) == _norm(cur.get("serve_player_side"))
            same_attempt = _norm(prev.get("serve_attempt")) == _norm(cur.get("serve_attempt"))
            same_score_state = prev.get("_score_before") == cur.get("_score_before")
            prev_obs, cur_obs = prev.get("_observed"), cur.get("_observed")
            prev_conf = float(prev.get("_confidence") or 0.0)
            cur_conf = float(cur.get("_confidence") or 0.0)
            same_reliable_side = bool(
                prev_obs and cur_obs
                and min(prev_conf, cur_conf) >= self.min_confidence
                and prev_obs == cur_obs
            )

            explicit = bool(prev.get("_is_let") and same_server)
            serve_in, serve_in_source = _serve_in_evidence(prev)
            short_rally = _safe_int(prev.get("rally_shots"), 0) <= 1
            inferred = bool(
                not prev.get("_is_let")
                and same_server
                and same_physical
                and same_reliable_side
                and serve_in
                and short_rally
            )

            if explicit or inferred:
                score_advanced = _score_changed(prev) or not same_score_state
                replay_info[i] = {
                    "explicit": explicit,
                    "inferred": inferred,
                    "score_advanced": score_advanced,
                    "attempt_changed": not same_attempt,
                    "previous_attempt": _norm(prev.get("serve_attempt")),
                    "current_attempt": _norm(cur.get("serve_attempt")),
                    "serve_in_source": serve_in_source,
                }

        checks: List[ServeCheck] = []
        last_first_side_by_game: Dict[Tuple[str, str, str], Optional[str]] = {}
        previous_row: Optional[dict] = None

        for i, row in enumerate(enriched):
            score_before = row["_score_before"]
            expected = expected_serve_side(score_before)
            attempt = (row.get("serve_attempt") or "").strip()
            attempt_norm = attempt.lower()
            server = (row.get("server") or "").strip()
            server_norm = server.lower()
            physical = _norm(row.get("serve_player_side"))
            point_index = (row.get("point_index") or "").strip()
            frame = (row.get("serve_start_frame") or "").strip()
            position = row["_serve_pos"]
            source = row["_position_source"]
            x = position[0] if position else None
            observed = row.get("_observed")
            confidence = float(row.get("_confidence") or 0.0)
            reliable = bool(observed and confidence >= self.min_confidence)
            is_let = bool(row.get("_is_let"))
            info = replay_info.get(i)
            replay_after_let = info is not None
            inferred_let_replay = bool(info and info["inferred"])
            missed_let_candidate = bool(
                info and info["inferred"]
                and (info["score_advanced"] or info["attempt_changed"])
            )
            details: List[str] = []

            model = row.get("_model")
            if model is not None and position is not None:
                low, high, midpoint = model
                details.append(
                    f"source={source} clusters={low:.0f}/{high:.0f}px midpoint={midpoint:.0f}px"
                )
            elif source != "none":
                details.append(f"source={source}; no reliable two-cluster model")

            if inferred_let_replay:
                prev_shots = _safe_int(previous_row.get("rally_shots") if previous_row else 0)
                details.append(
                    "retrospective let sequence: previous serve in "
                    f"({info['serve_in_source']}), rally_shots={prev_shots}, "
                    f"same server and same reliable service side={observed}"
                )
                if info["attempt_changed"]:
                    details.append(
                        "tracker attempt changed "
                        f"({info['previous_attempt']} -> {info['current_attempt']}); "
                        "attempt label is state to correct, not a let prerequisite"
                    )
                if info["score_advanced"]:
                    details.append("tracker score/point advanced across the apparent let replay")
                if missed_let_candidate:
                    details.append("previous sequence appears to be a MISSED LET")
                else:
                    details.append("inferred let replay; score and attempt remained consistent")

            score_side_mismatch = bool(reliable and expected and observed != expected)
            if missed_let_candidate:
                score_side_mismatch = False

            alternation_mismatch = False
            first_second_mismatch = False
            let_replay_mismatch = False
            is_first = attempt_norm in _FIRST
            is_second = attempt_norm in _SECOND
            games_token, _ = _split_score(score_before)

            if info and info["explicit"] and previous_row is not None:
                prev_attempt = _norm(previous_row.get("serve_attempt"))
                prev_physical = _norm(previous_row.get("serve_player_side"))
                prev_obs = previous_row.get("_observed")
                prev_conf = float(previous_row.get("_confidence") or 0.0)
                if attempt_norm != prev_attempt:
                    let_replay_mismatch = True
                    details.append(f"serve let must replay same attempt ({prev_attempt} -> {attempt_norm})")
                if prev_physical != physical:
                    let_replay_mismatch = True
                    details.append(f"serve let changed physical end ({prev_physical} -> {physical})")
                if (
                    prev_obs and observed
                    and min(prev_conf, confidence) >= self.min_confidence
                    and prev_obs != observed
                ):
                    let_replay_mismatch = True
                    details.append(f"serve let replay changed side ({prev_obs} -> {observed})")

            if is_first:
                game_key = (games_token, server_norm, physical)
                if replay_after_let:
                    details.append("let replay: first-serve alternation NOT advanced")
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

            if is_second and previous_row is not None and not replay_after_let:
                prev_attempt = _norm(previous_row.get("serve_attempt"))
                same_point_state = (
                    prev_attempt in _FIRST
                    and not previous_row.get("_is_let")
                    and _norm(previous_row.get("server")) == server_norm
                    and _norm(previous_row.get("serve_player_side")) == physical
                    and previous_row.get("_score_before") == score_before
                )
                if same_point_state:
                    prev_obs = previous_row.get("_observed")
                    prev_conf = float(previous_row.get("_confidence") or 0.0)
                    if (
                        prev_obs and observed
                        and min(prev_conf, confidence) >= self.min_confidence
                        and prev_obs != observed
                    ):
                        first_second_mismatch = True
                        details.append(f"immediate first/second serve changed side ({prev_obs} -> {observed})")

            if is_let:
                details.append("tracker-recorded serve let: next serve must replay same attempt and side")
            if score_side_mismatch:
                details.append(f"score expects {expected}, observed {observed}")

            if let_replay_mismatch:
                status = "LET_REPLAY_MISMATCH"
            elif missed_let_candidate:
                status = "MISSED_LET_CANDIDATE"
            elif inferred_let_replay:
                status = "INFERRED_LET_REPLAY"
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
                inferred_let_replay=inferred_let_replay,
                missed_let_candidate=missed_let_candidate,
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
        return [row for row in csv.DictReader(handle) if row and (row.get("point_index") or "").strip()]


def print_report(checks: Sequence[ServeCheck], only_problems: bool = False) -> int:
    score_mismatches = sum(c.score_side_mismatch for c in checks)
    first_second_mismatches = sum(c.first_second_mismatch for c in checks)
    alternation_mismatches = sum(c.alternation_mismatch for c in checks)
    let_replay_mismatches = sum(c.let_replay_mismatch for c in checks)
    lets = sum(c.is_let for c in checks)
    inferred_lets = sum(c.inferred_let_replay for c in checks)
    missed_lets = sum(c.missed_let_candidate for c in checks)
    low_confidence = sum(c.status == "LOW_CONFIDENCE" for c in checks)
    unknown = sum(c.status == "UNKNOWN" for c in checks)
    problem_serves = sum(
        c.score_side_mismatch
        or c.first_second_mismatch
        or c.alternation_mismatch
        or c.let_replay_mismatch
        or c.missed_let_candidate
        for c in checks
    )

    for check in checks:
        is_problem = (
            check.score_side_mismatch
            or check.first_second_mismatch
            or check.alternation_mismatch
            or check.let_replay_mismatch
            or check.missed_let_candidate
        )
        if only_problems and not is_problem:
            continue
        x_text = "?" if check.x is None else f"{check.x:.0f}"
        expected = check.expected or "?"
        observed = check.observed or "?"
        flags = []
        if check.is_let:
            flags.append("let=yes")
        if check.inferred_let_replay:
            flags.append("inferred_let_replay=yes")
        if check.missed_let_candidate:
            flags.append("missed_let_candidate=yes")
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
        f"tracker_lets={lets} inferred_let_replays={inferred_lets} "
        f"missed_let_candidates={missed_lets} "
        f"low_confidence={low_confidence} unknown={unknown}"
    )
    return problem_serves


def audit_file(path: Path, only_problems: bool = False) -> int:
    rows = read_point_history(path)
    guard = ServeSideConsistencyGuard()
    return print_report(guard.audit_rows(rows), only_problems=only_problems)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit tennis serve-side parity/alternation and retrospective let replays"
    )
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
