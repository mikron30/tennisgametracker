"""Retrospective serve-let rollback for the live tennis scorer.

Normal points are always scored immediately.  A rare missed let is confirmed
retrospectively only when the immediately following serve is by the same server
and starts from the same *server-position* service side.

The important distinction is deliberate: service side is inferred from the
server player's X position, not from the ball/toss X.  Toss/contact positions
move too much to be a reliable deuce/ad-side signal.
"""
from __future__ import annotations

import copy
import csv
import math
import os
import sys
import tempfile
from functools import wraps

_PATCHED = False
_TRACE_INSTALLED = False
_PREVIOUS_TRACE = None

_MIN_CLUSTER_SAMPLES = 4
_MIN_CLUSTER_SEPARATION_PX = 160.0
_MIN_CLUSTER_CONFIDENCE = 0.25
_EARLY_SAME_SIDE_DX_PX = 150.0
_EARLY_OPPOSITE_SIDE_DX_PX = 240.0
_MAX_REPLAY_GAP_FRAMES = 1200
# Shot events exclude the serve itself. A let can still be declared before
# the server plays the first ball after the receiver's return: 0 = no return,
# 1 = receiver returned serve, 2+ = server has already continued the rally.
_MAX_SHORT_RALLY_SHOTS = 1


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _score_snapshot(obj):
    """Capture score/stat state before an apparent point is awarded."""
    fields = (
        "point_stats",
        "serve_stats",
    )
    snapshot = {
        "score_points": list(getattr(obj, "score_points", [0, 0])),
        "score_games": list(getattr(obj, "score_games", [0, 0])),
        "score_game_index": int(getattr(obj, "score_game_index", 0)),
        "current_serve_attempt": int(getattr(obj, "current_serve_attempt", 1)),
        "last_scored_point_end_frame": getattr(obj, "_last_scored_point_end_frame", -1),
        "last_point_winner": getattr(obj, "_last_point_winner", None),
        "last_point_score_reason": getattr(obj, "_last_point_score_reason", None),
        "last_point_outcome_category": getattr(obj, "_last_point_outcome_category", None),
        "last_point_hit_count": getattr(obj, "_last_point_hit_count", 0),
        "last_confirmed_point_end_frame": getattr(obj, "_last_confirmed_point_end_frame", None),
    }
    for field in fields:
        snapshot[field] = copy.deepcopy(getattr(obj, field, None))
    return snapshot


def _restore_score_snapshot(obj, snapshot, *, remove_counted_serve_in=False, server_idx=None, attempt=1):
    obj.score_points = list(snapshot["score_points"])
    obj.score_games = list(snapshot["score_games"])
    obj.score_game_index = int(snapshot["score_game_index"])
    obj.current_serve_attempt = int(snapshot["current_serve_attempt"])

    if snapshot.get("point_stats") is not None:
        obj.point_stats = copy.deepcopy(snapshot["point_stats"])
    if snapshot.get("serve_stats") is not None:
        obj.serve_stats = copy.deepcopy(snapshot["serve_stats"])

    obj._last_scored_point_end_frame = snapshot.get("last_scored_point_end_frame", -1)
    obj._last_point_winner = snapshot.get("last_point_winner")
    obj._last_point_score_reason = snapshot.get("last_point_score_reason")
    obj._last_point_outcome_category = snapshot.get("last_point_outcome_category")
    obj._last_point_hit_count = snapshot.get("last_point_hit_count", 0)
    obj._last_confirmed_point_end_frame = snapshot.get("last_confirmed_point_end_frame")

    # The snapshot is taken at point end, after a successful serve may already
    # have been counted.  A let is not a counted serve-in, so remove exactly
    # that one recorded in-serve after restoring the snapshot.
    if remove_counted_serve_in:
        stats_all = getattr(obj, "serve_stats", None)
        idx = _safe_int(server_idx, -1)
        if isinstance(stats_all, list) and 0 <= idx < len(stats_all):
            stats = stats_all[idx]
            if isinstance(stats, dict):
                key = "first_in" if _safe_int(attempt, 1) <= 1 else "second_in"
                stats[key] = max(0, _safe_int(stats.get(key), 0) - 1)

    obj._serve_landed_in_current_attempt = False
    obj._serve_in_recorded_attempt = None
    obj._serve_start_requires_confirmation = False


def _serve_label(obj):
    fn = getattr(obj, "_serve_attempt_label", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    return "1st" if _safe_int(getattr(obj, "current_serve_attempt", 1), 1) <= 1 else "2nd"


def _score_text(obj):
    fn = getattr(obj, "_score_summary", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    games = getattr(obj, "score_games", [0, 0])
    points = getattr(obj, "score_points", [0, 0])
    return f"{games[0]}:{games[1]} {points[0]}:{points[1]}"


def _service_fault_reason(reason, category=None):
    text = " ".join((str(reason or ""), str(category or ""))).lower()
    normalized = str(category or "").lower().replace("-", "_").replace(" ", "_")
    if normalized in {"first_serve_fault", "double_fault", "serve_fault"}:
        return True
    return any(token in text for token in (
        "first serve fault",
        "double fault",
        "serve net then bounce outside",
        "serve bounce outside",
    ))


def _point_context(obj):
    context = getattr(obj, "_point_history_current", None)
    return context if isinstance(context, dict) else None


def _observation_from_context_dict(context):
    if not isinstance(context, dict):
        return None
    serve_context = context.get("serve_context")
    if not isinstance(serve_context, dict):
        return None
    position = serve_context.get("player_position")
    if not isinstance(position, (tuple, list)) or len(position) < 2:
        return None
    try:
        x = float(position[0])
        y = float(position[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    physical = str(serve_context.get("player_side") or "").strip().lower()
    if physical not in {"near", "far"}:
        physical = None
    source = str(serve_context.get("player_source") or serve_context.get("source") or "server")
    return {
        "x": x,
        "y": y,
        "physical_side": physical,
        "source": source,
    }


def _current_observation(obj):
    observed = _observation_from_context_dict(_point_context(obj))
    if observed is not None:
        obj._retro_let_current_observation = observed
        return observed
    cached = getattr(obj, "_retro_let_current_observation", None)
    return dict(cached) if isinstance(cached, dict) else None


def _context_observation_from_origin(obj, origin_pos):
    fn = getattr(obj, "_player_serve_context", None)
    if not callable(fn):
        return None
    try:
        context = fn(origin_pos)
    except Exception:
        return None
    return _observation_from_context_dict({"serve_context": context})


def _rally_shot_events(obj):
    context = _point_context(obj)
    if isinstance(context, dict):
        events = context.get("shot_events")
        if isinstance(events, list):
            return list(events)
    return []


def _rally_shots(obj):
    context = _point_context(obj)
    if isinstance(context, dict):
        events = context.get("shot_events")
        if isinstance(events, list):
            return len(events)
    return _safe_int(getattr(obj, "_point_hit_count", 0), 0)


def _post_serve_shot_player(obj, event):
    if not isinstance(event, dict):
        return None
    player = str(event.get("player") or "").strip()
    names = getattr(obj, "player_names", None)
    if not player or not isinstance(names, (list, tuple)):
        return None
    return player if player in {str(name).strip() for name in names} else None


def _let_window_open(obj, rally_shots, server_idx, events=None):
    """Return True only while a retrospective let is still plausible.

    Shot events exclude the serve itself.  A let may survive no post-serve shot,
    or one positively identified receiver return.  Once the server hits the
    next ball the rally has continued and a let is impossible.  Unknown shooter
    identity is deliberately not treated as positive evidence for a let.
    """
    shots = _safe_int(rally_shots, 0)
    if shots <= 0:
        return True
    if shots != 1:
        return False

    if events is None:
        events = _rally_shot_events(obj)
    if not isinstance(events, list) or len(events) != 1:
        return False

    names = getattr(obj, "player_names", None)
    idx = _safe_int(server_idx, -1)
    if not isinstance(names, (list, tuple)) or not (0 <= idx < len(names)):
        return False

    server_name = str(names[idx]).strip()
    shooter = _post_serve_shot_player(obj, events[0])
    if shooter is None:
        return False
    return shooter != server_name


def _server_idx_for_current_context(obj):
    context = _point_context(obj)
    if isinstance(context, dict):
        value = context.get("server_idx")
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    try:
        return int(obj._current_server_index())
    except Exception:
        return _safe_int(getattr(obj, "score_game_index", 0), 0) % 2


def _kmeans_1d(values):
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(finite) < _MIN_CLUSTER_SAMPLES:
        return None
    low, high = min(finite), max(finite)
    if high <= low:
        return None
    c0, c1 = low, high
    for _ in range(32):
        g0, g1 = [], []
        for value in finite:
            (g0 if abs(value - c0) <= abs(value - c1) else g1).append(value)
        if not g0 or not g1:
            return None
        n0, n1 = sum(g0) / len(g0), sum(g1) / len(g1)
        if abs(n0 - c0) < 0.01 and abs(n1 - c1) < 0.01:
            c0, c1 = n0, n1
            break
        c0, c1 = n0, n1
    low_center, high_center = min(c0, c1), max(c0, c1)
    if high_center - low_center < _MIN_CLUSTER_SEPARATION_PX:
        return None
    return low_center, high_center, (low_center + high_center) / 2.0


def _classify_service_side(x, physical_side, model):
    if model is None or physical_side not in {"near", "far"}:
        return None, 0.0
    low, high, midpoint = model
    x = float(x)
    if physical_side == "near":
        side = "right" if x >= midpoint else "left"
    else:
        side = "right" if x <= midpoint else "left"
    half = max(1.0, (high - low) / 2.0)
    confidence = min(1.0, abs(x - midpoint) / half)
    return side, confidence


def _sample_store(obj):
    store = getattr(obj, "_retro_let_side_samples", None)
    if not isinstance(store, dict):
        store = {}
        obj._retro_let_side_samples = store
    return store


def _remember_observation(obj, server_idx, observation, frame_index):
    if not isinstance(observation, dict):
        return
    physical = observation.get("physical_side")
    x = observation.get("x")
    if physical not in {"near", "far"} or not isinstance(x, (int, float)):
        return
    key = (int(server_idx), physical)
    store = _sample_store(obj)
    samples = store.setdefault(key, [])
    marker = (int(frame_index), float(x))
    if samples and samples[-1][0] == marker[0]:
        samples[-1] = marker
    else:
        samples.append(marker)
    if len(samples) > 48:
        del samples[:-48]


def _side_comparison(obj, pending, current_observation):
    """Return (decision, detail): decision is same/opposite/unknown."""
    if not isinstance(current_observation, dict):
        return "unknown", "server position unavailable"

    previous = pending.get("observation")
    if not isinstance(previous, dict):
        return "unknown", "previous server position unavailable"

    previous_physical = previous.get("physical_side")
    current_physical = current_observation.get("physical_side")
    if (
        previous_physical in {"near", "far"} and
        current_physical in {"near", "far"} and
        previous_physical != current_physical
    ):
        # For one immediate replay the same server should still be on the same
        # physical end.  A change here is not safe evidence for a let.
        return "opposite", f"physical_side={previous_physical}->{current_physical}"

    physical = current_physical or previous_physical
    try:
        previous_x = float(previous["x"])
        current_x = float(current_observation["x"])
    except (KeyError, TypeError, ValueError):
        return "unknown", "server X unavailable"

    # Build the same two-cluster server-position model used by the offline
    # consistency checker.  Include both positions being compared so the model
    # matures naturally during the first few points.
    history = []
    if physical in {"near", "far"}:
        key = (int(pending["server_idx"]), physical)
        history = [x for _frame, x in _sample_store(obj).get(key, [])]
    values = history + [previous_x, current_x]
    model = _kmeans_1d(values)
    if model is not None and physical in {"near", "far"}:
        old_side, old_conf = _classify_service_side(previous_x, physical, model)
        new_side, new_conf = _classify_service_side(current_x, physical, model)
        if (
            old_side is not None and new_side is not None and
            old_conf >= _MIN_CLUSTER_CONFIDENCE and new_conf >= _MIN_CLUSTER_CONFIDENCE
        ):
            decision = "same" if old_side == new_side else "opposite"
            return (
                decision,
                f"model={old_side}->{new_side} x={previous_x:.0f}->{current_x:.0f} "
                f"conf={old_conf:.2f}/{new_conf:.2f} "
                f"clusters={model[0]:.0f}/{model[1]:.0f}",
            )

    # Early in a match there may not yet be four reliable serve positions.
    # A tight same-position repeat is strong positive evidence; a large shift is
    # strong negative evidence.  The uncertain middle band is deliberately not
    # forced either way until the serve context refreshes or the rally ends.
    dx = abs(current_x - previous_x)
    if dx <= _EARLY_SAME_SIDE_DX_PX:
        return "same", f"early-dx={dx:.0f}px x={previous_x:.0f}->{current_x:.0f}"
    if dx >= _EARLY_OPPOSITE_SIDE_DX_PX:
        return "opposite", f"early-dx={dx:.0f}px x={previous_x:.0f}->{current_x:.0f}"
    return "unknown", f"ambiguous-dx={dx:.0f}px x={previous_x:.0f}->{current_x:.0f}"


def _rewrite_history_row_as_let(obj, pending):
    if not bool(getattr(obj, "write_point_history", False)):
        return
    path = str(getattr(obj, "point_history_file", "") or "")
    if not path or not os.path.isfile(path):
        return
    end_frame = str(pending.get("end_frame", ""))
    if not end_frame:
        return
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        target = None
        for row in reversed(rows):
            if str(row.get("point_end_frame") or "").strip() == end_frame:
                target = row
                break
        if target is None:
            return
        updates = {
            "end_reason": "Serve let (retrospective same-side replay)",
            "why": "short nonfault rally + immediate same-server service-side replay",
            "category": "serve_let",
            "point_awarded": "no",
            "winner": "",
            "current_score": _score_text(obj),
            "next_server": obj.player_names[obj._current_server_index()],
            "next_serve": _serve_label(obj),
        }
        for key, value in updates.items():
            if key in fieldnames:
                target[key] = value
        directory = os.path.dirname(os.path.abspath(path)) or "."
        fd, temp_path = tempfile.mkstemp(prefix="retro_let_", suffix=".csv", dir=directory)
        os.close(fd)
        try:
            with open(temp_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as exc:
        print(f"[RETRO_LET_HISTORY_WARNING] could not reconcile CSV: {exc}")


def _repair_current_history_after_rollback(obj):
    context = _point_context(obj)
    if not isinstance(context, dict):
        return
    context["server_idx"] = int(obj._current_server_index())
    context["serve_attempt"] = _serve_label(obj)


def _confirm_pending(obj, pending, detail, frame_now):
    temporary_score = _score_text(obj)
    _restore_score_snapshot(
        obj,
        pending["snapshot"],
        remove_counted_serve_in=bool(pending.get("serve_stat_counted")),
        server_idx=pending.get("server_idx"),
        attempt=pending.get("attempt", 1),
    )
    _repair_current_history_after_rollback(obj)
    _rewrite_history_row_as_let(obj, pending)
    print(
        f"[RETRO_LET_CONFIRMED] f{frame_now}: previous=f{pending['end_frame']} "
        f"server={obj.player_names[int(pending['server_idx'])]} hits={pending['hits']} {detail}"
    )
    print(
        f"[RETRO_LET_ROLLBACK] f{frame_now}: score={temporary_score}->{_score_text(obj)} "
        f"replay={_serve_label(obj)}"
    )
    obj._retro_let_pending = None
    return "confirmed"


def _discard_pending(obj, pending, detail, frame_now):
    print(
        f"[RETRO_LET_DISCARDED] f{frame_now}: previous=f{pending['end_frame']} {detail}"
    )
    obj._retro_let_pending = None
    return "discarded"


def _evaluate_pending(obj, *, final=False):
    pending = getattr(obj, "_retro_let_pending", None)
    if not isinstance(pending, dict):
        return "none"

    frame_now = int(getattr(obj, "frame_count", -1))
    gap = frame_now - int(pending.get("end_frame", frame_now))
    if gap < 0 or gap > _MAX_REPLAY_GAP_FRAMES:
        return _discard_pending(obj, pending, f"gap={gap}f", frame_now)

    current_observation = _current_observation(obj)
    current_server_idx = _server_idx_for_current_context(obj)
    before_game_index = int(pending["snapshot"]["score_game_index"])
    temporary_game_index = int(getattr(obj, "score_game_index", before_game_index))
    temp_completed_game = temporary_game_index != before_game_index

    # A temporary game-winning point changes score-derived server identity.
    # Without stable player identity we deliberately refuse to guess across that
    # boundary; physical near/far alone is ambiguous because players may change
    # ends after the game.
    if temp_completed_game:
        if final:
            return _discard_pending(
                obj, pending,
                "temporary point crossed game boundary; automatic let rollback withheld",
                frame_now,
            )
        return "unknown"

    if int(current_server_idx) != int(pending["server_idx"]):
        return _discard_pending(
            obj, pending,
            f"server={pending['server_idx']}->{current_server_idx}",
            frame_now,
        )

    decision, detail = _side_comparison(obj, pending, current_observation)
    if decision == "same":
        return _confirm_pending(obj, pending, detail, frame_now)
    if decision == "opposite":
        return _discard_pending(obj, pending, detail, frame_now)
    if final:
        return _discard_pending(obj, pending, f"{detail}; final observation unresolved", frame_now)
    return "unknown"


def _patch_tracker_class(cls):
    global _PATCHED
    if _PATCHED or getattr(cls, "_retro_let_scoring_installed", False):
        return

    original_record = cls._record_point_result
    original_start = cls._start_point_context
    original_refresh = getattr(cls, "_refresh_player_serve_context", None)

    @wraps(original_record)
    def record_point_result(self, reason, end_position=None, frame=None, history_end_frame=None):
        # If the next serve began with ambiguous geometry, its final serve
        # context is the last safe chance to confirm the previous let.  Do this
        # BEFORE scoring the current rally so a confirmed replay scores exactly
        # once from the restored state.
        _evaluate_pending(self, final=True)

        observation = _current_observation(self)
        server_idx = _server_idx_for_current_context(self)
        attempt = _safe_int(getattr(self, "current_serve_attempt", 1), 1)
        rally_events = _rally_shot_events(self)
        rally_shots = _rally_shots(self)
        let_window_open = _let_window_open(self, rally_shots, server_idx, rally_events)
        snapshot = _score_snapshot(self)
        pre_scored_frame = getattr(self, "_last_scored_point_end_frame", None)
        serve_stat_counted = bool(getattr(self, "_serve_landed_in_current_attempt", False))
        start_frame = getattr(self, "point_start_frame_internal", None)

        result = original_record(
            self,
            reason,
            end_position=end_position,
            frame=frame,
            history_end_frame=history_end_frame,
        )

        frame_now = int(getattr(self, "frame_count", -1))
        _remember_observation(self, server_idx, observation, start_frame if start_frame is not None else frame_now)

        winner = getattr(self, "_last_point_winner", None)
        category = getattr(self, "_last_point_outcome_category", None)
        scored_now = (
            getattr(self, "_last_scored_point_end_frame", None) == frame_now and
            pre_scored_frame != frame_now
        )
        awarded = scored_now and winner is not None
        nonfault = not _service_fault_reason(reason, category)
        short = let_window_open

        if awarded and nonfault and not short and rally_shots == 1:
            event = rally_events[0] if len(rally_events) == 1 and isinstance(rally_events[0], dict) else {}
            shooter = _post_serve_shot_player(self, event)
            print(
                f"[RETRO_LET_CLOSED_AFTER_SHOT] f{frame_now}: "
                f"server={self.player_names[int(server_idx)]} "
                f"shot_player={shooter or '?'} "
                f"shot_frame={event.get('frame', '?')} "
                f"label={event.get('label', '?')}"
            )

        # Every apparent point is already scored by original_record above.
        # We only arm a reversible candidate.  Same-side replay on the next
        # physical serve is the confirmation; otherwise this score stands.
        if awarded and nonfault and short and isinstance(observation, dict):
            if rally_shots >= 1:
                in_evidence = "receiver-returned-serve"
            elif serve_stat_counted:
                in_evidence = "serve-in-state"
            else:
                in_evidence = "tracker-treated-short-serve-as-live"

            self._retro_let_pending = {
                "snapshot": snapshot,
                "server_idx": int(server_idx),
                "attempt": int(attempt),
                "observation": dict(observation),
                "start_frame": start_frame,
                "end_frame": frame_now,
                "hits": int(rally_shots),
                "temporary_score": _score_text(self),
                "reason": str(reason or ""),
                "in_evidence": in_evidence,
                "serve_stat_counted": bool(serve_stat_counted),
            }
            print(
                f"[RETRO_LET_CANDIDATE] f{frame_now}: "
                f"server={self.player_names[int(server_idx)]} attempt={attempt} "
                f"server_x={observation['x']:.0f} physical={observation.get('physical_side') or '?'} "
                f"hits={rally_shots} evidence={in_evidence} "
                f"temporary_score={_score_text(self)}"
            )
        return result

    @wraps(original_start)
    def start_point_context(self, origin_pos, serve_start_frame=None, history_origin_pos=None):
        result = original_start(
            self,
            origin_pos,
            serve_start_frame=serve_start_frame,
            history_origin_pos=history_origin_pos,
        )

        observed = _observation_from_context_dict(_point_context(self))
        if observed is None:
            observed = _context_observation_from_origin(self, origin_pos)
        self._retro_let_current_observation = observed
        # Point-context creation is provisional. The main tracker can later
        # reject this serve candidate, so it is not safe evidence for a let.
        # Resolve the previous candidate only when the following rally reaches
        # the finalized point boundary in record_point_result().
        return result

    cls._record_point_result = record_point_result
    cls._start_point_context = start_point_context

    if callable(original_refresh):
        @wraps(original_refresh)
        def refresh_player_serve_context(self):
            result = original_refresh(self)
            observed = _observation_from_context_dict(_point_context(self))
            if observed is not None:
                self._retro_let_current_observation = observed
            # A refreshed serve context is still provisional and may belong to
            # a serve candidate that is subsequently rejected. Keep the server
            # observation current, but do not confirm/discard the pending let.
            return result
        cls._refresh_player_serve_context = refresh_player_serve_context

    cls._retro_let_scoring_installed = True
    _PATCHED = True


def install_retro_let_bootstrap():
    """Patch InteractiveBallAnalyzer once its first instance finishes __init__."""
    global _TRACE_INSTALLED, _PREVIOUS_TRACE
    if _TRACE_INSTALLED:
        return
    _TRACE_INSTALLED = True
    _PREVIOUS_TRACE = sys.gettrace()

    def tracer(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "__init__":
            obj = frame.f_locals.get("self")
            cls = getattr(obj, "__class__", None)
            if (
                cls is not None and
                cls.__name__ == "InteractiveBallAnalyzer" and
                hasattr(cls, "_record_point_result")
            ):
                _patch_tracker_class(cls)
                sys.settrace(_PREVIOUS_TRACE)
                return _PREVIOUS_TRACE
        return tracer

    sys.settrace(tracer)
