"""Retrospective serve-let rollback for the live tennis scorer.

The live tracker scores every apparent point immediately.  A rare missed let is
confirmed only when the next serve proves that the previous short, in-serve
sequence is being replayed from the same physical service side.  At that point
we restore the exact pre-point tennis state and let the replayed rally score
normally when it actually ends.

This module is installed at tracker start by ball_ai_recovery_probe.  The small
bootstrap exists because interactive_ball_analyzer.py is intentionally kept as
one large legacy file; the scoring behavior itself is isolated here and can be
moved into the class directly when that file is split into modules.
"""
from __future__ import annotations

import copy
import csv
import os
import sys
import tempfile
from functools import wraps

_PATCHED = False
_TRACE_INSTALLED = False
_PREVIOUS_TRACE = None


def _score_snapshot(obj):
    return {
        "score_points": list(getattr(obj, "score_points", [0, 0])),
        "score_games": list(getattr(obj, "score_games", [0, 0])),
        "score_game_index": int(getattr(obj, "score_game_index", 0)),
        "current_serve_attempt": int(getattr(obj, "current_serve_attempt", 1)),
        "point_stats": copy.deepcopy(getattr(obj, "point_stats", None)),
        "last_confirmed_point_end_frame": getattr(obj, "_last_confirmed_point_end_frame", None),
    }


def _restore_score_snapshot(obj, snapshot):
    obj.score_points = list(snapshot["score_points"])
    obj.score_games = list(snapshot["score_games"])
    obj.score_game_index = int(snapshot["score_game_index"])
    obj.current_serve_attempt = int(snapshot["current_serve_attempt"])
    if snapshot.get("point_stats") is not None:
        obj.point_stats = copy.deepcopy(snapshot["point_stats"])
    obj._last_confirmed_point_end_frame = snapshot.get("last_confirmed_point_end_frame")
    obj._serve_landed_in_current_attempt = False
    obj._serve_in_recorded_attempt = None


def _player_side_from_context(obj):
    context = getattr(obj, "_point_history_current", None)
    if not isinstance(context, dict):
        return None
    serve_context = context.get("serve_context")
    if not isinstance(serve_context, dict):
        return None
    side = str(serve_context.get("player_side") or "").strip().lower()
    return side if side in {"near", "far"} else None


def _serve_label(obj):
    fn = getattr(obj, "_serve_attempt_label", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    return "1st" if int(getattr(obj, "current_serve_attempt", 1)) <= 1 else "2nd"


def _score_text(obj):
    fn = getattr(obj, "_score_summary", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    return f"{obj.score_games[0]}:{obj.score_games[1]} {obj.score_points[0]}:{obj.score_points[1]}"


def _undo_let_serve_stat(obj, pending):
    """A let is not a counted first/second serve-in attempt."""
    stats_all = getattr(obj, "serve_stats", None)
    server_idx = int(pending.get("server_idx", -1))
    if not isinstance(stats_all, list) or not (0 <= server_idx < len(stats_all)):
        return
    stats = stats_all[server_idx]
    if not isinstance(stats, dict):
        return
    key = "first_in" if int(pending.get("attempt", 1)) <= 1 else "second_in"
    stats[key] = max(0, int(stats.get(key, 0)) - 1)


def _rewrite_history_row_as_let(obj, pending):
    """Keep the CSV consistent with a live rollback, when history is enabled."""
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
        restored_score = _score_text(obj)
        updates = {
            "end_reason": "Serve let (retrospective same-side replay)",
            "why": "serve in + short rally + immediate same-side replay",
            "category": "serve_let",
            "point_awarded": "no",
            "winner": "",
            "current_score": restored_score,
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


def _patch_tracker_class(cls):
    global _PATCHED
    if _PATCHED or getattr(cls, "_retro_let_scoring_installed", False):
        return
    original_record = cls._record_point_result
    original_start = cls._start_point_context

    @wraps(original_record)
    def record_point_result(self, reason, end_position=None, frame=None, history_end_frame=None):
        # Capture the tennis state immediately before the normal scorer changes it.
        serve_was_in = bool(getattr(self, "_serve_landed_in_current_attempt", False))
        attempt = int(getattr(self, "current_serve_attempt", 1))
        service_side = getattr(self, "_point_serve_start_side", None)
        player_side = _player_side_from_context(self)
        server_idx = int(self._current_server_index())
        short_hits = int(getattr(self, "_point_hit_count", 0))
        snapshot = _score_snapshot(self)
        start_frame = getattr(self, "point_start_frame_internal", None)

        result = original_record(self, reason, end_position=end_position, frame=frame,
                                 history_end_frame=history_end_frame)

        # Score normally first.  Only an awarded, short point that began with a
        # verified in-serve can become a retrospective-let candidate.
        awarded = getattr(self, "_last_point_winner", None) is not None
        if (
            awarded and serve_was_in and short_hits <= 1 and
            service_side in {"left", "right"}
        ):
            self._retro_let_pending = {
                "snapshot": snapshot,
                "server_idx": server_idx,
                "attempt": attempt,
                "service_side": service_side,
                "player_side": player_side,
                "start_frame": start_frame,
                "end_frame": int(getattr(self, "frame_count", -1)),
                "hits": short_hits,
                "temporary_score": _score_text(self),
                "reason": str(reason or ""),
            }
            print(
                f"[RETRO_LET_CANDIDATE] f{self.frame_count}: "
                f"server={self.player_names[server_idx]} attempt={attempt} "
                f"side={service_side} player_side={player_side or '?'} hits={short_hits} "
                f"temporary_score={_score_text(self)}"
            )
        return result

    @wraps(original_start)
    def start_point_context(self, origin_pos, serve_start_frame=None, history_origin_pos=None):
        result = original_start(self, origin_pos, serve_start_frame=serve_start_frame,
                                history_origin_pos=history_origin_pos)
        pending = getattr(self, "_retro_let_pending", None)
        if not isinstance(pending, dict):
            return result

        current_side = getattr(self, "_point_serve_start_side", None)
        current_player_side = _player_side_from_context(self)
        frame_now = int(getattr(self, "frame_count", -1))
        gap = frame_now - int(pending.get("end_frame", frame_now))
        same_service_side = current_side in {"left", "right"} and current_side == pending.get("service_side")

        previous_player_side = pending.get("player_side")
        if previous_player_side and current_player_side:
            same_server = previous_player_side == current_player_side
            server_evidence = f"player_side={previous_player_side}->{current_player_side}"
        else:
            # Safe fallback away from a game boundary.  If the temporary point
            # completed a game, score-derived server identity changed, so we do
            # not guess without physical player-side evidence.
            same_server = int(getattr(self, "score_game_index", 0)) == int(
                pending["snapshot"]["score_game_index"]
            )
            server_evidence = "score-game-index fallback"

        immediate = 0 <= gap <= 1200
        if same_service_side and same_server and immediate:
            temporary_score = _score_text(self)
            _restore_score_snapshot(self, pending["snapshot"])
            _undo_let_serve_stat(self, pending)

            # _start_point_history_row ran before this confirmation.  Repair the
            # new replay row so it uses the restored server/attempt state.
            current_history = getattr(self, "_point_history_current", None)
            if isinstance(current_history, dict):
                current_history["server_idx"] = int(self._current_server_index())
                current_history["serve_attempt"] = _serve_label(self)

            _rewrite_history_row_as_let(self, pending)
            print(
                f"[RETRO_LET_CONFIRMED] f{frame_now}: previous=f{pending['end_frame']} "
                f"serve_in=yes hits={pending['hits']} same_side={current_side} {server_evidence}"
            )
            print(
                f"[RETRO_LET_ROLLBACK] f{frame_now}: score={temporary_score}->{_score_text(self)} "
                f"replay={_serve_label(self)} side={current_side}"
            )
        else:
            print(
                f"[RETRO_LET_DISCARDED] f{frame_now}: previous=f{pending['end_frame']} "
                f"old_side={pending.get('service_side')} new_side={current_side} "
                f"same_server={same_server} gap={gap}f"
            )
        self._retro_let_pending = None
        return result

    cls._record_point_result = record_point_result
    cls._start_point_context = start_point_context
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
            if cls is not None and cls.__name__ == "InteractiveBallAnalyzer" and hasattr(cls, "_record_point_result"):
                _patch_tracker_class(cls)
                sys.settrace(_PREVIOUS_TRACE)
                return _PREVIOUS_TRACE
        return tracer

    sys.settrace(tracer)
