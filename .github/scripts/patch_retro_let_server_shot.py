from pathlib import Path

path = Path("retro_let_scoring.py")
text = path.read_text(encoding="utf-8")

old_helpers = '''def _rally_shots(obj):
    context = _point_context(obj)
    if isinstance(context, dict):
        events = context.get("shot_events")
        if isinstance(events, list):
            return len(events)
    return _safe_int(getattr(obj, "_point_hit_count", 0), 0)


def _let_window_open(rally_shots):
    """True only until the server plays the first ball after the return."""
    return _safe_int(rally_shots, 0) <= _MAX_SHORT_RALLY_SHOTS
'''

new_helpers = '''def _rally_shot_events(obj):
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
'''

if text.count(old_helpers) != 1:
    raise SystemExit(f"Expected exactly one old rally helper block, found {text.count(old_helpers)}")
text = text.replace(old_helpers, new_helpers, 1)

old_pre = '''        attempt = _safe_int(getattr(self, "current_serve_attempt", 1), 1)
        rally_shots = _rally_shots(self)
        snapshot = _score_snapshot(self)
'''
new_pre = '''        attempt = _safe_int(getattr(self, "current_serve_attempt", 1), 1)
        rally_events = _rally_shot_events(self)
        rally_shots = _rally_shots(self)
        let_window_open = _let_window_open(self, rally_shots, server_idx, rally_events)
        snapshot = _score_snapshot(self)
'''
if text.count(old_pre) != 1:
    raise SystemExit(f"Expected exactly one pre-record rally block, found {text.count(old_pre)}")
text = text.replace(old_pre, new_pre, 1)

old_short = '''        nonfault = not _service_fault_reason(reason, category)
        short = _let_window_open(rally_shots)

        # Every apparent point is already scored by original_record above.
'''
new_short = '''        nonfault = not _service_fault_reason(reason, category)
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
'''
if text.count(old_short) != 1:
    raise SystemExit(f"Expected exactly one old let-window call, found {text.count(old_short)}")
text = text.replace(old_short, new_short, 1)
path.write_text(text, encoding="utf-8")


test_path = Path("tests/test_retro_let_scoring.py")
tests = test_path.read_text(encoding="utf-8")
old_test = '''def test_let_window_closes_after_server_first_return():
    from retro_let_scoring import _let_window_open

    assert _let_window_open(0) is True
    assert _let_window_open(1) is True
    assert _let_window_open(2) is False
    assert _let_window_open(3) is False
'''
new_test = '''def test_let_window_requires_the_only_shot_to_be_receiver_return():
    from retro_let_scoring import _let_window_open

    state = _State()
    state.player_names = ["P1", "P2"]

    # No return yet: a let can still have stopped play.
    assert _let_window_open(state, 0, 0, []) is True

    # One positively identified receiver return: still possible.
    assert _let_window_open(
        state, 1, 0,
        [{"frame": 1800, "player": "P2", "label": "return"}],
    ) is True

    # Regression from the real video: the one recorded shot is P1, the server.
    # The rally therefore continued and the previous point must never become a let.
    assert _let_window_open(
        state, 1, 0,
        [{"frame": 1813, "player": "P1", "label": "racket contact"}],
    ) is False

    # Unknown shooter is not enough positive evidence to roll back a scored point.
    assert _let_window_open(
        state, 1, 0,
        [{"frame": 1813, "player": "", "label": "racket contact"}],
    ) is False

    # Two post-serve shots means the rally necessarily continued.
    assert _let_window_open(
        state, 2, 0,
        [
            {"frame": 1800, "player": "P2", "label": "return"},
            {"frame": 1813, "player": "P1", "label": "racket contact"},
        ],
    ) is False
'''
if tests.count(old_test) != 1:
    raise SystemExit(f"Expected exactly one old let-window test, found {tests.count(old_test)}")
tests = tests.replace(old_test, new_test, 1)
test_path.write_text(tests, encoding="utf-8")
