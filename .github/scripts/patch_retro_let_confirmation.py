from pathlib import Path

source_path = Path("retro_let_scoring.py")
text = source_path.read_text(encoding="utf-8")

old_constant = "_MAX_SHORT_RALLY_SHOTS = 1\n"
new_constant = """# Shot events exclude the serve itself. A let can still be declared before\n# the server plays the first ball after the receiver's return: 0 = no return,\n# 1 = receiver returned serve, 2+ = server has already continued the rally.\n_MAX_SHORT_RALLY_SHOTS = 1\n"""
if old_constant in text and "Shot events exclude the serve itself" not in text:
    text = text.replace(old_constant, new_constant, 1)

anchor = """def _rally_shots(obj):\n    context = _point_context(obj)\n    if isinstance(context, dict):\n        events = context.get(\"shot_events\")\n        if isinstance(events, list):\n            return len(events)\n    return _safe_int(getattr(obj, \"_point_hit_count\", 0), 0)\n\n\n"""
helper = """def _rally_shots(obj):\n    context = _point_context(obj)\n    if isinstance(context, dict):\n        events = context.get(\"shot_events\")\n        if isinstance(events, list):\n            return len(events)\n    return _safe_int(getattr(obj, \"_point_hit_count\", 0), 0)\n\n\ndef _let_window_open(rally_shots):\n    \"\"\"True only until the server plays the first ball after the return.\"\"\"\n    return _safe_int(rally_shots, 0) <= _MAX_SHORT_RALLY_SHOTS\n\n\n"""
if "def _let_window_open(" not in text:
    if text.count(anchor) != 1:
        raise SystemExit(f"Expected one rally-shots anchor, found {text.count(anchor)}")
    text = text.replace(anchor, helper, 1)

old_short = "        short = rally_shots <= _MAX_SHORT_RALLY_SHOTS\n"
new_short = "        short = _let_window_open(rally_shots)\n"
if old_short in text:
    text = text.replace(old_short, new_short, 1)
elif new_short not in text:
    raise SystemExit("Could not find short-rally let eligibility assignment")

old_start = """        self._retro_let_current_observation = observed\n        _evaluate_pending(self, final=False)\n        return result\n"""
new_start = """        self._retro_let_current_observation = observed\n        # Point-context creation is provisional. The main tracker can later\n        # reject this serve candidate, so it is not safe evidence for a let.\n        # Resolve the previous candidate only when the following rally reaches\n        # the finalized point boundary in record_point_result().\n        return result\n"""
if old_start in text:
    text = text.replace(old_start, new_start, 1)
elif "Point-context creation is provisional" not in text:
    raise SystemExit("Could not find provisional start evaluation block")

old_refresh = """            if observed is not None:\n                self._retro_let_current_observation = observed\n            _evaluate_pending(self, final=False)\n            return result\n"""
new_refresh = """            if observed is not None:\n                self._retro_let_current_observation = observed\n            # A refreshed serve context is still provisional and may belong to\n            # a serve candidate that is subsequently rejected. Keep the server\n            # observation current, but do not confirm/discard the pending let.\n            return result\n"""
if old_refresh in text:
    text = text.replace(old_refresh, new_refresh, 1)
elif "A refreshed serve context is still provisional" not in text:
    raise SystemExit("Could not find provisional refresh evaluation block")

source_path.write_text(text, encoding="utf-8")

test_path = Path("tests/test_retro_let_scoring.py")
tests = test_path.read_text(encoding="utf-8")

if "def test_let_window_closes_after_server_first_return" not in tests:
    tests += '''\n\ndef test_let_window_closes_after_server_first_return():\n    from retro_let_scoring import _let_window_open\n\n    assert _let_window_open(0) is True\n    assert _let_window_open(1) is True\n    assert _let_window_open(2) is False\n    assert _let_window_open(3) is False\n'''

if "def test_provisional_serve_context_does_not_resolve_pending" not in tests:
    tests += '''\n\ndef test_provisional_serve_context_does_not_resolve_pending(monkeypatch):\n    \"\"\"A false same-side serve start must not turn the previous point into a let.\"\"\"\n    import retro_let_scoring as rls\n\n    calls = []\n    monkeypatch.setattr(rls, \"_PATCHED\", False)\n    monkeypatch.setattr(\n        rls,\n        \"_evaluate_pending\",\n        lambda obj, *, final=False: calls.append(bool(final)) or \"unknown\",\n    )\n\n    class _Tracker:\n        def __init__(self):\n            self._point_history_current = None\n            self._retro_let_pending = {\"sentinel\": True}\n\n        def _record_point_result(self, reason, end_position=None, frame=None, history_end_frame=None):\n            return None\n\n        def _start_point_context(self, origin_pos, serve_start_frame=None, history_origin_pos=None):\n            self._point_history_current = {\n                \"server_idx\": 0,\n                \"serve_context\": {\n                    \"player_position\": [1900, 1500],\n                    \"player_side\": \"near\",\n                    \"player_source\": \"server\",\n                },\n            }\n            return None\n\n        def _refresh_player_serve_context(self):\n            return None\n\n    rls._patch_tracker_class(_Tracker)\n    tracker = _Tracker()\n    tracker._start_point_context((1900, 1500), serve_start_frame=200)\n    tracker._refresh_player_serve_context()\n\n    assert calls == []\n    assert tracker._retro_let_pending == {\"sentinel\": True}\n'''

test_path.write_text(tests, encoding="utf-8")
