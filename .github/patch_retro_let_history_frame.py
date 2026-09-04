from pathlib import Path

path = Path("retro_let_scoring.py")
text = path.read_text(encoding="utf-8")

old_safe = """def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
"""
new_safe = """def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _logical_point_end_frame(obj, frame=None, history_end_frame=None):
    \"\"\"Return the frame used by point-history CSV for this point result.

    Some point outcomes are finalized several frames after the actual landing or
    timeout boundary. In those cases frame_count is the callback frame while
    history_end_frame is the point_end_frame written to the CSV. A retrospective
    let must remember the latter so it rewrites the same row whose score it rolls
    back.
    \"\"\"
    for value in (history_end_frame, frame):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return _safe_int(getattr(obj, "frame_count", -1), -1)
"""
if text.count(old_safe) != 1:
    raise SystemExit(f"Expected one _safe_int block, found {text.count(old_safe)}")
text = text.replace(old_safe, new_safe, 1)

old_pre = """        observation = _current_observation(self)
        server_idx = _server_idx_for_current_context(self)
        attempt = _safe_int(getattr(self, "current_serve_attempt", 1), 1)
        rally_events = _rally_shot_events(self)
"""
new_pre = """        observation = _current_observation(self)
        server_idx = _server_idx_for_current_context(self)
        attempt = _safe_int(getattr(self, "current_serve_attempt", 1), 1)
        logical_end_frame = _logical_point_end_frame(self, frame, history_end_frame)
        rally_events = _rally_shot_events(self)
"""
if text.count(old_pre) != 1:
    raise SystemExit(f"Expected one pre-record block, found {text.count(old_pre)}")
text = text.replace(old_pre, new_pre, 1)

old_pending = """                \"start_frame\": start_frame,
                \"end_frame\": frame_now,
                \"hits\": int(rally_shots),
"""
new_pending = """                \"start_frame\": start_frame,
                \"end_frame\": int(logical_end_frame),
                \"record_frame\": int(frame_now),
                \"hits\": int(rally_shots),
"""
if text.count(old_pending) != 1:
    raise SystemExit(f"Expected one pending end-frame block, found {text.count(old_pending)}")
text = text.replace(old_pending, new_pending, 1)

old_print = """                f\"hits={rally_shots} evidence={in_evidence} \"
                f\"temporary_score={_score_text(self)}\"
"""
new_print = """                f\"hits={rally_shots} evidence={in_evidence} \"
                f\"history_end={logical_end_frame} record_frame={frame_now} \"
                f\"temporary_score={_score_text(self)}\"
"""
if text.count(old_print) != 1:
    raise SystemExit(f"Expected one candidate print block, found {text.count(old_print)}")
text = text.replace(old_print, new_print, 1)
path.write_text(text, encoding="utf-8")

test_path = Path("tests/test_retro_let_scoring.py")
tests = test_path.read_text(encoding="utf-8")
anchor = """def test_let_window_requires_the_only_shot_to_be_receiver_return():
"""
if tests.count(anchor) != 1:
    raise SystemExit(f"Expected test anchor once, found {tests.count(anchor)}")
new_test = """def test_logical_point_end_frame_prefers_history_frame_over_delayed_callback():
    from retro_let_scoring import _logical_point_end_frame

    state = _State()
    state.frame_count = 4677

    assert _logical_point_end_frame(state, frame=4677, history_end_frame=4663) == 4663
    assert _logical_point_end_frame(state, frame=4936, history_end_frame=None) == 4936
    assert _logical_point_end_frame(state, frame=None, history_end_frame=None) == 4677


"""
tests = tests.replace(anchor, new_test + anchor, 1)
test_path.write_text(tests, encoding="utf-8")
