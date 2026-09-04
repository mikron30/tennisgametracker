from retro_let_scoring import (
    _restore_score_snapshot,
    _score_snapshot,
    _side_comparison,
)


class _State:
    pass


def _observation(x, physical="near"):
    return {"x": float(x), "y": 1500.0, "physical_side": physical, "source": "server"}


def test_same_service_side_uses_server_position_not_toss_position():
    state = _State()
    state._retro_let_side_samples = {
        (0, "near"): [
            (22, 1805.0),
            (696, 1548.0),
            (1242, 1658.0),
            (1727, 1901.0),
        ]
    }
    pending = {
        "server_idx": 0,
        "observation": _observation(1901),
    }

    decision, detail = _side_comparison(state, pending, _observation(1830))

    assert decision == "same"
    assert "right->right" in detail


def test_normal_alternating_server_positions_are_opposite_service_sides():
    state = _State()
    state._retro_let_side_samples = {
        (0, "near"): [
            (22, 1805.0),
            (696, 1548.0),
            (1242, 1658.0),
            (1727, 1901.0),
        ]
    }
    pending = {
        "server_idx": 0,
        "observation": _observation(1805),
    }

    decision, detail = _side_comparison(state, pending, _observation(1548))

    assert decision == "opposite"
    assert "right->left" in detail


def test_early_match_fallback_can_confirm_same_side_before_cluster_matures():
    state = _State()
    pending = {
        "server_idx": 0,
        "observation": _observation(1901),
    }

    decision, detail = _side_comparison(state, pending, _observation(1830))

    assert decision == "same"
    assert "early-dx=71px" in detail


def test_score_snapshot_restores_game_state_and_removes_let_serve_in():
    state = _State()
    state.score_points = [3, 3]
    state.score_games = [2, 2]
    state.score_game_index = 4
    state.current_serve_attempt = 2
    state.point_stats = [{"won": 3}, {"won": 4}]
    state.serve_stats = [
        {"first_in": 3, "first_faults": 1, "second_in": 2, "double_faults": 0},
        {"first_in": 4, "first_faults": 0, "second_in": 1, "double_faults": 0},
    ]
    state._last_scored_point_end_frame = 100
    state._last_point_winner = 1
    state._last_point_score_reason = "previous"
    state._last_point_outcome_category = "winner"
    state._last_point_hit_count = 4
    state._last_confirmed_point_end_frame = 100
    state._serve_landed_in_current_attempt = True
    state._serve_in_recorded_attempt = 2
    state._serve_start_requires_confirmation = False

    snapshot = _score_snapshot(state)

    state.score_points = [0, 0]
    state.score_games = [2, 3]
    state.score_game_index = 5
    state.current_serve_attempt = 1
    state.point_stats[1]["won"] += 1
    state.serve_stats[0]["second_in"] = 99

    _restore_score_snapshot(
        state,
        snapshot,
        remove_counted_serve_in=True,
        server_idx=0,
        attempt=2,
    )

    assert state.score_points == [3, 3]
    assert state.score_games == [2, 2]
    assert state.score_game_index == 4
    assert state.current_serve_attempt == 2
    assert state.point_stats == [{"won": 3}, {"won": 4}]
    assert state.serve_stats[0]["second_in"] == 1


def test_let_window_requires_the_only_shot_to_be_receiver_return():
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


def test_provisional_serve_context_does_not_resolve_pending(monkeypatch):
    """A false same-side serve start must not turn the previous point into a let."""
    import retro_let_scoring as rls

    calls = []
    monkeypatch.setattr(rls, "_PATCHED", False)
    monkeypatch.setattr(
        rls,
        "_evaluate_pending",
        lambda obj, *, final=False: calls.append(bool(final)) or "unknown",
    )

    class _Tracker:
        def __init__(self):
            self._point_history_current = None
            self._retro_let_pending = {"sentinel": True}

        def _record_point_result(self, reason, end_position=None, frame=None, history_end_frame=None):
            return None

        def _start_point_context(self, origin_pos, serve_start_frame=None, history_origin_pos=None):
            self._point_history_current = {
                "server_idx": 0,
                "serve_context": {
                    "player_position": [1900, 1500],
                    "player_side": "near",
                    "player_source": "server",
                },
            }
            return None

        def _refresh_player_serve_context(self):
            return None

    rls._patch_tracker_class(_Tracker)
    tracker = _Tracker()
    tracker._start_point_context((1900, 1500), serve_start_frame=200)
    tracker._refresh_player_serve_context()

    assert calls == []
    assert tracker._retro_let_pending == {"sentinel": True}
