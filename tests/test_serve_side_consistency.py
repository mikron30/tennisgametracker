from serve_side_consistency import ServeSideConsistencyGuard, expected_serve_side


def test_score_parity_maps_to_tennis_service_side():
    assert expected_serve_side("0:0 0:0") == "right"
    assert expected_serve_side("0:0 15:0") == "left"
    assert expected_serve_side("0:0 30:0") == "right"
    assert expected_serve_side("0:0 15:15") == "right"
    assert expected_serve_side("0:0 30:15") == "left"
    assert expected_serve_side("0:0 40:40") == "right"
    assert expected_serve_side("0:0 AD:40") == "left"
    assert expected_serve_side("6:6 3:2") == "left"


def _row(score, point, x, attempt="1st"):
    return {
        "current_score": score,
        "server": "P1",
        "serve_attempt": attempt,
        "point_index": str(point),
        "serve_start_frame": str(point * 100),
        "serve_player_side": "near",
        "serve_ball_position": f"({x},1000)",
    }


def test_clean_alternation_passes():
    rows = [
        _row("0:0 15:0", 1, 2400),
        _row("0:0 15:15", 2, 1700),
        _row("0:0 30:15", 3, 2420),
        _row("0:0 30:30", 4, 1680),
    ]
    checks = ServeSideConsistencyGuard().audit_rows(rows)
    assert [c.status for c in checks] == ["OK", "OK", "OK", "OK"]


def test_repeated_service_side_is_flagged():
    rows = [
        _row("0:0 15:0", 1, 2400),
        _row("0:0 30:0", 2, 2350),  # should have moved to ad/left
        _row("0:0 40:0", 3, 2420),
        _row("0:1 0:0", 4, 1680),
    ]
    checks = ServeSideConsistencyGuard().audit_rows(rows)
    assert any(c.status == "MISMATCH" for c in checks)


def test_second_serve_stays_on_same_side():
    rows = [
        _row("0:0 0:0", 1, 2400),  # first-serve fault, score unchanged
        _row("0:0 15:0", 2, 2380, attempt="2nd"),
        _row("0:0 15:15", 3, 1700),
        _row("0:0 30:15", 4, 2420),
        _row("0:0 30:30", 5, 1680),
    ]
    checks = ServeSideConsistencyGuard().audit_rows(rows)
    second = checks[1]
    assert second.observed == "right"
    assert second.status == "OK"
