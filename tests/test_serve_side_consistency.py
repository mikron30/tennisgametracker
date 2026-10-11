from serve_side_consistency import (
    ServeSideConsistencyGuard,
    audit_file,
    expected_serve_side,
)


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
    assert any(c.score_side_mismatch or c.alternation_mismatch for c in checks)
    assert not any(c.inferred_let_replay for c in checks)


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


def test_audit_file_accepts_analyzer_string_history_path(tmp_path):
    history = tmp_path / "point_history.csv"
    history.write_text(
        "current_score,server,serve_attempt,point_index,serve_start_frame,"
        "serve_player_side,serve_ball_position\n"
        "0:0 15:0,P1,1st,1,10,near,\"(2400,1000)\"\n"
        "0:0 15:15,P1,1st,2,20,near,\"(1700,1000)\"\n"
        "0:0 30:15,P1,1st,3,30,near,\"(2420,1000)\"\n"
        "0:0 30:30,P1,1st,4,40,near,\"(1680,1000)\"\n",
        encoding="utf-8",
    )

    assert audit_file(str(history), only_problems=True) == 0
