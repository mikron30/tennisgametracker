from interactive_ball_analyzer import InteractiveBallAnalyzer


def _analyzer(base_end="near", dx=-1, dy=-1):
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer._base_serve_area_end = base_end
    analyzer.start_server_side = None
    analyzer._base_serve_area = {
        "x_min": 100,
        "x_max": 900,
        "y_min": 600,
        "y_max": 1000,
        "points": [],
    }
    analyzer._configured_far_serve_area = {
        "x_min": 200,
        "x_max": 800,
        "y_min": 0,
        "y_max": 120,
        "points": [],
    }
    analyzer._base_serve_direction_dx = dx
    analyzer._base_serve_direction_dy = dy
    analyzer._base_serve_ball_size_min = 3
    analyzer._base_serve_ball_size_max = 80
    analyzer.serve_ball_size_min = 3
    analyzer.serve_ball_size_max = 80
    analyzer.far_serve_ball_size_min = 1
    analyzer.far_serve_ball_size_max = 120
    analyzer.serve_direction_dx = dx
    analyzer.serve_direction_dy = dy
    analyzer._active_serve_area_end = None
    analyzer.frame_count = 0
    return analyzer


def _serve_geometry_by_game(analyzer, game_count=6):
    result = []
    for completed_games in range(game_count):
        analyzer.score_game_index = completed_games
        analyzer._apply_active_serve_geometry()
        result.append(
            (
                analyzer._active_serve_area_end,
                analyzer.serve_direction_dx,
                analyzer.serve_direction_dy,
            )
        )
    return result


def test_server_end_follows_server_rotation_and_tennis_end_changes():
    analyzer = _analyzer()

    assert _serve_geometry_by_game(analyzer) == [
        ("near", -1, -1),
        ("near", -1, -1),
        ("far", 1, 1),
        ("far", 1, 1),
        ("near", -1, -1),
        ("near", -1, -1),
    ]


def test_opposite_configured_end_rotates_the_complete_serve_vector():
    analyzer = _analyzer(base_end="far", dx=1, dy=1)

    assert _serve_geometry_by_game(analyzer, game_count=4) == [
        ("far", 1, 1),
        ("far", 1, 1),
        ("near", -1, -1),
        ("near", -1, -1),
    ]
