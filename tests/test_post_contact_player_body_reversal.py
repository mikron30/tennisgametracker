from interactive_ball_analyzer import InteractiveBallAnalyzer


def make_analyzer(candidate_zone='player_body', contact_frame=4158):
    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.frame_count = 4166
    analyzer._last_racket_contact_frame = contact_frame
    analyzer.ball_center = (1662, 637)
    analyzer.motion_history = [
        {'frame': 4162, 'pos': (1663, 671)},
        {'frame': 4163, 'pos': (1661, 648)},
        {'frame': 4164, 'pos': (1662, 637)},
    ]
    analyzer._player_point_zone = lambda point: candidate_zone
    return analyzer


def test_rejects_player_body_candidate_reversing_recent_flight():
    analyzer = make_analyzer()

    assert analyzer._post_contact_player_body_reversal(
        (1673, 643), previous_position=(1662, 637)
    )


def test_keeps_player_body_candidate_aligned_with_recent_flight():
    analyzer = make_analyzer()

    assert not analyzer._post_contact_player_body_reversal(
        (1664, 614), previous_position=(1662, 637)
    )


def test_does_not_apply_guard_outside_player_body_or_recent_contact():
    body_candidate = make_analyzer(candidate_zone=None)
    stale_contact = make_analyzer(contact_frame=4100)

    assert not body_candidate._post_contact_player_body_reversal(
        (1673, 643), previous_position=(1662, 637)
    )
    assert not stale_contact._post_contact_player_body_reversal(
        (1673, 643), previous_position=(1662, 637)
    )
