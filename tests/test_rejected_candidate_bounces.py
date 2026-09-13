"""Replay the long audit's rejected launches at f4663/f4667.

Pixel detection and AI arbitration are outside this fixture.  The real bounce
registration and rollback must not leave scoring evidence for rejected HSV.
"""
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.fixture
def analyzer(monkeypatch, tmp_path):
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture', lambda _: MagicMock())
    a = InteractiveBallAnalyzer(
        'unused', enable_player_tracking=False, enable_player_learning=False,
        point_history_file=str(tmp_path / 'points.csv'),
    )
    a.frame_count = 4657
    a.ball_center, a.ball_size = (1783, 815), 49.5
    a.last_motion = {'dx': -35., 'dy': 34., 'distance': 48.8}
    a.prev_motion = {'dx': -9., 'dy': 8., 'distance': 12.}
    a._player_reacq_protect_until_frame = 4729
    a._classify_ground_bounce = lambda *args, **kw: (True, 'Bounce in singles court', (255, 0, 0))
    a._racket_below_ball_motion_debug = lambda *args: {'available': False}
    a._night_return_crossed_net_after_serve_bounce = lambda *args: False
    return a


def register(a, frame_number):
    a.frame_count = frame_number
    return a._register_ground_bounce_from_context(
        {'origin': (1783, 815), 'expected': (1751, 775),
         'incoming_dx': -35., 'incoming_dy': 34., 'incoming_dist': 48.8},
        np.zeros((100, 100, 3), np.uint8), source_label='predicted launch',
    )


def test_rejected_launches_cannot_accumulate_double_bounce(analyzer):
    a = analyzer
    for frame_number in (4663, 4667):
        prior_bounce_frame = a.last_ground_bounce_frame
        before = a._snapshot_tracking_state_for_provisional_guard()
        assert register(a, frame_number)
        a.ball_center = (1719, 767)
        a._restore_tracking_state_for_provisional_guard(before)
        assert a.ball_center == (1783, 815)
        assert a.ground_bounce_count == 0
        assert a.last_ground_bounce_frame == prior_bounce_frame
        assert a._pending_rally_end_reason is None
        assert a.recent_bounce_markers == []
        assert not getattr(a, '_ground_bounce_debug_history', [])
        # Preserve the independently armed occlusion guard during rollback.
        assert a._player_reacq_protect_until_frame == 4729


def test_accepted_launches_still_register_double_bounce(analyzer):
    assert register(analyzer, 4663)
    assert register(analyzer, 4667)
    assert analyzer.ground_bounce_count == 2
    assert analyzer._pending_rally_end_reason == 'Ball bounced twice on court'
    assert len(analyzer._ground_bounce_debug_history) == 2


def test_rejection_preserves_previously_accepted_bounce(analyzer):
    a = analyzer
    assert register(a, 4663)
    before = a._snapshot_tracking_state_for_provisional_guard()
    accepted_markers = deepcopy(a.recent_bounce_markers)
    assert register(a, 4667)
    assert a._pending_rally_end_reason is not None
    a._restore_tracking_state_for_provisional_guard(before)
    assert a.ground_bounce_count == 1
    assert a.last_ground_bounce_frame == 4663
    assert a._pending_rally_end_reason is None
    assert a.recent_bounce_markers == accepted_markers


def test_repeated_retry_does_not_mutate_saved_serve_statistics(analyzer):
    a = analyzer
    a._serve_phase_active = a._awaiting_serve_bounce = True
    a.current_serve_attempt = 1
    before = a._snapshot_tracking_state_for_provisional_guard()
    original_stats = deepcopy(a.serve_stats)
    for frame_number in (4663, 4667):
        assert register(a, frame_number)
        assert a.serve_stats[0]['first_in'] == original_stats[0]['first_in'] + 1
        a._restore_tracking_state_for_provisional_guard(before)
        assert a.serve_stats == original_stats
        assert a._awaiting_serve_bounce
        assert a._serve_phase_active
        assert not a._serve_landed_in_current_attempt


def test_existing_review_event_is_not_modified_through_snapshot(analyzer):
    a = analyzer
    a.direction_change_events = [{'frame': 4663, 'pos': (1783, 815),
                                  'status': 'bad', 'reason': 'unconfirmed'}]
    before = a._snapshot_tracking_state_for_provisional_guard()
    assert register(a, 4663)
    assert a.direction_change_events[0]['status'] == 'good'
    a._restore_tracking_state_for_provisional_guard(before)
    assert a.direction_change_events[0]['status'] == 'bad'
