from types import SimpleNamespace

import numpy as np
import pytest

from interactive_ball_analyzer import InteractiveBallAnalyzer


@pytest.mark.parametrize('guard', ['stuck', 'occlusion'])
def test_ball_loss_does_not_freeze_player_detections_indefinitely(guard):
    calls = []
    tracker = SimpleNamespace(_last_detection_frame=0, detection_interval=5,
                              set_court_x_range=lambda r: None)

    def update(frame, number, net_y=None):
        if number - tracker._last_detection_frame >= tracker.detection_interval:
            calls.append(number)
            tracker._last_detection_frame = number
        return {'frame': number}

    tracker.update = update
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.player_tracker = tracker
    owner.far_serve_image_baseline = True
    owner._build_singles_sideline_model = lambda frame: None
    owner.last_motion = {'distance': 1 if guard == 'stuck' else 60}
    owner.stuck_frame_count = 5 if guard == 'stuck' else 1
    owner._player_reacq_protect_until_frame = 200 if guard == 'occlusion' else -1
    image = np.zeros((10, 10, 3), np.uint8)
    for number in range(1, 61):
        owner.frame_count = number
        assert owner._update_player_tracking(image) == {'frame': number}
    assert calls == [15, 30, 45, 60]


def test_normal_player_detection_cadence_is_preserved():
    calls = []
    tracker = SimpleNamespace(_last_detection_frame=0, detection_interval=5)

    def update(frame, number, net_y=None):
        if number - tracker._last_detection_frame >= 5:
            tracker._last_detection_frame = number
            calls.append(number)

    tracker.update = update
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.player_tracker = tracker
    owner.far_serve_image_baseline = True
    owner._build_singles_sideline_model = lambda frame: None
    owner.last_motion = {'distance': 60}
    owner._player_reacq_protect_until_frame = 200
    for number in range(1, 21):
        owner.frame_count = number
        owner._update_player_tracking(np.zeros((10, 10, 3), np.uint8))
    assert calls == [5, 10, 15, 20]


def test_legacy_recovery_preserves_profile_cadence_and_hog_range():
    owner = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    owner.frame_count=30
    owner.far_serve_image_baseline=False
    owner.last_motion={'distance':1}
    owner.stuck_frame_count=5
    tracker=SimpleNamespace(_last_detection_frame=0, detection_interval=5,
        update=lambda *a,**kw: {'held': tracker._last_detection_frame})
    owner.player_tracker=tracker
    owner._build_singles_sideline_model=lambda frame:None
    assert owner._update_player_tracking(np.zeros((10,10,3),np.uint8))=={'held':30}
    assert tracker.full_court_hog_enabled is False
