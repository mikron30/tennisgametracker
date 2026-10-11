from unittest.mock import Mock

import cv2

from video_positioning import position_capture


class ReferenceCapture:
    def __init__(self, position=0, codec='hevc', fail_at=None):
        self.position = position
        self.codec = sum(ord(c) << (8 * i) for i, c in enumerate(codec))
        self.fail_at = fail_at
        self.seeks = []
        self.decoded = []

    def get(self, prop):
        return self.position if prop == cv2.CAP_PROP_POS_FRAMES else self.codec

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seeks.append(value)
        self.position = int(value)
        return True

    def grab(self):
        if self.position == self.fail_at:
            return False
        self.decoded.append(self.position)
        self.position += 1
        return True


def test_midstream_hevc_keeps_all_references_and_next_requested_frame():
    cap = ReferenceCapture()
    assert position_capture(cap, 12)
    assert cap.decoded == list(range(12))
    assert cap.position == 12 and cap.seeks == []
    assert position_capture(cap, 12)
    assert len(cap.decoded) == 12  # Repeated initialization must not seek again.


def test_backward_hevc_seek_rebuilds_references_from_stream_start():
    cap = ReferenceCapture(position=20)
    assert position_capture(cap, 5)
    assert cap.seeks == [0]
    assert cap.decoded == list(range(5))


def test_failed_preroll_does_not_report_success_or_skip_missing_picture():
    cap = ReferenceCapture(fail_at=3)
    assert not position_capture(cap, 5)
    assert cap.position == 3 and cap.seeks == []


def test_other_codecs_keep_native_seek():
    cap = ReferenceCapture(codec='avc1')
    assert position_capture(cap, 20)
    assert cap.seeks == [20] and cap.decoded == []


def test_background_baseline_read_does_not_reset_main_decoder(tmp_path, monkeypatch):
    from interactive_ball_analyzer import InteractiveBallAnalyzer
    import numpy as np

    analyzer = InteractiveBallAnalyzer.__new__(InteractiveBallAnalyzer)
    analyzer.cap = Mock()
    analyzer.video_path = 'same-camera.mp4'
    analyzer.hsv_lower = np.array([20, 50, 130], dtype=np.uint8)
    analyzer.hsv_upper = np.array([85, 255, 255], dtype=np.uint8)
    analyzer.config_file = str(tmp_path / 'hsv_config.json')
    analyzer._frame0_hotspot_signature = None
    analyzer._frame0_hotspot_signature_value = lambda: 'test'
    analyzer._tracking_ball_size_max = lambda: 800
    analyzer._frame0_filter_specs = lambda: [('regular', analyzer.hsv_lower, analyzer.hsv_upper)]
    background = Mock()
    background.read.return_value = (True, np.zeros((20, 30, 3), dtype=np.uint8))
    opened = Mock(return_value=background)
    monkeypatch.setattr('interactive_ball_analyzer.cv2.VideoCapture', opened)
    assert analyzer._build_frame0_background_hotspots()
    opened.assert_called_once_with('same-camera.mp4')
    background.release.assert_called_once()
    analyzer.cap.set.assert_not_called()
    analyzer.cap.read.assert_not_called()
