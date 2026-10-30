import cv2
import numpy as np
from player_racket_tracker import PlayerRacketTracker


def test_person_detection_uses_full_court_instead_of_far_serve_rectangle():
    tracker = PlayerRacketTracker(frame_shape=(2160,3840), learning_enabled=False)
    tracker.full_court_hog_enabled = True
    tracker.set_court_x_range((1580,2180))
    tracker.set_court_region(dict(left=dict(a=0,b=1000),right=dict(a=0,b=2700),
                                 y_min=0,y_max=2160))
    # Boxes in the HOG input's 1100/3840 coordinate system: a real receiver
    # outside the serve rectangle and an adjacent-court false person.
    class Detector:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[638,169,66,131],[900,169,66,131]]), np.array([.3,.9])
    tracker._ensure_hog = lambda: Detector()
    detected = tracker._hog_detections(np.zeros((2160,3840,3),np.uint8))
    assert len(detected) == 1
    assert 2200 < detected[0]['bbox'][0] < 2250


def test_without_fitted_court_hog_keeps_horizontal_range_fallback():
    tracker = PlayerRacketTracker(frame_shape=(2160,3840), learning_enabled=False)
    tracker.set_court_x_range((1580,2180))
    class Detector:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[638,169,66,131]]), np.array([.3])
    tracker._ensure_hog = lambda: Detector()
    assert tracker._hog_detections(np.zeros((2160,3840,3),np.uint8)) == []


def test_legacy_profile_keeps_hog_range_even_with_fitted_sidelines():
    tracker=PlayerRacketTracker(frame_shape=(2160,3840),learning_enabled=False)
    tracker.set_court_x_range((1580,2180))
    tracker.set_court_region(dict(left=dict(a=0,b=1000),right=dict(a=0,b=2700),
                                 y_min=0,y_max=2160))
    class Detector:
        def detectMultiScale(self,*args,**kwargs):
            return np.array([[638,169,66,131]]),np.array([.3])
    tracker._ensure_hog=lambda:Detector()
    assert tracker._hog_detections(np.zeros((2160,3840,3),np.uint8))==[]
