"""Tune the player detector HSV values interactively."""

import cv2
from simple_player_detector import SimplePlayerDetector


def main():
    # Open video
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Go to frame 7 (where P2 is visible)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        return
    
    # Initialize detector
    detector = SimplePlayerDetector()
    
    # Run HSV tuner
    detector.tune_hsv(frame)
    
    cap.release()


if __name__ == "__main__":
    main()



