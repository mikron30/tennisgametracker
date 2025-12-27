#!/usr/bin/env python3
"""
Simple test to extract white lines from first frame and show the mask.
"""

import cv2
import numpy as np


def test_white_lines():
    """
    Extract white lines from the first frame and save the result.
    """
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    # Open video and get first frame
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Get right court region
    height, width = frame.shape[:2]
    right_start_x = int(width * 0.15)
    right_court = frame[:, right_start_x:width]
    
    print(f"Right court region: {right_court.shape[1]}x{right_court.shape[0]}")
    
    # Convert to HSV for white detection
    hsv = cv2.cvtColor(right_court, cv2.COLOR_BGR2HSV)
    
    # Define white color range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Create white mask
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    # Save images
    cv2.imwrite('original_right_court.jpg', right_court)
    cv2.imwrite('white_lines_mask.jpg', white_mask)
    
    print("Images saved:")
    print("- original_right_court.jpg (original right court)")
    print("- white_lines_mask.jpg (black and white with only white lines)")
    
    cap.release()


if __name__ == "__main__":
    test_white_lines()








