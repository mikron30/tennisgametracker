#!/usr/bin/env python3
"""
Test script for the improved tennis ball detection system.
This script allows you to test the serve area detection and ball tracking step by step.
"""

import cv2
import numpy as np
from tennis_ball_detector import TennisBallDetector, TennisVideoProcessor


def test_serve_area_detection():
    """
    Test the serve area detection on a single frame.
    """
    print("=== Testing Serve Area Detection ===")
    
    # Initialize detector
    detector = TennisBallDetector()
    
    # Open video and get first frame
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
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
    
    # Detect serve areas
    serve_areas = detector.detect_serve_areas(frame)
    print(f"Detected {len(serve_areas)} serve areas")
    
    # Draw serve areas on frame
    result_frame = frame.copy()
    for i, (x, y, w, h) in enumerate(serve_areas):
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result_frame, f"Serve Area {i+1}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        print(f"Serve Area {i+1}: x={x}, y={y}, w={w}, h={h}")
    
    # Resize frame to fit screen
    height, width = result_frame.shape[:2]
    screen_height = 800
    screen_width = 1200
    
    scale_h = screen_height / height
    scale_w = screen_width / width
    scale = min(scale_h, scale_w, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        result_frame = cv2.resize(result_frame, (new_width, new_height))
    
    # Show result
    cv2.imshow('Serve Area Detection Test', result_frame)
    print("Press any key to continue or 'q' to quit...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key == ord('q') or key == ord('Q')
    
    cap.release()


def test_ball_detection_accuracy():
    """
    Test ball detection accuracy on a few frames.
    """
    print("\n=== Testing Ball Detection Accuracy ===")
    
    # Initialize detector
    detector = TennisBallDetector()
    
    # Open video
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_count = 0
    test_frames = [0, 30, 60, 90, 120]  # Test every 30 frames
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"\nTesting frame {frame_num}:")
        
        # Test different detection methods
        methods = ["color", "hough", "combined"]
        for method in methods:
            balls = detector.detect_ball(frame, method=method, frame_number=frame_num)
            print(f"  {method}: {len(balls)} balls detected")
            
            if balls:
                for i, (x, y, r) in enumerate(balls):
                    print(f"    Ball {i+1}: center=({x}, {y}), radius={r}")
        
        # Show frame with detections
        balls = detector.detect_ball(frame, method="combined", frame_number=frame_num)
        result_frame = detector.draw_detections(frame, balls)
        
        # Resize frame to fit screen
        height, width = result_frame.shape[:2]
        screen_height = 800
        screen_width = 1200
        
        scale_h = screen_height / height
        scale_w = screen_width / width
        scale = min(scale_h, scale_w, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_frame = cv2.resize(result_frame, (new_width, new_height))
        
        cv2.imshow(f'Ball Detection Test - Frame {frame_num}', result_frame)
        print("Press any key for next frame or 'q' to quit...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q') or key == ord('Q'):
            print("Test stopped by user")
            break
    
    cap.release()


def test_serve_triggered_tracking():
    """
    Test the serve-triggered tracking system.
    """
    print("\n=== Testing Serve-Triggered Tracking ===")
    
    # Initialize detector and processor
    detector = TennisBallDetector()
    processor = TennisVideoProcessor(detector)
    
    # Process first 300 frames to see tracking activation
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_count = 0
    max_frames = 300
    
    print("Processing frames to test serve-triggered tracking...")
    print("Look for 'Tracking activated' messages in the output.")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect balls with tracking
        balls = detector.detect_ball(frame, method="combined", frame_number=frame_count)
        
        # Show frame with tracking status
        result_frame = detector.draw_detections(frame, balls)
        
        # Add frame info
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Balls: {len(balls)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize frame to fit screen
        height, width = result_frame.shape[:2]
        screen_height = 800
        screen_width = 1200
        
        scale_h = screen_height / height
        scale_w = screen_width / width
        scale = min(scale_h, scale_w, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_frame = cv2.resize(result_frame, (new_width, new_height))
        
        cv2.imshow('Serve-Triggered Tracking Test', result_frame)
        
        # Press 'q' to quit, any other key for next frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Tracking test stopped by user (Q key pressed)")
            break
        
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nTracking test completed. Processed {frame_count} frames.")
    print(f"Final tracking status: {'ACTIVE' if detector.tracking_active else 'INACTIVE'}")


def main():
    """
    Main function to run all tests.
    """
    print("Improved Tennis Ball Detection System - Test Suite")
    print("=" * 60)
    
    while True:
        print("\nChoose a test to run:")
        print("1. Test serve area detection")
        print("2. Test ball detection accuracy")
        print("3. Test serve-triggered tracking")
        print("4. Run all tests")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            test_serve_area_detection()
        elif choice == "2":
            test_ball_detection_accuracy()
        elif choice == "3":
            test_serve_triggered_tracking()
        elif choice == "4":
            test_serve_area_detection()
            test_ball_detection_accuracy()
            test_serve_triggered_tracking()
        elif choice == "5":
            print("Exiting test suite.")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
