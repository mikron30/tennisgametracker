#!/usr/bin/env python3
"""
Example usage of the Tennis Ball Detection System.
This script demonstrates how to use the detector programmatically.
"""

import cv2
import numpy as np
from tennis_ball_detector import TennisBallDetector, TennisVideoProcessor


def example_single_frame_detection():
    """
    Example of detecting tennis balls in a single frame.
    """
    print("=== Single Frame Detection Example ===")
    
    # Initialize detector
    detector = TennisBallDetector()
    
    # Create a sample frame (you would load an actual image here)
    # For demonstration, we'll create a simple test image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a yellow circle to simulate a tennis ball
    cv2.circle(frame, (320, 240), 20, (0, 255, 255), -1)  # Yellow circle
    
    # Detect balls
    balls = detector.detect_ball(frame, method="combined")
    
    print(f"Detected {len(balls)} balls in the frame")
    for i, (x, y, radius) in enumerate(balls):
        print(f"Ball {i+1}: Center=({x}, {y}), Radius={radius}")
    
    # Draw detections
    result_frame = detector.draw_detections(frame, balls)
    
    # Display result
    cv2.imshow('Tennis Ball Detection Example', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_video_processing():
    """
    Example of processing a tennis video file.
    """
    print("\n=== Video Processing Example ===")
    
    # Initialize detector and processor
    detector = TennisBallDetector()
    processor = TennisVideoProcessor(detector)
    
    # Example video path (replace with your actual video file)
    video_path = "sample_tennis_video.mp4"
    
    try:
        # Process video
        print(f"Processing video: {video_path}")
        detections = processor.process_video(
            video_path,
            output_path="output_with_detections.mp4",
            method="combined",
            show_preview=True
        )
        
        # Analyze results
        analysis = processor.analyze_detections(detections)
        print("\n=== Detection Analysis ===")
        print(f"Total frames: {analysis['total_frames']}")
        print(f"Frames with balls: {analysis['frames_with_balls']}")
        print(f"Detection rate: {analysis['detection_rate']:.2%}")
        print(f"Total detections: {analysis['total_detections']}")
        
        # Plot trajectory
        processor.plot_trajectory(detections, 1920, 1080)
        
    except FileNotFoundError:
        print(f"Video file '{video_path}' not found.")
        print("Please provide a valid tennis video file path.")
    except Exception as e:
        print(f"Error processing video: {e}")


def example_custom_parameters():
    """
    Example of using custom detection parameters.
    """
    print("\n=== Custom Parameters Example ===")
    
    # Create detector with custom parameters
    detector = TennisBallDetector(
        lower_color=(15, 80, 80),    # More restrictive lower bound
        upper_color=(35, 255, 255),  # More restrictive upper bound
        min_radius=8,                # Larger minimum radius
        max_radius=40                # Smaller maximum radius
    )
    
    print("Custom detector created with:")
    print(f"  Lower HSV: {detector.lower_color}")
    print(f"  Upper HSV: {detector.upper_color}")
    print(f"  Radius range: {detector.min_radius}-{detector.max_radius}")
    
    # You can now use this detector with your custom parameters
    processor = TennisVideoProcessor(detector)
    print("Custom detector is ready to use!")


def example_batch_processing():
    """
    Example of processing multiple video files.
    """
    print("\n=== Batch Processing Example ===")
    
    # List of video files to process
    video_files = [
        "tennis_match_1.mp4",
        "tennis_match_2.mp4",
        "tennis_match_3.mp4"
    ]
    
    detector = TennisBallDetector()
    processor = TennisVideoProcessor(detector)
    
    results = {}
    
    for video_file in video_files:
        try:
            print(f"\nProcessing: {video_file}")
            detections = processor.process_video(
                video_file,
                output_path=f"output_{video_file}",
                method="combined",
                show_preview=False  # Disable preview for batch processing
            )
            
            analysis = processor.analyze_detections(detections)
            results[video_file] = analysis
            
            print(f"  Detection rate: {analysis['detection_rate']:.2%}")
            print(f"  Total detections: {analysis['total_detections']}")
            
        except FileNotFoundError:
            print(f"  File not found: {video_file}")
        except Exception as e:
            print(f"  Error processing {video_file}: {e}")
    
    # Summary
    print("\n=== Batch Processing Summary ===")
    for video_file, analysis in results.items():
        print(f"{video_file}: {analysis['detection_rate']:.2%} detection rate")


def main():
    """
    Main function to run all examples.
    """
    print("Tennis Ball Detection System - Examples")
    print("=" * 50)
    
    # Run examples
    example_single_frame_detection()
    example_custom_parameters()
    
    # Uncomment the following lines to run video processing examples
    # (Make sure you have tennis video files available)
    
    # example_video_processing()
    # example_batch_processing()
    
    print("\nExamples completed!")
    print("\nTo process your own tennis videos:")
    print("1. Place your video files in the project directory")
    print("2. Update the video paths in the examples")
    print("3. Run: python tennis_ball_detector.py your_video.mp4")


if __name__ == "__main__":
    main()





