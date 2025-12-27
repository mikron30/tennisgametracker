#!/usr/bin/env python3
"""
Detect specific tennis court lines and mark them in purple on video.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


class CourtLineDetector:
    """
    Detects specific tennis court lines: singles sidelines, baselines, net, and service lines.
    """
    
    def __init__(self):
        self.court_lines = {
            'left_sideline': None,
            'right_sideline': None,
            'top_baseline': None,
            'bottom_baseline': None,
            'net_line': None,
            'top_service_line': None,
            'bottom_service_line': None
        }
    
    def extract_white_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract white lines from the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Binary mask with white lines
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define white color range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create white mask
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        return white_mask
    
    def detect_horizontal_lines(self, mask: np.ndarray) -> List[int]:
        """
        Detect horizontal lines (baselines, net, service lines).
        
        Args:
            mask: White lines mask
            
        Returns:
            List of Y coordinates of horizontal lines
        """
        # Use Hough Line Transform to find lines
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if it's horizontal (Y coordinates are similar)
                if abs(y2 - y1) < 10:
                    line_y = (y1 + y2) // 2
                    horizontal_lines.append(line_y)
        
        # Remove duplicates and sort
        horizontal_lines = sorted(list(set(horizontal_lines)))
        return horizontal_lines
    
    def detect_vertical_lines(self, mask: np.ndarray) -> List[int]:
        """
        Detect vertical lines (sidelines).
        
        Args:
            mask: White lines mask
            
        Returns:
            List of X coordinates of vertical lines
        """
        # Use Hough Line Transform to find lines
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        vertical_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if it's vertical (X coordinates are similar)
                if abs(x2 - x1) < 10:
                    line_x = (x1 + x2) // 2
                    vertical_lines.append(line_x)
        
        # Remove duplicates and sort
        vertical_lines = sorted(list(set(vertical_lines)))
        return vertical_lines
    
    def classify_court_lines(self, frame: np.ndarray) -> Dict:
        """
        Classify detected lines into specific court line types.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with classified court lines
        """
        height, width = frame.shape[:2]
        
        # Get right court region (focus on right court only)
        right_start_x = int(width * 0.15)
        right_court = frame[:, right_start_x:width]
        
        # Extract white lines
        white_mask = self.extract_white_lines(right_court)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self.detect_horizontal_lines(white_mask)
        vertical_lines = self.detect_vertical_lines(white_mask)
        
        # Adjust coordinates back to full frame
        horizontal_lines = [y for y in horizontal_lines]  # Y coordinates stay the same
        vertical_lines = [x + right_start_x for x in vertical_lines]  # X coordinates adjusted
        
        # Classify lines based on position
        court_lines = {
            'left_sideline': None,
            'right_sideline': None,
            'top_baseline': None,
            'bottom_baseline': None,
            'net_line': None,
            'top_service_line': None,
            'bottom_service_line': None
        }
        
        # Classify horizontal lines
        if len(horizontal_lines) >= 3:
            # Sort by Y coordinate
            horizontal_lines.sort()
            
            # Net is typically in the middle
            middle_idx = len(horizontal_lines) // 2
            court_lines['net_line'] = horizontal_lines[middle_idx]
            
            # Baselines are at the extremes
            court_lines['top_baseline'] = horizontal_lines[0]
            court_lines['bottom_baseline'] = horizontal_lines[-1]
            
            # Service lines are between net and baselines
            if len(horizontal_lines) >= 5:
                net_idx = middle_idx
                court_lines['top_service_line'] = horizontal_lines[net_idx - 1]
                court_lines['bottom_service_line'] = horizontal_lines[net_idx + 1]
        
        # Classify vertical lines (sidelines)
        if len(vertical_lines) >= 2:
            vertical_lines.sort()
            court_lines['left_sideline'] = vertical_lines[0]
            court_lines['right_sideline'] = vertical_lines[-1]
        
        return court_lines
    
    def draw_court_lines(self, frame: np.ndarray, court_lines: Dict) -> np.ndarray:
        """
        Draw detected court lines in purple on the frame.
        
        Args:
            frame: Input frame
            court_lines: Dictionary with court line coordinates
            
        Returns:
            Frame with court lines drawn in purple
        """
        result_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Purple color (BGR format)
        purple = (255, 0, 255)
        
        # Draw horizontal lines
        if court_lines['top_baseline'] is not None:
            cv2.line(result_frame, (0, court_lines['top_baseline']), 
                    (width, court_lines['top_baseline']), purple, 3)
            cv2.putText(result_frame, "TOP BASELINE", 
                       (10, court_lines['top_baseline'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        if court_lines['bottom_baseline'] is not None:
            cv2.line(result_frame, (0, court_lines['bottom_baseline']), 
                    (width, court_lines['bottom_baseline']), purple, 3)
            cv2.putText(result_frame, "BOTTOM BASELINE", 
                       (10, court_lines['bottom_baseline'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        if court_lines['net_line'] is not None:
            cv2.line(result_frame, (0, court_lines['net_line']), 
                    (width, court_lines['net_line']), purple, 4)
            cv2.putText(result_frame, "NET", 
                       (width - 100, court_lines['net_line'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, purple, 2)
        
        if court_lines['top_service_line'] is not None:
            cv2.line(result_frame, (0, court_lines['top_service_line']), 
                    (width, court_lines['top_service_line']), purple, 2)
            cv2.putText(result_frame, "TOP SERVICE", 
                       (10, court_lines['top_service_line'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        if court_lines['bottom_service_line'] is not None:
            cv2.line(result_frame, (0, court_lines['bottom_service_line']), 
                    (width, court_lines['bottom_service_line']), purple, 2)
            cv2.putText(result_frame, "BOTTOM SERVICE", 
                       (10, court_lines['bottom_service_line'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        # Draw vertical lines (sidelines)
        if court_lines['left_sideline'] is not None:
            cv2.line(result_frame, (court_lines['left_sideline'], 0), 
                    (court_lines['left_sideline'], height), purple, 3)
            cv2.putText(result_frame, "LEFT SIDELINE", 
                       (court_lines['left_sideline'] + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        if court_lines['right_sideline'] is not None:
            cv2.line(result_frame, (court_lines['right_sideline'], 0), 
                    (court_lines['right_sideline'], height), purple, 3)
            cv2.putText(result_frame, "RIGHT SIDELINE", 
                       (court_lines['right_sideline'] + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        return result_frame


def process_video_with_court_lines(video_path: str, output_path: str = None):
    """
    Process video and mark court lines in purple.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {width}x{height}, {fps} FPS")
    print("Press 'Q' key to stop processing and exit")
    
    # Setup video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize court line detector
    detector = CourtLineDetector()
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect court lines every 30 frames to update detection
            if frame_count % 30 == 0:
                court_lines = detector.classify_court_lines(frame)
            
            # Draw court lines on frame
            result_frame = detector.draw_court_lines(frame, court_lines)
            
            # Add frame info
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            if out:
                out.write(result_frame)
            
            # Resize for display
            display_frame = cv2.resize(result_frame, (1200, 800))
            cv2.imshow('Tennis Court Lines Detection', display_frame)
            
            # Check for 'q' key press to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Processing stopped by user (Q key pressed)")
                cv2.destroyAllWindows()
                break
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    print(f"Processing complete! Processed {frame_count} frames")
    if output_path:
        print(f"Output video saved to: {output_path}")


def main():
    """
    Main function to run court line detection on video.
    """
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "court_lines_detected.mp4"
    
    print("Tennis Court Lines Detection")
    print("=" * 40)
    print("Detecting:")
    print("- Singles sidelines (left and right)")
    print("- Baselines (top and bottom)")
    print("- Net line (middle)")
    print("- Service lines (high and low service boxes)")
    print("All lines will be marked in PURPLE")
    print()
    
    process_video_with_court_lines(video_path, output_path)


if __name__ == "__main__":
    main()








