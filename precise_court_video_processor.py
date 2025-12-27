#!/usr/bin/env python3
"""
Process tennis video with precise court line detection using manually marked coordinates.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def resize_to_fit_screen(frame: np.ndarray, max_width: int = 1200, max_height: int = 800) -> np.ndarray:
    """
    Resize frame to fit within specified screen dimensions while maintaining aspect ratio.
    """
    height, width = frame.shape[:2]
    
    scale_h = max_height / height
    scale_w = max_width / width
    scale = min(scale_h, scale_w, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame


class PreciseCourtVideoProcessor:
    """
    Process tennis video with precise court line detection.
    """
    
    def __init__(self):
        # Precise coordinates extracted from marked image (3264x2160)
        self.coordinates = {
            'court_corners': {
                'top_left': (400, 300),      # Point 5
                'top_right': (2800, 300),    # Point 6
                'bottom_left': (400, 1800),  # Point 3
                'bottom_right': (2800, 1800) # Point 4
            },
            'net_points': {
                'left': (400, 1050),         # Point 9
                'right': (2800, 1050)        # Point 10
            },
            'baseline_middle': {
                'top': (1600, 650),          # Point 2
                'bottom': (1600, 1450)       # Point 12
            },
            'service_corners': {
                'top_left': (400, 650),      # Point 1
                'top_right': (2800, 650),    # Top service box right
                'bottom_left': (400, 1450),  # Point 7
                'bottom_right': (2800, 1450) # Point 8
            }
        }
    
    def draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw precise court lines using the extracted coordinates.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with court lines drawn in purple
        """
        result_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Scale coordinates if frame size is different
        scale_x = width / 3264.0
        scale_y = height / 2160.0
        
        # Scale coordinates
        coords = self._scale_coordinates(scale_x, scale_y)
        
        # Purple color (BGR format)
        purple = (255, 0, 255)
        
        # Draw court outline (singles sidelines and baselines)
        # Top baseline
        cv2.line(result_frame, 
                (coords['court_corners']['top_left'][0], coords['court_corners']['top_left'][1]),
                (coords['court_corners']['top_right'][0], coords['court_corners']['top_right'][1]), 
                purple, 3)
        cv2.putText(result_frame, "TOP BASELINE", 
                   (coords['court_corners']['top_left'][0], coords['court_corners']['top_left'][1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Bottom baseline
        cv2.line(result_frame, 
                (coords['court_corners']['bottom_left'][0], coords['court_corners']['bottom_left'][1]),
                (coords['court_corners']['bottom_right'][0], coords['court_corners']['bottom_right'][1]), 
                purple, 3)
        cv2.putText(result_frame, "BOTTOM BASELINE", 
                   (coords['court_corners']['bottom_left'][0], coords['court_corners']['bottom_left'][1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Left singles sideline
        cv2.line(result_frame, 
                (coords['court_corners']['top_left'][0], coords['court_corners']['top_left'][1]),
                (coords['court_corners']['bottom_left'][0], coords['court_corners']['bottom_left'][1]), 
                purple, 3)
        cv2.putText(result_frame, "LEFT SIDELINE", 
                   (coords['court_corners']['top_left'][0] + 10, coords['court_corners']['top_left'][1] + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Right singles sideline
        cv2.line(result_frame, 
                (coords['court_corners']['top_right'][0], coords['court_corners']['top_right'][1]),
                (coords['court_corners']['bottom_right'][0], coords['court_corners']['bottom_right'][1]), 
                purple, 3)
        cv2.putText(result_frame, "RIGHT SIDELINE", 
                   (coords['court_corners']['top_right'][0] - 150, coords['court_corners']['top_right'][1] + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Draw net line
        cv2.line(result_frame, 
                (coords['net_points']['left'][0], coords['net_points']['left'][1]),
                (coords['net_points']['right'][0], coords['net_points']['right'][1]), 
                purple, 4)
        cv2.putText(result_frame, "NET", 
                   (coords['net_points']['left'][0] + 500, coords['net_points']['left'][1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, purple, 2)
        
        # Draw service lines
        # Top service line
        cv2.line(result_frame, 
                (coords['service_corners']['top_left'][0], coords['service_corners']['top_left'][1]),
                (coords['service_corners']['top_right'][0], coords['service_corners']['top_right'][1]), 
                purple, 2)
        cv2.putText(result_frame, "TOP SERVICE", 
                   (coords['service_corners']['top_left'][0], coords['service_corners']['top_left'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        # Bottom service line
        cv2.line(result_frame, 
                (coords['service_corners']['bottom_left'][0], coords['service_corners']['bottom_left'][1]),
                (coords['service_corners']['bottom_right'][0], coords['service_corners']['bottom_right'][1]), 
                purple, 2)
        cv2.putText(result_frame, "BOTTOM SERVICE", 
                   (coords['service_corners']['bottom_left'][0], coords['service_corners']['bottom_left'][1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        # Draw center service line (vertical)
        center_x = width // 2
        cv2.line(result_frame, 
                (center_x, coords['service_corners']['top_left'][1]),
                (center_x, coords['service_corners']['bottom_left'][1]), 
                purple, 2)
        cv2.putText(result_frame, "CENTER SERVICE", 
                   (center_x + 10, coords['baseline_middle']['top'][1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        return result_frame
    
    def _scale_coordinates(self, scale_x: float, scale_y: float) -> Dict:
        """
        Scale coordinates based on frame size.
        
        Args:
            scale_x: X scaling factor
            scale_y: Y scaling factor
            
        Returns:
            Scaled coordinates dictionary
        """
        scaled_coords = {}
        
        for category, points in self.coordinates.items():
            scaled_coords[category] = {}
            for name, (x, y) in points.items():
                scaled_coords[category][name] = (int(x * scale_x), int(y * scale_y))
        
        return scaled_coords
    
    def process_video(self, video_path: str, output_path: str = None):
        """
        Process tennis video with precise court line detection.
        
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        print("Press 'Q' key to stop processing and exit")
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw court lines on frame
                result_frame = self.draw_court_lines(frame)
                
                # Add frame info
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                if out:
                    out.write(result_frame)
                
                # Resize for display
                display_frame = resize_to_fit_screen(result_frame)
                cv2.imshow('Precise Tennis Court Lines Detection', display_frame)
                
                # Check for 'q' key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("Processing stopped by user (Q key pressed)")
                    cv2.destroyAllWindows()
                    break
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
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
    Main function to run precise court line detection on video.
    """
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "precise_court_lines.mp4"
    
    print("Precise Tennis Court Lines Detection")
    print("=" * 50)
    print("Using manually marked coordinates from 12 points:")
    print("- Court corners (4 points)")
    print("- Baseline corners (4 points)")
    print("- Net points (2 points)")
    print("- Baseline middle points (2 points)")
    print("All court lines will be drawn in PURPLE")
    print()
    
    processor = PreciseCourtVideoProcessor()
    processor.process_video(video_path, output_path)


if __name__ == "__main__":
    main()








