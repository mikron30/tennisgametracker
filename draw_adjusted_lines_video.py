"""
Draw tennis court lines on video using the adjusted 14-point coordinates.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class AdjustedCourtLineDrawer:
    def __init__(self):
        # Adjusted coordinates from the interactive tool (Full Video Frame)
        self.points = {
            1: (1043, 214),   # Top-left corner
            2: (1608, 170),   # Top-right corner
            3: (2042, 1449),  # Bottom-left corner
            4: (3002, 1060),  # Bottom-right corner
            5: (1120, 310),   # Baseline top-left
            6: (1782, 262),   # Baseline top-right
            7: (1549, 854),   # Baseline bottom-left
            8: (2490, 696),   # Baseline bottom-right
            9: (1041, 421),   # Net left
            10: (2229, 331),  # Net right
            11: (1468, 278),  # Center service line top (between 5 and 6)
            12: (2092, 764),  # Center service line bottom (between 7 and 8)
            13: (1334, 186),  # Middle of top baseline (between 1 and 2)
            14: (2656, 1216)  # Middle of bottom baseline (between 3 and 4)
        }
        
        # Screen dimensions for display
        self.screen_width = 1920
        self.screen_height = 1080
    
    def _resize_frame_to_fit_screen(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize frame to fit screen while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        
        scale_w = self.screen_width / width
        scale_h = self.screen_height / height
        scale = min(scale_w, scale_h, 1.0)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all court lines in YELLOW using the adjusted coordinates.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with court lines drawn in yellow
        """
        result_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Scale coordinates if video size differs from reference (3840x2160)
        if width != 3840 or height != 2160:
            scale_x = width / 3840
            scale_y = height / 2160
            scaled_points = {}
            for num, (x, y) in self.points.items():
                scaled_points[num] = (int(x * scale_x), int(y * scale_y))
        else:
            scaled_points = self.points
        
        # Yellow color (BGR format)
        yellow = (0, 255, 255)
        thickness = 3
        
        # Define all lines to draw
        lines = [
            # Left sideline (1-5-7-3)
            (1, 5, "LEFT SIDELINE"),
            (5, 7, None),
            (7, 3, None),
            # Right sideline (2-6-8-4)
            (2, 6, "RIGHT SIDELINE"),
            (6, 8, None),
            (8, 4, None),
            # Top baseline (1-13-2)
            (1, 13, "TOP BASELINE"),
            (13, 2, None),
            # Bottom baseline (3-14-4)
            (3, 14, "BOTTOM BASELINE"),
            (14, 4, None),
            # Inner baseline top (5-6)
            (5, 6, "SERVICE LINE"),
            # Inner baseline bottom (7-8)
            (7, 8, "SERVICE LINE"),
            # Net line (9-10)
            (9, 10, "NET"),
            # Center service line (11-12)
            (11, 12, "CENTER"),
        ]
        
        # Draw all lines
        for line_data in lines:
            if len(line_data) == 3:
                p1, p2, label = line_data
            else:
                p1, p2 = line_data
                label = None
            
            pt1 = scaled_points[p1]
            pt2 = scaled_points[p2]
            
            cv2.line(result_frame, pt1, pt2, yellow, thickness)
            
            # Draw label if provided
            if label:
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                cv2.putText(result_frame, label, (mid_x + 10, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
        
        return result_frame
    
    def process_video(self, input_path: str, output_path: str):
        """
        Process video and draw court lines on each frame.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file: {output_path}")
            cap.release()
            return
        
        print("Drawing YELLOW court lines using adjusted 14-point coordinates")
        print("Press 'Q' key to stop processing and exit")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw court lines
                frame_with_lines = self.draw_court_lines(frame)
                
                # Write to output
                out.write(frame_with_lines)
                
                # Display progress
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Display frame (resized to fit screen)
                display_frame, _ = self._resize_frame_to_fit_screen(frame_with_lines)
                cv2.imshow('Court Lines - Press Q to quit', display_frame)
                
                # Check for 'Q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user (Q key pressed)")
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Processed {frame_count} frames")
        print(f"Output video saved to: {output_path}")


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_with_adjusted_lines.mp4"
    
    print("Tennis Court Lines - Using Adjusted 14-Point Coordinates")
    print("=" * 50)
    print(f"Input video: {video_path}")
    print(f"Output video: {output_path}")
    print()
    
    drawer = AdjustedCourtLineDrawer()
    drawer.process_video(video_path, output_path)


if __name__ == "__main__":
    main()

