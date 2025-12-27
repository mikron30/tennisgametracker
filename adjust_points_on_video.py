"""
Interactive point adjustment tool for tennis court lines.
Shows the first frame of the video and allows dragging points to adjust their positions.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class InteractivePointAdjuster:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.window_name = "Adjust Court Points - Drag to move, Press 'S' to save, 'Q' to quit"
        
        # Original clicked points from previous session
        self.points = {
            1: (467, 214),   # Top-left corner
            2: (1033, 174),  # Top-right corner
            3: (1466, 1449), # Bottom-left corner
            4: (2426, 1060), # Bottom-right corner
            5: (546, 315),   # Baseline top-left
            6: (1199, 263),  # Baseline top-right
            7: (973, 854),   # Baseline bottom-left
            8: (1914, 696),  # Baseline bottom-right
            9: (465, 421),   # Net left
            10: (1653, 331), # Net right
            11: ((546 + 1199) // 2, (315 + 263) // 2),  # Middle between 5 and 6
            12: ((973 + 1914) // 2, (854 + 696) // 2), # Middle between 7 and 8
            13: ((467 + 1033) // 2, (214 + 174) // 2),  # Middle of top baseline (between 1 and 2)
            14: ((1466 + 2426) // 2, (1449 + 1060) // 2)  # Middle of bottom baseline (between 3 and 4)
        }
        
        # Scale points to full video frame
        self.scaled_points = {}
        
        # Mouse interaction state
        self.dragging_point = None
        self.mouse_pos = None
        
        # Display frame
        self.display_frame = None
        self.original_frame = None
        
        # Screen dimensions
        self.screen_width = 1920
        self.screen_height = 1080
        
    def _scale_points_to_video(self, video_width: int, video_height: int):
        """Scale points from right court coordinates to full video frame."""
        # Original right court dimensions
        right_court_width = 3264
        right_court_height = 2160
        
        # Right court starts at 15% from left
        right_court_start_x = int(video_width * 0.15)
        
        # Calculate scale factors
        scale_x = (video_width - right_court_start_x) / right_court_width
        scale_y = video_height / right_court_height
        
        # Scale all points
        for point_num, (x, y) in self.points.items():
            scaled_x = right_court_start_x + int(x * scale_x)
            scaled_y = int(y * scale_y)
            self.scaled_points[point_num] = [scaled_x, scaled_y]
    
    def _resize_frame_to_fit_screen(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize frame to fit screen while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        
        # Calculate scale to fit screen
        scale_w = self.screen_width / width
        scale_h = self.screen_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def _draw_points_and_lines(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Draw all points and connecting lines on the frame."""
        result = frame.copy()
        
        # Scale points for display
        display_points = {}
        for num, (x, y) in self.scaled_points.items():
            display_points[num] = (int(x * scale), int(y * scale))
        
        # Yellow for lines
        yellow = (0, 255, 255)
        thickness = 2
        
        # Draw court lines
        lines = [
            # Left sideline (1-5-7-3)
            (1, 5),
            (5, 7),
            (7, 3),
            # Right sideline (2-6-8-4)
            (2, 6),
            (6, 8),
            (8, 4),
            # Top baseline (1-13-2)
            (1, 13),
            (13, 2),
            # Bottom baseline (3-14-4)
            (3, 14),
            (14, 4),
            # Inner baseline top (5-6)
            (5, 6),
            # Inner baseline bottom (7-8)
            (7, 8),
            # Net line (9-10)
            (9, 10),
            # Center service line (11-12)
            (11, 12),
        ]
        
        for p1, p2 in lines:
            if p1 in display_points and p2 in display_points:
                cv2.line(result, display_points[p1], display_points[p2], yellow, thickness)
        
        # Draw points
        for num, (x, y) in display_points.items():
            # Draw circle
            color = (0, 255, 0) if self.dragging_point != num else (0, 0, 255)  # Green or red if dragging
            cv2.circle(result, (x, y), 10, color, -1)
            cv2.circle(result, (x, y), 12, (255, 255, 255), 2)  # White border
            
            # Draw point number
            cv2.putText(result, str(num), (x + 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Drag points to adjust positions",
            "Press 'S' to save coordinates",
            "Press 'Q' to quit without saving"
        ]
        y_offset = 30
        for instruction in instructions:
            cv2.putText(result, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        return result
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for dragging points."""
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near a point
            for num, (px, py) in self.scaled_points.items():
                display_px = int(px * scale)
                display_py = int(py * scale)
                distance = np.sqrt((x - display_px)**2 + (y - display_py)**2)
                if distance < 15:  # Within 15 pixels
                    self.dragging_point = num
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point is not None:
                # Update point position (convert back from display scale)
                self.scaled_points[self.dragging_point] = [int(x / scale), int(y / scale)]
                # Redraw
                self.display_frame = self._draw_points_and_lines(self.original_frame, scale)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None
    
    def _save_coordinates(self, video_width: int, video_height: int):
        """Save adjusted coordinates to file."""
        # Convert back to right court coordinates
        right_court_start_x = int(video_width * 0.15)
        right_court_width = 3264
        right_court_height = 2160
        
        scale_x = (video_width - right_court_start_x) / right_court_width
        scale_y = video_height / right_court_height
        
        # Save to file
        with open('adjusted_court_points.txt', 'w', encoding='utf-8') as f:
            f.write("Adjusted Court Points Coordinates\n")
            f.write("=" * 50 + "\n\n")
            f.write("Full Video Frame Coordinates:\n")
            f.write("-" * 50 + "\n")
            
            for num in sorted(self.scaled_points.keys()):
                x, y = self.scaled_points[num]
                f.write(f"Point {num:2d}: ({x:4d}, {y:4d})\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Right Court Region Coordinates (for reference):\n")
            f.write("-" * 50 + "\n")
            
            for num in sorted(self.scaled_points.keys()):
                x, y = self.scaled_points[num]
                # Convert back to right court coordinates
                orig_x = int((x - right_court_start_x) / scale_x)
                orig_y = int(y / scale_y)
                f.write(f"Point {num:2d}: ({orig_x:4d}, {orig_y:4d})\n")
        
        print("\nCoordinates saved to: adjusted_court_points.txt")
        print("\nFull Video Frame Coordinates:")
        print("-" * 50)
        for num in sorted(self.scaled_points.keys()):
            x, y = self.scaled_points[num]
            print(f"Point {num:2d}: ({x:4d}, {y:4d})")
    
    def run(self):
        """Run the interactive point adjustment tool."""
        # Open video and get first frame
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read first frame from video")
            return
        
        video_height, video_width = frame.shape[:2]
        print(f"Video dimensions: {video_width}x{video_height}")
        
        # Scale points to video coordinates
        self._scale_points_to_video(video_width, video_height)
        
        # Resize frame to fit screen
        self.original_frame, scale = self._resize_frame_to_fit_screen(frame)
        
        print(f"\nDisplay scale: {scale:.3f}")
        print(f"Display size: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}")
        
        # Initial draw
        self.display_frame = self._draw_points_and_lines(self.original_frame, scale)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback, scale)
        
        print("\nInteractive Point Adjustment")
        print("=" * 50)
        print("Instructions:")
        print("- Click and drag points to adjust their positions")
        print("- Press 'S' to save the adjusted coordinates")
        print("- Press 'Q' to quit without saving")
        print("\nPoint labels:")
        print("  1-4: Court corners (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right)")
        print("  5-6: Top baseline corners")
        print("  7-8: Bottom baseline corners")
        print("  9-10: Net line ends")
        print("  11-12: Center service line points")
        print("  13: Middle of top baseline (between 1 and 2)")
        print("  14: Middle of bottom baseline (between 3 and 4)")
        
        # Main loop
        while True:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting without saving...")
                break
            elif key == ord('s') or key == ord('S'):
                self._save_coordinates(video_width, video_height)
                print("\nCoordinates saved! You can now quit or continue adjusting.")
        
        cv2.destroyAllWindows()


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    adjuster = InteractivePointAdjuster(video_path)
    adjuster.run()


if __name__ == "__main__":
    main()

