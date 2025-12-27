"""
Interactive tool to mark:
1. Player 1 (Blue) - Left click
2. Player 2 (Red) - Right click  
3. Serve area (Green rectangle) - Press 'R' then drag rectangle where ball appears during serve
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class PlayerAndServeMarker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.window_name = "Mark Players & Serve Area - See instructions below"
        
        # Player positions
        self.player1_pos = None  # Blue
        self.player2_pos = None  # Red
        
        # Serve area (rectangle where to look for ball)
        self.serve_area = None  # (x1, y1, x2, y2)
        self.drawing_rect = False
        self.rect_start = None
        self.rect_end = None
        
        # Display
        self.display_frame = None
        self.original_frame = None
        self.display_scale = 1.0
        
        # Screen dimensions
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
    
    def _draw_markers(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Draw all markers on the frame."""
        result = frame.copy()
        
        # Draw Player 1 (Blue)
        if self.player1_pos is not None:
            x, y = self.player1_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            
            cv2.circle(result, (display_x, display_y), 30, (255, 0, 0), 3)
            cv2.putText(result, "Player 1", (display_x - 40, display_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw Player 2 (Red)
        if self.player2_pos is not None:
            x, y = self.player2_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            
            cv2.circle(result, (display_x, display_y), 30, (0, 0, 255), 3)
            cv2.putText(result, "Player 2", (display_x - 40, display_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw Serve Area (Green rectangle)
        if self.serve_area is not None:
            x1, y1, x2, y2 = self.serve_area
            display_x1 = int(x1 * scale)
            display_y1 = int(y1 * scale)
            display_x2 = int(x2 * scale)
            display_y2 = int(y2 * scale)
            
            cv2.rectangle(result, (display_x1, display_y1), (display_x2, display_y2), 
                         (0, 255, 0), 3)
            cv2.putText(result, "Serve Area", (display_x1, display_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw temporary rectangle while drawing
        if self.drawing_rect and self.rect_start is not None and self.rect_end is not None:
            cv2.rectangle(result, self.rect_start, self.rect_end, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Left Click: Mark Player 1 (Blue)",
            "Right Click: Mark Player 2 (Red)",
            "Press 'R' then drag: Draw serve area (Green)",
            "Press 'S' to save",
            "Press 'Q' to quit"
        ]
        y_offset = 30
        for instruction in instructions:
            cv2.putText(result, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Status
        status = []
        if self.player1_pos: status.append("P1✓")
        if self.player2_pos: status.append("P2✓")
        if self.serve_area: status.append("Serve✓")
        if status:
            cv2.putText(result, " ".join(status), (10, result.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing_rect:
                # Mark Player 1 (Blue)
                self.player1_pos = (int(x / scale), int(y / scale))
                print(f"Player 1 (Blue) marked at: ({self.player1_pos[0]}, {self.player1_pos[1]})")
                self.display_frame = self._draw_markers(self.original_frame, scale)
            else:
                # Start drawing rectangle
                self.rect_start = (x, y)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Mark Player 2 (Red)
            self.player2_pos = (int(x / scale), int(y / scale))
            print(f"Player 2 (Red) marked at: ({self.player2_pos[0]}, {self.player2_pos[1]})")
            self.display_frame = self._draw_markers(self.original_frame, scale)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_rect and self.rect_start is not None:
                self.rect_end = (x, y)
                self.display_frame = self._draw_markers(self.original_frame, scale)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing_rect and self.rect_start is not None:
                self.rect_end = (x, y)
                # Convert to original frame coordinates
                x1 = int(min(self.rect_start[0], self.rect_end[0]) / scale)
                y1 = int(min(self.rect_start[1], self.rect_end[1]) / scale)
                x2 = int(max(self.rect_start[0], self.rect_end[0]) / scale)
                y2 = int(max(self.rect_start[1], self.rect_end[1]) / scale)
                
                self.serve_area = (x1, y1, x2, y2)
                print(f"Serve area marked: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Reset drawing state
                self.drawing_rect = False
                self.rect_start = None
                self.rect_end = None
                
                self.display_frame = self._draw_markers(self.original_frame, scale)
    
    def _save_positions(self):
        """Save all positions to file."""
        if self.player1_pos is None or self.player2_pos is None or self.serve_area is None:
            print("\nError: All markers must be set before saving!")
            print(f"  Player 1: {'✓' if self.player1_pos else '✗'}")
            print(f"  Player 2: {'✓' if self.player2_pos else '✗'}")
            print(f"  Serve Area: {'✓' if self.serve_area else '✗'}")
            return False
        
        with open('tracking_config.txt', 'w', encoding='utf-8') as f:
            f.write("Player and Ball Tracking Configuration\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Player 1 (Blue): ({self.player1_pos[0]}, {self.player1_pos[1]})\n")
            f.write(f"Player 2 (Red): ({self.player2_pos[0]}, {self.player2_pos[1]})\n")
            f.write(f"Serve Area: ({self.serve_area[0]}, {self.serve_area[1]}, "
                   f"{self.serve_area[2]}, {self.serve_area[3]})\n")
        
        print("\nConfiguration saved to: tracking_config.txt")
        print(f"Player 1 (Blue): ({self.player1_pos[0]}, {self.player1_pos[1]})")
        print(f"Player 2 (Red): ({self.player2_pos[0]}, {self.player2_pos[1]})")
        print(f"Serve Area: {self.serve_area}")
        return True
    
    def run(self):
        """Run the interactive marking tool."""
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
        
        # Resize frame to fit screen
        self.original_frame, self.display_scale = self._resize_frame_to_fit_screen(frame)
        
        print(f"Display scale: {self.display_scale:.3f}")
        print(f"Display size: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}")
        
        # Initial draw
        self.display_frame = self._draw_markers(self.original_frame, self.display_scale)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback, self.display_scale)
        
        print("\nMark Players and Serve Area")
        print("=" * 50)
        print("Instructions:")
        print("1. Left click to mark Player 1 (Blue)")
        print("2. Right click to mark Player 2 (Red)")
        print("3. Press 'R' to start drawing serve area rectangle")
        print("4. Drag mouse to draw rectangle above server's head")
        print("5. Release to finish rectangle")
        print("6. Press 'S' to save configuration")
        print("7. Press 'Q' to quit")
        
        # Main loop
        saved = False
        while True:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                if not saved and (self.player1_pos or self.player2_pos or self.serve_area):
                    print("\nWarning: Configuration not saved!")
                print("\nQuitting...")
                break
            elif key == ord('r') or key == ord('R'):
                self.drawing_rect = True
                print("Draw rectangle mode activated - drag to draw serve area")
            elif key == ord('s') or key == ord('S'):
                if self._save_positions():
                    saved = True
                    print("\nConfiguration saved! You can now quit or continue adjusting.")
        
        cv2.destroyAllWindows()


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    marker = PlayerAndServeMarker(video_path)
    marker.run()


if __name__ == "__main__":
    main()











