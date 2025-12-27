"""
Interactive tool to mark initial positions of two players.
Player 1 = Blue, Player 2 = Red
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class PlayerMarker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.window_name = "Mark Players - Left click: Player 1 (Blue), Right click: Player 2 (Red), 'S' to save"
        
        # Player positions (will be scaled to full video frame)
        self.player1_pos = None  # Blue
        self.player2_pos = None  # Red
        
        # Display frame
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
        """Draw player markers on the frame."""
        result = frame.copy()
        
        # Draw Player 1 (Blue) marker
        if self.player1_pos is not None:
            x, y = self.player1_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            
            # Blue circle
            cv2.circle(result, (display_x, display_y), 30, (255, 0, 0), 3)
            cv2.putText(result, "Player 1", (display_x - 40, display_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw Player 2 (Red) marker
        if self.player2_pos is not None:
            x, y = self.player2_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            
            # Red circle
            cv2.circle(result, (display_x, display_y), 30, (0, 0, 255), 3)
            cv2.putText(result, "Player 2", (display_x - 40, display_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw instructions
        instructions = [
            "Left Click: Mark Player 1 (Blue)",
            "Right Click: Mark Player 2 (Red)",
            "Press 'S' to save positions",
            "Press 'Q' to quit"
        ]
        y_offset = 30
        for instruction in instructions:
            cv2.putText(result, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        return result
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for marking players."""
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark Player 1 (Blue) - convert from display scale to original
            self.player1_pos = (int(x / scale), int(y / scale))
            print(f"Player 1 (Blue) marked at: ({self.player1_pos[0]}, {self.player1_pos[1]})")
            # Redraw
            self.display_frame = self._draw_markers(self.original_frame, scale)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Mark Player 2 (Red) - convert from display scale to original
            self.player2_pos = (int(x / scale), int(y / scale))
            print(f"Player 2 (Red) marked at: ({self.player2_pos[0]}, {self.player2_pos[1]})")
            # Redraw
            self.display_frame = self._draw_markers(self.original_frame, scale)
    
    def _save_positions(self):
        """Save player positions to file."""
        if self.player1_pos is None or self.player2_pos is None:
            print("\nError: Both players must be marked before saving!")
            return False
        
        with open('player_positions.txt', 'w', encoding='utf-8') as f:
            f.write("Initial Player Positions\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Player 1 (Blue): ({self.player1_pos[0]}, {self.player1_pos[1]})\n")
            f.write(f"Player 2 (Red): ({self.player2_pos[0]}, {self.player2_pos[1]})\n")
        
        print("\nPlayer positions saved to: player_positions.txt")
        print(f"Player 1 (Blue): ({self.player1_pos[0]}, {self.player1_pos[1]})")
        print(f"Player 2 (Red): ({self.player2_pos[0]}, {self.player2_pos[1]})")
        return True
    
    def run(self):
        """Run the interactive player marking tool."""
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
        
        print("\nMark Initial Player Positions")
        print("=" * 50)
        print("Instructions:")
        print("- Left click to mark Player 1 (Blue)")
        print("- Right click to mark Player 2 (Red)")
        print("- Press 'S' to save positions")
        print("- Press 'Q' to quit")
        
        # Main loop
        saved = False
        while True:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                if not saved and (self.player1_pos is not None or self.player2_pos is not None):
                    print("\nWarning: Positions not saved!")
                print("\nQuitting...")
                break
            elif key == ord('s') or key == ord('S'):
                if self._save_positions():
                    saved = True
                    print("\nPositions saved! You can now quit or continue adjusting.")
        
        cv2.destroyAllWindows()


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    marker = PlayerMarker(video_path)
    marker.run()


if __name__ == "__main__":
    main()











