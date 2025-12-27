"""
Interactive tool to mark ball position while playing video.
User can pause video and click on the ball.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class VideoballMarker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.window_name = "Mark Ball - SPACE to pause/play, Click on ball, 'S' to save, 'Q' to quit"
        
        # Ball position
        self.ball_pos = None  # Green
        
        # Video state
        self.paused = False
        self.current_frame = None
        self.current_frame_number = 0
        
        # Display
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
    
    def _draw_marker(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Draw ball marker on the frame."""
        result = frame.copy()
        
        # Draw Ball (Green) marker
        if self.ball_pos is not None:
            x, y = self.ball_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            
            # Green circle
            cv2.circle(result, (display_x, display_y), 20, (0, 255, 0), 2)
            # Green X
            size = 15
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 255, 0), 2)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 255, 0), 2)
            cv2.putText(result, "Ball", (display_x - 30, display_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw status
        status_text = "PAUSED - Click on ball" if self.paused else "PLAYING - Press SPACE to pause"
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Draw frame number
        cv2.putText(result, f"Frame: {self.current_frame_number}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "SPACE: Pause/Play",
            "Click: Mark ball",
            "S: Save position",
            "Q: Quit"
        ]
        y_offset = result.shape[0] - 120
        for instruction in instructions:
            cv2.putText(result, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return result
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for marking ball."""
        if not self.paused:
            return
        
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark ball - convert from display scale to original
            self.ball_pos = (int(x / scale), int(y / scale))
            print(f"Ball marked at: ({self.ball_pos[0]}, {self.ball_pos[1]}) on frame {self.current_frame_number}")
    
    def _save_position(self):
        """Save ball position to file."""
        if self.ball_pos is None:
            print("\nError: Ball position must be marked before saving!")
            return False
        
        with open('ball_position.txt', 'w', encoding='utf-8') as f:
            f.write("Initial Ball Position\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Ball (Green): ({self.ball_pos[0]}, {self.ball_pos[1]})\n")
            f.write(f"Frame number: {self.current_frame_number}\n")
        
        print("\nBall position saved to: ball_position.txt")
        print(f"Ball (Green): ({self.ball_pos[0]}, {self.ball_pos[1]})")
        print(f"Frame number: {self.current_frame_number}")
        return True
    
    def run(self):
        """Run the interactive ball marking tool with video playback."""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Calculate display scale
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
        _, self.display_scale = self._resize_frame_to_fit_screen(test_frame)
        
        print(f"Display scale: {self.display_scale:.3f}")
        print("\nMark Ball Position on Video")
        print("=" * 50)
        print("Instructions:")
        print("- Video will start playing automatically")
        print("- Press SPACE to pause the video")
        print("- When paused, click on the ball to mark it")
        print("- The ball should be above the server's head at serve")
        print("- Press 'S' to save the position")
        print("- Press 'Q' to quit")
        print()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback, self.display_scale)
        
        saved = False
        
        try:
            while True:
                if not self.paused:
                    # Read next frame
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video reached")
                        break
                    
                    self.current_frame = frame
                    self.current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # Draw marker on current frame
                display_frame, _ = self._resize_frame_to_fit_screen(self.current_frame)
                display_frame = self._draw_marker(display_frame, self.display_scale)
                
                # Show frame
                cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(30 if not self.paused else 1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    if not saved and self.ball_pos is not None:
                        print("\nWarning: Position not saved!")
                    print("\nQuitting...")
                    break
                elif key == ord(' '):  # Space bar
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "PLAYING"
                    print(f"{status} at frame {self.current_frame_number}")
                elif key == ord('s') or key == ord('S'):
                    if self._save_position():
                        saved = True
                        print("\nPosition saved! You can continue or quit (Q).")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    marker = VideoballMarker(video_path)
    marker.run()


if __name__ == "__main__":
    main()











