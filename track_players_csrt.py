"""
Track two tennis players using OpenCV CSRT tracker.
Player 1 = Blue marker (X), Player 2 = Red marker (X)
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CSRTPlayerTracker:
    def __init__(self, initial_pos1: Tuple[int, int], initial_pos2: Tuple[int, int]):
        self.player1_initial = initial_pos1  # Blue
        self.player2_initial = initial_pos2  # Red
        
        # Bounding box size for initial detection
        self.bbox_size = 80
        
        # Screen dimensions
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Court coordinates (from adjusted points)
        self.court_points = {
            1: (1043, 214), 2: (1608, 170), 3: (2042, 1449), 4: (3002, 1060),
            5: (1120, 310), 6: (1782, 262), 7: (1549, 854), 8: (2490, 696),
            9: (1041, 421), 10: (2229, 331), 11: (1468, 278), 12: (2092, 764),
            13: (1334, 186), 14: (2656, 1216)
        }
        
        # Trackers
        self.tracker1 = None
        self.tracker2 = None
        
        # Current positions
        self.player1_bbox = None
        self.player2_bbox = None
    
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
    
    def _create_bbox_from_center(self, center: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Create bounding box from center point."""
        x, y = center
        half_size = self.bbox_size // 2
        return (x - half_size, y - half_size, self.bbox_size, self.bbox_size)
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point from bounding box."""
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def _draw_player_markers(self, frame: np.ndarray) -> np.ndarray:
        """Draw player markers on the frame."""
        result = frame.copy()
        
        # Draw Player 1 (Blue) with X
        if self.player1_bbox is not None:
            x, y = self._get_bbox_center(self.player1_bbox)
            size = 25
            thickness = 3
            
            # Draw X
            cv2.line(result, (x - size, y - size), (x + size, y + size), (255, 0, 0), thickness)
            cv2.line(result, (x - size, y + size), (x + size, y - size), (255, 0, 0), thickness)
            
            # Draw circle around X
            cv2.circle(result, (x, y), 35, (255, 0, 0), 2)
            
            # Draw bounding box
            bx, by, bw, bh = [int(v) for v in self.player1_bbox]
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
            
            # Draw label
            cv2.putText(result, "P1", (x + 40, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw Player 2 (Red) with X
        if self.player2_bbox is not None:
            x, y = self._get_bbox_center(self.player2_bbox)
            size = 25
            thickness = 3
            
            # Draw X
            cv2.line(result, (x - size, y - size), (x + size, y + size), (0, 0, 255), thickness)
            cv2.line(result, (x - size, y + size), (x + size, y - size), (0, 0, 255), thickness)
            
            # Draw circle around X
            cv2.circle(result, (x, y), 35, (0, 0, 255), 2)
            
            # Draw bounding box
            bx, by, bw, bh = [int(v) for v in self.player2_bbox]
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            
            # Draw label
            cv2.putText(result, "P2", (x + 40, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result
    
    def _draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw court lines in yellow."""
        result = frame.copy()
        yellow = (0, 255, 255)
        thickness = 2
        
        lines = [
            (1, 5), (5, 7), (7, 3),  # Left sideline
            (2, 6), (6, 8), (8, 4),  # Right sideline
            (1, 13), (13, 2),        # Top baseline
            (3, 14), (14, 4),        # Bottom baseline
            (5, 6), (7, 8),          # Service lines
            (9, 10),                 # Net
            (11, 12),                # Center service line
        ]
        
        for p1, p2 in lines:
            pt1 = self.court_points[p1]
            pt2 = self.court_points[p2]
            cv2.line(result, pt1, pt2, yellow, thickness)
        
        return result
    
    def process_video(self, input_path: str, output_path: str):
        """Process video and track players using CSRT tracker."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video file: {output_path}")
            cap.release()
            return
        
        print("Initializing CSRT trackers...")
        print("Press 'Q' key to stop processing and exit")
        
        frame_count = 0
        
        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read first frame")
                return
            
            # Initialize trackers with first frame
            self.tracker1 = cv2.TrackerCSRT_create()
            self.tracker2 = cv2.TrackerCSRT_create()
            
            # Create initial bounding boxes
            self.player1_bbox = self._create_bbox_from_center(self.player1_initial)
            self.player2_bbox = self._create_bbox_from_center(self.player2_initial)
            
            # Initialize trackers
            self.tracker1.init(frame, self.player1_bbox)
            self.tracker2.init(frame, self.player2_bbox)
            
            print(f"Player 1 (Blue) initialized at: {self.player1_initial}")
            print(f"Player 2 (Red) initialized at: {self.player2_initial}")
            print("Tracking players...")
            
            # Process first frame
            frame_with_lines = self._draw_court_lines(frame)
            frame_with_markers = self._draw_player_markers(frame_with_lines)
            out.write(frame_with_markers)
            frame_count += 1
            
            # Process remaining frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update trackers
                success1, bbox1 = self.tracker1.update(frame)
                success2, bbox2 = self.tracker2.update(frame)
                
                if success1:
                    self.player1_bbox = bbox1
                else:
                    print(f"Warning: Lost tracking of Player 1 at frame {frame_count}")
                
                if success2:
                    self.player2_bbox = bbox2
                else:
                    print(f"Warning: Lost tracking of Player 2 at frame {frame_count}")
                
                # Draw court lines
                frame_with_lines = self._draw_court_lines(frame)
                
                # Draw player markers
                frame_with_markers = self._draw_player_markers(frame_with_lines)
                
                # Write to output
                out.write(frame_with_markers)
                
                # Display progress
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # Display frame
                display_frame, _ = self._resize_frame_to_fit_screen(frame_with_markers)
                cv2.imshow('CSRT Player Tracking - Press Q to quit', display_frame)
                
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
    # Load player positions from file
    try:
        with open('player_positions.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Parse positions
            for line in lines:
                if "Player 1 (Blue):" in line:
                    pos_str = line.split(":")[1].strip()
                    pos_str = pos_str.replace("(", "").replace(")", "")
                    x, y = map(int, pos_str.split(","))
                    player1_pos = (x, y)
                elif "Player 2 (Red):" in line:
                    pos_str = line.split(":")[1].strip()
                    pos_str = pos_str.replace("(", "").replace(")", "")
                    x, y = map(int, pos_str.split(","))
                    player2_pos = (x, y)
    except FileNotFoundError:
        print("Error: player_positions.txt not found. Please run mark_players_initial.py first.")
        return
    
    print("Tennis Player Tracking with CSRT")
    print("=" * 50)
    print(f"Initial Player 1 (Blue) position: {player1_pos}")
    print(f"Initial Player 2 (Red) position: {player2_pos}")
    print()
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_with_csrt_tracking.mp4"
    
    tracker = CSRTPlayerTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()











