"""
Track two tennis players throughout the video using color-based detection and optical flow.
Player 1 = Blue marker (X), Player 2 = Red marker (X)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class PlayerTracker:
    def __init__(self, initial_pos1: Tuple[int, int], initial_pos2: Tuple[int, int]):
        self.player1_pos = initial_pos1  # Blue
        self.player2_pos = initial_pos2  # Red
        
        # Tracking parameters
        self.search_radius = 150  # How far to search for player movement
        
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
    
    def _create_court_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create a mask for the court area to focus player detection."""
        height, width = frame_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create polygon from court corners
        court_polygon = np.array([
            self.court_points[1],  # top-left
            self.court_points[2],  # top-right
            self.court_points[4],  # bottom-right
            self.court_points[3],  # bottom-left
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [court_polygon], 255)
        return mask
    
    def _detect_player_in_region(self, frame: np.ndarray, center: Tuple[int, int], 
                                  court_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect player position in a region around the center point.
        Uses color-based detection to find the player.
        """
        x, y = center
        height, width = frame.shape[:2]
        
        # Define search region
        x1 = max(0, x - self.search_radius)
        y1 = max(0, y - self.search_radius)
        x2 = min(width, x + self.search_radius)
        y2 = min(height, y + self.search_radius)
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        region_mask = court_mask[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect player clothing colors (adjust ranges as needed)
        # Look for darker colors (typical tennis clothing)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 150])
        
        # Also look for bright colors
        lower_bright = np.array([0, 50, 100])
        upper_bright = np.array([180, 255, 255])
        
        # Create masks
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Combine masks
        player_mask = cv2.bitwise_or(dark_mask, bright_mask)
        
        # Apply court mask
        player_mask = cv2.bitwise_and(player_mask, region_mask)
        
        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return center  # Keep previous position if not found
        
        # Find largest contour closest to center
        best_contour = None
        best_distance = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200 or area > 15000:  # Filter by area
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate distance to previous position
            dx = cx - (x - x1)
            dy = cy - (y - y1)
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < best_distance:
                best_distance = distance
                best_contour = contour
                best_cx = cx + x1
                best_cy = cy + y1
        
        if best_contour is not None:
            return (best_cx, best_cy)
        
        return center  # Keep previous position if not found
    
    def _draw_player_markers(self, frame: np.ndarray) -> np.ndarray:
        """Draw player markers on the frame."""
        result = frame.copy()
        
        # Draw Player 1 (Blue) with X
        if self.player1_pos is not None:
            x, y = self.player1_pos
            size = 25
            thickness = 3
            
            # Draw X
            cv2.line(result, (x - size, y - size), (x + size, y + size), (255, 0, 0), thickness)
            cv2.line(result, (x - size, y + size), (x + size, y - size), (255, 0, 0), thickness)
            
            # Draw circle around X
            cv2.circle(result, (x, y), 35, (255, 0, 0), 2)
            
            # Draw label
            cv2.putText(result, "P1", (x + 40, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw Player 2 (Red) with X
        if self.player2_pos is not None:
            x, y = self.player2_pos
            size = 25
            thickness = 3
            
            # Draw X
            cv2.line(result, (x - size, y - size), (x + size, y + size), (0, 0, 255), thickness)
            cv2.line(result, (x - size, y + size), (x + size, y - size), (0, 0, 255), thickness)
            
            # Draw circle around X
            cv2.circle(result, (x, y), 35, (0, 0, 255), 2)
            
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
        """Process video and track players."""
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
        
        # Create court mask
        court_mask = self._create_court_mask((height, width))
        
        print("Tracking players and drawing court lines...")
        print("Press 'Q' key to stop processing and exit")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Track players (every 3 frames to improve performance)
                if frame_count % 3 == 0:
                    self.player1_pos = self._detect_player_in_region(frame, self.player1_pos, court_mask)
                    self.player2_pos = self._detect_player_in_region(frame, self.player2_pos, court_mask)
                
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
                cv2.imshow('Player Tracking - Press Q to quit', display_frame)
                
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
    
    print("Tennis Player Tracking")
    print("=" * 50)
    print(f"Initial Player 1 (Blue) position: {player1_pos}")
    print(f"Initial Player 2 (Red) position: {player2_pos}")
    print()
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_with_player_tracking.mp4"
    
    tracker = PlayerTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()











