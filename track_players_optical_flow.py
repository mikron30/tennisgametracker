"""
Track two tennis players using Lucas-Kanade optical flow.
Player 1 = Blue marker (X), Player 2 = Red marker (X)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class OpticalFlowPlayerTracker:
    def __init__(self, initial_pos1: Tuple[int, int], initial_pos2: Tuple[int, int]):
        self.player1_initial = initial_pos1  # Blue
        self.player2_initial = initial_pos2  # Red
        
        # Current positions
        self.player1_points = None
        self.player2_points = None
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=15,
            qualityLevel=0.1,
            minDistance=5,
            blockSize=7
        )
        
        # Previous frame for optical flow
        self.prev_gray = None
        
        # Detection region size
        self.region_size = 60
        
        # Screen dimensions
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Court coordinates
        self.court_points = {
            1: (1043, 214), 2: (1608, 170), 3: (2042, 1449), 4: (3002, 1060),
            5: (1120, 310), 6: (1782, 262), 7: (1549, 854), 8: (2490, 696),
            9: (1041, 421), 10: (2229, 331), 11: (1468, 278), 12: (2092, 764),
            13: (1334, 186), 14: (2656, 1216)
        }
        
        # Frame counter for re-detection
        self.frame_count = 0
        self.redetect_interval = 30  # Re-detect features every N frames
    
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
    
    def _detect_features_in_region(self, gray: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Detect good features to track in a region around the center."""
        height, width = gray.shape
        x, y = center
        
        # Define region
        x1 = max(0, x - self.region_size)
        y1 = max(0, y - self.region_size)
        x2 = min(width, x + self.region_size)
        y2 = min(height, y + self.region_size)
        
        # Create mask for region
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        
        # Detect features
        features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        if features is None:
            # If no features found, return center point
            return np.array([[[float(x), float(y)]]], dtype=np.float32)
        
        return features
    
    def _get_center_from_points(self, points: np.ndarray) -> Tuple[int, int]:
        """Calculate center from tracked points."""
        if points is None or len(points) == 0:
            return None
        
        # Calculate mean position
        mean_x = np.mean(points[:, 0, 0])
        mean_y = np.mean(points[:, 0, 1])
        
        return (int(mean_x), int(mean_y))
    
    def _draw_player_markers(self, frame: np.ndarray) -> np.ndarray:
        """Draw player markers on the frame."""
        result = frame.copy()
        
        # Draw Player 1 (Blue) with X
        if self.player1_points is not None:
            center = self._get_center_from_points(self.player1_points)
            if center:
                x, y = center
                size = 25
                thickness = 3
                
                # Draw X
                cv2.line(result, (x - size, y - size), (x + size, y + size), (255, 0, 0), thickness)
                cv2.line(result, (x - size, y + size), (x + size, y - size), (255, 0, 0), thickness)
                
                # Draw circle around X
                cv2.circle(result, (x, y), 35, (255, 0, 0), 2)
                
                # Draw tracked points
                for point in self.player1_points:
                    px, py = point.ravel()
                    cv2.circle(result, (int(px), int(py)), 3, (255, 100, 100), -1)
                
                # Draw label
                cv2.putText(result, "P1", (x + 40, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw Player 2 (Red) with X
        if self.player2_points is not None:
            center = self._get_center_from_points(self.player2_points)
            if center:
                x, y = center
                size = 25
                thickness = 3
                
                # Draw X
                cv2.line(result, (x - size, y - size), (x + size, y + size), (0, 0, 255), thickness)
                cv2.line(result, (x - size, y + size), (x + size, y - size), (0, 0, 255), thickness)
                
                # Draw circle around X
                cv2.circle(result, (x, y), 35, (0, 0, 255), 2)
                
                # Draw tracked points
                for point in self.player2_points:
                    px, py = point.ravel()
                    cv2.circle(result, (int(px), int(py)), 3, (100, 100, 255), -1)
                
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
        """Process video and track players using optical flow."""
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
        
        print("Initializing optical flow tracker...")
        print("Press 'Q' key to stop processing and exit")
        
        self.frame_count = 0
        
        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read first frame")
                return
            
            # Convert to grayscale
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect initial features
            self.player1_points = self._detect_features_in_region(self.prev_gray, self.player1_initial)
            self.player2_points = self._detect_features_in_region(self.prev_gray, self.player2_initial)
            
            print(f"Player 1 (Blue) initialized with {len(self.player1_points)} features")
            print(f"Player 2 (Red) initialized with {len(self.player2_points)} features")
            print("Tracking players...")
            
            # Process first frame
            frame_with_lines = self._draw_court_lines(frame)
            frame_with_markers = self._draw_player_markers(frame_with_lines)
            out.write(frame_with_markers)
            self.frame_count += 1
            
            # Process remaining frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Track features using optical flow
                if self.player1_points is not None and len(self.player1_points) > 0:
                    new_points1, status1, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.player1_points, None, **self.lk_params
                    )
                    
                    # Keep only good points
                    if new_points1 is not None:
                        good_points1 = new_points1[status1.flatten() == 1]
                        if len(good_points1) > 0:
                            self.player1_points = good_points1.reshape(-1, 1, 2)
                        else:
                            # Re-detect features
                            center = self._get_center_from_points(self.player1_points)
                            if center:
                                self.player1_points = self._detect_features_in_region(gray, center)
                
                if self.player2_points is not None and len(self.player2_points) > 0:
                    new_points2, status2, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.player2_points, None, **self.lk_params
                    )
                    
                    # Keep only good points
                    if new_points2 is not None:
                        good_points2 = new_points2[status2.flatten() == 1]
                        if len(good_points2) > 0:
                            self.player2_points = good_points2.reshape(-1, 1, 2)
                        else:
                            # Re-detect features
                            center = self._get_center_from_points(self.player2_points)
                            if center:
                                self.player2_points = self._detect_features_in_region(gray, center)
                
                # Periodically re-detect features to maintain tracking
                if self.frame_count % self.redetect_interval == 0:
                    center1 = self._get_center_from_points(self.player1_points)
                    center2 = self._get_center_from_points(self.player2_points)
                    
                    if center1:
                        self.player1_points = self._detect_features_in_region(gray, center1)
                    if center2:
                        self.player2_points = self._detect_features_in_region(gray, center2)
                
                # Update previous frame
                self.prev_gray = gray.copy()
                
                # Draw court lines
                frame_with_lines = self._draw_court_lines(frame)
                
                # Draw player markers
                frame_with_markers = self._draw_player_markers(frame_with_lines)
                
                # Write to output
                out.write(frame_with_markers)
                
                # Display progress
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
                
                # Display frame
                display_frame, _ = self._resize_frame_to_fit_screen(frame_with_markers)
                cv2.imshow('Optical Flow Player Tracking - Press Q to quit', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user (Q key pressed)")
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Processed {self.frame_count} frames")
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
    
    print("Tennis Player Tracking with Optical Flow")
    print("=" * 50)
    print(f"Initial Player 1 (Blue) position: {player1_pos}")
    print(f"Initial Player 2 (Red) position: {player2_pos}")
    print()
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_with_optical_flow_tracking.mp4"
    
    tracker = OpticalFlowPlayerTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()











