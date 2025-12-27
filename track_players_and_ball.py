"""
Track two tennis players and the ball.
- Players tracked with optical flow (Blue X + circle, Red X + circle)
- Ball tracked only during play (Green X + circle)
- Ball detection starts in serve area and tracks until ball stops
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class PlayerAndBallTracker:
    def __init__(self, player1_pos: Tuple[int, int], player2_pos: Tuple[int, int],
                 serve_area: Tuple[int, int, int, int]):
        self.player1_initial = player1_pos  # Blue
        self.player2_initial = player2_pos  # Red
        self.serve_area = serve_area  # (x1, y1, x2, y2)
        
        # Player tracking
        self.player1_points = None
        self.player2_points = None
        
        # Ball tracking
        self.ball_pos = None
        self.ball_tracking = False
        self.ball_prev_pos = None
        self.ball_stationary_count = 0
        self.ball_lost_count = 0
        
        # Parameters
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.feature_params = dict(
            maxCorners=15,
            qualityLevel=0.1,
            minDistance=5,
            blockSize=7
        )
        
        self.prev_gray = None
        self.region_size = 60
        self.frame_count = 0
        self.redetect_interval = 30
        
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
    
    def _resize_frame_to_fit_screen(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize frame to fit screen."""
        height, width = frame.shape[:2]
        scale_w = self.screen_width / width
        scale_h = self.screen_height / height
        scale = min(scale_w, scale_h, 1.0)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def _detect_features_in_region(self, gray: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Detect features in region."""
        height, width = gray.shape
        x, y = center
        x1 = max(0, x - self.region_size)
        y1 = max(0, y - self.region_size)
        x2 = min(width, x + self.region_size)
        y2 = min(height, y + self.region_size)
        
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        
        features = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if features is None:
            return np.array([[[float(x), float(y)]]], dtype=np.float32)
        return features
    
    def _get_center_from_points(self, points: np.ndarray) -> Optional[Tuple[int, int]]:
        """Calculate center from points."""
        if points is None or len(points) == 0:
            return None
        mean_x = np.mean(points[:, 0, 0])
        mean_y = np.mean(points[:, 0, 1])
        return (int(mean_x), int(mean_y))
    
    def _detect_ball_in_serve_area(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect yellow/green tennis ball in serve area using color."""
        x1, y1, x2, y2 = self.serve_area
        
        # Extract serve area
        serve_region = frame[y1:y2, x1:x2]
        if serve_region.size == 0:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(serve_region, cv2.COLOR_BGR2HSV)
        
        # Yellow-green color range for tennis ball
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Create mask
        ball_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find circular contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 500:  # Ball size filter
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.5:  # Reasonably circular
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                return (cx, cy)
        
        return None
    
    def _track_ball(self, gray: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track ball using optical flow if already tracking."""
        if self.ball_pos is None:
            return None
        
        # Create points around ball
        x, y = self.ball_pos
        ball_points = np.array([[[float(x), float(y)]]], dtype=np.float32)
        
        # Track
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, ball_points, None, **self.lk_params
        )
        
        if new_points is not None and status[0][0] == 1:
            new_x, new_y = new_points[0][0]
            return (int(new_x), int(new_y))
        
        return None
    
    def _is_ball_moving(self, current_pos: Tuple[int, int]) -> bool:
        """Check if ball is moving."""
        if self.ball_prev_pos is None:
            return True
        
        dx = current_pos[0] - self.ball_prev_pos[0]
        dy = current_pos[1] - self.ball_prev_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance > 3  # Movement threshold
    
    def _draw_markers(self, frame: np.ndarray) -> np.ndarray:
        """Draw all markers."""
        result = frame.copy()
        
        # Draw Player 1 (Blue)
        if self.player1_points is not None:
            center = self._get_center_from_points(self.player1_points)
            if center:
                x, y = center
                size = 25
                # X
                cv2.line(result, (x - size, y - size), (x + size, y + size), (255, 0, 0), 3)
                cv2.line(result, (x - size, y + size), (x + size, y - size), (255, 0, 0), 3)
                # Circle
                cv2.circle(result, (x, y), 35, (255, 0, 0), 2)
                cv2.putText(result, "P1", (x + 40, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw Player 2 (Red)
        if self.player2_points is not None:
            center = self._get_center_from_points(self.player2_points)
            if center:
                x, y = center
                size = 25
                # X
                cv2.line(result, (x - size, y - size), (x + size, y + size), (0, 0, 255), 3)
                cv2.line(result, (x - size, y + size), (x + size, y - size), (0, 0, 255), 3)
                # Circle
                cv2.circle(result, (x, y), 35, (0, 0, 255), 2)
                cv2.putText(result, "P2", (x + 40, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw Ball (Green)
        if self.ball_pos is not None and self.ball_tracking:
            x, y = self.ball_pos
            size = 20
            # X
            cv2.line(result, (x - size, y - size), (x + size, y + size), (0, 255, 0), 3)
            cv2.line(result, (x - size, y + size), (x + size, y - size), (0, 255, 0), 3)
            # Circle
            cv2.circle(result, (x, y), 30, (0, 255, 0), 2)
            cv2.putText(result, "Ball", (x + 35, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw serve area (semi-transparent)
        if not self.ball_tracking:
            x1, y1, x2, y2 = self.serve_area
            overlay = result.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result, "Looking for ball...", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result
    
    def _draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw court lines."""
        result = frame.copy()
        yellow = (0, 255, 255)
        thickness = 2
        
        lines = [
            (1, 5), (5, 7), (7, 3),
            (2, 6), (6, 8), (8, 4),
            (1, 13), (13, 2),
            (3, 14), (14, 4),
            (5, 6), (7, 8),
            (9, 10),
            (11, 12),
        ]
        
        for p1, p2 in lines:
            pt1 = self.court_points[p1]
            pt2 = self.court_points[p2]
            cv2.line(result, pt1, pt2, yellow, thickness)
        
        return result
    
    def process_video(self, input_path: str, output_path: str):
        """Process video."""
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
        
        print("Tracking players and ball...")
        print("Press 'Q' key to stop")
        
        self.frame_count = 0
        
        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read first frame")
                return
            
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Initialize player tracking
            self.player1_points = self._detect_features_in_region(self.prev_gray, self.player1_initial)
            self.player2_points = self._detect_features_in_region(self.prev_gray, self.player2_initial)
            
            print(f"Players initialized")
            
            # Process first frame
            frame_with_lines = self._draw_court_lines(frame)
            frame_with_markers = self._draw_markers(frame_with_lines)
            out.write(frame_with_markers)
            self.frame_count += 1
            
            # Process remaining frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Track players
                if self.player1_points is not None and len(self.player1_points) > 0:
                    new_points1, status1, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.player1_points, None, **self.lk_params
                    )
                    if new_points1 is not None:
                        good_points1 = new_points1[status1.flatten() == 1]
                        if len(good_points1) > 0:
                            self.player1_points = good_points1.reshape(-1, 1, 2)
                
                if self.player2_points is not None and len(self.player2_points) > 0:
                    new_points2, status2, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.player2_points, None, **self.lk_params
                    )
                    if new_points2 is not None:
                        good_points2 = new_points2[status2.flatten() == 1]
                        if len(good_points2) > 0:
                            self.player2_points = good_points2.reshape(-1, 1, 2)
                
                # Re-detect player features periodically
                if self.frame_count % self.redetect_interval == 0:
                    center1 = self._get_center_from_points(self.player1_points)
                    center2 = self._get_center_from_points(self.player2_points)
                    if center1:
                        self.player1_points = self._detect_features_in_region(gray, center1)
                    if center2:
                        self.player2_points = self._detect_features_in_region(gray, center2)
                
                # Ball tracking logic
                if not self.ball_tracking:
                    # Look for ball in serve area
                    ball_detected = self._detect_ball_in_serve_area(frame)
                    if ball_detected:
                        self.ball_pos = ball_detected
                        self.ball_tracking = True
                        self.ball_stationary_count = 0
                        self.ball_lost_count = 0
                        print(f"Ball detected at frame {self.frame_count}: {self.ball_pos}")
                else:
                    # Track existing ball
                    new_ball_pos = self._track_ball(gray)
                    
                    if new_ball_pos:
                        # Check if ball is moving
                        if self._is_ball_moving(new_ball_pos):
                            self.ball_pos = new_ball_pos
                            self.ball_stationary_count = 0
                            self.ball_lost_count = 0
                        else:
                            self.ball_stationary_count += 1
                            if self.ball_stationary_count > 15:  # Ball stopped for ~0.6 sec
                                print(f"Ball stopped at frame {self.frame_count}")
                                self.ball_tracking = False
                                self.ball_pos = None
                                self.ball_stationary_count = 0
                    else:
                        self.ball_lost_count += 1
                        if self.ball_lost_count > 10:  # Lost tracking
                            print(f"Ball lost at frame {self.frame_count}")
                            self.ball_tracking = False
                            self.ball_pos = None
                            self.ball_lost_count = 0
                    
                    self.ball_prev_pos = self.ball_pos
                
                self.prev_gray = gray.copy()
                
                # Draw everything
                frame_with_lines = self._draw_court_lines(frame)
                frame_with_markers = self._draw_markers(frame_with_lines)
                out.write(frame_with_markers)
                
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
                
                # Display
                display_frame, _ = self._resize_frame_to_fit_screen(frame_with_markers)
                cv2.imshow('Player & Ball Tracking - Press Q to quit', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Complete! Processed {self.frame_count} frames")
        print(f"Output: {output_path}")


def main():
    # Load configuration
    try:
        with open('tracking_config.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
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
                elif "Serve Area:" in line:
                    area_str = line.split(":")[1].strip()
                    area_str = area_str.replace("(", "").replace(")", "")
                    x1, y1, x2, y2 = map(int, area_str.split(","))
                    serve_area = (x1, y1, x2, y2)
    except FileNotFoundError:
        print("Error: tracking_config.txt not found. Run mark_players_and_serve_area.py first.")
        return
    
    print("Player and Ball Tracking")
    print("=" * 50)
    print(f"Player 1 (Blue): {player1_pos}")
    print(f"Player 2 (Red): {player2_pos}")
    print(f"Serve Area: {serve_area}")
    print()
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_full_tracking.mp4"
    
    tracker = PlayerAndBallTracker(player1_pos, player2_pos, serve_area)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()











