"""
Track players and ball with manual ball initialization.
User can pause video and click on ball to start tracking.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ManualBallTracker:
    def __init__(self, player1_pos: Tuple[int, int], player2_pos: Tuple[int, int]):
        self.player1_initial = player1_pos
        self.player2_initial = player2_pos
        
        # Player tracking
        self.player1_points = None
        self.player2_points = None
        
        # Ball tracking
        self.ball_pos = None
        self.ball_tracking = False
        self.ball_prev_pos = None
        self.ball_stationary_count = 0
        self.ball_lost_count = 0
        
        # Video state
        self.paused = False
        self.current_frame = None
        self.current_gray = None
        
        # Parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
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
        self.display_scale = 1.0
        
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
    
    def _track_ball(self, gray: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track ball using optical flow."""
        if self.ball_pos is None or self.prev_gray is None:
            return None
        
        # Create points around ball for better tracking
        x, y = self.ball_pos
        ball_points = np.array([
            [[float(x), float(y)]],
            [[float(x-5), float(y-5)]],
            [[float(x+5), float(y-5)]],
            [[float(x-5), float(y+5)]],
            [[float(x+5), float(y+5)]]
        ], dtype=np.float32)
        
        # Track
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, ball_points, None, **self.lk_params
        )
        
        if new_points is not None:
            good_points = new_points[status.flatten() == 1]
            if len(good_points) > 0:
                # Use center of good points
                if good_points.ndim == 3:
                    new_x = int(np.mean(good_points[:, 0, 0]))
                    new_y = int(np.mean(good_points[:, 0, 1]))
                else:
                    new_x = int(np.mean(good_points[:, 0]))
                    new_y = int(np.mean(good_points[:, 1]))
                return (new_x, new_y)
        
        return None
    
    def _is_ball_moving(self, current_pos: Tuple[int, int]) -> bool:
        """Check if ball is moving."""
        if self.ball_prev_pos is None:
            return True
        
        dx = current_pos[0] - self.ball_prev_pos[0]
        dy = current_pos[1] - self.ball_prev_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance > 2  # Movement threshold
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark ball."""
        if not self.paused:
            return
        
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark ball position
            self.ball_pos = (int(x / scale), int(y / scale))
            self.ball_tracking = True
            self.ball_stationary_count = 0
            self.ball_lost_count = 0
            self.ball_prev_pos = None
            print(f"Ball marked at: {self.ball_pos} - Starting tracking")
    
    def _draw_markers(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Draw all markers."""
        result = frame.copy()
        
        # Draw Player 1 (Blue)
        if self.player1_points is not None:
            center = self._get_center_from_points(self.player1_points)
            if center:
                x, y = center
                display_x = int(x * scale)
                display_y = int(y * scale)
                size = int(25 * scale)
                # X
                cv2.line(result, (display_x - size, display_y - size), 
                        (display_x + size, display_y + size), (255, 0, 0), 3)
                cv2.line(result, (display_x - size, display_y + size), 
                        (display_x + size, display_y - size), (255, 0, 0), 3)
                # Circle
                cv2.circle(result, (display_x, display_y), int(35 * scale), (255, 0, 0), 2)
                cv2.putText(result, "P1", (display_x + int(40 * scale), display_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 0, 0), 2)
        
        # Draw Player 2 (Red)
        if self.player2_points is not None:
            center = self._get_center_from_points(self.player2_points)
            if center:
                x, y = center
                display_x = int(x * scale)
                display_y = int(y * scale)
                size = int(25 * scale)
                # X
                cv2.line(result, (display_x - size, display_y - size), 
                        (display_x + size, display_y + size), (0, 0, 255), 3)
                cv2.line(result, (display_x - size, display_y + size), 
                        (display_x + size, display_y - size), (0, 0, 255), 3)
                # Circle
                cv2.circle(result, (display_x, display_y), int(35 * scale), (0, 0, 255), 2)
                cv2.putText(result, "P2", (display_x + int(40 * scale), display_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 255), 2)
        
        # Draw Ball (Green)
        if self.ball_pos is not None and self.ball_tracking:
            x, y = self.ball_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            size = int(20 * scale)
            # X
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 255, 0), 3)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 255, 0), 3)
            # Circle
            cv2.circle(result, (display_x, display_y), int(30 * scale), (0, 255, 0), 2)
            cv2.putText(result, "Ball", (display_x + int(35 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), 2)
        
        # Draw status
        if self.paused:
            status_text = "PAUSED - Click ball | Arrows: navigate"
        else:
            status_text = "PLAYING - SPACE: pause"
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw frame number
        cv2.putText(result, f"Frame: {self.frame_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw tracking status
        if self.ball_tracking:
            cv2.putText(result, "Ball Tracking: ON", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw controls when paused
        if self.paused:
            controls = [
                "LEFT/RIGHT: 1 frame",
                "UP/DOWN: 30 frames"
            ]
            y_offset = result.shape[0] - 60
            for control in controls:
                cv2.putText(result, control, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                y_offset += 30
        
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
        """Process video with manual ball marking."""
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
        
        # Calculate display scale
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
        _, self.display_scale = self._resize_frame_to_fit_screen(test_frame)
        
        print("\nPlayer and Ball Tracking")
        print("=" * 50)
        print("Instructions:")
        print("- Video will start playing")
        print("- Press SPACE to pause/play")
        print("- When paused, click on the ball to start tracking")
        print("- Arrow RIGHT: Skip forward 1 frame")
        print("- Arrow LEFT: Go back 1 frame")
        print("- Arrow UP: Skip forward 30 frames (~1 sec)")
        print("- Arrow DOWN: Go back 30 frames (~1 sec)")
        print("- Press 'Q' to quit")
        print()
        
        window_name = "Track - SPACE: pause, Click: mark ball, Q: quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback, self.display_scale)
        
        self.frame_count = 0
        
        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read first frame")
                return
            
            self.current_frame = frame
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Initialize player tracking
            self.player1_points = self._detect_features_in_region(self.prev_gray, self.player1_initial)
            self.player2_points = self._detect_features_in_region(self.prev_gray, self.player2_initial)
            
            print("Players initialized - video ready")
            
            while True:
                if not self.paused:
                    # Read next frame
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video")
                        break
                    
                    self.current_frame = frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.current_gray = gray
                    
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
                    
                    # Track ball if tracking is active
                    if self.ball_tracking:
                        new_ball_pos = self._track_ball(gray)
                        
                        if new_ball_pos:
                            # Check if ball is moving
                            if self._is_ball_moving(new_ball_pos):
                                self.ball_pos = new_ball_pos
                                self.ball_stationary_count = 0
                                self.ball_lost_count = 0
                            else:
                                self.ball_stationary_count += 1
                                if self.ball_stationary_count > 20:  # Ball stopped
                                    print(f"Ball stopped at frame {self.frame_count}")
                                    self.ball_tracking = False
                                    self.ball_pos = None
                                    self.ball_stationary_count = 0
                        else:
                            self.ball_lost_count += 1
                            if self.ball_lost_count > 15:  # Lost tracking
                                print(f"Ball lost at frame {self.frame_count}")
                                self.ball_tracking = False
                                self.ball_pos = None
                                self.ball_lost_count = 0
                        
                        self.ball_prev_pos = self.ball_pos
                    
                    self.prev_gray = gray.copy()
                    self.frame_count += 1
                    
                    if self.frame_count % 30 == 0:
                        progress = (self.frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
                
                # Draw everything
                frame_with_lines = self._draw_court_lines(self.current_frame)
                display_frame = self._draw_markers(frame_with_lines, 1.0)
                
                # Write to output
                out.write(display_frame)
                
                # Display (resized)
                display_resized, _ = self._resize_frame_to_fit_screen(display_frame)
                cv2.imshow(window_name, display_resized)
                
                # Handle keyboard
                key = cv2.waitKey(30 if not self.paused else 1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("Quitting...")
                    break
                elif key == ord(' '):  # Space
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "PLAYING"
                    print(f"{status} at frame {self.frame_count}")
                elif key == 83:  # Right arrow (when paused)
                    if self.paused:
                        # Skip forward 1 frame
                        new_frame_num = min(self.frame_count + 1, total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.frame_count = new_frame_num
                            print(f"Forward to frame {self.frame_count}")
                elif key == 81:  # Left arrow (when paused)
                    if self.paused:
                        # Go back 1 frame
                        new_frame_num = max(self.frame_count - 1, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.frame_count = new_frame_num
                            print(f"Back to frame {self.frame_count}")
                elif key == 82:  # Up arrow (when paused)
                    if self.paused:
                        # Skip forward 30 frames
                        new_frame_num = min(self.frame_count + 30, total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.frame_count = new_frame_num
                            print(f"Skip forward to frame {self.frame_count}")
                elif key == 84:  # Down arrow (when paused)
                    if self.paused:
                        # Go back 30 frames
                        new_frame_num = max(self.frame_count - 30, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.frame_count = new_frame_num
                            print(f"Skip back to frame {self.frame_count}")
        
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
    except FileNotFoundError:
        print("Error: tracking_config.txt not found.")
        return
    
    print(f"Player 1 (Blue): {player1_pos}")
    print(f"Player 2 (Red): {player2_pos}")
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_manual_ball_tracking.mp4"
    
    tracker = ManualBallTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()

