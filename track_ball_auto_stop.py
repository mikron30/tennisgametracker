"""
Track players and ball with automatic pause when ball stops.
User marks ball position, system tracks it and auto-pauses when ball stops.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class AutoStopBallTracker:
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
        self.ball_template = None
        self.ball_search_region = 100
        
        # Video state
        self.paused = True  # Start paused
        self.current_frame = None
        self.current_gray = None
        self.ball_marked_pos = None  # For visual feedback
        
        # Parameters
        self.lk_params = dict(
            winSize=(21, 21),
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
    
    def _extract_ball_template(self, gray: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """Extract ball template for tracking."""
        x, y = center
        template_size = 30
        x1 = max(0, x - template_size)
        y1 = max(0, y - template_size)
        x2 = min(gray.shape[1], x + template_size)
        y2 = min(gray.shape[0], y + template_size)
        
        template = gray[y1:y2, x1:x2]
        return template
    
    def _track_ball_template_matching(self, gray: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track ball using template matching in search region."""
        if self.ball_template is None or self.ball_pos is None:
            return None
        
        x, y = self.ball_pos
        height, width = gray.shape
        
        # Define search region
        x1 = max(0, x - self.ball_search_region)
        y1 = max(0, y - self.ball_search_region)
        x2 = min(width, x + self.ball_search_region)
        y2 = min(height, y + self.ball_search_region)
        
        search_region = gray[y1:y2, x1:x2]
        
        if search_region.size == 0 or self.ball_template.size == 0:
            return None
        
        # Perform template matching
        try:
            result = cv2.matchTemplate(search_region, self.ball_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.5:  # Confidence threshold
                # Calculate center of match
                template_h, template_w = self.ball_template.shape
                match_x = x1 + max_loc[0] + template_w // 2
                match_y = y1 + max_loc[1] + template_h // 2
                return (match_x, match_y)
        except:
            pass
        
        return None
    
    def _track_ball_optical_flow(self, gray: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track ball using optical flow."""
        if self.ball_pos is None or self.prev_gray is None:
            return None
        
        x, y = self.ball_pos
        ball_points = np.array([
            [[float(x), float(y)]],
            [[float(x-5), float(y)]],
            [[float(x+5), float(y)]],
            [[float(x), float(y-5)]],
            [[float(x), float(y+5)]]
        ], dtype=np.float32)
        
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, ball_points, None, **self.lk_params
        )
        
        if new_points is not None:
            good_points = new_points[status.flatten() == 1]
            if len(good_points) > 0:
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
        
        return distance > 3  # Movement threshold
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark ball."""
        if not self.paused:
            return
        
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark ball position
            self.ball_marked_pos = (int(x / scale), int(y / scale))
            print(f"Ball marked at: {self.ball_marked_pos} - Press 'S' to save and start tracking")
    
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
                cv2.line(result, (display_x - size, display_y - size), 
                        (display_x + size, display_y + size), (255, 0, 0), 3)
                cv2.line(result, (display_x - size, display_y + size), 
                        (display_x + size, display_y - size), (255, 0, 0), 3)
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
                cv2.line(result, (display_x - size, display_y - size), 
                        (display_x + size, display_y + size), (0, 0, 255), 3)
                cv2.line(result, (display_x - size, display_y + size), 
                        (display_x + size, display_y - size), (0, 0, 255), 3)
                cv2.circle(result, (display_x, display_y), int(35 * scale), (0, 0, 255), 2)
                cv2.putText(result, "P2", (display_x + int(40 * scale), display_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 255), 2)
        
        # Draw Ball (Green) - tracking position
        if self.ball_pos is not None and self.ball_tracking:
            x, y = self.ball_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            size = int(20 * scale)
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 255, 0), 3)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 255, 0), 3)
            cv2.circle(result, (display_x, display_y), int(30 * scale), (0, 255, 0), 2)
            cv2.putText(result, "Ball", (display_x + int(35 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), 2)
        
        # Draw marked position (before saving)
        if self.ball_marked_pos is not None and not self.ball_tracking:
            x, y = self.ball_marked_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            size = int(20 * scale)
            # Draw semi-transparent green X
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 200, 0), 2)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 200, 0), 2)
            cv2.circle(result, (display_x, display_y), int(30 * scale), (0, 200, 0), 2)
            cv2.putText(result, "Press S to track", (display_x + int(35 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 200, 0), 2)
        
        # Draw status
        if self.ball_tracking:
            status_text = "TRACKING - Press D to advance frame | Click ball if lost"
        else:
            status_text = "PAUSED - Click ball, press S to save"
        
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(result, f"Frame: {self.frame_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.ball_tracking:
            cv2.putText(result, "Ball Tracking: ON", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw controls when paused
        if self.paused:
            controls = [
                "D/A: 1 frame | W/X: 30 frames",
                "Click ball + S: Start tracking",
                "SPACE: Continue | Q: Quit"
            ]
            y_offset = result.shape[0] - 90
            for control in controls:
                cv2.putText(result, control, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
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
        """Process video with auto-pause on ball stop."""
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
        
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
        _, self.display_scale = self._resize_frame_to_fit_screen(test_frame)
        
        print("\nManual Frame-by-Frame Ball Tracking")
        print("=" * 50)
        print("1. Video starts PAUSED")
        print("2. Use D/A/W/X to navigate:")
        print("   D = 1 frame forward")
        print("   A = 1 frame back")
        print("   W = 30 frames forward (~1 sec)")
        print("   X = 30 frames back (~1 sec)")
        print("3. Click on ball")
        print("4. Press 'S' to save ball position")
        print("5. Press 'D' to move forward - ball will track automatically")
        print("6. If tracking fails, click on ball again and press 'S'")
        print("7. Continue with 'D' for each frame")
        print("8. Press 'Q' to quit and save video")
        print()
        
        window_name = "Auto-Stop Tracking - Click ball + S to track, Arrows: navigate"
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
            
            print("Players initialized - Ready to mark ball")
            
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
                        # Try optical flow first
                        new_ball_pos = self._track_ball_optical_flow(gray)
                        
                        # If optical flow fails, try template matching
                        if new_ball_pos is None:
                            new_ball_pos = self._track_ball_template_matching(gray)
                        
                        if new_ball_pos:
                            # Check if ball is moving
                            if self._is_ball_moving(new_ball_pos):
                                self.ball_pos = new_ball_pos
                                self.ball_stationary_count = 0
                                self.ball_lost_count = 0
                            else:
                                self.ball_stationary_count += 1
                                if self.ball_stationary_count > 15:  # Ball stopped
                                    print(f"\n*** Ball stopped at frame {self.frame_count} - AUTO-PAUSED ***")
                                    print("Mark new ball position and press S to continue\n")
                                    self.ball_tracking = False
                                    self.ball_marked_pos = None
                                    self.ball_stationary_count = 0
                                    self.paused = True  # AUTO-PAUSE
                        else:
                            self.ball_lost_count += 1
                            if self.ball_lost_count > 10:  # Lost tracking
                                print(f"\n*** Ball lost at frame {self.frame_count} - AUTO-PAUSED ***")
                                print("Mark new ball position and press S to continue\n")
                                self.ball_tracking = False
                                self.ball_marked_pos = None
                                self.ball_lost_count = 0
                                self.paused = True  # AUTO-PAUSE
                        
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
                    if not self.ball_tracking:
                        print("Mark ball and press S first!")
                    else:
                        self.paused = not self.paused
                        status = "PAUSED" if self.paused else "TRACKING"
                        print(f"{status} at frame {self.frame_count}")
                elif key == ord('s') or key == ord('S'):
                    if self.ball_marked_pos is not None:
                        # Save ball position and enable tracking (but stay paused)
                        self.ball_pos = self.ball_marked_pos
                        self.ball_tracking = True
                        self.ball_stationary_count = 0
                        self.ball_lost_count = 0
                        self.ball_prev_pos = None
                        self.ball_marked_pos = None  # Clear marked position so user can mark again
                        
                        # Extract template
                        if self.current_gray is not None:
                            self.ball_template = self._extract_ball_template(self.current_gray, self.ball_pos)
                        
                        # Stay paused - user will advance frame by frame
                        print(f"Ball position saved at frame {self.frame_count}: {self.ball_pos}")
                        print("Press 'D' to advance and track ball frame by frame")
                elif key == ord('d') or key == ord('D'):  # Forward 1 frame
                    if self.paused:
                        new_frame_num = min(self.frame_count + 1, total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            # If ball tracking is active, try to track ball in new frame
                            if self.ball_tracking and self.ball_pos is not None:
                                # Try optical flow
                                new_ball_pos = self._track_ball_optical_flow(new_gray)
                                
                                # If optical flow fails, try template matching
                                if new_ball_pos is None:
                                    new_ball_pos = self._track_ball_template_matching(new_gray)
                                
                                if new_ball_pos:
                                    self.ball_pos = new_ball_pos
                                    print(f"Frame {new_frame_num} - Ball tracked to: {self.ball_pos}")
                                else:
                                    print(f"Frame {new_frame_num} - Ball tracking LOST! Click on ball and press S")
                                    self.ball_tracking = False
                                    self.ball_marked_pos = None  # Clear so user can mark new position
                            else:
                                print(f"Frame {new_frame_num}")
                            
                            self.current_gray = new_gray
                            self.prev_gray = new_gray.copy()
                            self.frame_count = new_frame_num
                elif key == ord('a') or key == ord('A'):  # Back 1 frame
                    if self.paused:
                        new_frame_num = max(self.frame_count - 1, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.prev_gray = self.current_gray.copy()
                            self.frame_count = new_frame_num
                            print(f"Frame {self.frame_count}")
                elif key == ord('w') or key == ord('W'):  # Forward 30 frames
                    if self.paused:
                        new_frame_num = min(self.frame_count + 30, total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.prev_gray = self.current_gray.copy()
                            self.frame_count = new_frame_num
                            print(f"Frame {self.frame_count}")
                elif key == ord('x') or key == ord('X'):  # Back 30 frames
                    if self.paused:
                        new_frame_num = max(self.frame_count - 30, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                        ret, frame = cap.read()
                        if ret:
                            self.current_frame = frame
                            self.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            self.prev_gray = self.current_gray.copy()
                            self.frame_count = new_frame_num
                            print(f"Frame {self.frame_count}")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"\nComplete! Processed {self.frame_count} frames")
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
    output_path = "tennis_auto_stop_tracking.mp4"
    
    tracker = AutoStopBallTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()

