"""
Robust ball tracking using color filtering and template matching.
Filters for yellow-green tennis ball colors and tracks frame by frame.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ColorBasedBallTracker:
    def __init__(self, player1_pos: Tuple[int, int], player2_pos: Tuple[int, int]):
        self.player1_initial = player1_pos
        self.player2_initial = player2_pos
        
        # Player tracking
        self.player1_points = None
        self.player2_points = None
        
        # Ball tracking
        self.ball_pos = None
        self.ball_tracking = False
        self.ball_search_radius = 80  # Search area around last position
        self.ball_template = None
        self.ball_template_size = 40  # Size of template area
        
        # Video state
        self.paused = True
        self.current_frame = None
        self.current_gray = None
        self.ball_marked_pos = None
        
        # Parameters for player tracking
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
    
    def _extract_ball_region(self, frame: np.ndarray, center: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract region around ball for color filtering."""
        x, y = center
        size = self.ball_template_size
        
        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(frame.shape[1], x + size)
        y2 = min(frame.shape[0], y + size)
        
        region = frame[y1:y2, x1:x2]
        return region, (x1, y1, x2, y2)
    
    def _filter_ball_color(self, frame: np.ndarray) -> np.ndarray:
        """
        Filter tennis ball by color - return binary mask where ball is white, rest is black.
        Tennis balls are typically yellow-green or bright yellow.
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Wider yellow-green color range for tennis ball
        # Range 1: Yellow-green (primary range)
        lower_yellow1 = np.array([15, 50, 50])
        upper_yellow1 = np.array([45, 255, 255])
        mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
        
        # Range 2: Bright yellow (secondary range)
        lower_yellow2 = np.array([20, 100, 150])
        upper_yellow2 = np.array([35, 255, 255])
        mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
        
        # Range 3: Light colors (for bright reflections on ball)
        lower_light = np.array([0, 0, 200])
        upper_light = np.array([180, 50, 255])
        mask3 = cv2.inRange(hsv, lower_light, upper_light)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _find_ball_in_region(self, frame: np.ndarray, search_center: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Find ball in search region using color filtering.
        Returns ball center position or None if not found.
        """
        x, y = search_center
        height, width = frame.shape[:2]
        
        # Define search region
        x1 = max(0, x - self.ball_search_radius)
        y1 = max(0, y - self.ball_search_radius)
        x2 = min(width, x + self.ball_search_radius)
        y2 = min(height, y + self.ball_search_radius)
        
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            return None
        
        # Filter for ball color
        ball_mask = self._filter_ball_color(search_region)
        
        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find best contour (closest to center and reasonably sized)
        best_contour = None
        best_distance = float('inf')
        region_center = (search_region.shape[1] // 2, search_region.shape[0] // 2)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (adjust based on ball size in video)
            if area < 20 or area > 1000:
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate distance to search center
            dx = cx - region_center[0]
            dy = cy - region_center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < best_distance:
                best_distance = distance
                best_contour = contour
                best_cx = cx + x1
                best_cy = cy + y1
        
        if best_contour is not None:
            return (best_cx, best_cy)
        
        return None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark ball."""
        if not self.paused:
            return
        
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Mark ball position
            self.ball_marked_pos = (int(x / scale), int(y / scale))
            print(f"Ball marked at: {self.ball_marked_pos} - Press 'S' to save")
    
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
            
            # Draw search area
            search_radius = int(self.ball_search_radius * scale)
            cv2.circle(result, (display_x, display_y), search_radius, (0, 255, 0), 1)
        
        # Draw marked position (before saving)
        if self.ball_marked_pos is not None and not self.ball_tracking:
            x, y = self.ball_marked_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            size = int(20 * scale)
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 200, 0), 2)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 200, 0), 2)
            cv2.circle(result, (display_x, display_y), int(30 * scale), (0, 200, 0), 2)
            cv2.putText(result, "Press S", (display_x + int(35 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 200, 0), 2)
        
        # Draw status
        if self.ball_tracking:
            status_text = "TRACKING - Press D | Click ball to reposition"
        else:
            status_text = "PAUSED - Click ball + press S"
        
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(result, f"Frame: {self.frame_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.ball_tracking:
            cv2.putText(result, "Ball Tracking: ON", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw controls
        controls = [
            "D/A: 1 frame | W/X: 30 frames",
            "Click + S: Mark ball | Q: Quit"
        ]
        y_offset = result.shape[0] - 60
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
        """Process video with color-based ball tracking."""
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
        
        print("\nColor-Based Ball Tracking")
        print("=" * 50)
        print("1. Navigate with D/A/W/X")
        print("2. Click on ball")
        print("3. Press 'S' - system filters for ball color")
        print("4. Press 'D' - ball tracks automatically using color filter")
        print("5. Click to reposition if needed")
        print()
        
        window_name = "Color Tracking - Click ball + S, then D to advance"
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
                # Draw everything
                frame_with_lines = self._draw_court_lines(self.current_frame)
                display_frame = self._draw_markers(frame_with_lines, 1.0)
                
                # Write to output
                out.write(display_frame)
                
                # Display (resized)
                display_resized, _ = self._resize_frame_to_fit_screen(display_frame)
                cv2.imshow(window_name, display_resized)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("Quitting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    if self.ball_marked_pos is not None:
                        self.ball_pos = self.ball_marked_pos
                        self.ball_tracking = True
                        self.ball_marked_pos = None
                        print(f"Ball saved at frame {self.frame_count}: {self.ball_pos}")
                        print("Press 'D' to advance - ball will track using color filter")
                elif key == ord('d') or key == ord('D'):
                    # Move forward 1 frame
                    new_frame_num = min(self.frame_count + 1, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                    ret, frame = cap.read()
                    if ret:
                        self.current_frame = frame
                        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Track players
                        if self.player1_points is not None and len(self.player1_points) > 0:
                            new_points1, status1, _ = cv2.calcOpticalFlowPyrLK(
                                self.prev_gray, new_gray, self.player1_points, None, **self.lk_params
                            )
                            if new_points1 is not None:
                                good_points1 = new_points1[status1.flatten() == 1]
                                if len(good_points1) > 0:
                                    self.player1_points = good_points1.reshape(-1, 1, 2)
                        
                        if self.player2_points is not None and len(self.player2_points) > 0:
                            new_points2, status2, _ = cv2.calcOpticalFlowPyrLK(
                                self.prev_gray, new_gray, self.player2_points, None, **self.lk_params
                            )
                            if new_points2 is not None:
                                good_points2 = new_points2[status2.flatten() == 1]
                                if len(good_points2) > 0:
                                    self.player2_points = good_points2.reshape(-1, 1, 2)
                        
                        # Re-detect player features periodically
                        if new_frame_num % self.redetect_interval == 0:
                            center1 = self._get_center_from_points(self.player1_points)
                            center2 = self._get_center_from_points(self.player2_points)
                            if center1:
                                self.player1_points = self._detect_features_in_region(new_gray, center1)
                            if center2:
                                self.player2_points = self._detect_features_in_region(new_gray, center2)
                        
                        # Track ball using color filtering
                        if self.ball_tracking and self.ball_pos is not None:
                            new_ball_pos = self._find_ball_in_region(frame, self.ball_pos)
                            
                            if new_ball_pos:
                                self.ball_pos = new_ball_pos
                                print(f"Frame {new_frame_num} - Ball at: {self.ball_pos}")
                            else:
                                print(f"Frame {new_frame_num} - Ball LOST! Click and press S")
                                self.ball_tracking = False
                        else:
                            print(f"Frame {new_frame_num}")
                        
                        self.current_gray = new_gray
                        self.prev_gray = new_gray.copy()
                        self.frame_count = new_frame_num
                elif key == ord('a') or key == ord('A'):
                    # Move back 1 frame
                    new_frame_num = max(self.frame_count - 1, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                    ret, frame = cap.read()
                    if ret:
                        self.current_frame = frame
                        self.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.prev_gray = self.current_gray.copy()
                        self.frame_count = new_frame_num
                        print(f"Frame {self.frame_count}")
                elif key == ord('w') or key == ord('W'):
                    # Forward 30 frames
                    new_frame_num = min(self.frame_count + 30, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                    ret, frame = cap.read()
                    if ret:
                        self.current_frame = frame
                        self.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        self.prev_gray = self.current_gray.copy()
                        self.frame_count = new_frame_num
                        print(f"Frame {self.frame_count}")
                elif key == ord('x') or key == ord('X'):
                    # Back 30 frames
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
    output_path = "tennis_color_tracking.mp4"
    
    tracker = ColorBasedBallTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()



