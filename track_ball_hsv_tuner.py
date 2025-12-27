"""
Ball tracking with interactive HSV filter tuner.
Allows user to adjust HSV ranges to perfectly isolate the tennis ball.
"""

import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional


class HSVFilterTuner:
    def __init__(self):
        # HSV range values - very restrictive for tennis ball only
        self.h_min = 50
        self.h_max = 80
        self.s_min = 150
        self.s_max = 255
        self.v_min = 200
        self.v_max = 255
        
        # Current region being tuned
        self.tuning_region = None
        self.tuning_window = "HSV Filter Tuner - Adjust until only BALL is white"
        self.config_file = "hsv_config.json"
        
        # Load saved values if they exist
        self.load_hsv_values()
    
    def save_hsv_values(self):
        """Save HSV values to a config file."""
        config = {
            'h_min': int(self.h_min),
            'h_max': int(self.h_max),
            's_min': int(self.s_min),
            's_max': int(self.s_max),
            'v_min': int(self.v_min),
            'v_max': int(self.v_max)
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nHSV values saved to {self.config_file}")
    
    def load_hsv_values(self):
        """Load HSV values from config file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                self.h_min = config.get('h_min', self.h_min)
                self.h_max = config.get('h_max', self.h_max)
                self.s_min = config.get('s_min', self.s_min)
                self.s_max = config.get('s_max', self.s_max)
                self.v_min = config.get('v_min', self.v_min)
                self.v_max = config.get('v_max', self.v_max)
                print(f"\nLoaded saved HSV values from {self.config_file}:")
                print(f"  H: {self.h_min}-{self.h_max}")
                print(f"  S: {self.s_min}-{self.s_max}")
                print(f"  V: {self.v_min}-{self.v_max}")
            except Exception as e:
                print(f"Could not load HSV config: {e}")
    
    def create_trackbars(self):
        """Create trackbars for HSV adjustment."""
        # Destroy existing window if it exists
        try:
            cv2.destroyWindow(self.tuning_window)
            for _ in range(10):
                cv2.waitKey(1)  # Allow window destruction to complete
        except:
            pass
        
        cv2.namedWindow(self.tuning_window, cv2.WINDOW_NORMAL)
        for _ in range(10):
            cv2.waitKey(1)  # Allow window creation to complete
        
        cv2.createTrackbar('H Min', self.tuning_window, self.h_min, 179, lambda x: None)
        cv2.createTrackbar('H Max', self.tuning_window, self.h_max, 179, lambda x: None)
        cv2.createTrackbar('S Min', self.tuning_window, self.s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', self.tuning_window, self.s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', self.tuning_window, self.v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', self.tuning_window, self.v_max, 255, lambda x: None)
    
    def get_trackbar_values(self):
        """Get current trackbar values."""
        self.h_min = cv2.getTrackbarPos('H Min', self.tuning_window)
        self.h_max = cv2.getTrackbarPos('H Max', self.tuning_window)
        self.s_min = cv2.getTrackbarPos('S Min', self.tuning_window)
        self.s_max = cv2.getTrackbarPos('S Max', self.tuning_window)
        self.v_min = cv2.getTrackbarPos('V Min', self.tuning_window)
        self.v_max = cv2.getTrackbarPos('V Max', self.tuning_window)
    
    def filter_with_current_values(self, frame: np.ndarray) -> np.ndarray:
        """Apply current HSV filter values to frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def auto_set_hsv_from_ball(self, region: np.ndarray, ball_center: Tuple[int, int]):
        """Automatically set HSV values based on ball position with ±10 range."""
        x, y = ball_center
        height, width = region.shape[:2]
        
        # Ensure ball center is within region bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Get HSV value at ball center
        h_val = int(hsv[y, x, 0])
        s_val = int(hsv[y, x, 1])
        v_val = int(hsv[y, x, 2])
        
        # Set range with ±10 tolerance
        self.h_min = max(0, h_val - 10)
        self.h_max = min(179, h_val + 10)
        self.s_min = max(0, s_val - 10)
        self.s_max = min(255, s_val + 10)
        self.v_min = max(0, v_val - 10)
        self.v_max = min(255, v_val + 10)
        
        print(f"Auto-set HSV from ball at ({x}, {y}):")
        print(f"  Ball HSV: ({h_val}, {s_val}, {v_val})")
        print(f"  Range: H({self.h_min}-{self.h_max}), S({self.s_min}-{self.s_max}), V({self.v_min}-{self.v_max})")

    def tune(self, region: np.ndarray, ball_center: Tuple[int, int] = None, auto_adjust: bool = False):
        """
        Interactive HSV tuning session.
        Returns True if user saved settings, False if cancelled.
        auto_adjust: If True, auto-calculate HSV from ball_center. If False, keep current values.
        """
        self.tuning_region = region
        
        # Auto-set HSV values if ball center is provided AND auto_adjust is True
        if ball_center is not None and auto_adjust:
            self.auto_set_hsv_from_ball(region, ball_center)
        else:
            print(f"\nUsing saved HSV values:")
            print(f"  H: {self.h_min}-{self.h_max}")
            print(f"  S: {self.s_min}-{self.s_max}")
            print(f"  V: {self.v_min}-{self.v_max}")
        
        self.create_trackbars()
        
        print("\nHSV Filter Tuning Mode")
        print("=" * 50)
        print("Adjust the sliders until ONLY the BALL is WHITE")
        print("Everything else should be BLACK")
        print("Press 'S' to SAVE and continue tracking")
        print("Press 'Q' to CANCEL")
        print()
        
        while True:
            # Get current values
            self.get_trackbar_values()
            
            # Apply filter
            mask = self.filter_with_current_values(region)
            
            # Create display image showing original and filtered side by side
            display = np.zeros((region.shape[0], region.shape[1] * 2, 3), dtype=np.uint8)
            display[:, :region.shape[1]] = region
            display[:, region.shape[1]:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Only show control instructions, no HSV values
            cv2.putText(display, "S: Save | Q: Cancel", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(self.tuning_window, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                print(f"\n{'='*50}")
                print(f"HSV Filter SAVED and will be used for all frames:")
                print(f"{'='*50}")
                print(f"  H_MIN: {self.h_min}")
                print(f"  H_MAX: {self.h_max}")
                print(f"  S_MIN: {self.s_min}")
                print(f"  S_MAX: {self.s_max}")
                print(f"  V_MIN: {self.v_min}")
                print(f"  V_MAX: {self.v_max}")
                print(f"{'='*50}")
                self.save_hsv_values()
                cv2.destroyWindow(self.tuning_window)
                return True
            elif key == ord('q') or key == ord('Q'):
                print("Filter tuning cancelled")
                cv2.destroyWindow(self.tuning_window)
                return False
    
    def get_filter_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current filter values as lower and upper bounds."""
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        return lower, upper


class ImprovedBallTracker:
    def __init__(self, player1_pos: Tuple[int, int], player2_pos: Tuple[int, int]):
        self.player1_initial = player1_pos
        self.player2_initial = player2_pos
        
        # Player tracking
        self.player1_points = None
        self.player2_points = None
        
        # Ball tracking
        self.ball_pos = None
        self.ball_tracking = False
        self.ball_search_radius = 80
        self.ball_marked_pos = None
        
        # Multiple ball detection
        self.detected_balls = []  # List of all detected ball positions
        self.false_positive_balls = []  # List of false positive positions to filter out
        
        # HSV filter - using the excellent values from the user's example
        self.hsv_tuner = HSVFilterTuner()
        # Get initial values from tuner (which loads saved values)
        self.hsv_lower, self.hsv_upper = self.hsv_tuner.get_filter_values()
        
        # Track if HSV has been set by user
        self.hsv_user_configured = os.path.exists(self.hsv_tuner.config_file)
        
        # Video state
        self.paused = True
        self.current_frame = None
        self.current_gray = None
        
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
    
    def _find_ball_with_hsv_filter(self, frame: np.ndarray, search_center: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find ball using HSV color filter in search region."""
        x, y = search_center
        height, width = frame.shape[:2]
        
        # Define search region
        x1 = max(0, x - self.ball_search_radius)
        y1 = max(0, y - self.ball_search_radius)
        x2 = min(width, x + self.ball_search_radius)
        y2 = min(height, y + self.ball_search_radius)
        
        search_region = frame[y1:y2, x1:x2]
        
        if search_region.size == 0:
            print(f"  DEBUG: Empty search region at {search_center}")
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Apply current HSV filter
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"  DEBUG: No contours found in search region at {search_center}, radius {self.ball_search_radius}")
            return None
        
        print(f"  DEBUG: Found {len(contours)} contours in search region")
        
        # Find best contour
        best_contour = None
        best_distance = float('inf')
        region_center = (search_region.shape[1] // 2, search_region.shape[0] // 2)
        
        valid_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 15 or area > 1500:
                continue
            
            valid_contours += 1
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            dx = cx - region_center[0]
            dy = cy - region_center[1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < best_distance:
                best_distance = distance
                best_contour = contour
                best_cx = cx + x1
                best_cy = cy + y1
        
        print(f"  DEBUG: {valid_contours} valid contours (area 15-1500)")
        
        if best_contour is not None:
            print(f"  DEBUG: Ball found at ({best_cx}, {best_cy})")
            return (best_cx, best_cy)
        
        print(f"  DEBUG: No valid ball contour found")
        return None
    
    def _detect_all_balls_in_frame(self, frame: np.ndarray) -> list:
        """Detect all balls in the entire frame using HSV filter."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply HSV filter
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Debug: Show HSV filter results
        white_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        print(f"  HSV Filter: {white_pixels}/{total_pixels} pixels match ({(white_pixels/total_pixels)*100:.1f}%)")
        
        # Clean up with larger kernel to better match tennis ball size
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to expand the ball detection area
        dilate_kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        
        # Save mask for debugging (first frame only)
        if self.frame_count == 0:
            cv2.imwrite("hsv_mask_debug.jpg", mask)
            print(f"  Saved HSV mask to hsv_mask_debug.jpg")
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_balls = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1 or area > 20:  # Very small tennis ball size: 1-20 pixels
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check circularity (balls should be roughly circular)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:  # More strict circularity for tennis balls
                continue
            
            # Check if this position is in false positive list
            is_false_positive = False
            for fp_x, fp_y in self.false_positive_balls:
                distance = np.sqrt((cx - fp_x)**2 + (cy - fp_y)**2)
                if distance < 50:  # Within 50 pixels of a false positive
                    is_false_positive = True
                    break
            
            if not is_false_positive:
                detected_balls.append((cx, cy, area))
        
        # Sort by a combination of area and circularity to find the best tennis ball
        def ball_score(ball_data):
            cx, cy, area = ball_data
            # Prefer larger balls but not too large (tennis ball size)
            size_score = min(area / 200.0, 1.0)  # Normalize to 0-1, peak at 200 pixels
            return size_score
        
        detected_balls.sort(key=ball_score, reverse=True)
        
        print(f"  DEBUG: Detected {len(detected_balls)} balls in frame")
        for i, (cx, cy, area) in enumerate(detected_balls):
            print(f"    Ball {i+1}: ({cx}, {cy}) area={area}")
        
        return detected_balls
    
    def _open_hsv_tuner(self, frame: np.ndarray, ball_center: Tuple[int, int]):
        """Open HSV tuner window with region around ball."""
        # Extract region around ball
        x, y = ball_center
        size = 80
        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(frame.shape[1], x + size)
        y2 = min(frame.shape[0], y + size)
        
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            print("Error: Cannot extract region")
            return False
        
        # Calculate ball center relative to the extracted region
        region_ball_center = (x - x1, y - y1)
        
        # Never auto-adjust - always use saved values
        auto_adjust = False
        
        # Open tuner with ball center for auto HSV setting
        result = self.hsv_tuner.tune(region, region_ball_center, auto_adjust)
        
        # If user saved, mark as configured
        if result:
            self.hsv_user_configured = True
        
        return result
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if not self.paused:
            return
        
        scale = param
        
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (int(x / scale), int(y / scale))
            
            # Check if clicking on a detected ball
            clicked_on_ball = False
            for i, (ball_x, ball_y, area) in enumerate(self.detected_balls):
                distance = np.sqrt((click_pos[0] - ball_x)**2 + (click_pos[1] - ball_y)**2)
                if distance < 50:  # Within 50 pixels of a detected ball
                    if i == 0:  # Clicked on the main ball (first in list)
                        # This is the actual tennis ball - select it
                        self.ball_pos = (ball_x, ball_y)
                        self.ball_tracking = True
                        print(f"Selected main ball at: ({ball_x}, {ball_y}) - area: {area}")
                        print("Press 'D' to continue tracking")
                    else:
                        # Mark as false positive
                        self.false_positive_balls.append(click_pos)
                        print(f"Marked false positive at: {click_pos}")
                        print(f"Total false positives: {len(self.false_positive_balls)}")
                    clicked_on_ball = True
                    break
            
            if not clicked_on_ball:
                # Regular ball marking
                self.ball_marked_pos = click_pos
                print(f"Ball marked at: {click_pos}")
                print("Press 'S' to open HSV tuner")
    
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
        
        # Draw all detected balls (Green circles)
        for i, (ball_x, ball_y, area) in enumerate(self.detected_balls):
            display_x = int(ball_x * scale)
            display_y = int(ball_y * scale)
            size = int(20 * scale)
            
            # Draw X
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 255, 0), 3)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 255, 0), 3)
            
            # Draw circle
            cv2.circle(result, (display_x, display_y), int(30 * scale), (0, 255, 0), 3)
            
            # Draw ball number
            cv2.putText(result, f"B{i+1}", (display_x + int(35 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), 2)
        
        # Draw main ball position (if tracking)
        if self.ball_pos is not None and self.ball_tracking:
            x, y = self.ball_pos
            display_x = int(x * scale)
            display_y = int(y * scale)
            size = int(25 * scale)
            cv2.line(result, (display_x - size, display_y - size), 
                    (display_x + size, display_y + size), (0, 200, 0), 4)
            cv2.line(result, (display_x - size, display_y + size), 
                    (display_x + size, display_y - size), (0, 200, 0), 4)
            cv2.circle(result, (display_x, display_y), int(35 * scale), (0, 200, 0), 4)
            cv2.putText(result, "MAIN", (display_x + int(40 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 200, 0), 2)
        
        # Draw marked position
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
            cv2.putText(result, "Press S to tune", (display_x + int(35 * scale), display_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 200, 0), 2)
        
        # Status
        if len(self.detected_balls) > 0:
            status_text = f"DETECTED {len(self.detected_balls)} BALLS - Click false positives to filter | Press D to advance"
        else:
            status_text = "No balls detected - Press D to search | Click + S to tune filter"
        
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(result, f"Frame: {self.frame_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Controls
        controls = [
            "D/A: 1 frame | W/X: 30 frames",
            "Click: Adjust ball | S: Tune filter | Q: Quit"
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
        
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
        _, self.display_scale = self._resize_frame_to_fit_screen(test_frame)
        
        print("\nAutomatic Multi-Ball Detection")
        print("=" * 50)
        print("1. Navigate with D/A/W/X to find balls")
        print("2. Press 'D' to advance frame - system detects ALL balls automatically")
        print("3. Green circles show all detected balls (B1, B2, B3...)")
        print("4. Click on FALSE POSITIVE balls to filter them out")
        print("5. Press 'S' to open HSV tuner if needed")
        print("6. System learns from your false positive clicks")
        print("7. Main ball (largest) is highlighted with 'MAIN' label")
        print()
        
        window_name = "HSV Ball Tracker - Click to adjust ball, S to tune, D to track"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback, self.display_scale)
        
        self.frame_count = 0
        
        try:
            # Read first frame
            ret, frame = cap.read()
            if not ret:
                return
            
            self.current_frame = frame
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.current_gray = self.prev_gray.copy()
            
            # Initialize players
            self.player1_points = self._detect_features_in_region(self.prev_gray, self.player1_initial)
            self.player2_points = self._detect_features_in_region(self.prev_gray, self.player2_initial)
            
            print("Ready to mark ball")
            
            while True:
                # Draw
                frame_with_lines = self._draw_court_lines(self.current_frame)
                display_frame = self._draw_markers(frame_with_lines, 1.0)
                out.write(display_frame)
                
                display_resized, _ = self._resize_frame_to_fit_screen(display_frame)
                cv2.imshow(window_name, display_resized)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    if self.ball_marked_pos is not None:
                        # Open HSV tuner using the marked ball position
                        print(f"\nOpening HSV tuner for ball at: {self.ball_marked_pos}")
                        if self._open_hsv_tuner(self.current_frame, self.ball_marked_pos):
                            # User saved filter settings
                            self.hsv_lower, self.hsv_upper = self.hsv_tuner.get_filter_values()
                            self.ball_pos = self.ball_marked_pos
                            self.ball_tracking = True
                            self.ball_marked_pos = None
                            print("Filter saved - Press 'D' to track ball")
                        else:
                            print("Filter tuning cancelled - mark ball again if needed")
                elif key == ord('d') or key == ord('D'):
                    # Move forward
                    new_frame_num = min(self.frame_count + 1, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_num)
                    ret, frame = cap.read()
                    if ret:
                        self.current_frame = frame
                        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # Track players
                        if self.player1_points is not None:
                            new_p1, s1, _ = cv2.calcOpticalFlowPyrLK(
                                self.prev_gray, new_gray, self.player1_points, None, **self.lk_params)
                            if new_p1 is not None:
                                good = new_p1[s1.flatten() == 1]
                                if len(good) > 0:
                                    self.player1_points = good.reshape(-1, 1, 2)
                        
                        if self.player2_points is not None:
                            new_p2, s2, _ = cv2.calcOpticalFlowPyrLK(
                                self.prev_gray, new_gray, self.player2_points, None, **self.lk_params)
                            if new_p2 is not None:
                                good = new_p2[s2.flatten() == 1]
                                if len(good) > 0:
                                    self.player2_points = good.reshape(-1, 1, 2)
                        
                        # Detect all balls in frame
                        self.detected_balls = self._detect_all_balls_in_frame(frame)
                        print(f"Frame {new_frame_num} - Detected {len(self.detected_balls)} balls")
                        
                        # If we have detected balls, use the largest one as main ball
                        if self.detected_balls:
                            # Use the largest detected ball as the main ball
                            main_ball = self.detected_balls[0]  # Already sorted by area
                            self.ball_pos = (main_ball[0], main_ball[1])
                            self.ball_tracking = True
                            print(f"  Main ball: {self.ball_pos} (area: {main_ball[2]})")
                        else:
                            print(f"  No balls detected")
                            self.ball_tracking = False
                        
                        self.prev_gray = new_gray.copy()
                        self.current_gray = new_gray
                        self.frame_count = new_frame_num
                elif key == ord('a') or key == ord('A'):
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
    try:
        with open('tracking_config.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if "Player 1 (Blue):" in line:
                    pos_str = line.split(":")[1].strip().replace("(", "").replace(")", "")
                    x, y = map(int, pos_str.split(","))
                    player1_pos = (x, y)
                elif "Player 2 (Red):" in line:
                    pos_str = line.split(":")[1].strip().replace("(", "").replace(")", "")
                    x, y = map(int, pos_str.split(","))
                    player2_pos = (x, y)
    except FileNotFoundError:
        print("Error: tracking_config.txt not found.")
        return
    
    print(f"Player 1 (Blue): {player1_pos}")
    print(f"Player 2 (Red): {player2_pos}")
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    output_path = "tennis_hsv_tuned_tracking.mp4"
    
    tracker = ImprovedBallTracker(player1_pos, player2_pos)
    tracker.process_video(video_path, output_path)


if __name__ == "__main__":
    main()



