import argparse
import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, List


class HSVFilterTuner:
    def __init__(self):
        # HSV range values
        self.h_min = 0
        self.h_max = 98
        self.s_min = 0
        self.s_max = 255
        self.v_min = 217
        self.v_max = 255
        
        # Current region being tuned
        self.tuning_region = None
        self.tuning_window = "HSV Filter Tuner - Adjust until only BALL is white"
        self.config_file = "hsv_config.json"
    
    def get_hsv_lower(self):
        return np.array([self.h_min, self.s_min, self.v_min])
    
    def get_hsv_upper(self):
        return np.array([self.h_max, self.s_max, self.v_max])
    
    def tune(self, region, ball_center, auto_adjust=False):
        """Tune HSV filter on the given region."""
        if region is None or region.size == 0:
            return False
        
        self.tuning_region = region
        
        # Create window
        cv2.namedWindow(self.tuning_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.tuning_window, 800, 600)
        
        # Create trackbars
        cv2.createTrackbar('H Min', self.tuning_window, self.h_min, 179, self._on_trackbar)
        cv2.createTrackbar('H Max', self.tuning_window, self.h_max, 179, self._on_trackbar)
        cv2.createTrackbar('S Min', self.tuning_window, self.s_min, 255, self._on_trackbar)
        cv2.createTrackbar('S Max', self.tuning_window, self.s_max, 255, self._on_trackbar)
        cv2.createTrackbar('V Min', self.tuning_window, self.v_min, 255, self._on_trackbar)
        cv2.createTrackbar('V Max', self.tuning_window, self.v_max, 255, self._on_trackbar)
        
        # Auto-adjust if requested
        if auto_adjust and ball_center:
            self._auto_adjust_hsv(ball_center)
        
        # Update display
        self._update_display()
        
        print("\nHSV Filter Tuning Mode")
        print("=" * 50)
        print("Adjust the sliders until ONLY the BALL is WHITE")
        print("Everything else should be BLACK")
        print("Press 'S' to SAVE and continue tracking")
        print("Press 'Q' to CANCEL")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self._save_hsv_values()
                cv2.destroyWindow(self.tuning_window)
                return True
            elif key == ord('q'):
                cv2.destroyWindow(self.tuning_window)
                return False
    
    def _on_trackbar(self, val):
        """Trackbar callback."""
        self.h_min = cv2.getTrackbarPos('H Min', self.tuning_window)
        self.h_max = cv2.getTrackbarPos('H Max', self.tuning_window)
        self.s_min = cv2.getTrackbarPos('S Min', self.tuning_window)
        self.s_max = cv2.getTrackbarPos('S Max', self.tuning_window)
        self.v_min = cv2.getTrackbarPos('V Min', self.tuning_window)
        self.v_max = cv2.getTrackbarPos('V Max', self.tuning_window)
        self._update_display()
    
    def _auto_adjust_hsv(self, ball_center):
        """Auto-adjust HSV values based on ball center."""
        if self.tuning_region is None:
            return
        
        x, y = ball_center
        size = 10
        
        # Extract ball region
        x1 = max(0, x - size)
        y1 = max(0, y - size)
        x2 = min(self.tuning_region.shape[1], x + size)
        y2 = min(self.tuning_region.shape[0], y + size)
        
        ball_region = self.tuning_region[y1:y2, x1:x2]
        if ball_region.size == 0:
            return
        
        # Convert to HSV
        hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        
        # Set HSV range with tolerance
        self.h_min = max(0, int(mean_hsv[0]) - 10)
        self.h_max = min(179, int(mean_hsv[0]) + 10)
        self.s_min = max(0, int(mean_hsv[1]) - 10)
        self.s_max = min(255, int(mean_hsv[1]) + 10)
        self.v_min = max(0, int(mean_hsv[2]) - 10)
        self.v_max = min(255, int(mean_hsv[2]) + 10)
        
        # Update trackbars
        cv2.setTrackbarPos('H Min', self.tuning_window, self.h_min)
        cv2.setTrackbarPos('H Max', self.tuning_window, self.h_max)
        cv2.setTrackbarPos('S Min', self.tuning_window, self.s_min)
        cv2.setTrackbarPos('S Max', self.tuning_window, self.s_max)
        cv2.setTrackbarPos('V Min', self.tuning_window, self.v_min)
        cv2.setTrackbarPos('V Max', self.tuning_window, self.v_max)
    
    def _update_display(self):
        """Update the display with current HSV filter."""
        if self.tuning_region is None:
            return
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.tuning_region, cv2.COLOR_BGR2HSV)
        
        # Apply HSV filter
        mask = cv2.inRange(hsv, self.get_hsv_lower(), self.get_hsv_upper())
        
        # Create side-by-side display
        height, width = self.tuning_region.shape[:2]
        display = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Left side: original
        display[:height, :width] = self.tuning_region
        
        # Right side: HSV filter result
        display[:height, width:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow(self.tuning_window, display)
    
    def _save_hsv_values(self):
        """Save HSV values to config file."""
        config = {
            "h_min": self.h_min,
            "h_max": self.h_max,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "v_min": self.v_min,
            "v_max": self.v_max
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        print("\n" + "=" * 50)
        print("HSV Filter SAVED and will be used for all frames:")
        print("=" * 50)
        print(f"  H_MIN: {self.h_min}")
        print(f"  H_MAX: {self.h_max}")
        print(f"  S_MIN: {self.s_min}")
        print(f"  S_MAX: {self.s_max}")
        print(f"  V_MIN: {self.v_min}")
        print(f"  V_MAX: {self.v_max}")
        print("=" * 50)
        print(f"\nHSV values saved to {self.config_file}")
        print("Filter saved - Press 'D' to track ball")


class FocusedBallTracker:
    def __init__(self, video_path: str, start_frame: int = 100):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.start_frame = max(0, start_frame)
        self.frame_count = self.start_frame
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Jump to requested start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        # Ball tracking state
        self.ball_center = None
        self.ball_size = None
        self.tracking_box_size = 100
        self.search_tolerance = 5
        
        # HSV filter
        self.hsv_lower = np.array([0, 0, 217])
        self.hsv_upper = np.array([98, 255, 255])
        
        # Load saved HSV values
        self.load_hsv_config()
        
        # Player positions (from previous system)
        self.player1_pos = (1266, 114)
        self.player2_pos = (3142, 1036)
        
        # Court lines (from previous system)
        self.court_points = self.load_court_points()
        
    def load_hsv_config(self):
        """Load HSV values from config file."""
        try:
            with open("hsv_config.json", "r") as f:
                config = json.load(f)
                self.hsv_lower = np.array([config["h_min"], config["s_min"], config["v_min"]])
                self.hsv_upper = np.array([config["h_max"], config["s_max"], config["v_max"]])
                print(f"Loaded HSV values: H:{config['h_min']}-{config['h_max']}, S:{config['s_min']}-{config['s_max']}, V:{config['v_min']}-{config['v_max']}")
        except:
            print("Using default HSV values")
    
    def load_court_points(self):
        """Load court points for drawing court lines."""
        try:
            with open("court_points_coordinates.txt", "r") as f:
                lines = f.readlines()
                points = []
                for line in lines:
                    if line.strip():
                        x, y = map(int, line.strip().split(','))
                        points.append((x, y))
                return points
        except:
            return []
    
    def draw_court_lines(self, frame):
        """Draw court lines on the frame."""
        if len(self.court_points) >= 14:
            # Draw court lines
            lines = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Outer court
                (4, 5), (5, 6), (6, 7), (7, 4),  # Inner court
                (8, 9), (10, 11),  # Net lines
                (12, 13),  # Center service line
            ]
            
            for start_idx, end_idx in lines:
                if start_idx < len(self.court_points) and end_idx < len(self.court_points):
                    start_point = self.court_points[start_idx]
                    end_point = self.court_points[end_idx]
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
    
    def get_search_region(self, frame):
        """Get the 100x100 search region around the ball center."""
        if self.ball_center is None:
            return None, None, None
        
        x, y = self.ball_center
        half_size = self.tracking_box_size // 2
        
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(frame.shape[1], x + half_size)
        y2 = min(frame.shape[0], y + half_size)
        
        search_region = frame[y1:y2, x1:x2]
        return search_region, (x1, y1), (x2, y2)
    
    def detect_ball_in_region(self, region, region_offset):
        """Detect ball in the search region using HSV filter."""
        if region is None or region.size == 0:
            return None, 0
        
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Apply HSV filter
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0
        
        # Find the best ball candidate
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check if area is within expected ball size range
            if self.ball_size is not None:
                min_area = max(50, self.ball_size - self.search_tolerance)
                max_area = self.ball_size + self.search_tolerance
                if area < min_area or area > max_area:
                    continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate score based on area and circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                score = area * circularity
                
                if score > best_score:
                    best_score = score
                    best_ball = (cx, cy, area)
        
        if best_ball:
            # Convert back to full frame coordinates
            x_offset, y_offset = region_offset
            full_x = best_ball[0] + x_offset
            full_y = best_ball[1] + y_offset
            return (full_x, full_y, best_ball[2]), best_ball[2]
        
        return None, 0
    
    def mark_ball_initial(self, frame):
        """Allow user to mark the initial ball position and calculate its size."""
        print("\n=== MARK BALL INITIAL POSITION ===")
        print("Click on the tennis ball to mark its position")
        print("Press 'S' to save and start tracking")
        print("Press 'Q' to quit")
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ball_center = (x, y)
                print(f"Ball marked at: ({x}, {y})")
                
                # Calculate ball size in the 100x100 region
                region, offset, _ = self.get_search_region(frame)
                if region is not None:
                    ball_data, bulb_size = self.detect_ball_in_region(region, offset)
                    if ball_data:
                        self.ball_size = bulb_size
                        print(f"Ball bulb size detected: {bulb_size} pixels")
                        print(f"Search tolerance: {bulb_size + self.search_tolerance} pixels")
                    else:
                        print("Could not detect ball size in marked region")
        
        cv2.namedWindow("Mark Ball", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mark Ball", mouse_callback)
        
        while True:
            display_frame = frame.copy()
            
            # Draw court lines
            self.draw_court_lines(display_frame)
            
            # Draw players
            cv2.circle(display_frame, self.player1_pos, 20, (255, 0, 0), 2)
            cv2.putText(display_frame, "P1", (self.player1_pos[0]-10, self.player1_pos[1]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.circle(display_frame, self.player2_pos, 20, (0, 0, 255), 2)
            cv2.putText(display_frame, "P2", (self.player2_pos[0]-10, self.player2_pos[1]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw search box if ball is marked
            if self.ball_center:
                x, y = self.ball_center
                half_size = self.tracking_box_size // 2
                cv2.rectangle(display_frame, 
                            (x - half_size, y - half_size), 
                            (x + half_size, y + half_size), 
                            (0, 255, 0), 2)
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                
                # Show ball size info
                if self.ball_size:
                    cv2.putText(display_frame, f"Ball Size: {self.ball_size}px", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Search: {self.ball_size + self.search_tolerance}px", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Mark Ball", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and self.ball_center and self.ball_size:
                print("Ball position and size saved!")
                break
            elif key == ord('q'):
                print("Exiting...")
                return False
        
        cv2.destroyWindow("Mark Ball")
        return True
    
    def track_ball_frame(self, frame):
        """Track ball in current frame."""
        if self.ball_center is None:
            return False
        
        # Get search region
        region, offset, _ = self.get_search_region(frame)
        if region is None:
            return False
        
        # Detect ball in region
        ball_data, detected_size = self.detect_ball_in_region(region, offset)
        
        if ball_data:
            # Update ball position
            self.ball_center = (ball_data[0], ball_data[1])
            print(f"Ball tracked at: ({ball_data[0]}, {ball_data[1]}) - Size: {detected_size}px")
            return True
        else:
            print("Ball lost in frame")
            return False
    
    def draw_tracking_info(self, frame):
        """Draw tracking information on frame."""
        if self.ball_center:
            # Draw search box
            x, y = self.ball_center
            half_size = self.tracking_box_size // 2
            cv2.rectangle(frame, 
                        (x - half_size, y - half_size), 
                        (x + half_size, y + half_size), 
                        (0, 255, 0), 2)
            
            # Draw ball center
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
            
            # Draw ball size info
            if self.ball_size:
                cv2.putText(frame, f"Ball: ({x}, {y})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Size: {self.ball_size}px", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def open_hsv_tuner(self, frame):
        """Open HSV tuner for the current ball position."""
        if self.ball_center is None:
            print("No ball position marked yet!")
            return False
        
        # Extract region around ball
        x, y = self.ball_center
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
        
        # Create HSV tuner
        tuner = HSVFilterTuner()
        result = tuner.tune(region, region_ball_center, auto_adjust=False)
        
        if result:
            # Update HSV values
            self.hsv_lower = tuner.get_hsv_lower()
            self.hsv_upper = tuner.get_hsv_upper()
            print("HSV values updated!")
            return True
        
        return False
    
    def process_video(self):
        """Process the entire video with interactive ball tracking."""
        print("Interactive Ball Tracking System")
        print("=" * 50)
        print("Controls:")
        print("W/A/S/D - Navigate frames (W=+30, A=-1, S=+1, D=+1)")
        print("S - Open HSV tuner")
        print("Click - Mark new ball position")
        print("Q - Quit")
        print("=" * 50)
        if self.start_frame > 0:
            print(f"Starting at frame {self.start_frame}")
        
        # Get first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read video")
            return
        
        # Resize frame to fit screen
        height, width = frame.shape[:2]
        scale = min(1920/width, 1080/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('interactive_ball_tracking.mp4', fourcc, self.fps, (new_width, new_height))
        
        # Mouse callback for ball marking
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ball_center = (x, y)
                print(f"Ball marked at: ({x}, {y})")
                
                # Calculate ball size in the 100x100 region
                region, offset, _ = self.get_search_region(frame)
                if region is not None:
                    ball_data, bulb_size = self.detect_ball_in_region(region, offset)
                    if ball_data:
                        self.ball_size = bulb_size
                        print(f"Ball bulb size: {bulb_size} pixels")
                    else:
                        print("Could not detect ball size")
        
        cv2.namedWindow("Interactive Ball Tracking", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Interactive Ball Tracking", mouse_callback)
        
        print(f"\nReady! Navigate with W/A/S/D, press S for HSV tuner")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Draw court lines and players
            self.draw_court_lines(resized_frame)
            
            # Draw players
            cv2.circle(resized_frame, self.player1_pos, 20, (255, 0, 0), 2)
            cv2.putText(resized_frame, "P1", (self.player1_pos[0]-10, self.player1_pos[1]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.circle(resized_frame, self.player2_pos, 20, (0, 0, 255), 2)
            cv2.putText(resized_frame, "P2", (self.player2_pos[0]-10, self.player2_pos[1]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Try to track ball if we have a position
            ball_detected = False
            if self.ball_center:
                ball_detected = self.track_ball_frame(resized_frame)
                if ball_detected:
                    self.draw_tracking_info(resized_frame)
                else:
                    cv2.putText(resized_frame, "BALL LOST - Click to mark new position", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show frame info
            cv2.putText(resized_frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                       (10, new_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if self.ball_center and self.ball_size:
                cv2.putText(resized_frame, f"Ball Size: {self.ball_size}px", (10, new_height-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Interactive Ball Tracking", resized_frame)
            out.write(resized_frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):  # +30 frames
                for _ in range(30):
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_count += 1
                    else:
                        break
            elif key == ord('a'):  # -1 frame
                # Go back one frame (not easily possible with OpenCV)
                print("Cannot go backward with this video format")
            elif key == ord('s'):  # +1 frame and open HSV tuner
                self.frame_count += 1
                if self.ball_center:
                    print("Opening HSV tuner...")
                    if self.open_hsv_tuner(resized_frame):
                        print("HSV filter updated!")
                    else:
                        print("HSV tuning cancelled")
                else:
                    print("Mark ball position first!")
            elif key == ord('d'):  # +1 frame
                self.frame_count += 1
        
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nTracking complete!")
        print(f"Processed {self.frame_count} frames")
        print(f"Output: interactive_ball_tracking.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive tennis ball tracker")
    parser.add_argument("--start-frame", type=int, default=100,
                        help="Frame index to start tracking from (default 100 for serve)")
    args = parser.parse_args()
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    tracker = FocusedBallTracker(video_path, start_frame=args.start_frame)
    tracker.process_video()
