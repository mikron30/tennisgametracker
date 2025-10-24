import cv2
import numpy as np
import json
from typing import Tuple, Optional


class InteractiveBallAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Ball analysis state
        self.ball_center = None
        self.ball_hsv = None
        self.ball_size = None
        self.tracking = False
        self.ball_stopped = False
        self.ball_velocity_history = []
        self.initial_ball_position = None
        
        # HSV filter (will be set based on clicked ball)
        self.hsv_lower = None
        self.hsv_upper = None
        
        # HSV values table for analysis
        self.hsv_table = []
        
        # Player positions
        self.player1_pos = (1266, 114)
        self.player2_pos = (3142, 1036)
        
    def load_hsv_config(self):
        """Load HSV values from config file."""
        try:
            with open("hsv_config.json", "r") as f:
                config = json.load(f)
                
                # Check if new format (with regular_court and behind_net)
                if "regular_court" in config and "behind_net" in config:
                    # New format with two HSV sets
                    self.hsv_regular = {
                        'lower': np.array([config["regular_court"]["h_min"], config["regular_court"]["s_min"], config["regular_court"]["v_min"]], dtype=np.uint8),
                        'upper': np.array([config["regular_court"]["h_max"], config["regular_court"]["s_max"], config["regular_court"]["v_max"]], dtype=np.uint8)
                    }
                    self.hsv_behind_net = {
                        'lower': np.array([config["behind_net"]["h_min"], config["behind_net"]["s_min"], config["behind_net"]["v_min"]], dtype=np.uint8),
                        'upper': np.array([config["behind_net"]["h_max"], config["behind_net"]["s_max"], config["behind_net"]["v_max"]], dtype=np.uint8)
                    }
                    self.net_area_y_min = config.get("net_area_y_min", 250)
                    self.net_area_y_max = config.get("net_area_y_max", 350)
                    
                    # Load high/low net boundaries if available
                    if "high_net_y_min" in config:
                        self.high_net_y_min = config["high_net_y_min"]
                        self.high_net_y_max = config["high_net_y_max"]
                    if "low_net_y_min" in config:
                        self.low_net_y_min = config["low_net_y_min"]
                        self.low_net_y_max = config["low_net_y_max"]
                    
                    # Set initial HSV to regular court
                    self.hsv_lower = self.hsv_regular['lower']
                    self.hsv_upper = self.hsv_regular['upper']
                    
                    print(f"Loaded DUAL HSV config:")
                    print(f"  Regular court: H:{config['regular_court']['h_min']}-{config['regular_court']['h_max']}, S:{config['regular_court']['s_min']}-{config['regular_court']['s_max']}, V:{config['regular_court']['v_min']}-{config['regular_court']['v_max']}")
                    print(f"  Behind net: H:{config['behind_net']['h_min']}-{config['behind_net']['h_max']}, S:{config['behind_net']['s_min']}-{config['behind_net']['s_max']}, V:{config['behind_net']['v_min']}-{config['behind_net']['v_max']}")
                    print(f"  Net area: Y={self.net_area_y_min}-{self.net_area_y_max}")
                    if hasattr(self, 'high_net_y_min') and hasattr(self, 'low_net_y_min'):
                        print(f"    HIGH NET: Y={self.high_net_y_min}-{self.high_net_y_max}")
                        print(f"    LOW NET: Y={self.low_net_y_min}-{self.low_net_y_max}")
                else:
                    # Old format (single HSV set)
                    self.hsv_lower = np.array([config["h_min"], config["s_min"], config["v_min"]], dtype=np.uint8)
                    self.hsv_upper = np.array([config["h_max"], config["s_max"], config["v_max"]], dtype=np.uint8)
                    self.hsv_regular = None
                    self.hsv_behind_net = None
                    print(f"Loaded HSV values: H:{config['h_min']}-{config['h_max']}, S:{config['s_min']}-{config['s_max']}, V:{config['v_min']}-{config['v_max']}")
                
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default HSV values")
            return None
    
    def select_hsv_for_position(self, y_position, at_edge=False):
        """Select appropriate HSV config based on ball Y position."""
        if self.hsv_regular is None or self.hsv_behind_net is None:
            # Single HSV mode
            if at_edge:
                # Use relaxed V_min at edges (ball may be darker when off-screen/returning)
                hsv_lower_edge = self.hsv_lower.copy()
                hsv_lower_edge[2] = max(70, self.hsv_lower[2] - 50)  # Lower V_min by 50
                return hsv_lower_edge, self.hsv_upper, "single_edge"
            return self.hsv_lower, self.hsv_upper, "single"
        
        # Check if ball is in net area
        if self.net_area_y_min <= y_position <= self.net_area_y_max:
            return self.hsv_behind_net['lower'], self.hsv_behind_net['upper'], "behind_net"
        elif at_edge:
            # Use behind_net HSV (more relaxed) at edges
            return self.hsv_behind_net['lower'], self.hsv_behind_net['upper'], "at_edge"
        else:
            return self.hsv_regular['lower'], self.hsv_regular['upper'], "regular_court"
    
    def mark_net_area(self, frame):
        """Interactive tool to mark the net area with 6 points (3 for high net, 3 for low net)."""
        print("\n" + "=" * 70)
        print("=== MARK NET AREA (6 POINTS) ===")
        print("=" * 70)
        print("Click 6 points clockwise around the net area:")
        print("  1. LEFT edge of HIGH net (top-left)")
        print("  2. MIDDLE of HIGH net (top-middle)")
        print("  3. RIGHT edge of HIGH net (top-right)")
        print("  4. RIGHT edge of LOW net (bottom-right)")
        print("  5. MIDDLE of LOW net (bottom-middle)")
        print("  6. LEFT edge of LOW net (bottom-left)")
        print("Press 'Q' or ESC when done (after marking all 6 points)")
        print("=" * 70)
        
        points = []
        window_name = "Mark Net Area - 6 Points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Resize frame for display
        screen_height = 900
        scale_factor = screen_height / frame.shape[0]
        new_width = int(frame.shape[1] * scale_factor)
        display_frame = cv2.resize(frame, (new_width, screen_height))
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                # Convert to original frame coordinates
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                points.append((orig_x, orig_y))
                
                # Label points based on position
                labels = ["HIGH NET Left", "HIGH NET Middle", "HIGH NET Right", 
                         "LOW NET Right", "LOW NET Middle", "LOW NET Left"]
                print(f"Point {len(points)} ({labels[len(points)-1]}): ({orig_x}, {orig_y})")
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            display = display_frame.copy()
            
            # Draw existing points with color coding
            for i, pt in enumerate(points):
                disp_x = int(pt[0] * scale_factor)
                disp_y = int(pt[1] * scale_factor)
                
                # Color: Green for high net (1-3), Blue for low net (4-6)
                color = (0, 255, 0) if i < 3 else (255, 0, 0)
                
                cv2.circle(display, (disp_x, disp_y), 10, color, -1)
                cv2.putText(display, str(i+1), (disp_x+15, disp_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw lines for high net (points 1-2-3)
            if len(points) >= 2:
                pt1 = (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor))
                pt2 = (int(points[1][0] * scale_factor), int(points[1][1] * scale_factor))
                cv2.line(display, pt1, pt2, (0, 255, 0), 2)
            if len(points) >= 3:
                pt2 = (int(points[1][0] * scale_factor), int(points[1][1] * scale_factor))
                pt3 = (int(points[2][0] * scale_factor), int(points[2][1] * scale_factor))
                cv2.line(display, pt2, pt3, (0, 255, 0), 2)
            
            # Draw lines for low net (points 4-5-6)
            if len(points) >= 5:
                pt4 = (int(points[3][0] * scale_factor), int(points[3][1] * scale_factor))
                pt5 = (int(points[4][0] * scale_factor), int(points[4][1] * scale_factor))
                cv2.line(display, pt4, pt5, (255, 0, 0), 2)
            if len(points) >= 6:
                pt5 = (int(points[4][0] * scale_factor), int(points[4][1] * scale_factor))
                pt6 = (int(points[5][0] * scale_factor), int(points[5][1] * scale_factor))
                cv2.line(display, pt5, pt6, (255, 0, 0), 2)
            
            # Draw connecting lines between high and low net
            if len(points) >= 4:
                pt1 = (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor))
                pt4 = (int(points[3][0] * scale_factor), int(points[3][1] * scale_factor))
                cv2.line(display, pt1, pt4, (0, 255, 255), 2)
            if len(points) >= 6:
                pt3 = (int(points[2][0] * scale_factor), int(points[2][1] * scale_factor))
                pt6 = (int(points[5][0] * scale_factor), int(points[5][1] * scale_factor))
                cv2.line(display, pt3, pt6, (0, 255, 255), 2)
                
                # Fill semi-transparent area
                overlay = display.copy()
                pts_array = np.array([
                    [int(points[0][0] * scale_factor), int(points[0][1] * scale_factor)],
                    [int(points[1][0] * scale_factor), int(points[1][1] * scale_factor)],
                    [int(points[2][0] * scale_factor), int(points[2][1] * scale_factor)],
                    [int(points[5][0] * scale_factor), int(points[5][1] * scale_factor)],
                    [int(points[4][0] * scale_factor), int(points[4][1] * scale_factor)],
                    [int(points[3][0] * scale_factor), int(points[3][1] * scale_factor)]
                ], np.int32)
                cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
                display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            
            # Status text
            status = f"Points marked: {len(points)}/6"
            if len(points) < 3:
                status += " (HIGH NET)"
            elif len(points) < 6:
                status += " (LOW NET)"
            else:
                status += " (COMPLETE - Press Q)"
            
            cv2.putText(display, status, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                if len(points) == 6:
                    break
                else:
                    print(f"Please mark all 6 points (currently {len(points)}/6)")
        
        cv2.destroyWindow(window_name)
        
        if len(points) == 6:
            # Points order: 0,1,2 = high net (left, middle, right)
            #               3,4,5 = low net (right, middle, left)
            high_net_y_coords = [points[0][1], points[1][1], points[2][1]]
            low_net_y_coords = [points[3][1], points[4][1], points[5][1]]
            
            self.high_net_y_min = min(high_net_y_coords)
            self.high_net_y_max = max(high_net_y_coords)
            self.low_net_y_min = min(low_net_y_coords)
            self.low_net_y_max = max(low_net_y_coords)
            
            # Overall net area encompasses both high and low
            self.net_area_y_min = min(self.high_net_y_min, self.low_net_y_min)
            self.net_area_y_max = max(self.high_net_y_max, self.low_net_y_max)
            
            print(f"\nNet area marked successfully!")
            print(f"HIGH NET: Y={self.high_net_y_min} - {self.high_net_y_max}")
            print(f"LOW NET: Y={self.low_net_y_min} - {self.low_net_y_max}")
            print(f"OVERALL NET AREA: Y={self.net_area_y_min} - {self.net_area_y_max}")
            
            # Save to config
            self.save_net_area_to_config()
            return True
        
        return False
    
    def save_net_area_to_config(self):
        """Save net area boundaries to config file."""
        try:
            with open("hsv_config.json", "r") as f:
                config = json.load(f)
            
            # Save overall net area
            config["net_area_y_min"] = int(self.net_area_y_min)
            config["net_area_y_max"] = int(self.net_area_y_max)
            
            # Save detailed high/low net boundaries if available
            if hasattr(self, 'high_net_y_min'):
                config["high_net_y_min"] = int(self.high_net_y_min)
                config["high_net_y_max"] = int(self.high_net_y_max)
            if hasattr(self, 'low_net_y_min'):
                config["low_net_y_min"] = int(self.low_net_y_min)
                config["low_net_y_max"] = int(self.low_net_y_max)
            
            with open("hsv_config.json", "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"Net area boundaries saved to hsv_config.json")
        except Exception as e:
            print(f"Error saving net area: {e}")
    
    def open_hsv_tuner(self, frame, point):
        """Open HSV filter tuner with 100x100 region around the ball."""
        x, y = point
        
        # Extract 100x100 region around the ball - MUST BE FRESH COPY
        region_size = 50  # 50 pixels radius = 100x100 total
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(frame.shape[1], x + region_size)
        y2 = min(frame.shape[0], y + region_size)
        
        print(f"\n=== EXTRACTING NEW REGION ===")
        print(f"Click position: ({x}, {y})")
        print(f"Extracting region: ({x1},{y1}) to ({x2},{y2}) from frame {frame.shape}")
        
        # IMPORTANT: Extract fresh region from the current frame
        region = frame[y1:y2, x1:x2].copy()  # .copy() ensures fresh data
        if region.size == 0:
            print("Error: Cannot extract 100x100 region around ball")
            return False
        
        print(f"Extracted region shape: {region.shape}")
        
        # Resize to exactly 100x100 if needed
        if region.shape[0] != 100 or region.shape[1] != 100:
            region = cv2.resize(region, (100, 100))
            print(f"Resized region to: {region.shape}")
        
        # Get initial HSV values at the clicked point (relative to region)
        rel_x = x - x1
        rel_y = y - y1
        
        # Adjust relative coordinates if resizing happened
        if x2 - x1 != 100:
            rel_x = int((rel_x / (x2 - x1)) * 100)
        if y2 - y1 != 100:
            rel_y = int((rel_y / (y2 - y1)) * 100)
        
        # Ensure relative coordinates are within bounds
        rel_x = max(0, min(99, rel_x))
        rel_y = max(0, min(99, rel_y))
        
        print(f"Relative coordinates: ({rel_x}, {rel_y}) in 100x100 region")
        
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        initial_hsv = hsv_region[rel_y, rel_x]
        
        print(f"Ball position: ({x}, {y})")
        print(f"Initial HSV: H={initial_hsv[0]}, S={initial_hsv[1]}, V={initial_hsv[2]}")
        print("Adjust sliders until ball appears WHITE, then press 'S' to save")
        
        # Create HSV tuner window
        tuner_window = "HSV Filter Tuner - 100x100"
        cv2.namedWindow(tuner_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tuner_window, 800, 600)
        cv2.waitKey(1)  # Allow window to be created
        
        # Use good starting HSV values from user's findings
        h_min = 20
        h_max = 90
        s_min = 20
        s_max = 255
        v_min = 70
        v_max = 255
        
        # Bulb size filter (2-10 pixels initially)
        bulb_min = 2
        bulb_max = 10
        
        # Create trackbars for HSV and bulb size
        cv2.createTrackbar('H Min', tuner_window, h_min, 179, lambda x: None)
        cv2.createTrackbar('H Max', tuner_window, h_max, 179, lambda x: None)
        cv2.createTrackbar('S Min', tuner_window, s_min, 255, lambda x: None)
        cv2.createTrackbar('S Max', tuner_window, s_max, 255, lambda x: None)
        cv2.createTrackbar('V Min', tuner_window, v_min, 255, lambda x: None)
        cv2.createTrackbar('V Max', tuner_window, v_max, 255, lambda x: None)
        cv2.createTrackbar('Bulb Min', tuner_window, bulb_min, 100, lambda x: None)
        cv2.createTrackbar('Bulb Max', tuner_window, bulb_max, 100, lambda x: None)
        cv2.waitKey(1)  # Allow trackbars to be created
        
        def update_display():
            # Get current trackbar values
            h_min = cv2.getTrackbarPos('H Min', tuner_window)
            h_max = cv2.getTrackbarPos('H Max', tuner_window)
            s_min = cv2.getTrackbarPos('S Min', tuner_window)
            s_max = cv2.getTrackbarPos('S Max', tuner_window)
            v_min = cv2.getTrackbarPos('V Min', tuner_window)
            v_max = cv2.getTrackbarPos('V Max', tuner_window)
            bulb_min = cv2.getTrackbarPos('Bulb Min', tuner_window)
            bulb_max = cv2.getTrackbarPos('Bulb Max', tuner_window)
            
            # Apply HSV filter to the fresh region (100x100)
            hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask_region = cv2.inRange(hsv_region, hsv_lower, hsv_upper)
            
            # Clean up mask for region
            kernel = np.ones((2, 2), np.uint8)
            mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_OPEN, kernel)
            mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, kernel)
            
            # Apply HSV filter to FULL FRAME and find all bulbs
            hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_full = cv2.inRange(hsv_full, hsv_lower, hsv_upper)
            mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)
            mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in full frame
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw detected bulbs on full frame
            full_frame_display = frame.copy()
            detected_bulbs = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if bulb_min <= area <= bulb_max:
                    # Draw green circle around detected bulb
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(full_frame_display, (cx, cy), 5, (0, 255, 0), 1)
                        cv2.putText(full_frame_display, f"{int(area)}", (cx+7, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                        detected_bulbs += 1
            
            # Show full frame with detected bulbs
            cv2.namedWindow("Detected Bulbs", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected Bulbs", 800, 450)
            cv2.imshow("Detected Bulbs", full_frame_display)
            
            # Create side-by-side display for 100x100 region
            display = np.zeros((120, 220, 3), dtype=np.uint8)
            display[10:110, 10:110] = region  # Original on left
            display[10:110, 120:220] = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR)  # Filter on right
            
            # Draw borders
            cv2.rectangle(display, (10, 10), (109, 109), (255, 255, 255), 1)
            cv2.rectangle(display, (120, 10), (219, 109), (255, 255, 255), 1)
            
            # Show coordinates with small + marker
            cv2.line(display, (rel_x + 8, rel_y + 10), (rel_x + 12, rel_y + 10), (0, 255, 0), 1)
            cv2.line(display, (rel_x + 10, rel_y + 8), (rel_x + 10, rel_y + 12), (0, 255, 0), 1)
            
            # Show info
            bulb_size_region = np.sum(mask_region > 0)
            cv2.putText(display, f"Size: {bulb_size_region}px", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"Found: {detected_bulbs}", (120, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow(tuner_window, display)
            return hsv_lower, hsv_upper, bulb_size_region
        
        # Initial display
        update_display()
        
        print("Controls:")
        print("- Adjust sliders to make ball WHITE")
        print("- Press 'S' to SAVE and continue")
        print("- Press 'Q' to CANCEL")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save HSV values
                hsv_lower, hsv_upper, bulb_size = update_display()
                self.hsv_lower = hsv_lower
                self.hsv_upper = hsv_upper
                self.ball_center = point
                self.ball_size = bulb_size
                self.tracking = True
                
                print(f"\nHSV Filter SAVED!")
                print(f"HSV Range: H={hsv_lower[0]}-{hsv_upper[0]}, S={hsv_lower[1]}-{hsv_upper[1]}, V={hsv_lower[2]}-{hsv_upper[2]}")
                print(f"Bulb size: {bulb_size} pixels")
                
                # Add to HSV table
                self.hsv_table.append({
                    'frame': self.frame_count,
                    'position': (x, y),
                    'hsv': initial_hsv.tolist(),
                    'bulb_size': bulb_size,
                    'hsv_range': [hsv_lower[0], hsv_upper[0], hsv_lower[1], hsv_upper[1], hsv_lower[2], hsv_upper[2]]
                })
                
                cv2.destroyWindow(tuner_window)
                cv2.destroyWindow("Detected Bulbs")
                return True
            elif key == ord('q'):
                print("HSV tuning cancelled")
                cv2.destroyWindow(tuner_window)
                cv2.destroyWindow("Detected Bulbs")
                return False
            else:
                # Update display
                update_display()
        
        return False
    
    def analyze_ball_at_point(self, frame, point):
        """Analyze ball at the clicked point - simple version without HSV tuner."""
        x, y = point
        
        # Store the initial marked position to return to after ball stops
        self.initial_ball_position = point
        
        # Extract small region around clicked point to calculate bulb size
        region_size = 10
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(frame.shape[1], x + region_size)
        y2 = min(frame.shape[0], y + region_size)
        
        region = frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_region, self.hsv_lower, self.hsv_upper)
        
        # Calculate bulb size
        bulb_size = np.sum(mask > 0)
        
        # Get HSV values at clicked point
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_values = hsv_frame[y, x]
        
        # Save analysis results
        self.ball_center = point
        self.ball_hsv = hsv_values
        self.ball_size = bulb_size
        self.tracking = True
        self.ball_stopped = False
        self.ball_velocity_history = []  # Track recent velocities to detect stop
        
        # Track if this is a recovery from a lost ball
        if hasattr(self, 'ball_center') and self.ball_center is not None and self.tracking:
            prev_pos = self.ball_center
            print(f"\n{'='*70}")
            print(f"=== BALL TRACKING LOST - MANUAL RECOVERY NEEDED ===")
            print(f"{'='*70}")
            print(f"Previous ball position: {prev_pos}")
            print(f"New marked position: ({x}, {y})")
            print(f"Distance jumped: {np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2):.1f}px")
            print(f"\n>>> ANALYZING WHY BALL WAS LOST <<<")
            
            # Check if HSV of new ball is within our filter range
            print(f"\nHSV Analysis:")
            print(f"  Current HSV filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
            print(f"  Ball HSV at new position: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]}")
            
            # Check each component
            h_in_range = self.hsv_lower[0] <= hsv_values[0] <= self.hsv_upper[0]
            s_in_range = self.hsv_lower[1] <= hsv_values[1] <= self.hsv_upper[1]
            v_in_range = self.hsv_lower[2] <= hsv_values[2] <= self.hsv_upper[2]
            
            print(f"  H in range: {h_in_range} {'[OK]' if h_in_range else '[OUT OF RANGE!]'}")
            print(f"  S in range: {s_in_range} {'[OK]' if s_in_range else '[OUT OF RANGE!]'}")
            print(f"  V in range: {v_in_range} {'[OK]' if v_in_range else '[OUT OF RANGE!]'}")
            
            if not (h_in_range and s_in_range and v_in_range):
                print(f"\n  *** ROOT CAUSE: Ball HSV is OUTSIDE filter range! ***")
                print(f"  *** SOLUTION: Need to widen HSV filter to include this ball ***")
            else:
                print(f"\n  *** Ball HSV is WITHIN filter range - other issue ***")
                print(f"  *** Possible causes: size changed, moved too fast, or occluded ***")
            
            print(f"{'='*70}")
            
            # Show interactive HSV filter tuner with bulb sizes
            self._open_interactive_hsv_tuner(frame, prev_pos)
        else:
            print(f"\n=== BALL MARKED AT ({x}, {y}) ===")
        
        print(f"HSV at point: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]}")
        print(f"HSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]} (Tennis ball optimized)")
        print(f"Bulb size: {bulb_size} pixels")
        print(f"Initial position saved: ({x}, {y})")
        print("Ball tracking enabled! Press 'D' to advance to next frame")
        
        # Add to table
        self.hsv_table.append({
            'frame': self.frame_count,
            'position': (x, y),
            'hsv': hsv_values.tolist(),
            'bulb_size': bulb_size,
            'hsv_range': [self.hsv_lower[0], self.hsv_upper[0], self.hsv_lower[1], self.hsv_upper[1], self.hsv_lower[2], self.hsv_upper[2]]
        })
        
        return True
    
    def _open_interactive_hsv_tuner(self, frame, search_center):
        """Open interactive HSV filter tuner with bulb size controls."""
        print(f"\n>>> OPENING DUAL HSV FILTER TUNER <<<")
        print(f"Showing search region around last known position: {search_center}")
        
        # Check if dual HSV mode is enabled
        is_dual_mode = hasattr(self, 'hsv_regular') and hasattr(self, 'hsv_behind_net')
        
        if is_dual_mode:
            print(f"Opening TWO windows:")
            print(f"  1. REGULAR COURT HSV (left window)")
            print(f"  2. BEHIND NET HSV (right window)")
            print(f"Adjust the appropriate window based on ball location")
        else:
            print(f"Adjust HSV min/max and Bulb size min/max sliders")
        
        print(f"Press 'Q' or ESC to close and continue")
        
        # Get search region (120px radius)
        search_radius = 120
        x_prev, y_prev = search_center
        x1 = max(0, x_prev - search_radius)
        y1 = max(0, y_prev - search_radius)
        x2 = min(frame.shape[1], x_prev + search_radius)
        y2 = min(frame.shape[0], y_prev + search_radius)
        
        search_frame = frame[y1:y2, x1:x2].copy()
        
        if is_dual_mode:
            # Create TWO windows for dual HSV mode
            window_regular = "1. REGULAR COURT HSV"
            window_behind_net = "2. BEHIND NET HSV"
            
            cv2.namedWindow(window_regular)
            cv2.namedWindow(window_behind_net)
            cv2.resizeWindow(window_regular, 800, 600)
            cv2.resizeWindow(window_behind_net, 800, 600)
            cv2.moveWindow(window_regular, 50, 50)
            cv2.moveWindow(window_behind_net, 900, 50)
            cv2.waitKey(1)
            
            # Regular court HSV values
            h_min_reg = self.hsv_regular['lower'][0]
            h_max_reg = self.hsv_regular['upper'][0]
            s_min_reg = self.hsv_regular['lower'][1]
            s_max_reg = self.hsv_regular['upper'][1]
            v_min_reg = self.hsv_regular['lower'][2]
            v_max_reg = self.hsv_regular['upper'][2]
            
            # Behind net HSV values
            h_min_net = self.hsv_behind_net['lower'][0]
            h_max_net = self.hsv_behind_net['upper'][0]
            s_min_net = self.hsv_behind_net['lower'][1]
            s_max_net = self.hsv_behind_net['upper'][1]
            v_min_net = self.hsv_behind_net['lower'][2]
            v_max_net = self.hsv_behind_net['upper'][2]
            
            bulb_min = 3
            bulb_max = 80
            
            # Create trackbars for REGULAR COURT window
            cv2.createTrackbar("H Min", window_regular, h_min_reg, 179, lambda x: None)
            cv2.createTrackbar("H Max", window_regular, h_max_reg, 179, lambda x: None)
            cv2.createTrackbar("S Min", window_regular, s_min_reg, 255, lambda x: None)
            cv2.createTrackbar("S Max", window_regular, s_max_reg, 255, lambda x: None)
            cv2.createTrackbar("V Min", window_regular, v_min_reg, 255, lambda x: None)
            cv2.createTrackbar("V Max", window_regular, v_max_reg, 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", window_regular, bulb_min, 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", window_regular, bulb_max, 200, lambda x: None)
            
            # Create trackbars for BEHIND NET window
            cv2.createTrackbar("H Min", window_behind_net, h_min_net, 179, lambda x: None)
            cv2.createTrackbar("H Max", window_behind_net, h_max_net, 179, lambda x: None)
            cv2.createTrackbar("S Min", window_behind_net, s_min_net, 255, lambda x: None)
            cv2.createTrackbar("S Max", window_behind_net, s_max_net, 255, lambda x: None)
            cv2.createTrackbar("V Min", window_behind_net, v_min_net, 255, lambda x: None)
            cv2.createTrackbar("V Max", window_behind_net, v_max_net, 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", window_behind_net, bulb_min, 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", window_behind_net, bulb_max, 200, lambda x: None)
            cv2.waitKey(1)
            
            while True:
                # Get REGULAR COURT trackbar values
                h_min_reg = cv2.getTrackbarPos("H Min", window_regular)
                h_max_reg = cv2.getTrackbarPos("H Max", window_regular)
                s_min_reg = cv2.getTrackbarPos("S Min", window_regular)
                s_max_reg = cv2.getTrackbarPos("S Max", window_regular)
                v_min_reg = cv2.getTrackbarPos("V Min", window_regular)
                v_max_reg = cv2.getTrackbarPos("V Max", window_regular)
                bulb_min_reg = cv2.getTrackbarPos("Bulb Min", window_regular)
                bulb_max_reg = cv2.getTrackbarPos("Bulb Max", window_regular)
                
                # Get BEHIND NET trackbar values
                h_min_net = cv2.getTrackbarPos("H Min", window_behind_net)
                h_max_net = cv2.getTrackbarPos("H Max", window_behind_net)
                s_min_net = cv2.getTrackbarPos("S Min", window_behind_net)
                s_max_net = cv2.getTrackbarPos("S Max", window_behind_net)
                v_min_net = cv2.getTrackbarPos("V Min", window_behind_net)
                v_max_net = cv2.getTrackbarPos("V Max", window_behind_net)
                bulb_min_net = cv2.getTrackbarPos("Bulb Min", window_behind_net)
                bulb_max_net = cv2.getTrackbarPos("Bulb Max", window_behind_net)
                
                # Apply REGULAR COURT HSV filter
                hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
                hsv_lower_reg = np.array([h_min_reg, s_min_reg, v_min_reg], dtype=np.uint8)
                hsv_upper_reg = np.array([h_max_reg, s_max_reg, v_max_reg], dtype=np.uint8)
                mask_reg = cv2.inRange(hsv_frame, hsv_lower_reg, hsv_upper_reg)
                contours_reg, _ = cv2.findContours(mask_reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                display_reg = search_frame.copy()
                bulb_count_reg = 0
                for contour in contours_reg:
                    area = cv2.contourArea(contour)
                    if area < bulb_min_reg or area > bulb_max_reg:
                        continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_reg, (cx, cy), 8, (0, 255, 0), 2)
                    cv2.putText(display_reg, f"{area:.0f}", (cx + 12, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    bulb_count_reg += 1
                
                mask_reg_bgr = cv2.cvtColor(mask_reg, cv2.COLOR_GRAY2BGR)
                combined_reg = np.hstack([display_reg, mask_reg_bgr])
                info_reg = f"REGULAR COURT | Bulbs: {bulb_count_reg} | H={h_min_reg}-{h_max_reg} S={s_min_reg}-{s_max_reg} V={v_min_reg}-{v_max_reg}"
                cv2.putText(combined_reg, info_reg, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_regular, combined_reg)
                
                # Apply BEHIND NET HSV filter
                hsv_lower_net = np.array([h_min_net, s_min_net, v_min_net], dtype=np.uint8)
                hsv_upper_net = np.array([h_max_net, s_max_net, v_max_net], dtype=np.uint8)
                mask_net = cv2.inRange(hsv_frame, hsv_lower_net, hsv_upper_net)
                contours_net, _ = cv2.findContours(mask_net, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                display_net = search_frame.copy()
                bulb_count_net = 0
                for contour in contours_net:
                    area = cv2.contourArea(contour)
                    if area < bulb_min_net or area > bulb_max_net:
                        continue
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(display_net, (cx, cy), 8, (255, 0, 0), 2)
                    cv2.putText(display_net, f"{area:.0f}", (cx + 12, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    bulb_count_net += 1
                
                mask_net_bgr = cv2.cvtColor(mask_net, cv2.COLOR_GRAY2BGR)
                combined_net = np.hstack([display_net, mask_net_bgr])
                info_net = f"BEHIND NET | Bulbs: {bulb_count_net} | H={h_min_net}-{h_max_net} S={s_min_net}-{s_max_net} V={v_min_net}-{v_max_net}"
                cv2.putText(combined_net, info_net, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(window_behind_net, combined_net)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
            
            # Close both windows
            cv2.destroyWindow(window_regular)
            cv2.destroyWindow(window_behind_net)
            cv2.waitKey(1)
            
            # Update both HSV configs
            self.hsv_regular['lower'] = hsv_lower_reg
            self.hsv_regular['upper'] = hsv_upper_reg
            self.hsv_behind_net['lower'] = hsv_lower_net
            self.hsv_behind_net['upper'] = hsv_upper_net
            
            # Determine which HSV to use as active based on ball position
            y_pos = search_center[1]
            is_in_net_area = (hasattr(self, 'net_area_y_min') and 
                             hasattr(self, 'net_area_y_max') and
                             self.net_area_y_min <= y_pos <= self.net_area_y_max)
            
            if is_in_net_area:
                self.hsv_lower = hsv_lower_net
                self.hsv_upper = hsv_upper_net
                h_min, h_max = h_min_net, h_max_net
                s_min, s_max = s_min_net, s_max_net
                v_min, v_max = v_min_net, v_max_net
                bulb_min, bulb_max = bulb_min_net, bulb_max_net
            else:
                self.hsv_lower = hsv_lower_reg
                self.hsv_upper = hsv_upper_reg
                h_min, h_max = h_min_reg, h_max_reg
                s_min, s_max = s_min_reg, s_max_reg
                v_min, v_max = v_min_reg, v_max_reg
                bulb_min, bulb_max = bulb_min_reg, bulb_max_reg
            
        else:
            # Single HSV mode (original behavior)
            tuner_window = "HSV Filter Tuner"
            cv2.namedWindow(tuner_window)
            cv2.resizeWindow(tuner_window, 1200, 600)
            cv2.waitKey(1)
            
            # Current HSV values
            h_min = self.hsv_lower[0]
            h_max = self.hsv_upper[0]
            s_min = self.hsv_lower[1]
            s_max = self.hsv_upper[1]
            v_min = self.hsv_lower[2]
            v_max = self.hsv_upper[2]
            bulb_min = 3
            bulb_max = 80
            
            # Create trackbars
            cv2.createTrackbar("H Min", tuner_window, h_min, 179, lambda x: None)
            cv2.createTrackbar("H Max", tuner_window, h_max, 179, lambda x: None)
            cv2.createTrackbar("S Min", tuner_window, s_min, 255, lambda x: None)
            cv2.createTrackbar("S Max", tuner_window, s_max, 255, lambda x: None)
            cv2.createTrackbar("V Min", tuner_window, v_min, 255, lambda x: None)
            cv2.createTrackbar("V Max", tuner_window, v_max, 255, lambda x: None)
            cv2.createTrackbar("Bulb Min", tuner_window, bulb_min, 100, lambda x: None)
            cv2.createTrackbar("Bulb Max", tuner_window, bulb_max, 200, lambda x: None)
            cv2.waitKey(1)
            
            while True:
                # Get current trackbar values
                h_min = cv2.getTrackbarPos("H Min", tuner_window)
                h_max = cv2.getTrackbarPos("H Max", tuner_window)
                s_min = cv2.getTrackbarPos("S Min", tuner_window)
                s_max = cv2.getTrackbarPos("S Max", tuner_window)
                v_min = cv2.getTrackbarPos("V Min", tuner_window)
                v_max = cv2.getTrackbarPos("V Max", tuner_window)
                bulb_min = cv2.getTrackbarPos("Bulb Min", tuner_window)
                bulb_max = cv2.getTrackbarPos("Bulb Max", tuner_window)
                
                # Apply HSV filter
                hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
                hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
                hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
                mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw detected bulbs on original frame
                display_frame = search_frame.copy()
                bulb_count = 0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < bulb_min or area > bulb_max:
                        continue
                        
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw circle and size label
                    cv2.circle(display_frame, (cx, cy), 8, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{area:.0f}", (cx + 12, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    bulb_count += 1
                
                # Create side-by-side display
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([display_frame, mask_bgr])
                
                # Add text info
                info_text = f"Bulbs detected: {bulb_count} | HSV: H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max} | Size: {bulb_min}-{bulb_max}px"
                cv2.putText(combined, info_text, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow(tuner_window, combined)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
            
            # Update HSV values with tuned values
            self.hsv_lower = hsv_lower
            self.hsv_upper = hsv_upper
            
            # Close tuner window
            try:
                cv2.destroyWindow(tuner_window)
                cv2.waitKey(1)
            except:
                pass
        
        # Save to config file for persistence
        # Load existing config to preserve structure
        try:
            with open('hsv_config.json', 'r') as f:
                hsv_config = json.load(f)
        except:
            hsv_config = {}
        
        # Update the appropriate section
        if is_dual_mode:
            # In dual mode, save BOTH configurations
            print(f"Saving BOTH HSV configurations:")
            print(f"  Regular Court: H={h_min_reg}-{h_max_reg}, S={s_min_reg}-{s_max_reg}, V={v_min_reg}-{v_max_reg}")
            print(f"  Behind Net: H={h_min_net}-{h_max_net}, S={s_min_net}-{s_max_net}, V={v_min_net}-{v_max_net}")
            
            hsv_config['regular_court'] = {
                'h_min': int(h_min_reg),
                'h_max': int(h_max_reg),
                's_min': int(s_min_reg),
                's_max': int(s_max_reg),
                'v_min': int(v_min_reg),
                'v_max': int(v_max_reg)
            }
            hsv_config['behind_net'] = {
                'h_min': int(h_min_net),
                'h_max': int(h_max_net),
                's_min': int(s_min_net),
                's_max': int(s_max_net),
                'v_min': int(v_min_net),
                'v_max': int(v_max_net)
            }
        else:
            # Single HSV mode - update entire config
            hsv_config = {
                'h_min': int(h_min),
                'h_max': int(h_max),
                's_min': int(s_min),
                's_max': int(s_max),
                'v_min': int(v_min),
                'v_max': int(v_max)
            }
        
        with open('hsv_config.json', 'w') as f:
            json.dump(hsv_config, f, indent=4)
        
        # Close tuner window(s) - already closed in dual mode
        if not is_dual_mode:
            try:
                cv2.destroyWindow(tuner_window)
                cv2.waitKey(1)
            except:
                pass
        
        print(f"Tuner closed. Updated HSV: H={h_min}-{h_max}, S={s_min}-{s_max}, V={v_min}-{v_max}")
        print(f"HSV values saved to hsv_config.json")
        print(f"Bulb size filter: {bulb_min}-{bulb_max}px\n")
    
    def track_ball_in_frame(self, frame):
        """Track ball in current frame using HSV filter with debug information."""
        if not self.tracking or self.hsv_lower is None:
            return None
        
        frame_height, frame_width = frame.shape[:2]
        
        # Store the last search position for when ball is lost
        if not hasattr(self, 'last_search_position'):
            self.last_search_position = None
        
        # Check if ball was near edge in previous frame
        if self.ball_center:
            x_prev, y_prev = self.ball_center
            self.last_search_position = (x_prev, y_prev)  # Remember this position
            edge_margin = 50  # pixels from edge (reduced from 100 to avoid false triggers)
            
            # Check if ball went off top edge
            if y_prev < edge_margin:
                print(f"\n  DEBUG: Ball near TOP edge (y={y_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at top edge for ball to return...")
                # Search along top edge
                x, y = x_prev, 10
                search_radius = 200  # Very wide search along edge for fast balls
            # Check if ball went off bottom edge
            elif y_prev > frame_height - edge_margin:
                print(f"\n  DEBUG: Ball near BOTTOM edge (y={y_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at bottom edge for ball to return...")
                x, y = x_prev, frame_height - 10
                search_radius = 200
            # Check if ball went off left edge
            elif x_prev < edge_margin:
                print(f"\n  DEBUG: Ball near LEFT edge (x={x_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at left edge for ball to return...")
                x, y = 10, y_prev
                search_radius = 200
            # Check if ball went off right edge
            elif x_prev > frame_width - edge_margin:
                print(f"\n  DEBUG: Ball near RIGHT edge (x={x_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at right edge for ball to return...")
                x, y = frame_width - 10, y_prev
                search_radius = 200
            else:
                # Normal tracking - increased radius to catch fast balls
                x, y = x_prev, y_prev
                search_radius = 120  # Increased from 80 to catch balls moving >80px/frame
        # First, check if ball stopped - if so, search from initial position
        elif hasattr(self, 'ball_stopped') and self.ball_stopped and hasattr(self, 'initial_ball_position'):
            print(f"\n  DEBUG: Ball stopped! Searching from initial position {self.initial_ball_position}")
            x, y = self.initial_ball_position
            search_radius = 80  # Larger radius for initial search
        else:
            search_frame = frame
            x1, y1 = 0, 0
            print(f"\n  DEBUG: No previous ball position, searching entire frame")
        
        # Search in region around the search position
        if self.ball_center or (hasattr(self, 'initial_ball_position') and self.initial_ball_position):
            x1 = max(0, x - search_radius)
            y1 = max(0, y - search_radius)
            x2 = min(frame.shape[1], x + search_radius)
            y2 = min(frame.shape[0], y + search_radius)
            
            search_frame = frame[y1:y2, x1:x2]
            # Determine which HSV config will be used
            _, _, hsv_mode_check = self.select_hsv_for_position(y)
            print(f"\n  DEBUG: Searching at ({x},{y}) with radius {search_radius}px, region: ({x1},{y1})-({x2},{y2}), HSV mode: {hsv_mode_check}")
        
        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        
        # Check if we should search with BOTH HSV filters
        # This happens when ball is approaching the net area (within 100px margin)
        is_dual_mode = hasattr(self, 'hsv_regular') and hasattr(self, 'hsv_behind_net')
        net_approach_margin = 100  # pixels before/after net area to check both filters
        
        should_check_both = False
        if is_dual_mode and hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            # Check if ball is approaching, in, or leaving net area
            if (self.net_area_y_min - net_approach_margin) <= y <= (self.net_area_y_max + net_approach_margin):
                should_check_both = True
        
        if should_check_both:
            # Search with BOTH filters and combine results
            print(f"  DEBUG: Ball near net area (Y={y}, net Y={self.net_area_y_min}-{self.net_area_y_max})")
            print(f"  DEBUG: Checking BOTH HSV filters to find best match")
            
            # Get contours from REGULAR COURT filter
            mask_reg = cv2.inRange(hsv_frame, self.hsv_regular['lower'], self.hsv_regular['upper'])
            kernel = np.ones((2, 2), np.uint8)
            mask_reg = cv2.morphologyEx(mask_reg, cv2.MORPH_OPEN, kernel)
            mask_reg = cv2.morphologyEx(mask_reg, cv2.MORPH_CLOSE, kernel)
            contours_reg, _ = cv2.findContours(mask_reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get contours from BEHIND NET filter
            mask_net = cv2.inRange(hsv_frame, self.hsv_behind_net['lower'], self.hsv_behind_net['upper'])
            mask_net = cv2.morphologyEx(mask_net, cv2.MORPH_OPEN, kernel)
            mask_net = cv2.morphologyEx(mask_net, cv2.MORPH_CLOSE, kernel)
            contours_net, _ = cv2.findContours(mask_net, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Tag each contour with its source
            contours = []
            for c in contours_reg:
                contours.append(('regular', c))
            for c in contours_net:
                contours.append(('behind_net', c))
            
            print(f"  DEBUG: Found {len(contours_reg)} regular court + {len(contours_net)} behind net = {len(contours)} total contours")
        else:
            # Normal single HSV filter search
            hsv_lower_use, hsv_upper_use, hsv_mode = self.select_hsv_for_position(y)
            
            # Apply HSV filter
            mask = cv2.inRange(hsv_frame, hsv_lower_use, hsv_upper_use)
            
            # Clean up
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [('single', c) for c in contours_raw]
            
            print(f"  DEBUG: Found {len(contours)} total contours in search region (mode: {hsv_mode})")
        
        if not contours:
            print("  DEBUG: [PROBLEM] No contours found in search region!")
            print(f"  DEBUG: HSV filter range ({hsv_mode}): H={hsv_lower_use[0]}-{hsv_upper_use[0]}, S={hsv_lower_use[1]}-{hsv_upper_use[1]}, V={hsv_lower_use[2]}-{hsv_upper_use[2]}")
            print(f"  DEBUG: Search region: ({x1},{y1})-({x2},{y2}), size: {x2-x1}x{y2-y1}px")
            if self.ball_center:
                print(f"  DEBUG: Previous ball position: {self.ball_center}")
                print(f"  DEBUG: KEEPING marker at last known position: {self.ball_center}")
            print(f"  DEBUG: REASON: Ball may have:")
            print(f"  DEBUG:   - Gone off screen (check edge detection)")
            print(f"  DEBUG:   - Changed color/lighting dramatically")
            print(f"  DEBUG:   - Moved faster than {search_radius}px/frame")
            print(f"  DEBUG:   - Be occluded by player/net")
            print(f"  DEBUG: Will continue searching in next frame at same position...")
            
            # Keep the ball_center at last position instead of losing it
            return self.ball_center
        
        # Find the best ball candidate using weighted score (distance + size similarity)
        best_contour = None
        best_score = float('inf')
        best_source = None
        candidates = []
        
        for i, (source, contour) in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Reject huge background regions first (> 500px is definitely not a ball)
            if area > 500:
                if i < 3:  # Only print first few to avoid spam
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (background region)")
                continue
            
            # More lenient size filter (3-80 pixels for tennis ball)
            # Ball can be very small when far away or large when close
            if area < 3 or area > 80:
                print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (outside 3-80)")
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            # Calculate position in original frame coordinates
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            
            # Calculate distance from previous position
            if self.ball_center:
                distance = np.sqrt((cx - self.ball_center[0])**2 + (cy - self.ball_center[1])**2)
            else:
                distance = 0
            
            # Calculate size difference (penalty for very different sizes)
            if self.ball_size and self.ball_size > 0:
                size_diff = abs(area - self.ball_size)
                size_ratio = size_diff / self.ball_size
            else:
                size_ratio = 0
            
            # Weighted score: distance + size penalty
            # Prefer candidates with similar size to previous ball
            # Distance is primary, but size consistency matters for fast-moving balls
            score = distance + (size_ratio * 30)  # 30px penalty per 100% size change
            
            candidates.append((i, source, cx, cy, area, distance, size_ratio, score))
            source_label = f"[{source}]" if source != 'single' else ""
            print(f"  DEBUG: Contour {i} {source_label} CANDIDATE - pos=({cx},{cy}), area={area:.1f}px, distance={distance:.1f}px, size_ratio={size_ratio:.2f}, score={score:.1f}")
            
            if score < best_score:
                best_score = score
                best_contour = contour
                best_source = source
        
        if best_contour is not None:
            # Update ball position
            M = cv2.moments(best_contour)
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            
            # Check if this is likely a false positive jump
            # If ball was at edge and closest match is far away, ball likely went off-screen
            if self.ball_center:
                x_prev, y_prev = self.ball_center
                actual_distance = np.sqrt((cx - x_prev)**2 + (cy - y_prev)**2)
                edge_threshold = 5  # pixels from edge
                jump_threshold = 50  # pixels - suspicious if ball "moves" more than this
                
                frame_height, frame_width = frame.shape[:2]
                at_edge = (y_prev < edge_threshold or y_prev > frame_height - edge_threshold or
                          x_prev < edge_threshold or x_prev > frame_width - edge_threshold)
                
                if at_edge and actual_distance > jump_threshold:
                    print(f"  DEBUG: [REJECTED] Candidate at ({cx},{cy}), distance={actual_distance:.1f}px")
                    print(f"  DEBUG: Ball was at edge (y={y_prev}), closest match is {actual_distance:.1f}px away")
                    print(f"  DEBUG: This is likely a FALSE POSITIVE - ball probably went off-screen")
                    print(f"  DEBUG: KEEPING marker at last edge position: {self.ball_center}")
                    print(f"  DEBUG: Will wait for ball to return...")
                    return self.ball_center
            
            source_label = f" from [{best_source}] filter" if best_source and best_source != 'single' else ""
            print(f"  DEBUG: [SELECTED] contour at ({cx},{cy}), score={best_score:.1f}{source_label}")
            
            # Get HSV values at new position
            hsv_values = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[cy, cx]
            bulb_size = cv2.contourArea(best_contour)
            
            # Calculate velocity (distance moved)
            if hasattr(self, 'ball_center') and self.ball_center:
                prev_x, prev_y = self.ball_center
                velocity = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
            else:
                velocity = 0
            
            # Update tracking data
            self.ball_center = (cx, cy)
            self.ball_hsv = hsv_values
            self.ball_size = bulb_size
            
            # Track velocity history (last 5 frames)
            if not hasattr(self, 'ball_velocity_history'):
                self.ball_velocity_history = []
            self.ball_velocity_history.append(velocity)
            if len(self.ball_velocity_history) > 5:
                self.ball_velocity_history.pop(0)
            
            # Detect if ball stopped (average velocity < 2 pixels/frame for 5 frames)
            if len(self.ball_velocity_history) >= 5:
                avg_velocity = sum(self.ball_velocity_history) / len(self.ball_velocity_history)
                if avg_velocity < 2.0:
                    if not hasattr(self, 'ball_stopped') or not self.ball_stopped:
                        print(f"  DEBUG: [BALL STOPPED] Avg velocity: {avg_velocity:.1f}px/frame")
                        print(f"  DEBUG: Will search from initial position {self.initial_ball_position} for next ball")
                        self.ball_stopped = True
                else:
                    self.ball_stopped = False
            
            # Add to HSV table
            self.hsv_table.append({
                'frame': self.frame_count,
                'position': (cx, cy),
                'hsv': hsv_values.tolist(),
                'bulb_size': bulb_size,
                'velocity': velocity,
                'hsv_range': [self.hsv_lower[0], self.hsv_upper[0], 
                             self.hsv_lower[1], self.hsv_upper[1], 
                             self.hsv_lower[2], self.hsv_upper[2]]
            })
            
            print(f"Frame {self.frame_count}: Ball at ({cx}, {cy}) - HSV: H={hsv_values[0]}, S={hsv_values[1]}, V={hsv_values[2]} - Size: {bulb_size:.1f}px - Velocity: {velocity:.1f}px/frame")
            
            return (cx, cy)
        
        print(f"  DEBUG: [PROBLEM] No valid candidate found!")
        print(f"  DEBUG: Total contours: {len(contours)}, Valid candidates: {len(candidates)}")
        if len(contours) > 0 and len(candidates) == 0:
            print(f"  DEBUG: All {len(contours)} contours were rejected by size filter (3-80px)")
            # Show the actual sizes that were rejected
            rejected_sizes = []
            for source, contour in contours[:5]:  # Show first 5
                rejected_sizes.append(f"{cv2.contourArea(contour):.1f}px")
            print(f"  DEBUG: Rejected sizes (first 5): {', '.join(rejected_sizes)}")
            print(f"  DEBUG: REASON: Ball size changed outside 3-80px range")
            print(f"  DEBUG:   - Ball may be too small (far away) or too large (very close)")
            print(f"  DEBUG:   - Consider adjusting size filter if ball is visible")
        if self.ball_center:
            print(f"  DEBUG: KEEPING marker at last known position: {self.ball_center}")
            print(f"  DEBUG: Will continue searching in next frame at same position...")
        # Keep the ball_center at last position instead of losing it
        return self.ball_center
    
    def draw_analysis_info(self, frame, scale=1.0):
        """Draw analysis information on the frame with proper scaling."""
        result = frame.copy()
        
        if self.ball_center:
            # Scale ball coordinates for display
            x = int(self.ball_center[0] * scale)
            y = int(self.ball_center[1] * scale)
            
            # Draw green circle around the ball
            cv2.circle(result, (x, y), 10, (0, 255, 0), 2)  # Circle outline
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)  # Center dot
            
            # Draw info text at the top
            if self.ball_hsv is not None and self.ball_size is not None:
                # Show current HSV values at clicked point
                hsv_text = f"Ball HSV: H={self.ball_hsv[0]}, S={self.ball_hsv[1]}, V={self.ball_hsv[2]}"
                cv2.putText(result, hsv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show HSV filter range
                filter_text = f"Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}"
                cv2.putText(result, filter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show bulb size
                size_text = f"Bulb Size: {self.ball_size:.0f}px"
                cv2.putText(result, size_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show ball position
                pos_text = f"Ball Pos: ({self.ball_center[0]}, {self.ball_center[1]})"
                cv2.putText(result, pos_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Scale player positions for display
        p1_x = int(self.player1_pos[0] * scale)
        p1_y = int(self.player1_pos[1] * scale)
        p2_x = int(self.player2_pos[0] * scale)
        p2_y = int(self.player2_pos[1] * scale)
        
        # Draw players
        cv2.circle(result, (p1_x, p1_y), 20, (255, 0, 0), 2)
        cv2.putText(result, "P1", (p1_x-10, p1_y-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.circle(result, (p2_x, p2_y), 20, (0, 0, 255), 2)
        cv2.putText(result, "P2", (p2_x-10, p2_y-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result
    
    def print_hsv_table(self):
        """Print the HSV values table."""
        if not self.hsv_table:
            print("No HSV data collected yet.")
            return
        
        print(f"\n=== HSV VALUES TABLE ({len(self.hsv_table)} entries) ===")
        print("Frame | Position | HSV Values | Bulb Size | HSV Range")
        print("-" * 80)
        
        for entry in self.hsv_table:
            hsv_str = f"H={entry['hsv'][0]}, S={entry['hsv'][1]}, V={entry['hsv'][2]}"
            range_str = f"H={entry['hsv_range'][0]}-{entry['hsv_range'][1]}, S={entry['hsv_range'][2]}-{entry['hsv_range'][3]}, V={entry['hsv_range'][4]}-{entry['hsv_range'][5]}"
            print(f"{entry['frame']:5d} | ({entry['position'][0]:3d},{entry['position'][1]:3d}) | {hsv_str} | {entry['bulb_size']:8.1f} | {range_str}")
        
        print("-" * 80)
    
    def process_video(self):
        """Process video with interactive ball analysis."""
        print("Interactive Ball Analyzer")
        print("=" * 50)
        print("1. Click on the tennis ball to mark and analyze it")
        print("2. System will wait for your mark before starting")
        print("3. Press 'D' to advance to next frame")
        print("4. Press 'N' to mark net area (6 points clockwise)")
        print("5. Press 'Q' to quit")
        print("=" * 50)
        
        cv2.namedWindow("Interactive Ball Analyzer", cv2.WINDOW_NORMAL)
        
        # Store scale factor for mouse coordinate conversion
        scale_factor = 1.0
        current_frame = None
        
        # Modified mouse callback that converts resized coordinates to original
        def mouse_callback_with_scale(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert clicked coordinates from resized frame to original frame
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                # Analyze ball at original coordinates
                self.analyze_ball_at_point(current_frame, (orig_x, orig_y))
        
        cv2.setMouseCallback("Interactive Ball Analyzer", mouse_callback_with_scale)
        
        # Skip to frame 144 and auto-start tracking at known ball position
        print(f"\nSkipping to frame 144...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 144)
        self.frame_count = 144
        
        # Auto-mark the first ball at frame 144 (known position from previous runs)
        # This allows tracking to start immediately without mouse click
        first_ball_position = (1977, 320)  # Approximate position of first ball at frame 144
        print(f"\nAuto-starting tracking at first ball position: {first_ball_position}")
        print(f"Press 'D' to advance frame by frame")
        
        # Read first frame and mark the ball automatically
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # Load HSV values from config file (or use defaults if not present)
            hsv_config = self.load_hsv_config()
            
            # Check if net area needs to be marked (first time setup)
            if hsv_config and "net_area_y_min" not in hsv_config:
                print("\n" + "=" * 70)
                print("FIRST TIME SETUP: Please mark the net area")
                print("=" * 70)
                if self.mark_net_area(frame):
                    print("Net area saved! This will be used for all future runs.")
                    # Reload config after marking net area
                    hsv_config = self.load_hsv_config()
                else:
                    print("Net area marking skipped. Using default values.")
            
            # Don't call analyze_ball_at_point during auto-start to avoid opening debug window
            # Just initialize the tracking parameters directly
            self.ball_center = first_ball_position
            self.tracking = True
            self.ball_stopped = False
            self.ball_velocity_history = []
            self.initial_ball_position = first_ball_position
            if not hsv_config:
                # Default values if config not found
                self.hsv_lower = np.array([20, 20, 70], dtype=np.uint8)
                self.hsv_upper = np.array([90, 255, 255], dtype=np.uint8)
            # If old format, set hsv_lower and hsv_upper (already done in load_hsv_config for new format)
            elif 'h_min' in hsv_config:
                self.hsv_lower = np.array([hsv_config['h_min'], hsv_config['s_min'], hsv_config['v_min']], dtype=np.uint8)
                self.hsv_upper = np.array([hsv_config['h_max'], hsv_config['s_max'], hsv_config['v_max']], dtype=np.uint8)
            # New format already loaded in load_hsv_config, just use the regular court as default
            
            print(f"\nBall tracking initialized at: {first_ball_position}")
            print(f"HSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
            
            # Display first frame (resize to fit screen)
            screen_height = 900
            scale_factor = screen_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale_factor)
            new_height = int(frame.shape[0] * scale_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Draw analysis info with scale factor
            display_frame = self.draw_analysis_info(resized_frame, scale=scale_factor)
            cv2.imshow("Interactive Ball Analyzer", display_frame)
            cv2.waitKey(1)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            current_frame = frame  # Store current frame for mouse callback
            
            # If tracking is enabled, automatically track ball in this frame
            if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
                tracked_position = self.track_ball_in_frame(frame)
                if tracked_position:
                    print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {self.ball_size:.1f}px")
                else:
                    print(f"Frame {self.frame_count}: [WARNING] Ball lost - click to re-mark")
            
            # Resize frame to fit screen
            height, width = frame.shape[:2]
            scale_factor = min(1920/width, 1080/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Draw analysis info with scale factor
            display_frame = self.draw_analysis_info(resized_frame, scale=scale_factor)
            
            # Show frame info
            cv2.putText(display_frame, f"Frame: {self.frame_count}/{self.total_frames}", 
                       (10, new_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if not self.tracking:
                cv2.putText(display_frame, "Click on ball to mark it | D=Next | N=Mark net | Q=Quit", 
                           (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Ball marked! D=Next | N=Mark net | T=Table | Q=Quit", 
                           (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Interactive Ball Analyzer", display_frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Mark net area
                print("\n=== MARKING NET AREA ===")
                if self.mark_net_area(frame):
                    print("Net area marked successfully! Updated config file.")
                    # Reload HSV config to get updated net area
                    self.load_hsv_config()
                else:
                    print("Net area marking cancelled.")
                # Redisplay current frame
                continue
            elif key == ord('d'):
                # Advance to next frame and automatically track
                if self.tracking and self.hsv_lower is not None:
                    print(f"\n--- ADVANCING TO FRAME {self.frame_count + 1} ---")
                    print("Automatically searching for ball in next frame...")
                    # Continue to next iteration to read next frame
                    # The ball will be tracked automatically
                    continue
                else:
                    print("\n--- ADVANCING TO FRAME (no tracking) ---")
                    print("No ball marked yet - click on ball first to enable tracking")
                    # Just continue to next frame without tracking
                    continue
            elif key == ord('t'):
                # Show HSV table
                self.print_hsv_table()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nAnalysis complete! Processed {self.frame_count} frames")


if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    analyzer = InteractiveBallAnalyzer(video_path)
    analyzer.process_video()
