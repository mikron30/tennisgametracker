import argparse
import cv2
import numpy as np
import json
import os
from simple_player_detector import SimplePlayerDetector
from typing import Tuple, Optional


class InteractiveBallAnalyzer:
    def __init__(self, video_path: str, start_frame: int = 100):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.start_frame = max(0, start_frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.frame_count = self.start_frame
        self.last_seen_frame = self.start_frame
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
        self.last_seen_frame = None
        self.edge_wait = False
        self.last_delta = None
        self.primary_hsv_lower = None
        self.primary_hsv_upper = None
        self.alt_hsv_lower = None
        self.alt_hsv_upper = None
        
        # HSV filter (will be set based on clicked ball)
        self.hsv_lower = None
        self.hsv_upper = None
        
        # HSV values table for analysis
        self.hsv_table = []
        
        # Player positions (will be detected dynamically)
        self.player1_pos = None
        self.player2_pos = None
        self.p1_bbox = None  # (x, y, w, h)
        self.p2_bbox = None  # (x, y, w, h)
        
        # Initialize simple player detector (focused on right court only)
        self.player_detector = SimplePlayerDetector()
        
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
                    
                    # Load serve area boundaries if available
                    if "serve_area_x_min" in config:
                        self.serve_area_points = config.get('serve_area_points', [])
                        self.serve_area_x_min = config['serve_area_x_min']
                        self.serve_area_x_max = config['serve_area_x_max']
                        self.serve_area_y_min = config['serve_area_y_min']
                        self.serve_area_y_max = config['serve_area_y_max']
                        print(f"  Serve area: X={self.serve_area_x_min}-{self.serve_area_x_max}, Y={self.serve_area_y_min}-{self.serve_area_y_max}")
                else:
                    # Old format (single HSV set)
                    self.hsv_lower = np.array([config["h_min"], config["s_min"], config["v_min"]], dtype=np.uint8)
                    self.hsv_upper = np.array([config["h_max"], config["s_max"], config["v_max"]], dtype=np.uint8)
                    self.hsv_regular = None
                    self.hsv_behind_net = None
                    print(f"Loaded HSV values: H:{config['h_min']}-{config['h_max']}, S:{config['s_min']}-{config['s_max']}, V:{config['v_min']}-{config['v_max']}")
            
                # Set primary/alt HSV ranges (primary = config, alt = capped legacy)
                self.primary_hsv_lower = self.hsv_lower.copy()
                self.primary_hsv_upper = self.hsv_upper.copy()
                self.alt_hsv_lower = self.primary_hsv_lower.copy()
                self.alt_hsv_upper = self.primary_hsv_upper.copy()
                # Legacy cap for alt to keep the older narrower range
                self.alt_hsv_upper[0] = min(self.alt_hsv_upper[0], 73)
                # Use primary as active by default (full config range)
                self.hsv_lower = self.primary_hsv_lower
                self.hsv_upper = self.primary_hsv_upper
                
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
        
        # Bulb size filter (wider to catch first frames)
        bulb_min = 1
        bulb_max = 30
        
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
    
    def track_ball_in_frame(self, frame, allow_inactive=False):
        """Track ball in current frame using HSV filter with debug information."""
        if (not self.tracking and not allow_inactive) or self.hsv_lower is None:
            return None
        
        frame_height, frame_width = frame.shape[:2]
        
        # Store the last search position for when ball is lost
        if not hasattr(self, 'last_search_position'):
            self.last_search_position = None
        
        early_frames = self.frame_count <= (self.start_frame + 10)

        # Check if ball was near edge in previous frame
        self.edge_wait = False

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
                self.edge_wait = True
            # Check if ball went off bottom edge
            elif y_prev > frame_height - edge_margin:
                print(f"\n  DEBUG: Ball near BOTTOM edge (y={y_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at bottom edge for ball to return...")
                x, y = x_prev, frame_height - 10
                search_radius = 200
                self.edge_wait = True
            # Check if ball went off left edge
            elif x_prev < edge_margin:
                print(f"\n  DEBUG: Ball near LEFT edge (x={x_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at left edge for ball to return...")
                x, y = 10, y_prev
                search_radius = 200
                self.edge_wait = True
            # Check if ball went off right edge
            elif x_prev > frame_width - edge_margin:
                print(f"\n  DEBUG: Ball near RIGHT edge (x={x_prev}), may have gone off-screen")
                print(f"  DEBUG: Waiting at right edge for ball to return...")
                x, y = frame_width - 10, y_prev
                search_radius = 200
                self.edge_wait = True
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
            # When tracking is inactive (serve scan), give a hint region around serve area if defined
            if allow_inactive and hasattr(self, 'serve_area_x_min') and hasattr(self, 'serve_area_x_max'):
                # Center search at middle of serve area to reduce false positives
                x = (self.serve_area_x_min + self.serve_area_x_max) // 2
                y = (self.serve_area_y_min + self.serve_area_y_max) // 2
                search_radius = max(self.serve_area_x_max - self.serve_area_x_min,
                                    self.serve_area_y_max - self.serve_area_y_min) // 2
                x1 = max(0, x - search_radius)
                y1 = max(0, y - search_radius)
                x2 = min(frame.shape[1], x + search_radius)
                y2 = min(frame.shape[0], y + search_radius)
                search_frame = frame[y1:y2, x1:x2]
                print(f"\n  DEBUG: Serve-scan hint search around ({x},{y}) radius {search_radius}px region ({x1},{y1})-({x2},{y2})")
            else:
                print(f"\n  DEBUG: No previous ball position, searching entire frame")
        
        # Search in region around the search position
        if self.ball_center or (hasattr(self, 'initial_ball_position') and self.initial_ball_position):
            if early_frames and 'search_radius' in locals():
                search_radius = max(search_radius, 250)  # wider window for first few frames
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
            use_alt_first = (self.frame_count == 127)
            primary_lower = self.alt_hsv_lower if use_alt_first and self.alt_hsv_lower is not None else hsv_lower_use
            primary_upper = self.alt_hsv_upper if use_alt_first and self.alt_hsv_upper is not None else hsv_upper_use
            alt_lower = hsv_lower_use if use_alt_first and self.alt_hsv_lower is not None else self.alt_hsv_lower
            alt_upper = hsv_upper_use if use_alt_first and self.alt_hsv_upper is not None else self.alt_hsv_upper
            # Apply primary HSV filter
            mask_primary = cv2.inRange(hsv_frame, primary_lower, primary_upper)
            # Apply alternate HSV filter (extended H) if defined
            mask_alt = None
            if alt_upper is not None and (alt_upper[0] != primary_upper[0] or use_alt_first):
                mask_alt = cv2.inRange(hsv_frame, alt_lower, alt_upper)
            
            # Clean up
            kernel = np.ones((2, 2), np.uint8)
            mask_primary = cv2.morphologyEx(mask_primary, cv2.MORPH_OPEN, kernel)
            mask_primary = cv2.morphologyEx(mask_primary, cv2.MORPH_CLOSE, kernel)
            if mask_alt is not None:
                mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_OPEN, kernel)
                mask_alt = cv2.morphologyEx(mask_alt, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours_raw, _ = cv2.findContours(mask_primary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [('primary', c) for c in contours_raw]
            if mask_alt is not None:
                contours_alt, _ = cv2.findContours(mask_alt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend([('alt', c) for c in contours_alt])
            
            print(f"  DEBUG: Found {len(contours)} total contours in search region (mode: {hsv_mode} incl alt)")
        
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
        predicted_point = None
        # Predict next position using last known motion if available
        if self.ball_center and hasattr(self, 'last_delta') and self.last_delta is not None:
            dx, dy = self.last_delta
            predicted_point = (self.ball_center[0] + int(dx), self.ball_center[1] + int(dy))
        
        for i, (source, contour) in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Reject huge background regions first (> 500px is definitely not a ball)
            if area > 500:
                if i < 3:  # Only print first few to avoid spam
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (background region)")
                continue
            
            # Size filter: tighter in inactive serve scan, looser when tracking
            if allow_inactive:
                if area < 1 or area > 80:
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (serve scan outside 1-80)")
                    continue
            else:
                if area < 1 or area > 150:
                    print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (outside 1-150)")
                    continue
            # Additional reject printed only when allow_inactive fell through
            if area < 1 or area > 150:
                print(f"  DEBUG: Contour {i} REJECTED - area={area:.1f}px (outside 1-150)")
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
            if predicted_point:
                pdx = cx - predicted_point[0]
                pdy = cy - predicted_point[1]
                predicted_distance = np.sqrt(pdx * pdx + pdy * pdy)
                # Give mild preference to being near predicted point
                score += predicted_distance * 0.5
            
            candidates.append((i, source, cx, cy, area, distance, size_ratio, score))
            source_label = f"[{source}]" if source != 'single' else ""
            print(f"  DEBUG: Contour {i} {source_label} CANDIDATE - pos=({cx},{cy}), area={area:.1f}px, distance={distance:.1f}px, size_ratio={size_ratio:.2f}, score={score:.1f}")
            
            if score < best_score:
                best_score = score
                best_contour = contour
                best_source = source

        # Early-serve bias: when starting and no previous ball, favor the highest (smallest y) valid contour
        if self.ball_center is None and self.frame_count <= self.start_frame + 10 and candidates:
            highest = min(candidates, key=lambda c: (c[3], c[4]))  # prioritize lowest y (higher on screen), then smaller area
            _, best_source, _, _, _, _, _, _ = highest
            best_contour = contours[highest[0]][1]
            print(f"  DEBUG: Early-serve bias -> picking highest contour (y={highest[3]}, area={highest[4]:.1f})")

        if best_contour is not None:
            # Update ball position
            M = cv2.moments(best_contour)
            cx = int(M["m10"] / M["m00"]) + x1
            cy = int(M["m01"] / M["m00"]) + y1
            self.last_seen_frame = self.frame_count
            self.edge_wait = False
            if hasattr(self, 'ball_center') and self.ball_center:
                prev_x, prev_y = self.ball_center
                self.last_delta = (cx - prev_x, cy - prev_y)
            else:
                self.last_delta = None
            
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
            size_cap = "1-80px (serve scan)" if allow_inactive else "1-150px"
            print(f"  DEBUG: All {len(contours)} contours were rejected by size filter ({size_cap})")
            # Show the actual sizes that were rejected
            rejected_sizes = []
            for source, contour in contours[:5]:  # Show first 5
                rejected_sizes.append(f"{cv2.contourArea(contour):.1f}px")
            print(f"  DEBUG: Rejected sizes (first 5): {', '.join(rejected_sizes)}")
            print(f"  DEBUG: REASON: Ball size changed outside 1-150px range")
            print(f"  DEBUG:   - Ball may be too small (far away) or too large (very close)")
            print(f"  DEBUG:   - Consider adjusting size filter if ball is visible")
            if predicted_point:
                print(f"  DEBUG: Predicted point was {predicted_point}, consider widening search around it")
        # Early-frame fallback: if nothing selected, try full-frame smallest contour
        if best_contour is None and early_frames:
            fallback_mask = cv2.inRange(frame, hsv_lower_use, hsv_upper_use) if 'hsv_lower_use' in locals() else None
            if fallback_mask is not None:
                fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                fallback_mask = cv2.morphologyEx(fallback_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
                fallback_contours, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                viable = [c for c in fallback_contours if 1 <= cv2.contourArea(c) <= 150]
                if viable:
                    smallest = min(viable, key=cv2.contourArea)
                    M = cv2.moments(smallest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self.ball_center = (cx, cy)
                        print(f"  DEBUG: Fallback picked smallest full-frame contour at ({cx},{cy}) area={cv2.contourArea(smallest):.1f}")
                        return self.ball_center
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
        
        # Draw detected players (P1 and P2)
        if self.p1_bbox is not None:
            x, y, w, h = self.p1_bbox
            # Scale bounding box for display
            x_scaled = int(x * scale)
            y_scaled = int(y * scale)
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)
            
            # Draw bounding box
            cv2.rectangle(result, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), 
                         (255, 0, 0), 2)
            
            # Draw center point
            center_x = int((x + w/2) * scale)
            center_y = int((y + h/2) * scale)
            cv2.circle(result, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Label
            cv2.putText(result, "P1", (x_scaled, y_scaled - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if self.p2_bbox is not None:
            x, y, w, h = self.p2_bbox
            # Scale bounding box for display
            x_scaled = int(x * scale)
            y_scaled = int(y * scale)
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)
            
            # Draw bounding box
            cv2.rectangle(result, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), 
                         (0, 0, 255), 2)
            
            # Draw center point
            center_x = int((x + w/2) * scale)
            center_y = int((y + h/2) * scale)
            cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Label
            cv2.putText(result, "P2", (x_scaled, y_scaled - 10), 
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
    
    def detect_and_track_players(self, frame):
        """Detect and track P1 and P2 on the court."""
        players = self.player_detector.detect_players(frame, debug=False)
        
        if len(players) >= 2:
            # Sort players by Y coordinate (top player = P1, bottom player = P2)
            players_sorted = sorted(players, key=lambda p: p[1])
            
            # P1 is the top player (server)
            self.p1_bbox = players_sorted[0]
            x, y, w, h = self.p1_bbox
            self.player1_pos = (x + w // 2, y + h // 2)  # Center of bbox
            
            # P2 is the bottom player
            self.p2_bbox = players_sorted[1]
            x, y, w, h = self.p2_bbox
            self.player2_pos = (x + w // 2, y + h // 2)  # Center of bbox
            
            return True
        elif len(players) == 1:
            # Only one player detected - assume it's P1 (server) if near serve area
            self.p1_bbox = players[0]
            x, y, w, h = self.p1_bbox
            self.player1_pos = (x + w // 2, y + h // 2)
            return True
        
        return False
    
    def detect_serve_position(self, frame):
        """Detect potential serve positions by looking for balls ABOVE P1's head in serve area."""
        # Check if serve area is configured
        if not hasattr(self, 'serve_area_x_min'):
            return None
        
        # Detect players to know where P1's head is
        self.detect_and_track_players(frame)
        
        # Restrict serve area to upper portion only (above P1's head)
        # Use P1's bounding box top as the reference
        if self.p1_bbox is not None:
            _, p1_y, _, p1_h = self.p1_bbox
            p1_head_y = p1_y  # Top of bounding box is the head
            serve_search_y_max = min(p1_head_y, self.serve_area_y_max)
        else:
            # Conservative default: only search upper portion of serve area
            serve_search_y_max = min(self.serve_area_y_min + (self.serve_area_y_max - self.serve_area_y_min) // 2, 
                                    self.serve_area_y_max)
        
        # Apply HSV filter to find potential balls
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
        
        # Find contours in the serve area (only upper portion above P1)
        serve_roi = mask[self.serve_area_y_min:serve_search_y_max, 
                        self.serve_area_x_min:self.serve_area_x_max]
        contours, _ = cv2.findContours(serve_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for ball-sized contours (prefer larger balls as they're more likely real)
        best_contour = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Tennis ball size range: 3-80 pixels (depending on distance from camera)
            # Minimum 3 pixels to avoid false positives
            if 3 <= area <= 80:
                # Calculate center in full frame coordinates
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + self.serve_area_x_min
                    cy = int(M["m01"] / M["m00"]) + self.serve_area_y_min
                    
                    # Only accept balls in the upper portion (above P1's head)
                    if cy < serve_search_y_max:
                        # Prefer larger balls (they're more likely to be the actual ball)
                        if area > best_area:
                            best_area = area
                            best_contour = (cx, cy)
        
        return best_contour
    
    def mark_serve_area(self, frame):
        """Mark the serve area with 4 points and save to config."""
        print("\n=== MARKING SERVE AREA ===")
        print("Click 4 points to define the serve area:")
        print("1. Top-left corner of serve area")
        print("2. Top-right corner of serve area") 
        print("3. Bottom-right corner of serve area")
        print("4. Bottom-left corner of serve area")
        print("Press ESC to cancel")
        
        # Create a copy of the frame for marking
        display_frame = frame.copy()
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"{len(points)}", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Mark Serve Area", display_frame)
                print(f"Point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow("Mark Serve Area", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mark Serve Area", mouse_callback)
        
        while len(points) < 4:
            cv2.imshow("Mark Serve Area", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow("Mark Serve Area")
                return False
        
        cv2.destroyWindow("Mark Serve Area")
        
        # Calculate serve area boundaries
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        serve_area_x_min = min(x_coords)
        serve_area_x_max = max(x_coords)
        serve_area_y_min = min(y_coords)
        serve_area_y_max = max(y_coords)
        
        print(f"\nServe area boundaries:")
        print(f"X: {serve_area_x_min} to {serve_area_x_max}")
        print(f"Y: {serve_area_y_min} to {serve_area_y_max}")
        
        # Save to config
        self.serve_area_points = points
        self.serve_area_x_min = serve_area_x_min
        self.serve_area_x_max = serve_area_x_max
        self.serve_area_y_min = serve_area_y_min
        self.serve_area_y_max = serve_area_y_max
        
        # Save to config file
        self.save_serve_area_to_config()
        
        return True
    
    def save_serve_area_to_config(self):
        """Save serve area configuration to hsv_config.json."""
        try:
            # Load existing config
            config = {}
            if os.path.exists('hsv_config.json'):
                with open('hsv_config.json', 'r') as f:
                    config = json.load(f)
            
            # Add serve area data
            config['serve_area_points'] = self.serve_area_points
            config['serve_area_x_min'] = self.serve_area_x_min
            config['serve_area_x_max'] = self.serve_area_x_max
            config['serve_area_y_min'] = self.serve_area_y_min
            config['serve_area_y_max'] = self.serve_area_y_max
            
            # Save updated config
            with open('hsv_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print("Serve area configuration saved to hsv_config.json")
            
        except Exception as e:
            print(f"Error saving serve area config: {e}")
    
    def detect_point_end(self, ball_position, frame):
        """Detect if a point has ended based on ball position and behavior."""
        height, width = frame.shape[:2]
        x, y = ball_position
        
        # Early serve grace: do not end point in first frames after start_frame
        if self.frame_count <= (self.start_frame + 15):
            return False, "Early-serve grace"
        # If we're waiting near an edge, don't end the point
        if getattr(self, 'edge_wait', False):
            return False, "Edge wait"
        
        # Check if ball is out of court bounds
        if x < 0 or x > width or y < 0 or y > height:
            return True, "Ball out of court bounds"
        
        # Check if ball is in the net area (if configured) - but be more careful
        if hasattr(self, 'net_area_y_min') and hasattr(self, 'net_area_y_max'):
            if self.net_area_y_min <= y <= self.net_area_y_max:
                # Ball is in net area - check if it's been there for multiple frames
                if not hasattr(self, 'net_area_frames'):
                    self.net_area_frames = 0
                self.net_area_frames += 1
                
                # Only consider it a net hit if ball has been in net area for several frames
                if self.net_area_frames > 5:
                    return True, "Ball hit the net"
            else:
                # Reset counter if ball is not in net area
                if hasattr(self, 'net_area_frames'):
                    self.net_area_frames = 0
        
        # Check if ball has been stationary for too long (double bounce)
        if hasattr(self, 'ball_velocity_history') and len(self.ball_velocity_history) > 10:
            recent_velocities = self.ball_velocity_history[-10:]
            avg_velocity = sum(recent_velocities) / len(recent_velocities)
            if avg_velocity < 5:  # Very slow movement
                return True, "Ball stopped (possible double bounce)"
        
        # Check if ball is near court edges (likely out)
        edge_margin = 50
        if (x < edge_margin or x > width - edge_margin or 
            y < edge_margin or y > height - edge_margin):
            return True, "Ball near court edge (likely out)"
        
        return False, "Point continues"
    
    def _open_serve_area_hsv_tuner(self, frame):
        """Open HSV tuner specifically for the serve area."""
        print("Opening HSV tuner for SERVE AREA...")
        
        # Extract serve area from frame
        serve_roi = frame[self.serve_area_y_min:self.serve_area_y_max, 
                         self.serve_area_x_min:self.serve_area_x_max]
        
        # Create tuner window
        tuner_window = "SERVE AREA HSV TUNER"
        cv2.namedWindow(tuner_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(tuner_window, 800, 600)
        
        # Initialize trackbars with current HSV values
        cv2.createTrackbar("H Min", tuner_window, int(self.hsv_lower[0]), 179, lambda x: None)
        cv2.createTrackbar("H Max", tuner_window, int(self.hsv_upper[0]), 179, lambda x: None)
        cv2.createTrackbar("S Min", tuner_window, int(self.hsv_lower[1]), 255, lambda x: None)
        cv2.createTrackbar("S Max", tuner_window, int(self.hsv_upper[1]), 255, lambda x: None)
        cv2.createTrackbar("V Min", tuner_window, int(self.hsv_lower[2]), 255, lambda x: None)
        cv2.createTrackbar("V Max", tuner_window, int(self.hsv_upper[2]), 255, lambda x: None)
        cv2.createTrackbar("Bulb Min", tuner_window, 2, 50, lambda x: None)
        cv2.createTrackbar("Bulb Max", tuner_window, 80, 200, lambda x: None)
        
        print("Adjust HSV values to detect balls in the serve area")
        print("Press 'Q' or ESC to close and save settings")
        
        while True:
            # Get trackbar values
            h_min = cv2.getTrackbarPos("H Min", tuner_window)
            h_max = cv2.getTrackbarPos("H Max", tuner_window)
            s_min = cv2.getTrackbarPos("S Min", tuner_window)
            s_max = cv2.getTrackbarPos("S Max", tuner_window)
            v_min = cv2.getTrackbarPos("V Min", tuner_window)
            v_max = cv2.getTrackbarPos("V Max", tuner_window)
            bulb_min = cv2.getTrackbarPos("Bulb Min", tuner_window)
            bulb_max = cv2.getTrackbarPos("Bulb Max", tuner_window)
            
            # Apply HSV filter to serve area
            hsv_serve = cv2.cvtColor(serve_roi, cv2.COLOR_BGR2HSV)
            hsv_lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            hsv_upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv_serve, hsv_lower, hsv_upper)
            
            # Find contours in serve area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw detected balls on serve area
            display_serve = serve_roi.copy()
            ball_count = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if bulb_min < area < bulb_max:
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw circle and size label
                        cv2.circle(display_serve, (cx, cy), 8, (0, 255, 0), 2)
                        cv2.putText(display_serve, f"{area:.0f}", (cx + 12, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        ball_count += 1
            
            # Create side-by-side display
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([display_serve, mask_bgr])
            
            # Add text info
            info_text = f"SERVE AREA | Balls: {ball_count} | HSV: H={h_min}-{h_max} S={s_min}-{s_max} V={v_min}-{v_max} | Size: {bulb_min}-{bulb_max}px"
            cv2.putText(combined, info_text, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(tuner_window, combined)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
        
        # Update HSV values with tuned values
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        
        # Save updated HSV values to config
        self.save_hsv_to_config()
        
        cv2.destroyWindow(tuner_window)
        print("Serve area HSV tuner closed. Settings saved.")
    
    def save_hsv_to_config(self):
        """Save current HSV values to config file."""
        try:
            # Load existing config
            config = {}
            if os.path.exists('hsv_config.json'):
                with open('hsv_config.json', 'r') as f:
                    config = json.load(f)
            
            # Update HSV values
            if "regular_court" in config:
                # Update regular court HSV values
                config["regular_court"]["h_min"] = int(self.hsv_lower[0])
                config["regular_court"]["h_max"] = int(self.hsv_upper[0])
                config["regular_court"]["s_min"] = int(self.hsv_lower[1])
                config["regular_court"]["s_max"] = int(self.hsv_upper[1])
                config["regular_court"]["v_min"] = int(self.hsv_lower[2])
                config["regular_court"]["v_max"] = int(self.hsv_upper[2])
            else:
                # Old format
                config["h_min"] = int(self.hsv_lower[0])
                config["h_max"] = int(self.hsv_upper[0])
                config["s_min"] = int(self.hsv_lower[1])
                config["s_max"] = int(self.hsv_upper[1])
                config["v_min"] = int(self.hsv_lower[2])
                config["v_max"] = int(self.hsv_upper[2])
            
            # Save updated config
            with open('hsv_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print("HSV values saved to hsv_config.json")
            
        except Exception as e:
            print(f"Error saving HSV config: {e}")
    
    def process_video(self):
        """Process video with intelligent tennis game analysis."""
        print("Intelligent Tennis Game Tracker")
        print("=" * 50)
        print("1. Automatically finds first serve position")
        print("2. Tracks ball through each point")
        print("3. Detects when point ends (out, net, double bounce)")
        print("4. Automatically waits for next serve")
        print("5. Press 'D' to advance frame by frame")
        print("6. Press 'N' to mark net area (6 points clockwise)")
        print("7. Press 'S' to mark serve area (4 points)")
        print("8. Press 'Q' to quit")
        print("=" * 50)
        if self.start_frame > 0:
            print(f"Starting at frame {self.start_frame}")
        
        cv2.namedWindow("Tennis Game Tracker", cv2.WINDOW_NORMAL)
        
        # Store scale factor for mouse coordinate conversion
        scale_factor = 1.0
        current_frame = None
        
        # Game state variables
        game_state = "SCANNING_FOR_SERVE"  # SCANNING_FOR_SERVE, TRACKING_POINT, POINT_ENDED, WAITING_FOR_SERVE
        point_start_frame = None
        point_end_frame = None
        serve_positions = []
        # Ensure capture starts at requested frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        early_serve_grace_frames = 15  # aggressively accept serve during first frames after start_frame
        
        # Modified mouse callback that converts resized coordinates to original
        def mouse_callback_with_scale(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Convert clicked coordinates from resized frame to original frame
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                # Analyze ball at original coordinates
                self.analyze_ball_at_point(current_frame, (orig_x, orig_y))
        
        cv2.setMouseCallback("Tennis Game Tracker", mouse_callback_with_scale)
        
        # Start from configured frame to scan for first serve
        print(f"\nStarting from frame {self.start_frame} to scan for first serve...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self.frame_count = self.start_frame
        
        # Load HSV values from config file (or use defaults if not present)
        hsv_config = self.load_hsv_config()
        
        # Check if net area needs to be marked (first time setup)
        if hsv_config and "net_area_y_min" not in hsv_config:
            print("\n" + "=" * 70)
            print("FIRST TIME SETUP: Please mark the net area")
            print("=" * 70)
            ret, frame = self.cap.read()
            if ret:
                if self.mark_net_area(frame):
                    print("Net area saved! This will be used for all future runs.")
                    # Reload config after marking net area
                    hsv_config = self.load_hsv_config()
                else:
                    print("Net area marking skipped. Using default values.")
        
        # Check if serve area needs to be marked (first time setup)
        if hsv_config and "serve_area_points" not in hsv_config:
            print("\n" + "=" * 70)
            print("FIRST TIME SETUP: Please mark the serve area")
            print("=" * 70)
            ret, frame = self.cap.read()
            if ret:
                if self.mark_serve_area(frame):
                    print("Serve area saved! This will be used for all future runs.")
                    # Reload config after marking serve area
                    hsv_config = self.load_hsv_config()
                else:
                    print("Serve area marking skipped. Using default values.")
        
        # Initialize HSV values
        if not hsv_config:
            # Default values if config not found
            self.hsv_lower = np.array([20, 20, 70], dtype=np.uint8)
            self.hsv_upper = np.array([90, 255, 255], dtype=np.uint8)
        # If old format, set hsv_lower and hsv_upper (already done in load_hsv_config for new format)
        elif 'h_min' in hsv_config:
            self.hsv_lower = np.array([hsv_config['h_min'], hsv_config['s_min'], hsv_config['v_min']], dtype=np.uint8)
            self.hsv_upper = np.array([hsv_config['h_max'], hsv_config['s_max'], hsv_config['v_max']], dtype=np.uint8)
        # New format already loaded in load_hsv_config, just use the regular court as default
        
        print(f"\nHSV Filter: H={self.hsv_lower[0]}-{self.hsv_upper[0]}, S={self.hsv_lower[1]}-{self.hsv_upper[1]}, V={self.hsv_lower[2]}-{self.hsv_upper[2]}")
        print(f"Game State: {game_state}")
        print(f"Press 'D' to advance frame by frame")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Early serve detection: while scanning and within grace window, attempt ball track and enter tracking
            if game_state == "SCANNING_FOR_SERVE" and self.frame_count <= (self.start_frame + early_serve_grace_frames):
                candidate = self.track_ball_in_frame(frame, allow_inactive=True)
                if candidate is not None:
                    print(f"Frame {self.frame_count}: Early serve found at {candidate}, entering TRACKING_POINT")
                    game_state = "TRACKING_POINT"
                    point_start_frame = self.frame_count
                    self.tracking = True
            
            self.frame_count += 1
            current_frame = frame  # Store current frame for mouse callback
            
            # Handle different game states
            if game_state == "SCANNING_FOR_SERVE":
                # Look for potential serve positions (ball in serve area)
                potential_serve = self.detect_serve_position(frame)
                if potential_serve:
                    print(f"\n{'='*70}")
                    print(f"SERVE DETECTED at frame {self.frame_count}!")
                    print(f"Ball position: {potential_serve}")
                    print(f"Starting to track the ball...")
                    print(f"{'='*70}\n")
                    serve_positions.append((self.frame_count, potential_serve))
                    # Start tracking this potential serve
                    self.ball_center = potential_serve
                    self.tracking = True
                    self.ball_stopped = False
                    self.ball_velocity_history = []
                    self.initial_ball_position = potential_serve
                    self.ball_size = None  # Will be set by track_ball_in_frame
                    point_start_frame = self.frame_count
                    game_state = "TRACKING_POINT"
            
            elif game_state == "TRACKING_POINT":
                # Update player positions while tracking
                self.detect_and_track_players(frame)
                
                # Track ball through the point
                tracked_position = None
                if self.tracking and self.hsv_lower is not None and self.ball_center is not None:
                    tracked_position = self.track_ball_in_frame(frame)
                if tracked_position:
                    size_text = f"{self.ball_size:.1f}px" if self.ball_size is not None else "unknown"
                    print(f"Frame {self.frame_count}: Ball tracked at {tracked_position} - Size: {size_text}")
                    
                    # Check if point has ended
                    point_ended, reason = self.detect_point_end(tracked_position, frame)
                    if point_ended:
                        point_end_frame = self.frame_count
                        print(f"Frame {self.frame_count}: POINT ENDED - {reason}")
                        print(f"Point duration: {point_end_frame - point_start_frame} frames")
                        game_state = "POINT_ENDED"
                        self.tracking = False
                    else:
                        print(f"Frame {self.frame_count}: Ball tracking continued")
                        # Ball lost - might be end of point
                        grace_limit = 45 if point_start_frame and self.frame_count <= (self.start_frame + 45) else 30
                        if self.frame_count - point_start_frame > grace_limit:  # Minimum point duration
                            point_end_frame = self.frame_count
                            print(f"Frame {self.frame_count}: POINT ENDED - Ball lost (likely out of court)")
                            print(f"Point duration: {point_end_frame - point_start_frame} frames")
                            game_state = "POINT_ENDED"
                            self.tracking = False
            
            elif game_state == "POINT_ENDED":
                # Wait a few frames then start scanning for next serve
                if self.frame_count - point_end_frame > 60:  # Wait 2 seconds (60 frames at 30fps)
                    print(f"Frame {self.frame_count}: Starting to scan for next serve...")
                    game_state = "SCANNING_FOR_SERVE"
                    point_start_frame = None
                    point_end_frame = None
                # If we are still within early grace and see the ball, resume tracking
                elif (self.frame_count <= (self.start_frame + 15)) or getattr(self, 'edge_wait', False) or (self.last_seen_frame and self.frame_count - self.last_seen_frame <= 200):
                    candidate = self.track_ball_in_frame(frame, allow_inactive=True)
                    if candidate is not None:
                        print(f"Frame {self.frame_count}: Ball re-found during grace, resuming TRACKING_POINT at {candidate}")
                        game_state = "TRACKING_POINT"
                        point_start_frame = self.frame_count
                        self.tracking = True
        
            elif game_state == "WAITING_FOR_SERVE":
                # This state can be used for manual intervention
                pass
            
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
            
            # Show game state and controls
            if game_state == "SCANNING_FOR_SERVE":
                cv2.putText(display_frame, f"SCANNING FOR SERVE | D=Next | N=Net | S=Serve | Q=Quit", 
                           (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif game_state == "TRACKING_POINT":
                cv2.putText(display_frame, f"TRACKING POINT (started frame {point_start_frame}) | D=Next | N=Net | S=Serve | Q=Quit", 
                           (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif game_state == "POINT_ENDED":
                cv2.putText(display_frame, f"POINT ENDED (waiting for next serve) | D=Next | N=Net | S=Serve | Q=Quit", 
                           (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(display_frame, f"Game State: {game_state} | D=Next | N=Net | S=Serve | Q=Quit", 
                           (10, new_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Tennis Game Tracker", display_frame)
            
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
            elif key == ord('s'):
                # Show HSV tuner for serve area
                print("\n=== SERVE AREA HSV TUNER ===")
                if hasattr(self, 'serve_area_x_min'):
                    print(f"Serve area: X={self.serve_area_x_min}-{self.serve_area_x_max}, Y={self.serve_area_y_min}-{self.serve_area_y_max}")
                    self._open_serve_area_hsv_tuner(frame)
                else:
                    print("No serve area configured. Marking serve area first...")
                    if self.mark_serve_area(frame):
                        print("Serve area marked successfully! Updated config file.")
                        # Reload HSV config to get updated serve area
                        self.load_hsv_config()
                        # Now open the HSV tuner for the serve area
                        self._open_serve_area_hsv_tuner(frame)
                    else:
                        print("Serve area marking cancelled.")
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
    parser = argparse.ArgumentParser(description="Interactive tennis ball analyzer/tracker")
    parser.add_argument("--start-frame", type=int, default=100,
                        help="Frame index to start from (default 100, e.g., serve)")
    args = parser.parse_args()
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    analyzer = InteractiveBallAnalyzer(video_path, start_frame=args.start_frame)
    analyzer.process_video()
