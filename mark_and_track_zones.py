"""
Interactive zone marker and MOG2 tracker.
1. Mark P1 zone (4 points) and P2 zone (4 points) by clicking
2. Tune MOG2 parameters to detect players
3. Save zones and settings
"""

import cv2
import numpy as np
import json
import os


class InteractiveZoneTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Zones (will be marked by user)
        self.p1_zone = None
        self.p2_zone = None
        
        # MOG2 parameters
        self.var_threshold = 8
        self.learning_rate = 0.001
        self.history = 200
        self.p1_min_area = 1000
        self.p1_max_area = 50000
        self.p2_min_area = 500
        self.p2_max_area = 30000
        
        # Background subtractors (created after zones are marked)
        self.p1_bg_subtractor = None
        self.p2_bg_subtractor = None
        
        # Last known player positions (for temporal consistency)
        self.last_p1_player = None
        self.last_p2_player = None
    
    def mark_zones(self):
        """Interactive zone marking."""
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read frame")
            return False
        
        print("\n" + "="*70)
        print("ZONE MARKING")
        print("="*70)
        print("\nMark P1 ZONE (4 corners):")
        print("  Click 4 points to define the P1 zone rectangle")
        print("  (top-left area where P1 is)")
        print("\nThen mark P2 ZONE (4 corners):")
        print("  Click 4 points to define the P2 zone rectangle")
        print("  (bottom-right area where P2 is)")
        print("\nPress ESC to cancel")
        print("="*70)
        
        # Mark P1 zone
        print("\n>>> Mark P1 ZONE (4 points)...")
        p1_points = self._mark_zone(frame, "P1 ZONE - Click 4 corners", (255, 0, 0))
        if len(p1_points) != 4:
            return False
        self.p1_zone = np.array(p1_points, dtype=np.int32)
        
        # Mark P2 zone
        print("\n>>> Mark P2 ZONE (4 points)...")
        p2_points = self._mark_zone(frame, "P2 ZONE - Click 4 corners", (0, 165, 255))
        if len(p2_points) != 4:
            return False
        self.p2_zone = np.array(p2_points, dtype=np.int32)
        
        # Save zones
        self._save_zones()
        
        print("\n[OK] Zones marked and saved!")
        return True
    
    def _mark_zone(self, frame, window_name, color):
        """Mark a single zone with 4 points."""
        points = []
        display = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(display, (x, y), 10, color, -1)
                cv2.putText(display, str(len(points)), (x + 15, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Draw lines between points
                if len(points) > 1:
                    cv2.line(display, points[-2], points[-1], color, 3)
                
                # Close polygon when 4 points
                if len(points) == 4:
                    cv2.line(display, points[-1], points[0], color, 3)
                    zone_array = np.array(points, dtype=np.int32)
                    overlay = display.copy()
                    cv2.fillPoly(overlay, [zone_array], color)
                    cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                
                cv2.imshow(window_name, display)
                print(f"  Point {len(points)}: ({x}, {y})")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while len(points) < 4:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return []
        
        cv2.waitKey(500)
        cv2.destroyWindow(window_name)
        return points
    
    def _save_zones(self):
        """Save zones to config file."""
        config = {
            'p1_zone': self.p1_zone.tolist(),
            'p2_zone': self.p2_zone.tolist()
        }
        with open('player_zones.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Zones saved to player_zones.json")
    
    def _load_zones(self):
        """Load zones from config file."""
        if os.path.exists('player_zones.json'):
            with open('player_zones.json', 'r') as f:
                config = json.load(f)
                self.p1_zone = np.array(config['p1_zone'], dtype=np.int32)
                self.p2_zone = np.array(config['p2_zone'], dtype=np.int32)
                print("Zones loaded from player_zones.json")
                return True
        return False
    
    def _load_parameters(self):
        """Load parameters from config file."""
        if os.path.exists('mog2_parameters.json'):
            with open('mog2_parameters.json', 'r') as f:
                config = json.load(f)
                self.var_threshold = config.get('var_threshold', 8)
                self.learning_rate = config.get('learning_rate', 0.001)
                self.history = config.get('history', 200)
                self.p1_min_area = config.get('p1_min_area', 1000)
                self.p1_max_area = config.get('p1_max_area', 50000)
                self.p2_min_area = config.get('p2_min_area', 500)
                self.p2_max_area = config.get('p2_max_area', 30000)
    
    def detect_player_in_zone(self, frame, zone, bg_subtractor, min_area, max_area, last_player=None):
        """Detect moving player in zone using MOG2 with intelligent background refresh."""
        # Create zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone], 255)
        
        # PROFESSIONAL SOLUTION: Adaptive learning rate based on detection confidence
        # - If player was just detected -> very low learning (don't learn player as background)
        # - If no player detected -> normal learning (adapt to environment changes)
        
        if last_player is not None:
            # Player was recently detected - use VERY LOW learning to preserve background model
            adaptive_learning_rate = -1  # -1 = no learning at all
        else:
            # No recent detection - use normal learning to adapt
            adaptive_learning_rate = self.learning_rate
        
        # Apply background subtraction with adaptive learning
        fg_mask = bg_subtractor.apply(frame, learningRate=adaptive_learning_rate)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Apply zone mask to only look at our zone
        fg_mask = cv2.bitwise_and(fg_mask, zone_mask)
        
        # Clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest valid contour
        best_player = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                if area > best_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x, y, w, h = cv2.boundingRect(contour)
                        best_player = {
                            'position': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area
                        }
                        best_area = area
        
        return best_player, fg_mask
    
    def _show_debug_window(self, frame, p1_player, p2_player, p1_mask, p2_mask, frame_num):
        """Show debug window with all contours and their sizes."""
        print(f"\n{'='*70}")
        print(f"DEBUG WINDOW - Frame {frame_num}")
        print(f"SENSITIVITY (varThreshold): {self.var_threshold}")
        print(f"{'='*70}")
        
        # Create debug display
        debug_frame = frame.copy()
        
        # Draw zones
        overlay = debug_frame.copy()
        cv2.fillPoly(overlay, [self.p1_zone], (255, 100, 100))
        cv2.fillPoly(overlay, [self.p2_zone], (100, 200, 255))
        debug_frame = cv2.addWeighted(debug_frame, 0.7, overlay, 0.3, 0)
        cv2.polylines(debug_frame, [self.p1_zone], True, (255, 0, 0), 3)
        cv2.polylines(debug_frame, [self.p2_zone], True, (0, 165, 255), 3)
        
        # Process P1 zone - find ALL contours
        print(f"\nP1 ZONE ANALYSIS:")
        print(f"  Area range: {self.p1_min_area} - {self.p1_max_area} pixels")
        
        contours_p1, _ = cv2.findContours(p1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Total contours found: {len(contours_p1)}")
        
        for i, contour in enumerate(contours_p1):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
            
            # Determine status
            if area < self.p1_min_area:
                status = f"REJECTED (too small: {area:.0f} < {self.p1_min_area})"
                color = (0, 0, 255)  # Red
            elif area > self.p1_max_area:
                status = f"REJECTED (too large: {area:.0f} > {self.p1_max_area})"
                color = (255, 0, 255)  # Magenta
            else:
                status = f"VALID ({area:.0f}px)"
                color = (0, 255, 0)  # Green
            
            # Draw on debug frame
            cv2.drawContours(debug_frame, [contour], -1, color, 2)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_frame, f"P1-{i}: {area:.0f}px", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"  Contour {i}: area={area:.0f}px, bbox=({x},{y},{w},{h}) - {status}")
        
        if p1_player:
            print(f"  >>> SELECTED: {p1_player['area']:.0f}px at {p1_player['bbox']}")
        else:
            print(f"  >>> NO VALID PLAYER FOUND")
        
        # Process P2 zone - find ALL contours
        print(f"\nP2 ZONE ANALYSIS:")
        print(f"  Area range: {self.p2_min_area} - {self.p2_max_area} pixels")
        
        contours_p2, _ = cv2.findContours(p2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Total contours found: {len(contours_p2)}")
        
        for i, contour in enumerate(contours_p2):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
            
            # Determine status
            if area < self.p2_min_area:
                status = f"REJECTED (too small: {area:.0f} < {self.p2_min_area})"
                color = (0, 0, 255)  # Red
            elif area > self.p2_max_area:
                status = f"REJECTED (too large: {area:.0f} > {self.p2_max_area})"
                color = (255, 0, 255)  # Magenta
            else:
                status = f"VALID ({area:.0f}px)"
                color = (0, 255, 0)  # Green
            
            # Draw on debug frame
            cv2.drawContours(debug_frame, [contour], -1, color, 2)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug_frame, f"P2-{i}: {area:.0f}px", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"  Contour {i}: area={area:.0f}px, bbox=({x},{y},{w},{h}) - {status}")
        
        if p2_player:
            print(f"  >>> SELECTED: {p2_player['area']:.0f}px at {p2_player['bbox']}")
        else:
            print(f"  >>> NO VALID PLAYER FOUND")
        
        print(f"{'='*70}\n")
        
        # Create mask display windows
        p1_mask_display = cv2.cvtColor(p1_mask, cv2.COLOR_GRAY2BGR)
        p2_mask_display = cv2.cvtColor(p2_mask, cv2.COLOR_GRAY2BGR)
        
        # Show windows
        cv2.namedWindow("DEBUG: Main Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEBUG: Main Frame", 1600, 900)
        cv2.imshow("DEBUG: Main Frame", debug_frame)
        
        cv2.namedWindow("DEBUG: P1 Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEBUG: P1 Mask", 800, 450)
        cv2.imshow("DEBUG: P1 Mask", p1_mask_display)
        
        cv2.namedWindow("DEBUG: P2 Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEBUG: P2 Mask", 800, 450)
        cv2.imshow("DEBUG: P2 Mask", p2_mask_display)
        
        print("Press any key in debug windows to close them...")
        cv2.waitKey(0)
        
        # Close debug windows
        cv2.destroyWindow("DEBUG: Main Frame")
        cv2.destroyWindow("DEBUG: P1 Mask")
        cv2.destroyWindow("DEBUG: P2 Mask")
    
    def tune_parameters(self):
        """Interactive parameter tuning."""
        # Reload parameters from config to ensure we have latest values
        self._load_parameters()
        
        print("\n" + "="*70)
        print("PARAMETER TUNING")
        print("="*70)
        print("Adjust parameters to detect both P1 and P2")
        print("The video will play from the beginning")
        print("="*70)
        
        # Create background subtractors
        self.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )
        self.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True
        )
        
        # Create window with trackbars
        cv2.namedWindow("Parameter Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parameter Tuner", 1600, 900)
        
        def nothing(x): pass
        
        cv2.createTrackbar("Sensitivity", "Parameter Tuner", self.var_threshold, 50, nothing)
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        paused = True  # Start paused for frame-by-frame analysis
        frame_num = 0
        advance_frame = False
        last_params = self.var_threshold
        
        print("\nStarting in PAUSED mode for frame-by-frame analysis")
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame (go back)")
        print("  'S' = Show debug window (all contours + masks)")
        print("  SPACE = Play/pause")
        print("  'R' = Restart from frame 0")
        print("  'Q' = Save and quit")
        print("\nTRACKBAR CONTROLS:")
        print("  Sensitivity = Lower = more sensitive to movement")
        print("\nBULB SIZE RANGES (fixed from config):")
        print(f"  P1: {self.p1_min_area} - {self.p1_max_area} pixels")
        print(f"  P2: {self.p2_min_area} - {self.p2_max_area} pixels")
        print("\nNOTE: Changing sensitivity will auto-restart from frame 0!")
        
        while True:
            if not paused or advance_frame:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Reset background subtractors
                    self.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=self.history,
                        varThreshold=self.var_threshold,
                        detectShadows=True
                    )
                    self.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=self.history,
                        varThreshold=self.var_threshold,
                        detectShadows=True
                    )
                    frame_num = 0
                    print("\n[LOOP] Restarting from beginning...")
                    continue
                
                frame_num += 1
                
                # Get trackbar values
                self.var_threshold = cv2.getTrackbarPos("Sensitivity", "Parameter Tuner")
                
                # Bulb sizes are fixed from config file (not on trackbars)
                # p1_min_area, p1_max_area, p2_min_area, p2_max_area stay as loaded
                
                # Check if sensitivity changed - DON'T auto-restart, just update
                current_params = self.var_threshold
                if current_params != last_params:
                    print(f"\n[SENSITIVITY CHANGED] {last_params} -> {self.var_threshold}")
                    print(f"  Press 'S' to see debug with new sensitivity")
                    print(f"  Press 'R' to restart from frame 0 with new sensitivity")
                    
                    # Update background subtractors with new sensitivity (but keep learned background)
                    self.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=200, varThreshold=self.var_threshold, detectShadows=True
                    )
                    self.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=200, varThreshold=self.var_threshold, detectShadows=True
                    )
                    
                    # Reset temporal tracking
                    self.last_p1_player = None
                    self.last_p2_player = None
                    
                    last_params = current_params
                
                # Detect players with temporal consistency
                p1_player, p1_mask = self.detect_player_in_zone(
                    frame, self.p1_zone, self.p1_bg_subtractor,
                    self.p1_min_area, self.p1_max_area, self.last_p1_player
                )
                
                p2_player, p2_mask = self.detect_player_in_zone(
                    frame, self.p2_zone, self.p2_bg_subtractor,
                    self.p2_min_area, self.p2_max_area, self.last_p2_player
                )
                
                # Update last known positions
                if p1_player:
                    self.last_p1_player = p1_player
                if p2_player:
                    self.last_p2_player = p2_player
                
                # Visualize
                display = frame.copy()
                
                # Draw zones
                overlay = display.copy()
                cv2.fillPoly(overlay, [self.p1_zone], (255, 100, 100))
                cv2.fillPoly(overlay, [self.p2_zone], (100, 200, 255))
                display = cv2.addWeighted(display, 0.85, overlay, 0.15, 0)
                
                cv2.polylines(display, [self.p1_zone], True, (255, 0, 0), 3)
                cv2.polylines(display, [self.p2_zone], True, (0, 165, 255), 3)
                
                # Draw P1
                p1_status = "NOT FOUND"
                if p1_player:
                    p1_status = f"DETECTED ({p1_player['area']:.0f}px)"
                    x, y, w, h = p1_player['bbox']
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 5)
                    cv2.circle(display, p1_player['position'], 15, (255, 0, 0), -1)
                    label = f"P1 ({p1_player['area']:.0f}px)"
                    cv2.rectangle(display, (x, y - 40), (x + 250, y - 5), (255, 0, 0), -1)
                    cv2.putText(display, label, (x + 5, y - 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Draw P2
                p2_status = "NOT FOUND"
                if p2_player:
                    p2_status = f"DETECTED ({p2_player['area']:.0f}px)"
                    x, y, w, h = p2_player['bbox']
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 165, 255), 5)
                    cv2.circle(display, p2_player['position'], 15, (0, 165, 255), -1)
                    label = f"P2 ({p2_player['area']:.0f}px)"
                    cv2.rectangle(display, (x, y - 40), (x + 250, y - 5), (0, 165, 255), -1)
                    cv2.putText(display, label, (x + 5, y - 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Status display
                status_color = (0, 255, 0) if (p1_player and p2_player) else (0, 165, 255) if (p1_player or p2_player) else (0, 0, 255)
                status_msg = "SUCCESS - Both detected!" if (p1_player and p2_player) else "Partial - Keep tuning" if (p1_player or p2_player) else "No players - Adjust sensitivity"
                
                cv2.rectangle(display, (10, 10), (900, 180), (0, 0, 0), -1)
                cv2.putText(display, status_msg, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                cv2.putText(display, f"Frame: {frame_num} | P1: {p1_status} | P2: {p2_status}", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                controls = "D=next | A=back | S=debug | SPACE=play/pause | Q=save&quit" if paused else "SPACE=pause | S=debug | Q=save&quit"
                cv2.putText(display, f"Sensitivity: {self.var_threshold} | {controls}", (20, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"P1 Bulb Size: {self.p1_min_area}-{self.p1_max_area}px | P2 Bulb Size: {self.p2_min_area}-{self.p2_max_area}px", (20, 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show
                cv2.imshow("Parameter Tuner", display)
                
                # Reset advance_frame flag
                advance_frame = False
            
            key = cv2.waitKey(30 if not paused else 0) & 0xFF
            if key == ord('q'):
                self._save_parameters()
                break
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'PAUSED' if paused else 'PLAYING'} at frame {frame_num}")
            elif key == ord('d'):
                # Advance one frame
                advance_frame = True
                print(f"\n[FORWARD] Advancing to frame {frame_num + 1}...")
            elif key == ord('a'):
                # Go back one frame
                if frame_num > 1:
                    frame_num = max(0, frame_num - 2)  # -2 because we'll increment by 1 on next read
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    advance_frame = True
                    print(f"\n[BACKWARD] Going back to frame {frame_num + 1}...")
                else:
                    print(f"\n[BACKWARD] Already at beginning (frame {frame_num})")
            elif key == ord('s'):
                # Show debug window with all contours and masks
                print(f"\n[DEBUG] Opening debug window for frame {frame_num}...")
                self._show_debug_window(frame, p1_player, p2_player, p1_mask, p2_mask, frame_num)
            elif key == ord('r'):
                # Manual restart from frame 0
                print(f"\n[MANUAL RESTART] Going back to frame 0...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                
                # Reset background subtractors with current parameters
                self.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=200, varThreshold=self.var_threshold, detectShadows=True
                )
                self.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=200, varThreshold=self.var_threshold, detectShadows=True
                )
                
                paused = True
                advance_frame = True  # Show first frame immediately
        
        cv2.destroyAllWindows()
    
    def _save_parameters(self):
        """Save parameters to config."""
        config = {
            'var_threshold': self.var_threshold,
            'learning_rate': self.learning_rate,
            'history': self.history,
            'p1_min_area': self.p1_min_area,
            'p1_max_area': self.p1_max_area,
            'p2_min_area': self.p2_min_area,
            'p2_max_area': self.p2_max_area
        }
        with open('mog2_parameters.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "="*70)
        print("FINAL SETTINGS SAVED")
        print("="*70)
        print(f"Sensitivity (varThreshold): {self.var_threshold}")
        print(f"P1 Area range: {self.p1_min_area} - {self.p1_max_area} pixels")
        print(f"P2 Area range: {self.p2_min_area} - {self.p2_max_area} pixels")
        print("Settings saved to mog2_parameters.json")
        print("="*70)


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = InteractiveZoneTracker(video_path)
    
    print("\n" + "="*70)
    print("INTERACTIVE ZONE MARKER AND MOG2 TRACKER")
    print("="*70)
    
    # Check if zones already exist
    if tracker._load_zones():
        print("\n[OK] Using saved zones from player_zones.json")
        print("Zones will be used permanently for all tracking")
    else:
        # Mark new zones only if they don't exist
        print("\nNo saved zones found. Please mark zones...")
        tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not tracker.mark_zones():
            print("Zone marking cancelled")
            return
    
    # Tune parameters
    tracker.tune_parameters()
    
    tracker.cap.release()
    print("\nDone!")


if __name__ == "__main__":
    main()

