import cv2
import json
import numpy as np

def load_zones():
    """Load P1 and P2 zones"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

class InteractivePlayerTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 10  # Start at frame 10
        self.paused = True
        
        # Load zones
        self.p1_zone, self.p2_zone = load_zones()
        
        # Color presets for auto-learn
        self.color_presets = {
            'White': [(0, 180, 0, 30, 200, 255)],
            'Black': [(0, 180, 0, 255, 0, 50)],
            'Red': [(0, 10, 100, 255, 100, 255), (170, 180, 100, 255, 100, 255)],
            'Orange': [(10, 25, 100, 255, 100, 255)],
            'Yellow': [(25, 35, 100, 255, 100, 255)],
            'Green': [(35, 85, 50, 255, 50, 255)],
            'Cyan': [(85, 95, 50, 255, 50, 255)],
            'Blue': [(95, 130, 50, 255, 50, 255)],
            'Purple': [(130, 160, 50, 255, 50, 255)],
            'Pink': [(160, 170, 50, 255, 100, 255)],
            'Gray': [(0, 180, 0, 50, 50, 200)],
            'Light_Blue': [(95, 115, 30, 100, 150, 255)],
            'Dark_Blue': [(100, 130, 100, 255, 50, 150)],
            'Light_Green': [(35, 85, 30, 100, 150, 255)],
            'Dark_Green': [(35, 85, 100, 255, 50, 150)],
        }
        
        # HSV filters (will be learned from clicks)
        self.p1_hsv_filter = None
        self.p2_hsv_filter = None
        
        # Tracking state
        self.p1_position = None
        self.p2_position = None
        self.search_window_size = 180
        self.miss_expansion = 1.8
        self.min_area = 900
        self.morph_close_iters = 2
        self.morph_open_iters = 1
        self.center_smooth_weight = 0.6
        
        # Setup mode
        self.setup_mode = True
        self.p1_click_point = None
        self.p2_click_point = None
        
        # Current frame
        self.current_display_frame = None
        self.current_original_frame = None
        
        print("=" * 70)
        print("INTERACTIVE PLAYER TRACKER")
        print("=" * 70)
        print("SETUP MODE - Click on players to learn their shirt colors")
        print("Press 'T' to auto-learn colors from the current frame")
        print("=" * 70)
        print("\nSETUP STEPS:")
        print("1. Navigate to frame where BOTH players are clearly visible")
        print("2. Click on P1 (bottom zone player) to learn shirt color")
        print("3. Click on P2 (top zone player) to learn shirt color")
        print("4. Tracking starts automatically!")
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10)")
        print("  'B' = Fast backward (-10)")
        print("  'S' = Show debug masks")
        print("  'T' = Auto-learn colors from current frame")
        print("  SPACE = Play/pause")
        print("  'R' = Restart from frame 0")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Interactive Player Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Interactive Player Tracker', 1600, 900)
        cv2.setMouseCallback('Interactive Player Tracker', self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to learn player colors"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if not self.setup_mode:
            return
        
        # Scale coordinates back to original frame size
        if self.current_original_frame is not None:
            orig_h, orig_w = self.current_original_frame.shape[:2]
            # We display at 0.5 scale
            x_orig = int(x * 2)
            y_orig = int(y * 2)
            
            # Check which zone the click is in
            p1_in_zone = cv2.pointPolygonTest(self.p1_zone, (x_orig, y_orig), False) >= 0
            p2_in_zone = cv2.pointPolygonTest(self.p2_zone, (x_orig, y_orig), False) >= 0
            
            if p1_in_zone and self.p1_hsv_filter is None:
                print(f"\n[CLICK] P1 at ({x_orig}, {y_orig})")
                self.p1_click_point = (x_orig, y_orig)
                self.learn_color_from_click(self.current_original_frame, x_orig, y_orig, "P1")
            elif p2_in_zone and self.p2_hsv_filter is None:
                print(f"\n[CLICK] P2 at ({x_orig}, {y_orig})")
                self.p2_click_point = (x_orig, y_orig)
                self.learn_color_from_click(self.current_original_frame, x_orig, y_orig, "P2")
            else:
                print(f"\n[CLICK] Click inside {'P1' if self.p1_hsv_filter is None else 'P2'} zone!")
            
            # Check if setup is complete
            if self.p1_hsv_filter is not None and self.p2_hsv_filter is not None:
                self.setup_mode = False
                self.current_frame = 0
                print("\n" + "=" * 70)
                print("SETUP COMPLETE! Starting tracking from frame 0...")
                print("=" * 70)
    
    def learn_color_from_click(self, frame, x, y, player_name):
        """Learn shirt color from a clicked point"""
        print(f"Learning {player_name} shirt color from click at ({x}, {y})...")
        
        # Sample a region around the click (30x30 pixels)
        sample_size = 30
        x1 = max(0, x - sample_size // 2)
        y1 = max(0, y - sample_size // 2)
        x2 = min(frame.shape[1], x + sample_size // 2)
        y2 = min(frame.shape[0], y + sample_size // 2)
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            print("[ERROR] Invalid region!")
            return
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Calculate median and std for HSV
        h_values = hsv_region[:, :, 0].flatten()
        s_values = hsv_region[:, :, 1].flatten()
        v_values = hsv_region[:, :, 2].flatten()
        
        h_median = np.median(h_values)
        h_std = np.std(h_values)
        s_median = np.median(s_values)
        s_std = np.std(s_values)
        v_median = np.median(v_values)
        v_std = np.std(v_values)
        
        # Create HSV range (median ± 2.5*std for wider tolerance)
        h_min = max(0, int(h_median - 2.5 * h_std))
        h_max = min(179, int(h_median + 2.5 * h_std))
        s_min = max(0, int(s_median - 2.5 * s_std))
        s_max = min(255, int(s_median + 2.5 * s_std))
        v_min = max(0, int(v_median - 2.5 * v_std))
        v_max = min(255, int(v_median + 2.5 * v_std))
        
        hsv_filter = (h_min, h_max, s_min, s_max, v_min, v_max)
        
        print(f"[OK] {player_name} HSV filter learned:")
        print(f"  H: {h_min}-{h_max} (median: {int(h_median)})")
        print(f"  S: {s_min}-{s_max} (median: {int(s_median)})")
        print(f"  V: {v_min}-{v_max} (median: {int(v_median)})")
        
        if player_name == "P1":
            self.p1_hsv_filter = hsv_filter
        else:
            self.p2_hsv_filter = hsv_filter
    
    def detect_player(self, frame, hsv_filter, search_region=None, zone_polygon=None):
        """Detect player using HSV filter"""
        if hsv_filter is None:
            return None, None
        
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_filter
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create HSV mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        
        # Apply search region if provided
        if search_region is not None:
            x, y, w, h = search_region
            # Clamp to frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(search_mask, (x, y), (x+w, y+h), 255, -1)
            hsv_mask = cv2.bitwise_and(hsv_mask, search_mask)
        elif zone_polygon is not None:
            # Use zone as search region on first detection
            zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(zone_mask, [zone_polygon], 255)
            hsv_mask = cv2.bitwise_and(hsv_mask, zone_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iters)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iters)
        
        # Find contours
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, hsv_mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum area threshold
        if area < self.min_area:
            return None, hsv_mask
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2
        
        return {
            'bbox': (x, y, w, h),
            'center': (center_x, center_y),
            'area': area
        }, hsv_mask
    
    def evaluate_presets_for_zone(self, hsv, zone_polygon):
        """Find best matching preset for a zone"""
        zone_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        best = None
        for color_name, ranges in self.color_presets.items():
            for hsv_range in ranges:
                h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
                mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                mask = cv2.bitwise_and(mask, zone_mask)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iters)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_open_iters)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if best is None or area > best['area']:
                    best = {
                        'color': color_name,
                        'area': area,
                        'hsv_range': hsv_range
                    }
        return best
    
    def auto_learn_from_frame(self, frame):
        """Auto-learn shirt colors from current frame using presets"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        p1_best = self.evaluate_presets_for_zone(hsv, self.p1_zone)
        p2_best = self.evaluate_presets_for_zone(hsv, self.p2_zone)
        
        min_accept_area = self.min_area * 0.8
        
        if p1_best and p1_best['area'] >= min_accept_area:
            self.p1_hsv_filter = p1_best['hsv_range']
            print(f"[AUTO] P1 -> {p1_best['color']} area {int(p1_best['area'])} using {p1_best['hsv_range']}")
        else:
            print("[AUTO] P1 color not confident enough")
        
        if p2_best and p2_best['area'] >= min_accept_area:
            self.p2_hsv_filter = p2_best['hsv_range']
            print(f"[AUTO] P2 -> {p2_best['color']} area {int(p2_best['area'])} using {p2_best['hsv_range']}")
        else:
            print("[AUTO] P2 color not confident enough")
        
        if self.p1_hsv_filter and self.p2_hsv_filter:
            self.setup_mode = False
            self.current_frame = 0
            print("\n" + "=" * 70)
            print("AUTO SETUP COMPLETE! Starting tracking from frame 0...")
            print("=" * 70)
    
    def get_search_region(self, last_position, frame_shape, scale=1.0):
        """Get search region around last position"""
        if last_position is None:
            return None
        
        center_x, center_y = last_position['center']
        window = int(self.search_window_size * scale)
        
        search_x = max(0, center_x - window)
        search_y = max(0, center_y - window)
        search_w = min(window * 2, frame_shape[1] - search_x)
        search_h = min(window * 2, frame_shape[0] - search_y)
        
        return (search_x, search_y, search_w, search_h)

    def smooth_position(self, prev_position, new_position):
        """Exponentially smooth center to reduce jitter."""
        if prev_position is None or new_position is None:
            return new_position
        
        px, py = prev_position['center']
        nx, ny = new_position['center']
        alpha = self.center_smooth_weight
        sx = int(alpha * px + (1 - alpha) * nx)
        sy = int(alpha * py + (1 - alpha) * ny)
        
        # Shift bbox so it stays centered on the smoothed point
        x, y, w, h = new_position['bbox']
        dx = sx - nx
        dy = sy - ny
        smoothed_bbox = (x + dx, y + dy, w, h)
        
        return {
            'bbox': smoothed_bbox,
            'center': (sx, sy),
            'area': new_position['area']
        }
    
    def process_frame(self, frame):
        """Process frame and detect both players"""
        p1_mask = None
        p2_mask = None
        
        if self.setup_mode:
            return None, None, p1_mask, p2_mask
        
        # Detect P1
        p1_search = self.get_search_region(self.p1_position, frame.shape)
        p1_detection, p1_mask = self.detect_player(
            frame, self.p1_hsv_filter, p1_search, self.p1_zone if p1_search is None else None)
        # Retry with expanded search if missed
        if p1_detection is None and self.p1_position is not None:
            expanded_search = self.get_search_region(self.p1_position, frame.shape, scale=self.miss_expansion)
            p1_detection, p1_mask = self.detect_player(
                frame, self.p1_hsv_filter, expanded_search, None)
        if p1_detection is not None:
            p1_detection = self.smooth_position(self.p1_position, p1_detection)
            self.p1_position = p1_detection
        
        # Detect P2
        p2_search = self.get_search_region(self.p2_position, frame.shape)
        p2_detection, p2_mask = self.detect_player(
            frame, self.p2_hsv_filter, p2_search, self.p2_zone if p2_search is None else None)
        # Retry with expanded search if missed
        if p2_detection is None and self.p2_position is not None:
            expanded_search = self.get_search_region(self.p2_position, frame.shape, scale=self.miss_expansion)
            p2_detection, p2_mask = self.detect_player(
                frame, self.p2_hsv_filter, expanded_search, None)
        if p2_detection is not None:
            p2_detection = self.smooth_position(self.p2_position, p2_detection)
            self.p2_position = p2_detection
        
        return p1_detection, p2_detection, p1_mask, p2_mask
    
    def draw_results(self, frame, p1_detection, p2_detection):
        """Draw tracking results"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 255, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (255, 255, 0), 2)
        cv2.putText(display, "P1 Zone", tuple(self.p1_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, "P2 Zone", tuple(self.p2_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Setup mode instructions
        if self.setup_mode:
            if self.p1_hsv_filter is None:
                cv2.putText(display, "CLICK on P1 (in P1 zone) to learn color", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif self.p2_hsv_filter is None:
                cv2.putText(display, "CLICK on P2 (in P2 zone) to learn color", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # Draw P1 click point
                if self.p1_click_point:
                    cv2.circle(display, self.p1_click_point, 15, (0, 255, 0), 3)
            cv2.putText(display, "Press 'T' to AUTO-LEARN both colors", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        else:
            # Tracking mode
            # Draw P1
            if p1_detection is not None:
                x, y, w, h = p1_detection['bbox']
                cx, cy = p1_detection['center']
                
                # Bounding box
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Center marker
                cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(display, (cx, cy), 12, (0, 255, 255), 2)
                
                # Label
                cv2.putText(display, f"P1", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Search window
                p1_search = self.get_search_region(self.p1_position, frame.shape)
                if p1_search:
                    sx, sy, sw, sh = p1_search
                    cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
            
            # Draw P2
            if p2_detection is not None:
                x, y, w, h = p2_detection['bbox']
                cx, cy = p2_detection['center']
                
                # Bounding box
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Center marker
                cv2.circle(display, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(display, (cx, cy), 12, (0, 255, 255), 2)
                
                # Label
                cv2.putText(display, f"P2", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Search window
                p2_search = self.get_search_region(self.p2_position, frame.shape)
                if p2_search:
                    sx, sy, sw, sh = p2_search
                    cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
            
            # Status
            cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            p1_status = f"P1: {int(p1_detection['area'])}px" if p1_detection else "P1: NOT FOUND"
            p2_status = f"P2: {int(p2_detection['area'])}px" if p2_detection else "P2: NOT FOUND"
            
            cv2.putText(display, p1_status, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                       (0, 255, 0) if p1_detection else (0, 0, 255), 2)
            cv2.putText(display, p2_status, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 0) if p2_detection else (0, 0, 255), 2)
        
        return display
    
    def show_debug(self, p1_mask, p2_mask):
        """Show debug masks"""
        if p1_mask is not None:
            p1_display = cv2.resize(p1_mask, (int(p1_mask.shape[1]*0.5), int(p1_mask.shape[0]*0.5)))
            cv2.imshow('Debug - P1 Mask', p1_display)
        
        if p2_mask is not None:
            p2_display = cv2.resize(p2_mask, (int(p2_mask.shape[1]*0.5), int(p2_mask.shape[0]*0.5)))
            cv2.imshow('Debug - P2 Mask', p2_display)
        
        print(f"\n[DEBUG] Frame {self.current_frame}")
        if self.p1_hsv_filter:
            h_min, h_max, s_min, s_max, v_min, v_max = self.p1_hsv_filter
            print(f"P1 HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"P1 Position: {self.p1_position}")
        
        if self.p2_hsv_filter:
            h_min, h_max, s_min, s_max, v_min, v_max = self.p2_hsv_filter
            print(f"P2 HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"P2 Position: {self.p2_position}")
    
    def run(self):
        """Main tracking loop"""
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            self.current_original_frame = frame.copy()
            
            p1_detection, p2_detection, p1_mask, p2_mask = self.process_frame(frame)
            display = self.draw_results(frame, p1_detection, p2_detection)
            
            # Store for mouse callback
            self.current_display_frame = display
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Interactive Player Tracker', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('f'):
                self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('b'):
                self.current_frame = max(self.current_frame - 10, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('s') and not self.setup_mode:
                self.show_debug(p1_mask, p2_mask)
            elif key == ord('t') and self.setup_mode:
                self.auto_learn_from_frame(self.current_original_frame)
            elif key == ord(' '):
                if not self.setup_mode:
                    self.paused = not self.paused
                    print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('r'):
                self.current_frame = 0
                self.p1_position = None
                self.p2_position = None
                print("Restarting from frame 0")
            elif key == ord('q'):
                break
            
            if not self.paused and not self.setup_mode:
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = InteractivePlayerTracker(video_path)
    tracker.run()

