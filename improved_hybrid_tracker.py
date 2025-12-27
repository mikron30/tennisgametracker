import cv2
import json
import numpy as np

def load_zones():
    """Load P1 and P2 zones"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

def load_parameters():
    """Load MOG2 parameters"""
    try:
        with open('mog2_parameters.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'var_threshold': 16,
            'p1_min_area': 5000,
            'p1_max_area': 150000,
            'p2_min_area': 10000,
            'p2_max_area': 300000
        }

class ImprovedHybridTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.paused = True
        
        # Load zones and parameters
        self.p1_zone, self.p2_zone = load_zones()
        params = load_parameters()
        self.var_threshold = params['var_threshold']
        self.p1_min_area = params['p1_min_area']
        self.p1_max_area = params['p1_max_area']
        self.p2_min_area = params['p2_min_area']
        self.p2_max_area = params['p2_max_area']
        
        # MOG2 for initial detection
        self.p1_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=self.var_threshold,
            detectShadows=True
        )
        self.p2_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=self.var_threshold,
            detectShadows=True
        )
        
        # HSV filters (learned from MOG2 detections)
        self.p1_hsv_filter = None
        self.p2_hsv_filter = None
        
        # Tracking mode
        self.mode = "MOG2"  # "MOG2" or "HSV"
        self.learning_rate = 0.001
        self.search_window_size = 250
        
        # Player positions
        self.p1_position = None
        self.p2_position = None
        
        print("=" * 70)
        print("IMPROVED HYBRID TRACKER - MOG2 + HSV with Validation")
        print("=" * 70)
        print("Strategy:")
        print("  1. Use MOG2 to detect players")
        print("  2. Learn DOMINANT shirt colors (not full region)")
        print("  3. Switch to HSV with zone + size validation")
        print("=" * 70)
        print(f"Sensitivity: {self.var_threshold}")
        print(f"P1 area range: {self.p1_min_area} - {self.p1_max_area}")
        print(f"P2 area range: {self.p2_min_area} - {self.p2_max_area}")
        print("=" * 70)
        print("\nWarmup: Learning background from first 20 frames...")
        
        # Warmup phase
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                self.apply_mog2_to_zones(frame, learn_only=True)
        
        print("Warmup complete!")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("\n" + "=" * 70)
        print("MODE: MOG2 Detection (will auto-switch to HSV)")
        print("=" * 70)
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10)")
        print("  'S' = Show debug window")
        print("  'L' = Learn colors NOW (switch to HSV mode)")
        print("  'M' = Toggle mode (MOG2 <-> HSV)")
        print("  SPACE = Play/pause")
        print("  'R' = Restart")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Improved Hybrid Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Improved Hybrid Tracker', 1600, 900)
    
    def apply_mog2_to_zones(self, frame, learn_only=False):
        """Apply MOG2 separately to each zone"""
        p1_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p1_zone_mask, [self.p1_zone], 255)
        
        p2_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p2_zone_mask, [self.p2_zone], 255)
        
        if learn_only:
            self.p1_mog2.apply(frame, learningRate=self.learning_rate)
            self.p2_mog2.apply(frame, learningRate=self.learning_rate)
            return None, None, None, None
        
        # Adaptive learning rate
        p1_lr = -1 if self.p1_position is not None else self.learning_rate
        p2_lr = -1 if self.p2_position is not None else self.learning_rate
        
        p1_fg = self.p1_mog2.apply(frame, learningRate=p1_lr)
        p2_fg = self.p2_mog2.apply(frame, learningRate=p2_lr)
        
        # Apply zone masks
        p1_fg = cv2.bitwise_and(p1_fg, p1_zone_mask)
        p2_fg = cv2.bitwise_and(p2_fg, p2_zone_mask)
        
        # Remove shadows
        p1_fg[p1_fg == 127] = 0
        p2_fg[p2_fg == 127] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        p1_fg = cv2.morphologyEx(p1_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        p1_fg = cv2.morphologyEx(p1_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        p2_fg = cv2.morphologyEx(p2_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        p2_fg = cv2.morphologyEx(p2_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Detect players
        p1_detection = self.detect_in_mask(p1_fg, self.p1_min_area, self.p1_max_area)
        p2_detection = self.detect_in_mask(p2_fg, self.p2_min_area, self.p2_max_area)
        
        return p1_detection, p2_detection, p1_fg, p2_fg
    
    def detect_in_mask(self, mask, min_area, max_area):
        """Detect player in mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
        
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        largest_contour, largest_area = valid_contours[0]
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx = x + w // 2
        cy = y + h // 2
        
        return {
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'area': largest_area,
            'mask_region': mask[y:y+h, x:x+w] if h > 0 and w > 0 else None
        }
    
    def learn_color_from_detection(self, frame, detection, player_name):
        """Learn DOMINANT shirt color from MOG2 detection - more selective"""
        if detection is None:
            print(f"[ERROR] Cannot learn {player_name} - no detection!")
            return None
        
        x, y, w, h = detection['bbox']
        
        # Extract player region
        player_region = frame[y:y+h, x:x+w]
        
        if player_region.size == 0:
            print(f"[ERROR] Empty region for {player_name}!")
            return None
        
        # Use the mask to get only player pixels
        mask_region = detection.get('mask_region')
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
        
        if mask_region is not None and mask_region.shape[:2] == hsv_region.shape[:2]:
            # Get only pixels that are foreground
            player_pixels = hsv_region[mask_region > 0]
        else:
            # Use all pixels in bounding box
            player_pixels = hsv_region.reshape(-1, 3)
        
        if len(player_pixels) < 100:
            print(f"[ERROR] Not enough pixels for {player_name}!")
            return None
        
        # Filter out very dark pixels (shadows) and very bright pixels (court lines)
        # Keep pixels with moderate saturation and value (likely shirt)
        filtered_mask = (player_pixels[:, 1] > 30) & (player_pixels[:, 2] > 50) & (player_pixels[:, 2] < 220)
        filtered_pixels = player_pixels[filtered_mask]
        
        if len(filtered_pixels) < 100:
            print(f"[WARNING] Not enough filtered pixels, using all pixels for {player_name}")
            filtered_pixels = player_pixels
        
        # Calculate HSV statistics on filtered pixels
        h_values = filtered_pixels[:, 0]
        s_values = filtered_pixels[:, 1]
        v_values = filtered_pixels[:, 2]
        
        # Use percentiles instead of median for more robust color detection
        h_median = np.percentile(h_values, 50)
        h_std = np.std(h_values)
        s_median = np.percentile(s_values, 50)
        s_std = np.std(s_values)
        v_median = np.percentile(v_values, 50)
        v_std = np.std(v_values)
        
        # More conservative range (2.0*std instead of 2.5*std)
        h_min = max(0, int(h_median - 2.0 * h_std))
        h_max = min(179, int(h_median + 2.0 * h_std))
        s_min = max(20, int(s_median - 2.0 * s_std))  # Minimum saturation of 20
        s_max = min(255, int(s_median + 2.0 * s_std))
        v_min = max(40, int(v_median - 2.0 * v_std))  # Minimum value of 40
        v_max = min(230, int(v_median + 2.0 * v_std))  # Maximum value of 230
        
        # Extra validation: if hue range is too wide, narrow it more
        if (h_max - h_min) > 60:
            h_min = max(0, int(h_median - 1.5 * h_std))
            h_max = min(179, int(h_median + 1.5 * h_std))
            print(f"  [INFO] Narrowed hue range for {player_name}")
        
        hsv_filter = (h_min, h_max, s_min, s_max, v_min, v_max)
        
        print(f"\n[OK] {player_name} HSV filter learned from MOG2 detection:")
        print(f"  H: {h_min}-{h_max} (median: {int(h_median)}, range: {h_max-h_min})")
        print(f"  S: {s_min}-{s_max} (median: {int(s_median)})")
        print(f"  V: {v_min}-{v_max} (median: {int(v_median)})")
        print(f"  Sampled {len(filtered_pixels)} filtered pixels from {len(player_pixels)} total")
        
        return hsv_filter
    
    def is_in_zone(self, center, zone_polygon):
        """Check if center point is inside zone"""
        result = cv2.pointPolygonTest(zone_polygon, center, False)
        return result >= 0
    
    def detect_by_hsv(self, frame, hsv_filter, zone_polygon, min_area, max_area, search_region=None):
        """Detect player using HSV filter with zone and size validation"""
        if hsv_filter is None:
            return None, None
        
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_filter
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        
        # Apply zone mask ALWAYS (not just search region)
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        # If search region provided, combine with zone
        if search_region is not None:
            x, y, w, h = search_region
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(search_mask, (x, y), (x+w, y+h), 255, -1)
            combined_mask = cv2.bitwise_and(zone_mask, search_mask)
            hsv_mask = cv2.bitwise_and(hsv_mask, combined_mask)
        else:
            hsv_mask = cv2.bitwise_and(hsv_mask, zone_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, hsv_mask
        
        # Find valid contours (within size range and inside zone)
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size validation
            if not (min_area <= area <= max_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            
            # Zone validation (center must be in zone)
            if not self.is_in_zone((cx, cy), zone_polygon):
                continue
            
            valid_detections.append({
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area
            })
        
        if not valid_detections:
            return None, hsv_mask
        
        # Return largest valid detection
        valid_detections.sort(key=lambda x: x['area'], reverse=True)
        return valid_detections[0], hsv_mask
    
    def get_search_region(self, last_position, frame_shape):
        """Get search region around last position"""
        if last_position is None:
            return None
        
        cx, cy = last_position['center']
        
        search_x = max(0, cx - self.search_window_size)
        search_y = max(0, cy - self.search_window_size)
        search_w = min(self.search_window_size * 2, frame_shape[1] - search_x)
        search_h = min(self.search_window_size * 2, frame_shape[0] - search_y)
        
        return (search_x, search_y, search_w, search_h)
    
    def process_frame(self, frame):
        """Process frame with current mode"""
        if self.mode == "MOG2":
            # Use MOG2 detection
            p1_det, p2_det, p1_mask, p2_mask = self.apply_mog2_to_zones(frame)
            
            if p1_det is not None:
                self.p1_position = p1_det
            if p2_det is not None:
                self.p2_position = p2_det
            
            # Auto-learn colors if both detected
            if (self.p1_position is not None and self.p2_position is not None and 
                self.p1_hsv_filter is None and self.p2_hsv_filter is None):
                
                print("\n" + "=" * 70)
                print("Both players detected! Learning colors...")
                print("=" * 70)
                
                self.p1_hsv_filter = self.learn_color_from_detection(frame, self.p1_position, "P1")
                self.p2_hsv_filter = self.learn_color_from_detection(frame, self.p2_position, "P2")
                
                if self.p1_hsv_filter and self.p2_hsv_filter:
                    self.mode = "HSV"
                    print("\n" + "=" * 70)
                    print("SWITCHED TO HSV COLOR TRACKING MODE")
                    print("=" * 70)
            
            return p1_det, p2_det, p1_mask, p2_mask
        
        else:  # HSV mode with validation
            # Use HSV color tracking with zone + size validation
            p1_search = self.get_search_region(self.p1_position, frame.shape)
            p1_det, p1_mask = self.detect_by_hsv(
                frame, self.p1_hsv_filter, self.p1_zone,
                self.p1_min_area, self.p1_max_area, p1_search)
            
            p2_search = self.get_search_region(self.p2_position, frame.shape)
            p2_det, p2_mask = self.detect_by_hsv(
                frame, self.p2_hsv_filter, self.p2_zone,
                self.p2_min_area, self.p2_max_area, p2_search)
            
            # Only update position if detection is valid
            if p1_det is not None:
                self.p1_position = p1_det
            if p2_det is not None:
                self.p2_position = p2_det
            
            return p1_det, p2_det, p1_mask, p2_mask
    
    def draw_results(self, frame, p1_det, p2_det):
        """Draw detection results"""
        display = frame.copy()
        
        # Draw zones (light)
        cv2.polylines(display, [self.p1_zone], True, (100, 100, 50), 1)
        cv2.polylines(display, [self.p2_zone], True, (100, 100, 50), 1)
        
        # Draw P1
        if p1_det is not None:
            x, y, w, h = p1_det['bbox']
            cx, cy = p1_det['center']
            
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.circle(display, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(display, (cx, cy), 15, (0, 255, 255), 3)
            
            cv2.putText(display, f"P1", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"{int(p1_det['area'])}px", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Search window
            if self.mode == "HSV":
                p1_search = self.get_search_region(self.p1_position, frame.shape)
                if p1_search:
                    sx, sy, sw, sh = p1_search
                    cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
        
        # Draw P2
        if p2_det is not None:
            x, y, w, h = p2_det['bbox']
            cx, cy = p2_det['center']
            
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.circle(display, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(display, (cx, cy), 15, (0, 255, 255), 3)
            
            cv2.putText(display, f"P2", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"{int(p2_det['area'])}px", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Search window
            if self.mode == "HSV":
                p2_search = self.get_search_region(self.p2_position, frame.shape)
                if p2_search:
                    sx, sy, sw, sh = p2_search
                    cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
        
        # Status bar
        cv2.rectangle(display, (0, 0), (700, 200), (0, 0, 0), -1)
        
        mode_color = (0, 255, 255) if self.mode == "HSV" else (255, 150, 0)
        cv2.putText(display, f"MODE: {self.mode}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        
        cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        p1_status = f"P1: {int(p1_det['area'])}px" if p1_det else "P1: NOT FOUND"
        p2_status = f"P2: {int(p2_det['area'])}px" if p2_det else "P2: NOT FOUND"
        
        cv2.putText(display, p1_status, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if p1_det else (0, 0, 255), 2)
        cv2.putText(display, p2_status, (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if p2_det else (0, 0, 255), 2)
        
        return display
    
    def show_debug(self, p1_mask, p2_mask):
        """Show debug windows"""
        if p1_mask is not None:
            cv2.imshow('Debug - P1 Mask', cv2.resize(p1_mask, 
                      (int(p1_mask.shape[1]*0.5), int(p1_mask.shape[0]*0.5))))
        
        if p2_mask is not None:
            cv2.imshow('Debug - P2 Mask', cv2.resize(p2_mask, 
                      (int(p2_mask.shape[1]*0.5), int(p2_mask.shape[0]*0.5))))
        
        print(f"\n[DEBUG] Frame {self.current_frame} - Mode: {self.mode}")
        print(f"P1: {self.p1_position}")
        print(f"P2: {self.p2_position}")
        
        if self.p1_hsv_filter:
            h_min, h_max, s_min, s_max, v_min, v_max = self.p1_hsv_filter
            print(f"P1 HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
        if self.p2_hsv_filter:
            h_min, h_max, s_min, s_max, v_min, v_max = self.p2_hsv_filter
            print(f"P2 HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
    
    def run(self):
        """Main loop"""
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            p1_det, p2_det, p1_mask, p2_mask = self.process_frame(frame)
            display = self.draw_results(frame, p1_det, p2_det)
            
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Improved Hybrid Tracker', display_resized)
            
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
            elif key == ord('s'):
                self.show_debug(p1_mask, p2_mask)
            elif key == ord('l'):
                if self.p1_position and self.p2_position:
                    print("\n[MANUAL] Learning colors from current detections...")
                    self.p1_hsv_filter = self.learn_color_from_detection(frame, self.p1_position, "P1")
                    self.p2_hsv_filter = self.learn_color_from_detection(frame, self.p2_position, "P2")
                    if self.p1_hsv_filter and self.p2_hsv_filter:
                        self.mode = "HSV"
                        print("Switched to HSV mode!")
                else:
                    print("[ERROR] Need both players detected first!")
            elif key == ord('m'):
                self.mode = "HSV" if self.mode == "MOG2" else "MOG2"
                print(f"Switched to {self.mode} mode")
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('r'):
                self.current_frame = 0
                self.p1_position = None
                self.p2_position = None
                print("Restarting...")
            elif key == ord('q'):
                break
            
            if not self.paused:
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = ImprovedHybridTracker(video_path)
    tracker.run()

