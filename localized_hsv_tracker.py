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

class LocalizedHSVTracker:
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
        
        # MOG2
        self.p1_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=self.var_threshold, detectShadows=True)
        self.p2_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=self.var_threshold, detectShadows=True)
        
        # HSV filters
        self.p1_hsv = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}
        self.p2_hsv = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}
        
        # Tracking boxes - THIS IS KEY!
        self.p1_search_box = None  # (x, y, w, h) - only search in this box
        self.p2_search_box = None
        
        # Tracking mode
        self.mode = "MOG2"  # "MOG2", "TUNING", "HSV_TRACKING"
        self.max_movement_per_frame = 100  # Slow movement
        self.search_box_padding = 80  # Pixels to expand around last detection
        
        # Player positions
        self.p1_position = None
        self.p2_position = None
        
        print("=" * 70)
        print("LOCALIZED HSV TRACKER")
        print("=" * 70)
        print("Strategy: MOG2 finds bulb -> HSV tracks ONLY within bulb area")
        print("Search area moves with player (slow movement)")
        print("=" * 70)
        
        # Warmup MOG2
        print("\n[WARMUP] Learning background...")
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                self.apply_mog2(frame, learn_only=True)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("[WARMUP] Complete!")
        
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  SPACE = Play/pause")
        print("  'T' = Start HSV tracking (from TUNING mode)")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Main View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Main View', 1600, 900)
    
    def apply_mog2(self, frame, learn_only=False):
        """Apply MOG2 for initial detection"""
        p1_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p1_zone_mask, [self.p1_zone], 255)
        
        p2_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p2_zone_mask, [self.p2_zone], 255)
        
        if learn_only:
            self.p1_mog2.apply(frame, learningRate=0.0005)
            self.p2_mog2.apply(frame, learningRate=0.0005)
            return None, None, None, None
        
        p1_lr = 0 if self.p1_position is not None else 0.0005
        p2_lr = 0 if self.p2_position is not None else 0.0005
        
        p1_fg = self.p1_mog2.apply(frame, learningRate=p1_lr)
        p2_fg = self.p2_mog2.apply(frame, learningRate=p2_lr)
        
        p1_fg = cv2.bitwise_and(p1_fg, p1_zone_mask)
        p2_fg = cv2.bitwise_and(p2_fg, p2_zone_mask)
        
        p1_fg[p1_fg == 127] = 0
        p2_fg[p2_fg == 127] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        p1_fg = cv2.morphologyEx(p1_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        p1_fg = cv2.morphologyEx(p1_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        p2_fg = cv2.morphologyEx(p2_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        p2_fg = cv2.morphologyEx(p2_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        p1_det = self.detect_in_mask(p1_fg, self.p1_min_area, self.p1_max_area)
        p2_det = self.detect_in_mask(p2_fg, self.p2_min_area, self.p2_max_area)
        
        return p1_det, p2_det, p1_fg, p2_fg
    
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
            'area': largest_area
        }
    
    def learn_shirt_colors_from_bulb(self, frame, detection, player_name):
        """Learn HSV colors from the BULB area only (not torso, the whole detected bulb)"""
        if detection is None:
            return None
        
        x, y, w, h = detection['bbox']
        
        # Extract the BULB region (the detected player area)
        bulb_region = frame[y:y+h, x:x+w]
        
        if bulb_region.size == 0:
            return None
        
        # Convert to HSV
        hsv_bulb = cv2.cvtColor(bulb_region, cv2.COLOR_BGR2HSV)
        pixels = hsv_bulb.reshape(-1, 3)
        
        # Filter out very dark and very bright pixels
        mask = (pixels[:, 2] > 60) & (pixels[:, 2] < 210)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) < 50:
            filtered_pixels = pixels
        
        # Calculate statistics
        h_values = filtered_pixels[:, 0]
        s_values = filtered_pixels[:, 1]
        v_values = filtered_pixels[:, 2]
        
        h_p25 = np.percentile(h_values, 25)
        h_p75 = np.percentile(h_values, 75)
        h_median = np.percentile(h_values, 50)
        h_iqr = h_p75 - h_p25
        
        s_p25 = np.percentile(s_values, 25)
        s_p75 = np.percentile(s_values, 75)
        s_median = np.percentile(s_values, 50)
        s_iqr = s_p75 - s_p25
        
        v_p25 = np.percentile(v_values, 25)
        v_p75 = np.percentile(v_values, 75)
        v_median = np.percentile(v_values, 50)
        v_iqr = v_p75 - v_p25
        
        # Create HSV range
        hsv_dict = {}
        hsv_dict['h_min'] = max(0, int(h_median - 1.5 * h_iqr))
        hsv_dict['h_max'] = min(179, int(h_median + 1.5 * h_iqr))
        hsv_dict['s_min'] = max(15, int(s_median - 1.5 * s_iqr))
        hsv_dict['s_max'] = min(255, int(s_median + 1.5 * s_iqr))
        hsv_dict['v_min'] = max(50, int(v_median - 1.5 * v_iqr))
        hsv_dict['v_max'] = min(220, int(v_median + 1.5 * v_iqr))
        
        print(f"{player_name} HSV from bulb ({w}x{h}):")
        print(f"  H: {hsv_dict['h_min']:3d} - {hsv_dict['h_max']:3d}")
        print(f"  S: {hsv_dict['s_min']:3d} - {hsv_dict['s_max']:3d}")
        print(f"  V: {hsv_dict['v_min']:3d} - {hsv_dict['v_max']:3d}")
        
        return hsv_dict
    
    def create_search_box(self, detection):
        """Create search box around detection with padding"""
        if detection is None:
            return None
        
        x, y, w, h = detection['bbox']
        
        # Expand box with padding
        search_x = max(0, x - self.search_box_padding)
        search_y = max(0, y - self.search_box_padding)
        search_w = w + 2 * self.search_box_padding
        search_h = h + 2 * self.search_box_padding
        
        return (search_x, search_y, search_w, search_h)
    
    def setup_tuning_windows(self):
        """Setup HSV tuning windows"""
        cv2.namedWindow('P1 HSV Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('P1 HSV Tuner', 600, 400)
        
        cv2.createTrackbar('H min', 'P1 HSV Tuner', self.p1_hsv['h_min'], 179, lambda x: self.update_hsv('p1', 'h_min', x))
        cv2.createTrackbar('H max', 'P1 HSV Tuner', self.p1_hsv['h_max'], 179, lambda x: self.update_hsv('p1', 'h_max', x))
        cv2.createTrackbar('S min', 'P1 HSV Tuner', self.p1_hsv['s_min'], 255, lambda x: self.update_hsv('p1', 's_min', x))
        cv2.createTrackbar('S max', 'P1 HSV Tuner', self.p1_hsv['s_max'], 255, lambda x: self.update_hsv('p1', 's_max', x))
        cv2.createTrackbar('V min', 'P1 HSV Tuner', self.p1_hsv['v_min'], 255, lambda x: self.update_hsv('p1', 'v_min', x))
        cv2.createTrackbar('V max', 'P1 HSV Tuner', self.p1_hsv['v_max'], 255, lambda x: self.update_hsv('p1', 'v_max', x))
        
        cv2.namedWindow('P2 HSV Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('P2 HSV Tuner', 600, 400)
        
        cv2.createTrackbar('H min', 'P2 HSV Tuner', self.p2_hsv['h_min'], 179, lambda x: self.update_hsv('p2', 'h_min', x))
        cv2.createTrackbar('H max', 'P2 HSV Tuner', self.p2_hsv['h_max'], 179, lambda x: self.update_hsv('p2', 'h_max', x))
        cv2.createTrackbar('S min', 'P2 HSV Tuner', self.p2_hsv['s_min'], 255, lambda x: self.update_hsv('p2', 's_min', x))
        cv2.createTrackbar('S max', 'P2 HSV Tuner', self.p2_hsv['s_max'], 255, lambda x: self.update_hsv('p2', 's_max', x))
        cv2.createTrackbar('V min', 'P2 HSV Tuner', self.p2_hsv['v_min'], 255, lambda x: self.update_hsv('p2', 'v_min', x))
        cv2.createTrackbar('V max', 'P2 HSV Tuner', self.p2_hsv['v_max'], 255, lambda x: self.update_hsv('p2', 'v_max', x))
    
    def update_hsv(self, player, param, value):
        """Update HSV parameter"""
        if player == 'p1':
            self.p1_hsv[param] = value
        else:
            self.p2_hsv[param] = value
    
    def detect_by_hsv_in_box(self, frame, hsv_dict, search_box, zone_polygon):
        """Detect player using HSV ONLY within search box"""
        if search_box is None:
            return None, None
        
        sx, sy, sw, sh = search_box
        
        # Clamp to frame bounds
        sx = max(0, sx)
        sy = max(0, sy)
        sw = min(sw, frame.shape[1] - sx)
        sh = min(sh, frame.shape[0] - sy)
        
        if sw <= 0 or sh <= 0:
            return None, None
        
        # Extract ONLY the search box region
        search_region = frame[sy:sy+sh, sx:sx+sw]
        
        # Convert to HSV
        hsv_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
        
        # Apply HSV filter
        lower = np.array([hsv_dict['h_min'], hsv_dict['s_min'], hsv_dict['v_min']])
        upper = np.array([hsv_dict['h_max'], hsv_dict['s_max'], hsv_dict['v_max']])
        hsv_mask = cv2.inRange(hsv_region, lower, upper)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Return full mask for visualization
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[sy:sy+sh, sx:sx+sw] = hsv_mask
            return None, full_mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 1000:  # Minimum threshold
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[sy:sy+sh, sx:sx+sw] = hsv_mask
            return None, full_mask
        
        # Get bounding box (relative to search region)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to absolute coordinates
        abs_x = sx + x
        abs_y = sy + y
        abs_cx = abs_x + w // 2
        abs_cy = abs_y + h // 2
        
        # Check if center is in zone
        in_zone = cv2.pointPolygonTest(zone_polygon, (abs_cx, abs_cy), False) >= 0
        
        if not in_zone:
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[sy:sy+sh, sx:sx+sw] = hsv_mask
            return None, full_mask
        
        detection = {
            'bbox': (abs_x, abs_y, w, h),
            'center': (abs_cx, abs_cy),
            'area': area
        }
        
        # Create full mask for visualization
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[sy:sy+sh, sx:sx+sw] = hsv_mask
        
        return detection, full_mask
    
    def draw_results(self, frame, p1_det, p2_det, p1_mask=None, p2_mask=None):
        """Draw detection results"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (100, 100, 50), 2)
        cv2.polylines(display, [self.p2_zone], True, (100, 100, 50), 2)
        
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
        
        # Draw P1 search box
        if self.p1_search_box is not None and self.mode == "HSV_TRACKING":
            sx, sy, sw, sh = self.p1_search_box
            cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2)
            cv2.putText(display, "P1 SEARCH", (sx, sy-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
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
        
        # Draw P2 search box
        if self.p2_search_box is not None and self.mode == "HSV_TRACKING":
            sx, sy, sw, sh = self.p2_search_box
            cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2)
            cv2.putText(display, "P2 SEARCH", (sx, sy-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Status bar
        cv2.rectangle(display, (0, 0), (900, 200), (0, 0, 0), -1)
        
        mode_color = {
            "MOG2": (255, 150, 0),
            "TUNING": (0, 255, 255),
            "HSV_TRACKING": (0, 255, 0)
        }[self.mode]
        
        cv2.putText(display, f"MODE: {self.mode}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        
        if self.mode == "TUNING":
            cv2.putText(display, "Adjust HSV, press 'T' to track", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        p1_status = f"P1: {int(p1_det['area'])}px" if p1_det else "P1: NOT FOUND"
        p2_status = f"P2: {int(p2_det['area'])}px" if p2_det else "P2: NOT FOUND"
        
        cv2.putText(display, p1_status, (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if p1_det else (0, 0, 255), 2)
        cv2.putText(display, p2_status, (10, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if p2_det else (0, 0, 255), 2)
        
        # Show tuning windows
        if self.mode == "TUNING" and p1_mask is not None and p2_mask is not None:
            p1_vis = cv2.cvtColor(p1_mask, cv2.COLOR_GRAY2BGR)
            p1_vis = cv2.resize(p1_vis, (600, 400))
            cv2.putText(p1_vis, "P1 HSV (in search box only)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('P1 HSV Tuner', p1_vis)
            
            p2_vis = cv2.cvtColor(p2_mask, cv2.COLOR_GRAY2BGR)
            p2_vis = cv2.resize(p2_vis, (600, 400))
            cv2.putText(p2_vis, "P2 HSV (in search box only)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('P2 HSV Tuner', p2_vis)
        
        return display
    
    def run(self):
        """Main loop"""
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            # Process based on mode
            if self.mode == "MOG2":
                p1_det, p2_det, p1_mask, p2_mask = self.apply_mog2(frame)
                
                if p1_det is not None:
                    self.p1_position = p1_det
                if p2_det is not None:
                    self.p2_position = p2_det
                
                # When both found, switch to tuning
                if self.p1_position and self.p2_position:
                    print(f"\n[FOUND] Both players at frame {self.current_frame}!")
                    print("\n[LEARNING] Extracting HSV from bulb areas...")
                    
                    # Learn HSV from the detected bulb areas
                    self.p1_hsv = self.learn_shirt_colors_from_bulb(frame, self.p1_position, "P1")
                    self.p2_hsv = self.learn_shirt_colors_from_bulb(frame, self.p2_position, "P2")
                    
                    # Create initial search boxes
                    self.p1_search_box = self.create_search_box(self.p1_position)
                    self.p2_search_box = self.create_search_box(self.p2_position)
                    
                    print("\n[SETUP] Creating tuning windows...")
                    self.setup_tuning_windows()
                    
                    self.mode = "TUNING"
                    self.paused = True
                    print("\n[PAUSED] Adjust HSV if needed, then press 'T' to start tracking")
            
            elif self.mode == "TUNING":
                # Show HSV detection in real-time within search boxes
                p1_det, p1_mask = self.detect_by_hsv_in_box(
                    frame, self.p1_hsv, self.p1_search_box, self.p1_zone)
                
                p2_det, p2_mask = self.detect_by_hsv_in_box(
                    frame, self.p2_hsv, self.p2_search_box, self.p2_zone)
            
            elif self.mode == "HSV_TRACKING":
                # Track using HSV ONLY within search boxes
                p1_det, p1_mask = self.detect_by_hsv_in_box(
                    frame, self.p1_hsv, self.p1_search_box, self.p1_zone)
                
                p2_det, p2_mask = self.detect_by_hsv_in_box(
                    frame, self.p2_hsv, self.p2_search_box, self.p2_zone)
                
                # Update positions and search boxes
                if p1_det is not None:
                    self.p1_position = p1_det
                    self.p1_search_box = self.create_search_box(p1_det)
                    print(f"Frame {self.current_frame}: P1 tracked at ({p1_det['center'][0]}, {p1_det['center'][1]})")
                else:
                    print(f"Frame {self.current_frame}: P1 LOST")
                
                if p2_det is not None:
                    self.p2_position = p2_det
                    self.p2_search_box = self.create_search_box(p2_det)
                    print(f"Frame {self.current_frame}: P2 tracked at ({p2_det['center'][0]}, {p2_det['center'][1]})")
                else:
                    print(f"Frame {self.current_frame}: P2 LOST")
            
            display = self.draw_results(frame, p1_det, p2_det, p1_mask, p2_mask)
            
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Main View', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                if self.mode != "TUNING":
                    self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            elif key == ord('t') and self.mode == "TUNING":
                self.mode = "HSV_TRACKING"
                self.current_frame = 0
                self.p1_position = None
                self.p2_position = None
                self.p1_search_box = self.create_search_box(self.p1_position) if self.p1_position else None
                self.p2_search_box = self.create_search_box(self.p2_position) if self.p2_position else None
                print("\n[HSV_TRACKING] Started from frame 0!")
                print("HSV will only search within moving boxes around players")
            elif key == ord(' '):
                if self.mode != "TUNING":
                    self.paused = not self.paused
                    print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('q'):
                break
            
            if not self.paused and self.mode != "TUNING":
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = LocalizedHSVTracker(video_path)
    tracker.run()

