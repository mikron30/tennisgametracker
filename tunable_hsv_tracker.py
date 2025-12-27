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

class TunableHSVTracker:
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
        
        # MOG2 for initial detection only
        self.p1_mog2 = None
        self.p2_mog2 = None
        
        # HSV filters - will be tunable
        self.p1_hsv = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}
        self.p2_hsv = {'h_min': 0, 'h_max': 179, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 255}
        
        # Tracking mode
        self.mode = "MOG2_INIT"  # "MOG2_INIT", "TUNING", "TRACKING"
        self.max_movement_per_frame = 150  # pixels - players move slowly
        
        # Player positions
        self.p1_position = None
        self.p2_position = None
        
        print("=" * 70)
        print("TUNABLE HSV TRACKER")
        print("=" * 70)
        print("1. Find players with MOG2")
        print("2. Learn shirt colors")
        print("3. TUNE HSV filters interactively")
        print("4. Track with HSV + movement constraints")
        print("=" * 70)
        
        cv2.namedWindow('Main View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Main View', 1600, 900)
        
        self.init_mog2_and_find_players()
    
    def init_mog2_and_find_players(self):
        """Initialize MOG2 and find both players"""
        print("\n[STEP 1] Initializing MOG2...")
        
        self.p1_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=self.var_threshold, detectShadows=True)
        self.p2_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=self.var_threshold, detectShadows=True)
        
        # Warmup MOG2
        print("[WARMUP] Learning background...")
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                self.apply_mog2(frame, learn_only=True)
        
        # Search for both players
        print("[SEARCH] Looking for both players...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_idx in range(min(50, self.total_frames)):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            p1_det, p2_det, _, _ = self.apply_mog2(frame)
            
            if p1_det is not None:
                self.p1_position = p1_det
            if p2_det is not None:
                self.p2_position = p2_det
            
            if self.p1_position and self.p2_position:
                self.current_frame = frame_idx
                print(f"[FOUND] Both players at frame {frame_idx}!")
                
                # Learn colors
                self.learn_shirt_colors(frame)
                
                # Switch to tuning mode
                self.mode = "TUNING"
                self.setup_tuning_windows()
                
                # Reset to frame 0 for tracking
                self.current_frame = 0
                self.p1_position = None
                self.p2_position = None
                
                return
        
        print("[ERROR] Could not find both players in first 50 frames!")
    
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
        
        p1_fg = self.p1_mog2.apply(frame, learningRate=0)
        p2_fg = self.p2_mog2.apply(frame, learningRate=0)
        
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
    
    def learn_shirt_colors(self, frame):
        """Learn shirt colors from torso regions"""
        print("\n[STEP 2] Learning shirt colors...")
        
        for player_name, detection, hsv_dict in [
            ("P1", self.p1_position, self.p1_hsv),
            ("P2", self.p2_position, self.p2_hsv)
        ]:
            if detection is None:
                continue
            
            x, y, w, h = detection['bbox']
            
            # Sample from torso
            torso_x1 = x + int(w * 0.35)
            torso_x2 = x + int(w * 0.65)
            torso_y1 = y + int(h * 0.20)
            torso_y2 = y + int(h * 0.60)
            
            torso_x1 = max(0, torso_x1)
            torso_x2 = min(frame.shape[1], torso_x2)
            torso_y1 = max(0, torso_y1)
            torso_y2 = min(frame.shape[0], torso_y2)
            
            torso_region = frame[torso_y1:torso_y2, torso_x1:torso_x2]
            
            if torso_region.size == 0:
                continue
            
            hsv_torso = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            pixels = hsv_torso.reshape(-1, 3)
            
            mask = (pixels[:, 1] > 25) & (pixels[:, 2] > 60) & (pixels[:, 2] < 210)
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < 50:
                filtered_pixels = pixels
            
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
            
            hsv_dict['h_min'] = max(0, int(h_median - 1.5 * h_iqr))
            hsv_dict['h_max'] = min(179, int(h_median + 1.5 * h_iqr))
            hsv_dict['s_min'] = max(15, int(s_median - 1.5 * s_iqr))
            hsv_dict['s_max'] = min(255, int(s_median + 1.5 * s_iqr))
            hsv_dict['v_min'] = max(50, int(v_median - 1.5 * v_iqr))
            hsv_dict['v_max'] = min(220, int(v_median + 1.5 * v_iqr))
            
            print(f"{player_name}: H({hsv_dict['h_min']}-{hsv_dict['h_max']}) "
                  f"S({hsv_dict['s_min']}-{hsv_dict['s_max']}) "
                  f"V({hsv_dict['v_min']}-{hsv_dict['v_max']})")
    
    def setup_tuning_windows(self):
        """Setup HSV tuning windows with trackbars"""
        print("\n[STEP 3] Setting up HSV tuning windows...")
        print("Adjust trackbars to fine-tune detection, then press 'T' to start tracking")
        
        # P1 tuning window
        cv2.namedWindow('P1 HSV Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('P1 HSV Tuner', 600, 400)
        
        cv2.createTrackbar('H min', 'P1 HSV Tuner', self.p1_hsv['h_min'], 179, lambda x: self.update_hsv('p1', 'h_min', x))
        cv2.createTrackbar('H max', 'P1 HSV Tuner', self.p1_hsv['h_max'], 179, lambda x: self.update_hsv('p1', 'h_max', x))
        cv2.createTrackbar('S min', 'P1 HSV Tuner', self.p1_hsv['s_min'], 255, lambda x: self.update_hsv('p1', 's_min', x))
        cv2.createTrackbar('S max', 'P1 HSV Tuner', self.p1_hsv['s_max'], 255, lambda x: self.update_hsv('p1', 's_max', x))
        cv2.createTrackbar('V min', 'P1 HSV Tuner', self.p1_hsv['v_min'], 255, lambda x: self.update_hsv('p1', 'v_min', x))
        cv2.createTrackbar('V max', 'P1 HSV Tuner', self.p1_hsv['v_max'], 255, lambda x: self.update_hsv('p1', 'v_max', x))
        
        # P2 tuning window
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
    
    def is_in_zone(self, center, zone_polygon):
        """Check if center point is inside zone"""
        result = cv2.pointPolygonTest(zone_polygon, center, False)
        return result >= 0
    
    def is_movement_valid(self, new_position, last_position):
        """Check if movement between frames is reasonable"""
        if last_position is None:
            return True
        
        last_cx, last_cy = last_position['center']
        new_cx, new_cy = new_position['center']
        
        distance = np.sqrt((new_cx - last_cx)**2 + (new_cy - last_cy)**2)
        
        return distance <= self.max_movement_per_frame
    
    def detect_by_hsv(self, frame, hsv_dict, zone_polygon, min_area, max_area, last_position=None):
        """Detect player using HSV filter with validation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([hsv_dict['h_min'], hsv_dict['s_min'], hsv_dict['v_min']])
        upper = np.array([hsv_dict['h_max'], hsv_dict['s_max'], hsv_dict['v_max']])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        
        # Apply zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        hsv_mask = cv2.bitwise_and(hsv_mask, zone_mask)
        
        # If we have last position, restrict search
        if last_position is not None:
            last_cx, last_cy = last_position['center']
            search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(search_mask, (last_cx, last_cy), self.max_movement_per_frame, 255, -1)
            hsv_mask = cv2.bitwise_and(hsv_mask, search_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, hsv_mask
        
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (min_area <= area <= max_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            
            if not self.is_in_zone((cx, cy), zone_polygon):
                continue
            
            detection = {
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area
            }
            
            # Check movement constraint
            if not self.is_movement_valid(detection, last_position):
                continue
            
            valid_detections.append(detection)
        
        if not valid_detections:
            return None, hsv_mask
        
        valid_detections.sort(key=lambda x: x['area'], reverse=True)
        return valid_detections[0], hsv_mask
    
    def draw_results(self, frame, p1_det, p2_det, p1_mask=None, p2_mask=None):
        """Draw detection results"""
        display = frame.copy()
        
        # Draw zones
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
            
            # Movement circle
            if self.p1_position:
                cv2.circle(display, (cx, cy), self.max_movement_per_frame, (0, 255, 255), 1)
        
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
            
            # Movement circle
            if self.p2_position:
                cv2.circle(display, (cx, cy), self.max_movement_per_frame, (0, 255, 255), 1)
        
        # Status bar
        cv2.rectangle(display, (0, 0), (800, 200), (0, 0, 0), -1)
        
        mode_text = "TUNING (press 'T' to track)" if self.mode == "TUNING" else "TRACKING"
        cv2.putText(display, f"MODE: {mode_text}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
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
        
        # Show HSV values
        cv2.putText(display, f"P1 HSV: H{self.p1_hsv['h_min']}-{self.p1_hsv['h_max']} "
                   f"S{self.p1_hsv['s_min']}-{self.p1_hsv['s_max']} "
                   f"V{self.p1_hsv['v_min']}-{self.p1_hsv['v_max']}", 
                   (10, display.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(display, f"P2 HSV: H{self.p2_hsv['h_min']}-{self.p2_hsv['h_max']} "
                   f"S{self.p2_hsv['s_min']}-{self.p2_hsv['s_max']} "
                   f"V{self.p2_hsv['v_min']}-{self.p2_hsv['v_max']}", 
                   (10, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show tuning windows with masks
        if self.mode == "TUNING" and p1_mask is not None and p2_mask is not None:
            # P1 tuner
            p1_vis = cv2.cvtColor(p1_mask, cv2.COLOR_GRAY2BGR)
            p1_vis = cv2.resize(p1_vis, (600, 400))
            cv2.putText(p1_vis, "P1 SHIRT DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('P1 HSV Tuner', p1_vis)
            
            # P2 tuner
            p2_vis = cv2.cvtColor(p2_mask, cv2.COLOR_GRAY2BGR)
            p2_vis = cv2.resize(p2_vis, (600, 400))
            cv2.putText(p2_vis, "P2 SHIRT DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('P2 HSV Tuner', p2_vis)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n[STEP 4] Starting tracking loop...")
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'T' = Start tracking (from tuning mode)")
        print("  SPACE = Play/pause")
        print("  'Q' = Quit")
        print("=" * 70)
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            # Detect players with HSV
            p1_det, p1_mask = self.detect_by_hsv(
                frame, self.p1_hsv, self.p1_zone,
                self.p1_min_area, self.p1_max_area, self.p1_position)
            
            p2_det, p2_mask = self.detect_by_hsv(
                frame, self.p2_hsv, self.p2_zone,
                self.p2_min_area, self.p2_max_area, self.p2_position)
            
            # Update positions only if detected
            if p1_det is not None:
                self.p1_position = p1_det
            if p2_det is not None:
                self.p2_position = p2_det
            
            display = self.draw_results(frame, p1_det, p2_det, p1_mask, p2_mask)
            
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Main View', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                if self.mode == "TRACKING":
                    print(f"Frame {self.current_frame}: P1={p1_det is not None}, P2={p2_det is not None}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
            elif key == ord('t') and self.mode == "TUNING":
                self.mode = "TRACKING"
                print("\n[TRACKING] Started! Use 'D' to step forward or SPACE to play")
            elif key == ord(' '):
                if self.mode == "TRACKING":
                    self.paused = not self.paused
                    print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('q'):
                break
            
            if not self.paused and self.mode == "TRACKING":
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = TunableHSVTracker(video_path)
    tracker.run()

