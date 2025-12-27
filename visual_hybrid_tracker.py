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

class VisualHybridTracker:
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
        
        # HSV filters
        self.p1_hsv_filter = None
        self.p2_hsv_filter = None
        
        # Tracking mode
        self.mode = "MOG2"
        self.learning_rate = 0.001
        self.search_window_size = 250
        
        # Player positions
        self.p1_position = None
        self.p2_position = None
        
        # Auto-pause flag
        self.auto_paused_for_learning = False
        
        print("=" * 70)
        print("VISUAL HYBRID TRACKER - Shows what is being sampled")
        print("=" * 70)
        print("Will AUTO-PAUSE when both players detected to show sampling")
        print("=" * 70)
        
        # Warmup
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                self.apply_mog2_to_zones(frame, learn_only=True)
        
        print("Warmup complete!")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'L' = Learn colors and switch to HSV mode")
        print("  'S' = Show debug masks")
        print("  'M' = Toggle mode (MOG2 <-> HSV)")
        print("  SPACE = Play/pause")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Main View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Main View', 1600, 900)
    
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
        
        p1_lr = -1 if self.p1_position is not None else self.learning_rate
        p2_lr = -1 if self.p2_position is not None else self.learning_rate
        
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
    
    def show_sampling_visualization(self, frame, detection, player_name):
        """Show detailed visualization of what pixels are being sampled"""
        if detection is None:
            return
        
        x, y, w, h = detection['bbox']
        cx, cy = detection['center']
        
        # Extract player region
        player_region = frame[y:y+h, x:x+w].copy()
        mask_region = detection.get('mask_region')
        
        if player_region.size == 0:
            return
        
        # Create visualization
        vis_height = 400
        vis_width = 1200
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # 1. Original player region
        region_h, region_w = player_region.shape[:2]
        scale = min(300 / region_w, 350 / region_h)
        scaled_w = int(region_w * scale)
        scaled_h = int(region_h * scale)
        player_scaled = cv2.resize(player_region, (scaled_w, scaled_h))
        visualization[20:20+scaled_h, 20:20+scaled_w] = player_scaled
        cv2.putText(visualization, "Original Region", (20, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 2. Mask region
        if mask_region is not None and mask_region.shape[:2] == (h, w):
            mask_vis = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR)
            mask_scaled = cv2.resize(mask_vis, (scaled_w, scaled_h))
            visualization[20:20+scaled_h, 340:340+scaled_w] = mask_scaled
            cv2.putText(visualization, "MOG2 Mask", (340, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 3. Filtered pixels (what will be sampled)
        hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
        if mask_region is not None and mask_region.shape[:2] == hsv_region.shape[:2]:
            player_pixels = hsv_region[mask_region > 0]
        else:
            player_pixels = hsv_region.reshape(-1, 3)
        
        # Filter pixels
        filtered_mask_2d = np.zeros((h, w), dtype=np.uint8)
        if mask_region is not None and mask_region.shape[:2] == hsv_region.shape[:2]:
            # Create 2D filtered mask
            for i in range(h):
                for j in range(w):
                    if mask_region[i, j] > 0:
                        s = hsv_region[i, j, 1]
                        v = hsv_region[i, j, 2]
                        if s > 30 and 50 < v < 220:
                            filtered_mask_2d[i, j] = 255
        
        filtered_vis = cv2.cvtColor(filtered_mask_2d, cv2.COLOR_GRAY2BGR)
        filtered_scaled = cv2.resize(filtered_vis, (scaled_w, scaled_h))
        visualization[20:20+scaled_h, 660:660+scaled_w] = filtered_scaled
        cv2.putText(visualization, "Filtered (Sampled)", (660, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 4. Info text
        info_y = 20
        cv2.putText(visualization, f"{player_name} Color Sampling", (20, vis_height - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(visualization, f"Bbox: ({x}, {y}, {w}, {h})", (20, vis_height - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(visualization, f"Center: ({cx}, {cy})", (20, vis_height - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(visualization, f"Area: {int(detection['area'])} pixels", (20, vis_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show filtered pixel count
        filtered_count = np.sum(filtered_mask_2d > 0)
        cv2.putText(visualization, f"Sampled pixels: {filtered_count}", (20, vis_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow(f'{player_name} Sampling', visualization)
    
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
        
        # Apply zone mask ALWAYS
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
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
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
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
            
            valid_detections.append({
                'bbox': (x, y, w, h),
                'center': (cx, cy),
                'area': area
            })
        
        if not valid_detections:
            return None, hsv_mask
        
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
    
    def learn_colors_and_show(self, frame):
        """Learn colors and show visualization"""
        if self.p1_position is None or self.p2_position is None:
            print("[ERROR] Need both players detected!")
            return False
        
        print("\n" + "=" * 70)
        print("LEARNING COLORS - Showing what is sampled...")
        print("=" * 70)
        
        # Show sampling visualizations
        self.show_sampling_visualization(frame, self.p1_position, "P1")
        self.show_sampling_visualization(frame, self.p2_position, "P2")
        
        # Learn colors (simplified version - using mask center region)
        p1_x, p1_y, p1_w, p1_h = self.p1_position['bbox']
        p2_x, p2_y, p2_w, p2_h = self.p2_position['bbox']
        
        # Sample from center of detected region (where shirt is most likely)
        p1_cx, p1_cy = self.p1_position['center']
        p2_cx, p2_cy = self.p2_position['center']
        
        sample_size = 50
        
        # P1 sampling
        p1_sample = frame[
            max(0, p1_cy-sample_size):min(frame.shape[0], p1_cy+sample_size),
            max(0, p1_cx-sample_size):min(frame.shape[1], p1_cx+sample_size)
        ]
        
        # P2 sampling
        p2_sample = frame[
            max(0, p2_cy-sample_size):min(frame.shape[0], p2_cy+sample_size),
            max(0, p2_cx-sample_size):min(frame.shape[1], p2_cx+sample_size)
        ]
        
        # Learn HSV from samples
        for sample, player_name in [(p1_sample, "P1"), (p2_sample, "P2")]:
            if sample.size == 0:
                continue
            
            hsv_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
            pixels = hsv_sample.reshape(-1, 3)
            
            # Filter pixels
            filtered = pixels[(pixels[:, 1] > 30) & (pixels[:, 2] > 50) & (pixels[:, 2] < 220)]
            
            if len(filtered) < 100:
                filtered = pixels
            
            h_med = np.percentile(filtered[:, 0], 50)
            h_std = np.std(filtered[:, 0])
            s_med = np.percentile(filtered[:, 1], 50)
            s_std = np.std(filtered[:, 1])
            v_med = np.percentile(filtered[:, 2], 50)
            v_std = np.std(filtered[:, 2])
            
            h_min = max(0, int(h_med - 2.0 * h_std))
            h_max = min(179, int(h_med + 2.0 * h_std))
            s_min = max(20, int(s_med - 2.0 * s_std))
            s_max = min(255, int(s_med + 2.0 * s_std))
            v_min = max(40, int(v_med - 2.0 * v_std))
            v_max = min(230, int(v_med + 2.0 * v_std))
            
            if (h_max - h_min) > 60:
                h_min = max(0, int(h_med - 1.5 * h_std))
                h_max = min(179, int(h_med + 1.5 * h_std))
            
            hsv_filter = (h_min, h_max, s_min, s_max, v_min, v_max)
            
            print(f"\n{player_name} HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"  Sampled from center region at ({p1_cx if player_name=='P1' else p2_cx}, {p1_cy if player_name=='P1' else p2_cy})")
            
            if player_name == "P1":
                self.p1_hsv_filter = hsv_filter
            else:
                self.p2_hsv_filter = hsv_filter
        
        print("\n" + "=" * 70)
        print("Press 'L' to accept and switch to HSV tracking")
        print("Press 'D' to try next frame")
        print("=" * 70)
        
        return True
    
    def process_frame(self, frame):
        """Process frame"""
        if self.mode == "MOG2":
            p1_det, p2_det, p1_mask, p2_mask = self.apply_mog2_to_zones(frame)
            
            if p1_det is not None:
                self.p1_position = p1_det
            if p2_det is not None:
                self.p2_position = p2_det
            
            # Auto-pause when both detected
            if (not self.auto_paused_for_learning and 
                self.p1_position is not None and self.p2_position is not None and
                self.p1_hsv_filter is None):
                
                print("\n[AUTO-PAUSE] Both players detected!")
                self.paused = True
                self.auto_paused_for_learning = True
                self.learn_colors_and_show(frame)
            
            return p1_det, p2_det, p1_mask, p2_mask
        
        else:  # HSV mode
            p1_search = self.get_search_region(self.p1_position, frame.shape)
            p1_det, p1_mask = self.detect_by_hsv(
                frame, self.p1_hsv_filter, self.p1_zone,
                self.p1_min_area, self.p1_max_area, p1_search)
            
            p2_search = self.get_search_region(self.p2_position, frame.shape)
            p2_det, p2_mask = self.detect_by_hsv(
                frame, self.p2_hsv_filter, self.p2_zone,
                self.p2_min_area, self.p2_max_area, p2_search)
            
            if p1_det is not None:
                self.p1_position = p1_det
            if p2_det is not None:
                self.p2_position = p2_det
            
            return p1_det, p2_det, p1_mask, p2_mask
    
    def draw_results(self, frame, p1_det, p2_det):
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
            
            # Show sample region in MOG2 mode
            if self.mode == "MOG2" and self.auto_paused_for_learning:
                sample_size = 50
                cv2.rectangle(display, 
                            (cx-sample_size, cy-sample_size), 
                            (cx+sample_size, cy+sample_size), 
                            (255, 0, 255), 2)
                cv2.putText(display, "Sample", (cx-40, cy-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
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
            
            # Show sample region in MOG2 mode
            if self.mode == "MOG2" and self.auto_paused_for_learning:
                sample_size = 50
                cv2.rectangle(display, 
                            (cx-sample_size, cy-sample_size), 
                            (cx+sample_size, cy+sample_size), 
                            (255, 0, 255), 2)
                cv2.putText(display, "Sample", (cx-40, cy-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
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
            cv2.imshow('Main View', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('l'):
                if self.p1_hsv_filter and self.p2_hsv_filter:
                    self.mode = "HSV"
                    print("\n[OK] Switched to HSV tracking mode!")
                elif self.p1_position and self.p2_position:
                    self.learn_colors_and_show(frame)
                else:
                    print("[ERROR] Need both players detected!")
            elif key == ord('s'):
                if p1_mask is not None:
                    cv2.imshow('Debug - P1 Mask', cv2.resize(p1_mask, 
                              (int(p1_mask.shape[1]*0.5), int(p1_mask.shape[0]*0.5))))
                if p2_mask is not None:
                    cv2.imshow('Debug - P2 Mask', cv2.resize(p2_mask, 
                              (int(p2_mask.shape[1]*0.5), int(p2_mask.shape[0]*0.5))))
            elif key == ord('m'):
                self.mode = "HSV" if self.mode == "MOG2" else "MOG2"
                print(f"Switched to {self.mode} mode")
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'PLAYING'}")
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
    
    tracker = VisualHybridTracker(video_path)
    tracker.run()

