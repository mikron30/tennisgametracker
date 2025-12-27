import cv2
import json
import numpy as np

def load_zones():
    """Load P1 and P2 zones"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

class ShirtColorTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.paused = True
        
        # Load zones
        self.p1_zone, self.p2_zone = load_zones()
        
        # Player tracking state
        self.p1_position = None  # (x, y, w, h)
        self.p2_position = None
        self.p1_hsv_range = None  # (h_min, h_max, s_min, s_max, v_min, v_max)
        self.p2_hsv_range = None
        self.search_window_size = 200  # Search within ±200 pixels of last position
        
        # Setup mode
        self.setup_mode = True  # Start in setup mode to learn shirt colors
        self.setup_step = 0  # 0 = waiting to learn P1, 1 = waiting to learn P2, 2 = tracking
        
        print("=" * 70)
        print("SHIRT COLOR TRACKER")
        print("=" * 70)
        print("SETUP MODE - Learn player shirt colors")
        print("=" * 70)
        print("\nSETUP STEPS:")
        print("1. Navigate to a frame where P1 (bottom zone) is clearly visible")
        print("2. Press '1' to learn P1's shirt color")
        print("3. Navigate to a frame where P2 (top zone) is clearly visible")
        print("4. Press '2' to learn P2's shirt color")
        print("5. Tracking will start automatically!")
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10 frames)")
        print("  'B' = Fast backward (-10 frames)")
        print("  '1' = Learn P1 shirt color (in setup)")
        print("  '2' = Learn P2 shirt color (in setup)")
        print("  'S' = Show debug window")
        print("  SPACE = Play/pause")
        print("  'R' = Restart from frame 0")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Shirt Color Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Shirt Color Tracker', 1600, 900)
    
    def learn_shirt_color(self, frame, zone_polygon, player_name):
        """Learn dominant shirt color in a zone"""
        print(f"\nLearning {player_name} shirt color...")
        
        # Create zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get pixels in zone
        zone_hsv = cv2.bitwise_and(hsv, hsv, mask=zone_mask)
        
        # Get non-zero pixels (inside zone)
        pixels = zone_hsv[zone_mask > 0]
        
        if len(pixels) == 0:
            print(f"[ERROR] No pixels found in {player_name} zone!")
            return None
        
        # Calculate median and standard deviation for each channel
        h_values = pixels[:, 0]
        s_values = pixels[:, 1]
        v_values = pixels[:, 2]
        
        # Filter out very dark pixels (likely shadows/background)
        bright_mask = v_values > 50
        if np.sum(bright_mask) < 100:
            print(f"[ERROR] Not enough bright pixels in {player_name} zone!")
            return None
        
        h_bright = h_values[bright_mask]
        s_bright = s_values[bright_mask]
        v_bright = v_values[bright_mask]
        
        # Use median ± 2*std for robust color range
        h_median = np.median(h_bright)
        h_std = np.std(h_bright)
        s_median = np.median(s_bright)
        s_std = np.std(s_bright)
        v_median = np.median(v_bright)
        v_std = np.std(v_bright)
        
        # Create HSV range (with bounds checking)
        h_min = max(0, int(h_median - 2 * h_std))
        h_max = min(179, int(h_median + 2 * h_std))
        s_min = max(0, int(s_median - 2 * s_std))
        s_max = min(255, int(s_median + 2 * s_std))
        v_min = max(0, int(v_median - 2 * v_std))
        v_max = min(255, int(v_median + 2 * v_std))
        
        hsv_range = (h_min, h_max, s_min, s_max, v_min, v_max)
        
        print(f"[OK] {player_name} shirt HSV range learned:")
        print(f"  H: {h_min}-{h_max} (median: {int(h_median)})")
        print(f"  S: {s_min}-{s_max} (median: {int(s_median)})")
        print(f"  V: {v_min}-{v_max} (median: {int(v_median)})")
        
        return hsv_range
    
    def detect_player_by_shirt(self, frame, hsv_range, search_region=None):
        """Detect player by shirt color in optional search region"""
        if hsv_range is None:
            return None, None
        
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create search mask (if last position known, search nearby)
        if search_region is not None:
            x, y, w, h = search_region
            search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(search_mask, (x, y), (x+w, y+h), 255, -1)
        else:
            search_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        # Create HSV mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        
        # Combine with search mask
        mask = cv2.bitwise_and(hsv_mask, search_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum area threshold (shirt should be at least 2000 pixels)
        if area < 2000:
            return None, mask
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h), mask
    
    def get_search_region(self, last_position):
        """Create search region around last known position"""
        if last_position is None:
            return None
        
        x, y, w, h = last_position
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Search window: ±200 pixels from center
        search_x = max(0, center_x - self.search_window_size)
        search_y = max(0, center_y - self.search_window_size)
        search_w = self.search_window_size * 2
        search_h = self.search_window_size * 2
        
        return (search_x, search_y, search_w, search_h)
    
    def process_frame(self, frame):
        """Process frame - detect players by shirt color"""
        p1_bbox = None
        p2_bbox = None
        p1_mask = None
        p2_mask = None
        
        if self.setup_mode:
            # In setup mode, just show the frame
            return p1_bbox, p2_bbox, p1_mask, p2_mask
        
        # Detect P1
        p1_search = self.get_search_region(self.p1_position)
        p1_bbox, p1_mask = self.detect_player_by_shirt(frame, self.p1_hsv_range, p1_search)
        if p1_bbox is not None:
            self.p1_position = p1_bbox
        
        # Detect P2
        p2_search = self.get_search_region(self.p2_position)
        p2_bbox, p2_mask = self.detect_player_by_shirt(frame, self.p2_hsv_range, p2_search)
        if p2_bbox is not None:
            self.p2_position = p2_bbox
        
        return p1_bbox, p2_bbox, p1_mask, p2_mask
    
    def draw_results(self, frame, p1_bbox, p2_bbox):
        """Draw detection results"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 255, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (255, 255, 0), 2)
        cv2.putText(display, "P1 Zone", tuple(self.p1_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, "P2 Zone", tuple(self.p2_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw setup instructions
        if self.setup_mode:
            if self.setup_step == 0:
                cv2.putText(display, "SETUP: Press '1' to learn P1 shirt color", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif self.setup_step == 1:
                cv2.putText(display, "P1 learned! Press '2' to learn P2 shirt color", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Draw P1
            if p1_bbox is not None:
                x, y, w, h = p1_bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(display, "P1", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw search window
                p1_search = self.get_search_region(self.p1_position)
                if p1_search:
                    sx, sy, sw, sh = p1_search
                    cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
            else:
                cv2.putText(display, "P1: NOT FOUND", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw P2
            if p2_bbox is not None:
                x, y, w, h = p2_bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(display, "P2", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw search window
                p2_search = self.get_search_region(self.p2_position)
                if p2_search:
                    sx, sy, sw, sh = p2_search
                    cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
            else:
                cv2.putText(display, "P2: NOT FOUND", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(display, "TRACKING MODE", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Frame info
        cv2.putText(display, f"Frame: {self.current_frame}", (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return display
    
    def show_debug(self, frame, p1_mask, p2_mask):
        """Show debug windows with HSV masks"""
        if p1_mask is not None:
            cv2.imshow('Debug - P1 Shirt Mask', cv2.resize(p1_mask, 
                      (int(p1_mask.shape[1]*0.5), int(p1_mask.shape[0]*0.5))))
        
        if p2_mask is not None:
            cv2.imshow('Debug - P2 Shirt Mask', cv2.resize(p2_mask, 
                      (int(p2_mask.shape[1]*0.5), int(p2_mask.shape[0]*0.5))))
        
        print("\n" + "=" * 70)
        print(f"DEBUG - Frame {self.current_frame}")
        print("=" * 70)
        
        if self.p1_hsv_range:
            h_min, h_max, s_min, s_max, v_min, v_max = self.p1_hsv_range
            print(f"P1 HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"P1 Position: {self.p1_position}")
        
        if self.p2_hsv_range:
            h_min, h_max, s_min, s_max, v_min, v_max = self.p2_hsv_range
            print(f"P2 HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"P2 Position: {self.p2_position}")
        
        print("=" * 70)
    
    def run(self):
        """Main tracking loop"""
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            p1_bbox, p2_bbox, p1_mask, p2_mask = self.process_frame(frame)
            display = self.draw_results(frame, p1_bbox, p2_bbox)
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Shirt Color Tracker', display_resized)
            
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
            elif key == ord('1') and self.setup_mode and self.setup_step == 0:
                # Learn P1 shirt color
                self.p1_hsv_range = self.learn_shirt_color(frame, self.p1_zone, "P1")
                if self.p1_hsv_range:
                    self.setup_step = 1
            elif key == ord('2') and self.setup_mode and self.setup_step == 1:
                # Learn P2 shirt color
                self.p2_hsv_range = self.learn_shirt_color(frame, self.p2_zone, "P2")
                if self.p2_hsv_range:
                    self.setup_mode = False
                    self.setup_step = 2
                    print("\n[OK] Setup complete! Starting tracking mode...")
                    print("Tip: Use 'S' to show debug masks")
            elif key == ord('s'):
                self.show_debug(frame, p1_mask, p2_mask)
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('r'):
                self.current_frame = 0
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
    
    tracker = ShirtColorTracker(video_path)
    tracker.run()

