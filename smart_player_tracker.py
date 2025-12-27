import cv2
import json
import numpy as np
from collections import defaultdict

def load_zones():
    """Load P1 and P2 zones"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

class SmartPlayerTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.paused = True
        
        # Load zones
        self.p1_zone, self.p2_zone = load_zones()
        
        # HSV filters (will be auto-detected)
        self.p1_hsv_filter = None
        self.p2_hsv_filter = None
        
        # Tracking state
        self.p1_position = None
        self.p2_position = None
        self.search_window_size = 300
        
        print("=" * 70)
        print("SMART PLAYER TRACKER")
        print("=" * 70)
        print("Step 1: Auto-analyzing frame 10 to find best HSV filters...")
        print("=" * 70)
        
        # Auto-detect HSV filters
        self.auto_detect_hsv_filters()
        
        print("\n" + "=" * 70)
        print("TRACKING MODE - Ready!")
        print("=" * 70)
        print("KEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10)")
        print("  'B' = Fast backward (-10)")
        print("  'S' = Show debug masks")
        print("  SPACE = Play/pause")
        print("  'R' = Restart from frame 0")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Smart Player Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Smart Player Tracker', 1600, 900)
    
    def get_zone_pixels(self, frame, zone_polygon):
        """Extract all pixels from a zone"""
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y_coords, x_coords = np.where(zone_mask > 0)
        pixels = hsv[y_coords, x_coords]
        coords = np.column_stack([x_coords, y_coords])
        
        return pixels, coords
    
    def cluster_colors(self, pixels, coords, num_bins=10):
        """Cluster pixels by HSV color"""
        h_bins = num_bins
        s_bins = 5
        v_bins = 5
        
        h_quantized = (pixels[:, 0].astype(np.int32) * h_bins // 180).astype(int)
        s_quantized = (pixels[:, 1].astype(np.int32) * s_bins // 256).astype(int)
        v_quantized = (pixels[:, 2].astype(np.int32) * v_bins // 256).astype(int)
        
        clusters = defaultdict(list)
        for i, (h, s, v) in enumerate(zip(h_quantized, s_quantized, v_quantized)):
            cluster_id = (h, s, v)
            clusters[cluster_id].append(i)
        
        return clusters
    
    def analyze_cluster_spatial(self, cluster_indices, coords):
        """Analyze cluster spatial distribution"""
        cluster_coords = coords[cluster_indices]
        
        if len(cluster_coords) < 100:
            return None
        
        x_min, y_min = cluster_coords.min(axis=0)
        x_max, y_max = cluster_coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        bbox_area = width * height
        
        if bbox_area == 0:
            return None
        
        cluster_area = len(cluster_coords)
        fill_ratio = cluster_area / bbox_area
        aspect_ratio = height / width if width > 0 else 0
        
        try:
            hull = cv2.convexHull(cluster_coords.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            compactness = cluster_area / hull_area if hull_area > 0 else 0
        except:
            compactness = fill_ratio
        
        return {
            'count': cluster_area,
            'bbox': (int(x_min), int(y_min), int(width), int(height)),
            'fill_ratio': fill_ratio,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'coords': cluster_coords
        }
    
    def score_cluster_as_player(self, stats):
        """Score cluster as player likelihood"""
        if stats is None:
            return 0
        
        score = 0
        
        # Size: 5000-200000 pixels
        if 5000 <= stats['count'] <= 200000:
            score += 30
        elif 2000 <= stats['count'] <= 300000:
            score += 10
        
        # Fill ratio: compact shapes
        if stats['fill_ratio'] > 0.5:
            score += 40
        elif stats['fill_ratio'] > 0.3:
            score += 20
        
        # Aspect ratio: person-shaped
        if 0.8 <= stats['aspect_ratio'] <= 3.0:
            score += 30
        elif 0.5 <= stats['aspect_ratio'] <= 4.0:
            score += 10
        
        return score
    
    def find_best_hsv_filter(self, frame, zone_polygon, zone_name):
        """Find best HSV filter for a zone"""
        print(f"\nAnalyzing {zone_name}...")
        
        pixels, coords = self.get_zone_pixels(frame, zone_polygon)
        clusters = self.cluster_colors(pixels, coords, num_bins=10)
        
        cluster_stats = []
        for cluster_id, cluster_indices in clusters.items():
            h, s, v = cluster_id
            
            # Convert to HSV range with expansion for better tracking
            h_min = max(0, h * 18 - 10)
            h_max = min(179, (h + 1) * 18 + 10)
            s_min = max(0, s * 51 - 20)
            s_max = min(255, (s + 1) * 51 + 20)
            v_min = max(0, v * 51 - 20)
            v_max = min(255, (v + 1) * 51 + 20)
            
            stats = self.analyze_cluster_spatial(cluster_indices, coords)
            if stats is None:
                continue
            
            score = self.score_cluster_as_player(stats)
            stats['hsv_range'] = (h_min, h_max, s_min, s_max, v_min, v_max)
            stats['score'] = score
            
            cluster_stats.append(stats)
        
        # Sort by score
        cluster_stats.sort(key=lambda x: x['score'], reverse=True)
        
        if cluster_stats:
            best = cluster_stats[0]
            h_min, h_max, s_min, s_max, v_min, v_max = best['hsv_range']
            print(f"  Best HSV: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"  Score: {best['score']}, Pixels: {best['count']}, Fill: {best['fill_ratio']*100:.1f}%")
            return best['hsv_range']
        
        return None
    
    def auto_detect_hsv_filters(self):
        """Auto-detect best HSV filters from frame 10"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
        ret, frame = self.cap.read()
        
        if not ret:
            print("[ERROR] Could not read frame 10!")
            return
        
        self.p1_hsv_filter = self.find_best_hsv_filter(frame, self.p1_zone, "P1")
        self.p2_hsv_filter = self.find_best_hsv_filter(frame, self.p2_zone, "P2")
        
        # Reset to frame 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
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
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, hsv_mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Minimum area threshold
        if area < 2000:
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
    
    def get_search_region(self, last_position):
        """Get search region around last position"""
        if last_position is None:
            return None
        
        center_x, center_y = last_position['center']
        
        search_x = max(0, center_x - self.search_window_size)
        search_y = max(0, center_y - self.search_window_size)
        search_w = self.search_window_size * 2
        search_h = self.search_window_size * 2
        
        return (search_x, search_y, search_w, search_h)
    
    def process_frame(self, frame):
        """Process frame and detect both players"""
        # Detect P1
        p1_search = self.get_search_region(self.p1_position)
        p1_detection, p1_mask = self.detect_player(
            frame, self.p1_hsv_filter, p1_search, self.p1_zone if p1_search is None else None)
        
        if p1_detection is not None:
            self.p1_position = p1_detection
        
        # Detect P2
        p2_search = self.get_search_region(self.p2_position)
        p2_detection, p2_mask = self.detect_player(
            frame, self.p2_hsv_filter, p2_search, self.p2_zone if p2_search is None else None)
        
        if p2_detection is not None:
            self.p2_position = p2_detection
        
        return p1_detection, p2_detection, p1_mask, p2_mask
    
    def draw_results(self, frame, p1_detection, p2_detection):
        """Draw tracking results"""
        display = frame.copy()
        
        # Draw zones (dimmed)
        cv2.polylines(display, [self.p1_zone], True, (100, 100, 50), 1)
        cv2.polylines(display, [self.p2_zone], True, (100, 100, 50), 1)
        
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
            p1_search = self.get_search_region(self.p1_position)
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
            p2_search = self.get_search_region(self.p2_position)
            if p2_search:
                sx, sy, sw, sh = p2_search
                cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 1)
        
        # Status info
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
            
            p1_detection, p2_detection, p1_mask, p2_mask = self.process_frame(frame)
            display = self.draw_results(frame, p1_detection, p2_detection)
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Smart Player Tracker', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"[FORWARD] Frame {self.current_frame}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"[BACKWARD] Frame {self.current_frame}")
            elif key == ord('f'):
                self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                print(f"[FAST FORWARD] Frame {self.current_frame}")
            elif key == ord('b'):
                self.current_frame = max(self.current_frame - 10, 0)
                print(f"[FAST BACKWARD] Frame {self.current_frame}")
            elif key == ord('s'):
                self.show_debug(p1_mask, p2_mask)
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('r'):
                self.current_frame = 0
                self.p1_position = None
                self.p2_position = None
                print("Restarting from frame 0")
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
    
    tracker = SmartPlayerTracker(video_path)
    tracker.run()

