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

class FinalPlayerTracker:
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
        
        # Create separate MOG2 for each player zone
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
        
        # Learning rate
        self.learning_rate = 0.001
        
        # Player positions
        self.p1_bbox = None
        self.p2_bbox = None
        
        print("=" * 70)
        print("FINAL PLAYER TRACKER - MOG2 Background Subtraction")
        print("=" * 70)
        print("Method: Separate MOG2 for each player zone")
        print(f"Sensitivity: {self.var_threshold}")
        print(f"P1 area range: {self.p1_min_area} - {self.p1_max_area}")
        print(f"P2 area range: {self.p2_min_area} - {self.p2_max_area}")
        print("=" * 70)
        print("\nWarmup: Learning background from first 20 frames...")
        
        # Warmup phase - learn background
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                self.apply_mog2_to_zones(frame, learn_only=True)
        
        print("Warmup complete!")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("\n" + "=" * 70)
        print("KEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame (WARNING: will reset MOG2)")
        print("  'F' = Fast forward (+10)")
        print("  'B' = Fast backward (-10)")
        print("  'S' = Show debug window")
        print("  SPACE = Play/pause")
        print("  'R' = Restart (with MOG2 reset)")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Player Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Player Tracker', 1600, 900)
    
    def apply_mog2_to_zones(self, frame, learn_only=False):
        """Apply MOG2 separately to each zone"""
        # Create zone masks
        p1_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p1_zone_mask, [self.p1_zone], 255)
        
        p2_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p2_zone_mask, [self.p2_zone], 255)
        
        # Apply MOG2 to full frame
        if learn_only:
            # During warmup, use normal learning
            p1_fg = self.p1_mog2.apply(frame, learningRate=self.learning_rate)
            p2_fg = self.p2_mog2.apply(frame, learningRate=self.learning_rate)
            return None, None, None, None
        
        # Adaptive learning rate
        p1_lr = -1 if self.p1_bbox is not None else self.learning_rate
        p2_lr = -1 if self.p2_bbox is not None else self.learning_rate
        
        p1_fg = self.p1_mog2.apply(frame, learningRate=p1_lr)
        p2_fg = self.p2_mog2.apply(frame, learningRate=p2_lr)
        
        # Apply zone masks
        p1_fg = cv2.bitwise_and(p1_fg, p1_zone_mask)
        p2_fg = cv2.bitwise_and(p2_fg, p2_zone_mask)
        
        # Remove shadows (value 127)
        p1_fg[p1_fg == 127] = 0
        p2_fg[p2_fg == 127] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        p1_fg = cv2.morphologyEx(p1_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        p1_fg = cv2.morphologyEx(p1_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        p2_fg = cv2.morphologyEx(p2_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        p2_fg = cv2.morphologyEx(p2_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Detect players
        p1_bbox, p1_area, p1_contours = self.detect_in_mask(p1_fg, self.p1_min_area, self.p1_max_area)
        p2_bbox, p2_area, p2_contours = self.detect_in_mask(p2_fg, self.p2_min_area, self.p2_max_area)
        
        return p1_bbox, p2_bbox, p1_fg, p2_fg
    
    def detect_in_mask(self, mask, min_area, max_area):
        """Detect player in foreground mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, contours
        
        # Find valid contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append((contour, area))
        
        if not valid_contours:
            return None, None, contours
        
        # Get largest valid contour
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        largest_contour, largest_area = valid_contours[0]
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate center
        cx = x + w // 2
        cy = y + h // 2
        
        return {
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'area': largest_area
        }, largest_area, contours
    
    def draw_results(self, frame, p1_bbox, p2_bbox):
        """Draw detection results"""
        display = frame.copy()
        
        # Draw zones (light)
        cv2.polylines(display, [self.p1_zone], True, (100, 100, 50), 1)
        cv2.polylines(display, [self.p2_zone], True, (100, 100, 50), 1)
        
        # Draw P1
        if p1_bbox is not None:
            x, y, w, h = p1_bbox['bbox']
            cx, cy = p1_bbox['center']
            
            # Bounding box
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Center marker
            cv2.circle(display, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(display, (cx, cy), 15, (0, 255, 255), 3)
            
            # Label with area
            cv2.putText(display, f"P1", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"{int(p1_bbox['area'])}px", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw P2
        if p2_bbox is not None:
            x, y, w, h = p2_bbox['bbox']
            cx, cy = p2_bbox['center']
            
            # Bounding box
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Center marker
            cv2.circle(display, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(display, (cx, cy), 15, (0, 255, 255), 3)
            
            # Label with area
            cv2.putText(display, f"P2", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"{int(p2_bbox['area'])}px", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status bar
        cv2.rectangle(display, (0, 0), (600, 160), (0, 0, 0), -1)
        
        cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        p1_status = f"P1: {int(p1_bbox['area'])}px" if p1_bbox else "P1: NOT FOUND"
        p2_status = f"P2: {int(p2_bbox['area'])}px" if p2_bbox else "P2: NOT FOUND"
        
        cv2.putText(display, p1_status, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if p1_bbox else (0, 0, 255), 2)
        cv2.putText(display, p2_status, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if p2_bbox else (0, 0, 255), 2)
        
        return display
    
    def show_debug(self, frame, p1_fg, p2_fg):
        """Show debug windows"""
        print("\n" + "=" * 70)
        print(f"DEBUG - Frame {self.current_frame}")
        print("=" * 70)
        
        # Show masks
        if p1_fg is not None:
            cv2.imshow('Debug - P1 Foreground', cv2.resize(p1_fg, 
                      (int(p1_fg.shape[1]*0.5), int(p1_fg.shape[0]*0.5))))
        
        if p2_fg is not None:
            cv2.imshow('Debug - P2 Foreground', cv2.resize(p2_fg, 
                      (int(p2_fg.shape[1]*0.5), int(p2_fg.shape[0]*0.5))))
        
        # Print info
        print(f"P1: {self.p1_bbox}")
        print(f"P2: {self.p2_bbox}")
        print(f"Area ranges: P1({self.p1_min_area}-{self.p1_max_area}) P2({self.p2_min_area}-{self.p2_max_area})")
        print("=" * 70)
    
    def reset_mog2(self):
        """Reset MOG2 and re-learn background"""
        print("\nResetting MOG2...")
        
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
        
        # Warmup
        current_pos = self.current_frame
        warmup_start = max(0, current_pos - 20)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
        for i in range(20):
            ret, frame = self.cap.read()
            if ret:
                self.apply_mog2_to_zones(frame, learn_only=True)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        print("MOG2 reset complete!")
    
    def run(self):
        """Main loop"""
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            p1_bbox, p2_bbox, p1_fg, p2_fg = self.apply_mog2_to_zones(frame)
            
            # Update tracking state
            if p1_bbox is not None:
                self.p1_bbox = p1_bbox
            if p2_bbox is not None:
                self.p2_bbox = p2_bbox
            
            display = self.draw_results(frame, p1_bbox, p2_bbox)
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Player Tracker', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('a'):
                print("[WARNING] Going backward - this may affect MOG2 background model")
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('f'):
                self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('b'):
                print("[WARNING] Going backward - this may affect MOG2 background model")
                self.current_frame = max(self.current_frame - 10, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('s'):
                self.show_debug(frame, p1_fg, p2_fg)
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('r'):
                self.current_frame = 0
                self.p1_bbox = None
                self.p2_bbox = None
                self.reset_mog2()
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
    
    tracker = FinalPlayerTracker(video_path)
    tracker.run()

