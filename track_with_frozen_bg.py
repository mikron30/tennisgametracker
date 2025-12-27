import cv2
import json
import numpy as np
import pickle
import os

def load_zones():
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

def load_parameters():
    with open('mog2_parameters.json', 'r') as f:
        return json.load(f)

class FrozenBackgroundTracker:
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
        
        # Load or create background model
        self.load_background_model()
        
        print("=" * 70)
        print("FROZEN BACKGROUND TRACKER")
        print("=" * 70)
        print(f"Using FROZEN background (learningRate=0)")
        print(f"Players will NEVER be absorbed into background!")
        print(f"Sensitivity: {self.var_threshold}")
        print(f"P1 bulb range: {self.p1_min_area} - {self.p1_max_area}")
        print(f"P2 bulb range: {self.p2_min_area} - {self.p2_max_area}")
        print("=" * 70)
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'S' = Show debug window")
        print("  SPACE = Play/pause")
        print("  'R' = Restart from frame 0")
        print("  'Q' = Quit")
        print("=" * 70)
        
        cv2.namedWindow('Frozen BG Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frozen BG Tracker', 1600, 900)
    
    def load_background_model(self):
        """Load trained background model or train from frames 0-50"""
        if os.path.exists('background_model.pkl'):
            print("Loading trained background model from 'background_model.pkl'...")
            with open('background_model.pkl', 'rb') as f:
                bg_model = pickle.load(f)
            
            print(f"  Trained on frames {bg_model['training_start']} to {bg_model['training_end']}")
            
            # Create MOG2 and train it on the same frames
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=bg_model['history'],
                varThreshold=bg_model['varThreshold'],
                detectShadows=True
            )
            
            # Re-train on the training frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, bg_model['training_start'])
            for frame_idx in range(bg_model['training_start'], bg_model['training_end'] + 1):
                ret, frame = self.cap.read()
                if ret:
                    self.mog2.apply(frame, learningRate=-1)
            
            print("  Background model loaded and re-trained!")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("[WARNING] No trained background model found!")
            print("Training on first 50 frames (may include players)...")
            
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=self.var_threshold,
                detectShadows=True
            )
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(50):
                ret, frame = self.cap.read()
                if ret:
                    self.mog2.apply(frame, learningRate=-1)
            
            print("  Quick background training complete")
            print("  TIP: Run 'train_background.py' to train on frames without players!")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def detect_in_zone(self, fgmask, zone_polygon, min_area, max_area, frame_shape):
        """Detect player in a specific zone"""
        # Create zone mask
        zone_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        # Apply zone mask
        zone_fg = cv2.bitwise_and(fgmask, zone_mask)
        
        # Find contours
        contours, _ = cv2.findContours(zone_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append((contour, area))
        
        if valid_contours:
            valid_contours.sort(key=lambda x: x[1], reverse=True)
            return valid_contours[0][0], valid_contours[0][1], contours
        
        return None, None, contours
    
    def process_frame(self, frame):
        """Process frame with FROZEN background (learningRate=0)"""
        # Apply MOG2 with FROZEN background
        fgmask = self.mog2.apply(frame, learningRate=0)  # FROZEN!
        
        # Remove shadows
        fgmask[fgmask == 127] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Detect in zones
        p1_contour, p1_area, p1_all = self.detect_in_zone(
            fgmask, self.p1_zone, self.p1_min_area, self.p1_max_area, frame.shape)
        p2_contour, p2_area, p2_all = self.detect_in_zone(
            fgmask, self.p2_zone, self.p2_min_area, self.p2_max_area, frame.shape)
        
        return fgmask, p1_contour, p1_area, p2_contour, p2_area, p1_all, p2_all
    
    def draw_results(self, frame, p1_contour, p1_area, p2_contour, p2_area):
        """Draw detection results"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 255, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (255, 255, 0), 2)
        
        # Draw P1
        if p1_contour is not None:
            x, y, w, h = cv2.boundingRect(p1_contour)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display, f"P1: {int(p1_area)}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw P2
        if p2_contour is not None:
            x, y, w, h = cv2.boundingRect(p2_contour)
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display, f"P2: {int(p2_area)}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Status info
        cv2.putText(display, f"Frame: {self.current_frame}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "FROZEN BG (LR=0)", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        p1_status = f"P1: {int(p1_area) if p1_area else 'NOT FOUND'}"
        p2_status = f"P2: {int(p2_area) if p2_area else 'NOT FOUND'}"
        
        cv2.putText(display, p1_status, (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if p1_area else (0, 0, 255), 2)
        cv2.putText(display, p2_status, (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 0) if p2_area else (0, 0, 255), 2)
        
        return display
    
    def show_debug(self, frame, fgmask, p1_all, p2_all):
        """Show debug window"""
        print("\n" + "=" * 70)
        print(f"DEBUG - Frame {self.current_frame}")
        print("=" * 70)
        
        print(f"\nP1 Zone: {len(p1_all)} contours, range {self.p1_min_area}-{self.p1_max_area}")
        for i, contour in enumerate(p1_all):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            status = "VALID" if self.p1_min_area <= area <= self.p1_max_area else "REJECTED"
            reason = "" if status == "VALID" else f"(too {'small' if area < self.p1_min_area else 'large'})"
            print(f"  Contour {i}: {int(area)}px at ({x},{y},{w},{h}) - {status} {reason}")
        
        print(f"\nP2 Zone: {len(p2_all)} contours, range {self.p2_min_area}-{self.p2_max_area}")
        for i, contour in enumerate(p2_all):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            status = "VALID" if self.p2_min_area <= area <= self.p2_max_area else "REJECTED"
            reason = "" if status == "VALID" else f"(too {'small' if area < self.p2_min_area else 'large'})"
            print(f"  Contour {i}: {int(area)}px at ({x},{y},{w},{h}) - {status} {reason}")
        
        print("=" * 70)
        
        # Show mask
        cv2.imshow('Debug - Foreground Mask', cv2.resize(fgmask, (int(fgmask.shape[1]*0.5), int(fgmask.shape[0]*0.5))))
    
    def run(self):
        """Main tracking loop"""
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            fgmask, p1_contour, p1_area, p2_contour, p2_area, p1_all, p2_all = self.process_frame(frame)
            display = self.draw_results(frame, p1_contour, p1_area, p2_contour, p2_area)
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Frozen BG Tracker', display_resized)
            
            key = cv2.waitKey(1 if not self.paused else 1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"[FORWARD] Frame {self.current_frame}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"[BACKWARD] Frame {self.current_frame}")
            elif key == ord('s'):
                self.show_debug(frame, fgmask, p1_all, p2_all)
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"{'PAUSED' if self.paused else 'PLAYING'}")
            elif key == ord('r'):
                self.current_frame = 0
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
    
    tracker = FrozenBackgroundTracker(video_path)
    tracker.run()



