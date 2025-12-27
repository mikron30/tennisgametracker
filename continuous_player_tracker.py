"""
Continuous player tracker using MOG2 on entire court.
1. Initialize P1 and P2 in their zones using MOG2
2. Once found, expand zones to cover entire court for continuous tracking
"""

import cv2
import numpy as np
import json
import os


class ContinuousPlayerTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Load zones
        self.p1_zone = None
        self.p2_zone = None
        self._load_zones()
        
        # Load MOG2 parameters
        self.var_threshold = 8
        self.learning_rate = 0.001
        self.p1_min_area = 1000
        self.p1_max_area = 50000
        self.p2_min_area = 500
        self.p2_max_area = 30000
        self._load_parameters()
        
        # Background subtractors - one for entire court
        self.bg_subtractor = None
        
        # Player states
        self.p1_initialized = False
        self.p2_initialized = False
        self.p1_bbox = None
        self.p2_bbox = None
        self.p1_center = None
        self.p2_center = None
        
        # Expanded zones for continuous tracking (entire court)
        self.p1_tracking_zone = None
        self.p2_tracking_zone = None
    
    def _load_zones(self):
        """Load zones from config."""
        if os.path.exists('player_zones.json'):
            with open('player_zones.json', 'r') as f:
                config = json.load(f)
                self.p1_zone = np.array(config['p1_zone'], dtype=np.int32)
                self.p2_zone = np.array(config['p2_zone'], dtype=np.int32)
                return True
        return False
    
    def _load_parameters(self):
        """Load MOG2 parameters from config."""
        if os.path.exists('mog2_parameters.json'):
            with open('mog2_parameters.json', 'r') as f:
                config = json.load(f)
                self.var_threshold = config.get('var_threshold', 8)
                self.learning_rate = config.get('learning_rate', 0.001)
                self.p1_min_area = config.get('p1_min_area', 1000)
                self.p1_max_area = config.get('p1_max_area', 50000)
                self.p2_min_area = config.get('p2_min_area', 500)
                self.p2_max_area = config.get('p2_max_area', 30000)
    
    def initialize_players(self, frame):
        """Initialize P1 and P2 using MOG2 in their zones."""
        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=self.var_threshold, detectShadows=True
            )
        
        # Apply MOG2 to entire frame
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask[fg_mask == 127] = 0
        
        # Try to find P1 if not initialized
        if not self.p1_initialized:
            p1_bbox = self._find_player_in_zone(fg_mask, self.p1_zone,
                                                self.p1_min_area, self.p1_max_area)
            if p1_bbox:
                self.p1_bbox = p1_bbox
                x, y, w, h = p1_bbox
                self.p1_center = (x + w//2, y + h//2)
                self.p1_initialized = True
                print(f"[INIT] P1 initialized at bbox {p1_bbox}")
        
        # Try to find P2 if not initialized
        if not self.p2_initialized:
            p2_bbox = self._find_player_in_zone(fg_mask, self.p2_zone,
                                                self.p2_min_area, self.p2_max_area)
            if p2_bbox:
                self.p2_bbox = p2_bbox
                x, y, w, h = p2_bbox
                self.p2_center = (x + w//2, y + h//2)
                self.p2_initialized = True
                print(f"[INIT] P2 initialized at bbox {p2_bbox}")
        
        return self.p1_initialized, self.p2_initialized
    
    def _find_player_in_zone(self, fg_mask, zone, min_area, max_area):
        """Find player in zone from foreground mask."""
        # Create zone mask
        zone_mask = np.zeros(fg_mask.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone], 255)
        
        # Apply zone mask to foreground
        fg_in_zone = cv2.bitwise_and(fg_mask, zone_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_in_zone = cv2.morphologyEx(fg_in_zone, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_in_zone = cv2.dilate(fg_in_zone, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_in_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest valid contour
        best_bbox = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                if area > best_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    best_bbox = (x, y, w, h)
                    best_area = area
        
        return best_bbox
    
    def track_players(self, frame):
        """Track P1 and P2 across the entire frame using MOG2."""
        players = []
        
        # Apply MOG2 to entire frame
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask[fg_mask == 127] = 0
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find all contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find P1 (closest to last known P1 position)
        if self.p1_initialized and self.p1_center:
            best_p1 = None
            best_p1_dist = float('inf')
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.p1_min_area <= area <= self.p1_max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w//2, y + h//2)
                    dist = np.sqrt((center[0] - self.p1_center[0])**2 + (center[1] - self.p1_center[1])**2)
                    
                    if dist < best_p1_dist and dist < 200:  # Within 200 pixels
                        best_p1_dist = dist
                        best_p1 = {'bbox': (x, y, w, h), 'center': center}
            
            if best_p1:
                self.p1_bbox = best_p1['bbox']
                self.p1_center = best_p1['center']
                players.append({'name': 'P1', 'bbox': self.p1_bbox, 'center': self.p1_center})
        
        # Find P2 (closest to last known P2 position)
        if self.p2_initialized and self.p2_center:
            best_p2 = None
            best_p2_dist = float('inf')
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.p2_min_area <= area <= self.p2_max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w//2, y + h//2)
                    dist = np.sqrt((center[0] - self.p2_center[0])**2 + (center[1] - self.p2_center[1])**2)
                    
                    # Make sure it's not the same as P1
                    if self.p1_bbox:
                        p1_center = (self.p1_bbox[0] + self.p1_bbox[2]//2, self.p1_bbox[1] + self.p1_bbox[3]//2)
                        dist_from_p1 = np.sqrt((center[0] - p1_center[0])**2 + (center[1] - p1_center[1])**2)
                        if dist_from_p1 < 100:  # Too close to P1, skip
                            continue
                    
                    if dist < best_p2_dist and dist < 200:  # Within 200 pixels
                        best_p2_dist = dist
                        best_p2 = {'bbox': (x, y, w, h), 'center': center}
            
            if best_p2:
                self.p2_bbox = best_p2['bbox']
                self.p2_center = best_p2['center']
                players.append({'name': 'P2', 'bbox': self.p2_bbox, 'center': self.p2_center})
        
        return players
    
    def visualize(self, frame, players, show_zones=False):
        """Visualize tracked players with bounding boxes."""
        display = frame.copy()
        
        # Optionally show initialization zones
        if show_zones:
            if self.p1_zone is not None:
                cv2.polylines(display, [self.p1_zone], True, (255, 0, 0), 2)
                cv2.putText(display, "P1 Init Zone", 
                           (self.p1_zone[0][0], self.p1_zone[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if self.p2_zone is not None:
                cv2.polylines(display, [self.p2_zone], True, (0, 165, 255), 2)
                cv2.putText(display, "P2 Init Zone",
                           (self.p2_zone[0][0], self.p2_zone[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Draw tracked players
        for player in players:
            name = player['name']
            bbox = player['bbox']
            center = player['center']
            
            x, y, w, h = bbox
            color = (255, 0, 0) if name == 'P1' else (0, 165, 255)
            
            # Draw bounding box (square)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 4)
            
            # Draw center point
            cv2.circle(display, center, 10, color, -1)
            cv2.circle(display, center, 10, (255, 255, 255), 2)
            
            # Draw label
            label = f"{name} [TRACKING]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(display, (x, y - lh - 15), (x + lw + 10, y - 5), color, -1)
            cv2.putText(display, label, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return display


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = ContinuousPlayerTracker(video_path)
    
    if tracker.p1_zone is None or tracker.p2_zone is None:
        print("ERROR: No zones found. Please run mark_and_track_zones.py first!")
        return
    
    print("\n" + "="*70)
    print("CONTINUOUS PLAYER TRACKER")
    print("="*70)
    print("Uses zone-based MOG2 initialization + CSRT continuous tracking")
    print("\nFeatures:")
    print("  - Initializes P1 and P2 in their zones")
    print("  - Tracks them continuously across entire court")
    print("  - Players can move anywhere and still be tracked")
    print("\nControls:")
    print("  SPACE = Pause/Resume")
    print("  'D' = Next frame (when paused)")
    print("  'Z' = Toggle zone overlay")
    print("  'R' = Reset and re-initialize players")
    print("  'Q' = Quit")
    print("="*70)
    
    cv2.namedWindow("Continuous Player Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Continuous Player Tracker", 1600, 900)
    
    # Start from beginning
    tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_num = 0
    paused = True
    show_zones = True
    advance_frame = False
    initialization_phase = True
    
    print("\n[PHASE 1] Initializing players (building background model)...")
    print("Press 'D' to advance frames until both players are detected")
    
    while True:
        if not paused or advance_frame:
            ret, frame = tracker.cap.read()
            if not ret:
                print("\nEnd of video")
                break
            
            frame_num += 1
            
            # Phase 1: Initialize players
            if initialization_phase:
                p1_init, p2_init = tracker.initialize_players(frame)
                
                if p1_init and p2_init:
                    print(f"\n[SUCCESS] Both players initialized at frame {frame_num}!")
                    print("[PHASE 2] Continuous tracking active. Press SPACE to play.")
                    initialization_phase = False
            
            # Phase 2: Track players continuously
            else:
                players = tracker.track_players(frame)
                
                # Re-initialize if tracking lost
                if not tracker.p1_initialized or not tracker.p2_initialized:
                    print(f"\n[WARN] Tracking lost at frame {frame_num}. Re-initializing...")
                    tracker.initialize_players(frame)
            
            # Get current players for visualization
            players = []
            if tracker.p1_initialized and tracker.p1_bbox:
                x, y, w, h = tracker.p1_bbox
                players.append({
                    'name': 'P1',
                    'bbox': tracker.p1_bbox,
                    'center': (x + w//2, y + h//2),
                    'tracking': True
                })
            if tracker.p2_initialized and tracker.p2_bbox:
                x, y, w, h = tracker.p2_bbox
                players.append({
                    'name': 'P2',
                    'bbox': tracker.p2_bbox,
                    'center': (x + w//2, y + h//2),
                    'tracking': True
                })
            
            # Visualize
            display = tracker.visualize(frame, players, show_zones)
            
            # Status overlay
            phase_text = "INITIALIZING" if initialization_phase else "TRACKING"
            phase_color = (0, 165, 255) if initialization_phase else (0, 255, 0)
            
            status_text = [
                f"[{phase_text}] Frame: {frame_num}",
                f"P1: {'INITIALIZED' if tracker.p1_initialized else 'SEARCHING...'}",
                f"P2: {'INITIALIZED' if tracker.p2_initialized else 'SEARCHING...'}",
            ]
            
            cv2.rectangle(display, (10, 10), (600, 150), (0, 0, 0), -1)
            for i, text in enumerate(status_text):
                cv2.putText(display, text, (20, 50 + i * 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, phase_color, 2)
            
            controls = "D=next | SPACE=play/pause | Z=zones | R=reset | Q=quit"
            cv2.putText(display, controls, (20, display.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Show
            cv2.imshow("Continuous Player Tracker", display)
            advance_frame = False
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"\n{'PAUSED' if paused else 'PLAYING'} at frame {frame_num}")
        elif key == ord('d'):
            advance_frame = True
        elif key == ord('z'):
            show_zones = not show_zones
        elif key == ord('r'):
            print("\n[RESET] Re-initializing players...")
            tracker.p1_initialized = False
            tracker.p2_initialized = False
            tracker.bg_subtractor = None
            initialization_phase = True
    
    tracker.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

