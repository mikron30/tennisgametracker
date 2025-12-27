"""
DEBUG version of zone tracker with detailed visualization.
Shows all detected contours, masks, and rejection reasons.
"""

import cv2
import numpy as np
import json
import os


class DebugZoneTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Load zones
        self.p1_zone = None
        self.p2_zone = None
        self._load_zones()
        
        # Load parameters
        self.var_threshold = 6
        self.learning_rate = 0.001
        self.p1_min_area = 1000
        self.p1_max_area = 50000
        self.p2_min_area = 500
        self.p2_max_area = 30000
        self._load_parameters()
        
        # Background subtractors
        self.p1_bg_subtractor = None
        self.p2_bg_subtractor = None
    
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
        """Load parameters from config."""
        if os.path.exists('mog2_parameters.json'):
            with open('mog2_parameters.json', 'r') as f:
                config = json.load(f)
                self.var_threshold = config.get('var_threshold', 6)
                self.p1_min_area = config.get('p1_min_area', 1000)
                self.p1_max_area = config.get('p1_max_area', 50000)
                self.p2_min_area = config.get('p2_min_area', 500)
                self.p2_max_area = config.get('p2_max_area', 30000)
    
    def detect_player_with_debug(self, frame, zone, bg_subtractor, min_area, max_area, zone_name):
        """Detect player with full debug info."""
        # Create zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone], 255)
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame, learningRate=self.learning_rate)
        fg_mask[fg_mask == 127] = 0  # Remove shadows
        fg_in_zone = cv2.bitwise_and(fg_mask, zone_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_cleaned = cv2.morphologyEx(fg_in_zone, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_cleaned = cv2.dilate(fg_cleaned, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze all contours
        print(f"\n{'='*70}")
        print(f"DEBUG: {zone_name}")
        print(f"{'='*70}")
        print(f"Total contours found: {len(contours)}")
        print(f"Area range: {min_area} - {max_area} pixels")
        print(f"Foreground pixels in zone: {np.sum(fg_in_zone > 0)}")
        print(f"Foreground pixels after cleanup: {np.sum(fg_cleaned > 0)}")
        
        # Track all candidates
        candidates = []
        best_player = None
        best_area = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            status = "CANDIDATE"
            reason = ""
            
            if area < min_area:
                status = "REJECTED"
                reason = f"Too small ({area:.0f} < {min_area})"
            elif area > max_area:
                status = "REJECTED"
                reason = f"Too large ({area:.0f} > {max_area})"
            elif aspect_ratio < 0.5 or aspect_ratio > 5.0:
                status = "REJECTED"
                reason = f"Bad aspect ratio ({aspect_ratio:.2f})"
            else:
                if area > best_area:
                    best_area = area
                    best_player = {
                        'bbox': (x, y, w, h),
                        'area': area,
                        'contour': contour
                    }
                    status = "BEST"
            
            if area > 100 or status in ["CANDIDATE", "BEST"]:  # Only show significant ones
                print(f"  Contour {i:2d}: area={area:6.0f}px, pos=({x:4d},{y:4d}), size=({w:3d}x{h:3d}), aspect={aspect_ratio:.2f} - {status} {reason}")
            
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'status': status,
                'reason': reason
            })
        
        if best_player:
            print(f"\n>>> SELECTED: bbox={best_player['bbox']}, area={best_player['area']:.0f}px")
        else:
            print(f"\n>>> NO VALID PLAYER FOUND!")
        
        return best_player, fg_in_zone, fg_cleaned, candidates
    
    def visualize_debug(self, frame, p1_result, p2_result, frame_num):
        """Create detailed debug visualization."""
        p1_player, p1_fg, p1_cleaned, p1_candidates = p1_result
        p2_player, p2_fg, p2_cleaned, p2_candidates = p2_result
        
        # Main display
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 0, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (0, 165, 255), 2)
        
        # Draw ALL P1 candidates (rejected and accepted)
        for cand in p1_candidates:
            x, y, w, h = cand['bbox']
            if cand['status'] == "BEST":
                color = (0, 255, 0)  # Green for selected
                thickness = 4
            elif cand['status'] == "CANDIDATE":
                color = (255, 255, 0)  # Cyan for candidate
                thickness = 2
            else:
                color = (0, 0, 255)  # Red for rejected
                thickness = 1
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            if cand['area'] > 100:
                cv2.putText(display, f"{cand['area']:.0f}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw ALL P2 candidates
        for cand in p2_candidates:
            x, y, w, h = cand['bbox']
            if cand['status'] == "BEST":
                color = (0, 255, 0)  # Green for selected
                thickness = 4
            elif cand['status'] == "CANDIDATE":
                color = (255, 255, 0)  # Cyan for candidate
                thickness = 2
            else:
                color = (0, 0, 255)  # Red for rejected
                thickness = 1
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            if cand['area'] > 100:
                cv2.putText(display, f"{cand['area']:.0f}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw final selections with labels
        if p1_player:
            x, y, w, h = p1_player['bbox']
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(display, f"P1 ({p1_player['area']:.0f}px)", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        if p2_player:
            x, y, w, h = p2_player['bbox']
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 165, 255), 5)
            cv2.putText(display, f"P2 ({p2_player['area']:.0f}px)", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Status
        status = "SUCCESS" if (p1_player and p2_player) else "PARTIAL" if (p1_player or p2_player) else "FAILED"
        status_color = (0, 255, 0) if status == "SUCCESS" else (0, 165, 255) if status == "PARTIAL" else (0, 0, 255)
        
        cv2.rectangle(display, (10, 10), (700, 100), (0, 0, 0), -1)
        cv2.putText(display, f"Frame {frame_num} - {status}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        cv2.putText(display, f"P1: {'FOUND' if p1_player else 'LOST'} | P2: {'FOUND' if p2_player else 'LOST'}",
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Create mask visualizations
        p1_mask_bgr = cv2.cvtColor(p1_cleaned, cv2.COLOR_GRAY2BGR)
        p2_mask_bgr = cv2.cvtColor(p2_cleaned, cv2.COLOR_GRAY2BGR)
        
        # Add zone overlays to masks
        p1_mask_overlay = p1_mask_bgr.copy()
        cv2.polylines(p1_mask_overlay, [self.p1_zone], True, (255, 0, 0), 2)
        
        p2_mask_overlay = p2_mask_bgr.copy()
        cv2.polylines(p2_mask_overlay, [self.p2_zone], True, (0, 165, 255), 2)
        
        # Resize masks for side-by-side display
        scale = 0.4
        display_resized = cv2.resize(display, None, fx=scale, fy=scale)
        p1_mask_resized = cv2.resize(p1_mask_overlay, None, fx=scale, fy=scale)
        p2_mask_resized = cv2.resize(p2_mask_overlay, None, fx=scale, fy=scale)
        
        # Add labels to masks
        cv2.putText(p1_mask_resized, "P1 FOREGROUND MASK", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(p2_mask_resized, "P2 FOREGROUND MASK", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Stack displays
        top_row = np.hstack([display_resized, p1_mask_resized])
        bottom_pad = np.zeros_like(p2_mask_resized)
        bottom_row = np.hstack([bottom_pad, p2_mask_resized])
        
        combined = np.vstack([top_row, bottom_row])
        
        # Add legend
        legend_height = 100
        legend = np.zeros((legend_height, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(legend, "GREEN=Selected | CYAN=Candidate | RED=Rejected | D=Next Frame | R=Restart | Q=Quit",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        final = np.vstack([combined, legend])
        
        return final


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    tracker = DebugZoneTracker(video_path)
    
    if tracker.p1_zone is None or tracker.p2_zone is None:
        print("ERROR: No zones found. Please run mark_and_track_zones.py first!")
        return
    
    print("\n" + "="*70)
    print("DEBUG ZONE TRACKER")
    print("="*70)
    print("Shows detailed detection info for debugging")
    print("\nFeatures:")
    print("  - All contours shown (green=selected, cyan=candidate, red=rejected)")
    print("  - Foreground masks for P1 and P2 zones")
    print("  - Detailed console output for each detection")
    print("\nControls:")
    print("  'D' = Next frame")
    print("  'R' = Restart from frame 0")
    print("  'Q' = Quit")
    print("="*70)
    
    # Initialize background subtractors
    tracker.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=tracker.var_threshold, detectShadows=True
    )
    tracker.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=tracker.var_threshold, detectShadows=True
    )
    
    cv2.namedWindow("Debug Zone Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Debug Zone Tracker", 1600, 1000)
    
    tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0
    
    while True:
        ret, frame = tracker.cap.read()
        if not ret:
            print("\nEnd of video")
            break
        
        frame_num += 1
        
        # Detect with debug info
        p1_result = tracker.detect_player_with_debug(
            frame, tracker.p1_zone, tracker.p1_bg_subtractor,
            tracker.p1_min_area, tracker.p1_max_area, "P1 ZONE"
        )
        
        p2_result = tracker.detect_player_with_debug(
            frame, tracker.p2_zone, tracker.p2_bg_subtractor,
            tracker.p2_min_area, tracker.p2_max_area, "P2 ZONE"
        )
        
        # Visualize
        display = tracker.visualize_debug(frame, p1_result, p2_result, frame_num)
        cv2.imshow("Debug Zone Tracker", display)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            continue  # Next frame
        elif key == ord('r'):
            print("\n[RESTART] Going back to frame 0...")
            tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            tracker.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=tracker.var_threshold, detectShadows=True
            )
            tracker.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=tracker.var_threshold, detectShadows=True
            )
    
    tracker.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



