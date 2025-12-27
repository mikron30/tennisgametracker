#!/usr/bin/env python3
"""
Manual forward-only navigator
Only allows pressing 'D' to advance frame-by-frame from the beginning
"""

import cv2
import numpy as np
import json

class ManualForwardNavigator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frame_num = 0
        
        # Load zones and parameters
        self.p1_zone, self.p2_zone = self._load_zones()
        self.params = self._load_parameters()
        
        # Initialize MOG2
        self.sensitivity = self.params['var_threshold']
        self.p1_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.params['history'],
            varThreshold=self.sensitivity,
            detectShadows=False
        )
        self.p2_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.params['history'],
            varThreshold=self.sensitivity,
            detectShadows=False
        )
        
        # Track last detection state
        self.p1_last_detected = False
        self.p2_last_detected = False
        
        # Current frame
        self.current_frame = None
        
    def _load_zones(self):
        """Load player zones from JSON"""
        with open('player_zones.json', 'r') as f:
            data = json.load(f)
        return data['p1_zone'], data['p2_zone']
    
    def _load_parameters(self):
        """Load MOG2 parameters from JSON"""
        with open('mog2_parameters.json', 'r') as f:
            return json.load(f)
    
    def detect_player_in_zone(self, frame, mog2, zone_points, min_area, max_area, last_detected):
        """Detect player in a specific zone using MOG2 with adaptive learning"""
        # Create mask for the zone
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        zone_array = np.array(zone_points, dtype=np.int32)
        cv2.fillPoly(mask, [zone_array], 255)
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Adaptive learning rate
        adaptive_rate = -1 if last_detected else self.params['learning_rate']
        fg_mask = mog2.apply(masked_frame, learningRate=adaptive_rate)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply zone mask
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find all contours > 100px
        all_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                all_contours.append({'area': area, 'valid': min_area <= area <= max_area})
        all_contours.sort(key=lambda x: x['area'], reverse=True)
        
        # Find valid contours
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            area = cv2.contourArea(largest)
            return (x, y, w, h), area, all_contours, fg_mask
        
        return None, None, all_contours, fg_mask
    
    def run(self, max_frame=10):
        """Run the manual forward navigator"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        print("="*70)
        print("MANUAL FORWARD-ONLY NAVIGATOR")
        print("="*70)
        print(f"Sensitivity: {self.sensitivity}")
        print(f"P1 area: {self.params['p1_min_area']:,} - {self.params['p1_max_area']:,}px")
        print(f"P2 area: {self.params['p2_min_area']:,} - {self.params['p2_max_area']:,}px")
        print()
        print("CONTROLS:")
        print("  'D' = Next frame (forward only)")
        print("  'Q' = Quit")
        print()
        print(f"Navigate through frames 0-{max_frame} by pressing 'D'")
        print("="*70)
        print()
        
        cv2.namedWindow('Manual Navigator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Manual Navigator', 1280, 720)
        
        waiting_for_keypress = True
        
        while self.frame_num <= max_frame:
            if not waiting_for_keypress:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"\nReached end of video at frame {self.frame_num}")
                    break
                
                self.current_frame = frame.copy()
                
                # Detect players
                p1_bbox, p1_area, p1_contours, p1_mask = self.detect_player_in_zone(
                    frame, self.p1_mog2, self.p1_zone,
                    self.params['p1_min_area'], self.params['p1_max_area'],
                    self.p1_last_detected
                )
                
                p2_bbox, p2_area, p2_contours, p2_mask = self.detect_player_in_zone(
                    frame, self.p2_mog2, self.p2_zone,
                    self.params['p2_min_area'], self.params['p2_max_area'],
                    self.p2_last_detected
                )
                
                # Update detection state
                p1_found = p1_bbox is not None
                p2_found = p2_bbox is not None
                
                # Update for next frame
                self.p1_last_detected = p1_found
                self.p2_last_detected = p2_found
                
                # Draw visualization
                display_frame = frame.copy()
                
                # Draw zones (yellow)
                cv2.polylines(display_frame, [np.array(self.p1_zone, dtype=np.int32)], True, (0, 255, 255), 2)
                cv2.polylines(display_frame, [np.array(self.p2_zone, dtype=np.int32)], True, (0, 255, 255), 2)
                
                # Draw detections (green boxes)
                if p1_bbox:
                    x, y, w, h = p1_bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                    cv2.putText(display_frame, f"P1: {p1_area:.0f}px", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if p2_bbox:
                    x, y, w, h = p2_bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                    cv2.putText(display_frame, f"P2: {p2_area:.0f}px", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Status text - BIG and CLEAR
                status = "BOTH DETECTED" if (p1_found and p2_found) else ("P1 ONLY" if p1_found else ("P2 ONLY" if p2_found else "NONE DETECTED"))
                color = (0, 255, 0) if (p1_found and p2_found) else ((0, 255, 255) if (p1_found or p2_found) else (0, 0, 255))
                
                # Black background for text
                cv2.rectangle(display_frame, (5, 5), (700, 90), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Frame {self.frame_num}",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(display_frame, status,
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Add legend
                cv2.rectangle(display_frame, (5, 100), (400, 180), (0, 0, 0), -1)
                cv2.putText(display_frame, "Press 'D' for next frame",
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'Q' to quit",
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Manual Navigator', display_frame)
                
                # Print to console
                status_str = "OK" if (p1_found and p2_found) else ("P1" if p1_found else ("P2" if p2_found else "NO"))
                p1_str = f"{p1_area:.0f}px" if p1_found else (f"NO (largest={p1_contours[0]['area']:.0f}px)" if p1_contours else "NO (no contours)")
                p2_str = f"{p2_area:.0f}px" if p2_found else (f"NO (largest={p2_contours[0]['area']:.0f}px)" if p2_contours else "NO (no contours)")
                print(f"Frame {self.frame_num:3d}: [{status_str:>2}] P1={p1_str:<25} P2={p2_str}")
                
                self.frame_num += 1
                waiting_for_keypress = True
            
            else:
                # Show current frame while waiting
                if self.current_frame is not None:
                    cv2.imshow('Manual Navigator', self.current_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('d'):
                waiting_for_keypress = False
        
        print(f"\nReached frame {max_frame}. Press 'Q' to quit or 'D' to continue...")
        
        # Wait for quit
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    navigator = ManualForwardNavigator(video_path)
    navigator.run(max_frame=10)


