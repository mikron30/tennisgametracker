#!/usr/bin/env python3
"""
Modified interactive tuner that logs detection results to a file
This will help us compare interactive results with debug script results
"""

import cv2
import numpy as np
import json
import os

class InteractiveLogger:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frame_num = 0
        self.paused = True
        
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
        
        # Log file
        self.log_data = []
        
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
            return (x, y, w, h), area, all_contours
        
        return None, None, all_contours
    
    def run(self):
        """Run the interactive logger"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        print("="*70)
        print("INTERACTIVE DETECTION LOGGER")
        print("="*70)
        print(f"Sensitivity: {self.sensitivity}")
        print(f"P1 area: {self.params['p1_min_area']:,} - {self.params['p1_max_area']:,}px")
        print(f"P2 area: {self.params['p2_min_area']:,} - {self.params['p2_max_area']:,}px")
        print()
        print("CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'Q' = Save log and quit")
        print("  SPACE = Play/pause")
        print()
        print("Starting at frame 0...")
        print("Navigate through frames 1-49 using 'D' key")
        print("="*70)
        print()
        
        cv2.namedWindow('Detection Logger')
        
        while True:
            if not self.paused or self.frame_num == 0:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"\nReached end of video at frame {self.frame_num}")
                    break
                
                # Detect players
                p1_bbox, p1_area, p1_contours = self.detect_player_in_zone(
                    frame, self.p1_mog2, self.p1_zone,
                    self.params['p1_min_area'], self.params['p1_max_area'],
                    self.p1_last_detected
                )
                
                p2_bbox, p2_area, p2_contours = self.detect_player_in_zone(
                    frame, self.p2_mog2, self.p2_zone,
                    self.params['p2_min_area'], self.params['p2_max_area'],
                    self.p2_last_detected
                )
                
                # Update detection state
                p1_found = p1_bbox is not None
                p2_found = p2_bbox is not None
                self.p1_last_detected = p1_found
                self.p2_last_detected = p2_found
                
                # Log results
                log_entry = {
                    'frame': self.frame_num,
                    'p1_detected': p1_found,
                    'p1_area': int(p1_area) if p1_area else None,
                    'p1_largest_contour': int(p1_contours[0]['area']) if p1_contours else None,
                    'p2_detected': p2_found,
                    'p2_area': int(p2_area) if p2_area else None,
                    'p2_largest_contour': int(p2_contours[0]['area']) if p2_contours else None
                }
                self.log_data.append(log_entry)
                
                # Draw visualization
                display_frame = frame.copy()
                
                # Draw zones
                cv2.polylines(display_frame, [np.array(self.p1_zone, dtype=np.int32)], True, (255, 255, 0), 2)
                cv2.polylines(display_frame, [np.array(self.p2_zone, dtype=np.int32)], True, (255, 255, 0), 2)
                
                # Draw detections
                if p1_bbox:
                    x, y, w, h = p1_bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"P1: {p1_area:.0f}px", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if p2_bbox:
                    x, y, w, h = p2_bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"P2: {p2_area:.0f}px", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Status text
                status = "BOTH" if (p1_found and p2_found) else ("P1" if p1_found else ("P2" if p2_found else "NONE"))
                cv2.putText(display_frame, f"Frame {self.frame_num} - Detected: {status}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Detection Logger', display_frame)
                
                # Print to console
                status_str = "OK" if (p1_found and p2_found) else ("P1" if p1_found else ("P2" if p2_found else "NO"))
                p1_str = f"{p1_area:.0f}px" if p1_found else f"NO (largest={p1_contours[0]['area']:.0f}px)" if p1_contours else "NO (no contours)"
                p2_str = f"{p2_area:.0f}px" if p2_found else f"NO (largest={p2_contours[0]['area']:.0f}px)" if p2_contours else "NO (no contours)"
                print(f"Frame {self.frame_num:3d}: [{status_str:>2}] P1={p1_str:<25} P2={p2_str}")
                
                self.frame_num += 1
            else:
                cv2.imshow('Detection Logger', frame)
            
            key = cv2.waitKey(30 if not self.paused else 1) & 0xFF
            
            if key == ord('q'):
                self._save_log()
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('d') and self.paused:
                self.paused = False
            elif key == ord('a') and self.paused and self.frame_num > 1:
                # Go back is tricky with MOG2, so we'll just warn
                print("Warning: Going back not supported with MOG2 state")
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _save_log(self):
        """Save log to file"""
        filename = 'interactive_detection_log.json'
        with open(filename, 'w') as f:
            json.dump({
                'sensitivity': self.sensitivity,
                'parameters': self.params,
                'frames': self.log_data
            }, f, indent=2)
        
        # Also create a summary
        p1_detected = sum(1 for f in self.log_data if f['p1_detected'])
        p2_detected = sum(1 for f in self.log_data if f['p2_detected'])
        both_detected = sum(1 for f in self.log_data if f['p1_detected'] and f['p2_detected'])
        total = len(self.log_data)
        
        print()
        print("="*70)
        print("LOG SAVED")
        print("="*70)
        print(f"File: {filename}")
        print(f"Total frames: {total}")
        print(f"P1 detected: {p1_detected}/{total} ({100*p1_detected/total:.1f}%)")
        print(f"P2 detected: {p2_detected}/{total} ({100*p2_detected/total:.1f}%)")
        print(f"Both detected: {both_detected}/{total} ({100*both_detected/total:.1f}%)")
        print("="*70)

if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    logger = InteractiveLogger(video_path)
    logger.run()

