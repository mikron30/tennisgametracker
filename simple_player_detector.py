"""
Simple player detector focused ONLY on the right tennis court.
Uses HSV color filtering for skin and clothing detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class SimplePlayerDetector:
    def __init__(self):
        """Initialize simple player detector with right court focus."""
        # Right court boundaries (X >= 1920 for 3840x2160 video)
        self.right_court_x_min = 1920
        
        # Player size thresholds (adjusted for 4K video: 3840x2160)
        # Players appear as 80-200px wide, 100-300px tall
        self.min_player_area = 2000  # Minimum 2000 pixels
        self.max_player_area = 60000  # Maximum 60000 pixels (increased for 4K)
        
        # HSV ranges for person detection
        # We look for: skin tones + white/light clothing
        
        # Skin detection (face, arms, legs)
        self.skin_lower = np.array([0, 20, 60], dtype=np.uint8)
        self.skin_upper = np.array([25, 255, 255], dtype=np.uint8)
        
        # White/light clothing
        self.light_lower = np.array([0, 0, 150], dtype=np.uint8)
        self.light_upper = np.array([180, 40, 255], dtype=np.uint8)
        
        # Gray clothing
        self.gray_lower = np.array([0, 0, 80], dtype=np.uint8)
        self.gray_upper = np.array([180, 50, 180], dtype=np.uint8)
    
    def detect_players(self, frame: np.ndarray, debug=False) -> List[Tuple[int, int, int, int]]:
        """
        Detect players in the right court only.
        
        Args:
            frame: Full video frame
            debug: If True, show debug visualizations
            
        Returns:
            List of (x, y, w, h) bounding boxes for detected players, sorted top-to-bottom
        """
        height, width = frame.shape[:2]
        
        # STEP 1: Extract ONLY the right court
        right_court = frame[:, self.right_court_x_min:]
        
        # STEP 2: Convert to HSV
        hsv = cv2.cvtColor(right_court, cv2.COLOR_BGR2HSV)
        
        # STEP 3: Create masks for different colors
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        light_mask = cv2.inRange(hsv, self.light_lower, self.light_upper)
        gray_mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)
        
        # STEP 4: Combine masks
        combined_mask = cv2.bitwise_or(skin_mask, light_mask)
        combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
        
        # STEP 5: Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # STEP 6: Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 7: Filter contours to find players
        player_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if not (self.min_player_area <= area <= self.max_player_area):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (person should be taller than wide)
            aspect_ratio = h / w if w > 0 else 0
            if not (1.0 <= aspect_ratio <= 3.5):
                continue
            
            # Adjust X coordinate back to full frame coordinates
            x_full = x + self.right_court_x_min
            
            player_candidates.append({
                'bbox': (x_full, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center_y': y + h // 2
            })
        
        # STEP 8: Sort players by Y coordinate (top = P1, bottom = P2)
        player_candidates.sort(key=lambda p: p['center_y'])
        
        # STEP 9: Return top 2 players (P1 and P2)
        players = [p['bbox'] for p in player_candidates[:2]]
        
        if debug:
            print(f"\n=== SimplePlayerDetector Debug ===")
            print(f"Right court region: X >= {self.right_court_x_min}")
            print(f"Found {len(contours)} total contours")
            print(f"Found {len(player_candidates)} player candidates:")
            for i, p in enumerate(player_candidates):
                x, y, w, h = p['bbox']
                print(f"  {i+1}. pos=({x},{y}), size=({w}x{h}), area={p['area']:.0f}, aspect={p['aspect_ratio']:.2f}")
            print(f"Returning {len(players)} players")
            
            # Show masks
            if len(player_candidates) > 0:
                # Draw bounding boxes on right court
                debug_frame = right_court.copy()
                for p in player_candidates:
                    x_rel = p['bbox'][0] - self.right_court_x_min
                    y = p['bbox'][1]
                    w = p['bbox'][2]
                    h = p['bbox'][3]
                    cv2.rectangle(debug_frame, (x_rel, y), (x_rel + w, y + h), (0, 255, 0), 3)
                
                # Resize for display
                scale = min(1920 / debug_frame.shape[1], 1080 / debug_frame.shape[0])
                new_w = int(debug_frame.shape[1] * scale)
                new_h = int(debug_frame.shape[0] * scale)
                debug_resized = cv2.resize(debug_frame, (new_w, new_h))
                mask_resized = cv2.resize(combined_mask, (new_w, new_h))
                
                cv2.imshow("Right Court with Detections", debug_resized)
                cv2.imshow("Combined Mask", mask_resized)
        
        return players
    
    def set_right_court_boundary(self, x_min: int):
        """Set the X coordinate where the right court starts."""
        self.right_court_x_min = x_min
        print(f"Right court boundary set to X >= {x_min}")
    
    def tune_hsv(self, frame: np.ndarray):
        """
        Interactive HSV tuner for player detection.
        
        Args:
            frame: Video frame to tune on
        """
        # Extract right court
        right_court = frame[:, self.right_court_x_min:]
        
        window_name = "Player Detection HSV Tuner"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 800)
        
        # Create trackbars for skin detection
        cv2.createTrackbar("Skin H Min", window_name, int(self.skin_lower[0]), 179, lambda x: None)
        cv2.createTrackbar("Skin H Max", window_name, int(self.skin_upper[0]), 179, lambda x: None)
        cv2.createTrackbar("Skin S Min", window_name, int(self.skin_lower[1]), 255, lambda x: None)
        cv2.createTrackbar("Skin S Max", window_name, int(self.skin_upper[1]), 255, lambda x: None)
        cv2.createTrackbar("Skin V Min", window_name, int(self.skin_lower[2]), 255, lambda x: None)
        cv2.createTrackbar("Skin V Max", window_name, int(self.skin_upper[2]), 255, lambda x: None)
        
        # Create trackbars for light clothing
        cv2.createTrackbar("Light V Min", window_name, int(self.light_lower[2]), 255, lambda x: None)
        cv2.createTrackbar("Light S Max", window_name, int(self.light_upper[1]), 255, lambda x: None)
        
        # Area thresholds
        cv2.createTrackbar("Min Area / 100", window_name, int(self.min_player_area / 100), 500, lambda x: None)
        cv2.createTrackbar("Max Area / 100", window_name, int(self.max_player_area / 100), 1000, lambda x: None)
        
        print("\n=== Player Detection HSV Tuner ===")
        print("Adjust trackbars to tune player detection")
        print("Press 'Q' or ESC to save and exit")
        
        while True:
            # Get trackbar values
            skin_h_min = cv2.getTrackbarPos("Skin H Min", window_name)
            skin_h_max = cv2.getTrackbarPos("Skin H Max", window_name)
            skin_s_min = cv2.getTrackbarPos("Skin S Min", window_name)
            skin_s_max = cv2.getTrackbarPos("Skin S Max", window_name)
            skin_v_min = cv2.getTrackbarPos("Skin V Min", window_name)
            skin_v_max = cv2.getTrackbarPos("Skin V Max", window_name)
            
            light_v_min = cv2.getTrackbarPos("Light V Min", window_name)
            light_s_max = cv2.getTrackbarPos("Light S Max", window_name)
            
            min_area = cv2.getTrackbarPos("Min Area / 100", window_name) * 100
            max_area = cv2.getTrackbarPos("Max Area / 100", window_name) * 100
            
            # Update detection parameters
            self.skin_lower = np.array([skin_h_min, skin_s_min, skin_v_min], dtype=np.uint8)
            self.skin_upper = np.array([skin_h_max, skin_s_max, skin_v_max], dtype=np.uint8)
            self.light_lower[2] = light_v_min
            self.light_upper[1] = light_s_max
            self.min_player_area = min_area
            self.max_player_area = max_area
            
            # Run detection
            players = self.detect_players(frame, debug=False)
            
            # Visualize
            hsv = cv2.cvtColor(right_court, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
            light_mask = cv2.inRange(hsv, self.light_lower, self.light_upper)
            gray_mask = cv2.inRange(hsv, self.gray_lower, self.gray_upper)
            combined_mask = cv2.bitwise_or(skin_mask, light_mask)
            combined_mask = cv2.bitwise_or(combined_mask, gray_mask)
            
            # Draw detections
            display = right_court.copy()
            for i, (x, y, w, h) in enumerate(players):
                x_rel = x - self.right_court_x_min
                color = (255, 0, 0) if i == 0 else (0, 0, 255)
                label = "P1" if i == 0 else "P2"
                cv2.rectangle(display, (x_rel, y), (x_rel + w, y + h), color, 3)
                cv2.putText(display, label, (x_rel, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Combine displays
            mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([display, mask_bgr])
            
            # Add info text
            info_text = f"Players: {len(players)} | Min Area: {min_area} | Max Area: {max_area}"
            cv2.putText(combined, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Resize for display
            scale = min(1400 / combined.shape[1], 800 / combined.shape[0])
            new_w = int(combined.shape[1] * scale)
            new_h = int(combined.shape[0] * scale)
            combined_resized = cv2.resize(combined, (new_w, new_h))
            
            cv2.imshow(window_name, combined_resized)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        cv2.destroyWindow(window_name)
        print("HSV tuner closed. Parameters saved.")



