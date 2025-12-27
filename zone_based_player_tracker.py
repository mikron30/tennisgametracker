"""
Zone-based player tracker using court geometry and HSV color filtering.
Each player is detected in their zone and tracked by their clothing color.
"""

import cv2
import numpy as np
import json


class ZoneBasedPlayerTracker:
    def __init__(self):
        """Initialize zone-based player tracker."""
        
        # Court points from adjusted_court_points.txt
        self.court_points = {
            1: (1043, 214),   # Top-left corner baseline
            2: (1608, 170),   # Top-right corner baseline (near net)
            3: (2042, 1449),  # Bottom-left corner baseline
            4: (3002, 1060),  # Bottom-right corner baseline (near net)
            5: (1120, 310),   # Service line left (top)
            6: (1782, 262),   # Service line right (top, near net)
            7: (1549, 854),   # Service line left (bottom)
            8: (2490, 696),   # Service line right (bottom, near net)
            9: (1041, 421),   # Net left
            10: (2229, 331),  # Net right
            11: (1468, 278),  # Center service line top
            12: (2092, 764),  # Center service line bottom
            13: (1334, 186),  # Top singles sideline
            14: (2656, 1216)  # Bottom singles sideline
        }
        
        # Define player zones (enlarged for better detection)
        # P1 ZONE: Between baseline and service line (near side) - ENLARGED
        self.p1_zone = np.array([
            (self.court_points[1][0] - 50, self.court_points[1][1] - 100),  # Extend up and left
            (self.court_points[13][0], self.court_points[13][1] - 100),
            (self.court_points[6][0] + 100, self.court_points[6][1]),
            (self.court_points[5][0] - 50, self.court_points[5][1] + 100),  # Extend down
        ], dtype=np.int32)
        
        # P2 ZONE: Between service line and net (far side) - ENLARGED
        self.p2_zone = np.array([
            (self.court_points[5][0] - 50, self.court_points[5][1] - 50),
            (self.court_points[6][0] + 100, self.court_points[6][1] - 50),
            (self.court_points[10][0] + 100, self.court_points[10][1] + 50),
            (self.court_points[9][0] - 50, self.court_points[9][1] + 50),
        ], dtype=np.int32)
        
        # Player HSV values (will be set during detection)
        self.p1_hsv_lower = None
        self.p1_hsv_upper = None
        self.p2_hsv_lower = None
        self.p2_hsv_upper = None
        
        # Player blob size ranges (for 4K video)
        self.p1_min_area = 1000   # P1 is closer, appears larger
        self.p1_max_area = 50000
        self.p2_min_area = 500    # P2 is farther, appears smaller
        self.p2_max_area = 30000
        
        # Last known positions
        self.p1_position = None
        self.p2_position = None
    
    def is_point_in_zone(self, point, zone):
        """Check if a point is inside a zone polygon."""
        return cv2.pointPolygonTest(zone, point, False) >= 0
    
    def detect_player_in_zone(self, frame, zone, min_area, max_area, zone_name="Zone", debug=False):
        """
        Detect largest moving blob in a specific zone.
        
        Returns:
            tuple: (center_position, hsv_values, area) or (None, None, None)
        """
        # Create mask for the zone
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [zone], 255)
        
        # Extract zone ROI
        zone_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to HSV
        hsv = cv2.cvtColor(zone_frame, cv2.COLOR_BGR2HSV)
        
        # Try multiple HSV ranges to find player-like colors
        # (skin, white, gray, blue, any bright clothing)
        hsv_ranges = [
            # Skin tones
            (np.array([0, 20, 50], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
            # White/light clothing
            (np.array([0, 0, 150], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8)),
            # Gray clothing
            (np.array([0, 0, 50], dtype=np.uint8), np.array([180, 50, 200], dtype=np.uint8)),
            # Blue clothing
            (np.array([90, 50, 50], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
            # Red/Orange clothing
            (np.array([0, 100, 100], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)),
        ]
        
        best_blob = None
        best_area = 0
        best_hsv_range = None
        
        for lower, upper in hsv_ranges:
            # Apply HSV filter
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Apply zone mask
            color_mask = cv2.bitwise_and(color_mask, mask)
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if min_area <= area <= max_area:
                    if area > best_area:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Get average HSV in this blob
                            blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            cv2.drawContours(blob_mask, [contour], 0, 255, -1)
                            mean_hsv = cv2.mean(hsv, mask=blob_mask)[:3]
                            
                            best_blob = {
                                'position': (cx, cy),
                                'hsv': mean_hsv,
                                'area': area,
                                'contour': contour,
                                'hsv_range': (lower, upper)
                            }
                            best_area = area
                            best_hsv_range = (lower, upper)
        
        if best_blob and debug:
            print(f"{zone_name}: Found player at {best_blob['position']}, "
                  f"area={best_blob['area']:.0f}px, "
                  f"HSV=({best_blob['hsv'][0]:.0f},{best_blob['hsv'][1]:.0f},{best_blob['hsv'][2]:.0f})")
        
        if best_blob:
            return best_blob['position'], best_blob['hsv'], best_blob['area'], best_blob['contour'], best_hsv_range
        
        return None, None, None, None, None
    
    def track_player_by_hsv(self, frame, hsv_lower, hsv_upper, zone, min_area, max_area, player_name="Player"):
        """Track a player using their stored HSV signature."""
        # Create zone mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [zone], 255)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply HSV filter
        color_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        color_mask = cv2.bitwise_and(color_mask, mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest valid blob
        best_blob = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                if area > best_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_blob = {'position': (cx, cy), 'area': area, 'contour': contour}
                        best_area = area
        
        if best_blob:
            return best_blob['position'], best_blob['area'], best_blob['contour']
        
        return None, None, None
    
    def initialize_players(self, frame, debug=False):
        """Initialize player detection by finding them in their zones."""
        print("\n" + "="*70)
        print("INITIALIZING PLAYER DETECTION")
        print("="*70)
        
        # Detect P1 in P1 zone
        p1_pos, p1_hsv, p1_area, p1_contour, p1_hsv_range = self.detect_player_in_zone(
            frame, self.p1_zone, self.p1_min_area, self.p1_max_area, "P1 Zone", debug=True
        )
        
        if p1_pos and p1_hsv:
            # Store P1 HSV with some tolerance
            lower, upper = p1_hsv_range
            self.p1_hsv_lower = np.array([
                max(0, lower[0]),
                max(0, lower[1]),
                max(0, lower[2])
            ], dtype=np.uint8)
            self.p1_hsv_upper = np.array([
                min(179, upper[0]),
                min(255, upper[1]),
                min(255, upper[2])
            ], dtype=np.uint8)
            self.p1_position = p1_pos
            print(f"[OK] P1 initialized at {p1_pos}, HSV range: {self.p1_hsv_lower} to {self.p1_hsv_upper}")
        else:
            print("[FAIL] P1 not found in zone")
        
        # Detect P2 in P2 zone
        p2_pos, p2_hsv, p2_area, p2_contour, p2_hsv_range = self.detect_player_in_zone(
            frame, self.p2_zone, self.p2_min_area, self.p2_max_area, "P2 Zone", debug=True
        )
        
        if p2_pos and p2_hsv:
            # Store P2 HSV with some tolerance
            lower, upper = p2_hsv_range
            self.p2_hsv_lower = np.array([
                max(0, lower[0]),
                max(0, lower[1]),
                max(0, lower[2])
            ], dtype=np.uint8)
            self.p2_hsv_upper = np.array([
                min(179, upper[0]),
                min(255, upper[1]),
                min(255, upper[2])
            ], dtype=np.uint8)
            self.p2_position = p2_pos
            print(f"[OK] P2 initialized at {p2_pos}, HSV range: {self.p2_hsv_lower} to {self.p2_hsv_upper}")
        else:
            print("[FAIL] P2 not found in zone")
        
        print("="*70)
        
        return (p1_pos is not None), (p2_pos is not None)
    
    def track_players(self, frame):
        """Track both players using their stored HSV signatures."""
        players = []
        
        # Track P1
        if self.p1_hsv_lower is not None:
            p1_pos, p1_area, p1_contour = self.track_player_by_hsv(
                frame, self.p1_hsv_lower, self.p1_hsv_upper,
                self.p1_zone, self.p1_min_area, self.p1_max_area, "P1"
            )
            if p1_pos:
                x, y, w, h = cv2.boundingRect(p1_contour)
                players.append({
                    'name': 'P1',
                    'position': p1_pos,
                    'bbox': (x, y, w, h),
                    'area': p1_area
                })
                self.p1_position = p1_pos
        
        # Track P2
        if self.p2_hsv_lower is not None:
            p2_pos, p2_area, p2_contour = self.track_player_by_hsv(
                frame, self.p2_hsv_lower, self.p2_hsv_upper,
                self.p2_zone, self.p2_min_area, self.p2_max_area, "P2"
            )
            if p2_pos:
                x, y, w, h = cv2.boundingRect(p2_contour)
                players.append({
                    'name': 'P2',
                    'position': p2_pos,
                    'bbox': (x, y, w, h),
                    'area': p2_area
                })
                self.p2_position = p2_pos
        
        return players
    
    def visualize(self, frame, players, show_zones=True):
        """Visualize players and zones."""
        display = frame.copy()
        
        # Draw zones with transparency
        if show_zones:
            overlay = display.copy()
            cv2.fillPoly(overlay, [self.p1_zone], (255, 100, 100))  # Blue for P1
            cv2.fillPoly(overlay, [self.p2_zone], (100, 100, 255))  # Red for P2
            display = cv2.addWeighted(display, 0.85, overlay, 0.15, 0)
            
            # Draw zone outlines
            cv2.polylines(display, [self.p1_zone], True, (255, 0, 0), 3)
            cv2.polylines(display, [self.p2_zone], True, (0, 0, 255), 3)
        
        # Draw players
        for player in players:
            name = player['name']
            pos = player['position']
            bbox = player['bbox']
            area = player['area']
            
            color = (255, 0, 0) if name == 'P1' else (0, 0, 255)
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 4)
            
            # Draw center
            cv2.circle(display, pos, 10, color, -1)
            
            # Draw label with background
            label = f"{name} ({area:.0f}px)"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(display, (x, y - label_h - 15), (x + label_w + 10, y - 5), color, -1)
            cv2.putText(display, label, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return display


def main():
    """Test the zone-based player tracker."""
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    tracker = ZoneBasedPlayerTracker()
    
    print("\n" + "="*70)
    print("ZONE-BASED PLAYER TRACKER")
    print("="*70)
    print("This tracker uses court zones + HSV color filtering")
    print("\nControls:")
    print("  SPACE = Pause/Resume")
    print("  'I' = Re-initialize players")
    print("  'Z' = Toggle zone overlay")
    print("  'Q' = Quit")
    print("="*70)
    
    # Initialize on frame 50
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return
    
    p1_found, p2_found = tracker.initialize_players(frame, debug=True)
    
    if not (p1_found and p2_found):
        print("\nWARNING: Not all players were initialized!")
        print("Press 'I' during playback to re-initialize")
    
    cv2.namedWindow("Zone-Based Player Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zone-Based Player Tracker", 1600, 900)
    
    paused = False
    show_zones = True
    frame_num = 50
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break
            
            frame_num += 1
            
            # Track players
            players = tracker.track_players(frame)
            
            # Visualize
            display = tracker.visualize(frame, players, show_zones=show_zones)
            
            # Add info
            info_text = f"Frame: {frame_num} | Players: {len(players)} | Press SPACE to pause"
            cv2.putText(display, info_text, (10, display.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Resize and show
            height, width = display.shape[:2]
            scale = min(1600 / width, 900 / height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            display_resized = cv2.resize(display, (new_w, new_h))
            
            cv2.imshow("Zone-Based Player Tracker", display_resized)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'PAUSED' if paused else 'PLAYING'}")
        elif key == ord('i'):
            print("\nRe-initializing players...")
            p1_found, p2_found = tracker.initialize_players(frame, debug=True)
        elif key == ord('z'):
            show_zones = not show_zones
            print(f"Zone overlay: {'ON' if show_zones else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

