"""
Zone-based MOG2 tracker for P1 and P2.
Uses court geometry zones + MOG2 background subtraction to detect moving players.
"""

import cv2
import numpy as np


class ZoneMOG2Tracker:
    def __init__(self):
        """Initialize zone-based MOG2 tracker."""
        
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
        
        # P1 ZONE: Top-left area (between baseline and service line, extended)
        self.p1_zone = np.array([
            (self.court_points[1][0] - 100, self.court_points[1][1] - 150),  # Extend up and left
            (self.court_points[13][0] + 50, self.court_points[13][1] - 150),
            (self.court_points[6][0] + 150, self.court_points[6][1] + 50),
            (self.court_points[5][0] - 100, self.court_points[5][1] + 150),  # Extend down
        ], dtype=np.int32)
        
        # P2 ZONE: Bottom-right area (between service line and baseline, extended)
        self.p2_zone = np.array([
            (self.court_points[7][0] - 100, self.court_points[7][1] - 100),
            (self.court_points[8][0] + 150, self.court_points[8][1] - 100),
            (self.court_points[14][0] + 150, self.court_points[14][1] + 100),
            (self.court_points[3][0] - 100, self.court_points[3][1] + 100),
        ], dtype=np.int32)
        
        # Create separate MOG2 background subtractors for each zone
        self.p1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=8,
            detectShadows=True
        )
        
        self.p2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=8,
            detectShadows=True
        )
        
        # Player size thresholds
        self.p1_min_area = 1000
        self.p1_max_area = 50000
        self.p2_min_area = 500
        self.p2_max_area = 30000
        
        self.learning_rate = 0.001
        
        # Last known positions
        self.p1_position = None
        self.p2_position = None
    
    def detect_player_in_zone(self, frame, zone, bg_subtractor, min_area, max_area, player_name="Player"):
        """Detect moving player in a specific zone using MOG2."""
        
        # Create zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone], 255)
        
        # Apply background subtraction to entire frame
        fg_mask = bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Apply zone mask to restrict to zone only
        fg_mask = cv2.bitwise_and(fg_mask, zone_mask)
        
        # Clean up mask with morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest valid contour
        best_player = None
        best_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                if area > best_area:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        best_player = {
                            'position': (cx, cy),
                            'bbox': (x, y, w, h),
                            'area': area
                        }
                        best_area = area
        
        return best_player, fg_mask
    
    def track_players(self, frame):
        """Track both P1 and P2 using MOG2 in their respective zones."""
        
        # Detect P1
        p1_player, p1_mask = self.detect_player_in_zone(
            frame, self.p1_zone, self.p1_bg_subtractor,
            self.p1_min_area, self.p1_max_area, "P1"
        )
        
        # Detect P2
        p2_player, p2_mask = self.detect_player_in_zone(
            frame, self.p2_zone, self.p2_bg_subtractor,
            self.p2_min_area, self.p2_max_area, "P2"
        )
        
        players = []
        
        if p1_player:
            self.p1_position = p1_player['position']
            players.append({'name': 'P1', **p1_player})
        
        if p2_player:
            self.p2_position = p2_player['position']
            players.append({'name': 'P2', **p2_player})
        
        return players, p1_mask, p2_mask
    
    def draw_court_lines(self, frame):
        """Draw court lines on the frame."""
        display = frame.copy()
        
        # Define court line connections
        lines = [
            # Baselines
            (1, 13, (0, 255, 0), "Top Baseline"),
            (13, 2, (0, 255, 0), None),
            (3, 14, (0, 255, 0), "Bottom Baseline"),
            (14, 4, (0, 255, 0), None),
            
            # Sidelines (following court geometry)
            (1, 5, (255, 0, 0), "Left Sideline"),
            (5, 7, (255, 0, 0), None),
            (7, 3, (255, 0, 0), None),
            (2, 6, (255, 0, 0), "Right Sideline"),
            (6, 8, (255, 0, 0), None),
            (8, 4, (255, 0, 0), None),
            
            # Service lines
            (5, 6, (0, 255, 255), "Service Line (top)"),
            (7, 8, (0, 255, 255), "Service Line (bottom)"),
            
            # Net
            (9, 10, (255, 255, 0), "Net"),
            
            # Center service line
            (11, 12, (255, 0, 255), "Center Service"),
        ]
        
        # Draw lines
        for line in lines:
            p1_idx, p2_idx, color, label = line
            p1 = self.court_points[p1_idx]
            p2 = self.court_points[p2_idx]
            cv2.line(display, p1, p2, color, 2)
        
        return display
    
    def visualize(self, frame, players, show_zones=True, show_lines=True, show_masks=False, p1_mask=None, p2_mask=None):
        """Visualize players, zones, and court lines."""
        
        # Start with court lines
        if show_lines:
            display = self.draw_court_lines(frame)
        else:
            display = frame.copy()
        
        # Draw zones with transparency
        if show_zones:
            overlay = display.copy()
            cv2.fillPoly(overlay, [self.p1_zone], (255, 100, 100))  # Blue for P1
            cv2.fillPoly(overlay, [self.p2_zone], (100, 200, 255))  # Orange for P2
            display = cv2.addWeighted(display, 0.88, overlay, 0.12, 0)
            
            # Draw zone outlines
            cv2.polylines(display, [self.p1_zone], True, (255, 0, 0), 3)
            cv2.polylines(display, [self.p2_zone], True, (0, 165, 255), 3)
            
            # Add zone labels
            cv2.putText(display, "P1 ZONE", (self.p1_zone[0][0] + 20, self.p1_zone[0][1] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            cv2.putText(display, "P2 ZONE", (self.p2_zone[0][0] + 20, self.p2_zone[0][1] + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        
        # Draw players
        for player in players:
            name = player['name']
            pos = player['position']
            bbox = player['bbox']
            area = player['area']
            
            color = (255, 0, 0) if name == 'P1' else (0, 165, 255)
            
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 5)
            
            # Draw center
            cv2.circle(display, pos, 15, color, -1)
            cv2.circle(display, pos, 15, (255, 255, 255), 2)
            
            # Draw label with background
            label = f"{name} ({area:.0f}px)"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.rectangle(display, (x, y - label_h - 25), (x + label_w + 15, y - 5), color, -1)
            cv2.putText(display, label, (x + 8, y - 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Optionally show masks overlay
        if show_masks and p1_mask is not None and p2_mask is not None:
            # Create colored overlay for masks
            mask_overlay = np.zeros_like(display)
            mask_overlay[p1_mask > 0] = [255, 100, 100]  # Blue for P1
            mask_overlay[p2_mask > 0] = [100, 200, 255]  # Orange for P2
            display = cv2.addWeighted(display, 0.85, mask_overlay, 0.15, 0)
        
        return display


def main():
    """Test the zone-based MOG2 tracker."""
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    tracker = ZoneMOG2Tracker()
    
    print("\n" + "="*70)
    print("ZONE-BASED MOG2 PLAYER TRACKER")
    print("="*70)
    print("Detects moving players in defined court zones using MOG2")
    print("\nFeatures:")
    print("  - P1 Zone: Top-left area (blue)")
    print("  - P2 Zone: Bottom-right area (orange)")
    print("  - Court lines displayed")
    print("  - MOG2 background subtraction per zone")
    print("\nControls:")
    print("  SPACE = Pause/Resume")
    print("  'Z' = Toggle zone overlay")
    print("  'L' = Toggle court lines")
    print("  'M' = Toggle mask overlay")
    print("  'Q' = Quit")
    print("="*70)
    
    cv2.namedWindow("Zone MOG2 Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zone MOG2 Tracker", 1600, 900)
    
    # Start from beginning to build background model
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    paused = False
    show_zones = True
    show_lines = True
    show_masks = False
    frame_num = 0
    
    print("\nBuilding background model (first 30 frames)...")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break
            
            frame_num += 1
            
            # Track players
            players, p1_mask, p2_mask = tracker.track_players(frame)
            
            # Visualize
            display = tracker.visualize(frame, players, show_zones, show_lines, show_masks, p1_mask, p2_mask)
            
            # Add info panel
            info_lines = [
                f"Frame: {frame_num}",
                f"P1: {'DETECTED' if any(p['name'] == 'P1' for p in players) else 'NOT FOUND'}",
                f"P2: {'DETECTED' if any(p['name'] == 'P2' for p in players) else 'NOT FOUND'}",
                "Press SPACE to pause"
            ]
            
            y_offset = 50
            for line in info_lines:
                cv2.putText(display, line, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                y_offset += 40
            
            # Resize and show
            height, width = display.shape[:2]
            scale = min(1600 / width, 900 / height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            display_resized = cv2.resize(display, (new_w, new_h))
            
            cv2.imshow("Zone MOG2 Tracker", display_resized)
        
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            status = 'PAUSED' if paused else 'PLAYING'
            print(f"\n{status} at frame {frame_num}")
        elif key == ord('z'):
            show_zones = not show_zones
            print(f"Zone overlay: {'ON' if show_zones else 'OFF'}")
        elif key == ord('l'):
            show_lines = not show_lines
            print(f"Court lines: {'ON' if show_lines else 'OFF'}")
        elif key == ord('m'):
            show_masks = not show_masks
            print(f"Mask overlay: {'ON' if show_masks else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nTracking complete!")


if __name__ == "__main__":
    main()



