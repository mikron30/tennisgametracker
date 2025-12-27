import cv2
import json
import numpy as np

def load_zones():
    """Load P1 and P2 zones"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

class AutoShirtFinder:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 10  # Start at frame 10 where players should be visible
        
        # Load zones
        self.p1_zone, self.p2_zone = load_zones()
        
        # Common shirt color ranges (expanded)
        self.color_presets = {
            'White': [(0, 180, 0, 30, 200, 255)],
            'Black': [(0, 180, 0, 255, 0, 50)],
            'Red': [(0, 10, 100, 255, 100, 255), (170, 180, 100, 255, 100, 255)],
            'Orange': [(10, 25, 100, 255, 100, 255)],
            'Yellow': [(25, 35, 100, 255, 100, 255)],
            'Green': [(35, 85, 50, 255, 50, 255)],
            'Cyan': [(85, 95, 50, 255, 50, 255)],
            'Blue': [(95, 130, 50, 255, 50, 255)],
            'Purple': [(130, 160, 50, 255, 50, 255)],
            'Pink': [(160, 170, 50, 255, 100, 255)],
            'Gray': [(0, 180, 0, 50, 50, 200)],
            'Light_Blue': [(95, 115, 30, 100, 150, 255)],
            'Dark_Blue': [(100, 130, 100, 255, 50, 150)],
            'Light_Green': [(35, 85, 30, 100, 150, 255)],
            'Dark_Green': [(35, 85, 100, 255, 50, 150)],
        }
        
        print("=" * 70)
        print("AUTOMATIC SHIRT COLOR FINDER")
        print("=" * 70)
        print(f"Testing {len(self.color_presets)} color presets on frame {self.current_frame}")
        print("\nTesting colors:")
        for color_name in self.color_presets.keys():
            print(f"  - {color_name}")
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10)")
        print("  'B' = Fast backward (-10)")
        print("  'T' = Test all colors on current frame")
        print("  'Q' = Quit")
        print("=" * 70)
    
    def test_color_in_zone(self, frame, zone_polygon, hsv_ranges, zone_name):
        """Test a color range in a specific zone"""
        # Create zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Try each HSV range (some colors have multiple ranges, like red)
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for hsv_range in hsv_ranges:
            h_min, h_max, s_min, s_max, v_min, v_max = hsv_range
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            color_mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)
        
        # Apply zone mask
        zone_color_mask = cv2.bitwise_and(combined_mask, zone_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        zone_color_mask = cv2.morphologyEx(zone_color_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        zone_color_mask = cv2.morphologyEx(zone_color_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(zone_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0, zone_color_mask
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        bbox = cv2.boundingRect(largest_contour)
        
        return bbox, area, zone_color_mask
    
    def test_all_colors(self, frame):
        """Test all color presets and rank them"""
        print("\n" + "=" * 70)
        print(f"TESTING ALL COLORS - Frame {self.current_frame}")
        print("=" * 70)
        
        p1_results = []
        p2_results = []
        
        # Test each color
        for color_name, hsv_ranges in self.color_presets.items():
            # Test P1
            p1_bbox, p1_area, p1_mask = self.test_color_in_zone(frame, self.p1_zone, hsv_ranges, "P1")
            if p1_area > 1000:  # Minimum threshold
                p1_results.append((color_name, p1_area, p1_bbox, p1_mask))
            
            # Test P2
            p2_bbox, p2_area, p2_mask = self.test_color_in_zone(frame, self.p2_zone, hsv_ranges, "P2")
            if p2_area > 1000:  # Minimum threshold
                p2_results.append((color_name, p2_area, p2_bbox, p2_mask))
        
        # Sort by area (largest first)
        p1_results.sort(key=lambda x: x[1], reverse=True)
        p2_results.sort(key=lambda x: x[1], reverse=True)
        
        # Print results
        print("\nP1 ZONE RESULTS (top 5):")
        for i, (color_name, area, bbox, _) in enumerate(p1_results[:5]):
            x, y, w, h = bbox
            print(f"  {i+1}. {color_name:15s} - {int(area):6d}px - bbox: ({x},{y},{w},{h})")
        
        print("\nP2 ZONE RESULTS (top 5):")
        for i, (color_name, area, bbox, _) in enumerate(p2_results[:5]):
            x, y, w, h = bbox
            print(f"  {i+1}. {color_name:15s} - {int(area):6d}px - bbox: ({x},{y},{w},{h})")
        
        print("=" * 70)
        
        # Show top 3 results for each player
        self.show_top_results(frame, p1_results[:3], p2_results[:3])
        
        return p1_results, p2_results
    
    def show_top_results(self, frame, p1_top3, p2_top3):
        """Show top 3 results for each player in grid"""
        # Create display grid
        h, w = frame.shape[:2]
        display_h = int(h * 0.4)
        display_w = int(w * 0.4)
        
        # P1 results
        print("\nDisplaying P1 top 3 results...")
        for i, (color_name, area, bbox, mask) in enumerate(p1_top3):
            # Show mask
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Draw bbox on mask
            if bbox:
                x, y, w_box, h_box = bbox
                cv2.rectangle(mask_display, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(mask_display, f"P1 - {color_name} ({int(area)}px)", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw zone outline
            cv2.polylines(mask_display, [self.p1_zone], True, (255, 255, 0), 2)
            
            # Resize and show
            mask_display = cv2.resize(mask_display, (display_w, display_h))
            cv2.imshow(f'P1 - Top {i+1}: {color_name}', mask_display)
        
        # P2 results
        print("Displaying P2 top 3 results...")
        for i, (color_name, area, bbox, mask) in enumerate(p2_top3):
            # Show mask
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Draw bbox on mask
            if bbox:
                x, y, w_box, h_box = bbox
                cv2.rectangle(mask_display, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(mask_display, f"P2 - {color_name} ({int(area)}px)", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw zone outline
            cv2.polylines(mask_display, [self.p2_zone], True, (255, 255, 0), 2)
            
            # Resize and show
            mask_display = cv2.resize(mask_display, (display_w, display_h))
            cv2.imshow(f'P2 - Top {i+1}: {color_name}', mask_display)
        
        print("Press any key in windows to continue...")
    
    def draw_frame(self, frame):
        """Draw current frame with zones"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 255, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (255, 255, 0), 2)
        cv2.putText(display, "P1 Zone", tuple(self.p1_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display, "P2 Zone", tuple(self.p2_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Frame info
        cv2.putText(display, f"Frame: {self.current_frame}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "Press 'T' to test colors", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return display
    
    def run(self):
        """Main loop"""
        cv2.namedWindow('Auto Shirt Finder', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Auto Shirt Finder', 1600, 900)
        
        # Auto-test on first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.test_all_colors(frame)
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            display = self.draw_frame(frame)
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Auto Shirt Finder', display_resized)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('f'):
                self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('b'):
                self.current_frame = max(self.current_frame - 10, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('t'):
                self.test_all_colors(frame)
            elif key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    finder = AutoShirtFinder(video_path)
    finder.run()

