import cv2
import numpy as np
import json
from typing import Tuple, Optional


class HSVFilterDebugger:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        
        # Load HSV values
        self.load_hsv_config()
        
        # Player positions
        self.player1_pos = (1266, 114)
        self.player2_pos = (3142, 1036)
        
    def load_hsv_config(self):
        """Load HSV values from config file."""
        try:
            with open("hsv_config.json", "r") as f:
                config = json.load(f)
                self.hsv_lower = np.array([config["h_min"], config["s_min"], config["v_min"]])
                self.hsv_upper = np.array([config["h_max"], config["s_max"], config["v_max"]])
                print(f"Loaded HSV values: H:{config['h_min']}-{config['h_max']}, S:{config['s_min']}-{config['s_max']}, V:{config['v_min']}-{config['v_max']}")
        except:
            print("Using default HSV values")
            self.hsv_lower = np.array([39, 54, 35])
            self.hsv_upper = np.array([77, 163, 181])
    
    def apply_hsv_filter(self, frame):
        """Apply HSV filter to the entire frame."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply HSV filter
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def find_contours_in_mask(self, mask):
        """Find all contours in the HSV mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1 or area > 20:  # Very small tennis ball size: 1-20 pixels
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            detected_objects.append((cx, cy, area))
        
        return detected_objects
    
    def draw_debug_info(self, frame, mask, detected_objects):
        """Draw debug information on the frame."""
        result = frame.copy()
        
        # Draw all detected objects
        for i, (cx, cy, area) in enumerate(detected_objects):
            # Draw circle around detected object
            cv2.circle(result, (cx, cy), 20, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw label
            label = f"OBJ{i+1}: {area:.0f}px"
            cv2.putText(result, label, (cx-30, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw players
        cv2.circle(result, self.player1_pos, 20, (255, 0, 0), 2)
        cv2.putText(result, "P1", (self.player1_pos[0]-10, self.player1_pos[1]-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.circle(result, self.player2_pos, 20, (0, 0, 255), 2)
        cv2.putText(result, "P2", (self.player2_pos[0]-10, self.player2_pos[1]-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show HSV filter info
        white_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (white_pixels / total_pixels) * 100
        
        cv2.putText(result, f"HSV Filter: {white_pixels}/{total_pixels} pixels ({percentage:.2f}%)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Detected Objects: {len(detected_objects)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def create_full_screen_debug(self, frame, mask):
        """Create a full screen debug view showing original and HSV filter side by side."""
        height, width = frame.shape[:2]
        
        # Create side-by-side display
        debug_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Left side: original frame with debug info
        left_frame = self.draw_debug_info(frame, mask, self.find_contours_in_mask(mask))
        debug_frame[:height, :width] = left_frame
        
        # Right side: HSV filter mask (white on black)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug_frame[:height, width:] = mask_colored
        
        # Add labels
        cv2.putText(debug_frame, "ORIGINAL + DETECTIONS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(debug_frame, "HSV FILTER MASK", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return debug_frame
    
    def process_video(self):
        """Process video with full HSV filter debugging."""
        print("HSV Filter Debugger")
        print("=" * 50)
        print("This will show you exactly what the HSV filter detects")
        print("Left side: Original frame with detected objects marked")
        print("Right side: HSV filter mask (white = detected)")
        print("Press 'D' to advance frame, 'Q' to quit")
        print("=" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Resize frame to fit screen
            height, width = frame.shape[:2]
            scale = min(1920/width, 1080/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Apply HSV filter
            mask = self.apply_hsv_filter(resized_frame)
            
            # Find detected objects
            detected_objects = self.find_contours_in_mask(mask)
            
            # Create debug display
            debug_frame = self.create_full_screen_debug(resized_frame, mask)
            
            # Show frame info
            cv2.putText(debug_frame, f"Frame: {self.frame_count}", 
                       (10, new_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show detected objects info
            for i, (cx, cy, area) in enumerate(detected_objects):
                print(f"  Object {i+1}: ({cx}, {cy}) area={area:.0f}px")
            
            cv2.imshow("HSV Filter Debug - Full Screen", debug_frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Advance frame
                pass
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nDebug complete! Processed {self.frame_count} frames")


if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    debugger = HSVFilterDebugger(video_path)
    debugger.process_video()
