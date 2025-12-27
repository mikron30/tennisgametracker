"""
Interactive HSV tuner for P1 and P2 separately.
Adjust HSV values and area sizes for each player independently.
"""

import cv2
import numpy as np
import json


def nothing(x):
    """Dummy callback for trackbars."""
    pass


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Go to frame 50 (where both players are visible)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot read frame")
        return
    
    print("\n" + "="*70)
    print("P1 and P2 HSV TUNER")
    print("="*70)
    print("\nAdjust trackbars to detect P1 (top-left) and P2 (bottom-right)")
    print("\nP1 TRACKBARS (left window):")
    print("  - Adjust H/S/V Min/Max to isolate P1's clothing color")
    print("  - Adjust Min/Max Area to set P1's size range")
    print("\nP2 TRACKBARS (right window):")
    print("  - Adjust H/S/V Min/Max to isolate P2's clothing color")
    print("  - Adjust Min/Max Area to set P2's size range")
    print("\nPress 'Q' to save settings and quit")
    print("="*70)
    
    # Create windows
    cv2.namedWindow("P1 HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.namedWindow("P2 HSV Tuner", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("P1 HSV Tuner", 700, 600)
    cv2.resizeWindow("P2 HSV Tuner", 700, 600)
    cv2.resizeWindow("Combined View", 1400, 900)
    
    # Wait for windows
    cv2.waitKey(100)
    
    # Create trackbars for P1 (gray/white clothing - person in top left)
    cv2.createTrackbar("P1 H Min", "P1 HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("P1 H Max", "P1 HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("P1 S Min", "P1 HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("P1 S Max", "P1 HSV Tuner", 50, 255, nothing)
    cv2.createTrackbar("P1 V Min", "P1 HSV Tuner", 50, 255, nothing)
    cv2.createTrackbar("P1 V Max", "P1 HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("P1 Min Area /100", "P1 HSV Tuner", 5, 100, nothing)
    cv2.createTrackbar("P1 Max Area /100", "P1 HSV Tuner", 500, 1000, nothing)
    
    # Create trackbars for P2 (gray/white clothing - person in bottom right)
    cv2.createTrackbar("P2 H Min", "P2 HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("P2 H Max", "P2 HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("P2 S Min", "P2 HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("P2 S Max", "P2 HSV Tuner", 50, 255, nothing)
    cv2.createTrackbar("P2 V Min", "P2 HSV Tuner", 50, 255, nothing)
    cv2.createTrackbar("P2 V Max", "P2 HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("P2 Min Area /100", "P2 HSV Tuner", 5, 100, nothing)
    cv2.createTrackbar("P2 Max Area /100", "P2 HSV Tuner", 300, 1000, nothing)
    
    # Define search zones to help guide detection
    # P1 zone: top-left area (where you circled in red)
    p1_search_zone = np.array([
        (900, 0),      # Top-left
        (1600, 0),     # Top-right
        (1600, 400),   # Bottom-right
        (900, 400)     # Bottom-left
    ], dtype=np.int32)
    
    # P2 zone: bottom-right area (where you circled in yellow)
    p2_search_zone = np.array([
        (2400, 800),   # Top-left
        (3200, 800),   # Top-right
        (3200, 1400),  # Bottom-right
        (2400, 1400)   # Bottom-left
    ], dtype=np.int32)
    
    while True:
        # Get P1 trackbar values
        p1_h_min = cv2.getTrackbarPos("P1 H Min", "P1 HSV Tuner")
        p1_h_max = cv2.getTrackbarPos("P1 H Max", "P1 HSV Tuner")
        p1_s_min = cv2.getTrackbarPos("P1 S Min", "P1 HSV Tuner")
        p1_s_max = cv2.getTrackbarPos("P1 S Max", "P1 HSV Tuner")
        p1_v_min = cv2.getTrackbarPos("P1 V Min", "P1 HSV Tuner")
        p1_v_max = cv2.getTrackbarPos("P1 V Max", "P1 HSV Tuner")
        p1_min_area = cv2.getTrackbarPos("P1 Min Area /100", "P1 HSV Tuner") * 100
        p1_max_area = cv2.getTrackbarPos("P1 Max Area /100", "P1 HSV Tuner") * 100
        
        # Get P2 trackbar values
        p2_h_min = cv2.getTrackbarPos("P2 H Min", "P2 HSV Tuner")
        p2_h_max = cv2.getTrackbarPos("P2 H Max", "P2 HSV Tuner")
        p2_s_min = cv2.getTrackbarPos("P2 S Min", "P2 HSV Tuner")
        p2_s_max = cv2.getTrackbarPos("P2 S Max", "P2 HSV Tuner")
        p2_v_min = cv2.getTrackbarPos("P2 V Min", "P2 HSV Tuner")
        p2_v_max = cv2.getTrackbarPos("P2 V Max", "P2 HSV Tuner")
        p2_min_area = cv2.getTrackbarPos("P2 Min Area /100", "P2 HSV Tuner") * 100
        p2_max_area = cv2.getTrackbarPos("P2 Max Area /100", "P2 HSV Tuner") * 100
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # === DETECT P1 ===
        p1_lower = np.array([p1_h_min, p1_s_min, p1_v_min], dtype=np.uint8)
        p1_upper = np.array([p1_h_max, p1_s_max, p1_v_max], dtype=np.uint8)
        
        # Create P1 zone mask
        p1_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p1_zone_mask, [p1_search_zone], 255)
        
        # Apply P1 HSV filter
        p1_mask = cv2.inRange(hsv, p1_lower, p1_upper)
        p1_mask = cv2.bitwise_and(p1_mask, p1_zone_mask)
        
        # Clean up P1 mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        p1_mask = cv2.morphologyEx(p1_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        p1_mask = cv2.morphologyEx(p1_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find P1 contours
        p1_contours, _ = cv2.findContours(p1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        p1_detected = []
        for contour in p1_contours:
            area = cv2.contourArea(contour)
            if p1_min_area <= area <= p1_max_area:
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    p1_detected.append({'bbox': (x, y, w, h), 'center': (cx, cy), 'area': area})
        
        # === DETECT P2 ===
        p2_lower = np.array([p2_h_min, p2_s_min, p2_v_min], dtype=np.uint8)
        p2_upper = np.array([p2_h_max, p2_s_max, p2_v_max], dtype=np.uint8)
        
        # Create P2 zone mask
        p2_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(p2_zone_mask, [p2_search_zone], 255)
        
        # Apply P2 HSV filter
        p2_mask = cv2.inRange(hsv, p2_lower, p2_upper)
        p2_mask = cv2.bitwise_and(p2_mask, p2_zone_mask)
        
        # Clean up P2 mask
        p2_mask = cv2.morphologyEx(p2_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        p2_mask = cv2.morphologyEx(p2_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find P2 contours
        p2_contours, _ = cv2.findContours(p2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        p2_detected = []
        for contour in p2_contours:
            area = cv2.contourArea(contour)
            if p2_min_area <= area <= p2_max_area:
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    p2_detected.append({'bbox': (x, y, w, h), 'center': (cx, cy), 'area': area})
        
        # === VISUALIZE ===
        # P1 view
        p1_display = frame.copy()
        overlay = p1_display.copy()
        cv2.fillPoly(overlay, [p1_search_zone], (255, 100, 100))
        p1_display = cv2.addWeighted(p1_display, 0.7, overlay, 0.3, 0)
        cv2.polylines(p1_display, [p1_search_zone], True, (0, 0, 255), 3)
        
        for p1 in p1_detected:
            x, y, w, h = p1['bbox']
            cv2.rectangle(p1_display, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.circle(p1_display, p1['center'], 10, (255, 0, 0), -1)
            cv2.putText(p1_display, f"P1 {p1['area']:.0f}px", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        p1_mask_bgr = cv2.cvtColor(p1_mask, cv2.COLOR_GRAY2BGR)
        p1_combined = np.hstack([p1_display, p1_mask_bgr])
        
        # P2 view
        p2_display = frame.copy()
        overlay = p2_display.copy()
        cv2.fillPoly(overlay, [p2_search_zone], (100, 200, 255))
        p2_display = cv2.addWeighted(p2_display, 0.7, overlay, 0.3, 0)
        cv2.polylines(p2_display, [p2_search_zone], True, (0, 165, 255), 3)
        
        for p2 in p2_detected:
            x, y, w, h = p2['bbox']
            cv2.rectangle(p2_display, (x, y), (x + w, y + h), (0, 165, 255), 4)
            cv2.circle(p2_display, p2['center'], 10, (0, 165, 255), -1)
            cv2.putText(p2_display, f"P2 {p2['area']:.0f}px", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        p2_mask_bgr = cv2.cvtColor(p2_mask, cv2.COLOR_GRAY2BGR)
        p2_combined = np.hstack([p2_display, p2_mask_bgr])
        
        # Combined view with both players
        combined_display = frame.copy()
        
        # Draw zones
        overlay = combined_display.copy()
        cv2.fillPoly(overlay, [p1_search_zone], (255, 100, 100))
        cv2.fillPoly(overlay, [p2_search_zone], (100, 200, 255))
        combined_display = cv2.addWeighted(combined_display, 0.85, overlay, 0.15, 0)
        
        # Draw P1
        for p1 in p1_detected:
            x, y, w, h = p1['bbox']
            cv2.rectangle(combined_display, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.circle(combined_display, p1['center'], 12, (255, 0, 0), -1)
            label = f"P1 ({p1['area']:.0f}px)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(combined_display, (x, y - lh - 20), (x + lw + 10, y - 5), (255, 0, 0), -1)
            cv2.putText(combined_display, label, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw P2
        for p2 in p2_detected:
            x, y, w, h = p2['bbox']
            cv2.rectangle(combined_display, (x, y), (x + w, y + h), (0, 165, 255), 5)
            cv2.circle(combined_display, p2['center'], 12, (0, 165, 255), -1)
            label = f"P2 ({p2['area']:.0f}px)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(combined_display, (x, y - lh - 20), (x + lw + 10, y - 5), (0, 165, 255), -1)
            cv2.putText(combined_display, label, (x + 5, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add status
        status = f"P1 Detected: {len(p1_detected)} | P2 Detected: {len(p2_detected)} | Press 'Q' to save"
        cv2.putText(combined_display, status, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Resize and show
        scale = 0.5
        p1_resized = cv2.resize(p1_combined, None, fx=scale, fy=scale)
        p2_resized = cv2.resize(p2_combined, None, fx=scale, fy=scale)
        combined_resized = cv2.resize(combined_display, None, fx=scale, fy=scale)
        
        cv2.imshow("P1 HSV Tuner", p1_resized)
        cv2.imshow("P2 HSV Tuner", p2_resized)
        cv2.imshow("Combined View", combined_resized)
        
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
    
    # Save settings
    settings = {
        'p1_hsv': {
            'h_min': int(p1_h_min),
            'h_max': int(p1_h_max),
            's_min': int(p1_s_min),
            's_max': int(p1_s_max),
            'v_min': int(p1_v_min),
            'v_max': int(p1_v_max),
            'min_area': int(p1_min_area),
            'max_area': int(p1_max_area)
        },
        'p2_hsv': {
            'h_min': int(p2_h_min),
            'h_max': int(p2_h_max),
            's_min': int(p2_s_min),
            's_max': int(p2_s_max),
            'v_min': int(p2_v_min),
            'v_max': int(p2_v_max),
            'min_area': int(p2_min_area),
            'max_area': int(p2_max_area)
        }
    }
    
    with open('player_hsv_config.json', 'w') as f:
        json.dump(settings, f, indent=2)
    
    print("\n" + "="*70)
    print("FINAL SETTINGS SAVED TO player_hsv_config.json")
    print("="*70)
    print("\nP1 Settings:")
    print(f"  HSV: H={p1_h_min}-{p1_h_max}, S={p1_s_min}-{p1_s_max}, V={p1_v_min}-{p1_v_max}")
    print(f"  Area: {p1_min_area}-{p1_max_area} pixels")
    print(f"  Detected: {len(p1_detected)} instance(s)")
    print("\nP2 Settings:")
    print(f"  HSV: H={p2_h_min}-{p2_h_max}, S={p2_s_min}-{p2_s_max}, V={p2_v_min}-{p2_v_max}")
    print(f"  Area: {p2_min_area}-{p2_max_area} pixels")
    print(f"  Detected: {len(p2_detected)} instance(s)")
    print("="*70)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



