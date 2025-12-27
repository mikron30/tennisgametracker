"""
Visualize the marked court lines and create player detection zones.
"""

import cv2
import numpy as np


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot read frame")
        return
    
    # Court points (full frame coordinates from adjusted_court_points.txt)
    court_points = {
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
    
    # Draw court lines on frame
    display = frame.copy()
    
    # Define court line connections (which points connect to form lines)
    lines = [
        # Baselines
        (1, 13, (0, 255, 0), "Top Baseline"),
        (13, 2, (0, 255, 0), None),
        (3, 14, (0, 255, 0), "Bottom Baseline"),
        (14, 4, (0, 255, 0), None),
        
        # Sidelines (following court geometry through service line points)
        # Left sideline: 1 -> 5 -> 7 -> 3
        (1, 5, (255, 0, 0), "Left Sideline"),
        (5, 7, (255, 0, 0), None),
        (7, 3, (255, 0, 0), None),
        # Right sideline: 2 -> 6 -> 8 -> 4
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
        p1 = court_points[p1_idx]
        p2 = court_points[p2_idx]
        cv2.line(display, p1, p2, color, 3)
        
        # Draw label at midpoint
        if label:
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(display, label, (mid_x + 10, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw points with numbers
    for idx, (x, y) in court_points.items():
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(display, str(idx), (x + 15, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Now define PLAYER ZONES based on court geometry
    # P1 zone: Near side (between baseline and service line, top half)
    # P2 zone: Far side (between service line and net, top half)
    
    # P1 ZONE (between top baseline and top service line)
    p1_zone = np.array([
        court_points[1],   # Top-left baseline
        court_points[13],  # Top singles sideline
        court_points[6],   # Service line right (top)
        court_points[5],   # Service line left (top)
    ], dtype=np.int32)
    
    # P2 ZONE (between top service line and net)
    p2_zone = np.array([
        court_points[5],   # Service line left (top)
        court_points[6],   # Service line right (top)
        court_points[10],  # Net right
        court_points[9],   # Net left
    ], dtype=np.int32)
    
    # Draw player zones with transparency
    overlay = display.copy()
    cv2.fillPoly(overlay, [p1_zone], (255, 100, 100))  # Blue zone for P1
    cv2.fillPoly(overlay, [p2_zone], (100, 100, 255))  # Red zone for P2
    display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
    
    # Add labels
    cv2.putText(display, "P1 ZONE", (1200, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(display, "P2 ZONE", (1300, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add instructions
    info_text = [
        "Court Lines and Player Zones",
        "P1 Zone = Blue (near baseline)",
        "P2 Zone = Red (near net)",
        "Green = Baselines | Blue = Sidelines",
        "Yellow = Net | Cyan = Service Lines",
        "Press any key to close..."
    ]
    
    y_offset = 50
    for text in info_text:
        cv2.putText(display, text, (50, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 40
    
    # Resize for display
    height, width = display.shape[:2]
    scale = min(1920 / width, 1080 / height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    display_resized = cv2.resize(display, (new_w, new_h))
    
    cv2.namedWindow("Court Lines and Player Zones", cv2.WINDOW_NORMAL)
    cv2.imshow("Court Lines and Player Zones", display_resized)
    
    print("\n" + "="*70)
    print("COURT LINES AND PLAYER ZONES")
    print("="*70)
    print("\nCourt points loaded from adjusted_court_points.txt")
    print(f"Total points: {len(court_points)}")
    print("\nPlayer zones defined:")
    print("  P1 Zone: Between baseline and service line (near side)")
    print("  P2 Zone: Between service line and net (far side)")
    print("\nPress any key to close...")
    print("="*70)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    
    # Save zone coordinates for later use
    print("\n" + "="*70)
    print("ZONE COORDINATES (for player detection):")
    print("="*70)
    print("P1 Zone polygon:", p1_zone.tolist())
    print("P2 Zone polygon:", p2_zone.tolist())
    print("="*70)


if __name__ == "__main__":
    main()

