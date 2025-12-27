import cv2
import numpy as np
from tennis_player_detector import TennisPlayerDetector

def main():
    """Debug player detection to see why it's failing."""
    
    # Open video
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Go to frame 7 (where we saw the issue)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Initialize player detector
    detector = TennisPlayerDetector()
    
    # Get right court ROI
    right_court, x_offset = detector.get_right_court_roi(frame)
    print(f"Right court starts at X={x_offset}, size: {right_court.shape}")
    
    # Create court mask
    court_mask = detector.create_court_mask(frame)
    print(f"Court mask created: {court_mask.shape}")
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Skin detection
    lower_skin = np.array([0, 30, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    print(f"Skin mask pixels: {np.sum(skin_mask > 0)}")
    
    # Blue clothing
    lower_blue = np.array([100, 50, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    print(f"Blue mask pixels: {np.sum(blue_mask > 0)}")
    
    # Light clothing (white, grey)
    lower_light = np.array([0, 0, 180], dtype=np.uint8)
    upper_light = np.array([180, 30, 255], dtype=np.uint8)
    light_mask = cv2.inRange(hsv, lower_light, upper_light)
    print(f"Light mask pixels: {np.sum(light_mask > 0)}")
    
    # Combine masks
    combined_mask = cv2.bitwise_or(skin_mask, blue_mask)
    combined_mask = cv2.bitwise_or(combined_mask, light_mask)
    print(f"Combined mask pixels: {np.sum(combined_mask > 0)}")
    
    # Apply court mask
    final_mask = cv2.bitwise_and(combined_mask, court_mask)
    print(f"Final mask pixels (after court mask): {np.sum(final_mask > 0)}")
    
    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\nFound {len(contours)} contours")
    
    # Analyze contours
    valid_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        
        # Check against player detector thresholds
        min_area = detector.min_person_area
        max_area = detector.max_person_area
        
        status = "VALID" if (min_area <= area <= max_area and 1.2 <= aspect_ratio <= 3.0) else "REJECTED"
        
        if area > 100:  # Only show significant contours
            print(f"  Contour {i}: area={area:.0f}px, pos=({x},{y}), size=({w}x{h}), aspect={aspect_ratio:.2f} - {status}")
            if status == "VALID":
                valid_contours.append((x, y, w, h))
    
    print(f"\nPlayer area thresholds: {detector.min_person_area} - {detector.max_person_area}")
    print(f"Valid player contours: {len(valid_contours)}")
    
    # Now run the actual detector
    print("\n=== Running actual player detector ===")
    players = detector.detect_players_advanced(frame)
    print(f"Detected {len(players)} players: {players}")
    
    # Visualize results
    display_frame = frame.copy()
    
    # Draw detected players
    for i, (x, y, w, h) in enumerate(players):
        color = (255, 0, 0) if i == 0 else (0, 0, 255)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
        label = "P1" if i == 0 else "P2"
        cv2.putText(display_frame, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Resize for display
    height, width = frame.shape[:2]
    scale_factor = min(1920/width, 1080/height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    display_resized = cv2.resize(display_frame, (new_width, new_height))
    court_mask_bgr = cv2.cvtColor(court_mask, cv2.COLOR_GRAY2BGR)
    court_mask_resized = cv2.resize(court_mask_bgr, (new_width, new_height))
    final_mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    final_mask_resized = cv2.resize(final_mask_bgr, (new_width, new_height))
    
    # Show results
    cv2.imshow("Original Frame with Detections", display_resized)
    cv2.imshow("Court Mask", court_mask_resized)
    cv2.imshow("Final Person Detection Mask", final_mask_resized)
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()

