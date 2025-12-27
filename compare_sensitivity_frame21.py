import cv2
import json
import numpy as np

def load_zones():
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

def detect_in_zone(fgmask, zone_polygon, min_area, max_area, frame_shape):
    """Detect player in a specific zone"""
    # Create zone mask
    zone_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(zone_mask, [zone_polygon], 255)
    
    # Apply zone mask to foreground mask
    zone_fg = cv2.bitwise_and(fgmask, zone_mask)
    
    # Find contours
    contours, _ = cv2.findContours(zone_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid_contours.append((contour, area))
    
    if valid_contours:
        # Return largest valid contour
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        return valid_contours[0][0], valid_contours[0][1], contours
    
    return None, None, contours

def test_sensitivity(video_path, frame_num, sensitivity, p1_zone, p2_zone):
    """Test a specific sensitivity value on a specific frame"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    # Create MOG2 with specific sensitivity
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=sensitivity,
        detectShadows=True
    )
    mog2.setVarThreshold(sensitivity)
    
    # Learn background from frames 0 to frame_num
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_num + 1):
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = mog2.apply(frame, learningRate=0.005)
    
    # Get the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None, None, None
    
    # Apply MOG2
    fgmask = mog2.apply(frame, learningRate=-1)
    
    # Remove shadows
    fgmask[fgmask == 127] = 0
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Detect in zones
    p1_contour, p1_area, p1_all_contours = detect_in_zone(fgmask, p1_zone, 5000, 150000, frame.shape)
    p2_contour, p2_area, p2_all_contours = detect_in_zone(fgmask, p2_zone, 10000, 300000, frame.shape)
    
    # Draw results
    result_frame = frame.copy()
    
    # Draw zones
    cv2.polylines(result_frame, [p1_zone], True, (255, 255, 0), 2)
    cv2.polylines(result_frame, [p2_zone], True, (255, 255, 0), 2)
    
    # Draw all contours in zones
    for contour in p1_all_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        color = (0, 255, 0) if (5000 <= area <= 150000) else (0, 0, 255)
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_frame, f"{int(area)}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    for contour in p2_all_contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        color = (0, 255, 0) if (10000 <= area <= 300000) else (0, 0, 255)
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_frame, f"{int(area)}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add text info
    p1_text = f"P1: {int(p1_area)}px" if p1_area else "P1: NOT FOUND"
    p2_text = f"P2: {int(p2_area)}px" if p2_area else "P2: NOT FOUND"
    
    cv2.putText(result_frame, f"Sensitivity: {sensitivity}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result_frame, p1_text, (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if p1_area else (0, 0, 255), 2)
    cv2.putText(result_frame, p2_text, (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if p2_area else (0, 0, 255), 2)
    
    cap.release()
    return result_frame, fgmask, (p1_area, p2_area, len(p1_all_contours), len(p2_all_contours))

# Main
video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
frame_num = 21

print("=" * 70)
print(f"COMPARING SENSITIVITIES FOR FRAME {frame_num}")
print("=" * 70)

p1_zone, p2_zone = load_zones()

# Test three sensitivities
sensitivities = [15, 16, 17]
results = []

for sens in sensitivities:
    print(f"\nTesting sensitivity {sens}...")
    frame, mask, stats = test_sensitivity(video_path, frame_num, sens, p1_zone, p2_zone)
    if frame is not None:
        results.append((sens, frame, mask, stats))
        p1_area, p2_area, p1_count, p2_count = stats
        print(f"  Sensitivity {sens}:")
        print(f"    P1: {int(p1_area) if p1_area else 'NOT FOUND'} ({p1_count} contours)")
        print(f"    P2: {int(p2_area) if p2_area else 'NOT FOUND'} ({p2_count} contours)")

print("\n" + "=" * 70)
print("DISPLAYING RESULTS - Press any key to close")
print("=" * 70)

# Display all results - resize to fit screen
scale = 0.4  # Scale down to 40% of original size
for sens, frame, mask, stats in results:
    # Resize frame and mask
    height, width = frame.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    frame_resized = cv2.resize(frame, (new_width, new_height))
    mask_resized = cv2.resize(mask, (new_width, new_height))
    
    window_name = f"Sensitivity {sens}"
    cv2.imshow(window_name, frame_resized)
    cv2.imshow(f"{window_name} - Mask", mask_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

