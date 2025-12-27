#!/usr/bin/env python3
"""
Find the exact marked points (purple circles with green numbers 1-12) on the tennis court image.
"""

import cv2
import numpy as np


def find_purple_circles_with_green_numbers(image_path: str):
    """
    Find purple circles with green numbers 1-12 on the tennis court image.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define purple color range (BGR: 255, 0, 255 -> HSV)
    # Purple in HSV: around 150-170 hue
    lower_purple = np.array([140, 50, 50])
    upper_purple = np.array([170, 255, 255])
    
    # Define green color range for numbers
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Create masks
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up purple mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, kernel)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of purple circles
    purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contours of green numbers
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(purple_contours)} purple contours")
    print(f"Found {len(green_contours)} green contours")
    
    # Create result image
    result_img = img.copy()
    
    # Store found points
    found_points = []
    
    # Process purple circles
    for i, contour in enumerate(purple_contours):
        # Check if contour is roughly circular and right size
        area = cv2.contourArea(contour)
        if area > 50 and area < 1000:  # Reasonable size for a circle
            # Get center of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw circle around detected point
                cv2.circle(result_img, (cx, cy), 15, (0, 255, 0), 2)
                
                # Try to find nearby green number
                number = find_nearby_green_number(cx, cy, green_contours, result_img)
                
                found_points.append((cx, cy, number))
                
                # Draw coordinates
                cv2.putText(result_img, f"({cx},{cy})", (cx + 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if number:
                    cv2.putText(result_img, f"#{number}", (cx + 20, cy + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Sort points by number
    found_points.sort(key=lambda x: x[2] if x[2] is not None else 999)
    
    # Save result
    cv2.imwrite('detected_points.jpg', result_img)
    cv2.imwrite('purple_mask.jpg', purple_mask)
    cv2.imwrite('green_mask.jpg', green_mask)
    
    print("\nDetected points:")
    for x, y, number in found_points:
        if number:
            print(f"Point {number}: ({x}, {y})")
        else:
            print(f"Point without number: ({x}, {y})")
    
    print(f"\nTotal found points: {len(found_points)}")
    print("Images saved:")
    print("- detected_points.jpg (with detected circles)")
    print("- purple_mask.jpg (purple color mask)")
    print("- green_mask.jpg (green color mask)")
    
    return found_points


def find_nearby_green_number(cx: int, cy: int, green_contours: list, img: np.ndarray) -> int:
    """
    Try to find a green number near the purple circle.
    """
    # Look for green contours near the purple circle
    for contour in green_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            gx = int(M["m10"] / M["m00"])
            gy = int(M["m01"] / M["m00"])
            
            # Check if green contour is near purple circle
            distance = np.sqrt((cx - gx)**2 + (cy - gy)**2)
            if distance < 50:  # Within 50 pixels
                # This is a rough approximation - we'd need OCR for exact numbers
                return None  # For now, just return None
    
    return None


def manual_point_detection(image_path: str):
    """
    Alternative method: manually click on points to get coordinates.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(img, str(len(points)), (x + 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Click Points', img)
    
    cv2.namedWindow('Click Points')
    cv2.setMouseCallback('Click Points', mouse_callback)
    
    cv2.imshow('Click Points', img)
    print("Click on the 12 marked points in order (1-12)")
    print("Press 'q' when done")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if len(points) == 12:
        print("\nAll 12 points collected:")
        for i, (x, y) in enumerate(points, 1):
            print(f"Point {i}: ({x}, {y})")
        
        # Save the marked image
        cv2.imwrite('manually_marked_points.jpg', img)
        print("Saved manually_marked_points.jpg")
        
        return points
    else:
        print(f"Only {len(points)} points collected. Need 12 points.")
        return None


def main():
    """
    Main function to find marked points.
    """
    image_path = 'original_right_court.jpg'
    
    print("Finding marked points on tennis court...")
    print("Method 1: Automatic detection of purple circles with green numbers")
    
    # Try automatic detection first
    found_points = find_purple_circles_with_green_numbers(image_path)
    
    if len(found_points) < 12:
        print(f"\nOnly found {len(found_points)} points automatically.")
        print("Method 2: Manual clicking on points")
        manual_points = manual_point_detection(image_path)
        
        if manual_points:
            found_points = manual_points
    
    return found_points


if __name__ == "__main__":
    main()








