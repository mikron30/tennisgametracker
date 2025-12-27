#!/usr/bin/env python3
"""
Interactive script to click on the 12 marked points on the tennis court image.
"""

import cv2
import numpy as np


def resize_to_fit_screen(frame: np.ndarray, max_width: int = 1200, max_height: int = 800) -> np.ndarray:
    """
    Resize frame to fit within specified screen dimensions while maintaining aspect ratio.
    """
    height, width = frame.shape[:2]
    
    scale_h = max_height / height
    scale_w = max_width / width
    scale = min(scale_h, scale_w, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame


def click_points_interactive():
    """
    Interactive function to click on the 12 marked points.
    """
    # Load the original image
    img = cv2.imread('original_right_court.jpg')
    if img is None:
        print("Error: Could not load original_right_court.jpg")
        return
    
    print(f"Original image size: {img.shape[1]}x{img.shape[0]}")
    
    # Store original image for coordinate conversion
    original_img = img.copy()
    original_height, original_width = img.shape[:2]
    
    # Resize for display
    display_img = resize_to_fit_screen(img)
    display_height, display_width = display_img.shape[:2]
    
    # Calculate scaling factors
    scale_x = original_width / display_width
    scale_y = original_height / display_height
    
    print(f"Display image size: {display_width}x{display_height}")
    print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
    
    # Store clicked points
    clicked_points = []
    current_point = 1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_point
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_point <= 12:
                # Convert display coordinates to original image coordinates
                original_x = int(x * scale_x)
                original_y = int(y * scale_y)
                
                clicked_points.append((original_x, original_y))
                
                # Draw point on display image
                cv2.circle(display_img, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(display_img, str(current_point), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Draw point on original image too
                cv2.circle(original_img, (original_x, original_y), 10, (0, 255, 0), -1)
                cv2.putText(original_img, str(current_point), (original_x + 15, original_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                print(f"Point {current_point}: Display({x}, {y}) -> Original({original_x}, {original_y})")
                current_point += 1
                
                cv2.imshow('Click Points (1-12)', display_img)
    
    # Create window and set mouse callback
    cv2.namedWindow('Click Points (1-12)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Click Points (1-12)', mouse_callback)
    
    # Show instructions
    cv2.putText(display_img, "Click on points 1-12 in order", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_img, "Press 'r' to reset, 'q' to quit", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Click Points (1-12)', display_img)
    
    print("\nInstructions:")
    print("1. Click on the 12 marked points in order (1, 2, 3, ..., 12)")
    print("2. Press 'r' to reset and start over")
    print("3. Press 'q' to quit when done")
    print("4. The image is resized to fit your screen")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            clicked_points = []
            current_point = 1
            display_img = resize_to_fit_screen(cv2.imread('original_right_court.jpg'))
            original_img = cv2.imread('original_right_court.jpg').copy()
            
            # Redraw instructions
            cv2.putText(display_img, "Click on points 1-12 in order", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_img, "Press 'r' to reset, 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Click Points (1-12)', display_img)
            print("Reset! Click on points 1-12 again.")
    
    cv2.destroyAllWindows()
    
    if len(clicked_points) == 12:
        print("\n" + "="*50)
        print("All 12 points collected successfully!")
        print("="*50)
        
        for i, (x, y) in enumerate(clicked_points, 1):
            print(f"Point {i:2d}: ({x:4d}, {y:4d})")
        
        # Save the marked original image
        cv2.imwrite('clicked_points_result.jpg', original_img)
        print("\nSaved clicked_points_result.jpg with all marked points")
        
        # Save coordinates to file
        with open('court_points_coordinates.txt', 'w') as f:
            f.write("Tennis Court Points Coordinates\n")
            f.write("="*40 + "\n")
            f.write(f"Image size: {original_width}x{original_height}\n\n")
            
            for i, (x, y) in enumerate(clicked_points, 1):
                f.write(f"Point {i:2d}: ({x:4d}, {y:4d})\n")
        
        print("Saved coordinates to court_points_coordinates.txt")
        
        return clicked_points
    else:
        print(f"\nOnly {len(clicked_points)} points collected. Need all 12 points.")
        return None


def main():
    """
    Main function to run interactive point clicking.
    """
    print("Interactive Tennis Court Point Selection")
    print("="*50)
    
    points = click_points_interactive()
    
    if points:
        print("\nSuccess! You can now use these coordinates for precise court line detection.")
    else:
        print("\nFailed to collect all 12 points. Please try again.")


if __name__ == "__main__":
    main()








