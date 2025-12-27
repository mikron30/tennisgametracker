#!/usr/bin/env python3
"""
Test script to extract and visualize tennis court lines from a single frame.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_court_lines_from_frame(video_path: str, frame_number: int = 0):
    """
    Extract tennis court lines from a specific frame and show the result.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract (default: 0 - first frame)
    """
    # Open video and get the specified frame
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Get right court region (focus on right court only)
    height, width = frame.shape[:2]
    right_start_x = int(width * 0.15)  # Start from 15% from left
    right_court = frame[:, right_start_x:width]
    
    print(f"Right court region: {right_court.shape[1]}x{right_court.shape[0]}")
    
    # Convert to different color spaces for better line detection
    gray = cv2.cvtColor(right_court, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(right_court, cv2.COLOR_BGR2HSV)
    
    # Method 1: Threshold for white lines
    # White lines should have high brightness and low saturation
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    
    # Method 2: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Method 3: Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    
    # Method 4: Otsu threshold
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create a combined mask for white lines
    # Combine different methods
    combined_mask = cv2.bitwise_or(white_mask, edges)
    combined_mask = cv2.bitwise_or(combined_mask, adaptive_thresh)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Create black and white image showing only white lines
    black_white_result = np.zeros_like(gray)
    black_white_result[combined_mask > 0] = 255
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(right_court, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Right Court')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # White mask
    axes[0, 2].imshow(white_mask, cmap='gray')
    axes[0, 2].set_title('White Lines Mask')
    axes[0, 2].axis('off')
    
    # Edge detection
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Edge Detection')
    axes[1, 0].axis('off')
    
    # Combined mask
    axes[1, 1].imshow(combined_mask, cmap='gray')
    axes[1, 1].set_title('Combined Mask')
    axes[1, 1].axis('off')
    
    # Final black and white result
    axes[1, 2].imshow(black_white_result, cmap='gray')
    axes[1, 2].set_title('Final: Only White Lines')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Also save the result
    cv2.imwrite('court_lines_black_white.jpg', black_white_result)
    cv2.imwrite('original_frame.jpg', right_court)
    
    print("Images saved:")
    print("- court_lines_black_white.jpg (black and white with only white lines)")
    print("- original_frame.jpg (original right court)")
    
    cap.release()


def main():
    """
    Main function to test court line extraction.
    """
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    print("Extracting tennis court lines from first frame...")
    print("This will show you different methods of line detection.")
    print("The final result will be black and white showing only the white court lines.")
    
    extract_court_lines_from_frame(video_path, 0)


if __name__ == "__main__":
    main()
