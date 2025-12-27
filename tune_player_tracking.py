"""
Interactive tuner for MOG2 player tracking parameters.
Adjust parameters in real-time to find optimal settings for detecting both P1 and P2.
"""

import cv2
import numpy as np


def nothing(x):
    """Dummy callback for trackbars."""
    pass


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\n=== MOG2 Player Tracking Parameter Tuner ===")
    print("Adjust the trackbars to tune player detection")
    print("\nControls:")
    print("  Use trackbars to adjust parameters")
    print("  Drag 'Frame' slider to jump to different frames")
    print("  'R' = Reset to current frame (rebuild background)")
    print("  'Q' = Quit and save settings")
    print(f"\nTotal frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Create windows
    cv2.namedWindow("Tuner Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Player Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Foreground Mask", cv2.WINDOW_NORMAL)
    
    # Wait for windows to be created
    cv2.waitKey(100)
    
    cv2.resizeWindow("Player Detection", 1400, 800)
    cv2.resizeWindow("Foreground Mask", 700, 400)
    cv2.resizeWindow("Tuner Controls", 600, 400)
    
    # Create trackbars
    cv2.createTrackbar("Frame", "Tuner Controls", 50, total_frames - 1, nothing)
    cv2.createTrackbar("Var Threshold", "Tuner Controls", 8, 50, nothing)
    cv2.createTrackbar("Learning Rate x1000", "Tuner Controls", 1, 50, nothing)  # 0.001 to 0.050
    cv2.createTrackbar("History", "Tuner Controls", 200, 1000, nothing)
    cv2.createTrackbar("Min Area / 100", "Tuner Controls", 15, 100, nothing)  # 1500 pixels
    cv2.createTrackbar("Max Area / 100", "Tuner Controls", 800, 1000, nothing)  # 80000 pixels
    cv2.createTrackbar("Morphology Open", "Tuner Controls", 2, 10, nothing)
    cv2.createTrackbar("Morphology Close", "Tuner Controls", 3, 10, nothing)
    cv2.createTrackbar("Dilate", "Tuner Controls", 2, 10, nothing)
    cv2.createTrackbar("Detect Shadows", "Tuner Controls", 1, 1, nothing)  # 0 or 1
    
    right_court_x_min = 1920
    last_frame_number = -1
    bg_subtractor = None
    
    while True:
        # Get trackbar values
        frame_number = cv2.getTrackbarPos("Frame", "Tuner Controls")
        var_threshold = cv2.getTrackbarPos("Var Threshold", "Tuner Controls")
        learning_rate = cv2.getTrackbarPos("Learning Rate x1000", "Tuner Controls") / 1000.0
        history = cv2.getTrackbarPos("History", "Tuner Controls")
        min_area = cv2.getTrackbarPos("Min Area / 100", "Tuner Controls") * 100
        max_area = cv2.getTrackbarPos("Max Area / 100", "Tuner Controls") * 100
        morph_open = cv2.getTrackbarPos("Morphology Open", "Tuner Controls")
        morph_close = cv2.getTrackbarPos("Morphology Close", "Tuner Controls")
        dilate_iter = cv2.getTrackbarPos("Dilate", "Tuner Controls")
        detect_shadows = cv2.getTrackbarPos("Detect Shadows", "Tuner Controls") == 1
        
        # Ensure min values
        if var_threshold < 1:
            var_threshold = 1
        if history < 10:
            history = 10
        if min_area < 100:
            min_area = 100
        if max_area < 1000:
            max_area = 1000
        
        # If frame changed or background subtractor not created, reset it
        if frame_number != last_frame_number or bg_subtractor is None:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 20))  # Start 20 frames before to build background
            
            # Create new background subtractor
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
            
            # Build background model by processing previous frames
            for i in range(20):
                ret, temp_frame = cap.read()
                if ret:
                    temp_right_court = temp_frame[:, right_court_x_min:]
                    bg_subtractor.apply(temp_right_court, learningRate=0.1)  # Fast learning for initial model
            
            last_frame_number = frame_number
        
        # Read current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            print("Cannot read frame")
            break
        
        # Extract right court
        right_court = frame[:, right_court_x_min:]
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(right_court, learningRate=learning_rate)
        
        # Remove shadows
        if detect_shadows:
            fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations
        if morph_open > 0:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=morph_open)
        
        if morph_close > 0:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=morph_close)
        
        if dilate_iter > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=dilate_iter)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find players
        players = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                if 0.8 <= aspect_ratio <= 4.0:
                    x_full = x + right_court_x_min
                    players.append((x_full, y, w, h, area))
        
        # Sort by Y coordinate
        players.sort(key=lambda p: p[1])
        
        # Draw results
        display = frame.copy()
        
        for i, (x, y, w, h, area) in enumerate(players[:2]):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)
            label = f"P{i+1} ({area:.0f}px)"
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 5)
            cv2.putText(display, label, (x, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw center
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(display, (center_x, center_y), 10, color, -1)
        
        # Add info text
        info_lines = [
            f"Frame: {frame_number}/{total_frames}",
            f"Players detected: {len(players[:2])}",
            f"Var Threshold: {var_threshold}",
            f"Learning Rate: {learning_rate:.4f}",
            f"Area: {min_area}-{max_area}px"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 40 + i * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Show foreground mask
        mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Resize for display
        height, width = frame.shape[:2]
        scale = min(1400 / width, 800 / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        
        display_resized = cv2.resize(display, (new_w, new_h))
        mask_resized = cv2.resize(mask_colored, (new_w // 2, new_h // 2))
        
        cv2.imshow("Player Detection", display_resized)
        cv2.imshow("Foreground Mask", mask_resized)
        
        # Handle keys
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            print("\n=== Final Settings ===")
            print(f"varThreshold = {var_threshold}")
            print(f"learning_rate = {learning_rate:.4f}")
            print(f"history = {history}")
            print(f"min_player_area = {min_area}")
            print(f"max_player_area = {max_area}")
            print(f"morph_open_iterations = {morph_open}")
            print(f"morph_close_iterations = {morph_close}")
            print(f"dilate_iterations = {dilate_iter}")
            print(f"detectShadows = {detect_shadows}")
            break
        elif key == ord('r'):
            # Reset background model
            last_frame_number = -1
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

