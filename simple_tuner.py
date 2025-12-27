"""
SIMPLE Player Tracking Tuner
Focus on the 3 most important parameters to detect both P1 and P2.
"""

import cv2
import numpy as np


def main():
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\n" + "="*70)
    print("SIMPLE PLAYER TRACKING TUNER")
    print("="*70)
    print("\nGOAL: Adjust sliders until BOTH P1 and P2 are detected")
    print("\nMAIN CONTROLS (adjust these to find both players):")
    print("  1. 'Sensitivity' - Lower = detects slower/still players")
    print("                     Start at 8, try going DOWN to 4-6 if P2 missing")
    print("  2. 'Min Size'    - Minimum player size in pixels")
    print("                     Start at 10 (=1000px), try 5 (=500px) if P2 missing")
    print("  3. 'Frame'       - Jump to different parts of the video")
    print("\nOTHER CONTROLS:")
    print("  SPACE = Pause/Play")
    print("  'Q' = Quit and save settings")
    print("="*70)
    
    # Create main window
    cv2.namedWindow("Player Tracking Tuner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Player Tracking Tuner", 1600, 900)
    
    # Create trackbars - SIMPLE VERSION with clear names
    def nothing(x): pass
    
    cv2.createTrackbar("Frame", "Player Tracking Tuner", 50, total_frames - 1, nothing)
    cv2.createTrackbar("Sensitivity (lower=more)", "Player Tracking Tuner", 8, 30, nothing)
    cv2.createTrackbar("Min Size (x100 pixels)", "Player Tracking Tuner", 10, 50, nothing)
    cv2.createTrackbar("Max Size (x100 pixels)", "Player Tracking Tuner", 500, 1000, nothing)
    
    right_court_x_min = 1920
    bg_subtractor = None
    last_frame = -1
    paused = False
    current_frame_num = 50
    
    while True:
        if not paused:
            # Get trackbar values
            frame_num = cv2.getTrackbarPos("Frame", "Player Tracking Tuner")
            sensitivity = cv2.getTrackbarPos("Sensitivity (lower=more)", "Player Tracking Tuner")
            min_size = cv2.getTrackbarPos("Min Size (x100 pixels)", "Player Tracking Tuner") * 100
            max_size = cv2.getTrackbarPos("Max Size (x100 pixels)", "Player Tracking Tuner") * 100
            
            # Ensure valid values
            if sensitivity < 1:
                sensitivity = 1
            if min_size < 100:
                min_size = 100
            if max_size < 1000:
                max_size = 1000
            
            # If frame changed significantly, rebuild background
            if abs(frame_num - last_frame) > 5 or bg_subtractor is None:
                print(f"\nBuilding background model around frame {frame_num}...")
                
                # Create new background subtractor
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=200,
                    varThreshold=sensitivity,
                    detectShadows=True
                )
                
                # Build background by processing 30 previous frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 30))
                for i in range(30):
                    ret, temp_frame = cap.read()
                    if ret:
                        temp_right_court = temp_frame[:, right_court_x_min:]
                        bg_subtractor.apply(temp_right_court, learningRate=0.05)
                
                last_frame = frame_num
                print(f"Background model ready. Sensitivity={sensitivity}, MinSize={min_size}px")
            
            # Read current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                print("Cannot read frame")
                break
            
            # Extract right court
            right_court = frame[:, right_court_x_min:]
            
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(right_court, learningRate=0.001)
            
            # Remove shadows (gray pixels = 127)
            fg_mask[fg_mask == 127] = 0
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            players = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if min_size <= area <= max_size:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Person shape check
                    if 0.5 <= aspect_ratio <= 5.0:
                        x_full = x + right_court_x_min
                        players.append({
                            'bbox': (x_full, y, w, h),
                            'area': area,
                            'y': y
                        })
            
            # Sort by Y coordinate (top to bottom)
            players.sort(key=lambda p: p['y'])
            
            # Draw results
            display = frame.copy()
            
            # Draw detected players
            for i, p in enumerate(players[:2]):
                x, y, w, h = p['bbox']
                area = p['area']
                
                if i == 0:
                    color = (255, 0, 0)  # Blue for P1
                    label = f"P1 ({area:.0f}px)"
                else:
                    color = (0, 0, 255)  # Red for P2
                    label = f"P2 ({area:.0f}px)"
                
                # Draw rectangle
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 5)
                
                # Draw label with background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                cv2.rectangle(display, (x, y - label_h - 20), (x + label_w + 10, y - 5), color, -1)
                cv2.putText(display, label, (x + 5, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                # Draw center point
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.circle(display, (center_x, center_y), 12, color, -1)
            
            # Add info panel
            info_bg = np.zeros((200, display.shape[1], 3), dtype=np.uint8)
            
            # Status indicator
            if len(players) >= 2:
                status = "SUCCESS - Both P1 and P2 detected!"
                status_color = (0, 255, 0)
            elif len(players) == 1:
                status = "WARNING - Only 1 player detected. Try LOWER Sensitivity or Min Size"
                status_color = (0, 165, 255)
            else:
                status = "ERROR - No players detected. Try LOWER Sensitivity"
                status_color = (0, 0, 255)
            
            cv2.putText(info_bg, status, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
            
            # Current settings
            settings_text = [
                f"Frame: {frame_num}/{total_frames}  |  Players: {len(players[:2])}",
                f"Sensitivity: {sensitivity}  |  Min Size: {min_size}px  |  Max Size: {max_size}px",
                "Press SPACE to pause  |  Press 'Q' to save and quit"
            ]
            
            for i, text in enumerate(settings_text):
                cv2.putText(info_bg, text, (20, 80 + i * 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Combine display with info panel
            combined = np.vstack([display, info_bg])
            
            # Resize for screen
            height, width = combined.shape[:2]
            scale = min(1600 / width, 900 / height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            combined_resized = cv2.resize(combined, (new_w, new_h))
            
            cv2.imshow("Player Tracking Tuner", combined_resized)
            
            current_frame_num = frame_num
        
        # Handle keys
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("\n" + "="*70)
            print("FINAL SETTINGS TO USE:")
            print("="*70)
            print(f"varThreshold = {sensitivity}")
            print(f"min_player_area = {min_size}")
            print(f"max_player_area = {max_size}")
            print("learning_rate = 0.001")
            print("history = 200")
            print("="*70)
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'PAUSED' if paused else 'PLAYING'}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



