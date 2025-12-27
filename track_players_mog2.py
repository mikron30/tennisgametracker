"""
Player tracking using MOG2 Background Subtraction.
Best for tennis videos with stationary camera and moving players.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class PlayerTrackerMOG2:
    def __init__(self):
        """Initialize MOG2 background subtractor for player tracking."""
        
        # Create MOG2 background subtractor
        # detectShadows=True helps remove shadows
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,           # Number of frames for background model (reduced for faster adaptation)
            varThreshold=8,        # Threshold for pixel classification (LOWER = more sensitive to detect P2)
            detectShadows=True     # Detect and mark shadows
        )
        
        # Right court boundary (for 3840x2160 video)
        self.right_court_x_min = 1920
        
        # Player size thresholds (adjusted for 4K video)
        # Players appear as large moving blobs: 80-250px wide, 150-400px tall
        # P2 (far side) appears smaller than P1 (near side)
        self.min_player_area = 1500   # Minimum 1500 pixels (lowered to catch P2)
        self.max_player_area = 80000  # Maximum 80000 pixels
        
        # Aspect ratio range for players (height/width)
        self.min_aspect_ratio = 0.8  # Can be wider when moving
        self.max_aspect_ratio = 4.0  # Can be very tall
        
        # Store previous player positions for tracking continuity
        self.p1_history = []
        self.p2_history = []
        
        # Learning rate for background model (-1 = automatic)
        self.learning_rate = 0.0005  # Very low rate = players stay visible longer even when still
    
    def detect_players(self, frame: np.ndarray, frame_number: int = 0, debug: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        Detect moving players using background subtraction.
        
        Args:
            frame: Full video frame
            frame_number: Current frame number
            debug: If True, show debug visualizations
            
        Returns:
            List of (x, y, w, h) bounding boxes for detected players, sorted top-to-bottom
        """
        height, width = frame.shape[:2]
        
        # STEP 1: Extract ONLY the right court
        right_court = frame[:, self.right_court_x_min:]
        
        # STEP 2: Apply background subtraction
        fg_mask = self.bg_subtractor.apply(right_court, learningRate=self.learning_rate)
        
        # STEP 3: Remove shadows (value 127 in MOG2)
        # Convert shadow pixels (127) to background (0)
        fg_mask[fg_mask == 127] = 0
        
        # STEP 4: Clean up the mask with morphological operations
        # Remove noise with opening
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        # Fill holes with closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Dilate to connect nearby regions
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)
        
        # STEP 5: Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 6: Filter contours to find players
        player_candidates = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if not (self.min_player_area <= area <= self.max_player_area):
                if debug and area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    print(f"  Contour {i}: REJECTED - area={area:.0f}px (outside {self.min_player_area}-{self.max_player_area})")
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (person should be reasonably shaped)
            aspect_ratio = h / w if w > 0 else 0
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                if debug:
                    print(f"  Contour {i}: REJECTED - aspect_ratio={aspect_ratio:.2f} (outside {self.min_aspect_ratio}-{self.max_aspect_ratio})")
                continue
            
            # Adjust X coordinate back to full frame coordinates
            x_full = x + self.right_court_x_min
            
            player_candidates.append({
                'bbox': (x_full, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center_y': y + h // 2,
                'center_x': x_full + w // 2
            })
            
            if debug:
                print(f"  Contour {i}: VALID - pos=({x_full},{y}), size=({w}x{h}), area={area:.0f}px, aspect={aspect_ratio:.2f}")
        
        # STEP 7: Sort players by Y coordinate (top = P1, bottom = P2)
        player_candidates.sort(key=lambda p: p['center_y'])
        
        # STEP 8: Return top 2 players (P1 and P2)
        players = [p['bbox'] for p in player_candidates[:2]]
        
        if debug:
            print(f"\n=== PlayerTrackerMOG2 Debug (Frame {frame_number}) ===")
            print(f"Right court region: X >= {self.right_court_x_min}")
            print(f"Found {len(contours)} total contours")
            print(f"Found {len(player_candidates)} player candidates:")
            for i, p in enumerate(player_candidates[:5]):  # Show top 5
                x, y, w, h = p['bbox']
                print(f"  {i+1}. pos=({x},{y}), size=({w}x{h}), area={p['area']:.0f}, aspect={p['aspect_ratio']:.2f}")
            print(f"Returning {len(players)} players")
        
        return players
    
    def reset_background(self):
        """Reset the background model (useful when scene changes dramatically)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        print("Background model reset")
    
    def set_learning_rate(self, rate: float):
        """
        Set the learning rate for background model.
        
        Args:
            rate: Learning rate (0.0 to 1.0)
                  0.0 = static background (no learning)
                  0.001 = very slow learning (good for stable scenes)
                  0.01 = moderate learning
                  -1 = automatic
        """
        self.learning_rate = rate
        print(f"Learning rate set to {rate}")
    
    def visualize_detection(self, frame: np.ndarray, players: List[Tuple[int, int, int, int]], 
                           show_mask: bool = True) -> np.ndarray:
        """
        Visualize player detections on the frame.
        
        Args:
            frame: Original frame
            players: List of (x, y, w, h) bounding boxes
            show_mask: If True, show the foreground mask alongside
            
        Returns:
            Visualization frame
        """
        display = frame.copy()
        
        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(players):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)
            label = "P1" if i == 0 else "P2"
            
            # Draw rectangle
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 5)
            
            # Draw label
            cv2.putText(display, label, (x, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(display, (center_x, center_y), 10, color, -1)
        
        if show_mask:
            # Get the foreground mask
            right_court = frame[:, self.right_court_x_min:]
            fg_mask = self.bg_subtractor.apply(right_court, learningRate=0)  # Don't update during visualization
            
            # Create a colored version of the mask
            mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[fg_mask == 255] = [0, 255, 0]  # Foreground in green
            mask_colored[fg_mask == 127] = [0, 165, 255]  # Shadows in orange
            
            # Overlay mask on right court portion of display
            display[:, self.right_court_x_min:] = cv2.addWeighted(
                display[:, self.right_court_x_min:], 0.7,
                mask_colored, 0.3, 0
            )
        
        return display


def main():
    """Test the MOG2 player tracker on video."""
    
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Initialize tracker
    tracker = PlayerTrackerMOG2()
    
    print("\n=== MOG2 Player Tracker ===")
    print("This tracker learns the background over the first ~50 frames")
    print("Players will be detected as moving objects")
    print("\nControls:")
    print("  SPACE = Pause/Resume")
    print("  'R' = Reset background model")
    print("  'D' = Toggle debug output")
    print("  'M' = Toggle mask overlay")
    print("  'Q' = Quit")
    print("\nStarting playback...")
    
    cv2.namedWindow("Player Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Player Tracking", 1600, 900)
    
    frame_number = 0
    paused = False
    debug = False
    show_mask = True
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video")
                break
            
            frame_number += 1
            
            # Detect players
            players = tracker.detect_players(frame, frame_number, debug=debug)
            
            # Visualize
            display = tracker.visualize_detection(frame, players, show_mask=show_mask)
            
            # Add frame info
            height, width = frame.shape[:2]
            info_text = f"Frame: {frame_number} | Players: {len(players)} | Press SPACE to pause"
            cv2.putText(display, info_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Resize for display
            scale = min(1600 / width, 900 / height)
            new_w = int(width * scale)
            new_h = int(height * scale)
            display_resized = cv2.resize(display, (new_w, new_h))
            
            cv2.imshow("Player Tracking", display_resized)
        
        # Handle keyboard input
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('r'):
            tracker.reset_background()
        elif key == ord('d'):
            debug = not debug
            print(f"Debug output: {'ON' if debug else 'OFF'}")
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"Mask overlay: {'ON' if show_mask else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking complete!")


if __name__ == "__main__":
    main()

