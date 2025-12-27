"""Test the simple player detector."""

import cv2
from simple_player_detector import SimplePlayerDetector


def main():
    # Open video
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Go to frame 7
    cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot read frame")
        return
    
    # Initialize detector
    detector = SimplePlayerDetector()
    
    # Detect players with debug output
    print("Testing SimplePlayerDetector on frame 7...")
    players = detector.detect_players(frame, debug=True)
    
    # Draw results on full frame
    display = frame.copy()
    for i, (x, y, w, h) in enumerate(players):
        color = (255, 0, 0) if i == 0 else (0, 0, 255)
        label = "P1" if i == 0 else "P2"
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 5)
        cv2.putText(display, label, (x, y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Resize for display
    height, width = frame.shape[:2]
    scale = min(1920 / width, 1080 / height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    display_resized = cv2.resize(display, (new_w, new_h))
    
    cv2.imshow("Player Detection Result", display_resized)
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()



