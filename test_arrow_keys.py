"""
Test script to find the correct arrow key codes on Windows.
"""

import cv2
import numpy as np

# Create a simple window
frame = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.putText(frame, "Press arrow keys to see their codes", (10, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(frame, "Press Q to quit", (10, 100),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.namedWindow("Arrow Key Test", cv2.WINDOW_NORMAL)

print("Press arrow keys to see their codes")
print("Press Q to quit")
print()

while True:
    cv2.imshow("Arrow Key Test", frame)
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break
    elif key == 255:
        continue
    else:
        print(f"Key pressed: {key} (0x{key:02x})")

cv2.destroyAllWindows()











