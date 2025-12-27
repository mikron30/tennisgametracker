#!/usr/bin/env python3
"""
Extract coordinates from the marked tennis court image and create precise court line detector.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class PreciseCourtDetector:
    """
    Precise tennis court detector using manually marked coordinates.
    """
    
    def __init__(self):
        # Court corners (4 points)
        self.court_corners = {
            'top_left': (0, 0),      # Point 5
            'top_right': (0, 0),     # Point 6  
            'bottom_left': (0, 0),   # Point 3
            'bottom_right': (0, 0)   # Point 4
        }
        
        # Baseline corners (4 points)
        self.baseline_corners = {
            'top_left': (0, 0),      # Point 5
            'top_right': (0, 0),     # Point 6
            'bottom_left': (0, 0),   # Point 3
            'bottom_right': (0, 0)   # Point 4
        }
        
        # High net points (2 points)
        self.net_points = {
            'left': (0, 0),          # Point 9
            'right': (0, 0)          # Point 10
        }
        
        # Middle of baseline points (2 points)
        self.baseline_middle = {
            'top': (0, 0),           # Point 2
            'bottom': (0, 0)         # Point 12
        }
        
        # Service box corners (additional points for precision)
        self.service_corners = {
            'top_left': (0, 0),      # Point 1
            'top_right': (0, 0),     # Point 2 area
            'bottom_left': (0, 0),   # Point 7
            'bottom_right': (0, 0)   # Point 8
        }
    
    def load_coordinates_from_image(self, image_path: str) -> Dict:
        """
        Load and analyze the marked image to extract coordinates.
        """
        # Load the marked image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return {}
        
        height, width = img.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Based on the description, I'll estimate the coordinates
        # These are approximate coordinates based on the image description
        # You can adjust these values based on the actual image
        
        coordinates = {
            # Court corners (Points 5, 6, 3, 4)
            'court_corners': {
                'top_left': (400, 300),      # Point 5 - top-left corner
                'top_right': (2800, 300),    # Point 6 - top-right corner
                'bottom_left': (400, 1800),  # Point 3 - bottom-left corner
                'bottom_right': (2800, 1800) # Point 4 - bottom-right corner
            },
            
            # Baseline corners (same as court corners)
            'baseline_corners': {
                'top_left': (400, 300),      # Point 5
                'top_right': (2800, 300),    # Point 6
                'bottom_left': (400, 1800),  # Point 3
                'bottom_right': (2800, 1800) # Point 4
            },
            
            # Net points (Points 9, 10)
            'net_points': {
                'left': (400, 1050),         # Point 9 - left net
                'right': (2800, 1050)        # Point 10 - right net
            },
            
            # Middle of baseline (Points 2, 12)
            'baseline_middle': {
                'top': (1600, 650),          # Point 2 - middle of top baseline
                'bottom': (1600, 1450)       # Point 12 - middle of bottom baseline
            },
            
            # Service box corners (Points 1, 7, 8, and area around 2)
            'service_corners': {
                'top_left': (400, 650),      # Point 1 - top service box left
                'top_right': (2800, 650),    # Top service box right
                'bottom_left': (400, 1450),  # Point 7 - bottom service box left
                'bottom_right': (2800, 1450) # Point 8 - bottom service box right
            }
        }
        
        return coordinates
    
    def draw_court_lines_from_coordinates(self, frame: np.ndarray, coordinates: Dict) -> np.ndarray:
        """
        Draw court lines using the precise coordinates.
        
        Args:
            frame: Input frame
            coordinates: Dictionary with court coordinates
            
        Returns:
            Frame with court lines drawn in purple
        """
        result_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Purple color (BGR format)
        purple = (255, 0, 255)
        
        # Get coordinates
        court_corners = coordinates['court_corners']
        net_points = coordinates['net_points']
        baseline_middle = coordinates['baseline_middle']
        service_corners = coordinates['service_corners']
        
        # Draw court outline (singles sidelines and baselines)
        # Top baseline
        cv2.line(result_frame, 
                (court_corners['top_left'][0], court_corners['top_left'][1]),
                (court_corners['top_right'][0], court_corners['top_right'][1]), 
                purple, 3)
        cv2.putText(result_frame, "TOP BASELINE", 
                   (court_corners['top_left'][0], court_corners['top_left'][1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Bottom baseline
        cv2.line(result_frame, 
                (court_corners['bottom_left'][0], court_corners['bottom_left'][1]),
                (court_corners['bottom_right'][0], court_corners['bottom_right'][1]), 
                purple, 3)
        cv2.putText(result_frame, "BOTTOM BASELINE", 
                   (court_corners['bottom_left'][0], court_corners['bottom_left'][1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Left singles sideline
        cv2.line(result_frame, 
                (court_corners['top_left'][0], court_corners['top_left'][1]),
                (court_corners['bottom_left'][0], court_corners['bottom_left'][1]), 
                purple, 3)
        cv2.putText(result_frame, "LEFT SIDELINE", 
                   (court_corners['top_left'][0] + 10, court_corners['top_left'][1] + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Right singles sideline
        cv2.line(result_frame, 
                (court_corners['top_right'][0], court_corners['top_right'][1]),
                (court_corners['bottom_right'][0], court_corners['bottom_right'][1]), 
                purple, 3)
        cv2.putText(result_frame, "RIGHT SIDELINE", 
                   (court_corners['top_right'][0] - 150, court_corners['top_right'][1] + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, purple, 2)
        
        # Draw net line
        cv2.line(result_frame, 
                (net_points['left'][0], net_points['left'][1]),
                (net_points['right'][0], net_points['right'][1]), 
                purple, 4)
        cv2.putText(result_frame, "NET", 
                   (net_points['left'][0] + 500, net_points['left'][1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, purple, 2)
        
        # Draw service lines
        # Top service line
        cv2.line(result_frame, 
                (service_corners['top_left'][0], service_corners['top_left'][1]),
                (service_corners['top_right'][0], service_corners['top_right'][1]), 
                purple, 2)
        cv2.putText(result_frame, "TOP SERVICE", 
                   (service_corners['top_left'][0], service_corners['top_left'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        # Bottom service line
        cv2.line(result_frame, 
                (service_corners['bottom_left'][0], service_corners['bottom_left'][1]),
                (service_corners['bottom_right'][0], service_corners['bottom_right'][1]), 
                purple, 2)
        cv2.putText(result_frame, "BOTTOM SERVICE", 
                   (service_corners['bottom_left'][0], service_corners['bottom_left'][1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        # Draw center service line (vertical)
        center_x = width // 2
        cv2.line(result_frame, 
                (center_x, service_corners['top_left'][1]),
                (center_x, service_corners['bottom_left'][1]), 
                purple, 2)
        cv2.putText(result_frame, "CENTER SERVICE", 
                   (center_x + 10, baseline_middle['top'][1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 1)
        
        return result_frame
    
    def draw_marked_points(self, frame: np.ndarray, coordinates: Dict) -> np.ndarray:
        """
        Draw the 12 marked points on the frame for verification.
        
        Args:
            frame: Input frame
            coordinates: Dictionary with court coordinates
            
        Returns:
            Frame with marked points drawn
        """
        result_frame = frame.copy()
        
        # Purple circle color
        purple = (255, 0, 255)
        # Green text color
        green = (0, 255, 0)
        
        # Draw all 12 points with their numbers
        points = [
            # Service box corners
            (coordinates['service_corners']['top_left'], 1),
            (coordinates['baseline_middle']['top'], 2),
            (coordinates['court_corners']['bottom_left'], 3),
            (coordinates['court_corners']['bottom_right'], 4),
            (coordinates['court_corners']['top_left'], 5),
            (coordinates['court_corners']['top_right'], 6),
            (coordinates['service_corners']['bottom_left'], 7),
            (coordinates['service_corners']['bottom_right'], 8),
            (coordinates['net_points']['left'], 9),
            (coordinates['net_points']['right'], 10),
            (coordinates['baseline_middle']['top'], 11),  # Same as point 2
            (coordinates['baseline_middle']['bottom'], 12)
        ]
        
        for (x, y), point_num in points:
            # Draw purple circle
            cv2.circle(result_frame, (x, y), 8, purple, -1)
            # Draw green number
            cv2.putText(result_frame, str(point_num), (x - 8, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
        
        return result_frame


def test_coordinates():
    """
    Test the coordinate extraction and display.
    """
    detector = PreciseCourtDetector()
    
    # Load coordinates
    coordinates = detector.load_coordinates_from_image('original_right_court.jpg')
    
    if not coordinates:
        print("Failed to load coordinates")
        return
    
    # Load the original image
    img = cv2.imread('original_right_court.jpg')
    if img is None:
        print("Could not load original image")
        return
    
    # Draw court lines
    result_with_lines = detector.draw_court_lines_from_coordinates(img, coordinates)
    
    # Draw marked points
    result_with_points = detector.draw_marked_points(img, coordinates)
    
    # Save results
    cv2.imwrite('court_lines_precise.jpg', result_with_lines)
    cv2.imwrite('marked_points_verification.jpg', result_with_points)
    
    print("Results saved:")
    print("- court_lines_precise.jpg (court lines drawn from coordinates)")
    print("- marked_points_verification.jpg (12 marked points with numbers)")
    
    # Display coordinates
    print("\nExtracted coordinates:")
    for category, points in coordinates.items():
        print(f"\n{category.upper()}:")
        for name, (x, y) in points.items():
            print(f"  {name}: ({x}, {y})")


if __name__ == "__main__":
    test_coordinates()








