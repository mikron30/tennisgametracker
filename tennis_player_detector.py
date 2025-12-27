import cv2
import numpy as np
from typing import List, Tuple, Optional
import argparse
import os


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


class TennisPlayerDetector:
    """
    A class to detect and track tennis players on the right side of the court.
    """
    
    def __init__(self, 
                 right_court_ratio: float = 0.5,
                 min_person_area: int = 2000,
                 max_person_area: int = 30000,
                 court_center_x: float = 0.75,
                 court_center_y: float = 0.5,
                 separation_line_x: float = 0.5,
                 separation_top_x: float = 0.15,
                 separation_bottom_x: float = 0.45):
        """
        Initialize the tennis player detector.
        
        Args:
            right_court_ratio: Ratio of frame width to consider as right court (0.5 = right half)
            min_person_area: Minimum area for person detection
            max_person_area: Maximum area for person detection
            court_center_x: X coordinate of court center (0.75 = 75% from left)
            court_center_y: Y coordinate of court center (0.5 = middle)
            separation_line_x: X coordinate of purple line separation (0.5 = middle)
            separation_top_x: X coordinate of line start at top (0.15 = 15% from left)
            separation_bottom_x: X coordinate of line end at bottom (0.45 = 45% from left)
        """
        self.right_court_ratio = right_court_ratio
        self.min_person_area = min_person_area
        self.max_person_area = max_person_area
        self.court_center_x = court_center_x
        self.court_center_y = court_center_y
        self.separation_line_x = separation_line_x
        self.separation_top_x = separation_top_x
        self.separation_bottom_x = separation_bottom_x
        self.players = []
        self.player_templates = []  # Store player templates for tracking
        
    def get_right_court_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Extract the right court region of interest.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (right_court_frame, x_offset)
        """
        height, width = frame.shape[:2]
        right_start_x = int(width * (1 - self.right_court_ratio))
        
        # Extract right court region
        right_court = frame[:, right_start_x:width]
        
        return right_court, right_start_x
    
    def create_court_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a mask for the right tennis court area using a diagonal line separation.
        
        Args:
            frame: Input frame
            
        Returns:
            Binary mask for the court area
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a diagonal line mask to separate left and right courts
        # The purple line appears diagonal due to camera perspective
        # Define diagonal line points (top-left to bottom-right with slight curve)
        top_x = int(width * 0.45)  # Start slightly left of center at top
        bottom_x = int(width * 0.55)  # End slightly right of center at bottom
        
        # Create curved diagonal line using multiple points for better perspective match
        # Based on the user's precise drawing: starts at 40% top, ends at 55% bottom
        top_x = int(width * self.separation_top_x)  # Start at 40% from left at top
        bottom_x = int(width * self.separation_bottom_x)  # End at 55% from left at bottom
        
        curve_points = []
        for y in range(0, height, 50):  # Sample every 50 pixels
            # Calculate x position with precise curve matching the user's drawing
            progress = y / height
            # Create a more pronounced curve that matches the perspective
            # The curve is concave towards the left court in upper half, then straightens
            if progress < 0.5:  # Upper half - more curved
                curve_offset = int(15 * np.sin(progress * np.pi * 2))  # Stronger curve
            else:  # Lower half - straighter
                curve_offset = int(5 * np.sin(progress * np.pi))  # Gentler curve
            
            x = int(top_x + (bottom_x - top_x) * progress + curve_offset)
            curve_points.append([x, y])
        
        # Add corner points to complete the polygon
        curve_points.extend([
            [width, height],      # Bottom right corner
            [width, 0]            # Top right corner
        ])
        
        points = np.array(curve_points, np.int32)
        
        # Fill the right side (right court area)
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def detect_players(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect tennis players in the right court using background subtraction and contour analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            List of (x, y, width, height) bounding boxes for detected players
        """
        # Get right court region
        right_court, x_offset = self.get_right_court_roi(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(right_court, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Use adaptive threshold to detect moving objects
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        players = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (reasonable size for a person)
            if self.min_person_area <= area <= self.max_person_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional filtering: person should be taller than wide
                aspect_ratio = h / w if w > 0 else 0
                if aspect_ratio > 1.2:  # Person should be taller than wide
                    # Adjust coordinates back to full frame
                    x += x_offset
                    players.append((x, y, w, h))
        
        # Sort players by x-coordinate (left to right)
        players.sort(key=lambda p: p[0])
        
        return players
    
    def detect_court_lines(self, frame: np.ndarray) -> dict:
        """
        Detect all tennis court lines including net, baselines, service lines, and sidelines.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with detected lines: net, baselines, service_lines, sidelines
        """
        height, width = frame.shape[:2]
        
        # Get right court region
        right_court, x_offset = self.get_right_court_roi(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(right_court, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection to find white lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Find all lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=50, maxLineGap=10)
        
        court_lines = {
            'net': None,
            'baselines': [],
            'service_lines': [],
            'sidelines': []
        }
        
        if lines is not None:
            court_height = right_court.shape[0]
            court_width = right_court.shape[1]
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Adjust coordinates to full frame
                x1 += x_offset
                x2 += x_offset
                y1 += 0  # No Y offset needed
                y2 += 0
                
                # Classify lines based on orientation and position
                is_horizontal = abs(y2 - y1) < 10
                is_vertical = abs(x2 - x1) < 10
                
                if is_horizontal:
                    # Horizontal lines - could be net, baselines, or service lines
                    line_y = (y1 + y2) // 2
                    
                    # Net is typically in the middle area of the court
                    if abs(line_y - court_height // 2) < court_height * 0.15:
                        if court_lines['net'] is None or abs(line_y - court_height // 2) < abs(court_lines['net'] - court_height // 2):
                            court_lines['net'] = line_y
                    
                    # Baselines are at the top and bottom of the court
                    elif line_y < court_height * 0.3 or line_y > court_height * 0.7:
                        court_lines['baselines'].append(line_y)
                    
                    # Service lines are between net and baselines
                    else:
                        court_lines['service_lines'].append(line_y)
                
                elif is_vertical:
                    # Vertical lines - sidelines
                    line_x = (x1 + x2) // 2
                    court_lines['sidelines'].append(line_x)
        
        return court_lines
    
    def detect_net_line(self, frame: np.ndarray) -> int:
        """
        Detect the net line (white line in the middle of the court).
        
        Args:
            frame: Input frame
            
        Returns:
            Y coordinate of the net line
        """
        court_lines = self.detect_court_lines(frame)
        if court_lines['net'] is not None:
            return court_lines['net']
        
        # Fallback to middle of frame if net not detected
        height, width = frame.shape[:2]
        return height // 2
    
    def detect_players_advanced(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Advanced player detection understanding court structure and net position.
        
        Args:
            frame: Input frame
            
        Returns:
            List of (x, y, width, height) bounding boxes for detected players
        """
        height, width = frame.shape[:2]
        
        # Detect net line to understand court structure
        net_y = self.detect_net_line(frame)
        
        # Create court mask to limit detection to court area only
        court_mask = self.create_court_mask(frame)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define ranges for skin color (more restrictive)
        lower_skin = np.array([0, 30, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Define ranges for clothing colors (blue, light colors, white)
        # Blue clothing
        lower_blue = np.array([100, 50, 50], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Light clothing (white, grey, light colors)
        lower_light = np.array([0, 0, 180], dtype=np.uint8)
        upper_light = np.array([180, 30, 255], dtype=np.uint8)
        light_mask = cv2.inRange(hsv, lower_light, upper_light)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(skin_mask, blue_mask)
        combined_mask = cv2.bitwise_or(combined_mask, light_mask)
        
        # Apply court mask to limit detection area
        combined_mask = cv2.bitwise_and(combined_mask, court_mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        players = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_person_area <= area <= self.max_person_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional filtering for person-like shapes
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 <= aspect_ratio <= 3.0:  # Person should be taller than wide
                    # Check if detection is within court bounds
                    center_x = x + w // 2
                    center_y = y + h // 2
                    if court_mask[center_y, center_x] > 0:
                        players.append((x, y, w, h))
        
        # Remove overlapping detections
        players = self._remove_overlapping_detections(players)
        
        # Sort players by their position relative to the net
        # Player 1: closer to camera (lower Y), Player 2: further from camera (higher Y)
        players.sort(key=lambda p: p[1])  # Sort by Y coordinate (top to bottom)
        
        # Limit to maximum 2 players and assign proper labels
        if len(players) >= 2:
            # Player 1 is the one closer to camera (lower Y coordinate)
            # Player 2 is the one further from camera (higher Y coordinate)
            return players[:2]
        else:
            return players
    
    def _remove_overlapping_detections(self, detections: List[Tuple[int, int, int, int]], 
                                     overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        Remove overlapping detections.
        """
        if len(detections) <= 1:
            return detections
        
        # Calculate overlap between bounding boxes
        filtered = []
        for i, det1 in enumerate(detections):
            is_duplicate = False
            for j, det2 in enumerate(detections):
                if i != j:
                    overlap = self._calculate_overlap(det1, det2)
                    if overlap > overlap_threshold:
                        # Keep the larger detection
                        area1 = det1[2] * det1[3]
                        area2 = det2[2] * det2[3]
                        if area1 < area2:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered.append(det1)
        
        return filtered
    
    def _calculate_overlap(self, box1: Tuple[int, int, int, int], 
                          box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap ratio between two bounding boxes.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, frame: np.ndarray, players: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw detected players and court area on the frame.
        
        Args:
            frame: Input frame
            players: List of detected players
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw curved diagonal purple line separation between left and right courts
        # Based on user's precise drawing: starts at 40% top, ends at 55% bottom
        top_x = int(width * self.separation_top_x)  # Start at 40% from left at top
        bottom_x = int(width * self.separation_bottom_x)  # End at 55% from left at bottom
        
        # Draw curved line using multiple points matching the user's drawing
        curve_points = []
        for y in range(0, height, 20):  # Sample every 20 pixels for smoother line
            progress = y / height
            # Create the precise curve matching the user's drawing
            if progress < 0.5:  # Upper half - more curved (concave towards left)
                curve_offset = int(15 * np.sin(progress * np.pi * 2))  # Stronger curve
            else:  # Lower half - straighter
                curve_offset = int(5 * np.sin(progress * np.pi))  # Gentler curve
            
            x = int(top_x + (bottom_x - top_x) * progress + curve_offset)
            curve_points.append((x, y))
        
        # Draw the curved line
        for i in range(len(curve_points) - 1):
            cv2.line(result_frame, curve_points[i], curve_points[i + 1], (255, 0, 255), 3)
        
        # Add court labels
        cv2.putText(result_frame, "LEFT COURT", (top_x - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(result_frame, "RIGHT COURT", (bottom_x + 20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Add arrow pointing to right court
        arrow_start = (bottom_x + 50, 100)
        arrow_end = (bottom_x + 150, 100)
        cv2.arrowedLine(result_frame, arrow_start, arrow_end, (255, 0, 255), 3, tipLength=0.3)
        
        # Detect and draw all court lines
        court_lines = self.detect_court_lines(frame)
        
        # Draw net line (white line in middle of court)
        if court_lines['net'] is not None:
            net_y = court_lines['net']
            cv2.line(result_frame, (0, net_y), (width, net_y), (255, 0, 255), 3)  # Purple net line
            cv2.putText(result_frame, "NET", (width - 100, net_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw baselines (top and bottom of court)
        for baseline_y in court_lines['baselines']:
            cv2.line(result_frame, (0, baseline_y), (width, baseline_y), (255, 0, 255), 2)  # Purple baseline
            cv2.putText(result_frame, "BASELINE", (10, baseline_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Draw service lines
        for service_y in court_lines['service_lines']:
            cv2.line(result_frame, (0, service_y), (width, service_y), (255, 0, 255), 2)  # Purple service line
            cv2.putText(result_frame, "SERVICE", (10, service_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Draw sidelines
        for sideline_x in court_lines['sidelines']:
            cv2.line(result_frame, (sideline_x, 0), (sideline_x, height), (255, 0, 255), 2)  # Purple sideline
            cv2.putText(result_frame, "SIDELINE", (sideline_x + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Draw detected players with specific colors and court position understanding
        player_colors = [(255, 0, 0), (0, 255, 255)]  # Blue for Player 1, Yellow for Player 2
        
        for i, (x, y, w, h) in enumerate(players):
            color = player_colors[i] if i < len(player_colors) else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(result_frame, (center_x, center_y), 8, color, -1)
            
            # Determine player position relative to actual net
            net_y = court_lines['net'] if court_lines['net'] is not None else height // 2
            if center_y < net_y:
                court_side = "Near Court (Behind Service Line)"
            else:
                court_side = "Far Court (Behind Service Line)"
            
            # Add player label with court position
            player_name = f"Player {i+1}"
            if i == 0:
                player_name = "Player 1 (Near Court)"
            elif i == 1:
                player_name = "Player 2 (Far Court)"
            
            cv2.putText(result_frame, player_name, (x, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add court position info
            cv2.putText(result_frame, court_side, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add coordinates
            cv2.putText(result_frame, f"({center_x}, {center_y})", (x, y + h + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add detection info
        cv2.putText(result_frame, f"Players in Court: {len(players)}/2", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add status
        if len(players) == 2:
            cv2.putText(result_frame, "Both Players Detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif len(players) == 1:
            cv2.putText(result_frame, "1 Player Detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(result_frame, "No Players Detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return result_frame


class TennisPlayerVideoProcessor:
    """
    A class to process tennis videos and detect players throughout the video.
    """
    
    def __init__(self, detector: TennisPlayerDetector):
        """
        Initialize the video processor.
        
        Args:
            detector: TennisPlayerDetector instance
        """
        self.detector = detector
        self.detection_history = []
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     method: str = "advanced", show_preview: bool = True) -> List[List[Tuple[int, int, int, int]]]:
        """
        Process a tennis video and detect players in each frame.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            method: Detection method ("basic" or "advanced")
            show_preview: Whether to show real-time preview
            
        Returns:
            List of detections for each frame
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Focusing on right court only")
        if show_preview:
            print("Press 'Q' key to stop processing and exit")
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect players in current frame
                if method == "basic":
                    players = self.detector.detect_players(frame)
                else:
                    players = self.detector.detect_players_advanced(frame)
                
                all_detections.append(players)
                
                # Draw detections on frame
                result_frame = self.detector.draw_detections(frame, players)
                
                # Add frame information
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                if out:
                    out.write(result_frame)
                
                # Show preview
                if show_preview:
                    # Resize frame to fit screen
                    display_frame = resize_to_fit_screen(result_frame)
                    cv2.imshow('Tennis Player Detection', display_frame)
                    
                    # Check for 'q' key press to exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("Processing stopped by user (Q key pressed)")
                        cv2.destroyAllWindows()  # Close all windows immediately
                        break
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:  # Every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            # Force close any remaining windows
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        
        print(f"Processing complete! Detected players in {len([d for d in all_detections if d])} frames")
        return all_detections
    
    def analyze_detections(self, detections: List[List[Tuple[int, int, int, int]]]) -> dict:
        """
        Analyze the detection results and provide statistics.
        
        Args:
            detections: List of detections for each frame
            
        Returns:
            Dictionary with analysis results
        """
        total_frames = len(detections)
        frames_with_players = len([d for d in detections if d])
        
        # Calculate player positions over time
        player_trajectories = {1: [], 2: []}
        for frame_idx, frame_detections in enumerate(detections):
            for i, player in enumerate(frame_detections[:2]):  # Max 2 players
                x, y, w, h = player
                center_x = x + w // 2
                center_y = y + h // 2
                player_trajectories[i + 1].append((frame_idx, center_x, center_y))
        
        analysis = {
            'total_frames': total_frames,
            'frames_with_players': frames_with_players,
            'detection_rate': frames_with_players / total_frames if total_frames > 0 else 0,
            'total_detections': sum(len(frame_detections) for frame_detections in detections),
            'player_trajectories': player_trajectories
        }
        
        return analysis


def main():
    """
    Main function to run the tennis player detection system.
    """
    parser = argparse.ArgumentParser(description='Tennis Player Detection System')
    parser.add_argument('video_path', help='Path to input tennis video file')
    parser.add_argument('--output', '-o', help='Path to save output video')
    parser.add_argument('--method', '-m', choices=['basic', 'advanced'], 
                       default='advanced', help='Detection method to use')
    parser.add_argument('--no-preview', action='store_true', 
                       help='Disable real-time preview')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    # Initialize detector and processor
    detector = TennisPlayerDetector()
    processor = TennisPlayerVideoProcessor(detector)
    
    try:
        # Process video
        print(f"Starting tennis player detection on: {args.video_path}")
        detections = processor.process_video(
            args.video_path,
            output_path=args.output,
            method=args.method,
            show_preview=not args.no_preview
        )
        
        # Analyze results
        analysis = processor.analyze_detections(detections)
        print("\n=== Detection Analysis ===")
        print(f"Total frames processed: {analysis['total_frames']}")
        print(f"Frames with player detections: {analysis['frames_with_players']}")
        print(f"Detection rate: {analysis['detection_rate']:.2%}")
        print(f"Total player detections: {analysis['total_detections']}")
        
        if args.output:
            print(f"\nOutput video saved to: {args.output}")
    
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
