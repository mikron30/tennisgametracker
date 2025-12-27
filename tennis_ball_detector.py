import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import argparse
import os


def resize_to_fit_screen(frame: np.ndarray, max_width: int = 1200, max_height: int = 800) -> np.ndarray:
    """
    Resize frame to fit within specified screen dimensions while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width for display
        max_height: Maximum height for display
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor to fit screen
    scale_h = max_height / height
    scale_w = max_width / width
    scale = min(scale_h, scale_w, 1.0)  # Don't scale up, only down
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(frame, (new_width, new_height))
    
    return frame


class TennisBallDetector:
    """
    A class to detect tennis balls in video frames using OpenCV.
    Uses color-based detection and contour analysis to locate tennis balls.
    Enhanced with serve area detection and improved tracking.
    """
    
    def __init__(self, 
                 lower_color: Tuple[int, int, int] = (15, 50, 50),
                 upper_color: Tuple[int, int, int] = (35, 255, 255),
                 min_radius: int = 3,
                 max_radius: int = 30,
                 serve_area_threshold: float = 0.1):
        """
        Initialize the tennis ball detector.
        
        Args:
            lower_color: Lower bound for HSV color range (default: yellow-green)
            upper_color: Upper bound for HSV color range (default: yellow-green)
            min_radius: Minimum radius for detected circles
            max_radius: Maximum radius for detected circles
            serve_area_threshold: Threshold for serve area detection
        """
        self.lower_color = np.array(lower_color)
        self.upper_color = np.array(upper_color)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.serve_area_threshold = serve_area_threshold
        self.serve_areas = []
        self.tracking_active = False
        self.last_ball_position = None
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame for better ball detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Preprocessed frame
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def detect_serve_areas(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential serve areas in the frame by looking for court lines and player positions.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of (x, y, width, height) tuples for serve areas
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection to find court lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        serve_areas = []
        if lines is not None:
            # Analyze the frame to find potential serve areas
            height, width = frame.shape[:2]
            
            # Look for areas in the top portion of the frame (where serves typically happen)
            top_region = int(height * 0.3)  # Top 30% of frame
            
            # Create a mask for potential serve areas
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Look for horizontal lines in the top region (baseline)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if y1 < top_region and y2 < top_region and abs(y2 - y1) < 10:
                    # This is likely a baseline in the serve area
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for serve area
                    x, y, w, h = cv2.boundingRect(contour)
                    serve_areas.append((x, y, w, h))
        
        return serve_areas
    
    def is_ball_in_serve_area(self, ball_pos: Tuple[int, int], serve_areas: List[Tuple[int, int, int, int]]) -> bool:
        """
        Check if a ball position is within any serve area.
        
        Args:
            ball_pos: (x, y) position of the ball
            serve_areas: List of serve areas
            
        Returns:
            True if ball is in a serve area
        """
        x, y = ball_pos
        for sx, sy, sw, sh in serve_areas:
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                return True
        return False
    
    def detect_ball_by_color(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect tennis balls using color-based segmentation with improved accuracy.
        
        Args:
            frame: HSV frame
            
        Returns:
            List of (x, y, radius) tuples for detected balls
        """
        # Create mask for tennis ball color (yellow-green)
        mask = cv2.inRange(frame, self.lower_color, self.upper_color)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply additional filtering to reduce noise
        mask = cv2.medianBlur(mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        balls = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # More restrictive area filtering
            if area > 20 and area < 2000:  # Reasonable range for tennis ball
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Filter by radius
                if self.min_radius <= radius <= self.max_radius:
                    # Additional circularity check
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Should be reasonably circular
                            balls.append((int(x), int(y), int(radius)))
        
        return balls
    
    def detect_ball_by_hough_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect tennis balls using Hough Circle Transform.
        
        Args:
            frame: Input frame
            
        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        balls = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                balls.append((x, y, r))
        
        return balls
    
    def detect_ball(self, frame: np.ndarray, method: str = "combined", 
                   frame_number: int = 0) -> List[Tuple[int, int, int]]:
        """
        Detect tennis balls in a frame using specified method with serve-triggered tracking.
        
        Args:
            frame: Input video frame
            method: Detection method ("color", "hough", or "combined")
            frame_number: Current frame number for tracking state
            
        Returns:
            List of (x, y, radius) tuples for detected balls
        """
        # Detect serve areas every 30 frames to update serve positions
        if frame_number % 30 == 0:
            self.serve_areas = self.detect_serve_areas(frame)
        
        # Detect balls using specified method
        if method == "color":
            hsv_frame = self.preprocess_frame(frame)
            balls = self.detect_ball_by_color(hsv_frame)
        
        elif method == "hough":
            balls = self.detect_ball_by_hough_circles(frame)
        
        elif method == "combined":
            # Use both methods and combine results
            hsv_frame = self.preprocess_frame(frame)
            color_balls = self.detect_ball_by_color(hsv_frame)
            hough_balls = self.detect_ball_by_hough_circles(frame)
            
            # Combine and remove duplicates
            all_balls = color_balls + hough_balls
            balls = self._remove_duplicate_detections(all_balls)
        
        else:
            raise ValueError("Method must be 'color', 'hough', or 'combined'")
        
        # Apply serve-triggered tracking logic
        filtered_balls = []
        
        for ball in balls:
            x, y, radius = ball
            
            # If tracking is not active, check if ball is in serve area
            if not self.tracking_active:
                if self.is_ball_in_serve_area((x, y), self.serve_areas):
                    self.tracking_active = True
                    self.last_ball_position = (x, y)
                    filtered_balls.append(ball)
                    print(f"Tracking activated at frame {frame_number} - Ball detected in serve area")
            else:
                # If tracking is active, check if ball movement is reasonable
                if self.last_ball_position is not None:
                    last_x, last_y = self.last_ball_position
                    distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                    
                    # Allow reasonable movement (max 100 pixels between frames)
                    if distance <= 100:
                        filtered_balls.append(ball)
                        self.last_ball_position = (x, y)
                    else:
                        # Ball moved too far, might be false positive
                        continue
                else:
                    filtered_balls.append(ball)
                    self.last_ball_position = (x, y)
        
        # If no balls detected for 10 consecutive frames, deactivate tracking
        if not filtered_balls and self.tracking_active:
            if not hasattr(self, 'no_ball_count'):
                self.no_ball_count = 0
            self.no_ball_count += 1
            
            if self.no_ball_count > 10:
                self.tracking_active = False
                self.last_ball_position = None
                self.no_ball_count = 0
                print(f"Tracking deactivated at frame {frame_number} - No ball detected for 10 frames")
        else:
            self.no_ball_count = 0
        
        return filtered_balls
    
    def _remove_duplicate_detections(self, balls: List[Tuple[int, int, int]], 
                                   threshold: int = 30) -> List[Tuple[int, int, int]]:
        """
        Remove duplicate ball detections that are too close to each other.
        
        Args:
            balls: List of detected balls
            threshold: Distance threshold for considering balls as duplicates
            
        Returns:
            Filtered list of unique ball detections
        """
        if len(balls) <= 1:
            return balls
        
        unique_balls = []
        for ball in balls:
            is_duplicate = False
            for unique_ball in unique_balls:
                distance = np.sqrt((ball[0] - unique_ball[0])**2 + (ball[1] - unique_ball[1])**2)
                if distance < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_balls.append(ball)
        
        return unique_balls
    
    def draw_detections(self, frame: np.ndarray, balls: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Draw detected balls and serve areas on the frame.
        
        Args:
            frame: Input frame
            balls: List of detected balls
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Draw serve areas
        for (x, y, w, h) in self.serve_areas:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result_frame, "Serve Area", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw detected balls
        for (x, y, radius) in balls:
            # Draw circle around detected ball
            cv2.circle(result_frame, (x, y), radius, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result_frame, (x, y), 2, (0, 0, 255), -1)
            
            # Add text label
            cv2.putText(result_frame, f"Ball", (x - 20, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw tracking status
        status_text = "TRACKING ACTIVE" if self.tracking_active else "WAITING FOR SERVE"
        color = (0, 255, 0) if self.tracking_active else (0, 0, 255)
        cv2.putText(result_frame, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result_frame


class TennisVideoProcessor:
    """
    A class to process tennis videos and detect balls throughout the video.
    """
    
    def __init__(self, detector: TennisBallDetector):
        """
        Initialize the video processor.
        
        Args:
            detector: TennisBallDetector instance
        """
        self.detector = detector
        self.detection_history = []
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     method: str = "combined", show_preview: bool = True) -> List[List[Tuple[int, int, int]]]:
        """
        Process a tennis video and detect balls in each frame.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            method: Detection method to use
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
                
                # Detect balls in current frame
                balls = self.detector.detect_ball(frame, method=method, frame_number=frame_count)
                all_detections.append(balls)
                
                # Draw detections on frame
                result_frame = self.detector.draw_detections(frame, balls)
                
                # Add frame information
                cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_frame, f"Balls detected: {len(balls)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                if out:
                    out.write(result_frame)
                
                # Show preview
                if show_preview:
                    # Resize frame to fit screen
                    display_frame = resize_to_fit_screen(result_frame)
                    cv2.imshow('Tennis Ball Detection', display_frame)
                    
                    # Check for 'q' key press to exit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("Processing stopped by user (Q key pressed)")
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
        
        print(f"Processing complete! Detected balls in {len([d for d in all_detections if d])} frames")
        return all_detections
    
    def analyze_detections(self, detections: List[List[Tuple[int, int, int]]]) -> dict:
        """
        Analyze the detection results and provide statistics.
        
        Args:
            detections: List of detections for each frame
            
        Returns:
            Dictionary with analysis results
        """
        total_frames = len(detections)
        frames_with_balls = len([d for d in detections if d])
        
        # Calculate ball positions over time
        ball_trajectories = []
        for frame_idx, frame_detections in enumerate(detections):
            for ball in frame_detections:
                ball_trajectories.append((frame_idx, ball[0], ball[1]))
        
        analysis = {
            'total_frames': total_frames,
            'frames_with_balls': frames_with_balls,
            'detection_rate': frames_with_balls / total_frames if total_frames > 0 else 0,
            'total_detections': sum(len(frame_detections) for frame_detections in detections),
            'ball_trajectories': ball_trajectories
        }
        
        return analysis
    
    def plot_trajectory(self, detections: List[List[Tuple[int, int, int]]], 
                       video_width: int, video_height: int):
        """
        Plot the ball trajectory over time.
        
        Args:
            detections: List of detections for each frame
            video_width: Width of the video
            video_height: Height of the video
        """
        # Extract ball positions
        x_positions = []
        y_positions = []
        frame_numbers = []
        
        for frame_idx, frame_detections in enumerate(detections):
            for ball in frame_detections:
                x_positions.append(ball[0])
                y_positions.append(ball[1])
                frame_numbers.append(frame_idx)
        
        if not x_positions:
            print("No ball detections found to plot")
            return
        
        # Create trajectory plot
        plt.figure(figsize=(12, 8))
        
        # Plot trajectory
        plt.subplot(2, 1, 1)
        plt.plot(x_positions, y_positions, 'b-', alpha=0.7, linewidth=2)
        plt.scatter(x_positions, y_positions, c=frame_numbers, cmap='viridis', s=50)
        plt.colorbar(label='Frame Number')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Tennis Ball Trajectory')
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        plt.grid(True, alpha=0.3)
        
        # Plot position over time
        plt.subplot(2, 1, 2)
        plt.plot(frame_numbers, x_positions, 'r-', label='X Position', linewidth=2)
        plt.plot(frame_numbers, y_positions, 'g-', label='Y Position', linewidth=2)
        plt.xlabel('Frame Number')
        plt.ylabel('Position (pixels)')
        plt.title('Ball Position Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to run the tennis ball detection system.
    """
    parser = argparse.ArgumentParser(description='Tennis Ball Detection System')
    parser.add_argument('video_path', help='Path to input tennis video file')
    parser.add_argument('--output', '-o', help='Path to save output video')
    parser.add_argument('--method', '-m', choices=['color', 'hough', 'combined'], 
                       default='combined', help='Detection method to use')
    parser.add_argument('--no-preview', action='store_true', 
                       help='Disable real-time preview')
    parser.add_argument('--plot-trajectory', action='store_true',
                       help='Plot ball trajectory after processing')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        return
    
    # Initialize detector and processor
    detector = TennisBallDetector()
    processor = TennisVideoProcessor(detector)
    
    try:
        # Process video
        print(f"Starting tennis ball detection on: {args.video_path}")
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
        print(f"Frames with ball detections: {analysis['frames_with_balls']}")
        print(f"Detection rate: {analysis['detection_rate']:.2%}")
        print(f"Total ball detections: {analysis['total_detections']}")
        
        # Plot trajectory if requested
        if args.plot_trajectory:
            cap = cv2.VideoCapture(args.video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            processor.plot_trajectory(detections, width, height)
        
        if args.output:
            print(f"\nOutput video saved to: {args.output}")
    
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
