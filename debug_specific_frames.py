#!/usr/bin/env python3
"""
Debug script to test specific frames with specific sensitivity
Shows detailed detection results for each frame
"""

import cv2
import numpy as np
import json

def load_zones():
    """Load player zones from JSON"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return data['p1_zone'], data['p2_zone']

def load_parameters():
    """Load MOG2 parameters from JSON"""
    with open('mog2_parameters.json', 'r') as f:
        return json.load(f)

def detect_player_in_zone(frame, mog2, zone_points, min_area, max_area, zone_name, learning_rate, last_detected):
    """Detect player in a specific zone using MOG2 with adaptive learning"""
    # Create mask for the zone
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    zone_array = np.array(zone_points, dtype=np.int32)
    cv2.fillPoly(mask, [zone_array], 255)
    
    # Apply mask to frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Adaptive learning rate: pause learning when player was detected in last frame
    adaptive_rate = -1 if last_detected else learning_rate
    fg_mask = mog2.apply(masked_frame, learningRate=adaptive_rate)
    
    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply zone mask to foreground mask
    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze all contours
    all_contours_info = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Only log contours larger than 100px
            all_contours_info.append({
                'area': area,
                'valid': min_area <= area <= max_area
            })
    
    # Sort by area (largest first)
    all_contours_info.sort(key=lambda x: x['area'], reverse=True)
    
    # Find valid contours
    valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    
    if valid_contours:
        # Find the largest valid contour
        largest = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area = cv2.contourArea(largest)
        return (x, y, w, h), area, all_contours_info
    
    return None, None, all_contours_info

def test_frames(video_path, start_frame, end_frame, sensitivity):
    """Test specific frames with given sensitivity"""
    
    print("\n" + "="*70)
    print(f"TESTING FRAMES {start_frame}-{end_frame} WITH SENSITIVITY {sensitivity}")
    print("="*70)
    
    # Load zones and parameters
    p1_zone, p2_zone = load_zones()
    params = load_parameters()
    
    print(f"\nParameters:")
    print(f"  P1 area range: {params['p1_min_area']:,} - {params['p1_max_area']:,} pixels")
    print(f"  P2 area range: {params['p2_min_area']:,} - {params['p2_max_area']:,} pixels")
    print(f"  Sensitivity: {sensitivity}")
    print(f"  History: {params['history']}")
    print(f"  Learning rate: {params['learning_rate']}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Initialize MOG2 for each zone
    p1_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=params['history'],
        varThreshold=sensitivity,
        detectShadows=False
    )
    
    p2_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=params['history'],
        varThreshold=sensitivity,
        detectShadows=False
    )
    
    # Warmup phase - let MOG2 learn the background
    # We need at least 10-20 frames for MOG2 to build a good background model
    warmup_end = max(10, start_frame)
    print(f"\nWarmup phase (frames 0-{warmup_end})...")
    print("Learning background model...")
    
    for frame_num in range(warmup_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update MOG2 with normal learning
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # P1 zone warmup
        p1_array = np.array(p1_zone, dtype=np.int32)
        cv2.fillPoly(mask, [p1_array], 255)
        p1_masked = cv2.bitwise_and(frame, frame, mask=mask)
        p1_mog2.apply(p1_masked, learningRate=params['learning_rate'])
        
        # P2 zone warmup
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        p2_array = np.array(p2_zone, dtype=np.int32)
        cv2.fillPoly(mask, [p2_array], 255)
        p2_masked = cv2.bitwise_and(frame, frame, mask=mask)
        p2_mog2.apply(p2_masked, learningRate=params['learning_rate'])
        
        if frame_num % 5 == 0:
            print(f"  Frame {frame_num}...")
    
    print(f"Warmup complete. Starting detection from frame {start_frame}...\n")
    
    # Statistics
    p1_detected_count = 0
    p2_detected_count = 0
    both_detected_count = 0
    total_frames = 0
    
    # Track detection state for adaptive learning
    p1_last_detected = False
    p2_last_detected = False
    
    # Test each frame
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"\nReached end of video at frame {frame_num}")
            break
        
        total_frames += 1
        
        # Detect P1
        p1_bbox, p1_area, p1_contours = detect_player_in_zone(
            frame, p1_mog2, p1_zone, 
            params['p1_min_area'], params['p1_max_area'], 
            "P1", params['learning_rate'], p1_last_detected
        )
        
        # Detect P2
        p2_bbox, p2_area, p2_contours = detect_player_in_zone(
            frame, p2_mog2, p2_zone, 
            params['p2_min_area'], params['p2_max_area'], 
            "P2", params['learning_rate'], p2_last_detected
        )
        
        # Update statistics
        p1_found = p1_bbox is not None
        p2_found = p2_bbox is not None
        
        if p1_found:
            p1_detected_count += 1
        if p2_found:
            p2_detected_count += 1
        if p1_found and p2_found:
            both_detected_count += 1
        
        # Update detection state for next frame's adaptive learning
        p1_last_detected = p1_found
        p2_last_detected = p2_found
        
        # Print frame results
        status = "OK" if (p1_found and p2_found) else ("P1" if p1_found else ("P2" if p2_found else "NO"))
        print(f"Frame {frame_num:3d}: [{status:>2}] ", end="")
        
        # P1 details
        if p1_found:
            print(f"P1={p1_area:,}px ", end="")
        else:
            if p1_contours:
                largest = p1_contours[0]
                print(f"P1=NO (largest={largest['area']:,}px) ", end="")
            else:
                print(f"P1=NO (no contours) ", end="")
        
        # P2 details
        if p2_found:
            print(f"P2={p2_area:,}px")
        else:
            if p2_contours:
                largest = p2_contours[0]
                print(f"P2=NO (largest={largest['area']:,}px)")
            else:
                print(f"P2=NO (no contours)")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total frames tested: {total_frames}")
    print(f"P1 detected: {p1_detected_count}/{total_frames} ({100*p1_detected_count/total_frames:.1f}%)")
    print(f"P2 detected: {p2_detected_count}/{total_frames} ({100*p2_detected_count/total_frames:.1f}%)")
    print(f"Both detected: {both_detected_count}/{total_frames} ({100*both_detected_count/total_frames:.1f}%)")
    print("="*70)
    
    cap.release()

if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    # Test frames 1-49 with sensitivity 16
    test_frames(video_path, start_frame=1, end_frame=49, sensitivity=16)

