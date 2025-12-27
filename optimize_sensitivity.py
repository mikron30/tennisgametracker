"""
Automatic sensitivity optimizer for MOG2 player tracking.
Tests different sensitivity values and finds the best one.
"""

import cv2
import numpy as np
import json
from mark_and_track_zones import InteractiveZoneTracker

def test_sensitivity(video_path, sensitivity, num_frames=50, debug=False):
    """Test a specific sensitivity value across multiple frames."""
    cap = cv2.VideoCapture(video_path)
    
    # Load zones
    with open('player_zones.json', 'r') as f:
        zones = json.load(f)
    p1_zone = np.array(zones['p1_zone'], dtype=np.int32)
    p2_zone = np.array(zones['p2_zone'], dtype=np.int32)
    
    # Load parameters
    with open('mog2_parameters.json', 'r') as f:
        params = json.load(f)
    
    # Create MOG2 with test sensitivity
    p1_bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=sensitivity, detectShadows=True
    )
    p2_bg = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=sensitivity, detectShadows=True
    )
    
    results = {
        'sensitivity': sensitivity,
        'frames_tested': 0,
        'p1_detected': 0,
        'p2_detected': 0,
        'both_detected': 0,
        'p1_sizes': [],
        'p2_sizes': []
    }
    
    # WARM-UP: Let MOG2 learn background for first 10 frames
    warmup_frames = 10
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            # Return empty results with zero rates
            results['p1_detection_rate'] = 0
            results['p2_detection_rate'] = 0
            results['both_detection_rate'] = 0
            return results
        
        # Just learn, don't detect yet
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [p1_zone], 255)
        p1_bg.apply(frame, learningRate=params['learning_rate'])
        
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [p2_zone], 255)
        p2_bg.apply(frame, learningRate=params['learning_rate'])
    
    # Now start actual testing
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        results['frames_tested'] += 1
        
        # Detect P1
        p1_player, p1_contours = detect_in_zone(frame, p1_zone, p1_bg, 
                                    params['p1_min_area'], params['p1_max_area'],
                                    params['learning_rate'])
        # Detect P2
        p2_player, p2_contours = detect_in_zone(frame, p2_zone, p2_bg,
                                    params['p2_min_area'], params['p2_max_area'],
                                    params['learning_rate'])
        
        # Debug output for first test
        if debug and frame_num < 3:
            print(f"    Frame {frame_num}: P1 contours={len(p1_contours)}, P2 contours={len(p2_contours)}")
        
        if p1_player:
            results['p1_detected'] += 1
            results['p1_sizes'].append(p1_player['area'])
        if p2_player:
            results['p2_detected'] += 1
            results['p2_sizes'].append(p2_player['area'])
        if p1_player and p2_player:
            results['both_detected'] += 1
    
    cap.release()
    
    # Calculate statistics
    if results['frames_tested'] > 0:
        results['p1_detection_rate'] = results['p1_detected'] / results['frames_tested'] * 100
        results['p2_detection_rate'] = results['p2_detected'] / results['frames_tested'] * 100
        results['both_detection_rate'] = results['both_detected'] / results['frames_tested'] * 100
    else:
        results['p1_detection_rate'] = 0
        results['p2_detection_rate'] = 0
        results['both_detection_rate'] = 0
    
    if results['p1_sizes']:
        results['p1_avg_size'] = np.mean(results['p1_sizes'])
        results['p1_min_size'] = np.min(results['p1_sizes'])
        results['p1_max_size'] = np.max(results['p1_sizes'])
    
    if results['p2_sizes']:
        results['p2_avg_size'] = np.mean(results['p2_sizes'])
        results['p2_min_size'] = np.min(results['p2_sizes'])
        results['p2_max_size'] = np.max(results['p2_sizes'])
    
    return results

def detect_in_zone(frame, zone, bg_subtractor, min_area, max_area, learning_rate):
    """Simplified player detection in zone."""
    # Create zone mask
    zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(zone_mask, [zone], 255)
    
    # Apply MOG2 with normal learning to build background model
    fg_mask = bg_subtractor.apply(frame, learningRate=learning_rate)
    fg_mask[fg_mask == 127] = 0
    fg_mask = cv2.bitwise_and(fg_mask, zone_mask)
    
    # Clean up
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_player = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            if area > best_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    best_player = {
                        'position': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    }
                    best_area = area
    
    return best_player, contours

def main():
    video_path = "06_right_court_lines.mp4"
    
    print("\n" + "="*70)
    print("AUTOMATIC SENSITIVITY OPTIMIZER")
    print("="*70)
    print(f"Testing first 50 frames with different sensitivity values...")
    print("This will take a few minutes...\n")
    
    # Test range of sensitivity values
    sensitivity_values = [8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 25]
    
    all_results = []
    
    for i, sens in enumerate(sensitivity_values):
        print(f"Testing sensitivity {sens}... ({i+1}/{len(sensitivity_values)})")
        results = test_sensitivity(video_path, sens, num_frames=50, debug=(i==0))
        all_results.append(results)
        
        print(f"  P1 detected: {results['p1_detected']}/50 ({results['p1_detection_rate']:.1f}%)")
        print(f"  P2 detected: {results['p2_detected']}/50 ({results['p2_detection_rate']:.1f}%)")
        print(f"  Both detected: {results['both_detected']}/50 ({results['both_detection_rate']:.1f}%)")
        if 'p1_avg_size' in results:
            print(f"  P1 avg size: {results['p1_avg_size']:.0f}px (range: {results['p1_min_size']:.0f}-{results['p1_max_size']:.0f})")
        if 'p2_avg_size' in results:
            print(f"  P2 avg size: {results['p2_avg_size']:.0f}px (range: {results['p2_min_size']:.0f}-{results['p2_max_size']:.0f})")
        print()
    
    # Find best sensitivity
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Sort by "both detected" rate
    all_results.sort(key=lambda x: x['both_detection_rate'], reverse=True)
    
    print("\nRanked by detection success (both players):\n")
    for i, result in enumerate(all_results[:5]):
        print(f"{i+1}. Sensitivity {result['sensitivity']}: "
              f"{result['both_detected']}/50 frames ({result['both_detection_rate']:.1f}%) "
              f"[P1: {result['p1_detection_rate']:.1f}%, P2: {result['p2_detection_rate']:.1f}%]")
    
    best = all_results[0]
    print(f"\n" + "="*70)
    print(f"RECOMMENDED SENSITIVITY: {best['sensitivity']}")
    print(f"="*70)
    print(f"  Both players detected: {best['both_detected']}/50 frames ({best['both_detection_rate']:.1f}%)")
    print(f"  P1 detection rate: {best['p1_detection_rate']:.1f}%")
    print(f"  P2 detection rate: {best['p2_detection_rate']:.1f}%")
    
    if 'p1_avg_size' in best:
        print(f"\n  Suggested P1 range: {int(best['p1_min_size'] * 0.8)}-{int(best['p1_max_size'] * 1.5)} pixels")
    if 'p2_avg_size' in best:
        print(f"  Suggested P2 range: {int(best['p2_min_size'] * 0.8)}-{int(best['p2_max_size'] * 1.5)} pixels")
    
    # Save results to JSON
    with open('sensitivity_optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: sensitivity_optimization_results.json")
    print("="*70)

if __name__ == "__main__":
    main()

