import cv2
import json
import numpy as np
from collections import defaultdict

def load_zones():
    """Load P1 and P2 zones"""
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

class PixelAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 10
        
        # Load zones
        self.p1_zone, self.p2_zone = load_zones()
        
        print("=" * 70)
        print("PIXEL COLOR ANALYZER")
        print("=" * 70)
        print("Strategy: Analyze ALL pixels in P1/P2 zones")
        print("  1. Extract all pixel colors (HSV)")
        print("  2. Cluster similar colors together")
        print("  3. Identify compact clusters (likely players)")
        print("  4. Reject large scattered clusters (fence/ground/lines)")
        print("=" * 70)
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10)")
        print("  'B' = Fast backward (-10)")
        print("  'SPACE' = Analyze current frame")
        print("  'Q' = Quit")
        print("=" * 70)
    
    def get_zone_pixels(self, frame, zone_polygon):
        """Extract all pixels from a zone"""
        # Create zone mask
        zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(zone_mask, [zone_polygon], 255)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get pixel coordinates and colors in zone
        y_coords, x_coords = np.where(zone_mask > 0)
        pixels = hsv[y_coords, x_coords]
        coords = np.column_stack([x_coords, y_coords])
        
        return pixels, coords
    
    def cluster_colors(self, pixels, coords, num_bins=10):
        """Cluster pixels by HSV color into bins"""
        # Quantize HSV values to create color bins
        h_bins = num_bins  # Hue: 0-179
        s_bins = 5   # Saturation: 0-255
        v_bins = 5   # Value: 0-255
        
        # Convert to int32 to avoid overflow
        h_quantized = (pixels[:, 0].astype(np.int32) * h_bins // 180).astype(int)
        s_quantized = (pixels[:, 1].astype(np.int32) * s_bins // 256).astype(int)
        v_quantized = (pixels[:, 2].astype(np.int32) * v_bins // 256).astype(int)
        
        # Create color clusters
        clusters = defaultdict(list)
        for i, (h, s, v) in enumerate(zip(h_quantized, s_quantized, v_quantized)):
            cluster_id = (h, s, v)
            clusters[cluster_id].append(i)
        
        return clusters
    
    def analyze_cluster_spatial_distribution(self, cluster_indices, coords):
        """Analyze if a cluster is spatially compact (player) or scattered (fence/ground)"""
        cluster_coords = coords[cluster_indices]
        
        if len(cluster_coords) < 100:  # Too small
            return None
        
        # Calculate bounding box
        x_min, y_min = cluster_coords.min(axis=0)
        x_max, y_max = cluster_coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        bbox_area = width * height
        
        if bbox_area == 0:
            return None
        
        # Calculate fill ratio (how much of bbox is filled with pixels)
        cluster_area = len(cluster_coords)
        fill_ratio = cluster_area / bbox_area
        
        # Calculate aspect ratio
        aspect_ratio = height / width if width > 0 else 0
        
        # Calculate compactness (using convex hull if available)
        try:
            hull = cv2.convexHull(cluster_coords.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            compactness = cluster_area / hull_area if hull_area > 0 else 0
        except:
            compactness = fill_ratio
        
        return {
            'count': cluster_area,
            'bbox': (int(x_min), int(y_min), int(width), int(height)),
            'fill_ratio': fill_ratio,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'coords': cluster_coords
        }
    
    def score_cluster_as_player(self, stats):
        """Score a cluster based on how likely it is to be a player"""
        if stats is None:
            return 0
        
        score = 0
        
        # Size: players should be 5000-200000 pixels
        if 5000 <= stats['count'] <= 200000:
            score += 30
        elif 2000 <= stats['count'] <= 300000:
            score += 10
        
        # Fill ratio: compact shapes (players) have higher fill ratio
        # Fence/ground/lines are scattered, fill ratio < 0.3
        if stats['fill_ratio'] > 0.5:
            score += 40
        elif stats['fill_ratio'] > 0.3:
            score += 20
        
        # Aspect ratio: people are taller than wide (0.8 to 3.0)
        if 0.8 <= stats['aspect_ratio'] <= 3.0:
            score += 30
        elif 0.5 <= stats['aspect_ratio'] <= 4.0:
            score += 10
        
        return score
    
    def analyze_zone(self, frame, zone_polygon, zone_name):
        """Analyze all pixels in a zone and find player colors"""
        print(f"\n{'=' * 70}")
        print(f"ANALYZING {zone_name} ZONE")
        print('=' * 70)
        
        # Get all pixels
        pixels, coords = self.get_zone_pixels(frame, zone_polygon)
        print(f"Total pixels in zone: {len(pixels)}")
        
        # Cluster by color
        clusters = self.cluster_colors(pixels, coords, num_bins=10)
        print(f"Total color clusters: {len(clusters)}")
        
        # Analyze each cluster
        cluster_stats = []
        for cluster_id, cluster_indices in clusters.items():
            h, s, v = cluster_id
            
            # Convert back to HSV range
            h_min = h * 180 // 10
            h_max = (h + 1) * 180 // 10 - 1
            s_min = s * 256 // 5
            s_max = (s + 1) * 256 // 5 - 1
            v_min = v * 256 // 5
            v_max = (v + 1) * 256 // 5 - 1
            
            stats = self.analyze_cluster_spatial_distribution(cluster_indices, coords)
            if stats is None:
                continue
            
            # Score as potential player
            score = self.score_cluster_as_player(stats)
            
            stats['hsv_range'] = (h_min, h_max, s_min, s_max, v_min, v_max)
            stats['score'] = score
            stats['cluster_id'] = cluster_id
            
            cluster_stats.append(stats)
        
        # Sort by score
        cluster_stats.sort(key=lambda x: x['score'], reverse=True)
        
        # Print top 10 clusters
        print(f"\nTop 10 clusters (by player likelihood score):")
        print(f"{'Rank':<5} {'Score':<6} {'Pixels':<8} {'Fill%':<7} {'Aspect':<7} {'HSV Range':<40} {'BBox'}")
        print('-' * 120)
        
        for i, stats in enumerate(cluster_stats[:10]):
            h_min, h_max, s_min, s_max, v_min, v_max = stats['hsv_range']
            x, y, w, h = stats['bbox']
            print(f"{i+1:<5} {stats['score']:<6} {stats['count']:<8} "
                  f"{stats['fill_ratio']*100:<6.1f}% {stats['aspect_ratio']:<7.2f} "
                  f"H:{h_min:3d}-{h_max:3d} S:{s_min:3d}-{s_max:3d} V:{v_min:3d}-{v_max:3d} "
                  f"({x},{y},{w},{h})")
        
        return cluster_stats
    
    def visualize_clusters(self, frame, zone_polygon, cluster_stats, zone_name):
        """Visualize top clusters"""
        print(f"\nVisualizing top 5 clusters for {zone_name}...")
        
        h, w = frame.shape[:2]
        display_h = int(h * 0.4)
        display_w = int(w * 0.4)
        
        for i, stats in enumerate(cluster_stats[:5]):
            # Create mask for this cluster
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            coords = stats['coords'].astype(np.int32)
            mask[coords[:, 1], coords[:, 0]] = 255
            
            # Convert to color for display
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Draw bounding box
            x, y, w_box, h_box = stats['bbox']
            cv2.rectangle(mask_color, (x, y), (x+w_box, y+h_box), (0, 255, 0), 3)
            
            # Draw zone outline
            cv2.polylines(mask_color, [zone_polygon], True, (255, 255, 0), 2)
            
            # Add info
            h_min, h_max, s_min, s_max, v_min, v_max = stats['hsv_range']
            cv2.putText(mask_color, f"{zone_name} #{i+1} - Score: {stats['score']}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(mask_color, f"Pixels: {stats['count']} Fill: {stats['fill_ratio']*100:.1f}%", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(mask_color, f"H:{h_min}-{h_max} S:{s_min}-{s_max} V:{v_min}-{v_max}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize and show
            mask_color = cv2.resize(mask_color, (display_w, display_h))
            cv2.imshow(f'{zone_name} - Cluster {i+1}', mask_color)
    
    def analyze_frame(self, frame):
        """Analyze current frame"""
        print("\n" + "=" * 70)
        print(f"ANALYZING FRAME {self.current_frame}")
        print("=" * 70)
        
        # Analyze P1
        p1_clusters = self.analyze_zone(frame, self.p1_zone, "P1")
        
        # Analyze P2
        p2_clusters = self.analyze_zone(frame, self.p2_zone, "P2")
        
        # Visualize
        self.visualize_clusters(frame, self.p1_zone, p1_clusters, "P1")
        self.visualize_clusters(frame, self.p2_zone, p2_clusters, "P2")
        
        # Recommendation
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)
        if p1_clusters:
            print(f"P1 BEST: Cluster #1 with score {p1_clusters[0]['score']}")
            h_min, h_max, s_min, s_max, v_min, v_max = p1_clusters[0]['hsv_range']
            print(f"  HSV Range: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"  Pixels: {p1_clusters[0]['count']}, Fill: {p1_clusters[0]['fill_ratio']*100:.1f}%")
        
        if p2_clusters:
            print(f"P2 BEST: Cluster #1 with score {p2_clusters[0]['score']}")
            h_min, h_max, s_min, s_max, v_min, v_max = p2_clusters[0]['hsv_range']
            print(f"  HSV Range: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})")
            print(f"  Pixels: {p2_clusters[0]['count']}, Fill: {p2_clusters[0]['fill_ratio']*100:.1f}%")
        
        print("=" * 70)
        print("TIP: Look at the visual windows to see which cluster best isolates the player!")
        print("=" * 70)
    
    def draw_frame(self, frame):
        """Draw current frame with zones"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 255, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (255, 255, 0), 2)
        cv2.putText(display, "P1 Zone", tuple(self.p1_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display, "P2 Zone", tuple(self.p2_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Frame info
        cv2.putText(display, f"Frame: {self.current_frame}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "Press SPACE to analyze", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return display
    
    def run(self):
        """Main loop"""
        cv2.namedWindow('Pixel Analyzer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pixel Analyzer', 1600, 900)
        
        # Auto-analyze first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            self.analyze_frame(frame)
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video")
                break
            
            display = self.draw_frame(frame)
            
            # Resize for display
            display_resized = cv2.resize(display, (int(display.shape[1]*0.5), int(display.shape[0]*0.5)))
            cv2.imshow('Pixel Analyzer', display_resized)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d'):
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('a'):
                self.current_frame = max(self.current_frame - 1, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord('f'):
                self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                print(f"Frame {self.current_frame}")
            elif key == ord('b'):
                self.current_frame = max(self.current_frame - 10, 0)
                print(f"Frame {self.current_frame}")
            elif key == ord(' '):
                self.analyze_frame(frame)
            elif key == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    analyzer = PixelAnalyzer(video_path)
    analyzer.run()

