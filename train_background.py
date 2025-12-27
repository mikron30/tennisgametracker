import cv2
import json
import numpy as np
import pickle

def load_zones():
    with open('player_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['p1_zone'], np.int32), np.array(data['p2_zone'], np.int32)

class BackgroundTrainer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.training_start = None
        self.training_end = None
        self.p1_zone, self.p2_zone = load_zones()
        
        print("=" * 70)
        print("BACKGROUND TRAINING - Find frames WITHOUT players in zones")
        print("=" * 70)
        print(f"Total frames: {self.total_frames}")
        print("\nKEYBOARD CONTROLS:")
        print("  'D' = Next frame")
        print("  'A' = Previous frame")
        print("  'F' = Fast forward (+10 frames)")
        print("  'B' = Fast backward (-10 frames)")
        print("  'S' = Set START of training range")
        print("  'E' = Set END of training range")
        print("  'T' = TRAIN background model (after setting start/end)")
        print("  'Q' = Quit")
        print("=" * 70)
    
    def draw_frame(self, frame):
        """Draw zones and info on frame"""
        display = frame.copy()
        
        # Draw zones
        cv2.polylines(display, [self.p1_zone], True, (255, 255, 0), 2)
        cv2.polylines(display, [self.p2_zone], True, (255, 255, 0), 2)
        cv2.putText(display, "P1 Zone", tuple(self.p1_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display, "P2 Zone", tuple(self.p2_zone[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show current frame
        cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show training range
        if self.training_start is not None:
            cv2.putText(display, f"Training START: {self.training_start}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if self.training_end is not None:
            cv2.putText(display, f"Training END: {self.training_end}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show instructions
        if self.training_start is None:
            cv2.putText(display, "Press 'S' to set START frame", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.training_end is None:
            cv2.putText(display, "Press 'E' to set END frame", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, "Press 'T' to TRAIN background model", 
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display
    
    def navigate(self):
        """Navigate through video to find training frames"""
        cv2.namedWindow('Background Training', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Background Training', 1600, 900)
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                print("End of video reached")
                break
            
            display = self.draw_frame(frame)
            
            # Resize for display
            height, width = display.shape[:2]
            scale = 0.5
            display_resized = cv2.resize(display, (int(width*scale), int(height*scale)))
            
            cv2.imshow('Background Training', display_resized)
            
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
            elif key == ord('s'):
                self.training_start = self.current_frame
                print(f"\n[OK] Training START set to frame {self.training_start}")
            elif key == ord('e'):
                if self.training_start is None:
                    print("\n[ERROR] Set START frame first (press 'S')")
                elif self.current_frame <= self.training_start:
                    print(f"\n[ERROR] END frame must be after START frame ({self.training_start})")
                else:
                    self.training_end = self.current_frame
                    print(f"[OK] Training END set to frame {self.training_end}")
                    print(f"[OK] Training range: {self.training_start} to {self.training_end} ({self.training_end - self.training_start + 1} frames)")
            elif key == ord('t'):
                if self.training_start is None or self.training_end is None:
                    print("\n[ERROR] Set both START and END frames first")
                else:
                    cv2.destroyAllWindows()
                    self.train_background()
                    return
            elif key == ord('q'):
                print("\nQuitting without training")
                cv2.destroyAllWindows()
                return
        
        cv2.destroyAllWindows()
    
    def train_background(self):
        """Train background model on selected frames"""
        print("\n" + "=" * 70)
        print("TRAINING BACKGROUND MODEL")
        print("=" * 70)
        print(f"Training on frames {self.training_start} to {self.training_end}")
        print(f"Total training frames: {self.training_end - self.training_start + 1}")
        
        # Create MOG2
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=16,
            detectShadows=True
        )
        
        # Train on selected frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.training_start)
        
        for frame_idx in range(self.training_start, self.training_end + 1):
            ret, frame = self.cap.read()
            if not ret:
                print(f"[ERROR] Could not read frame {frame_idx}")
                break
            
            # Apply with learning
            mog2.apply(frame, learningRate=-1)
            
            if (frame_idx - self.training_start) % 10 == 0:
                print(f"  Training progress: {frame_idx - self.training_start + 1}/{self.training_end - self.training_start + 1}")
        
        print("\n[OK] Background model trained successfully!")
        
        # Save the background model
        bg_model = {
            'training_start': self.training_start,
            'training_end': self.training_end,
            'history': 200,
            'varThreshold': 16,
            'background_image': mog2.getBackgroundImage()
        }
        
        with open('background_model.pkl', 'wb') as f:
            pickle.dump(bg_model, f)
        
        print("[OK] Background model saved to 'background_model.pkl'")
        
        # Save training config
        config = {
            'training_start': self.training_start,
            'training_end': self.training_end,
            'trained_frames': self.training_end - self.training_start + 1
        }
        
        with open('background_training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("[OK] Training config saved to 'background_training_config.json'")
        print("=" * 70)
        
        # Show background image
        bg_image = mog2.getBackgroundImage()
        if bg_image is not None:
            print("\nDisplaying learned background image...")
            cv2.namedWindow('Learned Background', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Learned Background', 1600, 900)
            
            # Resize for display
            height, width = bg_image.shape[:2]
            scale = 0.5
            bg_resized = cv2.resize(bg_image, (int(width*scale), int(height*scale)))
            
            cv2.imshow('Learned Background', bg_resized)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    video_path = "20251011124747503_FV3553362380_FV3553362.mp4"
    
    trainer = BackgroundTrainer(video_path)
    try:
        trainer.navigate()
    finally:
        trainer.cleanup()



