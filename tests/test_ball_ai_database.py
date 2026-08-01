import json
import tempfile
import unittest
from pathlib import Path

from ball_ai_database import BallAIDatabase


class BallAIDatabaseTests(unittest.TestCase):
    def test_sync_and_review_keep_original_label_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder, "dataset")
            run = root / "run_1"
            images = run / "images"
            images.mkdir(parents=True)
            image = images / "frame.jpg"
            image.write_bytes(b"placeholder")
            row = {
                "image": "images/frame.jpg",
                "video": "match.mp4",
                "video_id": "match",
                "source_frame": 14,
                "image_width": 60,
                "image_height": 40,
                "ball": {"center": [20, 10], "area": 12.5, "radius_hint": 2.0},
                "label_source": "tracker_accepted_position",
                "label_status": "tracked",
                "tracking_active": True,
                "stuck_frames": 0,
                "motion_distance_px": 7.0,
                "point_index": 2,
            }
            (run / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            database = BallAIDatabase(root / "ball_ai.sqlite")
            summary = database.sync_dataset_root(root)
            self.assertEqual(summary["frames_indexed"], 1)
            frame = database.connection.execute(
                "SELECT ball_x, ball_y FROM ball_frames WHERE run_id='run_1' AND source_frame=14"
            ).fetchone()
            self.assertEqual(frame, (20.0, 10.0))
            database.set_review("run_1", 14, "corrected", corrected_x=21, corrected_y=11)
            training = database.connection.execute(
                "SELECT ball_x, ball_y, review_status FROM training_frames WHERE run_id='run_1'"
            ).fetchone()
            self.assertEqual(training, (21.0, 11.0, "corrected"))
            original = database.connection.execute(
                "SELECT ball_x, ball_y FROM ball_frames WHERE run_id='run_1'"
            ).fetchone()
            self.assertEqual(original, (20.0, 10.0))
            database.close()


if __name__ == "__main__":
    unittest.main()
