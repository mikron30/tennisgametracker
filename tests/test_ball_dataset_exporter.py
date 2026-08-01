import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ball_dataset_exporter import BallDatasetExporter


class BallDatasetExporterTests(unittest.TestCase):
    def test_writes_raw_frame_and_pseudo_label_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            frame[10, 20] = (0, 255, 255)
            exporter = BallDatasetExporter(folder, "match.mp4")
            self.assertTrue(exporter.write_frame(
                frame,
                source_frame=12,
                ball_center=(20, 10),
                ball_area=12.5,
                stuck_frames=0,
                motion_distance=8.0,
                tracking_active=True,
                point_index=3,
            ))
            self.assertTrue(exporter.write_frame(
                frame,
                source_frame=13,
                ball_center=(21, 11),
                ball_area=None,
                stuck_frames=2,
                motion_distance=None,
                tracking_active=True,
                point_index=3,
            ))
            exporter.close()

            run_dir = next(Path(folder).iterdir())
            rows = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual([row["source_frame"] for row in rows], [12, 13])
            self.assertEqual(rows[0]["ball"]["center"], [20, 10])
            self.assertEqual(rows[0]["label_status"], "tracked")
            self.assertEqual(rows[1]["label_status"], "held_previous_position")
            self.assertTrue((run_dir / rows[0]["image"]).is_file())
            summary = json.loads((run_dir / "dataset_summary.json").read_text())
            self.assertEqual(summary["frames_saved"], 2)


if __name__ == "__main__":
    unittest.main()
