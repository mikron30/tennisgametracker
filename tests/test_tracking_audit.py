import json
import tempfile
import unittest
from pathlib import Path

from tracking_audit import (
    AuditProgress,
    _progress_frame_from_line,
    find_suspicious_events,
    point_endpoints,
    parse_track_log,
)


class TrackingAuditTests(unittest.TestCase):
    def test_progress_frame_parser_uses_tracker_events(self):
        self.assertEqual(_progress_frame_from_line("[TRACK] f123: pos=(1, 2)"), 123)
        self.assertEqual(_progress_frame_from_line("Frame 456: Ball tracked"), 456)
        self.assertEqual(
            _progress_frame_from_line("Analysis complete! Processed 1000 frames"),
            999,
        )

    def test_progress_writer_creates_text_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            progress = AuditProgress(folder)
            progress.update(
                status="running",
                stage="tracker regression",
                overall_percent=25.0,
                current_frame=50,
                end_frame=200,
                message="Tracking frame 50",
            )
            text = Path(folder, "progress.txt").read_text(encoding="utf-8")
            state = json.loads(Path(folder, "progress.json").read_text(encoding="utf-8"))
            self.assertIn("25.0%", text)
            self.assertEqual(state["current_frame"], 50)

    def test_parse_track_log_reads_positions_and_diagnostics(self):
        text = "\n".join([
            "[TRACK] f49: pos=(1797, 375) vel=18.0px stuck=0 vel_hist=[]",
            "[TRACK] f50: pos=(1779, 377) vel=18.1px stuck=0 vel_hist=[]",
            "Frame 51: [PLAYER-OCCLUSION] holding stale point",
            "[BALL_LOSS_DIAGNOSTIC] f51: reason=player occlusion pos=(1779,377) stuck=3 recovery=pending",
        ])
        tracks, events = parse_track_log(text)
        self.assertEqual(tracks[50]["pos"], [1779, 377])
        self.assertEqual({event["type"] for event in events}, {"player_occlusion", "ball_loss"})

    def test_find_suspicious_events_flags_gap_jump_and_stuck(self):
        tracks = {
            10: {"frame": 10, "pos": [100, 100], "velocity": 20.0, "stuck": 0},
            12: {"frame": 12, "pos": [800, 800], "velocity": 20.0, "stuck": 4},
        }
        events = find_suspicious_events(tracks, [])
        types = set(events[0]["types"])
        self.assertIn("trace_gap", types)
        self.assertIn("large_jump", types)
        self.assertIn("stuck", types)

    def test_point_endpoints_exposes_one_based_tracker_mapping(self):
        endpoints = point_endpoints([{
            "point_index": "1",
            "serve_start_frame": "22",
            "point_end_frame": "322",
            "end_position": "(2022,513)",
            "end_reason": "Ball hit the net",
            "category": "net_error",
            "winner": "P2",
        }])
        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0]["raw_end_frame"], 321)
        self.assertEqual(endpoints[0]["end_position"], [2022, 513])


if __name__ == "__main__":
    unittest.main()
