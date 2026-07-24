import unittest

from point_visual_auditor import PointVisualAuditAgent, parse_point


class PointVisualAuditorTests(unittest.TestCase):
    def setUp(self):
        self.agent = PointVisualAuditAgent("video.mp4", "history.csv", provider="local")
        self.agent.frame_height = 1080
        self.agent.config = {
            "net_area_points": [
                [100, 400],
                [500, 400],
                [500, 520],
                [100, 520],
            ]
        }
        self.available_metrics = {
            "frame_available": True,
            "position_available": True,
            "position_in_frame": True,
            "motion_mean": 5.0,
            "motion_max": 90,
            "hsv_match_fraction": None,
            "center_hsv": [30, 100, 180],
        }

    def test_parse_point_accepts_csv_format(self):
        self.assertEqual(parse_point("(1834,999)"), (1834, 999))
        self.assertEqual(parse_point("(1834, 999)"), (1834, 999))
        self.assertIsNone(parse_point(""))

    def test_serve_handoff_metrics_accept_verified_airborne_launch(self):
        trace = {
            1245: {"pos": (1723, 890)},
            1253: {"pos": (1615, 739)},
            1257: {"pos": (1562, 786)},
            1261: {"pos": (1515, 906)},
            1262: {"pos": (1615, 773)},
            1264: {"pos": (1759, 602)},
            1274: {"pos": (1965, 436)},
        }

        metrics = self.agent._serve_handoff_metrics(1242, (1762, 1023), trace)

        self.assertTrue(metrics["available"])
        self.assertGreater(metrics["launch_progress"], metrics["required_progress"])

    def test_serve_handoff_metrics_flags_player_takeover(self):
        trace = {
            1245: {"pos": (1723, 890)},
            1253: {"pos": (1615, 739)},
            1257: {"pos": (1562, 786)},
            1261: {"pos": (1515, 906)},
            1262: {"pos": (1515, 906)},
            1264: {"pos": (1670, 1141)},
            1274: {"pos": (1687, 1525)},
        }

        metrics = self.agent._serve_handoff_metrics(1242, (1762, 1023), trace)
        verdict, flags = self.agent._local_review(
            {"start_position": "(1762,1023)", "end_position": "(2000,500)"},
            self.available_metrics,
            self.available_metrics,
            metrics,
        )

        self.assertEqual(verdict, "needs_review")
        self.assertTrue(any("serve handoff" in flag for flag in flags))

    def test_direct_net_reason_requires_net_geometry(self):
        row = {
            "start_position": "(300,450)",
            "end_position": "(300,900)",
            "duration_frames": "80",
            "end_reason": "Ball hit the net",
        }
        verdict, flags = self.agent._local_review(
            row, self.available_metrics, self.available_metrics
        )
        self.assertEqual(verdict, "needs_review")
        self.assertTrue(any("net reason conflicts" in flag for flag in flags))

    def test_compound_net_then_bounce_uses_final_landing_position(self):
        row = {
            "start_position": "(300,450)",
            "end_position": "(300,900)",
            "duration_frames": "80",
            "end_reason": "Serve net then bounce outside right service box",
        }
        verdict, flags = self.agent._local_review(
            row, self.available_metrics, self.available_metrics
        )
        self.assertEqual(verdict, "likely_correct")
        self.assertFalse(any("net reason conflicts" in flag for flag in flags))


if __name__ == "__main__":
    unittest.main()
