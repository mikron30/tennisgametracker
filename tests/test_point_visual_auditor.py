import unittest
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

from point_visual_auditor import PointVisualAuditAgent, parse_point


class PointVisualAuditorTests(unittest.TestCase):
    def test_decode_matches_tracker_clock_and_preserves_hevc_reference_frames(self):
        import cv2
        import numpy as np
        for codec in ('MJPG','hevc'):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as directory:
                video=Path(directory)/'video.mp4'
                video.touch()
                seeks=[]
                class Capture:
                    def __init__(self,*args): self.position=0
                    def isOpened(self): return True
                    def release(self): pass
                    def get(self,key):
                        return {cv2.CAP_PROP_FRAME_WIDTH:4,cv2.CAP_PROP_FRAME_HEIGHT:4,
                                cv2.CAP_PROP_FPS:25,
                                cv2.CAP_PROP_FOURCC:cv2.VideoWriter_fourcc(*codec)}.get(key,0)
                    def set(self,key,value):
                        seeks.append(value)
                        self.position=int(value)
                    def grab(self):
                        self.last=self.position
                        self.position+=1
                        return self.position<=200
                    def retrieve(self):
                        return True,np.full((4,4,3),self.last,np.uint8)
                agent=PointVisualAuditAgent(video,'history.csv',provider='local')
                with patch('point_visual_auditor.cv2.VideoCapture',Capture):
                    agent._decode_frames([0,1,2,81,160])
                self.assertEqual(set(agent.frames),{0,1,2,81,160})
                for label in agent.frames:
                    self.assertEqual(int(agent.frames[label][0,0,0]),max(0,label-1))
                self.assertEqual(seeks,[0] if codec=='hevc' else [0,80,159])

    def test_game_one_history_is_included_by_default_and_explicit_filter_skips_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / 'history.csv'
            with history.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'current_score', 'point_index', 'serve_start_frame', 'point_end_frame'])
                writer.writeheader()
                writer.writerow(dict(current_score='0:0 15:0', point_index=1,
                                     serve_start_frame=100, point_end_frame=150))
            agent = PointVisualAuditAgent('video.mp4', history,
                output_dir=Path(directory) / 'audit', provider='local')
            self.assertEqual(len(agent._read_rows()), 1)
            agent.from_game = 2
            with patch.object(agent, '_decode_frames') as decode:
                summary = agent.run()
            decode.assert_not_called()
            self.assertTrue(summary['skipped'])
            self.assertEqual(summary['points_checked'], 0)
            self.assertEqual(summary['needs_review'], 0)
            self.assertIn('from_game=2', summary['reason'])

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

    def test_verified_post_hit_start_requires_contiguous_outgoing_flight(self):
        self.agent.config.update({
            'far_serve_area_points': [[100, 0], [500, 0], [500, 145], [100, 145]],
            'serve_area_points': [[100, 700], [500, 700], [500, 1000], [100, 1000]],
        })
        row = dict(point_index='2', serve_start_frame='100', point_end_frame='111',
                   start_position='(300,158)', end_position='(300,450)',
                   end_reason='Ball hit the net', duration_frames='11',
                   serve_player_side='far', serve_player_source='verified_serve_stance')
        self.agent.tracking_traces = {2: {100 + n: {'pos': (300, 158 + n * 30), 'stuck': 0}
                                          for n in range(4)}}
        verdict, flags = self.agent._local_review(row, self.available_metrics, self.available_metrics)
        self.assertEqual((verdict, flags), ('likely_correct', []))
        for invalid in ('missing_frame', 'static', 'opposite', 'too_far', 'unverified'):
            with self.subTest(invalid=invalid):
                changed = dict(row)
                self.agent.tracking_traces[2] = {
                    100 + n: {'pos': (300, 158 + n * 30), 'stuck': 0} for n in range(4)}
                if invalid == 'missing_frame':
                    del self.agent.tracking_traces[2][102]
                elif invalid in ('static', 'opposite'):
                    for n in range(4):
                        self.agent.tracking_traces[2][100 + n]['pos'] = (300, 158 - n * (10 if invalid == 'opposite' else 0))
                elif invalid == 'too_far':
                    changed['start_position'] = '(300,300)'
                else:
                    changed['serve_player_source'] = 'inferred'
                _, flags = self.agent._local_review(changed, self.available_metrics, self.available_metrics)
                self.assertTrue(any('outside both' in flag for flag in flags))

    def test_terminal_repair_disagreement_and_uncertainty_require_review(self):
        self.agent.tracking_traces = {9: {111: {'pos': (300, 450)}}}
        row = dict(point_index='9', point_end_frame='111', start_position='(300,450)',
                   end_position='(1300,450)', end_reason='Ball bounced twice on court',
                   category='tracking_uncertain')
        verdict, flags = self.agent._local_review(row, self.available_metrics, self.available_metrics)
        self.assertEqual(verdict, 'needs_review')
        self.assertTrue(any('differs from tracked ball by 1000px' in flag for flag in flags))
        self.assertTrue(any('score is provisional' in flag for flag in flags))

    def test_confirmed_serve_uses_real_trace_after_short_context_creation_delay(self):
        self.agent.config['far_serve_area_points'] = [[100,0],[500,0],[500,145],[100,145]]
        row = dict(point_index='2', serve_start_frame='100', serve_player_side='far',
                   serve_player_source='verified_serve_stance')
        self.agent.tracking_traces = {2: {n: dict(pos=(300,158+(n-100)*30),stuck=0)
                                         for n in range(103,107)}}
        self.assertTrue(self.agent._verified_serve_departure(row,(300,158)))
        self.assertFalse(self.agent._verified_serve_departure(row,(100,158)))
        self.agent.tracking_traces[2] = {n: dict(pos=(300,158+(n-100)*30),stuck=0)
                                        for n in range(105,109)}
        self.assertFalse(self.agent._verified_serve_departure(row,(300,158)))

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
