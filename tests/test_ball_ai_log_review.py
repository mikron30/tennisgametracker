import unittest
from ball_ai_log_review import parse_log


class LogReviewTests(unittest.TestCase):
    def test_log_verdict_does_not_verify_ball_labels(self):
        tracks, points = parse_log(
            '[TRACK] f12: pos=(123, 45) vel=7.0px stuck=0\n'
            '[POINT_AUDIT] point=1 start=10 end=15 verdict=likely_correct reason=net\n'
            '[VIDEO_CONTINUE] another video\n'
            '[TRACK] f12: pos=(900, 900) vel=7.0px stuck=0\n')
        self.assertEqual(tracks[12]['source_frame'], 11)
        self.assertEqual(tracks[12]['x'], 123)
        self.assertNotIn('label', tracks[12])
        self.assertEqual(points[0]['log_verdict'], 'likely_correct')


if __name__ == '__main__':
    unittest.main()
