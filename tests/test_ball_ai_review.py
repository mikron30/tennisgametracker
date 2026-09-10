import tempfile
import unittest
from pathlib import Path

from PIL import Image
from ball_ai_database import BallAIDatabase
from ball_ai_review import import_reviews


class PatchReviewTests(unittest.TestCase):
    def test_explicit_relabel_removes_opposite_patch_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image, video = root/'image.png', root/'video.mp4'
            Image.new('RGB', (32,32)).save(image)
            video.write_bytes(b'test video placeholder')
            db = BallAIDatabase(root/'db.sqlite')
            try:
                row = dict(image_path=str(image), video_path=str(video), source_frame=5,
                           x=16, y=16, label='negative', note='review evidence', reviewer='test')
                import_reviews(db, [row])
                import_reviews(db, [dict(row, label='positive')])
                self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM hard_negative_patches').fetchone()[0], 0)
                self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM hard_positive_patches').fetchone()[0], 1)
                self.assertEqual(db.connection.execute('SELECT video_path,source_frame FROM reviewed_image_sources').fetchone(),
                                 (str(video.resolve()), 5))
            finally:
                db.close()

    def test_invalid_batch_does_not_import_earlier_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image, video = root/'image.png', root/'video.mp4'
            Image.new('RGB', (32,32)).save(image)
            video.write_bytes(b'test video placeholder')
            db = BallAIDatabase(root/'db.sqlite')
            try:
                row = dict(image_path=str(image), video_path=str(video), source_frame=5,
                           x=16, y=16, label='positive', note='review evidence', reviewer='test')
                with self.assertRaises(ValueError):
                    import_reviews(db, [row, dict(row, x=100)])
                self.assertEqual(db.connection.execute('SELECT COUNT(*) FROM hard_positive_patches').fetchone()[0], 0)
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
