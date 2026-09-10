import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ball_local_ai import TrainingRow, reviewed_split, train


class ReviewedTrainingTests(unittest.TestCase):
    def test_split_keeps_all_labels_from_same_video_together(self):
        rows = [TrainingRow(f'{g}.png', 10, 10, g) for g in ['a', 'b', 'c']]
        patches = lambda database, table: [TrainingRow(r.image_path, 80 if table == 'hard_negative_patches' else 12,
                                                      10, r.group) for r in rows]
        with patch('ball_local_ai._training_rows', return_value=rows), patch(
                'ball_local_ai._reviewed_patch_rows', side_effect=patches):
            training, validation = reviewed_split(Path('unused'), 100, 42)
            training_small, validation_small = reviewed_split(Path('unused'), 1, 42)
        train_groups = {row.group for part in training for row in part}
        eval_groups = {row.group for part in validation for row in part}
        self.assertFalse(train_groups & eval_groups)
        self.assertEqual(validation, validation_small)
        self.assertTrue(all(validation))

    def test_unknown_patch_provenance_blocks_training(self):
        with patch('ball_local_ai._training_rows', return_value=[]), patch(
                'ball_local_ai._reviewed_patch_rows', side_effect=[[TrainingRow('x', 1, 2)], []]):
            with self.assertRaisesRegex(RuntimeError, 'provenance'):
                reviewed_split(Path('unused'), 100, 42)

    def test_single_video_is_not_a_held_out_evaluation(self):
        with patch('ball_local_ai._training_rows', return_value=[TrainingRow('x', 1, 2, 'video')]), patch(
                'ball_local_ai._reviewed_patch_rows', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'two source videos'):
                reviewed_split(Path('unused'), 100, 42)

    def test_conflicting_labels_block_training(self):
        row = TrainingRow('x', 1, 2, 'video')
        with patch('ball_local_ai._training_rows', return_value=[row]), patch(
                'ball_local_ai._reviewed_patch_rows', side_effect=[[row], []]):
            with self.assertRaisesRegex(RuntimeError, 'Conflicting'):
                reviewed_split(Path('unused'), 100, 42)

    def test_difficult_positive_is_not_counted_twice(self):
        rows = [TrainingRow(f'{g}.png', 10, 10, g) for g in ['a', 'b']]
        with patch('ball_local_ai._training_rows', return_value=rows), patch(
                'ball_local_ai._reviewed_patch_rows', side_effect=[[], rows]):
            training, validation = reviewed_split(Path('unused'), 100, 42)
        self.assertEqual(training[0], [])
        self.assertEqual(validation[0], [])
        self.assertEqual(len(training[2]) + len(validation[2]), 2)

    def test_existing_model_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder, 'model.pt')
            model.write_bytes(b'old model')
            with patch('ball_local_ai._torch', return_value=(None,) * 5):
                with self.assertRaises(FileExistsError):
                    train(Path('unused'), model, epochs=1, samples=100, batch_size=1, seed=42)
            self.assertEqual(model.read_bytes(), b'old model')


if __name__ == '__main__':
    unittest.main()
