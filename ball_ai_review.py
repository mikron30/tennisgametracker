"""Import explicit visual reviews with source provenance into the local database.

JSON rows: image_path, video_path, source_frame (zero based), label
(positive/negative), x, y, reviewer, note. Never infer labels from tracker status.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image
from ball_ai_database import BallAIDatabase


def import_reviews(database, rows):
    prepared = []
    for row in rows:
        image = Path(row['image_path']).resolve()
        video = Path(row['video_path']).resolve()
        x, y = float(row['x']), float(row['y'])
        if row['label'] not in {'positive', 'negative'}:
            raise ValueError('Labels must be explicitly positive or negative')
        if not row.get('reviewer') or not row.get('note') or not video.is_file():
            raise ValueError('Each review needs a reviewer, evidence note, and source video')
        with Image.open(image) as im:
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < im.width and 0 <= y < im.height):
                raise ValueError('Review center is outside the source image')
        if int(row['source_frame']) < 0:
            raise ValueError('Source frames are zero based')
        prepared.append((row, image, video, x, y))
    for row, image, video, x, y in prepared:
        database.connection.execute(
            'INSERT OR REPLACE INTO reviewed_image_sources(image_path,video_path,source_frame) VALUES (?,?,?)',
            (str(image), str(video), int(row['source_frame'])))
        method = database.add_hard_positive if row['label'] == 'positive' else database.add_hard_negative
        opposite = 'hard_negative_patches' if row['label'] == 'positive' else 'hard_positive_patches'
        database.connection.execute(
            f'DELETE FROM {opposite} WHERE image_path=? AND source_frame=? AND candidate_x=? AND candidate_y=?',
            (str(image), int(row['source_frame']), x, y))
        method(image, row['source_frame'], x, y, note=row['note'], source=row['reviewer'])
    return len(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--reviews', required=True, help='JSON array of explicit visual reviews')
    args = parser.parse_args()
    database = BallAIDatabase(args.database)
    try:
        print(json.dumps({'imported': import_reviews(database, json.loads(Path(args.reviews).read_text()))}))
    finally:
        database.close()


if __name__ == '__main__':
    main()
