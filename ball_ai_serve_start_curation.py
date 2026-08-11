"""Add visually reviewed false serve-start detections to the local AI DB.

The tracker model is a candidate scorer: it needs explicit examples of
ball-coloured/racket-adjacent blobs that are not tennis balls.  This small,
repeatable curation command materialises the reviewed 8012 caught-toss false
flight and records its detected centres as hard negatives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ball_ai_database import BallAIDatabase


REVIEWED_FALSE_8012 = (
    # The ball was tossed and caught: this later apparent netward streak is
    # HSV/background contamination, not a legal service flight.
    (8023, 2083, 344),
    (8024, 2104, 394),
    (8025, 2125, 446),
)

REVIEWED_TRUE_8527 = (
    # The subsequent second serve really travels to the net.  These matching
    # positives keep serve-start hard-negative training from rejecting all
    # bright, blurred far-side flight patches.
    (8533, 1879, 389),
    (8534, 1867, 407),
    (8535, 1859, 394),
    (8536, 1845, 383),
)


def curate_false_8012(video: Path, dataset_root: Path) -> int:
    output_dir = dataset_root / "manual_reviews" / "serve_start_false_8012"
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_dir = dataset_root / "manual_reviews" / "serve_start_true_8527"
    positive_dir.mkdir(parents=True, exist_ok=True)
    database = BallAIDatabase(dataset_root / "ball_ai.sqlite")
    written = 0
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    try:
        for frame_number, x, y in REVIEWED_FALSE_8012:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to decode frame {frame_number}")
            image_path = output_dir / f"frame_{frame_number:05d}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Unable to save {image_path}")
            database.add_hard_negative(
                image_path,
                frame_number,
                x,
                y,
                note="Reviewed caught toss at f8012; false serve/net trajectory",
                source="serve-start-review",
            )
            written += 1
        for frame_number, x, y in REVIEWED_TRUE_8527:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to decode frame {frame_number}")
            image_path = positive_dir / f"frame_{frame_number:05d}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Unable to save {image_path}")
            database.add_hard_positive(
                image_path,
                frame_number,
                x,
                y,
                note="Reviewed genuine f8527 second-serve net-fault flight",
                source="serve-start-review",
            )
            written += 1
    finally:
        capture.release()
        database.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate reviewed false serve-start candidates")
    parser.add_argument(
        "--video",
        default="04_Left_Court_tennis_20260623220627_20260623222559_1274169779.mp4",
    )
    parser.add_argument("--dataset-root", default="metadata/ball_dataset")
    args = parser.parse_args()
    written = curate_false_8012(Path(args.video).resolve(), Path(args.dataset_root).resolve())
    print(f"Added {written} reviewed serve-start patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
