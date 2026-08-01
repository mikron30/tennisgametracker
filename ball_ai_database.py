"""Local SQLite index for tracker-labelled ball-detection training data.

The database is deliberately local and model-agnostic.  It indexes the image
files written by ``ball_dataset_exporter.py`` and keeps later human/AI review
corrections separate from the original tracker pseudo labels.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


class BallAIDatabase:
    """Idempotently import ball-dataset JSONL manifests into SQLite."""

    SCHEMA_VERSION = 1

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dataset_runs (
                run_id TEXT PRIMARY KEY,
                run_dir TEXT NOT NULL,
                video_path TEXT,
                manifest_path TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_synced_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ball_frames (
                run_id TEXT NOT NULL REFERENCES dataset_runs(run_id) ON DELETE CASCADE,
                source_frame INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                video_path TEXT,
                video_id TEXT,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                ball_x REAL NOT NULL,
                ball_y REAL NOT NULL,
                ball_area REAL,
                radius_hint REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                label_source TEXT NOT NULL,
                label_status TEXT NOT NULL,
                tracking_active INTEGER NOT NULL,
                stuck_frames INTEGER NOT NULL,
                motion_distance_px REAL,
                point_index INTEGER,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (run_id, source_frame)
            );

            CREATE TABLE IF NOT EXISTS label_reviews (
                run_id TEXT NOT NULL,
                source_frame INTEGER NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed'
                    CHECK (review_status IN ('unreviewed', 'accepted', 'rejected', 'corrected')),
                corrected_x REAL,
                corrected_y REAL,
                note TEXT,
                reviewer TEXT,
                reviewed_at TEXT,
                PRIMARY KEY (run_id, source_frame),
                FOREIGN KEY (run_id, source_frame)
                    REFERENCES ball_frames(run_id, source_frame) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ball_frames_status
                ON ball_frames(label_status, tracking_active, source_frame);
            CREATE INDEX IF NOT EXISTS idx_ball_frames_point
                ON ball_frames(run_id, point_index, source_frame);

            CREATE VIEW IF NOT EXISTS training_frames AS
                SELECT
                    f.run_id,
                    f.source_frame,
                    f.image_path,
                    f.image_width,
                    f.image_height,
                    COALESCE(r.corrected_x, f.ball_x) AS ball_x,
                    COALESCE(r.corrected_y, f.ball_y) AS ball_y,
                    f.radius_hint,
                    f.bbox_x1,
                    f.bbox_y1,
                    f.bbox_x2,
                    f.bbox_y2,
                    f.label_status,
                    COALESCE(r.review_status, 'unreviewed') AS review_status,
                    f.point_index
                FROM ball_frames AS f
                LEFT JOIN label_reviews AS r
                    ON r.run_id = f.run_id AND r.source_frame = f.source_frame
                WHERE f.label_status = 'tracked'
                  AND f.tracking_active = 1
                  AND COALESCE(r.review_status, 'unreviewed') != 'rejected';
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_info(key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        self.connection.commit()

    @staticmethod
    def _as_optional_float(value: Any) -> Optional[float]:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bbox_values(ball: dict[str, Any]) -> tuple[Optional[float], ...]:
        bbox = ball.get("bbox_xyxy_hint")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return (None, None, None, None)
        return tuple(BallAIDatabase._as_optional_float(value) for value in bbox)

    def sync_manifest(self, manifest_path: str | Path) -> int:
        """Import one manifest; safe to call repeatedly while it is growing."""
        manifest = Path(manifest_path).resolve()
        run_dir = manifest.parent
        now = datetime.now().isoformat(timespec="seconds")
        run_id = run_dir.name
        imported = 0
        first_row: Optional[dict[str, Any]] = None
        with manifest.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError:
                    # The exporter writes a single JSON line at a time.  A
                    # watcher may see the file while a line is in flight.
                    continue
                if not isinstance(row, dict):
                    continue
                ball = row.get("ball")
                center = ball.get("center") if isinstance(ball, dict) else None
                if not isinstance(center, list) or len(center) != 2:
                    continue
                try:
                    frame_number = int(row["source_frame"])
                    width = int(row["image_width"])
                    height = int(row["image_height"])
                    ball_x = float(center[0])
                    ball_y = float(center[1])
                except (KeyError, TypeError, ValueError):
                    continue
                if first_row is None:
                    first_row = row
                    # Insert the parent before any frame so the foreign-key
                    # guarantee is valid even while a manifest is growing.
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO dataset_runs(
                            run_id, run_dir, video_path, manifest_path, first_seen_at, last_synced_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (run_id, str(run_dir), row.get("video"), str(manifest), now, now),
                    )
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self._bbox_values(ball)
                image_relative = str(row.get("image", ""))
                image_path = str((run_dir / image_relative).resolve())
                self.connection.execute(
                    """
                    INSERT INTO ball_frames (
                        run_id, source_frame, image_path, video_path, video_id,
                        image_width, image_height, ball_x, ball_y, ball_area, radius_hint,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, label_source, label_status,
                        tracking_active, stuck_frames, motion_distance_px, point_index, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, source_frame) DO UPDATE SET
                        image_path=excluded.image_path,
                        video_path=excluded.video_path,
                        video_id=excluded.video_id,
                        image_width=excluded.image_width,
                        image_height=excluded.image_height,
                        ball_x=excluded.ball_x,
                        ball_y=excluded.ball_y,
                        ball_area=excluded.ball_area,
                        radius_hint=excluded.radius_hint,
                        bbox_x1=excluded.bbox_x1,
                        bbox_y1=excluded.bbox_y1,
                        bbox_x2=excluded.bbox_x2,
                        bbox_y2=excluded.bbox_y2,
                        label_source=excluded.label_source,
                        label_status=excluded.label_status,
                        tracking_active=excluded.tracking_active,
                        stuck_frames=excluded.stuck_frames,
                        motion_distance_px=excluded.motion_distance_px,
                        point_index=excluded.point_index,
                        imported_at=excluded.imported_at
                    """,
                    (
                        run_id, frame_number, image_path, row.get("video"), row.get("video_id"),
                        width, height, ball_x, ball_y,
                        self._as_optional_float(ball.get("area")),
                        self._as_optional_float(ball.get("radius_hint")),
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        str(row.get("label_source") or "unknown"),
                        str(row.get("label_status") or "unknown"),
                        1 if row.get("tracking_active") else 0,
                        int(row.get("stuck_frames") or 0),
                        self._as_optional_float(row.get("motion_distance_px")),
                        int(row["point_index"]) if row.get("point_index") is not None else None,
                        now,
                    ),
                )
                imported += 1

        if first_row is not None:
            self.connection.execute(
                """
                INSERT INTO dataset_runs(run_id, run_dir, video_path, manifest_path, first_seen_at, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    run_dir=excluded.run_dir,
                    video_path=excluded.video_path,
                    manifest_path=excluded.manifest_path,
                    last_synced_at=excluded.last_synced_at
                """,
                (run_id, str(run_dir), first_row.get("video"), str(manifest), now, now),
            )
        self.connection.commit()
        return imported

    def sync_dataset_root(self, dataset_root: str | Path) -> dict[str, int]:
        root = Path(dataset_root).expanduser().resolve()
        manifests: Iterable[Path] = sorted(root.glob("*/manifest.jsonl"))
        rows_seen = 0
        for manifest in manifests:
            rows_seen += self.sync_manifest(manifest)
        frame_count = self.connection.execute("SELECT COUNT(*) FROM ball_frames").fetchone()[0]
        training_count = self.connection.execute("SELECT COUNT(*) FROM training_frames").fetchone()[0]
        held_count = self.connection.execute(
            "SELECT COUNT(*) FROM ball_frames WHERE label_status != 'tracked'"
        ).fetchone()[0]
        return {
            "manifests": len(list(root.glob("*/manifest.jsonl"))),
            "rows_seen": rows_seen,
            "frames_indexed": frame_count,
            "initial_training_frames": training_count,
            "held_or_nontracked_frames": held_count,
        }

    def set_review(
        self,
        run_id: str,
        source_frame: int,
        review_status: str,
        *,
        corrected_x: Optional[float] = None,
        corrected_y: Optional[float] = None,
        note: Optional[str] = None,
        reviewer: Optional[str] = None,
    ) -> None:
        """Store a review without ever changing the original tracker label."""
        if review_status not in {"unreviewed", "accepted", "rejected", "corrected"}:
            raise ValueError("Invalid review status")
        self.connection.execute(
            """
            INSERT INTO label_reviews(
                run_id, source_frame, review_status, corrected_x, corrected_y, note, reviewer, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, source_frame) DO UPDATE SET
                review_status=excluded.review_status,
                corrected_x=excluded.corrected_x,
                corrected_y=excluded.corrected_y,
                note=excluded.note,
                reviewer=excluded.reviewer,
                reviewed_at=excluded.reviewed_at
            """,
            (
                run_id, int(source_frame), review_status, corrected_x, corrected_y,
                note, reviewer, datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.connection.commit()


def _write_status(dataset_root: Path, database_path: Path, payload: dict[str, Any]) -> None:
    status = {
        "database": str(database_path),
        "dataset_root": str(dataset_root),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    path = dataset_root / "ball_ai_database_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/sync the local tennis-ball AI SQLite database")
    commands = parser.add_subparsers(dest="command")
    review_parser = commands.add_parser(
        "review",
        help="Record a human-verified label without changing the original tracker pseudo-label",
    )
    review_parser.add_argument("--run-id", required=True)
    review_parser.add_argument("--frame", type=int, required=True)
    review_parser.add_argument(
        "--status", required=True,
        choices=("unreviewed", "accepted", "rejected", "corrected"),
    )
    review_parser.add_argument("--x", type=float)
    review_parser.add_argument("--y", type=float)
    review_parser.add_argument("--note")
    review_parser.add_argument("--reviewer", default="local-ai-validation")
    parser.add_argument("--dataset-root", default="metadata/ball_dataset",
                        help="Folder containing timestamped ball-dataset runs")
    parser.add_argument("--database", default=None,
                        help="SQLite file path (default: <dataset-root>/ball_ai.sqlite)")
    parser.add_argument("--watch", action="store_true",
                        help="Continuously sync manifests as the exporter appends labels")
    parser.add_argument("--interval", type=float, default=10.0,
                        help="Watch polling interval in seconds (default: 10)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    database_path = Path(args.database).expanduser().resolve() if args.database else dataset_root / "ball_ai.sqlite"
    database = BallAIDatabase(database_path)
    try:
        if args.command == "review":
            if args.status == "corrected" and (args.x is None or args.y is None):
                parser.error("review --status corrected requires both --x and --y")
            exists = database.connection.execute(
                "SELECT 1 FROM ball_frames WHERE run_id = ? AND source_frame = ?",
                (args.run_id, args.frame),
            ).fetchone()
            if exists is None:
                parser.error("review target is not present in the database; sync the dataset first")
            database.set_review(
                args.run_id,
                args.frame,
                args.status,
                corrected_x=args.x,
                corrected_y=args.y,
                note=args.note,
                reviewer=args.reviewer,
            )
            print(json.dumps({
                "run_id": args.run_id,
                "source_frame": args.frame,
                "review_status": args.status,
                "corrected_x": args.x,
                "corrected_y": args.y,
            }, sort_keys=True))
            return
        while True:
            summary = database.sync_dataset_root(dataset_root)
            _write_status(dataset_root, database_path, summary)
            print(json.dumps(summary, sort_keys=True))
            if not args.watch:
                break
            time.sleep(max(1.0, float(args.interval)))
    except KeyboardInterrupt:
        print("Local AI database watcher stopped.")
    finally:
        database.close()


if __name__ == "__main__":
    main()
