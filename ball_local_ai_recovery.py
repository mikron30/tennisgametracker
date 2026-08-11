"""Guarded local-AI recovery for the HSV tennis-ball tracker.

This module deliberately does not import Torch into the main tracker process.
The tracker keeps using its normal HSV/motion logic.  Only when that logic is
stuck or has selected a player-region artifact do we invoke the dedicated
Python 3.10 local-AI runtime to rank real HSV contour candidates.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from ball_ai_recovery_probe import collect_candidates


class LocalBallAIRecovery:
    """Recover a short, physically continuous ball path after a bad track."""

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        *,
        python_executable: str | Path | None = None,
        work_dir: str | Path = "tmp/local_ai_recovery",
        lookback_frames: int = 4,
        cooldown_frames: int = 18,
        minimum_score: float = 0.985,
        maximum_candidates: int = 96,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.config_path = Path(config_path).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_frames = max(2, int(lookback_frames))
        self.cooldown_frames = max(1, int(cooldown_frames))
        self.minimum_score = float(minimum_score)
        self.maximum_candidates = max(8, int(maximum_candidates))
        self.last_attempt_frame = -1_000_000
        self.last_rejection: Optional[str] = None
        self._sequence = 0
        self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.events_path = self.work_dir / "events.jsonl"

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Local AI model not found: {self.model_path}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Local AI Python runtime not found: {self.python_executable}")

    def ready(self, frame_index: int) -> bool:
        return int(frame_index) - self.last_attempt_frame >= self.cooldown_frames

    @staticmethod
    def _distance(point_a, point_b) -> float:
        return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))

    def _candidate_subset(self, candidates: list[dict], anchor: Optional[tuple[int, int]]) -> list[dict]:
        if anchor is None:
            # Avoid passing thousands of line/background components through the
            # model.  Contour area is only a pre-filter; the model does the rank.
            return candidates[: self.maximum_candidates]
        ordered = sorted(candidates, key=lambda item: self._distance((item["x"], item["y"]), anchor))
        return ordered[: self.maximum_candidates]

    def _score(self, image: np.ndarray, source_frame: int, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        self._sequence += 1
        stem = f"f{int(source_frame):08d}_{self._sequence:04d}"
        image_path = self.work_dir / f"{stem}.jpg"
        candidate_path = self.work_dir / f"{stem}.json"
        try:
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError("Could not write local-AI recovery frame")
            candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
            command = [
                self.python_executable,
                str(Path(__file__).with_name("ball_local_ai.py")),
                "score",
                "--model", str(self.model_path),
                "--image", str(image_path),
                "--candidates", str(candidate_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "local AI scorer failed")
            return list(json.loads(completed.stdout.strip()))
        finally:
            # The event log retains decisions, not a growing copy of every raw
            # recovery frame.  The main dataset exporter is responsible for
            # durable source-frame retention.
            image_path.unlink(missing_ok=True)
            candidate_path.unlink(missing_ok=True)

    def _score_batch(self, samples: list[tuple[np.ndarray, int, list[dict]]]) -> list[list[dict]]:
        """Score a short temporal window with one model/runtime startup."""
        if not samples:
            return []
        self._sequence += 1
        stem = f"batch_{self._sequence:04d}"
        request_path = self.work_dir / f"{stem}.json"
        image_paths: list[Path] = []
        try:
            requests: list[dict] = []
            for index, (image, source_frame, candidates) in enumerate(samples):
                image_path = self.work_dir / f"{stem}_f{int(source_frame):08d}_{index}.jpg"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError("Could not write local-AI recovery frame")
                image_paths.append(image_path)
                requests.append({"image": str(image_path), "candidates": candidates})
            request_path.write_text(json.dumps(requests), encoding="utf-8")
            command = [
                self.python_executable,
                str(Path(__file__).with_name("ball_local_ai.py")),
                "score-batch",
                "--model", str(self.model_path),
                "--requests", str(request_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=75, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "local AI batch scorer failed")
            scored = json.loads(completed.stdout.strip())
            if not isinstance(scored, list) or len(scored) != len(samples):
                raise RuntimeError("local AI batch scorer returned malformed output")
            return [list(group) for group in scored]
        finally:
            request_path.unlink(missing_ok=True)
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)

    def _best_candidate(
        self,
        scored: list[dict],
        anchor: Optional[tuple[int, int]],
        player_zone: Callable[[tuple[int, int]], Optional[str]],
    ) -> Optional[dict]:
        ranked = sorted(scored, key=lambda item: float(item.get("ai_score", 0.0)), reverse=True)
        for candidate in ranked:
            score = float(candidate.get("ai_score", 0.0))
            point = (int(candidate["x"]), int(candidate["y"]))
            zone = player_zone(point)
            # A candidate inside a player silhouette needs almost conclusive
            # image evidence.  This is the crucial guard against replacing a
            # lost ball with torso/racket/shoe colour fragments.
            threshold = 0.9995 if zone is not None else self.minimum_score
            if score < threshold:
                continue
            if anchor is not None and self._distance(point, anchor) > 900.0:
                continue
            candidate = dict(candidate)
            candidate["player_zone"] = zone
            return candidate
        return None

    def _write_event(self, payload: dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def recover(
        self,
        frame_index: int,
        buffered_frames: Iterable[dict],
        *,
        predicted_position: Optional[tuple[int, int]],
        player_zone: Callable[[tuple[int, int]], Optional[str]],
        reason: str,
        force: bool = False,
    ) -> Optional[dict]:
        """Score a short frame history and return the final repaired position.

        The returned path has at least two accepted frames.  A single strong
        contour is useful evidence, but not enough to rewrite the tracker.
        """
        if not force and not self.ready(frame_index):
            return None
        self.last_rejection = None
        self.last_attempt_frame = int(frame_index)
        samples = list(buffered_frames)[-self.lookback_frames :]
        if len(samples) < 2:
            return None

        candidate_samples: list[tuple[dict, list[dict]]] = []
        anchor = predicted_position
        for sample in samples:
            image = sample.get("image")
            if image is None:
                continue
            normal_position = sample.get("normal_position")
            search_anchor = anchor or normal_position
            candidates = collect_candidates(
                image,
                self._config,
                min_area=3.0,
                max_area=2000.0,
                around=search_anchor,
                radius=950.0 if search_anchor is not None else None,
            )
            candidates = self._candidate_subset(candidates, search_anchor)
            candidate_samples.append((sample, candidates))

        try:
            # Keep test/research subclasses that override ``_score`` fully
            # deterministic.  The production scorer batches the window so
            # one recovery does not cold-start the GPU four times.
            if self.__class__ is LocalBallAIRecovery:
                scored_groups = self._score_batch([
                    (sample["image"], int(sample["frame"]), candidates)
                    for sample, candidates in candidate_samples
                ])
            else:
                scored_groups = [
                    self._score(sample["image"], int(sample["frame"]), candidates)
                    for sample, candidates in candidate_samples
                ]
        except Exception as error:
            self.last_rejection = "score-error"
            self._write_event({
                "frame": int(frame_index), "reason": reason, "accepted": False,
                "error": str(error), "stage": "score",
            })
            return None

        accepted: list[dict] = []
        anchor = predicted_position
        diagnostics: list[dict] = []
        for (sample, candidates), scored in zip(candidate_samples, scored_groups):
            selected = self._best_candidate(scored, anchor, player_zone)
            diagnostics.append({
                "frame": int(sample["frame"]), "candidates": len(candidates),
                "selected": selected,
            })
            if selected is None:
                continue
            point = (int(selected["x"]), int(selected["y"]))
            if accepted and self._distance(point, (accepted[-1]["x"], accepted[-1]["y"])) > 900.0:
                continue
            accepted.append({**selected, "frame": int(sample["frame"])})
            anchor = point

        final = accepted[-1] if len(accepted) >= 2 and accepted[-1]["frame"] == int(frame_index) else None
        rejection = None
        if final is not None:
            # A real occluded ball may be inside a player's box at the last
            # contact sample, but the path must enter or leave that box.  A
            # sequence made entirely from torso pixels is simply the player
            # moving through the frame, even when the patch classifier assigns
            # each small yellow/bright fragment an extremely high score.
            if accepted and all(sample.get("player_zone") == "player_body" for sample in accepted):
                final = None
                rejection = "all-player-body-path"

        if final is not None:
            # A single stationary HSV contour can score highly when it looks
            # ball-coloured, but it cannot be the outgoing flight that this
            # repair mechanism is meant to restore.  Do not turn such a
            # player/body/background artifact into a repeated local-AI
            # handoff.  Real terminal stationary balls are handled by the
            # regular endpoint logic rather than recovered as flight.
            path_spread = max(
                self._distance(
                    (int(sample["x"]), int(sample["y"])),
                    (int(accepted[0]["x"]), int(accepted[0]["y"])),
                )
                for sample in accepted
            )
            if path_spread < 4.0:
                final = None
                rejection = "static-recovery-path"
        payload = {
            "frame": int(frame_index), "reason": reason, "accepted": final is not None,
            "predicted_position": predicted_position, "path": accepted, "diagnostics": diagnostics,
            "rejection": rejection,
        }
        self.last_rejection = rejection
        self._write_event(payload)
        return final


def frame_buffer(size: int = 12):
    """Named constructor keeps the tracker implementation readable."""
    return deque(maxlen=max(4, int(size)))
