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
        # A player/racket occlusion is the one case where waiting for the normal
        # cooldown is dangerous: the HSV tracker can lock onto the player and
        # then use that false position as the next search anchor.  After a
        # player-region recovery attempt fails, allow a short bounded sequence
        # of immediate retries.  The window is deliberately not extended by
        # every retry, so this cannot turn local AI into the normal tracker.
        self._urgent_retry_until_frame = -1_000_000
        self.last_rejection: Optional[str] = None
        self._sequence = 0
        self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.events_path = self.work_dir / "events.jsonl"

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Local AI model not found: {self.model_path}")
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Local AI Python runtime not found: {self.python_executable}")

    def ready(self, frame_index: int) -> bool:
        frame_index = int(frame_index)
        if frame_index <= int(getattr(self, "_urgent_retry_until_frame", -1_000_000)):
            return True
        return frame_index - self.last_attempt_frame >= self.cooldown_frames

    @staticmethod
    def _distance(point_a, point_b) -> float:
        return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))

    @staticmethod
    def _sample_normal_position(sample: dict) -> Optional[tuple[int, int]]:
        point = sample.get("normal_position")
        if not isinstance(point, (tuple, list)) or len(point) < 2:
            return None
        try:
            return int(point[0]), int(point[1])
        except (TypeError, ValueError):
            return None

    def _build_recovery_corridor(self, samples: list[dict]) -> tuple[dict[int, tuple[int, int]], Optional[float], Optional[dict]]:
        """Detect a catastrophic normal-tracker jump and extrapolate the last sane lane.

        The normal tracker is still useful before an occlusion.  When its recent
        positions show a modest physical chain followed by a several-hundred-pixel
        jump, do not let that bad point become the anchor used to collect/rank AI
        candidates.  Instead extrapolate the last stable velocity for only the
        remainder of this short recovery window.

        This is intentionally conservative: ordinary direction changes, racket
        contacts, and perspective acceleration stay untouched unless the new step
        is both absolutely large and far beyond the immediately preceding speed.
        """
        positioned = []
        for sample in samples:
            pos = self._sample_normal_position(sample)
            if pos is None:
                continue
            try:
                frame = int(sample.get("frame"))
            except (TypeError, ValueError):
                continue
            positioned.append({"frame": frame, "pos": pos})

        if len(positioned) < 3:
            return {}, None, None

        stable = positioned[:2]
        outlier = None
        threshold = None
        for entry in positioned[2:]:
            previous = stable[-1]
            previous_previous = stable[-2]
            previous_speed = self._distance(previous["pos"], previous_previous["pos"])
            step = self._distance(entry["pos"], previous["pos"])
            # Never call a small/medium contact turn an outlier.  The bad case
            # seen in court footage is a 600+ px teleport after a 10-60 px chain.
            threshold = max(150.0, min(520.0, previous_speed * 3.2 + 60.0))
            if step > threshold:
                outlier = {
                    "frame": int(entry["frame"]),
                    "pos": tuple(entry["pos"]),
                    "step": float(step),
                    "threshold": float(threshold),
                }
                break
            stable.append(entry)

        if outlier is None or len(stable) < 2:
            return {}, None, None

        last_good = stable[-1]
        previous_good = stable[-2]
        frame_gap = max(1, int(last_good["frame"]) - int(previous_good["frame"]))
        vx = (float(last_good["pos"][0]) - float(previous_good["pos"][0])) / frame_gap
        vy = (float(last_good["pos"][1]) - float(previous_good["pos"][1])) / frame_gap
        stable_speed = math.hypot(vx, vy)

        # A true post-contact ball can accelerate, so the lane is deliberately
        # wider than one predicted step.  It is still far tighter than the old
        # 900 px acceptance radius that allowed a far-player ball to teleport
        # onto the near player's shoulder.
        corridor_cap = max(180.0, min(420.0, stable_speed * 3.5 + 60.0))
        anchors: dict[int, tuple[int, int]] = {}
        for sample in samples:
            try:
                frame = int(sample.get("frame"))
            except (TypeError, ValueError):
                continue
            if frame < int(outlier["frame"]):
                continue
            dt = frame - int(last_good["frame"])
            anchors[frame] = (
                int(round(float(last_good["pos"][0]) + vx * dt)),
                int(round(float(last_good["pos"][1]) + vy * dt)),
            )

        info = {
            "outlier_frame": int(outlier["frame"]),
            "outlier_position": tuple(outlier["pos"]),
            "outlier_step": float(outlier["step"]),
            "outlier_threshold": float(outlier["threshold"]),
            "last_good_frame": int(last_good["frame"]),
            "last_good_position": tuple(last_good["pos"]),
            "velocity": (float(vx), float(vy)),
            "stable_speed": float(stable_speed),
            "corridor_cap": float(corridor_cap),
        }
        return anchors, corridor_cap, info

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

    def rank_local_roi_candidates(
        self,
        frame_index: int,
        image: np.ndarray,
        *,
        anchor: tuple[int, int],
        radius: float = 25.0,
        maximum_candidates: int = 12,
    ) -> list[dict]:
        """Return model-ranked real HSV contours inside one bounded ROI.

        This function intentionally does not decide whether a candidate is safe
        enough to rewrite the tracker.  The caller can combine the relative AI
        ranking with trajectory, motion, size, and player-context evidence.
        """
        if image is None or anchor is None:
            return []
        roi_radius = max(8.0, float(radius))
        roi_anchor = (int(anchor[0]), int(anchor[1]))
        candidates = collect_candidates(
            image,
            self._config,
            min_area=3.0,
            max_area=2000.0,
            around=roi_anchor,
            radius=roi_radius,
        )
        candidates = self._candidate_subset(candidates, roi_anchor)[
            : max(1, int(maximum_candidates))
        ]
        if not candidates:
            return []

        scored = self._score(image, int(frame_index), candidates)
        ranked = sorted(
            scored,
            key=lambda item: float(item.get("ai_score", 0.0)),
            reverse=True,
        )
        result: list[dict] = []
        for index, candidate in enumerate(ranked):
            item = dict(candidate)
            item["roi_anchor"] = roi_anchor
            item["roi_radius"] = roi_radius
            item["roi_candidates"] = len(ranked)
            item["roi_rank"] = index + 1
            result.append(item)
        return result

    def rank_local_roi_candidate(
        self,
        frame_index: int,
        image: np.ndarray,
        *,
        anchor: tuple[int, int],
        radius: float = 25.0,
        maximum_candidates: int = 12,
    ) -> Optional[dict]:
        """Return the model's top contour inside one tightly bounded ROI."""
        ranked = self.rank_local_roi_candidates(
            frame_index,
            image,
            anchor=anchor,
            radius=radius,
            maximum_candidates=maximum_candidates,
        )
        if not ranked:
            return None
        selected = dict(ranked[0])
        selected["roi_score_margin"] = (
            float(selected.get("ai_score", 0.0)) -
            float(ranked[1].get("ai_score", 0.0))
            if len(ranked) >= 2 else None
        )
        return selected

    def _best_candidate(
        self,
        scored: list[dict],
        anchor: Optional[tuple[int, int]],
        player_zone: Callable[[tuple[int, int]], Optional[str]],
        *,
        max_anchor_distance: float = 900.0,
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
            if anchor is not None and self._distance(point, anchor) > float(max_anchor_distance):
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

        corridor_anchors, corridor_cap, corridor_info = self._build_recovery_corridor(samples)
        if corridor_info is not None:
            print(
                f"[LOCAL_AI_CORRIDOR] f{int(frame_index)}: ignoring normal jump "
                f"f{corridor_info['outlier_frame']} {corridor_info['outlier_position']} "
                f"step={corridor_info['outlier_step']:.1f}px > "
                f"{corridor_info['outlier_threshold']:.1f}px; "
                f"last_good={corridor_info['last_good_position']} "
                f"speed={corridor_info['stable_speed']:.1f}px "
                f"corridor={corridor_info['corridor_cap']:.1f}px"
            )

        candidate_samples: list[tuple[dict, list[dict]]] = []
        for sample in samples:
            image = sample.get("image")
            if image is None:
                continue
            normal_position = self._sample_normal_position(sample)
            try:
                sample_frame = int(sample.get("frame"))
            except (TypeError, ValueError):
                continue

            if corridor_info is not None:
                search_anchor = corridor_anchors.get(
                    sample_frame,
                    normal_position or predicted_position,
                )
                if sample_frame >= int(corridor_info["outlier_frame"]):
                    search_radius = max(
                        260.0,
                        min(950.0, float(corridor_cap or 420.0) * 1.8),
                    )
                else:
                    search_radius = 950.0
            else:
                search_anchor = predicted_position or normal_position
                search_radius = 950.0 if search_anchor is not None else None

            candidates = collect_candidates(
                image,
                self._config,
                min_area=3.0,
                max_area=2000.0,
                around=search_anchor,
                radius=search_radius if search_anchor is not None else None,
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
                "corridor": corridor_info,
            })
            return None

        accepted: list[dict] = []
        anchor = predicted_position
        diagnostics: list[dict] = []
        for (sample, candidates), scored in zip(candidate_samples, scored_groups):
            sample_frame = int(sample["frame"])
            normal_position = self._sample_normal_position(sample)
            restricted = (
                corridor_info is not None and
                sample_frame >= int(corridor_info["outlier_frame"])
            )
            if corridor_info is not None:
                sample_anchor = corridor_anchors.get(
                    sample_frame,
                    normal_position or anchor,
                )
            else:
                sample_anchor = anchor

            max_anchor_distance = float(corridor_cap or 900.0) if restricted else 900.0
            selected = self._best_candidate(
                scored,
                sample_anchor,
                player_zone,
                max_anchor_distance=max_anchor_distance,
            )
            diagnostics.append({
                "frame": sample_frame,
                "candidates": len(candidates),
                "anchor": sample_anchor,
                "anchor_cap": max_anchor_distance,
                "selected": selected,
            })
            if selected is None:
                continue

            point = (int(selected["x"]), int(selected["y"]))
            if accepted:
                step = self._distance(
                    point,
                    (accepted[-1]["x"], accepted[-1]["y"]),
                )
                if restricted:
                    max_step = max(
                        220.0,
                        min(520.0, float(corridor_cap or 420.0) * 1.35),
                    )
                else:
                    max_step = 900.0
                if step > max_step:
                    diagnostics[-1]["selected_rejected"] = (
                        f"path-step {step:.1f}px > {max_step:.1f}px"
                    )
                    continue

            accepted.append({**selected, "frame": sample_frame})
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

        if final is not None and corridor_info is not None:
            # During a detected teleport, a huge contour-size explosion is
            # another strong sign that the model has latched onto a shirt,
            # shoulder, or racket fragment just outside the player bbox.
            previous_areas = [
                float(sample.get("area", 0.0) or 0.0)
                for sample in accepted[:-1]
                if float(sample.get("area", 0.0) or 0.0) > 0.0
            ]
            final_area = float(final.get("area", 0.0) or 0.0)
            if previous_areas:
                baseline_area = float(np.median(previous_areas))
                area_cap = max(260.0, baseline_area * 10.0)
                if final_area > area_cap:
                    rejection = (
                        f"recovery-area-explosion:{final_area:.1f}>"
                        f"{area_cap:.1f}"
                    )
                    final = None

        # If the normal tracker has entered a player/racket region and AI could
        # not yet form a safe temporal path, the next few frames are exactly
        # where the real ball is most likely to emerge from the occlusion.
        # Do not let the ordinary 18-frame cooldown blind us during that short
        # handoff.  Arm this only once; subsequent failures inside the window
        # do not extend it indefinitely.
        if final is not None:
            self._urgent_retry_until_frame = -1_000_000
        elif str(reason).startswith("player-region:"):
            current_deadline = int(getattr(self, "_urgent_retry_until_frame", -1_000_000))
            if current_deadline < int(frame_index):
                self._urgent_retry_until_frame = int(frame_index) + 8
                print(
                    f"[LOCAL_AI_RAPID_RETRY] f{int(frame_index)}: "
                    f"{reason} recovery not safe yet; retry window through "
                    f"f{self._urgent_retry_until_frame}"
                )

        payload = {
            "frame": int(frame_index), "reason": reason, "accepted": final is not None,
            "predicted_position": predicted_position, "path": accepted, "diagnostics": diagnostics,
            "rejection": rejection, "corridor": corridor_info,
        }
        self.last_rejection = rejection
        self._write_event(payload)
        return final


def frame_buffer(size: int = 12):
    """Named constructor keeps the tracker implementation readable."""
    return deque(maxlen=max(4, int(size)))
