"""Lightweight player/racket context tracking for TennisGameTracker.

This module intentionally uses only OpenCV/Numpy.  It is a context tracker,
not a replacement for the ball tracker: detections are allowed to disappear,
and every classification carries a confidence score.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class PlayerTrack:
    side: str
    bbox: Optional[BBox] = None
    center: Optional[Point] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    last_frame: int = -1
    visible: bool = False
    head: Optional[Point] = None
    shoes: Optional[Point] = None
    racket: Optional[Dict] = None
    racket_side: Optional[str] = None

    def as_dict(self) -> Dict:
        return {
            "side": self.side,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "center": list(self.center) if self.center is not None else None,
            "velocity": [round(float(self.velocity[0]), 2), round(float(self.velocity[1]), 2)],
            "confidence": round(float(self.confidence), 3),
            "last_frame": int(self.last_frame),
            "visible": bool(self.visible),
            "head": list(self.head) if self.head is not None else None,
            "shoes": list(self.shoes) if self.shoes is not None else None,
            "racket": self.racket,
            "racket_side": self.racket_side,
        }


class PlayerRacketTracker:
    """Track the near/far player and racket using HOG plus motion fallback."""

    def __init__(
        self,
        frame_shape: Optional[Sequence[int]] = None,
        net_y: Optional[float] = None,
        detection_interval: int = 5,
        profile_path: Optional[str] = None,
        learning_enabled: bool = True,
        court_x_range: Optional[Tuple[float, float]] = None,
    ):
        self.frame_shape = tuple(frame_shape[:2]) if frame_shape is not None else None
        self.net_y = float(net_y) if net_y is not None else None
        self.court_x_range = tuple(court_x_range) if court_x_range is not None else None
        self.court_region = None
        self.detection_interval = max(1, int(detection_interval))
        self.profile_path = profile_path
        self.learning_enabled = bool(learning_enabled)
        self.tracks = {
            "near": PlayerTrack("near"),
            "far": PlayerTrack("far"),
        }
        self.prev_gray = None
        self.last_frame = None
        self.last_detections: List[Dict] = []
        self.point_events: List[Dict] = []
        self.point_positions: List[Dict] = []
        self.profile = {
            "version": 1,
            "shots": {"P1": {"forehand": 0, "backhand": 0, "unknown": 0},
                      "P2": {"forehand": 0, "backhand": 0, "unknown": 0}},
            "racket_sides": {"P1": {"left": 0, "right": 0}, "P2": {"left": 0, "right": 0}},
            "serve_sides": {"P1": {"left": 0, "right": 0}, "P2": {"left": 0, "right": 0}},
            "net_contacts": {"P1": 0, "P2": 0},
        }
        self._load_profile()
        self._hog = None
        self._last_detection_frame = -1000000

    def _load_profile(self):
        if not self.profile_path or not os.path.exists(self.profile_path):
            return
        try:
            with open(self.profile_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                for key in self.profile:
                    if key in payload and isinstance(payload[key], dict):
                        self.profile[key].update(payload[key])
        except Exception:
            # A corrupt learning file must never stop ball analysis.
            return

    def save_profile(self):
        if not self.learning_enabled or not self.profile_path:
            return
        try:
            directory = os.path.dirname(os.path.abspath(self.profile_path))
            os.makedirs(directory, exist_ok=True)
            temporary = self.profile_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self.profile, handle, indent=2, sort_keys=True)
            os.replace(temporary, self.profile_path)
        except Exception:
            return

    def set_geometry(self, frame_shape: Sequence[int], net_y: Optional[float] = None):
        self.frame_shape = tuple(frame_shape[:2])
        if net_y is not None:
            self.net_y = float(net_y)

    def set_court_x_range(self, x_range: Optional[Tuple[float, float]]):
        self.court_x_range = tuple(x_range) if x_range is not None else None

    def set_court_region(self, region: Optional[Dict]):
        """Set the main-court sideline region used for player candidates."""
        self.court_region = region if isinstance(region, dict) else None

    def _center_is_in_court(self, center: Point, frame_width: Optional[int] = None) -> bool:
        width = int(frame_width or (self.frame_shape[1] if self.frame_shape else 0))
        margin = max(80.0, width * 0.02) if width > 0 else 80.0
        if self.court_region:
            y = float(center[1])
            y_min = float(self.court_region.get("y_min", 0.0))
            y_max = float(self.court_region.get("y_max", float(self.frame_shape[0]) if self.frame_shape else 1e9))
            if y < y_min - margin or y > y_max + margin:
                return False
            left_line = self.court_region.get("left") or {}
            right_line = self.court_region.get("right") or {}
            try:
                left_x = float(left_line["a"]) * y + float(left_line["b"])
                right_x = float(right_line["a"]) * y + float(right_line["b"])
                return left_x - margin <= float(center[0]) <= right_x + margin
            except (KeyError, TypeError, ValueError):
                pass
        if self.court_x_range is None:
            return True
        left, right = self.court_x_range
        return left - margin <= float(center[0]) <= right + margin

    def begin_point(self):
        self.point_events = []
        self.point_positions = []

    @staticmethod
    def _box_center(box: BBox) -> Point:
        x, y, w, h = box
        return int(round(x + w * 0.5)), int(round(y + h * 0.5))

    @staticmethod
    def _clip_box(box: BBox, shape: Sequence[int]) -> BBox:
        h, w = shape[:2]
        x, y, bw, bh = [int(v) for v in box]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        bw = max(1, min(w - x, bw))
        bh = max(1, min(h - y, bh))
        return x, y, bw, bh

    @staticmethod
    def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
        px, py = float(point[0]), float(point[1])
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-6:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def _ensure_hog(self):
        if self._hog is None:
            try:
                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self._hog = False
        return self._hog if self._hog is not False else None

    def _hog_detections(self, frame: np.ndarray) -> List[Dict]:
        hog = self._ensure_hog()
        if hog is None:
            return []
        height, width = frame.shape[:2]
        scale = min(1.0, 1100.0 / max(width, 1))
        scaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 0.999 else frame
        try:
            boxes, weights = hog.detectMultiScale(
                scaled,
                # The far player is small in the 4K frame. A finer stride and
                # slightly lower threshold let us see that player without
                # changing the ball detector; court-region gating removes
                # adjacent-court detections afterward.
                winStride=(4, 4),
                padding=(8, 8),
                scale=1.03,
                hitThreshold=-0.80,
            )
        except Exception:
            return []
        result = []
        for box, weight in zip(boxes, weights):
            x, y, w, h = [int(round(float(v) / scale)) for v in box]
            if h < max(70, int(height * 0.055)) or w < 20:
                continue
            if self.court_x_range is not None:
                left, right = self.court_x_range
                # Keep HOG false positives on fences/walls outside the court
                # from becoming a slowly drifting player track.
                margin = max(80.0, width * 0.02)
                center_x = x + w * 0.5
                if not (left - margin <= center_x <= right + margin):
                    continue
            result.append({"bbox": self._clip_box((x, y, w, h), frame.shape),
                           "score": float(np.asarray(weight).reshape(-1)[0]) if np.asarray(weight).size else 0.0,
                           "source": "hog"})
        return result

    def _motion_detections(self, frame: np.ndarray) -> List[Dict]:
        if self.prev_gray is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_gray, gray)
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        result = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            if area < max(1500.0, width * height * 0.00012):
                continue
            if h < max(80, int(height * 0.06)) or h / max(w, 1) < 0.45:
                continue
            if w > width * 0.75 or h > height * 0.8:
                continue
            result.append({"bbox": self._clip_box((x, y, w, h), frame.shape),
                           "score": min(3.0, area / max(width * height * 0.01, 1.0)),
                           "source": "motion"})
        return result

    def _side_for_box(self, box: BBox, frame_height: int) -> Optional[str]:
        center_y = self._box_center(box)[1]
        net_y = self.net_y if self.net_y is not None else frame_height * 0.5
        if center_y >= net_y * 0.92:
            return "near"
        if center_y <= net_y * 1.08:
            return "far"
        return None

    def _candidate_score(self, candidate: Dict, side: str) -> float:
        box = candidate["bbox"]
        x, y, w, h = box
        cx, cy = self._box_center(box)
        score = float(candidate.get("score", 0.0))
        track = self.tracks[side]
        if track.bbox is not None:
            old_cx, old_cy = track.center or self._box_center(track.bbox)
            distance = math.hypot(cx - old_cx, cy - old_cy)
            score += max(0.0, 2.2 - distance / max(120.0, h * 0.8))
            old_area = track.bbox[2] * track.bbox[3]
            score += max(0.0, 1.0 - abs(math.log(max(1.0, (w * h) / max(old_area, 1)))) )
        # A standing player is taller than it is wide; slightly prefer that shape.
        score += min(2.0, h / max(w, 1))
        if side == "far":
            score += max(0.0, 1.0 - abs(cy - (self.net_y or 0.5 * self.frame_shape[0])) / max(1.0, self.frame_shape[0] * 0.35))
        return score

    def _candidate_is_continuous(self, candidate: Dict, track: PlayerTrack,
                                 frame_index: int) -> bool:
        """Reject a player detection that would teleport the track.

        HOG and motion contours can occasionally produce a convincing box on
        the opposite side of the image. A player may move between detector
        updates, but not by an arbitrary frame-sized jump, so keep the prior
        track when the candidate is both distant and non-overlapping.
        """
        candidate_box = candidate["bbox"]
        candidate_center = self._box_center(candidate_box)
        if not self._center_is_in_court(candidate_center):
            return False
        if track.bbox is None or track.center is None or track.last_frame < 0:
            return True
        gap = max(1, int(frame_index) - int(track.last_frame))
        old_x, old_y = track.center
        distance = math.hypot(candidate_center[0] - old_x, candidate_center[1] - old_y)
        old_height = max(1, track.bbox[3])
        # Scale for perspective, but cap the allowed step so a missed update
        # cannot eventually authorize a teleport across the court.
        max_distance = max(80.0, min(220.0, old_height * 0.45 + gap * 18.0))
        # Overlap alone is not sufficient: a very wide false box can overlap
        # the old box while its center has still teleported.
        return distance <= max_distance

    def _candidate_is_full_player_reacquisition(self, candidate: Dict,
                                                 track: PlayerTrack,
                                                 frame: np.ndarray,
                                                 side: str,
                                                 frame_index: int) -> bool:
        """Allow a safe re-lock when a stale 4K track lost the real player.

        On the 4K court camera a player can move from one lower-court side to
        the other between detector updates.  The normal continuity gate must
        reject arbitrary teleports, but that gate also rejected the clearly
        visible full player while retaining an old box on the opposite side.
        Reacquisition is deliberately narrow: it is only enabled for a 4K
        frame, a lower/upper-court full-height player-shaped box, and a strong
        HOG or large motion detection inside the fitted singles court.
        """
        if track.bbox is None or track.center is None or frame is None:
            return False
        height, width = frame.shape[:2]
        if width < 1920 or height < 1080:
            return False
        box = candidate.get("bbox")
        if not box or len(box) != 4:
            return False
        cx, cy = self._box_center(box)
        if not self._center_is_in_court((cx, cy), frame_width=width):
            return False
        x, y, w, h = [int(v) for v in box]
        if h < int(height * 0.18) or h / max(float(w), 1.0) < 1.15:
            return False
        # Do not relock on an upper-court/fence object for a near track (or
        # vice versa).  The small overlap around the net is intentionally
        # excluded because it is where HOG produces the most false boxes.
        if side == "near":
            if cy < height * 0.62 or y + h < height * 0.82:
                return False
        elif side == "far":
            if cy > height * 0.42 or y > height * 0.25:
                return False
        gap = max(1, int(frame_index) - int(track.last_frame))
        old_x, old_y = track.center
        distance = math.hypot(cx - old_x, cy - old_y)
        old_height = max(1, track.bbox[3])
        max_distance = max(80.0, min(220.0, old_height * 0.45 + gap * 18.0))
        if distance <= max_distance:
            return False
        source = str(candidate.get("source", "")).lower()
        detection_score = float(candidate.get("score", 0.0))
        if source == "hog" and detection_score >= -0.85:
            return True
        if source == "motion":
            area_ratio = float(w * h) / max(float(width * height), 1.0)
            return area_ratio >= 0.015 and h >= int(height * 0.28)
        return False

    def _detect_racket(self, frame: np.ndarray, track: PlayerTrack) -> Optional[Dict]:
        if track.bbox is None:
            return None
        x, y, w, h = track.bbox
        pad_x = int(w * 0.55)
        pad_y = int(h * 0.25)
        rx1 = max(0, x - pad_x)
        ry1 = max(0, y + int(h * 0.18) - pad_y)
        rx2 = min(frame.shape[1], x + w + pad_x)
        ry2 = min(frame.shape[0], y + int(h * 0.78) + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        min_len = max(18, int(min(w, h) * 0.16))
        max_gap = max(8, int(min(w, h) * 0.06))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=18,
                                minLineLength=min_len, maxLineGap=max_gap)
        if lines is None:
            return None
        candidates = []
        body_cx = x + w * 0.5
        hand_y_min = y + h * 0.25
        hand_y_max = y + h * 0.78
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(v) for v in line]
            gx1, gy1, gx2, gy2 = lx1 + rx1, ly1 + ry1, lx2 + rx1, ly2 + ry1
            length = math.hypot(gx2 - gx1, gy2 - gy1)
            center_x = (gx1 + gx2) * 0.5
            center_y = (gy1 + gy2) * 0.5
            if not (hand_y_min <= center_y <= hand_y_max):
                continue
            if length < min_len or length > max(w * 2.8, 260.0):
                continue
            # A racket usually extends to one side of the torso.
            side_distance = abs(center_x - body_cx)
            if side_distance < w * 0.08:
                continue
            candidates.append((length + side_distance * 0.25, (gx1, gy1, gx2, gy2)))
        if not candidates:
            return None
        best_score, (gx1, gy1, gx2, gy2) = max(candidates, key=lambda item: item[0])
        center = (int(round((gx1 + gx2) * 0.5)), int(round((gy1 + gy2) * 0.5)))
        side = "right" if center[0] >= body_cx else "left"
        return {
            "line": [gx1, gy1, gx2, gy2],
            "center": list(center),
            "length": round(math.hypot(gx2 - gx1, gy2 - gy1), 1),
            "side": side,
            "confidence": round(min(1.0, max(0.0, (best_score / max(w * 2.0, 1.0)) * 0.35)), 3),
        }

    def _assign(self, detections: Iterable[Dict], frame: np.ndarray, frame_index: int):
        candidates = list(detections)
        by_side = {"near": [], "far": []}
        for candidate in candidates:
            side = self._side_for_box(candidate["bbox"], frame.shape[0])
            if side is not None:
                by_side[side].append(candidate)
        for side, track in self.tracks.items():
            choices = [
                candidate for candidate in by_side[side]
                if self._candidate_is_continuous(candidate, track, frame_index)
                or self._candidate_is_full_player_reacquisition(
                    candidate, track, frame, side, frame_index
                )
            ]
            if choices:
                # When re-locking, a tall HOG person box is more precise than
                # the broad motion contour generated by the same movement.
                # Prefer that box while keeping the normal score ordering for
                # ordinary continuous updates.
                reacquisition_choices = [
                    candidate for candidate in choices
                    if self._candidate_is_full_player_reacquisition(
                        candidate, track, frame, side, frame_index
                    )
                ]
                if reacquisition_choices:
                    def reacquisition_score(candidate):
                        x, y, w, h = candidate["bbox"]
                        score = self._candidate_score(candidate, side)
                        if str(candidate.get("source", "")).lower() == "hog":
                            score += 2.5
                        elif w > h * 1.10:
                            score -= 0.5
                        return score
                    chosen = max(reacquisition_choices, key=reacquisition_score)
                else:
                    chosen = max(choices, key=lambda item: self._candidate_score(item, side))
                box = chosen["bbox"]
                new_center = self._box_center(box)
                old_center = track.center
                if old_center is not None:
                    velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                else:
                    velocity = (0.0, 0.0)
                track.bbox = box
                track.center = new_center
                track.velocity = velocity
                track.confidence = min(1.0, max(0.05, 0.45 + float(chosen.get("score", 0.0)) * 0.12))
                track.last_frame = int(frame_index)
                track.visible = True
                x, y, w, h = box
                track.head = (int(round(x + w * 0.5)), int(round(y + h * 0.14)))
                track.shoes = (int(round(x + w * 0.5)), int(round(y + h * 0.96)))
                track.racket = self._detect_racket(frame, track)
                track.racket_side = track.racket.get("side") if track.racket else track.racket_side
            else:
                track.visible = False
                track.confidence *= 0.90
                if track.confidence < 0.12:
                    track.bbox = None
                    track.center = None
                    track.racket = None
                    track.racket_side = None

    def update(self, frame: np.ndarray, frame_index: int, net_y: Optional[float] = None) -> Dict:
        if frame is None or frame.size == 0:
            return self.snapshot()
        self.set_geometry(frame.shape, net_y=net_y if net_y is not None else self.net_y)
        should_detect = (
            self.last_frame is None or
            int(frame_index) - self._last_detection_frame >= self.detection_interval
        )
        if should_detect:
            detections = self._hog_detections(frame)
            detections.extend(self._motion_detections(frame))
            self.last_detections = detections
            self._assign(detections, frame, frame_index)
            self._last_detection_frame = int(frame_index)
        else:
            for track in self.tracks.values():
                track.visible = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.last_frame = frame_index
        self.point_positions.append(self.snapshot())
        if len(self.point_positions) > 120:
            self.point_positions = self.point_positions[-120:]
        return self.snapshot()

    def _track_for_point(self, point: Optional[Point]) -> Optional[PlayerTrack]:
        if point is None:
            return None
        px, py = point
        best = None
        best_score = float("inf")
        for track in self.tracks.values():
            if track.bbox is None or track.center is None:
                continue
            x, y, w, h = track.bbox
            expanded = (x - int(w * 0.45), y - int(h * 0.25), int(w * 1.9), int(h * 1.5))
            ex, ey, ew, eh = expanded
            if ex <= px <= ex + ew and ey <= py <= ey + eh:
                score = math.hypot(px - track.center[0], py - track.center[1]) / max(1.0, h)
                if score < best_score:
                    best, best_score = track, score
        return best

    def classify_stroke(self, point: Optional[Point], player_name: str = "") -> Dict:
        track = self._track_for_point(point)
        if track is None or point is None:
            return {"stroke": "unknown", "confidence": 0.0, "player_side": None, "racket_side": None}
        x, y, w, h = track.bbox
        body_x = x + w * 0.5
        ball_side = "right" if point[0] >= body_x else "left"
        racket_side = track.racket_side
        if racket_side is None:
            return {"stroke": "unknown", "confidence": 0.2, "player_side": track.side, "racket_side": None}
        stroke = "forehand" if ball_side == racket_side else "backhand"
        confidence = min(1.0, 0.45 + track.confidence * 0.4)
        if track.racket and track.racket.get("confidence", 0.0) > 0.4:
            confidence = min(1.0, confidence + 0.15)
        return {
            "stroke": stroke,
            "confidence": round(confidence, 3),
            "player_side": track.side,
            "racket_side": racket_side,
            "player_bbox": list(track.bbox),
            "racket": track.racket,
        }

    def record_shot(self, frame_index: int, point: Optional[Point], player_name: str = "", label: Optional[str] = None) -> Dict:
        context = self.classify_stroke(point, player_name=player_name)
        event = {
            "frame": int(frame_index),
            "position": list(point) if point is not None else None,
            "player": player_name or "",
            "player_side": context.get("player_side"),
            "stroke": context.get("stroke", "unknown"),
            "confidence": context.get("confidence", 0.0),
            "racket_side": context.get("racket_side"),
            "label": label or "",
            "player_bbox": context.get("player_bbox"),
            "racket": context.get("racket"),
        }
        self.point_events.append(event)
        if self.learning_enabled:
            key = player_name if player_name in ("P1", "P2") else ("P1" if context.get("player_side") == "near" else "P2")
            stroke = event["stroke"] if event["stroke"] in ("forehand", "backhand") else "unknown"
            self.profile.setdefault("shots", {}).setdefault(key, {}).setdefault(stroke, 0)
            self.profile["shots"][key][stroke] += 1
            racket_side = event.get("racket_side")
            if racket_side in ("left", "right"):
                self.profile.setdefault("racket_sides", {}).setdefault(key, {}).setdefault(racket_side, 0)
                self.profile["racket_sides"][key][racket_side] += 1
            if label and "net" in label.lower():
                self.profile.setdefault("net_contacts", {}).setdefault(key, 0)
                self.profile["net_contacts"][key] += 1
            if len(self.point_events) % 4 == 0:
                self.save_profile()
        return event

    def _server_track_for_ball(self, point: Optional[Point]) -> Optional[PlayerTrack]:
        """Choose the server whose body is below the serve-ball position."""
        if point is None:
            return None
        px, py = point
        choices = []
        for track in self.tracks.values():
            if track.bbox is None or track.center is None:
                continue
            x, y, w, h = track.bbox
            cx, cy = track.center
            vertical_gap = float(cy - py)
            if vertical_gap <= max(20.0, h * 0.05):
                continue
            horizontal_gap = abs(float(cx - px))
            horizontal_limit = max(180.0, w * 1.60)
            if horizontal_gap > horizontal_limit:
                continue
            # Prefer a player directly below the ball, with a small penalty
            # for stale detections. The expected server name is supplied by
            # the analyzer separately; geometry decides which body is serving.
            score = (
                horizontal_gap / max(float(w), 1.0) +
                abs(vertical_gap - h * 0.70) / max(float(h), 1.0) * 0.35 +
                (0.25 if not track.visible else 0.0) -
                min(0.20, max(0.0, track.confidence - 0.5) * 0.20)
            )
            choices.append((score, track, vertical_gap, horizontal_gap))
        if not choices:
            return None
        return min(choices, key=lambda item: item[0])[1]

    def _server_detection_for_ball(self, point: Optional[Point]) -> Optional[Dict]:
        """Find a raw player box below the serve ball, before track assignment."""
        if point is None:
            return None
        px, py = point
        choices = []
        for detection in self.last_detections:
            box = detection.get("bbox") if isinstance(detection, dict) else None
            if not box:
                continue
            center = self._box_center(box)
            if not self._center_is_in_court(center):
                continue
            x, y, w, h = box
            vertical_gap = float(center[1] - py)
            if vertical_gap <= max(20.0, h * 0.05):
                continue
            horizontal_gap = abs(float(center[0] - px))
            if horizontal_gap > max(180.0, w * 1.60):
                continue
            side = self._side_for_box(box, self.frame_shape[0] if self.frame_shape else int(py * 2))
            if side is None:
                continue
            score = (
                horizontal_gap / max(float(w), 1.0) +
                abs(vertical_gap - h * 0.70) / max(float(h), 1.0) * 0.35 -
                min(0.20, max(0.0, float(detection.get("score", 0.0)) * 0.05))
            )
            choices.append((score, detection, side, center, vertical_gap, horizontal_gap))
        if not choices:
            return None
        _, detection, side, center, vertical_gap, horizontal_gap = min(choices, key=lambda item: item[0])
        result = dict(detection)
        result.update({
            "side": side,
            "center": list(center),
            "vertical_gap": round(vertical_gap, 1),
            "horizontal_gap": round(horizontal_gap, 1),
            "source": "serve_detection_below_ball",
        })
        return result

    def _receiver_detection_for_server(self, server_side: Optional[str]) -> Optional[Dict]:
        """Choose the receiver on the opposite side of the net."""
        if server_side not in ("near", "far"):
            return None
        desired_side = "far" if server_side == "near" else "near"
        choices = []
        for detection in self.last_detections:
            box = detection.get("bbox") if isinstance(detection, dict) else None
            if not box:
                continue
            center = self._box_center(box)
            if not self._center_is_in_court(center):
                continue
            side = self._side_for_box(box, self.frame_shape[0] if self.frame_shape else 2160)
            if side != desired_side:
                continue
            _, _, w, h = box
            shape_penalty = abs((float(h) / max(float(w), 1.0)) - 2.0) * 0.12
            score = shape_penalty - float(detection.get("score", 0.0)) * 0.20
            choices.append((score, detection, side, center))
        if not choices:
            for track in self.tracks.values():
                if track.side != desired_side or track.bbox is None or track.center is None:
                    continue
                if not self._center_is_in_court(track.center):
                    continue
                _, _, w, h = track.bbox
                score = abs((float(h) / max(float(w), 1.0)) - 2.0) * 0.12 - track.confidence * 0.20
                choices.append((score, track, desired_side, track.center))
        if not choices:
            return None
        _, selected, side, center = min(choices, key=lambda item: item[0])
        if isinstance(selected, PlayerTrack):
            return {
                "bbox": list(selected.bbox),
                "center": list(center),
                "side": side,
                "source": "tracked_receiver_opposite_side",
            }
        result = dict(selected)
        result.update({
            "bbox": list(result["bbox"]),
            "center": list(center),
            "side": side,
            "source": "serve_detection_opposite_side",
        })
        return result

    def _receiver_area(self, center: Optional[Sequence[int]]) -> Optional[str]:
        if center is None:
            return None
        midpoint = None
        if self.court_x_range is not None:
            midpoint = (float(self.court_x_range[0]) + float(self.court_x_range[1])) * 0.5
        elif self.frame_shape is not None:
            midpoint = self.frame_shape[1] * 0.5
        if midpoint is None:
            return None
        horizontal = "right" if float(center[0]) >= midpoint else "left"
        vertical = "near" if self.net_y is not None and center[1] >= self.net_y else "far"
        return f"{horizontal}_{vertical}"

    def serve_context(self, point: Optional[Point], player_name: str = "") -> Dict:
        track = self._server_track_for_ball(point)
        detection = self._server_detection_for_ball(point)
        if detection is not None:
            # Raw geometry is more reliable during the serve than a stale
            # near/far track that may have followed an adjacent object.
            track = None
        result = {
            "player": player_name or "",
            "player_side": detection.get("side") if detection is not None else (track.side if track is not None else None),
            "player_bbox": list(detection["bbox"]) if detection is not None else (list(track.bbox) if track is not None and track.bbox else None),
            "player_position": list(detection["center"]) if detection is not None else (list(track.center) if track is not None and track.center else None),
            "racket_side": track.racket_side if track is not None else None,
            "server_source": detection.get("source") if detection is not None else ("tracked_player_below_ball" if track is not None else None),
            "serve_x": int(point[0]) if point is not None else None,
            "serve_y": int(point[1]) if point is not None else None,
        }
        receiver = self._receiver_detection_for_server(result.get("player_side"))
        result.update({
            "receiver_side": receiver.get("side") if receiver is not None else None,
            "receiver_bbox": receiver.get("bbox") if receiver is not None else None,
            "receiver_position": receiver.get("center") if receiver is not None else None,
            "receiver_area": self._receiver_area(receiver.get("center")) if receiver is not None else None,
            "receiver_source": receiver.get("source") if receiver is not None else None,
        })
        if self.learning_enabled and player_name in ("P1", "P2") and point is not None:
            player_center = detection.get("center") if detection is not None else (track.center if track is not None else None)
            side = "right" if player_center is None or point[0] >= player_center[0] else "left"
            self.profile.setdefault("serve_sides", {}).setdefault(player_name, {}).setdefault(side, 0)
            self.profile["serve_sides"][player_name][side] += 1
        return result

    def adopt_serve_context(self, context: Optional[Dict], frame_index: int = -1):
        """Seed near/far tracks from the validated server/receiver boxes."""
        if not isinstance(context, dict):
            return
        for prefix in ("player", "receiver"):
            side = context.get(f"{prefix}_side")
            bbox = context.get(f"{prefix}_bbox")
            if side not in self.tracks or not bbox or len(bbox) != 4:
                continue
            track = self.tracks[side]
            track.bbox = tuple(int(v) for v in bbox)
            track.center = self._box_center(track.bbox)
            x, y, w, h = track.bbox
            track.head = (int(round(x + w * 0.5)), int(round(y + h * 0.14)))
            track.shoes = (int(round(x + w * 0.96)), int(round(y + h * 0.96)))
            track.confidence = max(track.confidence, 0.60)
            track.last_frame = int(frame_index)
            track.visible = True

    def point_zone(self, point: Point, include_body: bool = True) -> Optional[str]:
        """Return the tracked player region containing ``point``.

        The zones are deliberately geometric and frame-local.  They are used as
        an exclusion hint, not as a hard ball detector: a round, moving ball
        can still override the hint in :meth:`candidate_penalty`.
        """
        track = self._track_for_point(point)
        if track is None or track.bbox is None or float(track.confidence) < 0.30:
            return None
        x, y, w, h = track.bbox
        px, py = point
        if track.head is not None:
            head_x, head_y = track.head
            if (abs(px - head_x) <= max(28.0, w * 0.30)
                    and abs(py - head_y) <= max(24.0, h * 0.075)):
                return "player_head_hat"
        elif (x - int(w * 0.12) <= px <= x + int(w * 1.12)
              and y - int(h * 0.12) <= py <= y + int(h * 0.30)):
            return "player_head_hat"
        if (x - int(w * 0.15) <= px <= x + int(w * 1.15)
                and y + int(h * 0.80) <= py <= y + int(h * 1.15)):
            debug_zone = os.environ.get("TGT_DEBUG_PLAYER_ZONE", "").strip().lower() in (
                "1", "true", "yes", "on"
            )
            debug_range = os.environ.get("TGT_DEBUG_PLAYER_ZONE_RANGE", "").strip()
            tracker_frame = int(self.last_frame) if self.last_frame is not None else -1
            if debug_range:
                try:
                    left, right = debug_range.split(":", 1)
                    debug_zone = int(left) <= tracker_frame <= int(right)
                except (TypeError, ValueError):
                    pass
            if debug_zone:
                track_last = int(track.last_frame)
                age = tracker_frame - track_last if tracker_frame >= 0 and track_last >= 0 else None
                shoe_rect = (
                    x - int(w * 0.15),
                    y + int(h * 0.80),
                    x + int(w * 1.15),
                    y + int(h * 1.15),
                )
                expanded = (
                    x - int(w * 0.45),
                    y - int(h * 0.25),
                    x + int(w * 1.45),
                    y + int(h * 1.25),
                )
                print(
                    f"[PLAYER_ZONE_SHOES] tracker_frame={tracker_frame} point=({px},{py}) "
                    f"side={track.side} bbox={track.bbox} center={track.center} "
                    f"shoes_point={track.shoes} visible={track.visible} "
                    f"confidence={track.confidence:.3f} track_last={track_last} age={age} "
                    f"last_detection={self._last_detection_frame} "
                    f"shoe_rect={shoe_rect} expanded_track_rect={expanded}"
                )
            return "player_shoes"
        if track.racket and track.racket.get("line"):
            lx1, ly1, lx2, ly2 = track.racket["line"]
            distance = self._point_segment_distance((px, py), (lx1, ly1), (lx2, ly2))
            if distance <= max(14.0, min(w, h) * 0.085):
                return "racket_fragment"
        if include_body and (x - int(w * 0.08) <= px <= x + int(w * 1.08)
                              and y + int(h * 0.12) <= py <= y + int(h * 0.90)):
            return "player_body"
        return None

    def candidate_penalty(self, point: Point, area: float, motion_mean: float, motion_max: float,
                          contour=None, predicted_distance: Optional[float] = None,
                          head_only: bool = False) -> Tuple[float, Optional[str]]:
        """Return a conservative penalty for head/hat/shoe/racket fragments."""
        track = self._track_for_point(point)
        if track is None or track.bbox is None:
            return 0.0, None
        x, y, w, h = track.bbox
        px, py = point
        zone = self.point_zone(point)
        head_zone = zone == "player_head_hat"
        shoe_zone = zone == "player_shoes"
        racket_zone = zone == "racket_fragment"
        round_like = False
        if contour is not None and len(contour) >= 3:
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            if perimeter > 0 and contour_area > 0:
                round_like = (4.0 * math.pi * contour_area / (perimeter * perimeter)) >= 0.30
        strong_ball_evidence = (
            float(area) >= max(10.0, min(80.0, h * 0.04)) and
            (round_like or float(motion_max) >= 130.0) and
            (predicted_distance is None or predicted_distance <= max(35.0, min(100.0, h * 0.35)))
        )
        if strong_ball_evidence:
            return 0.0, None
        if shoe_zone and float(area) <= max(25.0, h * 0.05):
            return 1800.0, "player_shoes"
        if head_zone and float(area) <= max(18.0, h * 0.045):
            return 1800.0, "player_head_hat"
        if head_only:
            return 0.0, None
        if racket_zone and float(area) <= max(24.0, h * 0.06):
            return 1500.0, "racket_fragment"
        if zone == "player_body":
            # Body contours are usually elongated and much larger than the
            # ball.  Keep a modest penalty for larger blobs so a genuine ball
            # can win on motion/roundness, while stale body pixels lose.
            if float(area) <= max(90.0, h * 0.18) or float(motion_max) < 35.0:
                return 700.0, "player_body"
            return 180.0, "player_body"
        if head_zone or shoe_zone:
            return 150.0, "player_body"
        return 0.0, None

    def snapshot(self) -> Dict:
        return {"near": self.tracks["near"].as_dict(), "far": self.tracks["far"].as_dict()}

    def point_summary(self) -> Dict:
        samples = self.point_positions
        if len(samples) > 12:
            step = max(1, len(samples) // 12)
            samples = samples[::step][-12:]
        return {
            "tracks": self.snapshot(),
            "shots": list(self.point_events),
            "position_samples": samples,
            "learned_profile": self.profile,
        }
