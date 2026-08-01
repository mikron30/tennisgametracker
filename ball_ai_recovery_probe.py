"""Generate real HSV contour candidates for offline local-AI recovery checks.

The production recovery hook will use the same candidate source.  Keeping this
tool separate lets us prove that the model selects a visible ball before it is
allowed to influence the live tracker.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODES = ("regular_court", "alt1", "alt2", "alt3", "s_30", "h_10")


def collect_candidates(image: np.ndarray, config: dict, *, modes=DEFAULT_MODES,
                       min_area: float = 3.0, max_area: float = 8000.0,
                       around: tuple[int, int] | None = None,
                       radius: float | None = None) -> list[dict]:
    """Return de-duplicated actual HSV contours, preserving their source mode."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates: list[dict] = []
    for mode in modes:
        values = config.get(mode)
        if not isinstance(values, dict):
            continue
        low = np.array([values.get("h_min", 0), values.get("s_min", 0), values.get("v_min", 0)], dtype=np.uint8)
        high = np.array([values.get("h_max", 179), values.get("s_max", 255), values.get("v_max", 255)], dtype=np.uint8)
        if np.any(low > high):
            continue
        mask = cv2.inRange(hsv, low, high)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            x, y = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
            x0, y0, width, height = cv2.boundingRect(contour)
            candidates.append({"x": x, "y": y, "area": round(area, 2), "mode": mode,
                               "bbox": [x0, y0, width, height]})
    candidates.sort(key=lambda item: item["area"], reverse=True)
    deduplicated: list[dict] = []
    for candidate in candidates:
        if around is not None and radius is not None and math.hypot(
                candidate["x"] - around[0], candidate["y"] - around[1]) > radius:
            continue
        if any(math.hypot(candidate["x"] - kept["x"], candidate["y"] - kept["y"]) < 12
               for kept in deduplicated):
            continue
        deduplicated.append(candidate)
    return deduplicated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local-AI recovery candidates from a raw frame")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-area", type=float, default=3.0)
    parser.add_argument("--max-area", type=float, default=8000.0)
    parser.add_argument("--around", nargs=2, type=int, metavar=("X", "Y"),
                        help="Limit candidates to a recovery search area")
    parser.add_argument("--radius", type=float, default=500.0)
    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not decode {image_path}")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    around = tuple(args.around) if args.around else None
    candidates = collect_candidates(image, config, min_area=args.min_area, max_area=args.max_area,
                                    around=around, radius=args.radius if around else None)
    Path(args.output).write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(image_path), "candidates": len(candidates), "output": args.output}))


if __name__ == "__main__":
    main()
