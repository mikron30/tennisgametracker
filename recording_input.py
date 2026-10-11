"""Resolve downloaded recordings and exclude a configurable warm-up interval."""
from pathlib import Path
import math

DEFAULT_VIDEO_DIR = Path(r"C:\Users\Roy\Documents\carmeltennis\downloaded_tennis_videos")
DEFAULT_RECORDING = "court_1_2026-10-03_1100-1200.mp4"


def resolve_recording(directory, filename):
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(directory).expanduser() / path
    if not path.suffix:
        path = path.with_suffix(".mp4")
    if not path.is_file():
        raise ValueError(f"Recording does not exist: {path}")
    return str(path.resolve())


def match_search_start(fps, total_frames, requested_start, warmup_minutes):
    """Return the first eligible frame; this is a time gate, not visual proof."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot check match start: recording has invalid FPS")
    if not math.isfinite(warmup_minutes) or warmup_minutes < 0:
        raise ValueError("Warm-up minutes must be finite and non-negative")
    start = max(0, int(requested_start), math.ceil(warmup_minutes * 60 * fps))
    if total_frames <= 0 or start >= total_frames:
        raise ValueError("Match-start search falls outside the recording")
    return start
