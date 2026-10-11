"""Bound expensive automatic inference in video time, not wall-clock time."""
from collections import deque


class InferenceBudget:
    def __init__(self, interval_frames=8, window_frames=25, maximum_calls=2):
        self.interval = max(1, int(interval_frames))
        self.window = max(self.interval, int(window_frames))
        self.maximum = max(1, int(maximum_calls))
        self.frames = deque()

    def ready(self, frame):
        frame = int(frame)
        if self.frames and frame < self.frames[-1]:
            self.frames.clear()  # Interactive seek starts a new video interval.
        while self.frames and frame - self.frames[0] >= self.window:
            self.frames.popleft()
        return (not self.frames or frame - self.frames[-1] >= self.interval) and len(self.frames) < self.maximum

    def reserve(self, frame):
        if not self.ready(frame):
            return False
        self.frames.append(int(frame))
        return True
