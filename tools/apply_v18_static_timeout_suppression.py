#!/usr/bin/env python3
"""Apply V18: never score a stuck timeout on a recently suppressed static out artifact."""

from __future__ import annotations

import argparse
import pathlib
import py_compile
import shutil
import sys

TARGET = pathlib.Path("interactive_ball_analyzer.py")

OLD_BLOCK = """                            history_end_frame = (
                                self.frame_count
                                if static_timeout_artifact else
                                self._stuck_timeout_end_frame(
                                    point_start_frame=point_start_frame,
                                    frame=self.frame_count,
                                )
                            )
                            if static_timeout_artifact:
                                print(
                                    f\"Frame {self.frame_count}: [STUCK-TIMEOUT FRAME NOT BACKDATED] \"
                                    f\"static marker began at f{static_timeout_frame}\"
                                )
                            print(f\"Frame {self.frame_count}: POINT ENDED - {stuck_reason}\")
"""

NEW_BLOCK = """                            # V18: a marker already classified as a recent static
                            # out-bounce artifact must not be allowed to end the point
                            # through the generic stuck-timeout path. Keep recovery alive
                            # and let a later real ball observation or legitimate endpoint
                            # decide the point instead.
                            if static_timeout_artifact:
                                print(
                                    f\"Frame {self.frame_count}: [STUCK-TIMEOUT SUPPRESSED] \"
                                    f\"static marker {tuple(tracked_position)} follows suppressed \"
                                    f\"artifact {tuple(static_timeout_point)} from f{static_timeout_frame}; \"
                                    f\"continuing recovery instead of scoring\"
                                )
                                continue

                            history_end_frame = self._stuck_timeout_end_frame(
                                point_start_frame=point_start_frame,
                                frame=self.frame_count,
                            )
                            print(f\"Frame {self.frame_count}: POINT ENDED - {stuck_reason}\")
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f"[V18] {label}: already applied")
            return text
        raise RuntimeError(f"{label}: expected source block was not found")
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source block, found {count}")
    print(f"[V18] {label}: patching")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--target", type=pathlib.Path, default=TARGET)
    args = parser.parse_args()

    target = args.target
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    original = target.read_text(encoding="utf-8")
    patched = replace_once(
        original,
        OLD_BLOCK,
        NEW_BLOCK,
        "static artifact timeout suppression",
    )

    compile(patched, str(target), "exec")
    if args.check:
        print("[V18] check OK; source compiles after patch")
        return 0

    if patched == original:
        print("[V18] no changes needed")
        return 0

    backup = target.with_suffix(target.suffix + ".v17.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[V18] backup: {backup}")

    target.write_text(patched, encoding="utf-8")
    try:
        py_compile.compile(str(target), doraise=True)
    except Exception:
        shutil.copy2(backup, target)
        raise

    print("[V18] applied successfully; interactive_ball_analyzer.py compiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
