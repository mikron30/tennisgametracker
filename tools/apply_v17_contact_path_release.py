#!/usr/bin/env python3
"""Apply V17: release a proven normal-HSV serve path after repeated Contact Local-AI misses.

This patch is intentionally narrow. It changes only the Contact Local-AI trigger
arbitration during the short pre-bounce serve/contact window and adds quiet-log
visibility for the relevant diagnostics.
"""

from __future__ import annotations

import argparse
import pathlib
import py_compile
import shutil
import sys

TARGET = pathlib.Path("interactive_ball_analyzer.py")

OLD_CONTACT_BLOCK = """        if (
                (contact_near or watch_active) and proposed_speed >= jump_floor and
                (angle_delta >= 35.0 or prediction_error >= 50.0)):
            return (
                f'contact-jump:{proposed_speed:.0f}px>={jump_floor:.0f}/'
                f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
            )
"""

NEW_CONTACT_BLOCK = """        contact_jump_candidate = (
            (contact_near or watch_active) and proposed_speed >= jump_floor and
            (angle_delta >= 35.0 or prediction_error >= 50.0)
        )

        # V17: Contact Local-AI is an assist, not a permanent veto.
        #
        # During the short serve-contact/pre-bounce window, the normal HSV
        # tracker can already see the outgoing ball while the local model still
        # misses a blurred/contact-adjacent patch. The process loop deliberately
        # rolls that normal candidate back when Contact Local-AI misses. Keep
        # that protection for one-off racket/body fragments, but remember the
        # rejected normal proposals. Three consecutive netward proposals with
        # a coherent direction prove an independent causal ball path and let the
        # normal tracker keep the third frame.
        serve_contact_path_window = (
            contact_jump_candidate and
            tracked_position is not None and
            bool(getattr(self, '_awaiting_serve_bounce', False)) and
            int(getattr(self, '_serve_contact_grace_frames', 0) or 0) > 0 and
            int(getattr(self, 'ground_bounce_count', 0) or 0) == 0 and
            int(getattr(self, 'serve_direction_dy', 0) or 0) != 0
        )
        contact_normal_path = list(
            getattr(self, '_contact_normal_fallback_path', []) or []
        )

        if serve_contact_path_window:
            current_sample = {
                'frame': int(self.frame_count),
                'pos': tuple(tracked_position),
                'area': float(proposed_size),
            }
            if (
                    contact_normal_path and
                    int(current_sample['frame']) !=
                    int(contact_normal_path[-1].get('frame', -1000000)) + 1):
                contact_normal_path = []

            if contact_normal_path:
                previous_sample = contact_normal_path[-1]
                previous_pos = tuple(previous_sample['pos'])
                step_dx = float(tracked_position[0] - previous_pos[0])
                step_dy = float(tracked_position[1] - previous_pos[1])
                step_distance = math.hypot(step_dx, step_dy)
                serve_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
                netward_progress = -step_dy if serve_dy < 0 else step_dy
                if not (
                        20.0 <= step_distance <= 220.0 and
                        netward_progress >= 15.0):
                    contact_normal_path = []

            contact_normal_path.append(current_sample)
            contact_normal_path = contact_normal_path[-3:]
            self._contact_normal_fallback_path = contact_normal_path

            if len(contact_normal_path) == 3:
                p0 = tuple(contact_normal_path[0]['pos'])
                p1 = tuple(contact_normal_path[1]['pos'])
                p2 = tuple(contact_normal_path[2]['pos'])
                v1 = (
                    float(p1[0] - p0[0]),
                    float(p1[1] - p0[1]),
                )
                v2 = (
                    float(p2[0] - p1[0]),
                    float(p2[1] - p1[1]),
                )
                len1 = math.hypot(v1[0], v1[1])
                len2 = math.hypot(v2[0], v2[1])
                cosine = (
                    (v1[0] * v2[0] + v1[1] * v2[1]) /
                    max(1.0, len1 * len2)
                )
                serve_dy = int(getattr(self, 'serve_direction_dy', 0) or 0)
                total_netward = (
                    float(p0[1] - p2[1])
                    if serve_dy < 0 else
                    float(p2[1] - p0[1])
                )
                coherent_path = (
                    20.0 <= len1 <= 220.0 and
                    20.0 <= len2 <= 220.0 and
                    cosine >= 0.90 and
                    total_netward >= 100.0
                )
                if coherent_path:
                    self._contact_local_ai_state = None
                    self._contact_local_ai_cooldown_until_frame = max(
                        int(getattr(
                            self,
                            '_contact_local_ai_cooldown_until_frame',
                            -1000000,
                        )),
                        int(self.frame_count) + 2,
                    )
                    self._contact_normal_fallback_path = []
                    print(
                        f"[CONTACT_NORMAL_PATH_RELEASE] f{self.frame_count}: "
                        f"accepted normal HSV={tuple(tracked_position)} after "
                        f"3 coherent netward proposals "
                        f"path={[entry['pos'] for entry in contact_normal_path]} "
                        f"cos={cosine:.3f} net={total_netward:.1f}px"
                    )
                    return None
        else:
            self._contact_normal_fallback_path = []

        if contact_jump_candidate:
            return (
                f'contact-jump:{proposed_speed:.0f}px>={jump_floor:.0f}/'
                f'angle={angle_delta:.0f}deg/pred={prediction_error:.0f}px'
            )
"""

OLD_QUIET_BLOCK = """            text.startswith(\"[JUMP_REJECTED]\") or
            text.startswith(\"[POINT_IGNORED]\") or
"""

NEW_QUIET_BLOCK = """            text.startswith(\"[JUMP_REJECTED]\") or
            text.startswith(\"[CONTACT_LOCAL_AI\") or
            text.startswith(\"[CONTACT_NORMAL_PATH_RELEASE]\") or
            text.startswith(\"[LOCAL_AI_HANDOFF_STALL]\") or
            text.startswith(\"[POINT_IGNORED]\") or
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f"[V17] {label}: already applied")
            return text
        raise RuntimeError(f"{label}: expected source block was not found")
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source block, found {count}")
    print(f"[V17] {label}: patching")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="Verify that V17 can be applied without writing files",
    )
    parser.add_argument("--target", type=pathlib.Path, default=TARGET)
    args = parser.parse_args()

    target = args.target
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")

    original = target.read_text(encoding="utf-8")
    patched = replace_once(
        original, OLD_CONTACT_BLOCK, NEW_CONTACT_BLOCK,
        "three-frame Contact/HSV arbitration",
    )
    patched = replace_once(
        patched, OLD_QUIET_BLOCK, NEW_QUIET_BLOCK,
        "quiet contact diagnostics",
    )

    compile(patched, str(target), "exec")
    if args.check:
        print("[V17] check OK; source compiles after patch")
        return 0

    if patched == original:
        print("[V17] no changes needed")
        return 0

    backup = target.with_suffix(target.suffix + ".v16.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[V17] backup: {backup}")

    target.write_text(patched, encoding="utf-8")
    try:
        py_compile.compile(str(target), doraise=True)
    except Exception:
        shutil.copy2(backup, target)
        raise

    print("[V17] applied successfully; interactive_ball_analyzer.py compiles")
    print("[V17] review with: git diff -- interactive_ball_analyzer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
