from pathlib import Path

probe_path = Path('ball_ai_recovery_probe.py')
probe = probe_path.read_text(encoding='utf-8')
old_sig = '''def collect_candidates(image: np.ndarray, config: dict, *, modes=DEFAULT_MODES,
                       min_area: float = 3.0, max_area: float = 8000.0,
                       around: tuple[int, int] | None = None,
                       radius: float | None = None) -> list[dict]:'''
new_sig = '''def collect_candidates(image: np.ndarray, config: dict, *, modes=DEFAULT_MODES,
                       min_area: float = 3.0, max_area: float = 8000.0,
                       around: tuple[int, int] | None = None,
                       radius: float | None = None,
                       dedup_distance: float = 12.0) -> list[dict]:'''
if old_sig not in probe:
    raise SystemExit('collect_candidates signature marker not found')
probe = probe.replace(old_sig, new_sig, 1)
old_dedup = '''        if any(math.hypot(candidate["x"] - kept["x"], candidate["y"] - kept["y"]) < 12
               for kept in deduplicated):
            continue
'''
new_dedup = '''        dedup_radius = max(0.0, float(dedup_distance))
        if dedup_radius > 0.0 and any(
                math.hypot(candidate["x"] - kept["x"], candidate["y"] - kept["y"]) < dedup_radius
                for kept in deduplicated):
            continue
'''
if old_dedup not in probe:
    raise SystemExit('collect_candidates dedup marker not found')
probe = probe.replace(old_dedup, new_dedup, 1)
probe_path.write_text(probe, encoding='utf-8')

recovery_path = Path('ball_local_ai_recovery.py')
recovery = recovery_path.read_text(encoding='utf-8')
start = recovery.index('    def rank_local_roi_candidates(')
end = recovery.index('    def rank_local_roi_candidate(', start)
block = recovery[start:end]
old_call = '''            around=roi_anchor,
            radius=roi_radius,
        )'''
new_call = '''            around=roi_anchor,
            radius=roi_radius,
            # The default 12 px cross-HSV de-dup is useful for broad recovery,
            # but it can hide a small regular-HSV ball behind a larger alt-HSV
            # blob at nearly the same location.  Trajectory arbitration needs
            # to see both geometries so physics and AI can compare them.
            dedup_distance=4.0,
        )'''
if old_call not in block:
    raise SystemExit('rank_local_roi_candidates collect call marker not found')
block = block.replace(old_call, new_call, 1)
recovery = recovery[:start] + block + recovery[end:]
recovery_path.write_text(recovery, encoding='utf-8')

analyzer_path = Path('interactive_ball_analyzer.py')
analyzer = analyzer_path.read_text(encoding='utf-8')
old_print = '''                f"#{row['rank']} pos={row['position']} ai={row['ai_score']:.6f} "
                f"err={row['prediction_error']:.1f}/{prediction_cap:.1f}px "'''
new_print = '''                f"#{row['rank']} pos={row['position']} ai={row['ai_score']:.6f} "
                f"mode={row['mode'] or 'unknown'} "
                f"err={row['prediction_error']:.1f}/{prediction_cap:.1f}px "'''
if old_print not in analyzer:
    raise SystemExit('trajectory debug print marker not found')
analyzer = analyzer.replace(old_print, new_print, 1)
analyzer_path.write_text(analyzer, encoding='utf-8')

print('patched fine trajectory candidate dedup + debug source mode')
