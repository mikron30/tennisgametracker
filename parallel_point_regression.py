"""Replay independent tennis points concurrently; keep serve attempts together.

This is a fast isolation check, not a substitute for sequential match checks.
Expected scores/outcomes are used only by the comparator, never by detection.
"""
import argparse
import builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FIELDS = ('point_index', 'serve_start_frame', 'point_end_frame', 'server',
          'serve_attempt', 'category', 'winner', 'point_awarded', 'current_score',
          'next_server', 'next_serve', 'serve_player_side', 'receive_player_side')


def read_rows(path):
    csv.field_size_limit(20_000_000)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def make_cases(rows, initial_score='0:0 0:0', pre_roll=240, post_roll=16, baseline_start=None):
    if pre_roll < 0 or post_roll < 0 or not rows:
        raise ValueError('Need rows and non-negative frame padding')
    if baseline_start is not None and not 0 <= baseline_start < int(rows[0]['serve_start_frame']):
        raise ValueError('Baseline frame must precede the first reference serve')
    groups, pending = [], []
    previous_end = -1
    for row in rows:
        start, end = int(row['serve_start_frame']), int(row['point_end_frame'])
        if not previous_end < start <= end:
            raise ValueError('Reference events must be chronological and non-overlapping')
        previous_end = end
        pending.append(row)
        if row['point_awarded'] == 'yes':
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    cases, score, previous_end = [], initial_score, -post_roll-1
    for index, group in enumerate(groups):
        start = max(0, int(group[0]['serve_start_frame'])-pre_roll,
                    previous_end+post_roll+1)
        if index == 0 and baseline_start is not None:
            start = baseline_start
        end = int(group[-1]['point_end_frame'])+post_roll
        if index+1 < len(groups):
            end = min(end, int(groups[index+1][0]['serve_start_frame'])-1)
        if start >= int(group[0]['serve_start_frame']):
            raise ValueError('Insufficient gap before point; use a larger sequential chunk')
        cases.append(dict(case=index+1, start=start, end=end, score=score,
                          attempt=group[0]['serve_attempt'], server=group[0]['server'],
                          first_index=int(group[0]['point_index']), expected=group))
        score = group[-1]['current_score']
        previous_end = int(group[-1]['point_end_frame'])
    return cases


def differences(expected, actual):
    result = []
    if len(expected) != len(actual):
        result.append(dict(field='event_count', expected=len(expected), actual=len(actual)))
    for row_index, (wanted, got) in enumerate(zip(expected, actual)):
        for field in FIELDS:
            if field in wanted and wanted[field] != got.get(field):
                result.append(dict(event=row_index+1, field=field,
                                   expected=wanted[field], actual=got.get(field)))
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(manifest_path):
    spec = json.loads(Path(manifest_path).read_text())
    case, directory = spec['case'], Path(spec['directory'])
    config = Path(spec['config'])
    private_false = directory/'false_points.json'
    frozen_false = Path(spec['frozen_false'])
    if frozen_false.exists():
        private_false.write_bytes(frozen_false.read_bytes())
    original_open = builtins.open
    hotspot = config.with_name('frame0_hotspots_'+config.stem+'.json')

    def isolated_open(file, mode='r', *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and any(c in mode for c in 'wax+'):
            if Path(file).resolve() == hotspot:
                file = directory/hotspot.name
        return original_open(file, mode, *args, **kwargs)

    def install(cls):
        cls._steady_false_point_file_path = lambda self: private_false
        original_process = cls.process_video

        def process(self, *args, **kwargs):
            # Seed only the pre-point state, never the expected result or track.
            if self.player_names[self._current_server_index()] != case['server']:
                raise ValueError('Score seed does not imply reference server; need match-state checkpoint')
            self.current_serve_attempt = 2 if case['attempt'] == '2nd' else 1
            self._point_history_point_index = case['first_index']-1
            result = original_process(self, *args, **kwargs)
            if self.frame_count != case['end']:
                raise RuntimeError('Replay did not reach requested end frame')
            print('[PARALLEL_POINT_COMPLETE]', case['case'], flush=True)
            return result
        cls.process_video = process

    source = (ROOT/'interactive_ball_analyzer.py').read_text(encoding='utf-8-sig')
    marker = 'if __name__ == "__main__":'
    if source.count(marker) != 1:
        raise RuntimeError('CLI entry point changed; update replay hook')
    source = source.replace(marker, '_install_replay_hooks(InteractiveBallAnalyzer)\n\n'+marker)
    sys.argv = ['interactive_ball_analyzer.py', '--court', spec['court'], '--config', str(config),
                '--start-frame', str(case['start']), '--max-frames', str(case['end']-case['start']),
                '--start-score', case['score'], '--start-server-side', spec['initial_server_end'],
                '--headless', '--auto-play', '--disable-player-learning', '--no-audit-points',
                '--point-history-file', str(directory/'history.csv'),
                '--local-ai-recovery-dir', str(directory/'ai')]+spec['video_args']
    builtins.open = isolated_open
    try:
        exec(compile(source, str(ROOT/'interactive_ball_analyzer.py'), 'exec'),
             dict(__name__='__main__', __file__=str(ROOT/'interactive_ball_analyzer.py'),
                  _install_replay_hooks=install))
    finally:
        builtins.open = original_open


def run_case(manifest, timeout):
    spec = json.loads(manifest.read_text())
    directory = Path(spec['directory'])
    started = time.monotonic()
    try:
        with (directory/'replay.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                     '--worker', str(manifest)], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        histories = list(directory.glob('history_*.csv'))
        actual = read_rows(histories[0]) if len(histories) == 1 else []
        delta = differences(spec['case']['expected'], actual)
        complete = '[PARALLEL_POINT_COMPLETE]' in (directory/'replay.log').read_text(encoding='utf-8')
        return dict(case=spec['case']['case'], status='passed' if result.returncode == 0 and complete and not delta else 'needs_context_review',
                    returncode=result.returncode, complete=complete, differences=delta,
                    directory=str(directory), seconds=round(time.monotonic()-started, 2))
    except Exception as error:
        return dict(case=spec['case']['case'], status='error', error=str(error),
                    directory=str(directory), seconds=round(time.monotonic()-started, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, nargs='?')
    parser.add_argument('--court')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--initial-server-end', choices=['near', 'far'])
    parser.add_argument('--initial-score', default='0:0 0:0')
    parser.add_argument('--baseline-start-frame', type=int,
                        help='Original sequential initialization boundary for the first point/context')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pre-roll', type=int, default=240)
    parser.add_argument('--post-roll', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=1200, help='Per-case seconds')
    parser.add_argument('--cases', type=int, nargs='+', help='Subset of generated case numbers')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--video')
    parser.add_argument('--video-dir')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if not all([args.reference, args.court, args.config, args.initial_server_end]):
        parser.error('reference, --court, --config and --initial-server-end are required')
    if args.workers < 1 or args.timeout < 1:
        parser.error('workers and timeout must be positive')
    all_cases = make_cases(read_rows(args.reference), args.initial_score, args.pre_roll,
                          args.post_roll, args.baseline_start_frame)
    cases = all_cases
    if args.cases:
        if not set(args.cases) <= {c['case'] for c in cases}:
            parser.error('Unknown case number')
        cases = [c for c in cases if c['case'] in args.cases]
    output = (args.output or ROOT/'tmp'/'regression'/('parallel_points_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = args.config.resolve()
    # Freeze both code and input hashes. Any concurrent edit invalidates the run.
    paths = list(ROOT.glob('*.py'))+[config, args.reference.resolve()]
    paths += [p for p in config.parent.glob('*'+config.stem+'.json') if p != config and not p.name.startswith('frame0_hotspots_')]
    hashes = {str(p):file_hash(p) for p in paths}
    frozen_false = output/'initial_false_points.json'
    false_source = config.with_name('steady_false_points_'+config.stem+'.json')
    if false_source.exists():
        frozen_false.write_bytes(false_source.read_bytes())
    manifests = []
    for case in cases:
        directory = output/f"point_{case['case']:03d}"
        directory.mkdir()
        spec = dict(case=case, directory=str(directory), config=str(config), court=args.court,
                    initial_server_end=args.initial_server_end, frozen_false=str(frozen_false),
                    video_args=sum(([flag, value] for flag, value in [('--video', args.video), ('--video-dir', args.video_dir)] if value), []))
        manifest = directory/'manifest.json'
        manifest.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        manifests.append(manifest)
    (output/'plan.json').write_text(json.dumps(dict(cases=cases, all_cases=all_cases,
        baseline_start_frame=args.baseline_start_frame, hashes=hashes, workers=args.workers), indent=2), encoding='utf-8')
    print(f'{len(cases)} point cases; {sum(c["end"]-c["start"] for c in cases)} replay frames; output={output}', flush=True)
    if args.dry_run:
        return 0
    started, results = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run_case, p, args.timeout) for p in manifests]):
            result = future.result()
            results.append(result)
            print(f"Point {result['case']}: {result['status']} ({result['seconds']}s)", flush=True)
    unchanged = all(Path(p).exists() and file_hash(p) == value for p, value in hashes.items())
    passed = unchanged and all(r['status'] == 'passed' for r in results)
    report = dict(status='passed' if passed else 'needs_review', source_and_inputs_unchanged=unchanged,
                  seconds=round(time.monotonic()-started, 2), workers=args.workers,
                  cases=sorted(results, key=lambda r:r['case']),
                  limitation='Isolated replays reset tracking/player history. Differences require a wider-context replay; passing does not verify cross-point state or unreviewed winners.')
    (output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['status'], flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
