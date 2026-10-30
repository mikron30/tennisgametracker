"""Retry only failed isolated points with preceding-point context."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from parallel_point_regression import file_hash, run_case


def expanded_case(cases, number, count):
    index = next(i for i, case in enumerate(cases) if case['case'] == number)
    prefix = cases[max(0, index-count):index+1]
    first, last = prefix[0], prefix[-1]
    return dict(case=number, start=first['start'], end=last['end'], score=first['score'],
                attempt=first['attempt'], server=first['server'], first_index=first['first_index'],
                expected=[row for case in prefix for row in case['expected']])


def unchanged(hashes):
    return all(Path(p).exists() and file_hash(p) == value for p, value in hashes.items())


def retry(directory, cases, original, maximum, timeout):
    number = original['case']
    attempts = []
    base = json.loads((directory/f'point_{number:03d}'/'manifest.json').read_text())
    index = next(i for i, case in enumerate(cases) if case['case'] == number)
    for count in range(1, min(maximum, index)+1):
        expanded = expanded_case(cases, number, count)
        private = directory/f'point_{number:03d}'/f'context_{count:02d}'
        private.mkdir(exist_ok=False)
        spec = dict(base, case=expanded, directory=str(private))
        manifest = private/'manifest.json'
        manifest.write_text(json.dumps(spec, indent=2), encoding='utf-8')
        result = run_case(manifest, timeout)
        result['preceding_points'] = count
        attempts.append(result)
        if result['status'] == 'passed':
            return dict(case=number, status='passed_with_context', original=original, retries=attempts)
    return dict(case=number, status='needs_context_review', original=original, retries=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Completed parallel_points_* directory')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-context', type=int, default=2, help='Maximum preceding point cases')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if min(args.workers, args.max_context, args.timeout) < 1:
        parser.error('workers, max-context and timeout must be positive')
    directory = args.run.resolve()
    plan = json.loads((directory/'plan.json').read_text())
    cold = json.loads((directory/'results.json').read_text())
    if not cold['source_and_inputs_unchanged'] or not unchanged(plan['hashes']):
        parser.error('Sources/inputs changed since the cold batch; run a fresh batch')
    if (directory/'context_results.json').exists():
        parser.error('Context report already exists; use a fresh batch')
    hashes = dict(plan['hashes'])
    hashes[str(Path(__file__).resolve())] = file_hash(__file__)
    failed = [r for r in cold['cases'] if r['status'] != 'passed']
    print(f'Retrying {len(failed)} of {len(cold["cases"])} cases with preceding context', flush=True)
    results = [r for r in cold['cases'] if r['status'] == 'passed']
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(retry, directory, plan.get('all_cases', plan['cases']), r,
                                args.max_context, args.timeout):r
                for r in failed}
        for job in as_completed(jobs):
            try:
                result = job.result()
            except Exception as error:
                result = dict(case=jobs[job]['case'], status='error', error=str(error))
            results.append(result)
            print(f"Point {result['case']}: {result['status']}", flush=True)
    stable = unchanged(hashes)
    passed = stable and all(r['status'] in ('passed', 'passed_with_context') for r in results)
    elapsed = round(time.monotonic()-started, 2)
    report = dict(status='passed' if passed else 'needs_review', source_and_inputs_unchanged=stable,
                  cold_seconds=cold['seconds'], retry_seconds=elapsed,
                  total_seconds=round(cold['seconds']+elapsed, 2),
                  cases=sorted(results, key=lambda r:r['case']),
                  limitation='Passing checks only these reference windows. Full sequential regression still checks cumulative state and missed events outside windows.')
    (directory/'context_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['status'], flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
