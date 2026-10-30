"""Check the bounded court-one replay against its verified scoring history.

Replay frames 33500 through 42130, initial score 0:0 and initial server end
far, with hsv_config_court1_trial.json. Compare the resulting CSV here.
"""
import argparse
import csv
import json
from pathlib import Path


def compare(reference, candidate):
    csv.field_size_limit(20_000_000)
    with reference.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        expected = list(reader)
    with candidate.open(encoding='utf-8-sig', newline='') as stream:
        actual = list(csv.DictReader(stream))
    differences = []
    if len(actual) != len(expected):
        differences.append(dict(field='event_count', expected=len(expected), actual=len(actual)))
    for event, (wanted, got) in enumerate(zip(expected, actual), 1):
        for field in fields:
            if wanted[field] != got.get(field):
                differences.append(dict(event=event, field=field,
                                        expected=wanted[field], actual=got.get(field)))
    return dict(status='passed' if not differences else 'failed',
                reference=str(reference), candidate=str(candidate),
                expected_events=len(expected), actual_events=len(actual),
                differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path)
    parser.add_argument('--reference', type=Path,
                        default=Path(__file__).with_name('regression_court1_first_two_games.csv'),
                        help='Reviewed fixture; use regression_court1_through_double_fault.csv for all 20 events')
    parser.add_argument('--output', type=Path, help='Optional JSON comparison report')
    args = parser.parse_args()
    reference = args.reference
    result = compare(reference, args.history)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"{result['status'].upper()}: {result['actual_events']} events; "
          f"{len(result['differences'])} differences from verified court-one history")
    for difference in result['differences'][:6]:
        print(json.dumps(difference))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
