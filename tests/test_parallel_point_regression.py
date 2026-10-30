from pathlib import Path

import pytest

from parallel_point_regression import differences, make_cases, read_rows
from retry_parallel_point_context import expanded_case


def reference():
    return read_rows(Path(__file__).resolve().parents[1]/'regression_court1_through_double_fault.csv')


def test_all_events_are_covered_once_with_faults_and_second_serves_together():
    rows = reference()
    cases = make_cases(rows)
    assert [r for case in cases for r in case['expected']] == rows
    for case in cases:
        assert case['start'] < int(case['expected'][0]['serve_start_frame'])
        assert case['end'] >= int(case['expected'][-1]['point_end_frame'])
    last = cases[-1]
    assert [r['serve_attempt'] for r in last['expected']] == ['1st', '2nd']
    assert last['score'] == '2:0 15:0'
    assert last['expected'][-1]['current_score'] == '2:0 15:15'
    assert last['first_index'] == 19


def test_before_score_survives_game_change_without_seeding_expected_winner():
    cases = make_cases(reference())
    game_two = next(c for c in cases if c['expected'][0]['server'] == 'P2')
    assert game_two['score'] == '1:0 0:0'
    near = next(c for c in cases if c['expected'][0]['serve_player_side'] == 'near')
    assert near['score'] == '2:0 0:0'
    assert near['server'] == 'P1'


@pytest.mark.parametrize('field,value', [('winner','P1'), ('point_end_frame','43067'),
                                       ('serve_player_side','far'), ('current_score','2:0 30:0')])
def test_comparison_rejects_wrong_score_endpoint_winner_and_end(field, value):
    expected = reference()[-1:]
    actual = [dict(expected[0], **{field:value})]
    assert differences(expected, actual) == [dict(event=1,field=field,
                                                 expected=expected[0][field],actual=value)]


def test_reference_does_not_allow_silent_missing_or_extra_events():
    rows = reference()
    assert differences(rows, rows) == []
    assert differences(rows, rows[:-1])[0]['field'] == 'event_count'
    with pytest.raises(ValueError, match='chronological'):
        make_cases(list(reversed(rows)))


def test_retry_restores_preceding_score_and_checks_both_points():
    cases = make_cases(reference())
    expanded = expanded_case(cases, 2, 1)
    assert expanded['score'] == '0:0 0:0'
    assert expanded['first_index'] == 1
    assert expanded['start'] == cases[0]['start']
    assert expanded['end'] == cases[1]['end']
    assert [r['point_index'] for r in expanded['expected']] == ['1','2','3']
    assert expanded['case'] == 2


def test_double_fault_context_starts_before_previous_awarded_point():
    cases = make_cases(reference())
    expanded = expanded_case(cases, 13, 1)
    assert expanded['score'] == '2:0 0:0'
    assert expanded['first_index'] == 18
    assert [r['point_index'] for r in expanded['expected']] == ['18','19','20']


def test_original_analysis_boundary_is_preserved_for_expanded_first_context():
    cases = make_cases(reference(), baseline_start=33500)
    expanded = expanded_case(cases, 2, 1)
    assert expanded['start'] == 33500
    assert expanded['score'] == '0:0 0:0'
    assert expanded['end'] == 34455
    assert [r['point_index'] for r in expanded['expected']] == ['1','2','3']
    assert cases[1:] == make_cases(reference())[1:]


def test_baseline_boundary_cannot_skip_first_reference_serve():
    with pytest.raises(ValueError, match='Baseline frame'):
        make_cases(reference(), baseline_start=33736)
