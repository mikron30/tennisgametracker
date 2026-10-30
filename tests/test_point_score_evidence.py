from types import SimpleNamespace

import pytest

from point_score_evidence import unresolved_end_requires_review


def owner(**changes):
    values=dict(frame_count=100,point_start_frame_internal=50,
        _short_rewind=SimpleNamespace(point=50,unresolved=True,hard_uncertainty=False),
        far_serve_image_baseline=True,ball_size=20,stuck_frame_count=0,
        _last_tracked_candidate_motion_frame=100,
        _last_tracked_candidate_motion_mean=20,_last_tracked_candidate_motion_max=100)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('reason',['POINT_TIMEOUT','Ball lost (likely out of court)',
                                 'Ball stopped on player side'])
def test_unresolved_loss_cannot_award_a_timeout_point_under_another_label(reason):
    assert unresolved_end_requires_review(owner(),reason)


@pytest.mark.parametrize('reason',['Ball bounced out of court (right sideline)','Ball hit the net'])
@pytest.mark.parametrize('weak',[dict(_last_tracked_candidate_motion_mean=1.5),
    dict(_last_tracked_candidate_motion_frame=90),dict(ball_size=1),dict(stuck_frame_count=4)])
def test_weak_terminal_cannot_turn_failed_recovery_into_a_confident_score(reason,weak):
    obj=owner(**weak)
    assert unresolved_end_requires_review(obj,reason)
    assert not unresolved_end_requires_review(obj,reason,measured_terminal=True)


def test_fresh_terminal_and_unrelated_or_resolved_loss_keep_normal_scoring():
    assert not unresolved_end_requires_review(owner(),'Ball bounced out of court (left sideline)')
    obj=owner(_last_tracked_candidate_motion_mean=0)
    obj._short_rewind.point=20
    assert not unresolved_end_requires_review(obj,'Ball hit the net')
    obj._short_rewind.point=50;obj._short_rewind.unresolved=False
    assert not unresolved_end_requires_review(obj,'Ball stopped on player side')


def test_hard_ownership_uncertainty_stays_unresolved_even_with_terminal_pixels():
    obj=owner();obj._short_rewind.hard_uncertainty=True
    assert unresolved_end_requires_review(obj,'Ball hit the net',measured_terminal=True)


def test_one_failed_replay_cannot_score_a_weak_terminal_before_retry_budget_expires():
    obj=owner(_last_tracked_candidate_motion_mean=0)
    obj._short_rewind.unresolved=False
    obj._short_rewind.first_problem=95
    obj._short_rewind.last_rejection='no-coherent-ai-path'
    assert unresolved_end_requires_review(obj,'Ball hit the net')
    assert unresolved_end_requires_review(obj,'Ball stopped on player side')
    assert not unresolved_end_requires_review(obj,'Ball hit the net',measured_terminal=True)


@pytest.mark.parametrize('invalid',[None,'far','first_fault','receiver_hit','other_reason'])
def test_legal_serve_net_return_cannot_be_scored_with_a_missing_receiver_hit(invalid):
    from types import SimpleNamespace
    obj=SimpleNamespace(far_serve_image_baseline=True,_active_serve_area_end='near',
        _serve_landed_in_current_attempt=True,_last_serve_bounce_was_in=True,_point_hit_count=1)
    if invalid=='far':obj._active_serve_area_end='far'
    if invalid=='first_fault':obj._serve_landed_in_current_attempt=False
    if invalid=='receiver_hit':obj._point_hit_count=2
    reason='Ball bounced out of court' if invalid=='other_reason' else 'Ball hit the net'
    assert unresolved_end_requires_review(obj,reason)==(invalid is None)
