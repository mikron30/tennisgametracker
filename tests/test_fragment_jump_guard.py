import ast
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tracking_evidence import unsupported_fragment_jump


def metrics(**changes):
    result = dict(prev_distance=227.3, predicted_distance=210.0,
                  motion_mean=4.1, motion_max=19.0)
    result.update(changes)
    return result


@pytest.mark.parametrize('height,step', [(2160,227.3), (1080,114.0), (720,95.0)])
def test_distant_weak_fragment_rejected_at_multiple_resolutions(height, step):
    assert unsupported_fragment_jump(17.0, 3.5, metrics(prev_distance=step), height)


@pytest.mark.parametrize('changes', [dict(motion_mean=15), dict(motion_max=80),
                                    dict(predicted_distance=15), dict(prev_distance=30)])
def test_motion_or_trajectory_supported_recovery_remains_eligible(changes):
    assert not unsupported_fragment_jump(17.0, 3.5, metrics(**changes), 2160)


def test_weak_jump_without_prediction_requires_evidence():
    assert unsupported_fragment_jump(17, 3.5, metrics(predicted_distance=None), 2160)


def test_moderate_fence_motion_does_not_certify_one_pixel_jump():
    assert unsupported_fragment_jump(17,1,metrics(prev_distance=166,
                                                motion_mean=8.6,motion_max=44),2160)


def test_widened_search_prediction_cannot_certify_a_fragment_jump():
    assert unsupported_fragment_jump(17,1,metrics(prev_distance=166,predicted_distance=0,
                                                motion_mean=8.6,motion_max=44),2160,
                                     previous_pos=(1384,254),candidate_pos=(1219,272),
                                     previous_motion=dict(dx=-12,dy=26))


def test_measured_motion_can_support_a_small_fast_ball():
    assert not unsupported_fragment_jump(17,3.5,metrics(predicted_distance=200),2160,
                                         previous_pos=(1384,254),candidate_pos=(1219,272),
                                         previous_motion=dict(dx=-165,dy=18))


@pytest.mark.parametrize('previous, candidate', [(None,3.5), (3.5,1), (17,12), (17,8)])
def test_no_size_collapse_is_not_rejected(previous, candidate):
    assert not unsupported_fragment_jump(previous, candidate, metrics(), 2160)


@lru_cache(maxsize=1)
def tracker_class():
    source = Path(__file__).resolve().parents[1] / 'interactive_ball_analyzer.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'InteractiveBallAnalyzer')
    return cls


@lru_cache(maxsize=2)
def override_guard(name='_should_accept_hsv_override'):
    source = Path(__file__).resolve().parents[1] / 'interactive_ball_analyzer.py'
    method = next(n for n in tracker_class().body if isinstance(n, ast.FunctionDef)
                  and n.name == name)
    namespace = {'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[method.name]


@pytest.mark.parametrize('label', ['alt1', 'alt2', 'alt4', 'alt6'])
@pytest.mark.parametrize('current_pos', [None, (1384,254)])
def test_override_guard_does_not_depend_on_current_filter_or_night(label, current_pos):
    # Execute the real method with only its image-metric boundary stubbed.
    analyzer = SimpleNamespace(ball_size=17,
        _collect_override_candidate_metrics=lambda *args: metrics())
    assert not override_guard()(analyzer, label, dict(pos=(1327,34),area=3.5),
                                     current_pos,17,(1384,254),(1380,280),
                                     np.zeros((2160,1),dtype=np.uint8))


def resolver(visible=None, motion=None):
    return SimpleNamespace(
        ball_center=(1384,254),ball_size=17,max_ball_speed=400,
        last_motion=dict(dx=-12,dy=26,distance=28.6),
        _collect_override_candidate_metrics=lambda *args: metrics(
            prev_distance=175.3,**(motion or {})),
        _find_night_visible_ball_candidate=lambda *args: visible)


def resolve(analyzer):
    return override_guard('_resolve_fragment_jump')(
        analyzer,(1216,304),1,(1384,254),17,
        np.zeros((2160,1,3),dtype=np.uint8),np.zeros((2160,1),dtype=np.uint8))


def test_primary_fragment_holds_without_poisoning_motion():
    analyzer = resolver()
    original_motion = dict(analyzer.last_motion)
    assert resolve(analyzer) == {'hold':True}
    assert analyzer.ball_center == (1384,254)
    assert analyzer.ball_size == 17
    assert analyzer.last_motion == original_motion


def test_primary_fragment_can_recover_a_bounded_visible_moving_ball():
    visible = dict(pos=(1262,450),area=110,motion_mean=40,motion_max=160)
    result = resolve(resolver(visible))
    assert result['recovery']['pos'] == visible['pos']
    assert 'recovery_label' not in visible


@pytest.mark.parametrize('changes',[dict(pos=(2700,450)),dict(motion_max=44),
                                  dict(motion_mean=8),dict(area=3.5)])
def test_unconfirmed_alternative_does_not_replace_rejected_primary(changes):
    visible = dict(pos=(1262,450),area=110,motion_mean=40,motion_max=160)
    visible.update(changes)
    assert resolve(resolver(visible)) == {'hold':True}


def test_strong_primary_motion_does_not_trigger_broad_recovery():
    assert resolve(resolver(motion=dict(motion_max=100))) is None


def test_rejected_primary_restores_filter_even_after_in_place_mutation():
    method = next(n for n in tracker_class().body if isinstance(n,ast.FunctionDef)
                  and n.name == 'track_ball_in_frame')
    capture = next(n for n in method.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='trusted_hsv_state' for t in n.targets))
    restore = next(n for n in ast.walk(method) if isinstance(n,ast.For)
                   and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Attribute)
                   and isinstance(n.iter.func.value,ast.Name)
                   and n.iter.func.value.id == 'trusted_hsv_state')
    functions = ast.parse('def capture(self):\n    pass\ndef restore(self, trusted_hsv_state):\n    pass')
    functions.body[0].body = [capture,ast.Return(value=ast.Name(id='trusted_hsv_state',ctx=ast.Load()))]
    functions.body[1].body = [restore]
    namespace = {'np':np}
    exec(compile(ast.fix_missing_locations(functions),'<real filter rollback>','exec'),namespace)
    analyzer = SimpleNamespace(hsv_lower=np.array([20,50,130]),hsv_upper=np.array([85,255,255]),
                               using_alt2_hsv=True,using_alt6_hsv=False)
    previous = namespace['capture'](analyzer)
    analyzer.hsv_lower[:] = [70,0,30]
    analyzer.using_alt2_hsv = False
    analyzer.using_alt6_hsv = True
    namespace['restore'](analyzer,previous)
    assert analyzer.hsv_lower.tolist() == [20,50,130]
    assert analyzer.hsv_upper.tolist() == [85,255,255]
    assert analyzer.using_alt2_hsv is True
    assert analyzer.using_alt6_hsv is False
