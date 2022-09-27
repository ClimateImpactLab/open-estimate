import pytest
from unittest.mock import MagicMock
from openest.generate.curvegen import (CurveGenerator,
                                       DelayedCurveGenerator,
                                       TransformCurveGenerator)


@pytest.fixture()
def mock_curvegen():
    """Return mocked CurveGenerator

    The Curve Generator has a MagicMocked `get_curve()` method,
    returns 'foobar'.
    """
    cg = CurveGenerator(['indepunits'], 'depenunit')
    cg.get_curve = MagicMock(return_value='foobar')
    return cg


def test_transformcurvegenerator(mock_curvegen):
    """Ensure TransformedCurveGenerator.get_curve() does get_curve() correctly"""
    mock_curvegen_list = [mock_curvegen]
    # Transform prints output "curve" and args as uppercase strings.
    tcg = TransformCurveGenerator(lambda *a: str(a).upper(),
                                  'this_is_a_description',
                                  *mock_curvegen_list)

    args_in = ['region', 1984, 'foobar']
    kwargs_in = {'knights_say': 'ni! ni!'}
    victim = tcg.get_curve(*args_in, **kwargs_in)

    tcg.curvegens[0].get_curve.assert_called_with(*args_in, **kwargs_in)
    assert victim == "('REGION', 'FOOBAR')"


def test_delayedcurvegenerator(mock_curvegen):
    """Check that DelayedCurveGenerator delays are handled correctly"""
    dcg = DelayedCurveGenerator(mock_curvegen)
    year = 1984
    args_in = ['a_region', year, 'foobar']
    kwargs_in = {'weather': 'Its warm'}

    victim1 = dcg.get_curve(*args_in, **kwargs_in)
    assert victim1 == 'foobar'
    assert dcg.last_years == {'a_region': 1984}
    assert dcg.last_curves == {'a_region': 'foobar'}

    dcg.curve_cache['a_region'] = 'barfoo'
    victim2 = dcg.get_curve(*args_in, **kwargs_in)
    assert victim2 == 'barfoo'

    args_in = ['a_region', year + 1, 'foobar']
    dcg.last_curves['a_region'] = 'barfoo'
    victim3 = dcg.get_curve(*args_in, **kwargs_in)
    assert victim3 == 'barfoo'
    assert dcg.last_years == {'a_region': 1985}
    assert dcg.last_curves == {'a_region': 'foobar'}
