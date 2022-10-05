import numpy as np
import pandas as pd
import xarray as xr
import pytest

from openest.generate.base import Constant
from openest.generate.daily import YearlyDayBins
from openest.generate.functions import Scale, Instabase, SpanInstabase, Clip, Sum, Product, FractionSum
from .test_daily import test_curve


class MockAppCalc:
    """Mocks openest.generate.calculation.CustomFunctionalCalculation-like for ease

    This primarily allows us to prime the Application generators with simple
    Sequences. Is it an Application? Is it a Calculation? It's kinda both!
    """
    def __init__(self, years, values, unitses):
        self.unitses = unitses
        # Mimic generator nature of the data, test if data leaks.
        self.data = zip(years, values)

    def apply(self, *args, **kwargs):
        return self

    def column_info(self):
        return [{'name': 'mockname', 'title': 'mocktitle', 'description': 'mockdescription'}]

    def push(self, *args, **kwargs):
        for x in self.data:
            yield x


def make_year_ds(year, values):
    year = str(year)
    time_coord = pd.date_range(year + '-01-01', periods=365)
    return xr.Dataset({'x': (['time'], values)},
                      coords={'time': time_coord})


def test_scale():
    application = Scale(Constant(33, 'widgets'), {'test': 1. / 11}, 'widgets', 'subwigs').test()
    for yearresult in application.push(make_year_ds(2000, np.random.rand(365))):
        np.testing.assert_equal(yearresult[1], 3)

    def check_units():
        app = Scale(Constant(33, 'widgets'), {'test': 1.8}, 'deg. C', 'deg F.').test()
        for _ in app.push(make_year_ds(2000, np.random.rand(365))):
            return

    np.testing.assert_raises(Exception, check_units)


def test_instabase():
    calc = Instabase(YearlyDayBins(test_curve, 'days', lambda ds: ds['x']), 2)
    app1 = calc.test()
    app2 = calc.test()
    for _ in app1.push(make_year_ds(1800, [-10] * 300 + [10] * 65)):
        raise AssertionError  # Should get nothing here
    for _ in app2.push(make_year_ds(1800, [10] * 365)):
        raise AssertionError  # Should get nothing here
    for yearresult in app1.push(make_year_ds(2000, [-10] * 65 + [10] * 300)):
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 65. / 300.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
    for yearresult in app2.push(make_year_ds(2000, [10] * 365)):
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 1.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)


def test_spaninstabase():
    calc = SpanInstabase(YearlyDayBins(test_curve, 'days', lambda ds: ds['x']), 2, 2)
    app1 = calc.test()
    app2 = calc.test()
    for _ in app1.push(make_year_ds(1800, [-10] * 300 + [10] * 65)):
        raise AssertionError  # Should get nothing here
    for _ in app2.push(make_year_ds(1800, [10] * 365)):
        raise AssertionError  # Should get nothing here
    for yearresult in app1.push(make_year_ds(2000, [-10] * 65 + [10] * 300)):
        print(app1.denomterms)
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 65. / 300.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
    for yearresult in app2.push(make_year_ds(2000, [10] * 365)):
        print(app2.denomterms)
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 1.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)


class TestSum:
    """
    Basic tests for openest.generate.functions.Sum
    """
    @pytest.mark.parametrize(
        "unshift_flag,n_expected",
        [(True, 3), (False, 1)],
        ids=['shifted', 'unshifted'],
    )
    def test_units_append(self, unshift_flag, n_expected):
        """Tests that Sum instances correctly append units from the original subcalcs
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[1.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[2.0], unitses=unit)

        sum_calc = Sum([subcalc_mock1, subcalc_mock2], unshift=unshift_flag)
        assert sum_calc.unitses == unit * n_expected

    def test_apply(self):
        """Test Sum.apply() actually sums values
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[1.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[2.0], unitses=unit)

        sum_calc = Sum([subcalc_mock1, subcalc_mock2])
        victim_gen = sum_calc.apply('foobar_region').push('not_a_ds')
        assert next(victim_gen) == [0, 3.0, 1.0, 2.0]

    def test_apply_memory(self):
        """Test that Sum.apply() doesn't hold memory between yields
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0, 1], values=[1.0, 2.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0, 1], values=[2.0, 3.0], unitses=unit)
        sum_calc = Sum([subcalc_mock1, subcalc_mock2])

        victim_gen = sum_calc.apply('foobar_region').push('not_a_ds')
        iter0 = next(victim_gen)
        iter1 = next(victim_gen)
        assert (iter0 == [0, 3.0, 1.0, 2.0]) and (iter1 == [1, 5.0, 2.0, 3.0])

    def test_column_info(self):
        """Ensure Sum.column_info() appends dict with correct keys
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[1.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        sum_calc = Sum([subcalc_mock1, subcalc_mock2])
        victim = sum_calc.column_info()
        # Check that first dict is from Sum instance
        assert victim[0]['name'] == 'sum'
        assert list(victim[0].keys()) == ['name', 'title', 'description']

    def test_describe(self):
        """Ensure Sum.describe() returns dict with correct keys
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[1.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        sum_calc = Sum([subcalc_mock1, subcalc_mock2])
        victim = sum_calc.describe()
        assert list(victim.keys()) == ['input_timerate', 'output_timerate', 'arguments', 'description']


class TestProduct:
    """
    Basic tests for openest.generate.functions.Product
    """
    @pytest.mark.parametrize(
        "unshift_flag",
        [(True), (False)],
        ids=['shifted', 'unshifted'],
    )
    def test_units_append(self, unshift_flag):
        """Tests that Product correctly tacks on units from the original subcalcs
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[3.0], unitses=unit)
        in_calcs = [subcalc_mock1, subcalc_mock2]

        prod_calc = Product(in_calcs, unshift=unshift_flag)

        expected = [" * ".join(unit * len(in_calcs))]
        if unshift_flag:
            expected += unit * len(in_calcs)
        assert prod_calc.unitses == expected

    def test_apply(self):
        """Test Product.apply() actually multiplies values
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[3.0], unitses=unit)

        prod_calc = Product([subcalc_mock1, subcalc_mock2])
        victim_gen = prod_calc.apply('foobar_region').push('not_a_ds')
        assert next(victim_gen) == [0, 6.0, 2.0, 3.0]

    def test_apply_memory(self):
        """Test that Product.apply() doesn't hold memory between yields
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0, 1], values=[2.0, 3.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0, 1], values=[3.0, 4.0], unitses=unit)
        prod_calc = Product([subcalc_mock1, subcalc_mock2])

        victim_gen = prod_calc.apply('foobar_region').push('not_a_ds')
        iter0 = next(victim_gen)
        iter1 = next(victim_gen)
        assert (iter0 == [0, 6.0, 2.0, 3.0]) and (iter1 == [1, 12.0, 3.0, 4.0])

    def test_column_info(self):
        """Ensure Product.column_info() appends dict with correct keys
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[3.0], unitses=unit)
        prod_calc = Product([subcalc_mock1, subcalc_mock2])
        victim = prod_calc.column_info()
        # Check that first dict is from Sum instance
        assert victim[0]['name'] == 'product'
        assert list(victim[0].keys()) == ['name', 'title', 'description']

    def test_describe(self):
        """Ensure Product.describe() returns dict with correct keys
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[3.0], unitses=unit)
        prod_calc = Product([subcalc_mock1, subcalc_mock2])
        victim = prod_calc.describe()
        assert list(victim.keys()) == ['input_timerate', 'output_timerate', 'arguments', 'description']

    def test_enable_deltamethod_exception(self):
        """Ensure that Product.enable_deltamethod() raises an exception
        """
        unit = ['fakeunit']
        subcalc_mock1 = MockAppCalc(years=[0], values=[2.0], unitses=unit)
        subcalc_mock2 = MockAppCalc(years=[0], values=[3.0], unitses=unit)
        prod_calc = Product([subcalc_mock1, subcalc_mock2])

        with pytest.raises(AttributeError):
            prod_calc.enable_deltamethod()


class TestFractionSum:
    """
    Basic tests for openest.generate.functions.FractionSum
    """
    def _subcalcs_stuffer(self, values1, values2, fvalues, years, unit):
        """Helper returning list of subcalcs for vanilla FunctionSum input"""
        frac_mock = MockAppCalc(years=years, values=fvalues,
                                unitses=["unitless"])
        subcalc_mock1 = MockAppCalc(years=years, values=values1, unitses=unit)
        subcalc_mock2 = MockAppCalc(years=years, values=values2, unitses=unit)
        return [subcalc_mock1, frac_mock, subcalc_mock2]

    @pytest.mark.parametrize(
        "unshift_flag,n_expected",
        [(True, 2), (False, 1)],
        ids=['shifted', 'unshifted'],
    )
    def test_units_append(self, unshift_flag, n_expected):
        """Tests that FractionSum correctly appends units from the original subcalcs
        """
        unit = ['fakeunit']
        subcalcs = self._subcalcs_stuffer(values1=[1.0], values2=[2.0],
                                          fvalues=[0.1], years=[0], unit=unit)
        fsum_calc = FractionSum(subcalcs, unshift=unshift_flag)

        if unshift_flag:
            expected_units = unit + [unit for c in subcalcs for unit in c.unitses]
        else:
            expected_units = unit
        assert fsum_calc.unitses == expected_units

    def test_apply(self):
        """Test FractionSum.apply() does math good
        """
        unit = ['fakeunit']
        subcalcs = self._subcalcs_stuffer(values1=[1.0], values2=[2.0],
                                          fvalues=[0.1], years=[0], unit=unit)
        fsum_calc = FractionSum(subcalcs)
        victim_gen = fsum_calc.apply('foobar_region').push('not_a_ds')
        assert np.allclose(next(victim_gen), [0, 1.9, 1.0, 0.1, 2.0])

    @pytest.mark.parametrize(
        "fracweight",
        [-2, 1.2],
        ids=['too low', 'too high'],
    )
    def test_apply_oobweight(self, fracweight):
        """Test FractionSum.apply() raises except if fraction weight outside [0, 1]
        """
        unit = ['fakeunit']
        subcalcs = self._subcalcs_stuffer(values1=[1.0], values2=[2.0],
                                          fvalues=[fracweight], years=[0],
                                          unit=unit)
        fsum_calc = FractionSum(subcalcs)
        victim_gen = fsum_calc.apply('foobar_region').push('not_a_ds')
        with pytest.raises(ValueError):
            next(victim_gen)

    def test_apply_memory(self):
        """Test that Sum.apply() doesn't hold memory between yields
        """
        unit = ['fakeunit']
        subcalcs = self._subcalcs_stuffer(values1=[1.0, 2.0], values2=[2.0, 3.0],
                                          fvalues=[0.1, 0.2], years=[0, 1], unit=unit)
        fsum_calc = FractionSum(subcalcs)

        victim_gen = fsum_calc.apply('foobar_region').push('not_a_ds')
        iter0 = next(victim_gen)
        iter1 = next(victim_gen)
        assert (np.allclose(iter0, [0, 1.9, 1.0, 0.1, 2.0])
                and np.allclose(iter1, [1, 2.8, 2.0, 0.2, 3.0]))

    def test_column_info(self):
        """Ensure Sum.column_info() appends dict with correct keys
        """
        unit = ['fakeunit']
        subcalcs = self._subcalcs_stuffer(values1=[1.0], values2=[2.0],
                                          fvalues=[0.1], years=[0], unit=unit)
        fsum_calc = FractionSum(subcalcs)
        victim = fsum_calc.column_info()
        # Check that first dict is from Sum instance
        assert victim[0]['name'] == 'fractionsum'
        assert list(victim[0].keys()) == ['name', 'title', 'description']

    def test_describe(self):
        """Ensure Sum.describe() returns dict with correct keys
        """
        unit = ['fakeunit']
        subcalcs = self._subcalcs_stuffer(values1=[1.0], values2=[2.0],
                                          fvalues=[0.1], years=[0], unit=unit)
        fsum_calc = FractionSum(subcalcs)
        victim = fsum_calc.describe()
        assert list(victim.keys()) == ['input_timerate', 'output_timerate', 'arguments', 'description']


class TestClip:
    """Basic tests for openest.generate.functions.Clip
    """
    def test_units_append(self):
        """Tests that Clip instances correctly append units from the original subcalc
        """
        unit = ['fakeunit']
        subcalc_mock = MockAppCalc(years=[0], values=[1.0], unitses=unit)
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        assert clipped_calc.unitses == unit * 2

    @pytest.mark.parametrize(
        "in_years,in_values,expected",
        [
            ([0], [-0.5], [0, 0.0, -0.5]),
            ([1], [0.0], [1, 0.0, 0.0]),
            ([2], [0.5], [2, 0.5, 0.5]),
        ],
        ids=['outside clip', 'clip edge', 'inside clip'],
    )
    def test_apply(self, in_years, in_values, expected):
        """Test Clip.apply() clips values in, out, and on edge of interval
        """
        subcalc_mock = MockAppCalc(
            years=in_years,
            values=in_values,
            unitses=['fakeunit'],
        )
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim_gen = clipped_calc.apply('foobar_region').push('not_a_ds')
        assert next(victim_gen) == expected

    def test_apply_memory(self):
        """Test that Clip.apply() doesn't hold memory between yields
        """
        subcalc_mock = MockAppCalc(
            years=[0, 1, 2],
            values=[0.0, 0.5, 1.0],
            unitses=['fakeunit'],
        )
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim_gen = clipped_calc.apply('foobar_region').push('not_a_ds')
        iter0 = next(victim_gen)
        iter1 = next(victim_gen)
        iter2 = next(victim_gen)
        assert (iter0 == [0, 0.0, 0.0]) and (iter1 == [1, 0.5, 0.5]) and (iter2 == [2, 1.0, 1.0])

    def test_column_info(self):
        """Ensure Clip.column_info() appends dict with correct keys
        """
        subcalc_mock = MockAppCalc(years=[0], values=[1.0], unitses=['fakeunit'])
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim = clipped_calc.column_info()
        # Check that first dict is from Clip instance
        assert victim[0]['name'] == 'clip'
        assert list(victim[0].keys()) == ['name', 'title', 'description']

    def test_describe(self):
        """Ensure Clip.describe() returns dict with correct keys
        """
        subcalc_mock = MockAppCalc(years=[0], values=[1.0], unitses=['fakeunit'])
        clipped_calc = Clip(subcalc_mock, 0.0, 1.0)
        victim = clipped_calc.describe()
        assert list(victim.keys()) == ['input_timerate', 'output_timerate', 'arguments', 'description']

class TestSequentialProcess:
    """Tests for openest.generate.functions.SequentialProcess
    """
    def test_doublereport(self):
        subcalc_step1 = MockAppCalc(years=[0], values=[1.0], unitses=['none'])
        subcalc_step2 = MockAppCalc(years=[0], values=[2.0], unitses=['none'])
        seqcalc = SequentialProcess(subcalc_step1, subcalc_step2, reportstep1=True)
        for yearresult in seqcalc.apply('a region').push('not a ds'):
            np.testing.assert_equal(yearresult[1], 2.)
            np.testing.assert_equal(yearresult[1], 1.)
