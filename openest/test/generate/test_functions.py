import numpy as np
import xarray as xr
from openest.generate.base import Constant
from openest.generate.daily import YearlyDayBins
from openest.generate.functions import Scale, Instabase, SpanInstabase
from test_daily import test_model

def make_year_ds(year, values):
    return xr.Dataset({'x': (['time'], values)},
                      coords={'time': map(lambda nn: datetime.date(year, 1, 1) + datetime.timedelta(nn), np.arange(365))})

def test_Scale():
    application = Scale(Constant(33, 'widgets'), {'test': 1. / 11}, 'widgets', 'subwigs').test()
    for yearresult in application.push(make_year_ds(2000, np.random.rand(365))):
        np.testing.assert_equal(yearresult[1], 3)

    def check_units():
        application = Scale(Constant(33, 'widgets'), {'test': 1.8}, 'deg. C', 'deg F.').test()
        for yearresult in application.push(make_year_ds(2000, np.random.rand(365))):
            return

    np.testing.assert_raises(Exception, check_units)

def test_Instabase():
    calc = Instabase(YearlyDayBins(test_model, 'days'), 2)
    app1 = calc.test()
    app2 = calc.test()
    for yearresult in app1.push(make_year_ds(1800, [-10] * 300 + [10] * 65)):
        np.testing.assert_equal(True, False) # Should get nothing here
    for yearresult in app2.push(make_year_ds(1800, [10] * 365)):
        np.testing.assert_equal(True, False) # Should get nothing here
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


def test_SpanInstabase():
    calc = SpanInstabase(YearlyDayBins(test_model, 'days'), 2, 2)
    app1 = calc.test()
    app2 = calc.test()
    for yearresult in app1.push(make_year_ds(1800, [-10] * 300 + [10] * 65)):
        np.testing.assert_equal(True, False) # Should get nothing here
    for yearresult in app2.push(make_year_ds(1800, [10] * 365)):
        np.testing.assert_equal(True, False) # Should get nothing here
    for yearresult in app1.push(make_year_ds(2000, [-10] * 65 + [10] * 300)):
        print app1.denomterms
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 65. / 300.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)
    for yearresult in app2.push(make_year_ds(2000, [10] * 365)):
        print app2.denomterms
        if yearresult[0] == 1:
            np.testing.assert_equal(yearresult[1], 1.)
        if yearresult[0] == 2:
            np.testing.assert_equal(yearresult[1], 1.)


