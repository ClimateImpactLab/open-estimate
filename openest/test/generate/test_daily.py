import numpy as np
from openest.models.bin_model import BinModel
from openest.models.spline_model import SplineModel
from openest.generate.daily import YearlyDayBins, Constant
from openest.generate.weatherslice import DailyWeatherSlice

test_model = BinModel([-40, 0, 80], SplineModel.create_gaussian({0: (0, 1), 1: (1, 1)}, order=range(2)))

def test_YearlyDayBins():
    application = YearlyDayBins(test_model, 'days').test()
    for (year, result) in application.push(DailyWeatherSlice(1800 + np.arange(365), [-10] * 300 + [10] * 65)):
        np.testing.assert_equal(result, 65.)
    for (year, result) in application.push(DailyWeatherSlice(2000 + np.arange(365), [-10] * 65 + [10] * 300)):
        np.testing.assert_equal(result, 300.)

def test_Constant():
    application = Constant(33, 'widgets').test()
    for (year, result) in application.push(DailyWeatherSlice(1800 + np.arange(365), np.random.rand(365))):
        np.testing.assert_equal(result, 33)
    for (year, result) in application.push(DailyWeatherSlice(2000 + np.arange(365), np.random.rand(365))):
        np.testing.assert_equal(result, 33)
