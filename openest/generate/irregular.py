import numpy as np
import xarray as xr
from .calculation import Calculation, ApplicationByIrregular
from . import formatting, arguments, diagnostic
from .formatting import FormatElement
from .curvegen import CurveGenerator

try:
    import bottleneck as bn
except ImportError:
    # Use numpy instead...
    bn = np

class YearlySumIrregular(Calculation):
    def __init__(self, units, curvegen, curve_description):
        super(YearlySumIrregular, self).__init__([units])
        assert isinstance(curvegen, CurveGenerator)

        self.curvegen = curvegen
        self.curve_description = curve_description

    def format(self, lang):
        variable = formatting.get_variable()
        if lang == 'latex':
            result = self.curvegen.format_call(lang, variable + "_t")
            result.update({'main': FormatElement(r"\sum_{s \in y(t)} %s" % result['main'].repstr,
                                                 [variable + '_t'] + result['main'].dependencies),
                           variable + '_t': FormatElement("Weather", is_abstract=True)})
        elif lang == 'julia':
            result = self.curvegen.format_call(lang, variable + "[t]")
            result.update({'main': FormatElement(r"sum(%s)" % result['main'].repstr,
                                                 [variable + '[t]'] + result['main'].dependencies),
                           variable + '[t]': FormatElement("Weather", is_abstract=True)})

        formatting.add_label('response', result)
            
        return result

    def apply(self, region, *args):
        checks = dict(lastyear=-np.inf)

        def generate(region, year, weather, **kw):
            # Ensure that we aren't called with a year twice
            assert year > checks['lastyear'], "Push of %d, but already did %d." % (year, checks['lastyear'])
            checks['lastyear'] = year

            curve = self.curvegen.get_curve(region, year, *args, weather=weather) # Passing in original (not weather-changed) data

            result = bn.nansum(curve(weather))

            if isinstance(weather, xr.Dataset):
                for var in weather._variables:
                    if var not in ['time', 'year'] and var not in weather.coords:
                        diagnostic.record(region, year, var, float(bn.nansum(weather._variables[var]._data)))
            else:
                diagnostic.record(region, year, 'avgv', float(bn.nansum(weather)))
                        
            if not np.isnan(result):
                yield year, result

        return ApplicationByIrregular(region, generate)

    def column_info(self):
        description = "The total result across a year of weather applied to " + self.curve_description
        return [dict(name='response', title='Direct marginal response', description=description)]

    def partial_derivative(self, covariate, covarunit):
        return YearlySumIrregular(self.unitses[0] + '/' + covarunit,
                                  self.curvegen.get_partial_derivative_curvegen(covariate, covarunit),
                                  "Partial derivative of " + self.curve_description + " w.r.t. " + covariate)

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='year',
                    arguments=[arguments.output_unit.rename('units'),
                               arguments.curvegen, arguments.curve_description],
                    description="Apply a curve to values and take the sum over each year.")
