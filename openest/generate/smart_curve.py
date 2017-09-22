import numpy as np
import juliatools, latextools, formatting
from statsmodels.distributions.empirical_distribution import StepFunction

## Smart Curves fall back on Curve logic, but take xarray DataSets and know which variables they want

class SmartCurve(object):
    def __init__(self):
        self.xx = [-np.inf, np.inf] # Backwards compatibility to functions expecting curves
    
    def __call__(self, ds):
        raise NotImplementedError("call not implemented")

    def format(self, lang, dsname):
        raise NotImplementedError()

    @staticmethod
    def format_call(lang, curve, *args):
        if isinstance(curve, SmartCurve):
            if len(args) == 1:
                return curve.format(lang, args[0])
            else:
                return curve.format(lang, '[' + ', '.join(args) + ']')

        if lang == 'latex':
            return latextools.call(curve, None, None, *args)
        elif lang == 'julia':
            return juliatools.call(curve, None, None, *args)

class CurveCurve(SmartCurve):
    def __init__(self, curve, variable):
        super(CurveCurve, self).__init__()
        self.curve = curve
        self.variable = variable

    def __call__(self, ds):
        return self.curve(ds[self.variable])

    def format(self, lang, dsname):
        return SmartCurve.format_call(self.curve, lang, "%s[%s]" % (dsname, self.variable))

class ConstantCurve(SmartCurve):
    def __init__(self, constant, dimension):
        super(ConstantCurve, self).__init__()
        self.constant = constant
        self.dimension = dimension

    def __call__(self, ds):
        return np.repeat(self.constant, len(ds[self.dimension]))

    def format(self, lang, dsname):
        return {'main': FormatElement(str(self.contant), None)}
    
class LinearCurve(CurveCurve):
    def __init__(self, slope, variable):
        super(LinearCurve, self).__init__(lambda x: slope * x, variable)

class StepCurve(CurveCurve):
    def __init__(self, xxlimits, levels, variable):
        step_function = StepFunction(xxlimits[1:-1], levels[1:], ival=levels[0])
        super(StepCurve, self).__init__(step_function, variable)

        self.xxlimits = xxlimits
        self.levels = levels

class CoefficientsCurve(SmartCurve):
    def __init__(self, coeffs, variables):
        super(CoefficientsCurve, self).__init__()
        self.coeffs = coeffs
        self.variables = variables

        assert isinstance(variables, list) and len(variables) == len(coeffs)

    def __call__(self, ds):
        result = np.zeros(ds[self.variables[0]].shape)
        for ii in range(len(self.variables)):
            #result += self.coeffs[ii] * ds[self.variables[ii]].values # TOO SLOW
            result += self.coeffs[ii] * ds._variables[self.variables[ii]]._data
            
        return result

    def format(self, lang, dsname):
        coeffvar = formatting.get_variable()
        if lang == 'latex':
            return {'main': FormatElement(r"(%s) \cdot \vec{%s}" % (', '.join(["%s[%s]" % (dsname, varname) for varname in self.variables]), coeffvar), None)}
        elif lang == 'julia':
            return {'main': FormatElement(' + '.join(["%s[%s] * %s_%d" % (dsname, self.variables[ii], coeffvar, ii + 1) for ii in range(len(self.variables))]), None)}

class SelectiveInputCurve(SmartCurve):
    """Assumes input is a matrix, and only pass selected input columns to child curve."""
    
    def __init__(self, curve, variable):
        super(SelectiveInputCurve, self).__init__()
        self.curve = curve
        self.variable = variable

    def __call__(self, ds):
        return self.curve(ds[self.variable]._data)

    def format(self, lang, dsname):
        return SmartCurve.format_call(self.curve, lang, "%s[%s]" % (dsname, self.variable))
