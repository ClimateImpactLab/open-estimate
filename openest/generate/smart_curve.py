"""Curve classes that apply to xarray Datasets.

Curves are mathematical functions on one or more independent
variables. The basic form of the curves classes is in
`models/curve.py`. The curve classes defined here, derived from
`SmartCurve`, take Datasets as arguments.

Smart Curves fall back on Curve logic, but take xarray DataSets and
know which variables they want.
"""

import numpy as np
from . import juliatools, latextools, formatting, diagnostic, formattools
from statsmodels.distributions.empirical_distribution import StepFunction
from openest.models import curve as curve_module
    
class SmartCurve(object):
    def __init__(self):
        self.xx = [-np.inf, np.inf] # Backwards compatibility to functions expecting curves
        self.deltamethod = False
    
    def __call__(self, ds):
        raise NotImplementedError("call not implemented")

    @property
    def univariate(self):
        raise NotImplementedError("univariate not implemented")
    
    def format(self, lang):
        raise NotImplementedError()

    @staticmethod
    def format_call(lang, curve, *args):
        if isinstance(curve, SmartCurve):
            return curve.format(lang)

        if lang == 'latex':
            return latextools.call(curve, None, *args)
        elif lang == 'julia':
            return juliatools.call(curve, None, *args)

class CurveCurve(SmartCurve):
    def __init__(self, curve, variable):
        super(CurveCurve, self).__init__()
        self.curve = curve
        self.variable = variable

    def __call__(self, ds):
        return self.curve(ds[self.variable])

    def format(self, lang):
        return SmartCurve.format_call(self.curve, lang, self.variable)

class ConstantCurve(SmartCurve):
    def __init__(self, constant, dimension):
        super(ConstantCurve, self).__init__()
        self.constant = constant
        self.dimension = dimension

    def __call__(self, ds):
        return np.repeat(self.constant, len(ds[self.dimension]))

    def format(self, lang):
        return {'main': formatting.FormatElement(str(self.constant))}
    
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

        assert isinstance(variables, list) and len(variables) == len(coeffs), "Variables do not match coefficients: %s <> %s" % (variables, coeffs)

    def __call__(self, ds):
        result = np.zeros(ds[self.variables[0]].shape)
        for ii in range(len(self.variables)):
            #result += self.coeffs[ii] * ds[self.variables[ii]].values # TOO SLOW
            result += self.coeffs[ii] * ds._variables[self.variables[ii]]._data
            
        return result

    def format(self, lang):
        coeffvar = formatting.get_variable()
        if lang == 'latex':
            return {'main': formatting.FormatElement(r"(%s) \cdot \vec{%s}" % (', '.join([varname for varname in self.variables]), coeffvar))}
        elif lang == 'julia':
            return {'main': formatting.FormatElement(' + '.join(["%s * %s_%d" % (self.variables[ii], coeffvar, ii + 1) for ii in range(len(self.variables))]))}

class ZeroInterceptPolynomialCurve(CoefficientsCurve):
    def __init__(self, coeffs, variables, allow_raising=False, descriptions=None):
        super(ZeroInterceptPolynomialCurve, self).__init__(coeffs, variables)
        if descriptions is None:
            descriptions = {}
        self.allow_raising = allow_raising
        self.descriptions = descriptions

        self.getters = [((lambda ds, var=variable: ds._variables[var]) if isinstance(variable, str) else variable)
                        for variable in self.variables]
    
    def __call__(self, ds):
        result = self.coeffs[0] * self.getters[0](ds)._data
            
        for ii in range(1, len(self.variables)):
            if not self.allow_raising or self.variables[ii] in ds._variables:
                #result += self.coeffs[ii] * ds[self.variables[ii]].values # TOO SLOW
                result += self.coeffs[ii] * self.getters[ii](ds)._data
            else:
                result += self.coeffs[ii] * (self.getters[0](ds)._data ** (ii + 1))
                    
        return result

    @property
    def univariate(self):
        return curve_module.ZeroInterceptPolynomialCurve([-np.inf, np.inf], self.coeffs)
    
    def format(self, lang):
        coeffvar = formatting.get_variable()
        variable = formatting.get_variable()
        funcvars = {}

        repterms = []
        if lang == 'latex':
            if isinstance(self.variables[0], str):
                repterms.append(r"%s_1 %s" % (coeffvar, variable))
            else:
                funcvar = formatting.get_function()
                funcvars[self.variables[0]] = funcvar
                repterms.append(r"%s_1 %s(%s)" % (coeffvar, funcvar, variable))
        elif lang == 'julia':
            if isinstance(self.variables[0], str):
                repterms.append(r"%s[1] * %s" % (coeffvar, variable))
            else:
                funcvar = formatting.get_function()
                funcvars[self.variables[0]] = funcvar
                repterms.append(r"%s[1] * %s(%s)" % (coeffvar, funcvar, variable))
        
        for ii in range(1, len(self.variables)):
            if lang == 'latex':
                if isinstance(self.variables[0], str):
                    repterms.append(r"%s_1 %s^%d" % (coeffvar, variable, ii + 1))
                else:
                    funcvar = formatting.get_function()
                    funcvars[self.variables[ii]] = funcvar
                    repterms.append(r"%s_1 %s(%s)^%d" % (coeffvar, funcvar, variable, ii + 1))
            elif lang == 'julia':
                if isinstance(self.variables[0], str):
                    repterms.append(r"%s[1] * %s^%d" % (coeffvar, variable, ii + 1))
                else:
                    funcvar = formatting.get_function()
                    funcvars[self.variables[ii]] = funcvar
                    repterms.append(r"%s[1] * %s(%s)^%d" % (coeffvar, funcvar, variable, ii + 1))

        result = {'main': formatting.FormatElement(' + '.join(repterms))}
        for variable in funcvars:
            result[funcvars[variable]] = formatting.FormatElement(self.descriptions.get(variable, "Unknown"))

        return result

    
class SumByTimePolynomialCurve(SmartCurve):
    """Equivalent to `ZeroInterceptPolynomialCurve`, but with a different coefficient per timestep.

    Parameters
    ----------
    coeffmat : array_like
        Matrix of K (order) x T (timesteps)
    variables : list of str or function
        Name of variable in DataSet or getter function for each exponent term
    allow_raising : bool, optional
        Can we just raise the linear term to an exponent, or should each bein the ds (default)
    descriptions : dict of str => str
        Description of each getter function
    """
    def __init__(self, coeffmat, variables, allow_raising=False, descriptions=None):
        super(SumByTimePolynomialCurve, self).__init__()
        self.coeffmat = coeffmat # K x T
        assert len(self.coeffmat.shape) == 2
        self.variables = variables
        self.allow_raising = allow_raising
        if descriptions is None:
            descriptions = {}
        self.descriptions = descriptions

        self.getters = [(lambda ds: ds._variables[variable]) if isinstance(variable, str) else variable for variable in self.variables] # functions return vector of length T

    def __call__(self, ds):
        maxtime = self.coeffmat.shape[1]
        lindata = self.getters[0](ds)._data[:maxtime]
        result = np.sum(self.coeffmat[0, :len(lindata)] * lindata)
        
        for ii in range(1, len(self.variables)):
            if not self.allow_raising or self.variables[ii] in ds._variables:
                termdata = self.getters[ii](ds)._data[:maxtime]
                result += np.sum(self.coeffmat[ii, :len(lindata)] * termdata) # throws error if length mismatch
            else:
                result += np.sum(self.coeffmat[ii, :len(lindata)] * (lindata ** (ii + 1)))
                
        return result

    @property
    def univariate(self):
        raise NotImplementedError("Probably want to define a matrix-taking curve before this.")

    def format(self, lang):
        coeffvar = formatting.get_variable()
        variable = formatting.get_variable()
        funcvars = {}

        repterms = []
        if lang == 'latex':
            if isinstance(self.variables[0], str):
                repterms.append(r"%s_1 \cdot %s" % (coeffvar, variable))
            else:
                funcvar = formatting.get_function()
                funcvars[self.variables[0]] = funcvar
                repterms.append(r"%s_1 \cdot %s(%s)" % (coeffvar, funcvar, variable))
        elif lang == 'julia':
            if isinstance(self.variables[0], str):
                repterms.append(r"sum(%s[1,:] * %s)" % (coeffvar, variable))
            else:
                funcvar = formatting.get_function()
                funcvars[self.variables[0]] = funcvar
                repterms.append(r"sum(%s[1,:] * %s(%s))" % (coeffvar, funcvar, variable))
        
        for ii in range(1, len(self.variables)):
            if lang == 'latex':
                if isinstance(self.variables[0], str):
                    repterms.append(r"%s_1 \cdot %s^%d" % (coeffvar, variable, ii + 1))
                else:
                    funcvar = formatting.get_function()
                    funcvars[self.variables[ii]] = funcvar
                    repterms.append(r"%s_1 \cdot %s(%s)^%d" % (coeffvar, funcvar, variable, ii + 1))
            elif lang == 'julia':
                if isinstance(self.variables[0], str):
                    repterms.append(r"sum(%s[1,:] * %s^%d)" % (coeffvar, variable, ii + 1))
                else:
                    funcvar = formatting.get_function()
                    funcvars[self.variables[ii]] = funcvar
                    repterms.append(r"sum(%s[1,:] * %s(%s)^%d)" % (coeffvar, funcvar, variable, ii + 1))

        result = {'main': formatting.FormatElement(' + '.join(repterms))}
        for variable in funcvars:
            result[funcvars[variable]] = formatting.FormatElement(self.descriptions.get(variable, "Unknown"))

        return result

class SumByTimeCoefficientsCurve(SmartCurve):
    """Equivalent to `TransformCoefficientsCurve`, but with a different coefficient per timestep.

    Parameters
    ----------
    coeffmat : array_like
        Matrix of K (#predictors) x T (timesteps)
    transforms : list of functions
        Functions of DataSet to return each predictor
    descriptions : list of str
        Descriptions of each transformation/predictor
    diagnames : list of str
        Keys to be used for each predictor in the diagnostic files, or None for no-recording
    """
    def __init__(self, coeffmat, transforms, descriptions, diagnames=None):
        super(SumByTimeCoefficientsCurve, self).__init__()
        self.coeffmat = coeffmat # K x T
        assert len(coeffmat.shape) == 2 or np.all(coeffmat == 0)
        self.transforms = transforms
        self.descriptions = descriptions
        self.diagnames = diagnames

        assert isinstance(transforms, list) and len(transforms) == coeffmat.shape[0], "Transforms do not match coefficients: %s <> %s" % (transforms, coeffmat.shape)
        assert diagnames is None or isinstance(diagnames, list) and len(diagnames) == len(transforms)

    def __call__(self, ds):
        if np.all(self.coeffmat == 0):
            # Happens with edge case of conditional suffixes
            return 0
            
        maxtime = self.coeffmat.shape[1]

        result = None
        for ii in range(len(self.transforms)):
            predictor = self.transforms[ii](ds)._data.ravel()[:maxtime]
            if self.diagnames:
                diagnostic.record(ds.region, ds.year, self.diagnames[ii], np.sum(predictor))
            if result is None:
                result = np.sum(self.coeffmat[ii, :] * predictor)
            else:
                result += np.sum(self.coeffmat[ii, :] * predictor)
                
        return result

    @property
    def univariate(self):
        raise NotImplementedError("Probably want to define a matrix-taking curve before this.")

    def format(self, lang):
        raise NotImplementedError()
    
class CubicSplineCurve(CoefficientsCurve):
    def __init__(self, coeffs, knots, variables, allow_raising=False):
        super(CubicSplineCurve, self).__init__(coeffs, variables)
        self.allow_raising = allow_raising
        self.knots = knots

    
    def __call__(self, ds):
        result = np.zeros(ds[self.variables[0]].shape)

        try:
            for ii in range(len(self.variables)):
                result += self.coeffs[ii] * ds._variables[self.variables[ii]]._data

            return result
        except KeyError as ex:
            # This should only catch KeyErrors coming from coming from
            # ds._variables[x].
            if self.allow_raising:
                return curve_module.CubicSplineCurve(self.knots, self.coeffs)(ds._variables[self.variables[0]]._data)
            raise ex

    @property
    def univariate(self):
        return curve_module.CubicSplineCurve(self.knots, self.coeffs)

class TransformCoefficientsCurve(SmartCurve):
    """Use a transformation of ds to produce each predictor.

    Parameters
    ----------
    coeffs : array_like
        Vector of coefficients on each [transformed] predictor
    transforms : list of functions
        Functions of DataSet to return each predictor
    descriptions : list of str
        Descriptions of each transformation/predictor
    diagnames : list of str (optional)
        Keys to be used for each predictor in the diagnostic files, or None for no-recording
    univariate_curve : UnivariateCurve (optional)
        If a univariate function is requested, can we produce one?
    """
    def __init__(self, coeffs, transforms, descriptions, diagnames=None, univariate_curve=None):
        super(TransformCoefficientsCurve, self).__init__()
        self.coeffs = coeffs
        self.transforms = transforms
        self.descriptions = descriptions
        self.diagnames = diagnames
        self._univariate_curve = univariate_curve

        assert isinstance(transforms, list) and len(transforms) == len(coeffs), "Transforms do not match coefficients: %s <> %s" % (transforms, coeffs)
        assert diagnames is None or isinstance(diagnames, list) and len(diagnames) == len(transforms)

    def __call__(self, ds):
        result = None
        for ii in range(len(self.transforms)):
            predictor = self.transforms[ii](ds)
            if self.diagnames:
                diagnostic.record(ds.region, ds.year, self.diagnames[ii], np.sum(predictor._data))
            if result is None:
                result = self.coeffs[ii] * predictor._data
            else:
                result += self.coeffs[ii] * predictor._data

        return result

    def format(self, lang):
        coeffvar = formatting.get_variable()
        funcvars = [formatting.get_function() for transform in self.transforms]
        if lang == 'latex':
            result = {'main': formatting.FormatElement(r"(%s) \cdot \vec{%s}" % (', '.join(["%s" % funcvars[ii] for ii in range(len(funcvars))]), coeffvar))}
        elif lang == 'julia':
            result = {'main': formatting.FormatElement(' + '.join(["%s() * %s_%d" % (funcvars[ii], coeffvar, ii + 1) for ii in range(len(funcvars))]))}

        for ii in range(len(funcvars)):
            result[funcvars[ii]] = formatting.FormatElement(self.descriptions[ii])

        return result

    @property
    def univariate(self):
        if self._univariate_curve is not None:
            return self._univariate_curve

        raise NotImplementedError("univariate transform not specified")

class SelectiveInputCurve(SmartCurve):
    """Assumes input is a matrix, and only pass selected input columns to child curve."""
    
    def __init__(self, curve, variable):
        super(SelectiveInputCurve, self).__init__()
        self.curve = curve
        self.variable = variable

    def __call__(self, ds):
        return self.curve(ds[self.variable]._data)

    def format(self, lang, dsname):
        return SmartCurve.format_call(self.curve, lang, self.variable)
    
class SumCurve(SmartCurve):
    def __init__(self, curves):
        super(SmartCurve, self).__init__()
        self.curves = curves

    def __call__(self, ds):
        total = 0
        for curve in self.curves:
            total += curve(ds)
        return total

    def format(self, lang):
        formatteds = [SmartCurve.format_call(self.curves[ii], lang, self.variable) for ii in range(len(self.curves))]
        return formattools.join(' + ', formatteds)

class ProductCurve(SmartCurve):
    def __init__(self, curve1, curve2):
        super(ProductCurve, self).__init__()
        self.curve1 = curve1
        self.curve2 = curve2

    def __call__(self, ds):
        return self.curve1(ds) * self.curve2(ds)

    def format(self, lang):
        return formatting.build_recursive({'latex': r"(%s) (%s)",
                                           'julia': r"(%s) .* (%s)"}, lang,
                                          self.curve1, self.curve2)

class ShiftedCurve(SmartCurve):
    def __init__(self, curve, offset):
        super(ShiftedCurve, self).__init__()
        self.curve = curve
        self.offset = offset

    def __call__(self, ds):
        return self.curve(ds) + self.offset

    @property
    def univariate(self):
        return curve_module.ShiftedCurve(self.curve.univariate, self.offset)
    
    def format(self, lang):
        return formatting.build_recursive({'latex': r"(%s + " + str(self.offset) + ")",
                                           'julia': r"(%s + " + str(self.offset) + ")"},
                                          lang, self.curve)

class ClippedCurve(curve_module.ClippedCurve, SmartCurve):
    @property
    def univariate(self):
        return curve_module.ClippedCurve(self.curve.univariate, self.cliplow)

class OtherClippedCurve(curve_module.OtherClippedCurve, SmartCurve):
    @property
    def univariate(self):
        return curve_module.OtherClippedCurve(self.clipping_curve.univariate, self.curve.univariate, self.clipy)

class MinimumCurve(curve_module.MinimumCurve, SmartCurve):
    @property
    def univariate(self):
        return curve_module.MinimumCurve(self.curve1.univariate, self.curve2.univariate)

class MaximumCurve(curve_module.MaximumCurve, SmartCurve):
    @property
    def univariate(self):
        return curve_module.MaximumCurve(self.curve1.univariate, self.curve2.univariate)

