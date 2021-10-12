import traceback

from . import formatting
from . import juliatools
from . import latextools

from openest.models.curve import FlatCurve


class CurveGenerator(object):
    """Abstract Base Class for curve generators

    Parameters
    ----------
    indepunits : sequence of strs
        Independent variable units.
    depenunit : str
        Dependent variable unit.
    """
    def __init__(self, indepunits, depenunit):
        self.indepunits = indepunits
        self.depenunit = depenunit

    def get_curve(self, region, year, *args, **kw):
        """Returns an object of type Curve."""
        raise NotImplementedError()

    def format_call(self, lang, *args):
        raise NotImplementedError()

    def get_partial_derivative_curvegen(self, covariate, covarunit):
        """
        Returns a CurveGenerator that calculates the partial
        derivative with respect to a covariate.
        """
        print(self.__class__)
        raise NotImplementedError()


class ConstantCurveGenerator(CurveGenerator):
    def __init__(self, indepunits, depenunit, curve):
        super(ConstantCurveGenerator, self).__init__(indepunits, depenunit)
        self.curve = curve

    def get_curve(self, region, year, *args, **kw):
        return self.curve

    def format_call(self, lang, *args):
        result = self.curve.format_call(lang, args[0])
        result['main'].unit = self.depenunit
        return result

    def get_partial_derivative_curvegen(self, covariate, covarunit):
        """
        Returns a CurveGenerator that calculates the partial
        derivative with respect to a covariate.
        """
        return ConstantCurveGenerator(self.indepunits, self.depenunit + '/' + covarunit, FlatCurve(0))
    
class TransformCurveGenerator(CurveGenerator):
    """
    Parameters
    ----------
    transform : callable
        Curve transformation function. Must take 'region' and *args
        argument of CurveGenerator-like objects.
    description : str
    curvegens : Sequence of CurveGenerator-like objects
    """
    def __init__(self, transform, description, *curvegens, partial_derivatve_transform=None):
        super(TransformCurveGenerator, self).__init__(curvegens[0].indepunits, curvegens[0].depenunit)
        assert description is not None, "Please provide a description."
        self.curvegens = curvegens
        self.description = description
        self.transform = transform
        self.deltamethod_passthrough = False
        self.partial_derivative_transform = partial_derivatve_transform

    def get_curve(self, region, year, *args, **kw):
        """

        Parameters
        ----------
        region : str
            Passed to `self.transform() and `get_curve()` for each
            CurveGenerator in `self.curvegens`.
        year : int
            Passed to `self.transform() and `get_curve()` for each
            CurveGenerator in `self.curvegens`.
        args :
            Passed on to `get_curve()` for each CurveGenerator in
            `self.curvegens`.
        kw :
            Passed on to `get_curve()`  for each CurveGenerator in
            `self.curvegens`.

        Returns
        -------
        Returned value from `self.transform()`.
        """
        try:
            return self.transform(region, *tuple(
                [curvegen.get_curve(region, year, *args, **kw) for curvegen in self.curvegens]))
        except Exception as ex:
            print(self.curvegens)
            print(args, kw)
            traceback.print_exc()
            raise ex

    def get_lincom_terms(self, region, year, predictors, origds):
        if self.deltamethod_passthrough:
            return self.curvegens[0].get_lincom_terms(region, year, predictors, origds)
        else:
            raise NotImplementedError("Cannot produce deltamethod terms for transform %s" % self.description)

    def get_lincom_terms_simple(self, predictors, covariates=None):
        if covariates is None:
            covariates = {}
        if self.deltamethod_passthrough:
            return self.curvegens[0].get_lincom_terms_simple(predictors, covariates)
        else:
            raise NotImplementedError("Cannot produce deltamethod terms for transform %s" % self.description)

    def format_call(self, lang, *args):
        if self.deltamethod_passthrough and len(self.curvegens):
            # No calculation change
            return self.curvegens[0].format_call(lang, *args)
            
        try:
            result = {}
            curveargs = []
            for curvegen in self.curvegens:
                equation = curvegen.format_call(lang, *args)
                result.update(equation)
                curveargs.append(equation['main'])

            if self.description is None and len(curveargs) == 1:
                # Pretend like nothing happened (used, e.g., for smart_curve transformation)
                pass  # already in main
            elif lang == 'latex':
                result.update(latextools.call(self.transform, self.description, *tuple(curveargs)))
            elif lang == 'julia':
                result.update(juliatools.call(self.transform, self.description, *tuple(curveargs)))
            
            return result
        except Exception as ex:
            print(self.curvegens)
            raise

    def get_partial_derivative_curvegen(self, covariate, covarunit):
        """
        Returns a CurveGenerator that calculates the partial
        derivative with respect to a covariate.
        """
        if self.deltamethod_passthrough:
            return self.curvegens[0].get_partial_derivative_curvegen(covariate, covarunit)
        elif self.partial_derivative_transform:
            derivcurvegens = [curvegen.get_partial_derivative_curvegen(covariate, covarunit) for curvegen in self.curvegens]
            return TransformCurveGenerator(self.partial_derivative_transform, "Partial derivative of " + self.description,
                                           *tuple(derivcurvegens), *tuple(curvegens))
        else:
            raise NotImplementedError("Cannot produce partial derivative for transform %s" % self.description)
    
class DelayedCurveGenerator(CurveGenerator):
    """

    Parameters
    ----------
    curvegen : CurveGenerator-like
    """
    def __init__(self, curvegen):
        super(DelayedCurveGenerator, self).__init__(curvegen.indepunits, curvegen.depenunit)
        self.curvegen = curvegen
        self.last_curves = {}
        self.last_years = {}
        
    def get_curve(self, region, year, *args, **kwargs):
        """

        Parameters
        ----------
        region : str
        year : int
        args:
            Passed to `self.get_next_curve()`.
        kwargs:
            Passed to `self.get_next_curve()`. Should have an entry for key
            'weather', which is poped before passing to
            `self.get_next_curve()`.

        Returns
        -------
        openest.generate.SmartCurve-like
        """
        if self.last_years.get(region, None) == year:
            return self.last_curves[region]
        
        if region not in self.last_curves:
            # Calculate no-weather before update covariates by calling with weather
            weather = kwargs['weather']
            del kwargs['weather']
            curve = self.get_next_curve(region, year, *args, **kwargs)
            kwargs['weather'] = weather
        else:
            curve = self.last_curves[region]

        self.last_curves[region] = self.get_next_curve(region, year, *args, **kwargs)
        self.last_years[region] = year
        return curve

    def get_next_curve(self, region, year, *args, **kwargs):
        """

        Parameters
        ----------
        region : str
            Passed to `self.curvegen.get_curve()`.
        year : int
            Passed to `self.curvegen.get_curve()`.
        args:
            Passed to `self.curvegen.get_curve()`.
        kwargs:
            Passed to `self.curvegen.get_curve()`.

        Returns
        -------
        openest.generate.SmartCurve-like

        """
        return self.curvegen.get_curve(region, year, *args, **kwargs)

    def format_call(self, lang, *args):
        argstrs = []
        extradeps = []
        elements = {}
        for arg in args:
            if isinstance(arg, str):
                argstrs.append(arg)
            else:
                if arg['main'].is_primitive:
                    argstrs.append(formatting.get_repstr(arg['main']))
                else:
                    var = get_variable()
                    argstrs.append(var)
                    args[var] = arg['main']
                    extradeps.append(var)
                elements.update(arg)

        #try:
        if lang == 'latex':
            elements.update(self.curvegen.format_call(lang, *tuple([x.replace('t', '{t-1}') for x in argstrs])))
        elif lang == 'julia':
            elements.update(self.curvegen.format_call(lang, *tuple([x.replace('t', 't-1') for x in argstrs])))
        elements['main'].unit = self.depenunit
        elements['main'].dependencies.extend(extradeps)
        return elements
        #except Exception as ex:
        #    print self.curvegen
        #    raise ex

    def get_partial_derivative_curvegen(self, covariate, covarunit):
        """
        Returns a CurveGenerator that calculates the partial
        derivative with respect to a covariate.
        """
        return DelayedCurveGenerator(self.curvegen.get_partial_derivative_curvegen(covariate, covarunit))

class FunctionCurveGenerator(CurveGenerator):
    def __init__(self, indepunits, depenunits, covargen, curvegenfunc):
        super(FunctionCurveGenerator, self).__init__(indepunits, depenunits)
        self.covargen = covargen
        self.curvegenfunc = curvegenfunc

    def get_curve(self, region, year, *args, **kwargs):
        covars = self.covargen.offer_update(region, year, kwargs['weather'])
        return self.curvegenfunc(covars)
