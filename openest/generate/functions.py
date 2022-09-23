import numpy as np
from . import juliatools, latextools, calculation, diagnostic, arguments, formatting
from .formatting import FormatElement, ParameterFormatElement

"""Scale the results by the value in scale_dict, or the mean value (if it is set).
make_generator: we encapsulate this function, passing in data and opporting on outputs
func: default operation is to multiple (scale), but can do other things (e.g., - for re-basing)
"""
class Scale(calculation.RecursiveCalculation):
    def __init__(self, subcalc, scale_dict, from_units, to_units, func=lambda x, y: x*y, latexpair=(r"\bar{I}", "Region-specific scaling")):
        super().__init__([subcalc], [to_units] + subcalc.unitses)
        assert(subcalc.unitses[0] == from_units)

        self.subcalc = subcalc
        self.scale_dict = scale_dict
        self.func = func
        self.latexpair = latexpair
        self.from_units = from_units

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        scaledesc = FormatElement(self.latexpair[1])
        if lang == 'latex':
            elements.update(latextools.call(self.func,
                                            "Scaling function", value,
                                            self.latexpair[0]))
            elements[self.latexpair[0]] = scaledesc
            return elements
        elif lang == 'julia':
            variable = formatting.get_variable()
            elements.update(latextools.call(self.func,
                                            "Scaling function", value, variable))
            elements[variable] = scaledesc
            return elements

    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            if region in self.scale_dict:
                return self.func(result, self.scale_dict[region])
            else:
                return self.func(result, self.scale_dict['mean'])

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Scaled ' + infos[0]['title']
        equation = latextools.english_function(self.func, infos[0]['name'], self.latexpair[1])
        description = "Computed from the %s variable, as %s." % (infos[0]['name'], equation)
        return [dict(name='scaled', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.region_dictionary, arguments.input_unit,
                               arguments.output_unit, arguments.input_reduce.optional(),
                               arguments.latexpair],
                    description="Scale each result by a region-specific value.")

"""
Transform all results by a function.
"""
class Transform(calculation.RecursiveCalculation):
    def __init__(self, subcalc, from_units, to_units, func, description, long_description):
        super().__init__([subcalc], [to_units] + subcalc.unitses)
        assert(subcalc.unitses[0] == from_units)

        self.subcalc = subcalc
        self.func = func
        self.description = description
        self.long_description = long_description
        self.from_units = from_units

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        if lang == 'latex':
            elements.update(latextools.call(self.func,
                                            self.long_description, elements['main']))
        elif lang == 'julia':
            elements.update(juliatools.call(self.func,
                                            self.long_description, elements['main']))
        return elements

    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return self.func(result)

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = self.description
        description = self.long_description
        return [dict(name='transformed', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.input_unit,
                               arguments.output_unit, arguments.input_change,
                               arguments.description, arguments.description.rename('long_description')],
                    description="Apply an arbitrary function to the results.")

    
class Instabase(calculation.CustomFunctionalCalculation):
    """Re-base the results of make_generator(...) to the values in baseyear
    baseyear is the year to use as the 'denominator'; None for the first year
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    Tacks on the value to the front of the results
    """

    def __init__(self, subcalc, baseyear, func=lambda x, y: x / y, units='portion', skip_on_missing=True, unshift=True):
        super(Instabase, self).__init__(subcalc, subcalc.unitses[0], units, unshift, baseyear, func, skip_on_missing)
        self.unshift = unshift
        self.baseyear = baseyear
        self.denom = None # The value in the baseyear
        self.pastresults = [] # results before baseyear

    def format_handler(self, equation, lang, baseyear, func, skip_on_missing):
        eqvar = formatting.get_variable(equation)
        if lang == 'latex':
            result = latextools.call(func, "Re-basing function", eqvar,
                                     r"\left[%s\right]_{t = %d}" % (eqvar, baseyear))
        elif lang == 'julia':
            result = juliatools.call(func, "Re-basing function", eqvar,
                                     "%s[findfirst(year .== %d)" % (eqvar, baseyear))
        result['main'].dependencies.append(eqvar)
        result[eqvar] = equation

        formatting.add_label('rebased', result)
        return result

    def init_apply(self):
        self.pastresults = [] # don't copy this across instances!

    def pushhandler(self, ds, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(ds):
            year = yearresult[0]
            result = yearresult[1]

            # Should we base everything off this year?
            if year == baseyear or (baseyear is None and self.denom is None):
                self.denom = result

                # Print out all past results, relative to this year
                for self.pastresult in self.pastresults:
                    yield [self.pastresult[0], func(self.pastresult[1], self.denom)] + list(self.pastresult[1:])

            if self.denom is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
            else:
                # calculate this and tack it on
                yield [year, func(result, self.denom)] + list(yearresult[1:])

    def donehandler(self, baseyear, func, skip_on_missing):
        if self.denom is None and skip_on_missing:
            # Never got to this year: just write out results
            for pastresult in self.pastresults:
                yield pastresult

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Rebased ' + infos[0]['title']
        description = "The result calculated relative to the year %d, by re-basing variable %s." % (self.baseyear, infos[0]['name'])
        return [dict(name='rebased', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.year,
                               arguments.input_reduce.optional(), arguments.output_unit.optional(),
                               arguments.skip_on_missing.optional()],
                    description="Translate all results relative to the baseline year.")

class SpanInstabase(Instabase):
    """Re-base the results of a calculation to the average of values between two years.
    Default func constructs a porportional change; x - y makes simple difference.
    skip_on_missing: If we never encounter the year and this is false,
      still print out the existing results.
    """
    def __init__(self, subcalc, year1, year2, func=lambda x, y: x - y, units='portion', skip_on_missing=True, unshift=True, baseline_diagname='baseline'):
        print(baseline_diagname)
        super(SpanInstabase, self).__init__(subcalc, (year1 + year2) / 2, func, units, skip_on_missing, unshift)
        self.year1 = year1
        self.year2 = year2
        self.denomterms = []
        self.baseline_diagname = baseline_diagname

    def format_handler(self, equation, lang, baseyear, func, skip_on_missing):
        eqvar = formatting.get_variable(equation)
        baselinevar = formatting.get_parametername(self.baseline_diagname, lang)
        if lang == 'latex':
            result = latextools.call(func, "Re-basing function", eqvar, baselinevar)
        elif lang == 'julia':
            result = juliatools.call(func, "Re-basing function", eqvar, baselinevar)
        if isinstance(eqvar, str):
            result['main'].dependencies.append(eqvar)
            result[eqvar] = equation
        else:
            # FormatElement
            result['main'].dependencies.extend(eqvar.dependencies)
        result[self.baseline_diagname] = ParameterFormatElement(self.baseline_diagname, formatting.get_parametername(self.baseline_diagname, lang))
        result['main'].dependencies.append(self.baseline_diagname)

        formatting.add_label('rebased', result)
        return result

    def init_apply(self):
        self.denomterms = [] # don't copy this across instances!
        self.pastresults = []

    def pushhandler(self, ds, baseyear, func, skip_on_missing):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(ds):
            year = yearresult[0]
            result = yearresult[1]

            # Should we base everything off this year?
            if year == self.year2:
                self.denomterms.append(result)
                if not self.deltamethod:
                    self.denom = np.mean(self.denomterms)
                else:
                    self.denom = np.mean(self.denomterms, 0)

                # Print out all past results, re-based
                for pastresult in self.pastresults:
                    diagnostic.record(self.region, pastresult[0], self.baseline_diagname, self.denom)
                    if self.unshift:
                        yield [pastresult[0], func(pastresult[1], self.denom)] + list(pastresult[1:])
                    else:
                        yield [pastresult[0], func(pastresult[1], self.denom)]

            if self.denom is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
                if year >= self.year1:
                    self.denomterms.append(result)
            else:
                diagnostic.record(self.region, year, self.baseline_diagname, self.denom)
                # calculate this and tack it on
                if self.unshift:
                    yield [year, func(result, self.denom)] + list(yearresult[1:])
                else:
                    yield [year, func(result, self.denom)]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.year.rename('year1').describe("The starting year"),
                               arguments.year.rename('year2').describe("The ending year"),
                               arguments.input_reduce.optional(), arguments.output_unit.optional(),
                               arguments.skip_on_missing.optional(), arguments.label.rename('baseline_diagname').optional()],
                    description="Translate all results relative to a span of baseline years.")

class InstaZScore(calculation.CustomFunctionalCalculation):
    """
    Collects up to `baseyear` of values and then uses them to represent all values as a z-score.
    """

    def __init__(self, subcalc, lastyear, units='z-score'):
        super(InstaZScore, self).__init__(subcalc, subcalc.unitses[0], units, True, lastyear)
        self.lastyear = lastyear
        self.mean = None # The mean to subtract off
        self.sdev = None # The sdev to divide by
        self.pastresults = [] # results before lastyear

    def init_apply(self):
        self.pastresults = [] # don't copy this across instances!

    def pushhandler(self, ds, lastyear):
        """
        Returns an interator of (yyyy, value, ...).
        """
        for yearresult in self.subapp.push(ds):
            year = yearresult[0]
            result = yearresult[1]

            # Have we collected all the data?
            if year == lastyear or (lastyear is None and self.mean is None):
                self.mean = np.mean([mx[1] for mx in self.pastresults])
                self.sdev = np.std([mx[1] for mx in self.pastresults])

                # Print out all past results, now that we have them
                for pastresult in self.pastresults:
                    yield [pastresult[0], (pastresult[1] - self.mean) / self.sdev] + list(pastresult[1:])

            if self.mean is None:
                # Keep track of this until we have a base
                self.pastresults.append(yearresult)
            else:
                # calculate this and tack it on
                yield [year, (result - self.mean) / self.sdev] + list(yearresult[1:])

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Z-Score of ' + infos[0]['title']
        description = "Z-scores of %s calculated relative to the years before %d." % (infos[0]['name'], self.lastyear)
        return [dict(name='zscore', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.year,
                               arguments.output_unit.optional()],
                    description="Translate all results to z-scores against results up to a given year.")


class Sum(calculation.RecursiveCalculation):
    """Sum two or more subcalculations

    Parameters
    ----------
    subcalcs : Sequence of ``openest.generate.calculation.Calculation``
    unshift : bool, optional
    """
    def __init__(self, subcalcs, unshift=True):
        fullunitses = subcalcs[0].unitses[:]
        for ii in range(1, len(subcalcs)):
            assert subcalcs[0].unitses[0] == subcalcs[ii].unitses[0], "%s <> %s" % (subcalcs[0].unitses[0], subcalcs[ii].unitses[0])
            fullunitses.extend(subcalcs[ii].unitses)
        if unshift:
            super().__init__(subcalcs, [subcalcs[0].unitses[0]] + fullunitses)
        else:
            super().__init__(subcalcs, [subcalcs[0].unitses[0]])

        self.unshift = unshift

    def format(self, lang, *args, **kwargs):
        mains = []
        elements = {}
        alldeps = set()
        for subcalc in self.subcalcs:
            elements.update(subcalc.format(lang, *args, **kwargs))
            mains.append(elements['main'])
            alldeps.update(elements['main'].dependencies)
            
        if lang in ['latex', 'julia']:
            elements['main'] = FormatElement(' + '.join([main.repstr for main in mains]), list(alldeps))

        formatting.add_label('sum', elements)
        return elements
        
    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """
        def generate(year, results):
            if not self.deltamethod:
                return np.sum([x[1] if x is not None else np.nan for x in results])
            else:
                return np.sum([x[1] if x is not None else np.nan for x in results], 0)

        # Prepare the generator from our encapsulated operations
        subapps = [subcalc.apply(region, *args, **kwargs) for subcalc in self.subcalcs]
        return calculation.ApplicationPassCall(region, subapps, generate, unshift=self.unshift)

    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infoses = [subcalc.column_info() for subcalc in self.subcalcs]
        title = 'Sum of previous results'
        description = 'Sum of ' + ', '.join([infos[0]['title'] for infos in infoses])

        fullinfos = []
        for infos in infoses:
            fullinfos.extend(infos)
        return [dict(name='sum', title=title, description=description)] + fullinfos

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates
        are supported.
        """
        return Sum([subcalc.partial_derivative(covariate, covarunit) for subcalc in self.subcalcs])

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculationss, arguments.unshift.optional()],
                    description="Sum the results of multiple previous calculations.")


class Product(calculation.RecursiveCalculation):
    """Product of two or more subcalculations

    Note that this does not support delta-method runs.

    Parameters
    ----------
    subcalcs : Sequence of ``openest.generate.calculation.Calculation``
    unshift : bool, optional
    """

    def __init__(self, subcalcs, unshift=True):
        # Unit of result is product of input subcalc units.
        units_product = [" * ".join([s.unitses[0] for s in subcalcs])]
        if unshift:
            fullunitses = [unit for calc in subcalcs for unit in calc.unitses]
            super().__init__(subcalcs, units_product + fullunitses)
        else:
            super().__init__(subcalcs, units_product)

        self.unshift = unshift

    def format(self, lang, *args, **kwargs):
        mains = []
        elements = {}
        alldeps = set()
        for subcalc in self.subcalcs:
            elements.update(subcalc.format(lang, *args, **kwargs))
            mains.append(elements['main'])
            alldeps.update(elements['main'].dependencies)

        if lang in ['latex', 'julia']:
            elements['main'] = FormatElement(' * '.join([main.repstr for main in mains]), list(alldeps))

        formatting.add_label('product', elements)
        return elements

    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """

        def generate(year, results):
            return np.prod([x[1] if x is not None else np.nan for x in results])

        # Prepare the generator from our encapsulated operations
        subapps = [subcalc.apply(region, *args, **kwargs) for subcalc in self.subcalcs]
        return calculation.ApplicationPassCall(region, subapps, generate, unshift=self.unshift)

    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infoses = [subcalc.column_info() for subcalc in self.subcalcs]
        title = 'Product of previous results'
        description = 'Product of ' + ', '.join([infos[0]['title'] for infos in infoses])
        fullinfos = [info for infos in infoses for info in infos]
        return [dict(name='product', title=title, description=description)] + fullinfos

    def enable_deltamethod(self):
        """Enable delta-method calculations

        Delta-method is unsupported for this calculation.
        """
        raise AttributeError(f'{self.__class__.__name__} does not support enabling deltamethod')

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates
        are supported.
        """
        # Partial deriv should be sum of products
        chain_products = []
        for i, subcalc in enumerate(self.subcalcs):

            # product of (∂subcalc / ∂covariate) and all other subcalcs
            chain_products.append(
                Product(
                    [subcalc.partial_derivative(covariate, covarunit)]
                    + self.subcalcs[:i]
                    + self.subcalcs[(i + 1):]
                )
            )

        return Sum(chain_products)

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculationss, arguments.unshift.optional()],
                    description="Product of results from multiple previous calculations.")


class FractionSum(calculation.RecursiveCalculation):
    """Sum of subcalculations weighted on fractions of unity.

    Parameters
    ----------
    subcalcs : Sequence of ``openest.generate.calculation.Calculation``
        Sequence of alternating values to be weighted and fraction weights.
        If there are p values to be weighted, there should be p - 1 weights,
        as the weighted value of the last weight is 1 - the sum of previous
        weights. Length of sequence must be odd. The units of the values to be
        weighted must be the same. Weights must be unitless.
    unshift : bool, optional
    """

    def __init__(self, subcalcs, unshift=True):

        if not len(subcalcs) % 2:
            raise ValueError("len of `subcalcs` should be odd")

        # Collect units. Check that fractions are unitless and other subcalcs
        # have same units.
        fullunitses = subcalcs[0].unitses[:]
        for ii in range(1, len(subcalcs)):
            if ii % 2:
                assert subcalcs[ii].unitses[0] == "unitless"
            else:
                assert subcalcs[0].unitses[0] == subcalcs[ii].unitses[0], "%s <> %s" % (subcalcs[0].unitses[0], subcalcs[ii].unitses[0])
            fullunitses.extend(subcalcs[ii].unitses)
        if unshift:
            super().__init__(subcalcs, [subcalcs[0].unitses[0]] + fullunitses)
        else:
            super().__init__(subcalcs, [subcalcs[0].unitses[0]])
        self.unshift = unshift

    def format(self, lang, *args, **kwargs):
        mains = []
        elements = {}
        alldeps = set()
        for subcalc in self.subcalcs:
            elements.update(subcalc.format(lang, *args, **kwargs))
            mains.append(elements['main'])
            alldeps.update(elements['main'].dependencies)

        if lang in ['latex', 'julia']:
            # Do str formatting that looks like math...
            subcalc_strs = [main.repstr for main in mains]

            # Collect products to sum. Start with all subcalcs but last.
            to_sum = []
            for i in range(0, len(subcalc_strs) - 1, 2):
                to_sum.append(' * '.join(subcalc_strs[i:i + 1]))

            # Put together the last product; it's weight's based on sum of
            # prev weights...
            last_weight = ' - '.join(subcalc_strs[1:-1:2])
            last_product = f"(1 - {last_weight}) * {subcalc_strs[-1]}"
            to_sum.append(last_product)

            elements['main'] = FormatElement(' + '.join(to_sum), list(alldeps))

        formatting.add_label(self.__class__.__name__.lower(), elements)
        return elements

    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """

        def generate(year, results):
            results_a = np.array([r[1] for r in results])  # Extract values.

            # Point to original array data. Values are not copied.
            values = results_a[::2]
            weights = results_a[1::2]

            if np.sum(weights) > 1 or np.any(weights < 0):
                raise ValueError("fraction weight results must be within [0, 1]")

            out = np.sum(weights * values[:-1])
            out += values[-1] * (1 - weights.sum())  # Add weighted last value.
            return out

        # Prepare the generator from our encapsulated operations
        subapps = [c.apply(region, *args, **kwargs) for c in self.subcalcs]
        return calculation.ApplicationPassCall(region, subapps, generate, unshift=self.unshift)

    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infoses = [subcalc.column_info() for subcalc in self.subcalcs]
        title = 'Sum of results weighted on fractions of unity'
        description = f'{self.__class__.__name__} of ' + ', '.join([infos[0]['title'] for infos in infoses])
        fullinfos = [info for infos in infoses for info in infos]
        return [dict(name=self.__class__.__name__.lower(), title=title, description=description)] + fullinfos

    def enable_deltamethod(self):
        """Enable delta-method calculations

        Delta-method is unsupported for this calculation.
        """
        raise AttributeError(f'{self.__class__.__name__} does not support enabling deltamethod')

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates
        are supported.
        """
        # Sum of products.
        # First we get all weight, values except last value.
        out = []
        for i in range(0, len(self.subcalcs) - 1, 2):
            out.append(Product(self.subcalcs[i:i + 1]))
        # Append negated weighted last value product before getting partial
        # from Sum.
        out.append(self.subcalcs[-1])
        final_weight = ConstantScale(Sum(self.subcalcs[1:-1:2]), -1.0)
        out.append(Product([final_weight, self.subcalcs[-1]]))
        return Sum(out).partial_derivative(covariate, covarunit)

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculationss, arguments.unshift.optional()],
                    description="Sum of subcalculations weighted on fractions of unity.")


"""
ConstantScale
"""
class ConstantScale(calculation.RecursiveCalculation):
    def __init__(self, subcalc, coeff):
        super().__init__([subcalc], [subcalc.unitses[0]] + subcalc.unitses)
        self.coeff = coeff

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
            
        if lang in ['latex', 'julia']:
            elements['main'] = FormatElement('%f * (%s)' % (self.coeff, elements['main'].repstr), elements['main'].dependencies)

        return elements
        
    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return self.coeff * result

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Previous result multiplied by %f' % self.coeff
        description = 'Previous result multiplied by %f' % self.coeff

        return [dict(name='constscale', title=title, description=description)] + infos

    def partial_derivative(self, covariate, covarunit):
        return Sum([subcalc.partial_derivative(covariate, covarunit) for subcalc in self.subcalcs],
                   unshift=self.unshift)

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.coefficient],
                    description="Multiply the result by a constant factor.")

class Positive(calculation.RecursiveCalculation):
    """
    Return 0 if subcalc is less than 0
    """
    def __init__(self, subcalc):
        super().__init__([subcalc], [subcalc.unitses[0]] + subcalc.unitses)
        self.subcalc = subcalc

    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return result if result > 0 else 0

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'Positive-only form of ' + infos[0]['title']
        description = 'The value of ' + infos[0]['title'] + ', if positive and otherwise 0.'

        return [dict(name='positive', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation],
                    description="Return the maximum of a previous result or 0.")

class Exponentiate(calculation.RecursiveCalculation):
    def __init__(self, subcalc, errorvar):
        assert subcalc.unitses[0][:3] == 'log'
        super().__init__([subcalc], [subcalc.unitses[0][3:].strip()] + subcalc.unitses)
        self.subcalc = subcalc
        self.errorvar = errorvar

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        if lang == 'latex':
            elements.update({'main': FormatElement(r"\exp{%s + %f/2}" % (elements['main'].repstr, self.errorvar))})
        elif lang == 'julia':
            elements.update({'main': FormatElement(r"exp(%s + %f/2)" % (elements['main'].repstr, self.errorvar))})
        return elements
            
    def apply(self, region, *args, **kwargs):
        def generate(year, result):
            return np.exp(result + 0.5*self.errorvar)

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def column_info(self):
        infos = self.subcalc.column_info()
        title = 'exp(' + infos[0]['title'] + ')'
        description = 'Exponentiation of ' + infos[0]['title']

        return [dict(name='exp', title=title, description=description)] + infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.variance.rename('errorvar')],
                    description="Return the the exponentiation of a previous result.")

class AuxiliaryResult(calculation.RecursiveCalculation):
    """
    Produce an additional output, but then pass the main result on.
    """
    def __init__(self, subcalc_main, subcalc_aux, auxname, keeplastonly=True):
        if keeplastonly:
            super().__init__([subcalc_main, subcalc_aux], [subcalc_main.unitses[0], subcalc_aux.unitses[0]] + subcalc_main.unitses[1:])
        else:
            super().__init__([subcalc_main, subcalc_aux], [subcalc_main.unitses[0]] + subcalc_aux.unitses + subcalc_main.unitses[1:])
        self.subcalc_main = subcalc_main
        self.subcalc_aux = subcalc_aux
        self.auxname = auxname
        self.keeplastonly = keeplastonly

    def format(self, lang, *args, **kwargs):
        beforeauxlen = len(formatting.format_labels)
        auxres = self.subcalc_aux.format(lang, *args, **kwargs)
        formatting.format_labels = formatting.format_labels[:beforeauxlen] # drop any new ones added
        formatting.add_label(self.auxname, auxres)
        return self.subcalc_main.format(lang, *args, **kwargs)

    def apply(self, region, *args, **kwargs):
        subapp_main = self.subcalc_main.apply(region, *args, **kwargs)
        subapp_aux = self.subcalc_aux.apply(region, *args, **kwargs)
        return AuxiliaryResultApplication(region, subapp_main, subapp_aux, self.keeplastonly)

    def partial_derivative(self, covariate, covarunit):
        """
        Returns a new calculation object that calculates the partial
        derivative with respect to a given variable; currently only covariates are supported.
        """
        return AuxiliaryResult(self.subcalc_main.partial_derivative(covariate, covarunit),
                               self.subcalc_aux.partial_derivative(covariate, covarunit), self.auxname)
        
    def column_info(self):
        infos_main = self.subcalc_main.column_info()
        infos_aux = self.subcalc_aux.column_info()
        infos_aux[0]['name'] = self.auxname

        if self.keeplastonly:
            return [infos_main[0], infos_aux[0]] + infos_main[1:]
        else:
            return [infos_main[0]] + infos_aux + infos_main[1:]

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.calculation.describe("An auxiliary calculation, placed behind the main calculation."), arguments.label],
                    description="Add an additional result to the columns.")


class AuxillaryResult(AuxiliaryResult):
    """Deprecated variation of AuxiliaryResult

    Emits a FutureWarning whenever used. Exists for backwards compatibility
    and legacy support.
    """
    def init(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "`AuxillaryResult` is deprecated, please use `AuxiliaryResult`",
            FutureWarning
        )
        super().__init__(*args, **kwargs)


class AuxiliaryResultApplication(calculation.Application):
    """
    Perform both main and auxiliary calculation, and order as main[0], aux, main[1:]
    """
    def __init__(self, region, subapp_main, subapp_aux, keeplastonly):
        super(AuxiliaryResultApplication, self).__init__(region)
        self.subapp_main = subapp_main
        self.subapp_aux = subapp_aux
        self.keeplastonly = keeplastonly

    def push(self, ds):
        for yearresult in self.subapp_main.push(ds):
            for yearresult_aux in self.subapp_aux.push(ds):
                pass # Just take the last one
            if self.keeplastonly:
                yield list(yearresult[0:2]) + [yearresult_aux[1]] + list(yearresult[2:])
            else:
                yield list(yearresult[0:2]) + list(yearresult_aux[1:]) + list(yearresult[2:])

    def done(self):
        self.subapp_aux.done()
        return self.subapp_main.done()


class AuxillaryResultApplication(AuxiliaryResultApplication):
    """Deprecated variation of AuxiliaryResultApplication

    Emits a FutureWarning whenever used. Exists for backwards compatibility
    and legacy support.
    """
    def init(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "`AuxillaryResultApplication` is deprecated, please use `AuxiliaryResultApplication`",
            FutureWarning
        )
        super().__init__(*args, **kwargs)


class KeepOnly(calculation.RecursiveCalculation):
    """
    Keep only a subset of the calculation results, with given names.
    """
    def __init__(self, subcalc, names):
        self.subcalc = subcalc
        self.iskept = [info['name'] in names for info in subcalc.column_info()]
        super().__init__([subcalc], self.keeplist(subcalc.unitses))

    def keeplist(self, lst):
        assert len(lst) == len(self.iskept), "Given %d <> %d when choosing which to keep." % (len(lst), len(self.iskept))
        return [lst[ii] for ii in range(len(lst)) if self.iskept[ii]]
        
    def format(self, lang, *args, **kwargs):
        return self.subcalc.format(lang, *args, **kwargs)
            
    def apply(self, region, *args, **kwargs):
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return KeepOnlyApplication(region, subapp, self.keeplist)

    def column_info(self):
        return self.keeplist(self.subcalc.column_info())

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.column_names],
                    description="Only keep a subset of the preliminary calculations.")

class KeepOnlyApplication(calculation.Application):
    def __init__(self, region, subapp, keeplist):
        super(KeepOnlyApplication, self).__init__(region)
        self.subapp = subapp
        self.keeplist = keeplist

    def push(self, ds):
        for yearresult in self.subapp.push(ds):
            yield [yearresult[0]] + self.keeplist(yearresult[1:])

    def done(self):
        return self.subapp.done()

class Clip(calculation.RecursiveCalculation):
    """Clip the values in a subcalculation.

    Given a (min, max) interval of values in a subcalculation, values outside
    of this interval are clipped to the interval edges. For example, with the
    intervals ``[0.0, 1.0]``, values less than 0 become 0 and values greater
    than 1 become 1.

    Parameters
    ----------
    subcalc : openest.generate.calculation.Calculation
    clip_min : float
    clip_max : float
    """
    def __init__(self, subcalc, clip_min, clip_max):
        super().__init__([subcalc], [subcalc.unitses[0]] + subcalc.unitses)
        self.subcalc = subcalc
        self.min = float(clip_min)
        self.max = float(clip_max)

    def apply(self, region, *args, **kwargs):
        """Apply calculation to all subcalculations (`self.subcalc`)

        All parameters are passed to ``self.subcalc.apply()``.

        Parameters
        ----------
        region : str
        args
        kwargs

        Returns
        -------
        openest.generate.Calculation.ApplicationPassCall
        """
        def generate(year, result):
            # This is where the actual clipping happens
            return max(self.min, min(result, self.max))

        # Prepare the generator from our encapsulated operations
        subapp = self.subcalc.apply(region, *args, **kwargs)
        return calculation.ApplicationPassCall(region, subapp, generate, unshift=True)

    def format(self, lang, *args, **kwargs):
        elements = self.subcalc.format(lang, *args, **kwargs)
        if lang == 'latex':
            elements['main'] = FormatElement("\\min(\\max(%s, %f), %f)" % (elements['main'].repstr, self.min, self.min),
                                             elements['main'].dependencies)
        elif lang == 'julia':
            elements['main'] = FormatElement("min(max(%s, %f), %f)" % (elements['main'].repstr, self.min, self.min),
                                             elements['main'].dependencies)
            
        return elements
    
    def column_info(self):
        """Get column information of values output from this calculation.

        Returns
        -------
        Sequence of dicts
            Each dict contains:

                ``"name"``
                    Short-length title of this calculation

                ``"title"``
                    Long-length title of this calculation

                ``"description"``
                    Long-form description of this calculation
        """
        infos = self.subcalc.column_info()
        title_str = str(infos[0]['title'])
        title = f'Clipped form of {title_str}'
        description = f'The value of {title_str}, clipped to the interval [{self.min}, {self.max}].'
        return [dict(name='clip', title=title, description=description)] + infos

    @staticmethod
    def describe():
        """Get computer-readable description of the calculation

        Returns
        -------
        dict
            This contains:

            ``"input_timerate"``
                Expected time rate of data, day, month, year, or any.

            ``"output_timerate"``
                Expected time rate of data, day, month, year, or same.

            ``"arguments"``
                A list of subclasses of arguments.ArgumentType, describing each
                constructor argument.

            ``"description"``
                Text description.

        """
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.numeric.rename('min'), arguments.numeric.rename('max')],
                    description="Return the clipped values of a previous result.")

class Reword(calculation.RecursiveCalculation):
    """Changes the label and/or description of a subcalculation.

    Parameters:
    subcalc : openest.generate.calculation.Calculation
    name : str
    description : str (optional)
    """
    def __init__(self, subcalc, name=None, title=None, description=None):
        super().__init__([subcalc], subcalc.unitses)
        self.subcalc = subcalc
        self.name = name
        self.title = title
        self.description = description

    def format(self, lang, *args, **kwargs):
        result = self.subcalc.format(lang, *args, **kwargs)
        formatting.add_label(self.name, result)
        return result
        
    def apply(self, region, *args, **kwargs):
        return self.subcalc.apply(region, *args, **kwargs)

    def column_info(self):
        infos = self.subcalc.column_info()
        if self.name is not None:
            infos[0]['name'] = self.name
        if self.title is not None:
            infos[0]['title'] = self.title
        if self.description is not None:
            infos[0]['description'] = self.description
        return infos

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.label.rename('name').optional(),
                               arguments.label.rename('title').optional(), arguments.description.optional()],
                    description="Change the name and/or description of a result.")

class SequentialProcess(calculation.RecursiveCalculation):
    def __init__(self, step1, step2, reportstep1=False):
        if reportstep1:
            super().__init__([step2, step1], step2.unitses + step1.unitses)
        else:
            super().__init__([step2, step1], step2.unitses)
        self.step1 = step1
        self.step2 = step2
        self.reportstep1 = reportstep1

    def format(self, lang, *args, **kwargs):
        beforeauxlen = len(formatting.format_labels)
        auxres = self.step1.format(lang, *args, **kwargs)
        formatting.format_labels = formatting.format_labels[:beforeauxlen] # drop any new ones added
        return self.step2.format(lang, *args, **kwargs)

    def apply(self, region, *args, **kwargs):
        app1 = self.step1.apply(region, *args, **kwargs)
        app2 = self.step2.apply(region, *args, **kwargs)
        return SequentialProcessApplication(region, app1, app2, self.reportstep1)

    def partial_derivative(self, covariate, covarunit):
        return SequentialProcess(self.step1.partial_derivative(covariate, covarunit),
                                 self.step2.partial_derivative(covariate, covarunit))


    def column_info(self):
        if self.reportstep1:
            return self.step2.column_info() + self.step1.column_info()
        else:
            return self.step2.column_info()

    @staticmethod
    def describe():
        return dict(input_timerate='any', output_timerate='same',
                    arguments=[arguments.calculation, arguments.calculation],
                    description="Perform a first calculation, then return the result of a second.")

class SequentialProcessApplication(calculation.Application):
    def __init__(self, region, app1, app2, reportapp1):
        super(SequentialProcessApplication, self).__init__(region)
        self.app1 = app1
        self.app2 = app2
        self.reportapp1 = reportapp1
        
    def push(self, ds):
        for yearresult1 in self.app1.push(ds):
            for yearresult2 in self.app2.push(ds):
                pass
            if self.reportapp1:
                yield list(yearresult2) + list(yearresult1[1:])
            else:
                yield yearresult2

    def done(self):
        self.app1.done()
        return self.app2.done()
