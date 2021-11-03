import numpy as np
import xarray as xr
from openest.generate import fast_dataset
from openest.curves.basic import UnivariateCurve


class UShapedCurve(UnivariateCurve):
    """A curve that returns values that clips another curve to maintain a u-shape.

    Parameters
    ----------
    curve : UnivariateCurve
        The curve to be clipped.
    midtemp : float
        The temperature forced to be at the lowest point in the curve.
    gettas : function(xs) -> array_like
        Returns the core temperature variable from whatever the curve is called with.
    ordered : boolean or 'maintain'
        If 'maintain', return order according to tas; if True, increasing tas order
    fillxxs : array_like
        Temperature values for a grid of additional sampled temperatures
    fillyys : array_like
        The dependent variables that correspond to fillxxs
    direction : string
        Either boatpose (u-shaped) or downdog (inverse-u-shaped)
    """
    def __init__(self, curve, midtemp, gettas, ordered='maintain', fillxxs=None, fillyys=None, direction='boatpose'):
        super(UShapedCurve, self).__init__(curve.xx)
        self.curve = curve
        self.midtemp = midtemp
        self.gettas = gettas
        self.ordered = ordered
        self.fillxxs = [] if fillxxs is None else fillxxs
        self.fillyys = [] if fillyys is None else fillyys
        assert direction in ['boatpose', 'downdog'], "Unknown direction for u-shaped clipping."
        self.direction = direction

    def __call__(self, xs):
        values = self.curve(xs)
        tas = tas_original = np.array(self.gettas(xs))

        # Add in the grid for completeness
        if len(self.fillxxs) > 0:
            tas = np.concatenate((tas, self.fillxxs))
            values = np.concatenate((values, self.fillyys))

        # Order into increasing values left and right of midtemp
        order = np.argsort(tas)
        orderedtas = tas[order]
        orderedvalues = values[order]

        lowvalues = orderedvalues[orderedtas < self.midtemp]
        if self.direction == 'boatpose':
            lowvalues2 = np.maximum.accumulate(lowvalues[::-1])
        elif self.direction == 'downdog':
            lowvalues2 = np.minimum.accumulate(lowvalues[::-1])

        highvalues = orderedvalues[orderedtas >= self.midtemp]
        if self.direction == 'boatpose':
            highvalues2 = np.maximum.accumulate(highvalues)
        elif self.direction == 'downdog':
            highvalues2 = np.minimum.accumulate(highvalues)

        if self.ordered == 'maintain':
            # Put it all back together
            increasing = np.concatenate((lowvalues2[::-1], highvalues2))
            # Undo the ordering
            tasorder = np.empty(len(increasing))
            tasorder[order] = increasing
            # Return just the given values
            return tasorder[:len(tas_original)]
        else:
            if len(self.fillxxs) > 0:
                tokeeps = order < len(tas_original)
                lowkeeps = tokeeps[orderedtas < self.midtemp]
                highkeeps = tokeeps[orderedtas >= self.midtemp]

                # Remove the fillins
                lowvalues2 = lowvalues2[lowkeeps[::-1]]
                highvalues2 = highvalues2[highkeeps]

            if self.ordered:
                return np.concatenate((lowvalues2[::-1], highvalues2))
            else:
                return np.concatenate((lowvalues2, highvalues2))

        
class UShapedDynamicCurve(UnivariateCurve):
    def __init__(self, curve, midtemp, gettas, unicurve, ordered='maintain', numfills=50, direction='boatpose'):
        super(UShapedDynamicCurve, self).__init__(curve.xx)
        self.curve = curve
        self.midtemp = midtemp
        self.gettas = gettas
        self.unicurve = unicurve
        self.ordered = ordered
        self.numfills = numfills
        self.direction = direction

    def __call__(self, xs):
        tas = self.gettas(xs)
        fillxxs = np.arange(np.min(tas), np.max(tas), self.numfills)[1:-1]
        fillyys = self.unicurve(tas)
        ucurve = UShapedCurve(self.curve, self.midtemp, self.gettas, self.ordered, fillxxs, fillyys, direction=self.direction)
        return ucurve(xs)

    
# Return tmarginal evaluated at the innermost edge of plateaus
class UShapedClipping(UnivariateCurve):
    """Clip marginal effects to zero where a curve is u-clipped

    Parameters
    ----------
    curve : UnivariateCurve
        Curve that has been u-clipped (but may not be a UShapedCurve).
    tmarginal_curve : UnivariateCurve
        Curve representing the marginal effects wrt temperature.
    midtemp : float
        The temperature forced to be at the lowest point in the curve.
    gettas : function(xs) -> array_like
        Returns the core temperature variable from whatever the curve is called with.
    ordered : boolean or 'maintain'
        Must use the same value as the corresponding UShapedCurve
    direction : string
        Either boatpose (u-shaped) or downdog (inverse-u-shaped)
    """
    def __init__(self, curve, tmarginal_curve, midtemp, gettas, ordered='maintain', direction='boatpose'):
        super(UShapedClipping, self).__init__(curve.xx)
        self.curve = curve
        self.tmarginal_curve = tmarginal_curve
        self.midtemp = midtemp
        self.gettas = gettas
        self.ordered = ordered
        assert direction in ['boatpose', 'downdog'], "Unknown direction for u-shaped clipping."
        self.direction = direction

    def __call__(self, xs):
        tas = np.array(self.gettas(xs))
        order = np.argsort(tas)
        orderedtas = tas[order]

        n_below = sum(orderedtas < self.midtemp)
        
        if self.ordered == 'maintain':
            # Order the clipped values to low..., high...
            clippedvalues = self.curve(xs)
            orderedvalues = clippedvalues[order]

            increasingvalues = np.concatenate((orderedvalues[:n_below][::-1], orderedvalues[n_below:]))
        else:
            increasingvalues = self.curve(xs)  # these are ordered as low..., high...

        # Find the plateaus
        increasingplateaus = np.diff(increasingvalues) == 0

        # Apply tmarginal at index just-before clipping (boat raises all)
        lowindicesofordered = np.arange(n_below)[::-1]  # [N-1 ... 0]
        if len(lowindicesofordered) > 1:
            lowindicesofordered[np.concatenate(([False], increasingplateaus[:len(lowindicesofordered)-1]))] = n_below
            if self.direction == 'boatpose':
                lowindicesofordered = np.minimum.accumulate(lowindicesofordered)
            elif self.direction == 'downdog':
                lowindicesofordered = np.maximum.accumulate(lowindicesofordered)
                
        highindicesofordered = np.arange(sum(orderedtas >= self.midtemp)) + n_below  # [N ... T-1]
        if len(highindicesofordered) > 1:
            highindicesofordered[np.concatenate(([False], increasingplateaus[-len(highindicesofordered)+1:]))] = n_below
            if self.direction == 'boatpose':
                highindicesofordered = np.maximum.accumulate(highindicesofordered)
            elif self.direction == 'downdog':
                highindicesofordered = np.minimum.accumulate(highindicesofordered)

        # Construct the results
        if isinstance(xs, xr.Dataset):
            newxs = fast_dataset.reorder_coord(xs, 'time', np.concatenate((order[lowindicesofordered], order[highindicesofordered])))
            increasingresults = self.tmarginal_curve(newxs)  # ordered low..., high...
        elif len(xs.shape) == 2:
            increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered], :]),
                                                self.tmarginal_curve(xs[order[highindicesofordered], :])))  # ordered low..., high...
        else:
            increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered]]),
                                                self.tmarginal_curve(xs[order[highindicesofordered]])))  # ordered low..., high...
        increasingresults[increasingvalues <= 0] = 0  # replace truly clipped with 0

        if self.ordered == 'maintain':
            orderedresults = np.concatenate((increasingresults[:n_below][::-1], increasingresults[n_below:]))
            
            # Undo the ordering
            tasorder = np.empty(len(orderedresults))
            tasorder[order] = orderedresults
            return tasorder
        elif self.ordered:
            return np.concatenate((increasingresults[tas < self.midtemp][::-1], increasingresults[tas >= self.midtemp]))
        else:
            return increasingresults
