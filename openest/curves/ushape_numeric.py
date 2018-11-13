class UShapedCurve(UnivariateCurve):
    def __init__(self, curve, mintemp, gettas):
        super(UShapedCurve, self).__init__(curve.xx)
        self.curve = curve
        self.mintemp = mintemp
        self.gettas = gettas

    def __call__(self, xs):
        values = self.curve(xs)
        tas = self.gettas(xs)
        order = np.argsort(tas)
        orderedtas = tas[order]
        orderedvalues = values[order]

        lowvalues = orderedvalues[orderedtas < self.mintemp]
        lowvalues2 = np.maximum.accumulate(lowvalues[::-1])

        highvalues = orderedvalues[orderedtas >= self.mintemp]
        highvalues2 = np.maximum.accumulate(highvalues)

        return np.concatenate((lowvalues2, highvalues2))

# Return tmarginal evaluated at the innermost edge of plateaus
class UShapedClipping(UnivariateCurve):
    def __init__(self, curve, tmarginal_curve, mintemp, gettas):
        super(UShapedClipping, self).__init__(curve.xx)
        self.curve = curve
        self.tmarginal_curve = tmarginal_curve
        self.mintemp = mintemp
        self.gettas = gettas

    def __call__(self, xs):
        increasingvalues = self.curve(xs) # these are ordered as low..., high...
        increasingplateaus = np.diff(increasingvalues) == 0

        tas = self.gettas(xs)
        order = np.argsort(tas)
        orderedtas = tas[order]

        n_below = sum(orderedtas < self.mintemp)
        
        lowindicesofordered = np.arange(n_below)[::-1] # [N-1 ... 0]
        if len(lowindicesofordered) > 1:
            lowindicesofordered[np.concatenate(([False], increasingplateaus[:len(lowindicesofordered)-1]))] = n_below
            lowindicesofordered = np.minimum.accumulate(lowindicesofordered)
        
        highindicesofordered = np.arange(sum(orderedtas >= self.mintemp)) + n_below # [N ... T-1]
        if len(highindicesofordered) > 1:
            highindicesofordered[np.concatenate(([False], increasingplateaus[-len(highindicesofordered)+1:]))] = n_below
            highindicesofordered = np.maximum.accumulate(highindicesofordered)

        increasingresults = np.concatenate((self.tmarginal_curve(xs[order[lowindicesofordered], :]), self.tmarginal_curve(xs[order[highindicesofordered], :]))) # ordered low..., high...
        increasingresults[increasingvalues <= 0] = 0 # replace truly clipped with 0

        return increasingresults
