import numpy as np
from daily import *
from functions import *
from shortterm import *

def logscalefunc(x, s):
    return s * (np.exp(x) - 1)

def c2f(x):
    return x * 1.8 + 32

