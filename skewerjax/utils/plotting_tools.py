import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
import itertools
import numpy as np


def ten_power_formatter(y, pos):
    # Zero stays zero
    if np.isclose(y, 0.0):
        return "0"
    # Get exponent
    exp = int(np.round(np.log10(abs(y))))
    # Only label exact powers of ten, otherwise empty label
    if np.isclose(abs(y), 10.0**exp):
        return rf"$10^{{{exp}}}$"
    return ""