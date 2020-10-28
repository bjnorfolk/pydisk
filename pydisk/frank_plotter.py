"""Frankenstein is an open source code that uses a Gaussian process to reconstruct the 1D radial brightness profile of a disc non-parametrically.

The Frank_Plotter class contains all tools necessary to read, model, and plot visibilities.
"""

import numpy as np

from .utils import PACKAGES

class Frank_Plotter:
	"""