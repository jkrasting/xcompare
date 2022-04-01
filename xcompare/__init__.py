""" xcompare: tool for comparing two xarray objects """

from .version import __version__

from . import compare_xy
from . import coord_util
from . import standard_grid

from . import plot
from . import xr_stats
from . import xcompare
from .xcompare import compare_datasets
from .plot import plot_three_panel
