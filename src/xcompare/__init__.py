""" xcompare: GFDL-Specific tool for comparing Xarray Datsets """

import importlib.metadata as ilm

msg = ilm.metadata("xcompare")

__name__ = msg["Name"]
__version__ = msg["Version"]
__license__ = msg["License"]
__email__ = msg["Maintainer-email"]
__description__ = msg["Summary"]
__requires__ = msg["Requires-Dist"]
__requires_python__ = msg["Requires-Python"]

from . import mom
from . import plot
from . import xr_stats
from . import xcompare

from .xcompare import compare_datasets
from .xcompare import AREA_VARS
from .xcompare import TIME_DIMS
from .xcompare import LAT_DIMS
from .xcompare import LON_DIMS
from .xcompare import Z_DIMS
from .xcompare import infer_dim_name
from .xcompare import infer_var_name
from .xcompare import ordered_list_extraction
from .xcompare import dataset_vars
from .xcompare import extract_var_from_dataset
from .xcompare import reorder_dims
from .xcompare import equal_horiz_dims

from .plot import plot_three_panel
