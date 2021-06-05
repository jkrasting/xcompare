""" Main module for xcompare """

import warnings
import numpy as np
import xarray as xr
import xesmf as xe

from .xr_stats import xr_stats_2d

warnings.filterwarnings(
    "ignore",
    message="``output_sizes`` should be given in the ``dask_gufunc_kwargs`` parameter",
)


__all__ = [
    "TIME_DIMS",
    "LAT_DIMS",
    "LON_DIMS",
    "Z_DIMS",
    "infer_dim_name",
    "dataset_vars",
    "extract_var_from_dataset",
    "reorder_dims",
]

# recognized rectilinear dimension names for cell centers
RECT_LAT_DIMS = ["lat", "latitude"]
RECT_LON_DIMS = ["lon", "longitude"]

# recognized curvilinear cell center dim names
CURV_LAT_CENTERS = ["yh"]
CURV_LON_CENTERS = ["xh"]

# recognized curvilinear cell corner dim names
CURV_LAT_CORNERS = ["yq"]
CURV_LON_CORNERS = ["xq"]

# recognized vertical dimension names
Z_ATM_DIMS = ["level", "z"]
Z_OCN_DIMS = ["z_l", "z_i", "depth"]

# ocean grids
OCEAN_GRIDS = [("yh", "xh")]

# time dimensions
TIME_DIMS = ["time", "t", "tax", "month", "year", "day", "date"]

# union of all dimensions
LAT_DIMS = RECT_LAT_DIMS + CURV_LAT_CENTERS + CURV_LAT_CORNERS
LON_DIMS = RECT_LON_DIMS + CURV_LON_CENTERS + CURV_LON_CORNERS
Z_DIMS = Z_ATM_DIMS + Z_OCN_DIMS


def compare_datasets(ds1, ds2, varlist=None):
    """Compares two Xarray datasets

    Parameters
    ----------
    ds1: xarray.Dataset
        First dataset
    ds2: xarray.Dataset
        Second dataset
    varlist: list
        List of variables

    Returns
    -------
    tuple of xarray.Dataset objects
        (ds1, ds2, difference dataset)
    """

    if varlist is None:
        vars1 = set(dataset_vars(ds1))
        vars2 = set(dataset_vars(ds2))
        varlist = list(vars1.intersection(vars2))

    ds1 = extract_var_from_dataset(ds1, varlist=varlist)
    ds2 = extract_var_from_dataset(ds2, varlist=varlist)

    ds1_orig = ds1.copy()
    ds2_orig = ds2.copy()

    ds1_orig.load()
    ds2_orig.load()

    vars1 = set(dataset_vars(ds1))
    vars2 = set(dataset_vars(ds2))
    varlist = list(vars1.intersection(vars2))

    if equal_horiz_dims(ds1, ds2):
        diff = ds1 - ds2
        if "area" in ds1.variables:
            area = ds1["area"]
        elif "area" in ds2.variables:
            area = ds2["area"]
        else:
            area = None

    else:
        ds1.load()
        ds2.load()
        if np.multiply(*ds1[varlist[0]].shape[-2::]) > np.multiply(
            *ds2[varlist[0]].shape[-2::]
        ):
            regridder = xe.Regridder(ds1, ds2, "bilinear")
            attrs = {x: ds1[x].attrs for x in varlist}
            ds1 = regridder(ds1)
            for x in varlist:
                ds1[x] = ds1[x].assign_attrs(attrs[x])
            area = ds2["area"] if "area" in ds2.variables else None
        else:
            regridder = xe.Regridder(ds2, ds1, "bilinear")
            attrs = {x: ds2[x].attrs for x in varlist}
            ds2 = regridder(ds2)
            for x in varlist:
                ds2[x] = ds2[x].assign_attrs(attrs[x])
            area = ds1["area"] if "area" in ds1.variables else None
        # create difference array
        diff = ds1 - ds2

    # include cell area
    diff["area"] = area if area is not None else diff[varlist[0]] * 0.0

    if diff["area"].sum() > 0.0:
        # compute statistics
        for x in varlist:
            var1 = ds1[x]
            var2 = ds2[x]
            if len(var1.shape) == 2 and len(var2.shape) == 2:
                diff[x] = diff[x].assign_attrs(
                    xr_stats_2d(var1, var2, diff["area"], fmt="dict")
                )

    return {
        "ds1": ds1,
        "ds2": ds2,
        "ds1_orig": ds1_orig,
        "ds2_orig": ds2_orig,
        "diff": diff,
    }


def equal_horiz_dims(ds1, ds2):
    """Determines if two datasets have the same horizontal dimensions

    Parameters
    ----------
    ds1: xarray.Dataset
        First dataset
    ds2: xarray.Dataset
        Second dataset

    Returns
    -------
    bool
    """
    assert ("lon" in ds1.variables) and (
        "lat" in ds1.variables
    ), "Dataset 1 must contain lat & lon"
    assert ("lon" in ds2.variables) and (
        "lat" in ds2.variables
    ), "Dataset 2 must contain lat & lon"

    return (ds1.lon.shape == ds2.lon.shape) and (ds1.lat.shape == ds2.lat.shape)


def extract_var_from_dataset(ds, varlist=None):
    """Extracts a list of variables from their parent dataset

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset
    varlist: list
        List of variables

    Returns
    -------
    xarray.Dataset
    """
    result = xr.Dataset()
    varlist = list(ds.variables) if varlist is None else varlist
    if "area" in ds.variables:
        varlist = varlist + ["area"]
    varlist = sorted(list(set(varlist)))
    for var in varlist:

        # setup a local array
        _arr = ds[var]

        # remove singleton dimensions and reorder
        _arr = _arr.squeeze().reset_coords(drop=True)
        _arr = reorder_dims(_arr)

        # create a tuple of existing coordinates
        coords = _arr.coords
        coords = [(x, ds[x]) for x in coords]

        # infer the x and y dimension name
        xdim = infer_dim_name(_arr, LON_DIMS)
        ydim = infer_dim_name(_arr, LAT_DIMS)
        horiz_coord = (xdim, ydim)

        if xdim is None or ydim is None:
            continue

        if horiz_coord == ("xh", "yh"):
            result["area"] = ds["areacello"]
            coords = [
                ("lon", ds["geolon"]) if x[0] == "xh" else (x[0], x[1]) for x in coords
            ]
            coords = [
                ("lat", ds["geolat"]) if x[0] == "yh" else (x[0], x[1]) for x in coords
            ]
            rename_latlon_coords = False
        else:
            rename_latlon_coords = True

        # reassign coordinates
        coords = dict(coords)
        _arr = _arr.assign_coords(coords)

        if rename_latlon_coords:
            try:
                _arr = _arr.rename({xdim: "lon", ydim: "lat"})
            except Exception as e:
                print(e)
                print(var)

        # append to new dataset
        result[var] = _arr

    return result


def dataset_vars(ds):
    """Returns a list of variables in a dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset

    Returns
    -------
    list
    """
    varlist = set(ds.variables)
    dimlist = set(ds.dims)
    coordlist = set(ds.coords)
    # remove dims and coords
    varlist = varlist - dimlist
    varlist = varlist - coordlist
    return list(varlist)


def infer_dim_name(arr, dimlist):
    """Infers dimension name from expected list

    Parameters
    ----------
    arr : xarray.DataArray
        Input array
    dimlist : list
        List of expected/acceptable dimensions

    Returns
    -------
    str
    """
    arrdim = [x for x in arr.dims if x in dimlist]
    assert len(arrdim) <= 1, f"Multiple dimensions found: {arrdim}"
    return arrdim[0] if len(arrdim) == 1 else None


def reorder_dims(arr):
    """Reorders DataArray to expected dimension order of
    time, depth, latitude, and longitude

    Parameters
    ----------
    arr: xarray.DataArray
        Input data array

    Returns
    -------
    xarray.DataArray
    """
    tdim = infer_dim_name(arr, TIME_DIMS)
    ydim = infer_dim_name(arr, LAT_DIMS)
    xdim = infer_dim_name(arr, LON_DIMS)
    zdim = infer_dim_name(arr, Z_DIMS)

    dims = [tdim, zdim, ydim, xdim]
    dims = [x for x in dims if x is not None]
    dims = tuple(dims)

    return arr.transpose(..., *dims)
