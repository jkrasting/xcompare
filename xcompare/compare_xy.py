""" compare_xy.py: module for x-y (lat-lon) field comparison """

import warnings
import xesmf as xe
import xarray as xr

from xcompare.coord_util import (
    rename_coords_xy,
    identical_xy_coords,
    list_dset_dimset,
    fix_bounds_attributes,
    remove_unused_bounds_attrs,
    valid_xy_dims,
    extract_dimset,
)
from xcompare.standard_grid import generate_standard_grid
from xcompare.xr_stats import xr_stats_2d


__all__ = [
    "compare_arrays",
    "determine_target",
    "compare_datasets",
    "regrid_dataset",
    "wrap_compare_datasets",
]


def compare_arrays(
    arr1, arr2, verbose=False, weights=None, save_weights=False, resolution=None
):
    """Function to compare two data array objects

    This functions returns the difference between to Xarray data arrays.

    Parameters
    ----------
    arr1 : xarray.core.dataarray.DataArray
        First input data array
    arr2 : xarray.core.dataarray.DataArray
        Second input data array
    verbose : bool, optional
        Verbose output, by default False
    weights : str, path-like, optional
        Pre-calculated weights file for xESMF, by default None
    save_weights : bool, optional
        Save xESMF weights, by default False
    resolution : str, optional
        Target resolution of "low", "high" or "common",
        by default None

    Returns
    -------
    Tuple[xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset]
        Difference array and weights dataset
    """

    # squeeze the arrays before starting
    arr1 = arr1.squeeze()
    arr2 = arr2.squeeze()

    errors = []
    try:
        assert len(arr1.dims) == 2, "First array does not have exactly two dimensions"
    except AssertionError as exception:
        warnings.warn(str(exception))
        errors.append(exception)

    try:
        assert len(arr2.dims) == 2, "Second array does not have exactly two dimensions"
    except AssertionError as exception:
        warnings.warn(str(exception))
        errors.append(exception)

    if len(errors) > 0:
        raise ValueError("Input arrays do not have the correct number of dimensions.")

    # xesmf expects datasets, so convert them here
    ds1 = xr.Dataset({"var": arr1})
    ds2 = xr.Dataset({"var": arr2})

    result = compare_datasets(
        ds1,
        ds2,
        verbose=verbose,
        weights=weights,
        save_weights=save_weights,
        resolution=resolution,
    )

    # Return array and weights dataset
    result = (result[0]["var"].rename("difference"), result[1])

    return result


def determine_target(ds1, ds2, resolution=None):
    """Function to determine target resolution

    This function compares to dataset objects and returns the
    target and source datasets depending on the requested
    resolution convention:

        "low" : regrid target is the lower resolution dataset
        "high" : regrid target is the higher resolution dataset
        "common" : regrid target is a standard 1-deg lat-lon dataset

    If `resolution=None`, ds2 is the target resolution

    Parameters
    ----------
    ds1 : xarray.core.dataset.Dataset
        First input dataset
    ds2 : xarray.core.dataset.Dataset
        Second input dataset
    resolution : str, optional
        Target resolution of "low", "high" or "common",
        by default None

    Returns
    -------
    Tuple[xarray.core.dataset.Dataset, xarray.core.dataset.Dataset]
        Target resolution dataset, source resolution dataset
    """

    # calculate the number of grid points and determine which dataset
    # is lower resolution
    num_pts_ds1 = len(ds1.lon) * len(ds1.lat)
    num_pts_ds2 = len(ds2.lon) * len(ds2.lat)

    ds1_is_lower_res = bool(num_pts_ds1 <= num_pts_ds2)

    if resolution == "low":
        source = ds2 if ds1_is_lower_res else ds1
        target = ds1 if ds1_is_lower_res else ds2

    elif resolution == "high":
        source = ds1 if ds1_is_lower_res else ds2
        target = ds2 if ds1_is_lower_res else ds1

    elif resolution == "common":
        target = generate_standard_grid()
        source = None

    elif resolution is None:
        source = ds1
        target = ds2

    else:
        raise ValueError(f"Unknown option for resolution: {resolution}")

    return (target, source)


def compare_datasets(
    ds1,
    ds2,
    verbose=False,
    weights=None,
    stats=True,
    save_weights=False,
    resolution=None,
):
    """Function to compare two dataset objects

    This functions returns the difference between to Xarray datasets.

    Parameters
    ----------
    ds1 : xarray.core.dataset.Dataset
        First input dataset
    ds2 : xarray.core.dataset.Dataset
        Second input dataset
    verbose : bool, optional
        Verbose output, by default False
    weights : str, path-like, optional
        Pre-calculated weights file for xESMF, by default None
    stats : bool, optional
        Calculate statistics. Requires area field be present
        in the target dataset. By default, True
    save_weights : bool, optional
        Save xESMF weights, by default False
    resolution : str, optional
        Target resolution of "low", "high" or "common",
        by default None


    Returns
    -------
    Tuple[xarray.core.dataset.Dataset, xarray.core.dataset.Dataset]
        Difference dataset and weights dataset
    """

    comparables = rename_coords_xy(ds1, ds2)

    # squeeze the objects
    comparables = [x.squeeze() for x in comparables]

    target, _ = determine_target(*comparables, resolution=resolution)

    # loop over the datasets and regrid if they do not match the target grid
    comparables = list(comparables)

    weights_ds = None

    for num, comp in enumerate(comparables):
        if not identical_xy_coords(comp, target):

            comparables[num], weights_ds = regrid_dataset(
                comp, target, weights=weights, verbose=verbose
            )

    # difference the two datasets
    diff = comparables[0] - comparables[1]

    print(diff)
    print(target)

    # calculate comparison statistics
    if stats:
        try:
            area = target.cf["area"]
            print(area)
        except KeyError as _:
            area = None
            warnings.warn("Unable to determine cell area. Stats will not be provided")

        if area is not None:
            for var in diff.keys():
                try:
                    _stats = xr_stats_2d(
                        comparables[0][var],
                        comparables[1][var],
                        area,
                        fmt="dict",
                    )
                    diff[var].attrs = {**diff[var].attrs, **_stats}
                except Exception as exception:
                    warnings.warn(f"Unable to calculate statistics for {var}")

    result = (diff, weights_ds)

    return result


def regrid_dataset(source, target, weights=None, verbose=False):
    """Function to regrid datasets

    This function invokes the xESMF regridder and regrids a source
    dataset to the same grid as the target dataset.

    Parameters
    ----------
    source : xarray.core.dataset.Dataset
        Source xarray dataset
    target : xarray.core.dataset.Dataset
        Target xarray dataset
    weights : str, path-like, optional
        Pre-calculated weights file for xESMF, by default None
    verbose : bool, optional
        Verbose output, by default False

    Returns
    -------
    Tuple[xarray.core.dataset.Dataset, xarray.core.dataset.Dataset]
        Regridded dataset and weights dataset
    """
    if verbose:
        print("Starting regridder")
        if weights is not None:
            print(f"Using precalculated weights file: {weights}")
        else:
            print("Generating weights - this step may a take a while.")

    try:
        regridder = xe.Regridder(
            source,
            target,
            "bilinear",
            weights=weights,
            unmapped_to_nan=True,
        )
    except Exception as exception:
        if weights is not None:
            warnings.warn(str(exception))
            exception = RuntimeError(
                "Regridding failed. "
                + "Check weights file or remove to regenerate weights."
            )
        raise exception

    wgt = regridder.weights.data
    dim = "n_s"
    weights_ds = xr.Dataset(
        {
            "S": (dim, wgt.data),
            "col": (dim, wgt.coords[1, :] + 1),
            "row": (dim, wgt.coords[0, :] + 1),
        }
    )

    # perform the regridding

    return (regridder(source), weights_ds)


def wrap_compare_datasets(ds1, ds2, weights=None, verbose=False, resolution=None):
    """Wrapper function to compare to xarray datasets.

    This function compares two xarray dataset object but performs cleaning
    operations before calling `compare_datasets`:

        - squeezes the datasets to remove unused dimensions
        - fixes non-CF compliant bounds attributes
        - detects a mixture of different horizontal grid conventions in
          a single dataset (e.g. cell centers, corners, etc.) and regrids
          all fields to a common standard 1-degree lat-lon grid

    Parameters
    ----------
    ds1 : xarray.core.dataset.Dataset
        First input xarray dataset
    ds2 : xarray.core.dataset.Dataset
        Second input xarray dataset
    weights : str, path-like, optional
        Pre-calculated weights file for xESMF, by default None
    verbose : bool, optional
        Verbose output, by default False
    resolution : str, optional
        Target resolution of "low", "high" or "common",
        by default None


    Returns
    -------
    Tuple[xarray.core.dataset.Dataset, xarray.core.dataset.Dataset]
        Difference dataset and weights dataset
    """

    ds1 = ds1.squeeze()
    ds2 = ds2.squeeze()

    ds1 = fix_bounds_attributes(ds1)
    ds2 = fix_bounds_attributes(ds2)

    if "static_fields" in ds1.attrs.keys():
        ds1_static_vars = ds1.attrs["static_fields"]
        ds1_vars = [x for x in list(ds1.keys()) if x not in ds1_static_vars]
    else:
        ds1_static_vars = []
        ds1_vars = ds1.keys()

    dimlist_1 = list_dset_dimset(ds1)
    ds1_vars = [x for x in ds1_vars if ds1[x].dims in dimlist_1]

    if "static_fields" in ds2.attrs.keys():
        ds2_static_vars = ds2.attrs["static_fields"]
        ds2_vars = [x for x in list(ds2.keys()) if x not in ds2_static_vars]
    else:
        ds2_static_vars = []
        ds2_vars = ds2.keys()

    dimlist_2 = list_dset_dimset(ds2)
    ds2_vars = [x for x in ds2_vars if ds2[x].dims in dimlist_2]

    varlist = list(set(ds1_vars).intersection(set(ds2_vars)))

    # a bit hackey in terms of letting lat-lon grids pass through
    dimlist_1 = [(None, None)] if len(dimlist_1) == 0 else dimlist_1
    dimlist_2 = [(None, None)] if len(dimlist_2) == 0 else dimlist_2

    if len(dimlist_1) == 1 and len(dimlist_2) == 1:
        diff, weights_ds = compare_datasets(
            ds1, ds2, save_weights=True, verbose=verbose, resolution=resolution
        )

    else:
        warnings.warn("Datasets have multiple coordinates")
        target_ds = generate_standard_grid()

        _ds1 = []

        for dim in dimlist_1:
            if valid_xy_dims(ds1, dim):
                result = regrid_dataset(extract_dimset(ds1, dim), target_ds)
                _ds1.append(result[0])

        _ds1 = xr.merge(_ds1)

        for var in ds1_static_vars:
            if var in list(_ds1.keys()):
                _ds1 = _ds1.drop(var)

        _ds2 = []
        for dim in dimlist_2:
            if valid_xy_dims(ds2, dim):
                result = regrid_dataset(extract_dimset(ds2, dim), target_ds)
                _ds2.append(result[0])

        _ds2 = xr.merge(_ds2)

        for var in ds2_static_vars:
            if var in list(_ds2.keys()):
                _ds2 = _ds2.drop(var)

        _ds2 = _ds2.merge(target_ds)

        diff, weights_ds = compare_datasets(_ds1, _ds2, verbose=verbose)

    for var in list(diff.keys()):
        diff[var].attrs = {**ds1[var].attrs, **diff[var].attrs}

    return diff, weights_ds
