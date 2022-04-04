""" compare_xy.py: module for x-y (lat-lon) field comparison """

import warnings
import xesmf as xe
import xarray as xr

from xcompare.coord_util import (
    rename_coords_xy,
    identical_xy_coords,
    list_dset_dimset,
    fix_bounds_attributes,
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

    # calculate comparison statistics
    if stats:
        try:
            area = target.cf["area"]
        except KeyError as _:
            area = None
            warnings.warn("Unable to determine cell area. Stats will not be provided")

        if area is not None:
            for var in diff.keys():
                _stats = xr_stats_2d(
                    comparables[0][var],
                    comparables[1][var],
                    area,
                    fmt="dict",
                )
                diff[var].attrs = {**diff[var].attrs, **_stats}

    result = (diff, weights_ds)

    return result


def regrid_dataset(source, target, weights=None, verbose=False):
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

    ds1 = ds1.squeeze()

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
