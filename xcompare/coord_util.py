""" coord_util.py: coordinate utilities """

import warnings
import numpy as np
import xarray as xr

__all__ = [
    "associate_ocean_coords",
    "identical_xy_coords",
    "rename_coords_xy",
    "reset_nominal_coords",
]


def associate_ocean_coords(arr, static=None, prefix=None):
    """Function to associate ocean coordinates with a data array

    MOM-specific function to associate grid info with DataArray object.
    This function adds appropriate geo coords, cell area, and wet masks.
    It also sets integer nominal coords.

    Parameters
    ----------
    arr : xarray.core.dataarray.DataArray
        Input data array
    static : xarray.core.dataset.Dataset
        Static dataset containing grid information
    prefix : str, list, optional
        Variable prefixes to associate, by default
        ["geolon", "geolat", "area", "wet"]

    Returns
    -------
    xarray.core.dataarray.DataArray
        Data array with associated coordinates

    See Also
    --------
    reset_nominal_coords : sets integer nominal coordinates
    """

    prefix = ["geolon", "geolat", "area", "wet"] if prefix is None else prefix

    assert isinstance(
        arr, xr.DataArray
    ), "Input to this function must be an xarray DataArray"

    # get x,y axis for array
    errors = []
    try:
        xaxis = arr.cf["X"].name
    except KeyError as exception:
        warnings.warn(str(exception))
        errors.append(exception)

    try:
        yaxis = arr.cf["Y"].name
    except KeyError as exception:
        warnings.warn(str(exception))
        errors.append(exception)

    if len(errors) > 0:
        raise ValueError("Unable to diagnose standard CF Axes for input array.")

    array_axes = (yaxis, xaxis)

    coord_vars = {}
    for pref in prefix:
        _res = {
            x: tuple(static[x].dims)
            for x in static.variables
            if x.startswith(pref) and tuple(static[x].dims) == array_axes
        }
        coord_vars = {**coord_vars, **_res}

    arr = reset_nominal_coords(arr)
    static = reset_nominal_coords(static)

    for k, _ in coord_vars.items():
        arr.coords[k] = static[k]

    return arr


def identical_xy_coords(ds1, ds2, xcoord="lon", ycoord="lat"):
    """Function to test if datasets have identical xy coordinates

    Parameters
    ----------
    ds1 : xarray.core.dataset.Dataset
        First input dataset
    ds2 : xarray.core.dataset.Dataset
        Second input dataset
    xcoord : str, optional
        Name of x-coordinate, by default "lon"
    ycoord : str
        Name of y-coordinate, by default "lat"

    Returns
    -------
    bool
        True if datasets have identical horizontal coordinates
    """

    result = bool(ds1[xcoord].equals(ds2[xcoord]) & ds1[ycoord].equals(ds2[ycoord]))

    return result


def rename_coords_xy(*ds):
    """Function to rename latitude and longitude coordinates

    This function discovers CF-convention latitude and longitude
    coordinates and renames them `lat` and `lon`, respectively

    Parameters
    ----------
    ds : xarray.core.dataset.Dataset, or list
        Input datasets to rename

    Returns
    -------
    xarray.core.dataset.Dataset
    """

    # rename lat/lon variables, xesmf expects specific names

    ds = list(ds)
    ds = [
        x.rename({x.cf["longitude"].name: "lon", x.cf["latitude"].name: "lat"})
        for x in ds
    ]
    result = ds[0] if len(ds) == 1 else tuple(ds)

    return result


def reset_nominal_coords(obj, coords=None):
    """Function to reset nominal coordinates

    This function resets nominal 1-D coordinates to integer values.
    These coordinates are typically found to accompany 2-D lat/lon
    coordinates that are common in curvilinear grids.

    The function sets the nominal coordinates to a monotonically
    increasing array of integer values.

    Parameters
    ----------
    obj : xarray.core.dataset.Dataset, or xarray.core.dataarray.DataArray
        Input xarray object
    coords : str, list
        Coordinate names to reset, by default ["xh", "yh", "xq", "yq"]

    Returns
    -------
    Union[xarray.core.dataset.Dataset, xarray.core.dataarray.DataArray]
    """

    coords = ["xh", "yh", "xq", "yq"] if coords is None else coords
    coords = list(coords) if isinstance(coords, tuple) else coords
    coords = [coords] if not isinstance(coords, list) else coords

    coords = list(set(coords).intersection(set(obj.coords)))

    result = obj.assign_coords(
        {x: np.arange(1, len(obj[x].values) + 1) for x in coords}
    )

    for coord in coords:
        result[coord].attrs = obj[coord].attrs
        if "units" in result[coord].attrs.keys():
            del result[coord].attrs["units"]

    return result
