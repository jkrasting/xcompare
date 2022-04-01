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

    # MOM-specific function to associate grid info with DataArray object
    # Adds appropriate geo coords, cell area, and wet masks
    # Sets integer nominal coords
    # See also: reset_nominal_coords

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


def identical_xy_coords(ds1, ds2):

    # Tests if two datasets have the same xy coorindates
    # Requires that the datasets be renamed to lat/lon
    # See also: rename_coords_xy

    result = bool(ds1.lon.equals(ds2.lon) & ds1.lat.equals(ds2.lat))

    return result


def rename_coords_xy(*ds):

    # rename lat/lon variables, xesmf expects specific names

    ds = list(ds)
    ds = [
        x.rename({x.cf["longitude"].name: "lon", x.cf["latitude"].name: "lat"})
        for x in ds
    ]
    return tuple(ds)


def reset_nominal_coords(obj, coords=None):

    # Function to set nominal coordinates to integer values
    # This function resets the nominal coordinates to integer values
    # and copies associated metadata. This function can be used on
    # either a DataArray or a Dataset object
    #
    # obj : input xarray object
    # coords : str, or iterable of coords to reset

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
