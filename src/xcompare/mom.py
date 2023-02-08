""" mom.py - module for bringing MOM output into CF compliance """

import itertools
import numpy as np
import xarray as xr

__all__ = [
    "area_fields_from_static",
    "map_coords_to_vars",
    "refine_cf_fields",
    "xy_coords_from_static",
]


def area_fields_from_static(dset, area="areacello"):
    """Function to find cell area fields from static file

    This function extracts cell area fields whose names contain a specified string.

    The CF standard_name attribute `cell_area` is added to the fields.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset containing time-invariant (static) grid coordinates
    area : str
        Naming convention for cell area, by default 'areacello'

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset containing cell area fields
    """
    dset_area = xr.Dataset()

    for var in dset.variables:
        if area in str(dset[var].name):
            dset_area[var] = dset[var]
            dset_area[var].attrs["standard_name"] = "cell_area"

    return dset_area


def map_coords_to_vars(dset, dset_coords, reset_index_coords=True):
    """Function to assign multi-dimensional coords to variables

    This function iterates over variables in a dataset and adds coordinate variables
    contained in a separate dataset. Matching is performed based on like dimensions.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    dset_coords : xarray.core.dataset.Dataset
        Dataset containing coordinates
    reset_index_coords : bool, optional
        Reset index coordinates with integer values, by default True

    Returns
    -------
    xarray.core.dataset.Dataset
        Similar to input dataset with added coordinates
    """
    ds_mapped = xr.Dataset()
    for var in sorted(dset.variables):
        _var = dset[var]
        for coord in sorted(dset_coords.coords):
            if set(dset_coords[coord].dims).issubset(set(dset[var].dims)):
                _var = _var.assign_coords({coord: dset_coords[coord]})

        ds_mapped[var] = _var

    old_coords = list(
        set(
            list(
                itertools.chain(
                    *[list(dset_coords[x].dims) for x in dset_coords.coords]
                )
            )
        )
    )

    if reset_index_coords:
        ds_mapped = reset_dim_coord(ds_mapped, old_coords)

    return ds_mapped


def refine_cf_fields(dset, dset_static, include_area=True, reset_index_coords=True):
    """Function to refine/fix CF attributes for MOM ocean data

    This function combines an xarray dataset containing MOM ocean output
    with a dataset containing static information.

    The multi-dimensional coordinates (geolon,geolat) are combined with each
    variable and the appropriate cell area fields are included in the
    resulting dataset.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    dset_static : xarray.core.dataset.Dataset
        Dataset containing static grid information
    include_area : bool, optional
        Include cell areas in resulting dataset, by default True
    reset_index_coords : bool, optional
        Reset index coordinates (e.g. xh/yh) to integer values,
        by default True

    Returns
    -------
    xarray.core.dataset.Dataset
        Corrected CF-compliant dataset
    """

    coords = xy_coords_from_static(dset_static, reset_index_coords=reset_index_coords)
    mapped = map_coords_to_vars(dset, coords, reset_index_coords=reset_index_coords)

    if include_area:
        areas = area_fields_from_static(dset_static)
        areas = map_coords_to_vars(areas, coords, reset_index_coords=reset_index_coords)
        result = xr.merge([mapped, areas])
    else:
        result = mapped

    return result


def reset_dim_coord(dset, coords):
    """Function to reset index coordinate variables

    This function resets index coordinate variables with integer values

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    coords : List[str]
        List of coordinate names to reset

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset with adjusted index coordinate variables
    """

    coords = [coords] if not isinstance(coords, list) else coords

    for idx in sorted(coords):
        _updated_coord = xr.DataArray(np.arange(1, len(dset[idx]) + 1), dims=(idx))
        _updated_coord.attrs = {k: v for k, v in dset[idx].attrs.items() if k == "axis"}
        dset = dset.assign_coords({idx: _updated_coord})

    return dset


def xy_coords_from_static(
    dset,
    xcoord="geolon",
    ycoord="geolat",
    reset_index_coords=True,
):
    """Function to extract x-y multi-dimensional coordinates from ocean static file

    This function identifies coordinate variables whose names contain specified
    strings and returns them in a stand-alone dataset

    Exising index coordinates are reset to integer values but this behavior can be
    turned off.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Dataset containing time-invariant (static) grid coordinates
    xcoord : str, optional
        Naming convention for x-coordinate, by default 'geolon'
    ycoord : str, optional
        Naming convention for y-coordinate, by default 'geolat'
    reset_index_coords : bool, optional
        Reset index coordinates with integer values, by default True

    Returns
    -------
    xarray.core.dataset.Dataset
        Dataset containing coordinate variables
    """
    # initialize an output dataset to hold the results
    ds_coords = xr.Dataset()

    # resolve coordinate variables that contain the xcoord and
    # ycoord string names
    for var in dset.variables:
        if xcoord in str(dset[var].name):
            ds_coords[var] = dset[var]
            ds_coords[var].attrs = {
                **dset[var].attrs,
                "long_name": "longitude",
                "units": "degrees_east",
                "standard_name": "longitude",
            }

        if ycoord in str(dset[var].name):
            ds_coords[var] = dset[var]
            ds_coords[var].attrs = {
                **dset[var].attrs,
                "long_name": "latitude",
                "units": "degrees_north",
                "standard_name": "latitude",
            }

    # get a list of non-index (i.e. multi-dimensional) coordinates
    non_index_coords = {
        x: ds_coords[x] for x in set(ds_coords.variables) - set(ds_coords.coords)
    }

    result = xr.Dataset()
    result = result.assign_coords(non_index_coords)

    # promote multidimensional coordinate variables to true xarray coordinates
    # and reset the exisitng index coordinates (default behavior)
    if reset_index_coords:
        coords = list(set(ds_coords.coords) - set(non_index_coords.keys()))
        result = reset_dim_coord(result, coords)

    # leave the multidimensional coorindate variables as-is in the dataset
    else:
        # Add CF-compliant metadata if retaining the index coordinates
        for dim in ds_coords.dims:
            if "x" in dim:
                result[dim].attrs = {
                    "long_name": "x coordinate of projection",
                    "units": "degrees",
                    "axis": "X",
                    "standard_name": "projection_x_coordinate",
                }

            if "y" in dim:
                result[dim].attrs = {
                    "long_name": "y coordinate of projection",
                    "units": "degrees",
                    "axis": "Y",
                    "standard_name": "projection_y_coordinate",
                }

    return result
