""" coord_util.py: coordinate utilities """

import warnings
import numpy as np
import xarray as xr

__all__ = [
    "associate_ocean_coords",
    "associate_ocean_coords_array",
    "associate_ocean_coords_dataset",
    "dims_to_geo_coords",
    "extract_dimset",
    "fix_bounds_attributes",
    "identical_xy_coords",
    "infer_coordinate_system",
    "list_dset_dimset",
    "remove_unused_bounds_attrs",
    "rename_coords_xy",
    "reset_nominal_coords",
    "valid_xy_dims",
]


def associate_ocean_coords(obj, static, prefix=None):
    """Function to associate ocean coordinates with an xarray object

    MOM-specific function to associate grid info with an xarray object.
    This function adds appropriate geo coords, cell area, and wet masks.
    It also sets integer nominal coords.

    Parameters
    ----------
    obj : xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset
        Input xarray object
    static : xarray.core.dataset.Dataset
        Static dataset containing grid information
    prefix : str, list, optional
        Variable prefixes to associate, by default
        ["geolon", "geolat", "area", "wet"]

    Returns
    -------
    xarray.core.dataarray.DataArray, xarray.core.dataset.Dataset
        Xarray object with associated coordinates

    See Also
    --------
    reset_nominal_coords : sets integer nominal coordinates
    associate_ocean_coords_array : associates coordinates for DataArray objects
    associate_ocean_coords_dataset : associates coordinates for Dataset objects
    """

    if isinstance(obj, xr.DataArray):
        result = associate_ocean_coords_array(obj, static, prefix=prefix)
    elif isinstance(obj, xr.Dataset):
        result = associate_ocean_coords_dataset(obj, static)
    else:
        raise ValueError("Input object must be xarray DataArray or Dataset")

    return result


def associate_ocean_coords_array(arr, static, prefix=None):
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


def associate_ocean_coords_dataset(dset, static):
    """Function to associate ocean coordinates with a dataset

    MOM-specific function to associate grid info with Dataset object.
    This function adds appropriate geo coords, cell area, and wet masks.
    It also sets integer nominal coords.

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    static : xarray.core.dataset.Dataset
        Static dataset containing grid information

    Returns
    -------
    xarray.core.dataset.Dataset
        Data array with associated coordinates

    See Also
    --------
    reset_nominal_coords : sets integer nominal coordinates
    """

    dset = reset_nominal_coords(dset)
    static = reset_nominal_coords(static)

    assert isinstance(
        dset, xr.Dataset
    ), "Input to this function must be an xarray Dataset"

    dimset = list_dset_dimset(dset)
    static = extract_dimset(static, dimset)

    dset = dset.copy().squeeze()

    static_varlist = []
    for var in list(static.keys()):
        if var not in list(dset.keys()):
            dset[var] = static[var]
            static_varlist.append(var)

            if var.startswith("geolat"):
                dset[var].attrs["standard_name"] = "latitude"

            if var.startswith("geolon"):
                dset[var].attrs["standard_name"] = "longitude"

        else:
            warnings.warn(
                f"Static variable {var} already exists in dataset ... skipping"
            )

    dset.attrs = {**dset.attrs, "static_fields": static_varlist}

    dimset = list_dset_dimset(dset)
    dsout = []
    for dim in dimset:
        _dsout = xr.Dataset()
        _ds = extract_dimset(dset, dim)
        geocoords = dims_to_geo_coords(dset, dim)
        for var in set(_ds.keys()) - set(geocoords):
            _dsout[var] = _ds[var].assign_coords(
                {geocoords[0]: _ds[geocoords[0]], geocoords[1]: _ds[geocoords[1]]}
            )
        dsout.append(_dsout)

    dsout = xr.merge(dsout)
    dsout.attrs = dset.attrs

    return dsout


def dims_to_geo_coords(dset, dims):
    """Function to associate dimensions to geocentric coordinates

    This function accepts an xarray dataset and returns the geocentric
    coordinates (e.g. geolon/geolat) associated with a tuple of dimension names

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    dims : tuple(str,str)
        Dimension names, e.g. ("yh","xh")

    Returns
    -------
    xarray.core.dataset.Dataset
        Data array with associated coordinates
    """

    assert len(dims) == 2

    dim = dims[0]
    geoname = "geolon" if dim[0] == "x" else "geolat"
    candidate_0 = [
        x for x in dset.variables if (x.startswith(geoname) and dset[x].dims == dims)
    ]

    dim = dims[1]
    geoname = "geolon" if dim[0] == "x" else "geolat"
    candidate_1 = [
        x for x in dset.variables if (x.startswith(geoname) and dset[x].dims == dims)
    ]

    assert len(candidate_0) == 1
    assert len(candidate_1) == 1

    return (candidate_0[0], candidate_1[0])


def extract_dimset(dset, dimset):
    """Function to subset a dataset based on coordinates

    This function accepts a dimension paring and returns all
    variables in the source dataset that match that dimension pairing

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    dimset : Tuple[str, str]
        Dimension pair, e.g. ("yh","xh")

    Returns
    -------
    xarray.core.dataset.Dataset
        Subset of source dataset with variables that match dimension pairing
    """
    dimset = [dimset] if not isinstance(dimset, list) else dimset
    dset = dset.squeeze()
    ds_out = xr.Dataset()

    # this used to be in a for loop ...
    # for dim in dimset:

    varlist = [x for x in list(dset.keys()) if dset[x].dims in dimset]
    for var in varlist:
        ds_out[var] = dset[var]

    return ds_out


def fix_bounds_attributes(obj):
    """Function to fix incorrect bounds attributes

    Some bounds variables incorrectly include an "axis" or "cartesian_axis"
    attribute that is not CF-compliant. This function removes those
    attributes if present

    Parameters
    ----------
    obj : xarray.core.dataset.Dataset, or xarray.core.dataarray.DataArray
        Input xarray object

    Returns
    -------
    xarray.core.dataset.Dataset, or xarray.core.dataarray.DataArray
        Object with corrected bounds attributes
    """

    obj = obj.copy()
    coords = set(obj.cf.coordinates.keys())

    potential_bounds = []

    for coord in coords.intersection({"latitude", "longitude"}):

        res = obj.cf[[coord]] if isinstance(obj, xr.Dataset) else obj.cf[coord]
        variables = list(res.variables) if isinstance(obj, xr.Dataset) else [res.name]

        bounds_attr_names = ["edges", "bounds"]
        for var in variables:
            for bounds_attr in bounds_attr_names:

                if isinstance(obj, xr.Dataset):
                    if bounds_attr in res[var].attrs.keys():
                        potential_bounds.append(res[var].attrs[bounds_attr])
                else:
                    if bounds_attr in obj.coords[var].attrs.keys():
                        potential_bounds.append(obj.coords[var].attrs[bounds_attr])

    remove_attrs = ["axis", "cartesian_axis", "units"]
    for var in potential_bounds:
        for attr in remove_attrs:

            if isinstance(obj, xr.Dataset):
                if attr in obj[var].attrs.keys():
                    del obj[var].attrs[attr]
            else:
                if var in obj.coords:
                    if attr in obj.coords[var].attrs.keys():
                        del obj.coords[var].attrs[attr]

    return obj


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


def infer_coordinate_system(obj):
    """Function to determine coordinate system

    This function determines the coordinate system of reference for
    an xarray object.

    Parameters
    ----------
    obj : xarray.core.dataset.Dataset, or xarray.core.dataarray.DataArray
        Input xarray object

    Returns
    -------
    str
       Coordinates of object - "yx","zy", or "zx"
    """
    obj = obj.copy()

    if isinstance(obj, xr.DataArray):
        obj = xr.Dataset({"array": obj, **obj.coords})
        for coord in obj.coords:
            if "bounds" in obj.coords[coord].attrs.keys():
                del obj.coords[coord].attrs["bounds"]

    obj = obj.squeeze().reset_coords(drop=True)

    used_coords = []
    for var in obj.keys():
        used_coords = used_coords + list(obj[var].coords)
    used_coords = set(used_coords)
    drop_list = list(set(obj.coords) - used_coords)
    obj = obj.drop(drop_list)

    try:
        _ = obj.cf["time"]
        raise KeyError(
            "Time coordinate found. The package is only intended "
            + "for two-dimensional (xy,yz) objects."
        )
    except KeyError:
        pass

    # try:
    xcoord = obj.cf[["X"]].coords
    xcoord = str(",").join(list(xcoord))
    # except KeyError:
    #    xcoord = None

    try:
        ycoord = obj.cf[["Y"]].coords
        ycoord = str(",").join(list(ycoord))
    except KeyError:
        ycoord = None

    try:
        zcoord = obj.cf[["Z"]].coords
        zcoord = str(",").join(list(zcoord))
    except KeyError:
        zcoord = None

    poss_coords = [(ycoord, xcoord), (zcoord, ycoord), (zcoord, xcoord)]

    bad_coords = [x for x in poss_coords if x[0] is None or x[1] is None]
    coords = list((set(poss_coords) - set(bad_coords)))

    assert (
        len(coords) == 1
    ), "Dataset has incorrect number of coordinates. Must contain only (x,y) or (y,z) coords."

    if coords == [(ycoord, xcoord)]:
        result = "yx"
    elif coords == [(zcoord, ycoord)]:
        result = "zy"
    elif coords == [(zcoord, xcoord)]:
        result = "zx"
    else:
        raise RuntimeError("Catastrophic failure of coordinate inference")

    return result


def list_dset_dimset(dset):
    """Function to resolve xy coordinate pairings in a dataset

    This function returns valid xy coordinate parings in a dataset

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset

    Returns
    -------
    List[Tuple[str, str]]
        List of coordinate pairings, e.g. [("yh","xh"),("yq","xq")]
    """
    dset = dset.copy().squeeze()
    dims = [dset[x].dims for x in list(dset.keys())]
    dims = list(set(dims))
    dims = [x for x in dims if len(x) == 2]
    dims = [x for x in dims if valid_xy_dims(dset, x)]
    return dims


def remove_unused_bounds_attrs(obj):
    """Function to remove unused bounds attributes

    This function removes "bounds" attributes from xarray object coordinates
    if the coordinates are no longer associated with the object

    Parameters
    ----------
    obj : xarray.core.dataset.Dataset, or xarray.core.dataarray.DataArray
        Input xarray object

    Returns
    -------
     xarray.core.dataset.Dataset, or xarray.core.dataarray.DataArray
    """

    obj = obj.copy()

    coords = list(obj.coords)
    bounds = {
        x: obj.coords[x].attrs["bounds"]
        for x in coords
        if "bounds" in obj.coords[x].attrs.keys()
    }

    if isinstance(obj, xr.Dataset):
        for key, val in bounds.items():
            if val not in obj.keys():
                del obj.coords[key].attrs["bounds"]

    elif isinstance(obj, xr.DataArray):
        for key, val in bounds.items():
            if val not in coords:
                del obj.coords[key].attrs["bounds"]

    return obj


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

        # if coord in ["xh","xq"]:
        #    result[coord].attrs["standard_name"] = "longitude"

        # if coord in ["yh", "yq"]:
        #    result[coord].attrs["standard_name"] = "latitude"

    return result


def valid_xy_dims(dset, dim):
    """Function to determine if xy coordinates are valid

    This function tests if the specified xy coordinates are valid
    and CF-compliant

    Parameters
    ----------
    dset : xarray.core.dataset.Dataset
        Input dataset
    dim : Tuple[str, str]
        Set of dimensions, e.g. ("yh","xh")

    Returns
    -------
    bool
        True if coordinates are valid and compliant
    """

    _ds = extract_dimset(dset, dim)
    _ds = remove_unused_bounds_attrs(_ds)

    try:
        _ = _ds.cf[["X"]]
        _ = _ds.cf[["Y"]]
        valid = True

    except KeyError:
        valid = False

    return valid
