""" standard_grid.py: module for standard lat-lon grids """

import numpy as np
import xarray as xr

__all__ = ["cell_area", "generate_standard_grid", "grid_area"]


def cell_area(corners, radius_earth=6371.0e3):
    """Function to calculate cell area on a spherical grid

    Calculates cell area given an array of 4 corners [x0,x1,y0,y1]

    Parameters
    ----------
    corners : numpy.ndarray
        Cell corners in degrees - [x0,x1,y0,y1]
    radius_earth : float, optional
        Radius of the Earth, by default 6371.0e3 m

    Returns
    -------
    numpy.float64
        Cell area in units of m2

    See Also
    --------
    grid_area : generated 2-dimensional field of cell areas
    """

    assert len(corners) == 4

    return (
        (np.pi / 180.0)
        * radius_earth
        * radius_earth
        * np.abs(np.sin(np.radians(corners[2])) - np.sin(np.radians(corners[3])))
        * np.abs(corners[0] - corners[1])
    )


def generate_standard_grid(delta_x=1.0, delta_y=1.0):
    """Function to generate a standard grid dataset

    This function generates an xarray dataset with lat/lon coordinates and
    bounds generated on a standard spherical grid. The corresponding cell
    area field is also returned. The resulting dataset is CF-compliant

    The default resolution is 1x1-degree but any values for dx and dy can
    be given as long as the longitude and latitude coordinates are evenly
    divisible by dx and dy.

    Parameters
    ----------
    delta_x : float, optional
        Longitudinal resolution in degrees, by default 1.0
    delta_y : float, optional
        Latitudinal resolution in degrees, by default 1.0

    Returns
    -------
    xarray.core.dataset.Dataset
        CF-compliant dataset on a standard spherical grid

    See Also
    --------
    grid_area : generates cell area field
    """

    assert (
        np.mod(360.0, delta_x) == 0.0
    ), "Longitude must by evenly divisible by delta_x"
    assert np.mod(180.0, delta_y) == 0.0, "Latitude must by evenly divisible by delta_y"

    # generates xarray dataset with a standard lat-lon grid
    lat = xr.DataArray(np.arange(-90.0 + (delta_y / 2), 90.0, delta_y), dims="lat")
    lat.attrs = {
        "long_name": "latitude",
        "standard_name": "latitude",
        "axis": "Y",
        "units": "degrees_N",
        "bounds": "lat_bnds",
    }

    lon = xr.DataArray(np.arange(0.0 + (delta_x / 2), 360.0, delta_x), dims="lon")
    lon.attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "axis": "X",
        "units": "degrees_E",
        "bounds": "lon_bnds",
    }

    lat_bnds = np.arange(-90.0, 90.0 + delta_y, delta_y)
    lat_bnds = np.array(list(zip(lat_bnds[0:-1], lat_bnds[1::])))
    lat_bnds = xr.DataArray(lat_bnds, dims=("lat", "bnds"))
    lat_bnds.attrs = {
        "long_name": "latitude bounds",
        "units": "degrees_N",
        "axis": "Y",
    }

    lon_bnds = np.arange(0.0, 360.0 + delta_x, delta_x)
    lon_bnds = np.array(list(zip(lon_bnds[0:-1], lon_bnds[1::])))
    lon_bnds = xr.DataArray(lon_bnds, dims=("lon", "bnds"))
    lon_bnds.attrs = {
        "long_name": "longitude bounds",
        "units": "degrees_E",
        "axis": "X",
    }

    bnds = xr.DataArray([1.0, 2.0], dims="bnds", attrs={"long_name": "vertex number"})

    area = xr.DataArray(grid_area(lat, lon), dims=("lat", "lon"))
    area.attrs = {
        "standard_name": "cell_area",
        "long_name": "Grid cell area",
        "units": "m2",
    }

    dset = xr.Dataset(
        {
            "lat": lat,
            "lon": lon,
            "lat_bnds": lat_bnds,
            "lon_bnds": lon_bnds,
            "bnds": bnds,
            "area": area,
        }
    )

    return dset


def grid_area(lat, lon):
    """Function to generate a 2-dimensional gridded array of cell areas

    Given 1-dimensional arrays for latitude and longitude  (in degrees), this
    function returns a 2-dimensional array of cell areas

    Parameters
    ----------
    lat : numpy.ndarray
        Latitude coordinate vector in degrees
    lon : numpy.ndarray
        Longitude coordinate vector in degrees

    Returns
    -------
    numpy.ndarray
        Two dimensional array of cell areas

    See Also
    --------
    cell_area : calculates individual grid cell areas
    """

    num_y = len(lat)
    num_x = len(lon)
    # calcualted a gridded field of cell areas given vectors of lat and lon

    delta_y = list(set(np.diff(lat)))
    delta_x = list(set(np.diff(lon)))

    assert len(delta_y) == 1, "Non-uniform latitude coordinate"
    assert len(delta_x) == 1, "Non-uniform longitude coordinate"

    delta_y_2 = delta_y[0] / 2.0
    delta_x_2 = delta_x[0] / 2.0

    lon = list(lon)
    lat = list(lat)

    lon_bnds = list(zip([x - delta_x_2 for x in lon], [x + delta_x_2 for x in lon]))
    lat_bnds = list(zip([x - delta_y_2 for x in lat], [x + delta_y_2 for x in lat]))

    lon_bnds = np.tile(np.array(lon_bnds)[None, :, :], (num_y, 1, 1))
    lat_bnds = np.tile(np.array(lat_bnds)[:, None, :], (1, num_x, 1))

    # stack the bounds
    bounds = np.concatenate((lon_bnds, lat_bnds), axis=-1)

    area = np.apply_along_axis(cell_area, -1, bounds)

    return area
