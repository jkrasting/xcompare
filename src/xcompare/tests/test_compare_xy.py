""" test_compare_xy.py : unit and functional tests for xy comparison functions """

import io
import requests
import numpy as np
import xarray as xr

from xcompare import standard_grid
from xcompare import compare_xy

URL = (
    "https://github.com/jkrasting/mdtf_test_data/raw/main/"
    + "mdtf_test_data/resources/ocean_static_5deg.nc"
)
response = requests.get(URL, allow_redirects=True)
data = io.BytesIO(response.content)
static = xr.open_dataset(data)

np.random.seed(123)
random = xr.DataArray(
    np.random.normal(size=static.areacello.shape),
    dims=("yh", "xh"),
    coords={"yh": static.yh, "xh": static.xh},
)
dset1 = xr.Dataset({"random": random})

random = xr.DataArray(
    np.random.normal(size=static.areacello.shape),
    dims=("yh", "xh"),
    coords={"yh": static.yh[-1::-1], "xh": static.xh},
)
dset2 = xr.Dataset({"random": random})

dset3 = standard_grid.generate_standard_grid(delta_x=5.0, delta_y=5.0)
random = xr.DataArray(
    np.random.normal(size=dset3.area.shape),
    dims=("lat", "lon"),
    coords=({"lat": dset3.lat, "lon": dset3.lon}),
)
dset3["random"] = random

dset4 = standard_grid.generate_standard_grid(delta_x=6.0, delta_y=6.0)
random = xr.DataArray(
    np.random.normal(size=dset4.area.shape),
    dims=("lat", "lon"),
    coords=({"lat": dset4.lat, "lon": dset4.lon}),
)
dset4["random"] = random


def test_compare_arrays_1():
    """test to compare two data arrays - tripolar to latlon"""
    result = compare_xy.compare_arrays(dset1.random, dset3.random)
    assert isinstance(result[0], xr.DataArray)
    assert isinstance(result[1], xr.Dataset)
    assert np.allclose(result[0].sum(), 2.86851976)


def test_compare_arrays_2():
    """test to compare two data arrays - latlon to tripolar"""
    result = compare_xy.compare_arrays(dset3.random, dset1.random)
    assert isinstance(result[0], xr.DataArray)
    assert isinstance(result[1], xr.Dataset)
    assert np.allclose(result[0].sum(), -2.86851976)


def test_determine_target():
    """test to determine target resolution"""
    result, _ = compare_xy.determine_target(dset3, dset4)
    assert result.equals(dset4)
    result, _ = compare_xy.determine_target(dset3, dset4, resolution="high")
    assert result.equals(dset3)
    result, _ = compare_xy.determine_target(dset3, dset4, resolution="common")
    assert not result.equals(dset3) and not result.equals(dset4)


def test_compare_datasets():
    """tests for lower-level dataset comparison"""
    result = compare_xy.compare_datasets(dset1, dset3)
    assert isinstance(result[0], xr.Dataset)
    assert isinstance(result[1], xr.Dataset)
    assert np.allclose(result[0].random.sum(), 2.86851976)


def test_regrid_dataset():
    """test regridding function"""
    result = compare_xy.regrid_dataset(dset3, dset4)
    assert isinstance(result[0], xr.Dataset)
    assert isinstance(result[1], xr.Dataset)
    assert result[0].coords["lon"].shape == dset4.coords["lon"].shape
    assert result[0].coords["lat"].shape == dset4.coords["lat"].shape


def test_wrap_compare_datasets():
    """test for higher level dataset comparison wrapper"""
    result = compare_xy.wrap_compare_datasets(dset1, dset3)
    assert isinstance(result[0], xr.Dataset)
    assert isinstance(result[1], xr.Dataset)
    assert np.allclose(result[0].random.sum(), 2.86851976)
