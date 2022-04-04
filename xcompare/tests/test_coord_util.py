""" test_coord_util.py : unit tests for coordinate routines """

import io
import requests
import numpy as np
import xarray as xr

from xcompare import coord_util
from xcompare import standard_grid

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

random2 = xr.DataArray(
    np.random.normal(size=static.areacello.shape),
    dims=("yh", "xh"),
    coords={"yh": static.yh[-1::-1], "xh": static.xh},
)
dset2 = xr.Dataset({"random": random2})

dset3 = standard_grid.generate_standard_grid()


def test_identical_xy_coords_1():
    """tests function that evaluates if coords area equal in datasets"""
    assert coord_util.identical_xy_coords(dset1, dset1, xcoord="xh", ycoord="yh")
    assert not coord_util.identical_xy_coords(dset1, dset2, xcoord="xh", ycoord="yh")


def test_extract_dimset():
    """tests extraction of variables from dataset based on dimensionality"""
    result = coord_util.extract_dimset(static, ("yq", "xh"))
    answer = ["geolat_v", "geolon_v", "wet_v"]
    assert sorted(result.keys()) == answer


def test_fix_bounds_attributes():
    """tests the removal of incorrect axis attributes in bounds variables"""
    dset3.lon_bnds.attrs = {**dset3.lon_bnds.attrs, "axis": "X"}
    dset3.lat_bnds.attrs = {**dset3.lat_bnds.attrs, "axis": "Y"}
    result = coord_util.fix_bounds_attributes(dset3)
    assert "axis" not in result.lon_bnds.attrs.keys()
    assert "axis" not in result.lat_bnds.attrs.keys()


def test_identical_xy_coords_2():
    """tests function that evaluates if coords area equal in data arrays"""
    assert coord_util.identical_xy_coords(
        dset1.random, dset1.random, xcoord="xh", ycoord="yh"
    )
    assert not coord_util.identical_xy_coords(
        dset1.random, dset2.random, xcoord="xh", ycoord="yh"
    )


def test_list_dset_dimset():
    """tests discovery of dimension pairings in a dataset"""
    result = sorted(coord_util.list_dset_dimset(static))
    answer = [("yh", "xh"), ("yh", "xq"), ("yq", "xh"), ("yq", "xq")]
    assert result == answer


def test_reset_nominal_coords_1():
    """tests that nominal corodinate reset in datasets"""
    result = coord_util.reset_nominal_coords(static)
    assert result.xh.min() == 1
    assert result.xh.max() == 72
    assert result.yh.min() == 1
    assert result.yh.max() == 36
    assert result.xq.min() == 1
    assert result.xq.max() == 73
    assert result.yq.min() == 1
    assert result.yq.max() == 37


def test_reset_nominal_coords_2():
    """tests that nominal corodinate reset in data arrays"""
    result = coord_util.reset_nominal_coords(dset1.random)
    assert result.xh.min() == 1
    assert result.xh.max() == 72
    assert result.yh.min() == 1
    assert result.yh.max() == 36


def test_associate_ocean_coords_1():
    """tests that ocean static coords are associated with data array"""
    result = coord_util.associate_ocean_coords(dset1.random, static=static)
    assert result.xh.min() == 1
    assert result.xh.max() == 72
    assert result.yh.min() == 1
    assert result.yh.max() == 36
    assert len(result.coords) == 7


def test_associate_ocean_coords_2():
    """tests that ocean static coords are associated with dataset"""
    result = coord_util.associate_ocean_coords(dset1, static)
    assert result.xh.min() == 1
    assert result.xh.max() == 72
    assert result.yh.min() == 1
    assert result.yh.max() == 36
    print(sorted(result.attrs["static_fields"]))
    answer = ["area_t", "areacello", "depth_ocean", "geolat", "geolon", "wet"]
    assert sorted(result.attrs["static_fields"]) == answer


def test_rename_coords_xy_1():
    """tests that CF lat/lon coords can be identfied and renamed in a dataset"""
    result = coord_util.associate_ocean_coords(dset1.random, static=static)
    dset = xr.Dataset({"random": result})
    assert "lat" not in dset.coords
    assert "lon" not in dset.coords
    dset = coord_util.rename_coords_xy(dset)
    assert "lat" in dset.coords
    assert "lon" in dset.coords


def test_rename_coords_xy_2():
    """tests that CF lat/lon coords can be identfied and renamed in a data array"""
    arr = coord_util.associate_ocean_coords(dset1.random, static=static)
    assert "lat" not in arr.coords
    assert "lon" not in arr.coords
    arr = coord_util.rename_coords_xy(arr)
    assert "lat" in arr.coords
    assert "lon" in arr.coords


def test_valid_xy_dims():
    """tests function to determine if xy coords are CF-compliant/valid"""
    assert coord_util.valid_xy_dims(static, ("yh", "xh"))
