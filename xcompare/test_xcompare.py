import pytest
import numpy as np
import xarray as xr
from .xcompare import (
    compare_datasets,
    dataset_vars,
    infer_dim_name,
    reorder_dims,
    equal_horiz_dims,
    extract_var_from_dataset,
)
from .xcompare import LON_DIMS, LAT_DIMS, Z_DIMS, TIME_DIMS

rng = np.random.default_rng(1)
coords = {
    "lat": xr.DataArray(np.arange(-82.5, 90.0, 15.0), name="lat", dims="lat"),
    "xh": xr.DataArray(np.arange(-165.0, 180.0, 30.0), name="xh", dims="xh"),
    "depth": xr.DataArray([0.0, 50.0, 100.0], name="depth", dims="depth"),
    "t": xr.DataArray(np.arange(1, 6, 1), name="t", dims="t"),
}
arr3d = xr.DataArray(
    rng.random((12, 12, 3, 5)), coords=coords, dims=[k for k, v in coords.items()]
)
arr2d = arr3d.mean(dim="depth").squeeze()

ds1 = xr.Dataset()
ds1["varname1"] = arr3d
ds1["varname2"] = arr2d
ds1["area"] = arr2d.isel(t=0).squeeze()

rng = np.random.default_rng(2)
coords = {
    "lat": xr.DataArray(np.arange(-85.0, 90.0, 10.0), name="lat", dims="lat"),
    "xh": xr.DataArray(np.arange(0.0, 360.0, 15.0), name="xh", dims="xh"),
    "depth": xr.DataArray([0.0, 50.0, 100.0], name="depth", dims="depth"),
    "t": xr.DataArray(np.arange(1, 6, 1), name="t", dims="t"),
}
arr3d = xr.DataArray(
    rng.random((18, 24, 3, 5)), coords=coords, dims=[k for k, v in coords.items()]
)
arr2d = arr3d.mean(dim="depth").squeeze()

ds2 = xr.Dataset()
ds2["varname1"] = arr3d
ds2["varname2"] = arr2d
ds2["area"] = arr2d.isel(t=0).squeeze()


@pytest.mark.parametrize(
    "dimlist,varname",
    [(LON_DIMS, "xh"), (LAT_DIMS, "lat"), (TIME_DIMS, "t"), (Z_DIMS, "depth")],
)
def test_infer_dim_name(dimlist, varname):
    result = infer_dim_name(arr3d, dimlist)
    assert result == varname


@pytest.mark.parametrize(
    "arr,expected", [(arr3d, ["t", "depth", "lat", "xh"]), (arr2d, ["t", "lat", "xh"])]
)
def test_reorder_dims(arr, expected):
    orig_dims = arr.dims
    new_dims = reorder_dims(arr).dims
    assert orig_dims != new_dims


def test_extract_var_from_dataset():
    result = extract_var_from_dataset(ds1, ["varname1", "varname2"])
    assert "lat" in result.variables
    assert "lon" in result.variables
    assert "varname1" in result.variables
    assert "varname2" in result.variables


@pytest.mark.xfail
def test_equal_horiz_dims_1():
    equal_horiz_dims(ds1, ds2)


def test_equal_horiz_dims_2():
    _ds1 = extract_var_from_dataset(ds1, ["varname1", "varname2"])
    assert equal_horiz_dims(_ds1, _ds1)


@pytest.mark.xfail
def test_equal_horiz_dims_3():
    _ds1 = extract_var_from_dataset(ds1, ["varname1", "varname2"])
    _ds2 = extract_var_from_dataset(ds2, ["varname1", "varname2"])
    assert equal_horiz_dims(_ds1, _ds2)


def test_compare_datasets_1():
    _ds1 = ds1.mean(dim="depth")

    result = compare_datasets(
        _ds1, _ds1, varlist=["varname1", "varname2"], timeavg=True
    )
    result = result["diff"]

    assert result.varname1.attrs["bias"] == 0.0
    assert result.varname1.attrs["rmse"] == 0.0
    assert result.varname1.attrs["rsquared"] == 1.0

    assert result.varname2.attrs["bias"] == 0.0
    assert result.varname2.attrs["rmse"] == 0.0
    assert result.varname2.attrs["rsquared"] == 1.0


def test_compare_datasets_2():
    _ds1 = ds1.mean(dim="depth")
    _ds2 = ds2.mean(dim="depth")

    result = compare_datasets(
        _ds1, _ds2, varlist=["varname1", "varname2"], timeavg=True
    )
    result = result["diff"]

    assert np.allclose(result.varname1.attrs["bias"], 0.0031753039070744104)
    assert np.allclose(result.varname1.attrs["rmse"], 0.08174864295535195)
    assert np.allclose(result.varname1.attrs["rsquared"], -0.10642072198515153)

    assert np.allclose(result.varname2.attrs["bias"], 0.0031753039070744104)
    assert np.allclose(result.varname2.attrs["rmse"], 0.08174864295535195)
    assert np.allclose(result.varname2.attrs["rsquared"], -0.10642072198515153)


def test_compare_datasets_3():
    _ds1 = ds1.mean(dim="depth")
    _ds2 = ds2.mean(dim="depth")

    result = compare_datasets(_ds1, _ds2)
    result = result["diff"]

    answers = np.array([-0.0248607, 0.01296314, -0.00738818, 0.01785682, -0.02044864])
    assert np.allclose(np.array(result["varname1"].mean(axis=(-2, -1))), answers)


def test_dataset_vars():
    result = dataset_vars(ds1)
    assert sorted(result) == ["area", "varname1", "varname2"]
