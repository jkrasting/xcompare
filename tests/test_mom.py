""" test_mom.py -- unit and functional tests for mom module """

import pytest
import dask
import xarray as xr
from xcompare import mom


TEST_DATA_SOURCE = "https://extranet.gfdl.noaa.gov/~John.Krasting/xcompare"
# TEST_DATA_SOURCE = "~/xcompare_test_data"

ds_esm2 = xr.open_zarr(f"{TEST_DATA_SOURCE}/ESM2G_sample_zarr", use_cftime=True).load()

ds_static_esm2 = xr.open_zarr(
    f"{TEST_DATA_SOURCE}/ESM2G_static_zarr", decode_times=False
).load()

ds_esm4 = xr.open_zarr(
    f"{TEST_DATA_SOURCE}/ESM4_sample_zarr",
    use_cftime=True,
).load()

ds_static_esm4 = xr.open_zarr(
    f"{TEST_DATA_SOURCE}/ESM4_static_zarr",
    decode_times=False,
).load()

dask.config.set({"tokenize.ensure-deterministic": True})


def test_xy_coords_from_static_1():
    result = mom.xy_coords_from_static(ds_static_esm2)
    assert dask.base.tokenize(result) == "b719bc7ca7557362ab173d60cefea97b"


def test_xy_coords_from_static_2():
    result = mom.xy_coords_from_static(ds_static_esm4)
    assert dask.base.tokenize(result) == "6129788779876be0389bfb5b0c01cbb7"


def test_xy_coords_from_static_3():
    result = mom.xy_coords_from_static(ds_static_esm2, reset_index_coords=False)
    assert dask.base.tokenize(result) == "49750d7d77975b4b103ebc84581e635d"


def test_xy_coords_from_static_4():
    result = mom.xy_coords_from_static(ds_static_esm4, reset_index_coords=False)
    assert dask.base.tokenize(result) == "f0d1e9d3f6c035a5af6e3435ae40b2ca"


def test_area_fields_from_static_1():
    result = mom.area_fields_from_static(ds_static_esm2)
    assert dask.base.tokenize(result) == "a9ce4dd4c0a268b74cbac6bacd853750"


def test_area_fields_from_static_2():
    result = mom.area_fields_from_static(ds_static_esm4)
    assert dask.base.tokenize(result) == "f14fc9c917f3c6332f9e60ddde59780c"


def test_reset_dim_coord():
    result = mom.reset_dim_coord(ds_static_esm4, ["xh", "yh"])
    assert dask.base.tokenize(result) == "a811c7b274edb72705ee671a706dd06b"


def test_map_coords_to_vars_1():
    coords = mom.xy_coords_from_static(ds_static_esm4)
    result = mom.map_coords_to_vars(ds_esm4, coords)
    assert dask.base.tokenize(result) != dask.base.tokenize(ds_esm4)
    assert dask.base.tokenize(result) == "08684f89c3b4c80f4c709a7e1cd39742"


def test_map_coords_to_vars_2():
    coords = mom.xy_coords_from_static(ds_static_esm4)
    result = mom.map_coords_to_vars(ds_esm4, coords, reset_index_coords=False)
    assert dask.base.tokenize(result) != dask.base.tokenize(ds_esm4)
    assert dask.base.tokenize(result) == "3f3eafaf972019fb3998ad5d1ba25055"


def test_refine_cf_fields_1():
    result = mom.refine_cf_fields(ds_esm4, ds_static_esm4)
    assert dask.base.tokenize(result) == "4ce3e679093ec6f2e1cb063a714988c1"


def test_refine_cf_fields_2():
    result = mom.refine_cf_fields(ds_esm4, ds_static_esm4, reset_index_coords=False)
    assert dask.base.tokenize(result) == "864158f81c6150f455361c574258dd06"


def test_refine_cf_fields_3():
    result = mom.refine_cf_fields(ds_esm4, ds_static_esm4, include_area=False)
    assert dask.base.tokenize(result) == "08684f89c3b4c80f4c709a7e1cd39742"
