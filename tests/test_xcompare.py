import pytest
import cftime
import numpy as np
import xarray as xr
from xcompare import (
    compare_datasets,
    dataset_vars,
    infer_dim_name,
    infer_var_name,
    ordered_list_extraction,
    reorder_dims,
    equal_horiz_dims,
    extract_var_from_dataset,
)
from xcompare import LON_DIMS, LAT_DIMS, Z_DIMS, TIME_DIMS, AREA_VARS


def xr_times_from_tuples(timetuple, boundstuple, timefmt="gfdl"):
    """[summary]
    Parameters
    ----------
    timetuple : list of tuples of ints
        List of tuples containing time coordinate values [(Y,M,D,H,...) ...]
    boundstuple : list of tuples of ints
        List of tuples containing time bounds values [((Y,M,D,...)(Y,M,D,...)) ...]
    timefmt : str, optional
        Modeling center time format, either "gfdl" or "ncar", by default "ncar"
    Returns
    -------
    xarray.Dataset
        Returns an xarray dataset
    """

    dset_out = xr.Dataset()
    nbnds = np.array([0, 1])

    times = [cftime.DatetimeNoLeap(*x, calendar="noleap") for x in timetuple]
    bounds = [cftime.DatetimeNoLeap(*x, calendar="noleap") for x in boundstuple]
    bounds = list(zip(bounds[0:-1], bounds[1::]))

    if timefmt == "gfdl":
        bounds_index_name = "bnds"
        bnds_attrs = {"long_name": "time axis boundaries"}
        time_attrs = {
            "long_name": "time",
            "cartesian_axis": "T",
            "calendar_type": "noleap",
            "bounds": "time_bnds",
        }
    else:
        bounds_index_name = "nbnds"
        bnds_attrs = {"long_name": "time interval endpoints"}
        time_attrs = {"long_name": "time", "bounds": "time_bnds"}

    dims = (("time", times), (bounds_index_name, nbnds))

    dset_out["time_bnds"] = xr.DataArray(
        bounds,
        coords=dims,
        attrs=bnds_attrs,
    )

    dset_out["time"] = xr.DataArray(
        times,
        dims={"time": times},
        coords={"time": (times)},
        attrs=time_attrs,
    )

    if timefmt == "gfdl":
        dset_out["average_T1"] = (("time"), [x[0] for x in bounds])
        dset_out.average_T1.attrs = {"long_name": "Start time for average period"}

        dset_out["average_T2"] = (("time"), [x[1] for x in bounds])
        dset_out.average_T2.attrs = {"long_name": "End time for average period"}

        dset_out["average_DT"] = (("time"), [(x[1] - x[0]) for x in bounds])
        dset_out.average_DT.attrs = {"long_name": "Length of average period"}

    if timefmt == "ncar":
        dset_out["date"] = (
            ("time"),
            [int(x.strftime("%Y%m%d")) for x in dset_out.time.values],
        )
        dset_out.date.attrs = {"long_name": "current date (YYYYMMDD)"}

    if bounds_index_name in list(dset_out.variables):
        dset_out = dset_out.drop_vars(bounds_index_name)
    startyear = str(dset_out.time.values[0].strftime("%Y")).replace(" ", "0")
    dset_out.attrs["base_time_unit"] = f"days since {startyear}-01-01"

    return dset_out


def generate_annual_time_axis(startyear, nyears):
    """Construct a monthly noleap time dimension with associated bounds
    Parameters
    ----------
    startyear : int
        Start year for requested time axis
    nyears : int
        Number of years in requested time axis
    Returns
    -------
    xarray.DataArray
        time and time_bnds xarray DataArray types
    """

    nyears = nyears + 1

    years = np.arange(startyear, startyear + nyears)
    months = [7 for x in years]
    days = [15 for x in years]
    timetuple = list(zip(years, months, days))[0:-1]

    months = [1] * len(months)
    days = [1] * len(days)
    boundstuple = list(zip(years, months, days))

    return xr_times_from_tuples(timetuple, boundstuple)


def test_gen():
    ds1 = generate_annual_time_axis(1, 5)
    rng = np.random.default_rng(1)
    coords = {
        "lat": xr.DataArray(np.arange(-82.5, 90.0, 15.0), name="lat", dims="lat"),
        "xh": xr.DataArray(np.arange(-165.0, 180.0, 30.0), name="xh", dims="xh"),
        "depth": xr.DataArray([0.0, 50.0, 100.0], name="depth", dims="depth"),
        "time": ds1.time,
    }
    arr3d = xr.DataArray(
        rng.random((12, 12, 3, 5)), coords=coords, dims=[k for k, v in coords.items()]
    )
    arr2d = arr3d.mean(dim="depth").squeeze()
    ds1["varname1"] = arr3d
    ds1["varname2"] = arr2d
    ds1["area"] = arr2d.isel(time=0).squeeze()
    assert True


ds1 = generate_annual_time_axis(1, 5)
rng = np.random.default_rng(1)
coords = {
    "lat": xr.DataArray(np.arange(-82.5, 90.0, 15.0), name="lat", dims="lat"),
    "xh": xr.DataArray(np.arange(-165.0, 180.0, 30.0), name="xh", dims="xh"),
    "depth": xr.DataArray([0.0, 50.0, 100.0], name="depth", dims="depth"),
    "time": ds1.time,
}
arr3d = xr.DataArray(
    rng.random((12, 12, 3, 5)), coords=coords, dims=[k for k, v in coords.items()]
)
arr2d = arr3d.mean(dim="depth").squeeze()
ds1["varname1"] = arr3d
ds1["varname2"] = arr2d
ds1["areacella"] = arr2d.isel(time=0).squeeze()


ds2 = generate_annual_time_axis(1, 5)
rng = np.random.default_rng(2)
coords = {
    "lat": xr.DataArray(np.arange(-85.0, 90.0, 10.0), name="lat", dims="lat"),
    "xh": xr.DataArray(np.arange(0.0, 360.0, 15.0), name="xh", dims="xh"),
    "depth": xr.DataArray([0.0, 50.0, 100.0], name="depth", dims="depth"),
    "time": ds2.time,
}
arr3d = xr.DataArray(
    rng.random((18, 24, 3, 5)), coords=coords, dims=[k for k, v in coords.items()]
)
arr2d = arr3d.mean(dim="depth").squeeze()
ds2 = xr.Dataset()
ds2["varname1"] = arr3d
ds2["varname2"] = arr2d
ds2["areacello"] = arr2d.isel(time=0).squeeze()


@pytest.mark.parametrize(
    "dimlist,varname",
    [(LON_DIMS, "xh"), (LAT_DIMS, "lat"), (TIME_DIMS, "time"), (Z_DIMS, "depth")],
)
def test_infer_dim_name(dimlist, varname):
    result = infer_dim_name(arr3d, dimlist)
    assert result == varname


def test_infer_var_name():
    result = infer_var_name(ds1, AREA_VARS)
    assert result == "areacella"


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
    _ds1 = ds1.isel(depth=0)

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
    _ds1 = ds1.isel(depth=0)
    _ds2 = ds2.isel(depth=0)

    result = compare_datasets(
        _ds1, _ds2, varlist=["varname1", "varname2"], timeavg=True
    )
    result = result["diff"]

    assert np.allclose(result.varname1.attrs["bias"], -0.0878525676497386)
    assert np.allclose(result.varname1.attrs["rmse"], 0.1929865013387339)
    assert np.allclose(result.varname1.attrs["rsquared"], -0.0209345804727127)

    assert np.allclose(result.varname2.attrs["bias"], -0.0906134824978717)
    assert np.allclose(result.varname2.attrs["rmse"], 0.1433717794017723)
    assert np.allclose(result.varname2.attrs["rsquared"], -0.0432083346534093)


def test_compare_datasets_3():
    _ds1 = ds1.isel(depth=0)
    _ds2 = ds2.isel(depth=0)

    result = compare_datasets(_ds1, _ds2)
    result = result["diff"]

    answers = np.array([-0.1097941, -0.0547705, -0.0717734, -0.0506908, -0.0791244])
    assert np.allclose(np.array(result["varname1"].mean(axis=(-2, -1))), answers)


def test_dataset_vars():
    result = dataset_vars(ds1)
    assert sorted(result) == [
        "areacella",
        "average_DT",
        "average_T1",
        "average_T2",
        "time_bnds",
        "varname1",
        "varname2",
    ]


def test_ordered_list_extraction():
    list1 = ["canteloupes", "pears", "apricots"]
    list2 = ["grapes", "canteloupes", "peaches", "mellons", "apricots", "apples"]
    result = ordered_list_extraction(list1, list2)
    assert result == ["canteloupes", "apricots"]
