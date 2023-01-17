""" Unit tests for statistics """

import pytest
import numpy as np
import xarray as xr
from xcompare.xr_stats import corr, cov, xr_stats_2d


def test_corr():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    rng3 = np.random.default_rng(3)

    arr1 = rng1.random((20, 20))
    arr2 = rng2.random((20, 20))
    area = rng3.random((20, 20))

    result = corr(arr1, arr2, area)

    assert np.allclose(result, -0.009438481988987536)


def test_cov():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    rng3 = np.random.default_rng(3)

    arr1 = rng1.random((20, 20))
    arr2 = rng2.random((20, 20))
    area = rng3.random((20, 20))

    result = cov(arr1, arr2, area)

    assert np.allclose(result, -0.0007627592728940184)


@pytest.mark.parametrize("fmt,fmttype", [("list", list), ("dict", dict)])
def test_xr_stats_2d(fmt, fmttype):
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    rng3 = np.random.default_rng(3)

    arr1 = xr.DataArray(rng1.random((20, 20)))
    arr2 = xr.DataArray(rng2.random((20, 20)))

    area = rng3.random((20, 20))
    area[5, 5] = np.nan
    area = xr.DataArray(area)

    result = xr_stats_2d(arr1, arr2, area, fmt=fmt)

    assert len(result) == 3
    assert isinstance(result, fmttype)
