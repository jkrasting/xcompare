""" test_standard_grid.py: unit tests for standard grids """

import numpy as np
from xcompare import standard_grid


def test_generate_standard_grid_1():
    """tests standard grid generation"""
    result = standard_grid.generate_standard_grid()
    assert sorted(list(result.variables)) == [
        "area",
        "bnds",
        "lat",
        "lat_bnds",
        "lon",
        "lon_bnds",
    ]
    assert len(result.lon) == 360
    assert len(result.lat) == 180


def test_generate_standard_grid_2():
    """tests grid generation with different dx/dy"""
    result = standard_grid.generate_standard_grid(delta_x=5, delta_y=10)
    assert sorted(list(result.variables)) == [
        "area",
        "bnds",
        "lat",
        "lat_bnds",
        "lon",
        "lon_bnds",
    ]
    assert len(result.lon) == 72
    assert len(result.lat) == 18


def test_cell_area():
    """test cell area calculation and symmetry across the equator"""
    result1 = standard_grid.cell_area([0.0, 1.0, 0.0, 1.0])
    result2 = standard_grid.cell_area([0.0, -1.0, 0.0, -1.0])
    assert np.allclose(result1, 12363683990.261118)
    assert result1 == result2


def test_grid_area():
    """tests grid cell area field generation"""
    ds1 = standard_grid.generate_standard_grid()
    area1 = standard_grid.grid_area(np.array(ds1.lat), np.array(ds1.lon))
    assert np.allclose(area1.sum(), 510064471909788.25)

    ds2 = standard_grid.generate_standard_grid(delta_x=5, delta_y=10)
    area2 = standard_grid.grid_area(np.array(ds2.lat), np.array(ds2.lon))
    assert np.allclose(area1.sum(), area2.sum())
