""" Compare.py: class object for comparisons """

import warnings
import xarray as xr
from xcompare import compare_xy
from xcompare import coord_util

__all__ = ["Compare"]


class Compare:
    def __init__(
        self,
        obj1,
        obj2,
        verbose=False,
        weights=None,
        save_weights=True,
        resolution=None,
        coord=None,
    ):

        validate_objects(obj1, obj2)

        obj1 = obj1.squeeze()
        obj2 = obj2.squeeze()

        # check coordinates before proceeding
        if coord is None:
            coord = coord_util.infer_coordinate_system(obj1)
            coords2 = coord_util.infer_coordinate_system(obj2)
            assert coord == coords2

        if coord == "yx":

            # catch and fix erroneous bounds
            obj1 = coord_util.fix_bounds_attributes(obj1)
            obj2 = coord_util.fix_bounds_attributes(obj2)

            self.comparables = (obj1, obj2)

            if isinstance(obj1, xr.DataArray):
                if verbose:
                    print("Performing Comparison on Arrays")
                result = compare_xy.compare_arrays(
                    *self.comparables,
                    verbose=verbose,
                    weights=weights,
                    save_weights=save_weights,
                    resolution=resolution,
                )

            elif isinstance(obj1, xr.Dataset):
                if verbose:
                    print("Performing Comparison on Datasets")
                result = compare_xy.wrap_compare_datasets(
                    *self.comparables,
                    verbose=verbose,
                    weights=weights,
                    resolution=resolution,
                )

            (self.difference, self.weights) = result

        else:

            raise ValueError(f"Coordinate system f{coord} is not currently supported")


def validate_objects(obj1, obj2):

    errors = []

    for obj in [obj1, obj2]:
        try:
            assert isinstance(
                obj, (xr.DataArray, xr.Dataset)
            ), f"Input object is {type(obj)};must be in Xarray format."
        except AssertionError as exception:
            warnings.warn(str(exception))
            errors.append(exception)

    try:
        assert isinstance(obj1, type(obj2)), (
            "Comparison must be made on two like objects, "
            + "either xarray.DataArray or xarray.Dataset"
        )
    except AssertionError as exception:
        warnings.warn(str(exception))
        errors.append(exception)

    if len(errors) > 0:
        raise ValueError("Errors found in input arguments")
