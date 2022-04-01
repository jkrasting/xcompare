""" Compare.py: class object for comparisons """

import warnings
import xarray as xr
from xcompare import compare_xy

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
    ):

        validate_objects(obj1, obj2)
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
            result = compare_xy.compare_datasets(
                *self.comparables,
                verbose=verbose,
                weights=weights,
                save_weights=save_weights,
                resolution=resolution,
            )

        (self.difference, self.weights) = result


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
