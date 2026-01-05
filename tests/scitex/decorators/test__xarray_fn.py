#!/usr/bin/env python3
# Timestamp: "2025-04-30 16:25:56 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/tests/scitex/decorators/test__xarray_fn.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/decorators/test__xarray_fn.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
# Required for scitex.decorators module
pytest.importorskip("tqdm")

# Optional dependencies
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
xr = pytest.importorskip("xarray")

from scitex.decorators import xarray_fn


@pytest.fixture
def test_data():
    """Create test data for tests."""
    return {
        "list": [1.0, 2.0, 3.0],
        "numpy": np.array([1.0, 2.0, 3.0]),
        "pandas_series": pd.Series([1.0, 2.0, 3.0]),
        "pandas_df": pd.DataFrame({"col1": [1.0, 2.0, 3.0]}),
        "torch": torch.tensor([1.0, 2.0, 3.0]),
        "xarray": xr.DataArray([1.0, 2.0, 3.0]),
    }


def test_xarray_fn_with_list_input(test_data):
    """Test xarray_fn with list input."""

    @xarray_fn
    def dummy_function(arr):
        # Check that input is indeed a DataArray
        assert isinstance(arr, xr.DataArray)
        return arr + 1.0

    # Input is a list, output should be list
    result = dummy_function(test_data["list"])
    assert isinstance(result, list)
    assert result == [2.0, 3.0, 4.0]


def test_xarray_fn_with_xarray_input(test_data):
    """Test xarray_fn with xarray input."""

    @xarray_fn
    def dummy_function(arr):
        assert isinstance(arr, xr.DataArray)
        return arr * 2.0

    # Input is xarray, output should be xarray
    result = dummy_function(test_data["xarray"])
    assert isinstance(result, xr.DataArray)
    xr.testing.assert_allclose(result, xr.DataArray([2.0, 4.0, 6.0]))


def test_xarray_fn_with_numpy_input(test_data):
    """Test xarray_fn with numpy input."""

    @xarray_fn
    def dummy_function(arr):
        assert isinstance(arr, xr.DataArray)
        return arr * 3.0

    # Input is numpy, output should be numpy
    result = dummy_function(test_data["numpy"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([3.0, 6.0, 9.0]))


def test_xarray_fn_nested_decorator(test_data):
    """Test nested decorator behavior with xarray_fn."""

    # Create a dummy decorator to simulate nesting
    def dummy_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set nested context
            wrapper._current_decorator = "dummy_decorator"
            return func(*args, **kwargs)

        wrapper._is_wrapper = True
        return wrapper

    # Apply both decorators (nested)
    @xarray_fn
    @dummy_decorator
    def nested_function(arr):
        # In nested mode, the type should pass through unchanged from dummy_decorator
        assert not isinstance(arr, xr.DataArray)
        return arr

    with patch("scitex.decorators._xarray_fn.is_nested_decorator", return_value=True):
        # Input list should stay as list due to nested context
        result = nested_function(test_data["torch"])
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, test_data["torch"])

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_xarray_fn.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-30 15:41:19 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/decorators/_xarray_fn.py
# # ----------------------------------------
# import os
#
# __FILE__ = "./src/scitex/decorators/_xarray_fn.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# from functools import wraps
# from typing import Any as _Any
# from typing import Callable
# 
# import numpy as np
#
# from ._converters import is_nested_decorator
# 
# 
# def xarray_fn(func: Callable) -> Callable:
#     @wraps(func)
#     def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
#         # Skip conversion if already in a nested decorator context
#         if is_nested_decorator():
#             results = func(*args, **kwargs)
#             return results
# 
#         # Set the current decorator context
#         wrapper._current_decorator = "xarray_fn"
# 
#         # Store original object for type preservation
#         original_object = args[0] if args else None
# 
#         # Convert args to xarray DataArrays
#         def to_xarray(data):
#             import xarray as xr
#             import pandas as pd
#             import torch
#
#             if isinstance(data, xr.DataArray):
#                 return data
#             elif isinstance(data, np.ndarray):
#                 return xr.DataArray(data)
#             elif isinstance(data, list):
#                 return xr.DataArray(data)
#             elif hasattr(data, "__class__") and data.__class__.__name__ == "Tensor":
#                 return xr.DataArray(data.detach().cpu().numpy())
#             elif hasattr(data, "__class__") and data.__class__.__name__ == "DataFrame":
#                 return xr.DataArray(data.values)
#             elif hasattr(data, "__class__") and data.__class__.__name__ == "Series":
#                 return xr.DataArray(data.values)
#             else:
#                 return xr.DataArray([data])
# 
#         converted_args = [to_xarray(arg) for arg in args]
#         converted_kwargs = {k: to_xarray(v) for k, v in kwargs.items()}
# 
#         # Assertion to ensure all args are converted to xarray DataArrays
#         import xarray as xr
#
#         for arg_index, arg in enumerate(converted_args):
#             assert isinstance(arg, xr.DataArray), (
#                 f"Argument {arg_index} not converted to DataArray: {type(arg)}"
#             )
#
#         results = func(*converted_args, **converted_kwargs)
# 
#         # Convert results back to original input types
#         import xarray as xr
#
#         if isinstance(results, xr.DataArray):
#             if original_object is not None:
#                 if isinstance(original_object, list):
#                     return results.values.tolist()
#                 elif isinstance(original_object, np.ndarray):
#                     return results.values
#                 elif (
#                     hasattr(original_object, "__class__")
#                     and original_object.__class__.__name__ == "Tensor"
#                 ):
#                     import torch
#
#                     return torch.tensor(results.values)
#                 elif (
#                     hasattr(original_object, "__class__")
#                     and original_object.__class__.__name__ == "DataFrame"
#                 ):
#                     import pandas as pd
#
#                     return pd.DataFrame(results.values)
#                 elif (
#                     hasattr(original_object, "__class__")
#                     and original_object.__class__.__name__ == "Series"
#                 ):
#                     import pandas as pd
#
#                     return pd.Series(results.values.flatten())
#             return results
# 
#         return results
# 
#     # Mark as a wrapper for detection
#     wrapper._is_wrapper = True
#     wrapper._decorator_type = "xarray_fn"
#     return wrapper
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/decorators/_xarray_fn.py
# --------------------------------------------------------------------------------
