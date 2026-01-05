#!/usr/bin/env python3
# Time-stamp: "2025-06-02 14:35:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/io/_load_modules/test__pickle.py

"""Tests for pickle file loading functionality.

This module tests the _load_pickle function from scitex.io._load_modules._pickle,
which handles loading pickle files including gzip-compressed pickles.
"""

import gzip
import os
import pickle
import tempfile
from collections import OrderedDict, namedtuple
from datetime import date, datetime

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")

# Define namedtuple at module level for pickling support
Point = namedtuple("Point", ["x", "y"])


def test_load_pickle_basic():
    """Test loading a basic pickle file."""
    from scitex.io._load_modules._pickle import _load_pickle

    # Test various data types
    test_data = {
        "string": "Hello World",
        "integer": 42,
        "float": 3.14159,
        "list": [1, 2, 3, 4, 5],
        "dict": {"nested": {"key": "value"}},
        "tuple": (1, "two", 3.0),
        "set": {1, 2, 3},
        "none": None,
        "bool": True,
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        loaded_data = _load_pickle(temp_path)
        assert loaded_data == test_data
        assert isinstance(loaded_data, dict)
        assert loaded_data["string"] == "Hello World"
        assert loaded_data["list"] == [1, 2, 3, 4, 5]
    finally:
        os.unlink(temp_path)


def test_load_pickle_with_pickle_extension():
    """Test loading a file with .pickle extension."""
    from scitex.io._load_modules._pickle import _load_pickle

    test_data = ["item1", "item2", "item3"]

    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
        pickle.dump(test_data, f)
        temp_path = f.name

    try:
        loaded_data = _load_pickle(temp_path)
        assert loaded_data == test_data
    finally:
        os.unlink(temp_path)


def test_load_pickle_compressed():
    """Test loading gzip-compressed pickle files."""
    from scitex.io._load_modules._pickle import _load_pickle

    # Create a large data structure to benefit from compression
    large_data = {f"key_{i}": list(range(100)) for i in range(100)}

    with tempfile.NamedTemporaryFile(suffix=".pkl.gz", delete=False) as f:
        with gzip.open(f.name, "wb") as gz:
            pickle.dump(large_data, gz)
        temp_path = f.name

    try:
        loaded_data = _load_pickle(temp_path)
        assert loaded_data == large_data
        assert len(loaded_data) == 100
        assert loaded_data["key_50"] == list(range(100))
    finally:
        os.unlink(temp_path)


def test_load_pickle_complex_objects():
    """Test loading pickle with complex Python objects."""
    import numpy as np

    from scitex.io._load_modules._pickle import _load_pickle

    # Use module-level namedtuple for pickling support
    point1 = Point(10, 20)
    point2 = Point(30, 40)

    complex_data = {
        "namedtuples": [point1, point2],
        "ordered_dict": OrderedDict([("a", 1), ("b", 2), ("c", 3)]),
        "datetime": datetime(2024, 1, 15, 12, 30, 45),
        "date": date(2024, 6, 15),
        "nested": {
            "point": point1,
            "data": [1, 2, 3],
            "array": np.array([1.0, 2.0, 3.0]),
        },
        "frozenset": frozenset([1, 2, 3]),
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(complex_data, f)
        temp_path = f.name

    try:
        loaded_data = _load_pickle(temp_path)
        assert len(loaded_data["namedtuples"]) == 2
        assert loaded_data["namedtuples"][0].x == 10
        assert loaded_data["namedtuples"][0].y == 20
        assert loaded_data["ordered_dict"]["a"] == 1
        assert loaded_data["datetime"].year == 2024
        assert loaded_data["nested"]["point"] == point1
        np.testing.assert_array_equal(
            loaded_data["nested"]["array"], np.array([1.0, 2.0, 3.0])
        )
    finally:
        os.unlink(temp_path)


def test_load_pickle_invalid_extension():
    """Test that loading non-pickle file raises ValueError."""
    from scitex.io._load_modules._pickle import _load_pickle

    # _load_pickle validates extensions and raises ValueError
    with pytest.raises(
        ValueError, match="File must have .pkl, .pickle, or .pkl.gz extension"
    ):
        _load_pickle("test.txt")

    with pytest.raises(
        ValueError, match="File must have .pkl, .pickle, or .pkl.gz extension"
    ):
        _load_pickle("/path/to/file.json")


def test_load_pickle_corrupted_file():
    """Test handling of corrupted pickle files."""
    from scitex.io._load_modules._pickle import _load_pickle

    # Create a file with invalid pickle data
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(b"This is not valid pickle data")
        temp_path = f.name

    try:
        with pytest.raises(pickle.UnpicklingError):
            _load_pickle(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_pickle_nonexistent_file():
    """Test loading a nonexistent file."""
    from scitex.io._load_modules._pickle import _load_pickle

    with pytest.raises(FileNotFoundError):
        _load_pickle("/nonexistent/path/file.pkl")


def test_load_pickle_numpy_arrays():
    """Test loading pickle containing numpy arrays."""
    import numpy as np

    from scitex.io._load_modules._pickle import _load_pickle

    # Create data with numpy arrays
    numpy_data = {
        "array_1d": np.array([1, 2, 3, 4, 5]),
        "array_2d": np.random.rand(10, 20),
        "array_3d": np.ones((5, 5, 5)),
        "structured": np.array(
            [(1, "a"), (2, "b")], dtype=[("num", "i4"), ("char", "U1")]
        ),
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(numpy_data, f)
        temp_path = f.name

    try:
        loaded_data = _load_pickle(temp_path)
        np.testing.assert_array_equal(loaded_data["array_1d"], numpy_data["array_1d"])
        np.testing.assert_array_almost_equal(
            loaded_data["array_2d"], numpy_data["array_2d"]
        )
        assert loaded_data["array_3d"].shape == (5, 5, 5)
    finally:
        os.unlink(temp_path)


def test_load_pickle_protocol_versions():
    """Test loading pickles saved with different protocol versions."""
    from scitex.io._load_modules._pickle import _load_pickle

    test_data = {"protocol_test": True, "data": [1, 2, 3]}

    # Test different pickle protocols (0-5, where 5 is the highest in Python 3.8+)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(test_data, f, protocol=protocol)
            temp_path = f.name

        try:
            loaded_data = _load_pickle(temp_path)
            assert loaded_data == test_data
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_pickle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-14 07:41:33 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/io/_load_modules/_pickle.py
#
# import pickle
# import gzip
#
#
# def _load_pickle(lpath, **kwargs):
#     """Load pickle file (compressed or uncompressed)."""
#     if lpath.endswith(".pkl.gz"):
#         # Handle gzip compressed pickle
#         with gzip.open(lpath, "rb") as f:
#             return pickle.load(f, **kwargs)
#     elif lpath.endswith(".pkl") or lpath.endswith(".pickle"):
#         # Handle regular pickle
#         with open(lpath, "rb") as f:
#             return pickle.load(f, **kwargs)
#     else:
#         raise ValueError("File must have .pkl, .pickle, or .pkl.gz extension")
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_pickle.py
# --------------------------------------------------------------------------------
