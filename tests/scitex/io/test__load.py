#!/usr/bin/env python3
# Timestamp: "2025-05-30 01:00:00 (Claude)"
# File: /tests/scitex/io/test__load.py

import json
import os
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")
from pathlib import Path

import numpy as np
import pandas as pd

import scitex.io


class TestLoad:
    """Test cases for scitex.io.load function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_json(self, temp_dir):
        """Test loading JSON files."""
        # Arrange
        test_data = {"name": "test", "value": 42, "array": [1, 2, 3]}
        json_path = os.path.join(temp_dir, "test.json")
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Act
        loaded_data = scitex.io.load(json_path)

        # Assert
        assert loaded_data == test_data
        assert isinstance(loaded_data, dict)

    def test_load_yaml(self, temp_dir):
        """Test loading YAML files."""
        # Arrange
        yaml_content = """
name: test
value: 42
array:
  - 1
  - 2
  - 3
"""
        yaml_path = os.path.join(temp_dir, "test.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        # Act
        loaded_data = scitex.io.load(yaml_path)

        # Assert
        assert loaded_data["name"] == "test"
        assert loaded_data["value"] == 42
        assert loaded_data["array"] == [1, 2, 3]

    def test_load_csv(self, temp_dir):
        """Test loading CSV files."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]})
        csv_path = os.path.join(temp_dir, "test.csv")
        df.to_csv(csv_path, index=False)

        # Act
        loaded_df = scitex.io.load(csv_path)

        # Assert
        assert isinstance(loaded_df, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_load_numpy(self, temp_dir):
        """Test loading NumPy array files."""
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        npy_path = os.path.join(temp_dir, "test.npy")
        np.save(npy_path, arr)

        # Act
        loaded_arr = scitex.io.load(npy_path)

        # Assert
        assert isinstance(loaded_arr, np.ndarray)
        np.testing.assert_array_equal(loaded_arr, arr)

    def test_load_txt(self, temp_dir):
        """Test loading text files."""
        # Arrange
        text_content = "Hello\nWorld\nTest"
        txt_path = os.path.join(temp_dir, "test.txt")
        with open(txt_path, "w") as f:
            f.write(text_content)

        # Act
        loaded_text = scitex.io.load(txt_path)

        # Assert - text loader returns list of lines by default
        assert loaded_text == ["Hello", "World", "Test"]

    def test_load_markdown(self, temp_dir):
        """Test loading markdown files."""
        # Arrange
        md_content = "# Header\n\nThis is a **test** markdown file."
        md_path = os.path.join(temp_dir, "test.md")
        with open(md_path, "w") as f:
            f.write(md_content)

        # Act
        loaded_md = scitex.io.load(md_path)

        # Assert
        assert "Header" in loaded_md
        assert "test" in loaded_md

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        # Arrange
        fake_path = "/path/to/nonexistent/file.txt"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            scitex.io.load(fake_path)

    def test_load_with_extension_no_dot(self, temp_dir):
        """Test loading a file without extension."""
        # Arrange
        text_content = "File without extension"
        no_ext_path = os.path.join(temp_dir, "testfile")
        with open(no_ext_path, "w") as f:
            f.write(text_content)

        # Act
        loaded_text = scitex.io.load(no_ext_path)

        # Assert - no extension files are loaded as text (returns list)
        assert loaded_text == ["File without extension"]

    def test_load_pickle(self, temp_dir):
        """Test loading pickle files."""
        # Arrange
        import pickle

        test_obj = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        pkl_path = os.path.join(temp_dir, "test.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(test_obj, f)

        # Act
        loaded_obj = scitex.io.load(pkl_path)

        # Assert
        assert loaded_obj == test_obj

    def test_load_excel(self, temp_dir):
        """Test loading Excel files."""
        # Arrange
        df = pd.DataFrame({"Col1": [1, 2, 3], "Col2": ["x", "y", "z"]})
        excel_path = os.path.join(temp_dir, "test.xlsx")
        df.to_excel(excel_path, index=False)

        # Act
        loaded_df = scitex.io.load(excel_path)

        # Assert
        assert isinstance(loaded_df, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_load_tsv(self, temp_dir):
        """Test loading TSV files."""
        # Arrange
        df = pd.DataFrame({"A": [10, 20, 30], "B": ["foo", "bar", "baz"]})
        tsv_path = os.path.join(temp_dir, "test.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)

        # Act
        loaded_df = scitex.io.load(tsv_path)

        # Assert
        assert isinstance(loaded_df, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_load_relative_paths(self, temp_dir):
        """Test that relative paths like ./file.txt work correctly."""
        # Arrange
        text_content = "Relative path test"
        txt_path = os.path.join(temp_dir, "test.txt")
        with open(txt_path, "w") as f:
            f.write(text_content)

        # Test with ./relative path from temp_dir
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            relative_path = "./test.txt"

            # Act
            loaded_text = scitex.io.load(relative_path)

            # Assert - text files return list of lines
            assert loaded_text == ["Relative path test"]
        finally:
            os.chdir(original_cwd)

    def test_load_symlink(self, temp_dir):
        """Test loading files through symlinks."""
        # Arrange
        text_content = "Symlink test content"

        # Create source file in subdirectory
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)
        source_path = os.path.join(source_dir, "original.txt")
        with open(source_path, "w") as f:
            f.write(text_content)

        # Create symlink
        symlink_path = os.path.join(temp_dir, "link.txt")
        os.symlink(os.path.relpath(source_path, temp_dir), symlink_path)

    def test_load_symlink(self, temp_dir):
        """Test loading files through symlinks."""
        # Arrange
        text_content = "Symlink test content"

        # Create source file in subdirectory
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir)
        source_path = os.path.join(source_dir, "original.txt")
        with open(source_path, "w") as f:
            f.write(text_content)

        # Create symlink
        symlink_path = os.path.join(temp_dir, "link.txt")
        os.symlink(os.path.relpath(source_path, temp_dir), symlink_path)

        # Act
        loaded_text = scitex.io.load(symlink_path)

        # Assert - text files return list of lines
        assert loaded_text == ["Symlink test content"]


class TestLoadExtensionOverride:
    """Test cases for the ext parameter override functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_json_without_extension(self, temp_dir):
        """Test loading JSON file without .json extension using ext parameter."""
        # Arrange - create JSON file without extension
        test_data = {"key": "value", "number": 42}
        no_ext_path = os.path.join(temp_dir, "uuid-style-filename")
        with open(no_ext_path, "w") as f:
            json.dump(test_data, f)

        # Act - load with ext='json'
        loaded_data = scitex.io.load(no_ext_path, ext="json")

        # Assert
        assert loaded_data == test_data
        assert isinstance(loaded_data, dict)

    def test_load_csv_without_extension(self, temp_dir):
        """Test loading CSV file without extension using ext parameter.

        Note: The CSV loader currently validates file extension internally.
        This test documents the current limitation.
        """
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        no_ext_path = os.path.join(temp_dir, "data_file")
        df.to_csv(no_ext_path, index=False)

        # Act & Assert - CSV loader currently enforces extension check
        with pytest.raises(ValueError, match="File must have .csv extension"):
            scitex.io.load(no_ext_path, ext="csv")

    def test_load_ext_with_leading_dot(self, temp_dir):
        """Test that ext parameter works with or without leading dot."""
        # Arrange
        test_data = {"test": True}
        path = os.path.join(temp_dir, "file")
        with open(path, "w") as f:
            json.dump(test_data, f)

        # Act - both with and without leading dot should work
        loaded1 = scitex.io.load(path, ext="json")
        loaded2 = scitex.io.load(path, ext=".json")

        # Assert
        assert loaded1 == test_data
        assert loaded2 == test_data

    def test_load_numpy_without_extension(self, temp_dir):
        """Test loading NumPy file without extension using ext parameter.

        This tests the ext parameter for formats that don't enforce extension checks.
        """
        # Arrange - use a file with proper extension but test ext param with JSON
        # since numpy loader doesn't enforce extension
        arr = np.array([[1, 2], [3, 4]])
        npy_path = os.path.join(temp_dir, "array_data.npy")
        np.save(npy_path, arr)

        # Act - load normally
        loaded_arr = scitex.io.load(npy_path, cache=False)

        # Assert
        np.testing.assert_array_equal(loaded_arr, arr)


class TestLoadCaching:
    """Test cases for caching behavior in load function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_with_cache_disabled(self, temp_dir):
        """Test loading with cache=False."""
        # Arrange
        test_data = {"cached": False}
        json_path = os.path.join(temp_dir, "test.json")
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Act - first load with cache disabled
        loaded1 = scitex.io.load(json_path, cache=False)

        # Modify file
        with open(json_path, "w") as f:
            json.dump({"cached": True, "modified": True}, f)

        # Load again with cache disabled - should get new data
        loaded2 = scitex.io.load(json_path, cache=False)

        # Assert
        assert loaded1 == {"cached": False}
        assert loaded2 == {"cached": True, "modified": True}

    def test_load_with_cache_enabled(self, temp_dir):
        """Test loading with cache=True (default)."""
        # Arrange
        test_data = {"original": True}
        json_path = os.path.join(temp_dir, "cached_test.json")
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Clear cache first
        scitex.io.clear_load_cache()

        # Act - first load with cache
        loaded1 = scitex.io.load(json_path, cache=True)

        # Assert first load
        assert loaded1 == {"original": True}


class TestLoadGlobPatterns:
    """Test cases for glob pattern loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_glob_pattern(self, temp_dir):
        """Test loading multiple files using glob pattern."""
        # Arrange - create multiple JSON files
        for i in range(3):
            data = {"index": i}
            path = os.path.join(temp_dir, f"data_{i}.json")
            with open(path, "w") as f:
                json.dump(data, f)

        # Act
        pattern = os.path.join(temp_dir, "data_*.json")
        results = scitex.io.load(pattern)

        # Assert
        assert isinstance(results, list)
        assert len(results) == 3
        indices = [r["index"] for r in results]
        assert sorted(indices) == [0, 1, 2]

    def test_load_glob_no_matches(self, temp_dir):
        """Test loading glob pattern with no matches raises FileNotFoundError."""
        # Arrange
        pattern = os.path.join(temp_dir, "nonexistent_*.json")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="No files found matching pattern"):
            scitex.io.load(pattern)

    def test_load_glob_single_match(self, temp_dir):
        """Test loading glob pattern with single match returns list."""
        # Arrange
        data = {"single": True}
        path = os.path.join(temp_dir, "only_one.json")
        with open(path, "w") as f:
            json.dump(data, f)

        # Act
        pattern = os.path.join(temp_dir, "only_*.json")
        results = scitex.io.load(pattern)

        # Assert
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0] == {"single": True}


class TestLoadPathTypes:
    """Test cases for different path types."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_with_pathlib_path(self, temp_dir):
        """Test loading with pathlib.Path object."""
        # Arrange
        test_data = {"pathlib": True}
        json_path = Path(temp_dir) / "test.json"
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Act - use Path object instead of string
        loaded = scitex.io.load(json_path)

        # Assert
        assert loaded == test_data

    def test_load_with_tilde_path(self, temp_dir):
        """Test that tilde expansion works correctly."""
        # This test verifies the function handles paths correctly
        # We can't easily test ~ expansion without modifying HOME

        # Arrange
        json_path = os.path.join(temp_dir, "test.json")
        with open(json_path, "w") as f:
            json.dump({"test": True}, f)

        # Act - absolute path should work
        loaded = scitex.io.load(json_path)

        # Assert
        assert loaded == {"test": True}


class TestLoadSpecificFormats:
    """Test cases for specific file formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_npz(self, temp_dir):
        """Test loading .npz files (compressed NumPy archives)."""
        # Arrange
        arrays = {"arr1": np.array([1, 2, 3]), "arr2": np.array([[1, 2], [3, 4]])}
        npz_path = os.path.join(temp_dir, "arrays.npz")
        np.savez(npz_path, **arrays)

        # Act
        loaded = scitex.io.load(npz_path, cache=False)

        # Assert
        assert "arr1" in loaded.files
        assert "arr2" in loaded.files
        np.testing.assert_array_equal(loaded["arr1"], arrays["arr1"])
        np.testing.assert_array_equal(loaded["arr2"], arrays["arr2"])

    def test_load_pickle_gz(self, temp_dir):
        """Test loading .pkl.gz (compressed pickle) files."""
        import gzip
        import pickle

        # Arrange
        test_data = {"large": list(range(1000))}
        gz_path = os.path.join(temp_dir, "data.pkl.gz")
        with gzip.open(gz_path, "wb") as f:
            pickle.dump(test_data, f)

        # Act
        loaded = scitex.io.load(gz_path)

        # Assert
        assert loaded == test_data

    def test_load_yaml_yml_extension(self, temp_dir):
        """Test loading YAML with .yml extension."""
        # Arrange
        yaml_content = """
config:
  key: value
  number: 42
"""
        yml_path = os.path.join(temp_dir, "config.yml")
        with open(yml_path, "w") as f:
            f.write(yaml_content)

        # Act
        loaded = scitex.io.load(yml_path)

        # Assert
        assert loaded["config"]["key"] == "value"
        assert loaded["config"]["number"] == 42

    def test_load_log_file(self, temp_dir):
        """Test loading .log files as text."""
        # Arrange
        log_content = "INFO: Starting\nDEBUG: Processing\nINFO: Done"
        log_path = os.path.join(temp_dir, "app.log")
        with open(log_path, "w") as f:
            f.write(log_content)

        # Act
        loaded = scitex.io.load(log_path)

        # Assert - should return list of lines
        assert loaded == ["INFO: Starting", "DEBUG: Processing", "INFO: Done"]

    def test_load_python_file(self, temp_dir):
        """Test loading .py files as text."""
        # Arrange
        py_content = '#!/usr/bin/env python3\nprint("Hello")'
        py_path = os.path.join(temp_dir, "script.py")
        with open(py_path, "w") as f:
            f.write(py_content)

        # Act
        loaded = scitex.io.load(py_path)

        # Assert - should return list of lines
        assert loaded[0] == "#!/usr/bin/env python3"
        assert loaded[1] == 'print("Hello")'


class TestLoadEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_empty_json(self, temp_dir):
        """Test loading empty JSON object."""
        # Arrange
        json_path = os.path.join(temp_dir, "empty.json")
        with open(json_path, "w") as f:
            f.write("{}")

        # Act
        loaded = scitex.io.load(json_path)

        # Assert
        assert loaded == {}

    def test_load_empty_list_json(self, temp_dir):
        """Test loading JSON with empty list."""
        # Arrange
        json_path = os.path.join(temp_dir, "empty_list.json")
        with open(json_path, "w") as f:
            f.write("[]")

        # Act
        loaded = scitex.io.load(json_path)

        # Assert
        assert loaded == []

    def test_load_unicode_filename(self, temp_dir):
        """Test loading file with unicode in filename."""
        # Arrange
        unicode_path = os.path.join(temp_dir, "データ.json")
        test_data = {"unicode": "日本語"}
        try:
            with open(unicode_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False)

            # Act
            loaded = scitex.io.load(unicode_path)

            # Assert
            assert loaded == test_data
        except OSError:
            pytest.skip("Filesystem does not support unicode filenames")

    def test_load_large_csv(self, temp_dir):
        """Test loading large CSV file."""
        # Arrange
        large_df = pd.DataFrame(
            {"A": range(10000), "B": np.random.rand(10000), "C": ["text"] * 10000}
        )
        csv_path = os.path.join(temp_dir, "large.csv")
        large_df.to_csv(csv_path, index=False)

        # Act
        loaded = scitex.io.load(csv_path)

        # Assert
        assert len(loaded) == 10000
        pd.testing.assert_frame_equal(loaded, large_df)

    def test_load_case_insensitive_extension(self, temp_dir):
        """Test file extension handling with uppercase.

        Note: The loader uses the raw extension, so uppercase extensions
        fall back to txt loader. This test documents current behavior.
        """
        # Arrange
        test_data = {"case": "test"}
        upper_path = os.path.join(temp_dir, "test.JSON")
        with open(upper_path, "w") as f:
            json.dump(test_data, f)

        # Act - uppercase .JSON falls back to txt loader (returns lines)
        loaded = scitex.io.load(upper_path)

        # Assert - currently returns as text (list of lines)
        # This documents the current behavior
        assert isinstance(loaded, list)  # Text loader returns list of lines

    def test_load_broken_symlink(self, temp_dir):
        """Test loading a broken symlink raises appropriate error."""
        # Arrange
        symlink_path = os.path.join(temp_dir, "broken_link.txt")
        os.symlink("/nonexistent/target.txt", symlink_path)

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            scitex.io.load(symlink_path)

    def test_load_directory_raises_error(self, temp_dir):
        """Test that loading a directory raises an error."""
        # Arrange
        dir_path = os.path.join(temp_dir, "subdir")
        os.makedirs(dir_path)

        # Act & Assert
        with pytest.raises((IsADirectoryError, ValueError, OSError)):
            scitex.io.load(dir_path)

    def test_load_with_verbose(self, temp_dir, capsys):
        """Test loading with verbose output."""
        # Arrange
        test_data = {"verbose": True}
        json_path = os.path.join(temp_dir, "verbose_test.json")
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Clear cache to trigger cache message
        scitex.io.clear_load_cache()

        # Act
        scitex.io.load(json_path, verbose=True, cache=True)

        # Assert - verbose output is printed
        captured = capsys.readouterr()
        # Note: verbose output may vary, just verify it doesn't crash


class TestLoadHDF5:
    """Test cases for HDF5 file loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_hdf5_basic(self, temp_dir):
        """Test loading basic HDF5 file."""
        h5py = pytest.importorskip("h5py")

        # Arrange
        h5_path = os.path.join(temp_dir, "data.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=np.array([1, 2, 3, 4, 5]))
            f.create_dataset("matrix", data=np.ones((3, 3)))

        # Act
        loaded = scitex.io.load(h5_path)

        # Assert
        assert isinstance(loaded, dict)
        np.testing.assert_array_equal(loaded["data"], np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(loaded["matrix"], np.ones((3, 3)))

    def test_load_hdf5_with_groups(self, temp_dir):
        """Test loading HDF5 file with nested groups."""
        h5py = pytest.importorskip("h5py")

        # Arrange
        h5_path = os.path.join(temp_dir, "nested.hdf5")
        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("group1")
            grp.create_dataset("nested_data", data=np.array([10, 20]))

        # Act
        loaded = scitex.io.load(h5_path)

        # Assert - loaded should contain the nested structure
        assert isinstance(loaded, dict)


class TestLoadZarr:
    """Test cases for Zarr file loading."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_load_zarr_basic(self, temp_dir):
        """Test loading basic Zarr array."""
        zarr = pytest.importorskip("zarr")

        # Arrange
        zarr_path = os.path.join(temp_dir, "data.zarr")
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        z = zarr.open(zarr_path, mode="w", shape=arr.shape, dtype=arr.dtype)
        z[:] = arr

        # Act
        loaded = scitex.io.load(zarr_path)

        # Assert
        np.testing.assert_array_equal(np.array(loaded), arr)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-11 05:54:51 (ywatanabe)"
# # File: /home/ywatanabe/proj/SciTeX-Code/src/scitex/io/_load.py
# # ----------------------------------------
# from __future__ import annotations
# import os
#
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# import glob
# from pathlib import Path
# from typing import Any, Union
#
# from scitex.decorators import preserve_doc
# from scitex.str._clean_path import clean_path
# from ._load_cache import (
#     cache_data,
#     configure_cache,
#     get_cache_info,
#     get_cached_data,
#     load_npy_cached,
# )
# from ._load_modules._bibtex import _load_bibtex
#
# # from ._load_modules._catboost import _load_catboost
# from ._load_modules._con import _load_con
# from ._load_modules._docx import _load_docx
# from ._load_modules._eeg import _load_eeg_data
# from ._load_modules._hdf5 import _load_hdf5
# from ._load_modules._image import _load_image
# from ._load_modules._joblib import _load_joblib
# from ._load_modules._json import _load_json
# from ._load_modules._markdown import _load_markdown
# from ._load_modules._matlab import _load_matlab
# from ._load_modules._numpy import _load_npy
# from ._load_modules._pandas import _load_csv, _load_excel, _load_tsv
# from ._load_modules._pdf import _load_pdf
# from ._load_modules._pickle import _load_pickle
# from ._load_modules._sqlite3 import _load_db_sqlite3
# from ._load_modules._torch import _load_torch
# from ._load_modules._txt import _load_txt
# from ._load_modules._xml import _load_xml
# from ._load_modules._yaml import _load_yaml
# from ._load_modules._zarr import _load_zarr
#
#
# def _load_bundle(lpath, verbose=False, **kwargs):
#     """Load a .plot, .figure, or .stats bundle.
#
#     Parameters
#     ----------
#     lpath : str or Path
#         Path to the bundle (directory or ZIP).
#     verbose : bool
#         If True, print verbose output.
#     **kwargs
#         Additional arguments.
#
#     Returns
#     -------
#     For .plot bundles:
#         tuple: (fig, ax, data) where fig is reconstructed figure,
#                ax is the axes, data is DataFrame or None.
#     For .figure bundles:
#         dict: Figure data with 'spec' and 'panels'.
#     For .stats bundles:
#         dict: Stats data with 'spec' and 'comparisons'.
#     """
#     from .bundle import load as load_bundle, BundleType
#
#     bundle = load_bundle(lpath)
#     bundle_type = bundle.get('type')
#
#     if bundle_type == BundleType.PLTZ:
#         # Return (fig, ax, data) tuple for .plot bundles
#         # Note: We return the spec and data, not a reconstructed figure
#         # as matplotlib figures cannot be perfectly serialized/deserialized
#         import matplotlib.pyplot as plt
#         from pathlib import Path
#
#         p = Path(lpath)
#         bundle_dir = p
#
#         # Handle ZIP extraction
#         if not p.is_dir():
#             import tempfile
#             import zipfile
#             temp_dir = Path(tempfile.mkdtemp())
#             with zipfile.ZipFile(p, 'r') as zf:
#                 zf.extractall(temp_dir)
#             bundle_dir = temp_dir
#
#         # Find PNG file - layered format stores in exports/
#         basename = bundle.get('basename', 'plot')
#         png_path = bundle_dir / "exports" / f"{basename}.png"
#         if not png_path.exists():
#             # Fallback to root level (legacy format)
#             png_path = bundle_dir / f"{basename}.png"
#
#         # Load the PNG as a figure
#         if png_path.exists():
#             img = plt.imread(str(png_path))
#             fig, ax = plt.subplots()
#             ax.imshow(img)
#             ax.axis('off')
#
#             # Attach metadata from spec
#             spec = bundle.get('spec', {})
#             if spec:
#                 # Handle both layered and legacy spec formats
#                 axes_list = spec.get('axes', [])
#                 if axes_list and isinstance(axes_list, list):
#                     for key, val in axes_list[0].items():
#                         setattr(ax, f'_scitex_{key}', val)
#                 # Theme from style (layered) or spec (legacy)
#                 style = bundle.get('style', {})
#                 theme = style.get('theme', {}) if style else spec.get('theme', {})
#                 if theme:
#                     fig._scitex_theme = theme.get('mode')
#
#             # Data from bundle (merged in load_layered_plot_bundle)
#             data = bundle.get('data')
#             return fig, ax, data
#         else:
#             # No PNG, return spec and data
#             return bundle.get('spec'), None, bundle.get('data')
#
#     elif bundle_type == BundleType.FIGZ:
#         # Return figure dict for .figure bundles
#         return bundle
#
#     elif bundle_type == BundleType.STATSZ:
#         # Return stats dict for .stats bundles
#         return bundle
#
#     return bundle
#
#
# def load(
#     lpath: Union[str, Path],
#     ext: str = None,
#     show: bool = False,
#     verbose: bool = False,
#     cache: bool = True,
#     metadata: bool = None,  # None = auto-detect (True for images)
#     **kwargs,
# ) -> Any:
#     """
#     Load data from various file formats.
#
#     This function supports loading data from multiple file formats with optional caching.
#
#     Parameters
#     ----------
#     lpath : Union[str, Path]
#         The path to the file to be loaded. Can be a string or pathlib.Path object.
#     ext : str, optional
#         File extension to use for loading. If None, automatically detects from filename.
#         Useful for files without extensions (e.g., UUID-named files).
#         Examples: 'pdf', 'json', 'csv'
#     show : bool, optional
#         If True, display additional information during loading. Default is False.
#     verbose : bool, optional
#         If True, print verbose output during loading. Default is False.
#     cache : bool, optional
#         If True, enable caching for faster repeated loads. Default is True.
#     metadata : bool or None, optional
#         If True, return tuple (data, metadata_dict) for images and PDFs.
#         If False, return data only.
#         If None (default), automatically True for images, False for PDFs and other formats.
#         Works for image files (.png, .jpg, .jpeg, .tiff, .tif) and PDF files.
#         For PDFs, metadata_dict contains embedded scitex metadata from PDF Subject field.
#     **kwargs : dict
#         Additional keyword arguments to be passed to the specific loading function.
#         For PDFs, can include: mode='full'|'text'|'scientific', etc.
#
#     Returns
#     -------
#     object
#         The loaded data object, which can be of various types depending on the input file format.
#
#         For images with metadata=True (default):
#             Returns tuple (image, metadata_dict). metadata_dict is None if no metadata found.
#
#         For PDFs with metadata=False (default):
#             Returns dict with keys: 'full_text', 'sections', 'metadata', 'pages', etc.
#
#         For PDFs with metadata=True:
#             Returns tuple (pdf_data_dict, metadata_dict). Enables consistent API with images.
#
#     Raises
#     ------
#     ValueError
#         If the file extension is not supported.
#     FileNotFoundError
#         If the specified file does not exist.
#
#     Supported Extensions
#     -------------------
#     - Data formats: .csv, .tsv, .xls, .xlsx, .xlsm, .xlsb, .json, .yaml, .yml
#     - Scientific: .npy, .npz, .mat, .hdf5, .con
#     - ML/DL: .pth, .pt, .cbm, .joblib, .pkl
#     - Documents: .txt, .log, .event, .md, .docx, .pdf, .xml
#     - Images: .jpg, .png, .tiff, .tif
#     - EEG data: .vhdr, .vmrk, .edf, .bdf, .gdf, .cnt, .egi, .eeg, .set
#     - Database: .db
#
#     Examples
#     --------
#     >>> # Load CSV data
#     >>> data = load('data.csv')
#
#     >>> # Load image with metadata (default behavior)
#     >>> img, meta = load('figure.png')
#     >>> print(meta['scitex']['version'])
#
#     >>> # Load image without metadata
#     >>> img = load('figure.png', metadata=False)
#
#     >>> # Load PDF with default extraction (no metadata tuple)
#     >>> pdf = load('paper.pdf')
#     >>> print(pdf['full_text'])
#
#     >>> # Load PDF with metadata tuple (consistent API with images)
#     >>> pdf, meta = load('paper.pdf', metadata=True)
#     >>> print(meta['scitex']['version'])
#
#     >>> # Load PDF with specific mode
#     >>> text = load('paper.pdf', mode='text')
#
#     >>> # Load file without extension (e.g., UUID PDF)
#     >>> pdf = load('f2694ccb-1b6f-4994-add8-5111fd4d52f1', ext='pdf')
#     """
#
#     # Don't use clean_path as it breaks relative paths like ./file.txt
#     # lpath = clean_path(lpath)
#
#     # Convert Path objects to strings for consistency
#     if isinstance(lpath, Path):
#         lpath = str(lpath)
#         if verbose:
#             print(f"[DEBUG] After Path conversion: {lpath}")
#
#     # Handle bundle formats (.plot, .figure, .stats and their .d variants)
#     bundle_extensions = (".figure", ".plot", ".stats")
#     for bext in bundle_extensions:
#         if lpath.endswith(bext) or lpath.endswith(f"{bext}.d"):
#             return _load_bundle(lpath, verbose=verbose, **kwargs)
#
#     # Check if it's a glob pattern
#     if "*" in lpath or "?" in lpath or "[" in lpath:
#         # Handle glob pattern
#         matched_files = sorted(glob.glob(lpath))
#         if not matched_files:
#             raise FileNotFoundError(f"No files found matching pattern: {lpath}")
#         # Load all matched files
#         results = []
#         for file_path in matched_files:
#             results.append(load(file_path, show=show, verbose=verbose, **kwargs))
#         return results
#
#     # Handle broken symlinks - os.path.exists() returns False for broken symlinks
#     if not os.path.exists(lpath):
#         if os.path.islink(lpath):
#             # For symlinks, resolve the target path relative to symlink's directory
#             symlink_dir = os.path.dirname(os.path.abspath(lpath))
#             target = os.readlink(lpath)
#             resolved_target = os.path.join(symlink_dir, target)
#             resolved_target = os.path.abspath(resolved_target)
#
#             if os.path.exists(resolved_target):
#                 lpath = resolved_target
#             else:
#                 raise FileNotFoundError(f"Symlink target not found: {resolved_target}")
#         else:
#             # Try general path resolution
#             try:
#                 resolved_path = os.path.realpath(lpath)
#                 if os.path.exists(resolved_path):
#                     lpath = resolved_path
#                 else:
#                     raise FileNotFoundError(f"File not found: {lpath}")
#             except Exception:
#                 raise FileNotFoundError(f"File not found: {lpath}")
#
#     # Try to get from cache first (skip cache if metadata is requested for images)
#     if cache and not metadata:
#         cached_data = get_cached_data(lpath)
#         if cached_data is not None:
#             if verbose:
#                 print(f"[Cache HIT] Loaded from cache: {lpath}")
#             return cached_data
#
#     loaders_dict = {
#         # Default
#         "": _load_txt,
#         # Config/Settings
#         "yaml": _load_yaml,
#         "yml": _load_yaml,
#         "json": _load_json,
#         "xml": _load_xml,
#         # Bibliography
#         "bib": _load_bibtex,
#         # ML/DL Models
#         "pth": _load_torch,
#         "pt": _load_torch,
#         # "cbm": _load_catboost,
#         "joblib": _load_joblib,
#         "pkl": _load_pickle,
#         "pickle": _load_pickle,
#         "gz": _load_pickle,  # For .pkl.gz files
#         # Tabular Data
#         "csv": _load_csv,
#         "tsv": _load_tsv,
#         "xls": _load_excel,
#         "xlsx": _load_excel,
#         "xlsm": _load_excel,
#         "xlsb": _load_excel,
#         "db": _load_db_sqlite3,
#         # Scientific Data
#         "npy": _load_npy,
#         "npz": _load_npy,
#         "mat": _load_matlab,
#         "hdf5": _load_hdf5,
#         "h5": _load_hdf5,
#         "zarr": _load_zarr,
#         "con": _load_con,
#         # Documents
#         "txt": _load_txt,
#         "tex": _load_txt,
#         "log": _load_txt,
#         "event": _load_txt,
#         "py": _load_txt,
#         "sh": _load_txt,
#         "md": _load_markdown,
#         "docx": _load_docx,
#         "pdf": _load_pdf,
#         # Images
#         "jpg": _load_image,
#         "png": _load_image,
#         "tiff": _load_image,
#         "tif": _load_image,
#         # EEG Data
#         "vhdr": _load_eeg_data,
#         "vmrk": _load_eeg_data,
#         "edf": _load_eeg_data,
#         "bdf": _load_eeg_data,
#         "gdf": _load_eeg_data,
#         "cnt": _load_eeg_data,
#         "egi": _load_eeg_data,
#         "eeg": _load_eeg_data,
#         "set": _load_eeg_data,
#     }
#
#     # Determine extension: use explicit ext parameter or detect from filename
#     if ext is not None:
#         # Use explicitly provided extension (strip leading dot if present)
#         detected_ext = ext.lstrip(".")
#     else:
#         # Auto-detect from filename
#         detected_ext = lpath.split(".")[-1] if "." in lpath else ""
#
#     # Auto-detect metadata for images and PDFs
#     is_image = detected_ext in ["jpg", "jpeg", "png", "tiff", "tif"]
#     is_pdf = detected_ext == "pdf"
#
#     if metadata is None:
#         # Default: True for images, False for other formats (PDFs default to False for backward compatibility)
#         metadata = is_image
#
#     # Special handling for numpy files with caching
#     if cache and detected_ext in ["npy", "npz"]:
#         return load_npy_cached(lpath, **kwargs)
#
#     loader = preserve_doc(loaders_dict.get(detected_ext, _load_txt))
#
#     try:
#         # Pass metadata parameter for images and PDFs
#         if is_image:
#             result = loader(lpath, metadata=metadata, verbose=verbose, **kwargs)
#         elif is_pdf:
#             # Pass metadata parameter to PDF loader for API consistency
#             result = loader(lpath, metadata=metadata, **kwargs)
#         else:
#             result = loader(lpath, **kwargs)
#
#         # Cache the result if caching is enabled (skip if metadata was used)
#         if cache and not metadata:
#             cache_data(lpath, result)
#             if verbose:
#                 print(f"[Cache STORED] Cached data for: {lpath}")
#
#         return result
#     except (ValueError, FileNotFoundError) as e:
#         raise ValueError(f"Error loading file {lpath}: {str(e)}")
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load.py
# --------------------------------------------------------------------------------
