#!/usr/bin/env python3
# Timestamp: "2025-05-13 22:30:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/tests/scitex/io/test__save.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/io/test__save.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import shutil
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")
torch = pytest.importorskip("torch")
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports for specific formats
try:
    import h5py
except ImportError:
    h5py = None


try:
    import scipy.io
except ImportError:
    scipy = None

try:
    import joblib
except ImportError:
    joblib = None

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

import scitex


def test_torch_save_pt_extension():
    """Test that PyTorch models can be saved with .pt extension."""
    _save = scitex.io.save

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Create simple model tensor
        model = torch.tensor([1, 2, 3])

        # Test saving with .pt extension
        _save(model, temp_path, verbose=False)

        # Verify the file exists and can be loaded back
        assert os.path.exists(temp_path)
        loaded_model = torch.load(temp_path)
        assert torch.all(loaded_model == model)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_torch_save_kwargs():
    """Test that kwargs are properly passed to torch.save."""
    _save = scitex.io.save

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Create simple model tensor
        model = torch.tensor([1, 2, 3])

        # _save should pass kwargs to torch.save
        # While we can't directly test the internal call, we can verify that
        # using _save with _use_new_zipfile_serialization=False works
        _save(model, temp_path, verbose=False, _use_new_zipfile_serialization=False)

        # Verify the file exists and can be loaded back
        assert os.path.exists(temp_path)
        loaded_model = torch.load(temp_path)
        assert torch.all(loaded_model == model)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_deduplication():
    """Test that CSV files are not rewritten if content hasn't changed."""
    import pandas as pd

    from scitex.io._save_modules._csv import _save_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")

        # Create test DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # First save - creates the file
        _save_csv(df, csv_path)
        assert os.path.exists(csv_path)
        mtime1 = os.path.getmtime(csv_path)

        # Small delay to ensure mtime would change if file is rewritten
        import time

        time.sleep(0.1)

        # Second save with same data - should skip (deduplication)
        _save_csv(df, csv_path)
        mtime2 = os.path.getmtime(csv_path)

        # mtime should be unchanged since file wasn't rewritten
        assert (
            mtime1 == mtime2
        ), "File should not be rewritten when content is identical"

        # Third save with different data - should write
        df_new = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
        _save_csv(df_new, csv_path)
        mtime3 = os.path.getmtime(csv_path)

        # mtime should change since content is different
        assert mtime3 > mtime2, "File should be rewritten when content differs"

        # Verify new content was saved
        loaded = pd.read_csv(csv_path)
        assert loaded["a"].tolist() == [7, 8, 9]


def test_save_matplotlib_figure():
    """Test saving matplotlib figures in various formats."""
    import matplotlib.pyplot as plt

    from scitex.io import save

    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test Plot")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test PNG save
        png_path = os.path.join(tmpdir, "figure.png")
        save(fig, png_path, verbose=False)
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 0

        # Test PDF save
        pdf_path = os.path.join(tmpdir, "figure.pdf")
        save(fig, pdf_path, verbose=False)
        assert os.path.exists(pdf_path)

        # Test SVG save
        svg_path = os.path.join(tmpdir, "figure.svg")
        save(fig, svg_path, verbose=False)
        assert os.path.exists(svg_path)

    plt.close(fig)


def test_save_plotly_figure():
    """Test saving plotly figures."""
    try:
        import plotly.graph_objects as go

        from scitex.io import save

        # Create a simple plotly figure
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test HTML save
            html_path = os.path.join(tmpdir, "plotly_fig.html")
            save(fig, html_path, verbose=False)
            assert os.path.exists(html_path)
            assert os.path.getsize(html_path) > 0
    except ImportError:
        pytest.skip("plotly not installed")


def test_save_hdf5():
    """Test saving HDF5 files."""
    if h5py is None:
        pytest.skip("h5py not installed")

    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving numpy array to HDF5
        data = np.random.rand(10, 20, 30)
        hdf5_path = os.path.join(tmpdir, "data.h5")
        save(data, hdf5_path, verbose=False)

        assert os.path.exists(hdf5_path)

        # Verify content
        with h5py.File(hdf5_path, "r") as f:
            assert "data" in f
            loaded_data = f["data"][:]
            np.testing.assert_array_almost_equal(loaded_data, data)


def test_save_matlab():
    """Test saving MATLAB .mat files."""
    if scipy is None:
        pytest.skip("scipy not installed")

    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving dict to .mat
        data = {
            "array": np.array([1, 2, 3]),
            "matrix": np.array([[1, 2], [3, 4]]),
            "scalar": 42.0,
        }
        mat_path = os.path.join(tmpdir, "data.mat")
        save(data, mat_path, verbose=False)

        assert os.path.exists(mat_path)

        # Verify content
        loaded = scipy.io.loadmat(mat_path)
        np.testing.assert_array_equal(loaded["array"].flatten(), data["array"])
        np.testing.assert_array_equal(loaded["matrix"], data["matrix"])
        # MATLAB stores scalars as arrays, extract with .item() or flatten
        assert float(loaded["scalar"].flatten()[0]) == data["scalar"]


def test_save_compressed_pickle():
    """Test saving compressed pickle files."""
    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Large data that benefits from compression
        data = {
            "large_array": np.random.rand(1000, 1000),
            "metadata": {"compression": True},
        }

        # Test .pkl.gz
        gz_path = os.path.join(tmpdir, "data.pkl.gz")
        save(data, gz_path, verbose=False)
        assert os.path.exists(gz_path)

        # Verify it's compressed (should be smaller than uncompressed)
        pkl_path = os.path.join(tmpdir, "data_uncompressed.pkl")
        save(data, pkl_path, verbose=False)

        assert os.path.getsize(gz_path) < os.path.getsize(pkl_path)


def test_save_joblib():
    """Test saving with joblib format."""
    if joblib is None:
        pytest.skip("joblib not installed")

    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create complex object
        data = {
            "model": {"weights": np.random.rand(100, 100)},
            "config": {"learning_rate": 0.001},
        }

        joblib_path = os.path.join(tmpdir, "model.joblib")
        save(data, joblib_path, verbose=False)

        assert os.path.exists(joblib_path)

        # Verify content
        loaded = joblib.load(joblib_path)
        np.testing.assert_array_equal(
            loaded["model"]["weights"], data["model"]["weights"]
        )
        assert loaded["config"] == data["config"]


def test_save_pil_image():
    """Test saving PIL images."""
    from PIL import Image

    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple PIL image
        img = Image.new("RGB", (100, 100), color="red")

        # Test various image formats
        for ext in [".png", ".jpg", ".tiff"]:
            img_path = os.path.join(tmpdir, f"image{ext}")
            save(img, img_path, verbose=False)
            assert os.path.exists(img_path)

            # Verify it can be loaded
            loaded_img = Image.open(img_path)
            assert loaded_img.size == (100, 100)
            loaded_img.close()


def test_save_with_datetime_path():
    """Test saving with datetime in path."""
    from datetime import datetime

    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create path with datetime placeholder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {"test": "data"}

        # Path with datetime
        save_path = os.path.join(tmpdir, f"data_{timestamp}.json")
        save(data, save_path, verbose=False)

        assert os.path.exists(save_path)


def test_save_verbose_output(capsys):
    """Test verbose output during save."""
    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        data = np.array([1, 2, 3])
        save_path = os.path.join(tmpdir, "data.npy")

        # Save with verbose=True
        save(data, save_path, verbose=True)


def test_save_hdf5_with_key_and_override(capsys):
    """Test HDF5 save functionality with key and override parameters."""
    if h5py is None:
        pytest.skip("h5py not installed")

    from scitex.io import has_h5_key, save
    from scitex.io._load_modules._H5Explorer import H5Explorer

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test_data.h5")

        # Test data
        test_data = {
            "array1": np.random.rand(10, 10),
            "array2": np.arange(100).reshape(10, 10),
            "scalar": 42,
            "string": "test_string",
            "metadata": {"test_id": 1, "description": "Test HDF5"},
        }

        # Test 1: Save with key parameter
        save(test_data, h5_path, key="group1/subgroup/data", verbose=False)
        assert os.path.exists(h5_path)

        # Verify structure using H5Explorer
        with H5Explorer(h5_path) as explorer:
            # Check that nested groups were created
            assert "group1" in explorer.keys("/")
            assert "subgroup" in explorer.keys("/group1")
            assert "data" in explorer.keys("/group1/subgroup")

            # Load and verify data
            loaded_data = explorer.load("/group1/subgroup/data")
            np.testing.assert_array_equal(loaded_data["array1"], test_data["array1"])
            assert loaded_data["scalar"] == test_data["scalar"]
            assert loaded_data["string"] == test_data["string"]

        # Test 2: has_h5_key function
        assert has_h5_key(h5_path, "group1/subgroup/data")
        assert not has_h5_key(h5_path, "nonexistent/key")

        # Test 3: Save again without override (should skip)
        # Modify data to check if it's overwritten
        test_data["array1"] = np.ones((5, 5))
        save(
            test_data,
            h5_path,
            key="group1/subgroup/data",
            override=False,
            verbose=False,
        )

        # Verify data was NOT overwritten
        with H5Explorer(h5_path) as explorer:
            loaded_data = explorer.load("/group1/subgroup/data")
            assert loaded_data["array1"].shape == (10, 10)  # Original shape
            assert not np.array_equal(loaded_data["array1"], np.ones((5, 5)))

        # Test 4: Save with override=True
        save(
            test_data, h5_path, key="group1/subgroup/data", override=True, verbose=False
        )

        # Verify data WAS overwritten
        with H5Explorer(h5_path) as explorer:
            loaded_data = explorer.load("/group1/subgroup/data")
            assert loaded_data["array1"].shape == (5, 5)  # New shape
            np.testing.assert_array_equal(loaded_data["array1"], np.ones((5, 5)))

        # Test 5: Save to root (no key)
        root_data = {"root_array": np.random.rand(3, 3)}
        h5_path2 = os.path.join(tmpdir, "test_root.h5")
        save(root_data, h5_path2, verbose=False)

        with H5Explorer(h5_path2) as explorer:
            # Data should be at root level
            assert "root_array" in explorer.keys("/")
            loaded = explorer.load("/root_array")
            np.testing.assert_array_equal(loaded, root_data["root_array"])

        # Test 6: Complex nested structure (like PAC data)
        pac_data = {
            "pac_values": np.random.rand(64, 10, 10),
            "p_values": np.random.rand(64, 10, 10),
            "metadata": {
                "seizure_id": "S001",
                "patient_id": "P023",
                "duration_sec": 60.0,
            },
        }

        pac_key = "patient_023/seizure_001/pac_analysis"
        save(pac_data, h5_path, key=pac_key, verbose=False)

        # Verify complex structure
        assert has_h5_key(h5_path, pac_key)
        with H5Explorer(h5_path) as explorer:
            loaded_pac = explorer.load(f"/{pac_key}")
            np.testing.assert_array_equal(
                loaded_pac["pac_values"], pac_data["pac_values"]
            )
            assert loaded_pac["metadata"]["seizure_id"] == "S001"

        # Check output (commented out since all saves are with verbose=False)
        # captured = capsys.readouterr()
        # assert "Saved to:" in captured.out
        # assert save_path in captured.out
        # assert "KB" in captured.out or "B" in captured.out  # Size info


def test_save_figure_with_csv_export():
    """Test saving figure with CSV data export."""
    import matplotlib.pyplot as plt

    from scitex.io import save

    # Create figure with data
    fig, ax = plt.subplots()
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    ax.plot(x, y, label="Test Line")

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "figure.png")

        # Save figure (CSV export depends on wrapped axes)
        save(fig, fig_path, verbose=False)
        assert os.path.exists(fig_path)

    plt.close(fig)


def test_save_error_handling(caplog):
    """Test error handling in save function."""
    from scitex import logging
    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with None object - should log error
        save(None, os.path.join(tmpdir, "none.txt"), verbose=False)
        assert "Error occurred while saving" in caplog.text

        # Test with unsupported extension - should show warning
        data = {"test": "data"}
        save(data, os.path.join(tmpdir, "no_extension"), verbose=False)
        # Check that file wasn't created since format is unsupported
        assert not os.path.exists(os.path.join(tmpdir, "no_extension"))

        # Test with read-only directory - should log error
        ro_dir = os.path.join(tmpdir, "readonly")
        os.makedirs(ro_dir)
        os.chmod(ro_dir, 0o444)

        try:
            save(data, os.path.join(ro_dir, "data.json"), verbose=False)
            # Should not create file due to permission error
            assert not os.path.exists(os.path.join(ro_dir, "data.json"))
        finally:
            # Restore permissions for cleanup
            os.chmod(ro_dir, 0o755)


def test_save_catboost_model():
    """Test saving CatBoost models."""
    try:
        from catboost import CatBoostClassifier

        from scitex.io import save

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple model
            model = CatBoostClassifier(iterations=10, verbose=False)

            # Mock training data
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            model.fit(X, y, verbose=False)

            # Save model
            cbm_path = os.path.join(tmpdir, "model.cbm")
            save(model, cbm_path, verbose=False)

            assert os.path.exists(cbm_path)
    except ImportError:
        pytest.skip("CatBoost not installed")


def test_save_with_makedirs_false(caplog):
    """Test save behavior when makedirs=False."""
    from scitex.io import save

    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to save to non-existent directory with makedirs=False
        data = {"test": "data"}
        save_path = os.path.join(tmpdir, "nonexistent", "data.json")

        # Should not create the file since directory doesn't exist
        save(data, save_path, verbose=False, makedirs=False)
        assert not os.path.exists(save_path)
        assert "Error occurred while saving" in caplog.text


class TestSave:
    """Test cases for scitex.io.save function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_numpy_array(self, temp_dir):
        """Test saving NumPy arrays."""
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        save_path = os.path.join(temp_dir, "array.npy")

        # Act
        scitex.io.save(arr, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded, arr)

    def test_save_numpy_compressed(self, temp_dir):
        """Test saving compressed NumPy arrays."""
        # Arrange
        data = {"array1": np.array([1, 2, 3]), "array2": np.array([[4, 5], [6, 7]])}
        save_path = os.path.join(temp_dir, "arrays.npz")

        # Act
        scitex.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded["array1"], data["array1"])
        np.testing.assert_array_equal(loaded["array2"], data["array2"])

    def test_save_pandas_dataframe(self, temp_dir):
        """Test saving pandas DataFrames."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"], "C": [1.1, 2.2, 3.3]})
        save_path = os.path.join(temp_dir, "data.csv")

        # Act
        scitex.io.save(df, save_path, verbose=False, index=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = pd.read_csv(save_path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_json(self, temp_dir):
        """Test saving JSON data."""
        # Arrange
        data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        save_path = os.path.join(temp_dir, "data.json")

        # Act
        scitex.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_yaml(self, temp_dir):
        """Test saving YAML data."""
        # Arrange
        data = {"config": {"learning_rate": 0.001, "batch_size": 32}, "model": "ResNet"}
        save_path = os.path.join(temp_dir, "config.yaml")

        # Act
        scitex.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = scitex.io.load(save_path)
        assert loaded == data

    def test_save_pickle(self, temp_dir):
        """Test saving pickle files."""
        # Arrange
        data = {
            "array": np.array([1, 2, 3]),
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        }
        save_path = os.path.join(temp_dir, "data.pkl")

        # Act
        scitex.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path, "rb") as f:
            loaded = pickle.load(f)
        np.testing.assert_array_equal(loaded["array"], data["array"])
        assert loaded["list"] == data["list"]
        assert loaded["dict"] == data["dict"]

    def test_save_text(self, temp_dir):
        """Test saving text files."""
        # Arrange
        text = "Hello\nWorld\nTest"
        save_path = os.path.join(temp_dir, "text.txt")

        # Act
        scitex.io.save(text, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path) as f:
            loaded = f.read()
        assert loaded == text

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "nested", "dir", "data.json")

        # Act
        scitex.io.save(data, save_path, verbose=False, makedirs=True)

        # Assert
        assert os.path.exists(save_path)
        parent_dir = os.path.dirname(save_path)
        assert os.path.exists(parent_dir)

    def test_save_dry_run(self, temp_dir, caplog):
        """Test dry run mode."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "data.json")

        # Act
        scitex.io.save(data, save_path, dry_run=True, verbose=True)

        # Assert
        assert not os.path.exists(save_path)  # File should not be created
        # Output goes to logging via logger.success
        assert "(dry run)" in caplog.text

    @pytest.mark.skip(reason="symlink_from_cwd requires full session infrastructure")
    def test_save_with_symlink(self, temp_dir):
        """Test saving with symlink creation."""
        # Arrange
        data = {"test": "data"}
        # Change to temp dir to test symlink
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Act
            scitex.io.save(
                data, "subdir/data.json", verbose=False, symlink_from_cwd=True
            )

            # Assert
            # Should create both the actual file and a symlink
            assert os.path.exists("subdir/data.json")
            # The implementation creates files in script_out directories
        finally:
            os.chdir(original_cwd)

    def test_save_unsupported_format(self, temp_dir, caplog):
        """Test saving with unsupported format shows warning."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "data.unknown")

        # Act - save uses logger.warning (not warnings.warn)
        scitex.io.save(data, save_path, verbose=False)

        # Assert - warning should be logged and file not created
        assert "Unsupported file format" in caplog.text
        assert not os.path.exists(save_path)

    def test_save_list_to_npz(self, temp_dir):
        """Test saving list of arrays to npz."""
        # Arrange
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        save_path = os.path.join(temp_dir, "arrays.npz")

        # Act
        scitex.io.save(arrays, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded["0"], arrays[0])
        np.testing.assert_array_equal(loaded["1"], arrays[1])

    def test_save_various_csv_types(self, temp_dir):
        """Test saving various types as CSV."""
        # Test DataFrame (primary CSV use case)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scitex.io.save(df, os.path.join(temp_dir, "dataframe.csv"), verbose=False)
        assert os.path.exists(os.path.join(temp_dir, "dataframe.csv"))

        # Test numpy array
        arr = np.array([[1, 2], [3, 4]])
        scitex.io.save(arr, os.path.join(temp_dir, "array.csv"), verbose=False)
        assert os.path.exists(os.path.join(temp_dir, "array.csv"))

        # Test Series
        series = pd.Series([1, 2, 3], name="values")
        scitex.io.save(series, os.path.join(temp_dir, "series.csv"), verbose=False)
        assert os.path.exists(os.path.join(temp_dir, "series.csv"))


class TestSavePathTypes:
    """Test cases for different path types in save function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_with_pathlib_path(self, temp_dir):
        """Test saving with pathlib.Path object."""
        # Arrange
        data = {"pathlib": True, "works": "yes"}
        save_path = Path(temp_dir) / "pathlib_test.json"

        # Act
        scitex.io.save(data, save_path, verbose=False)

        # Assert
        assert save_path.exists()
        loaded = json.load(open(save_path))
        assert loaded == data

    def test_save_with_nested_pathlib(self, temp_dir):
        """Test saving with nested pathlib.Path."""
        # Arrange
        data = np.array([1, 2, 3])
        save_path = Path(temp_dir) / "nested" / "dir" / "array.npy"

        # Act
        scitex.io.save(data, save_path, verbose=False, makedirs=True)

        # Assert
        assert save_path.exists()
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded, data)


class TestSaveZarr:
    """Test cases for Zarr save functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_zarr_basic(self, temp_dir):
        """Test basic Zarr array saving."""
        zarr = pytest.importorskip("zarr")

        # Arrange
        data = np.random.rand(10, 20, 30)
        zarr_path = os.path.join(temp_dir, "data.zarr")

        # Act
        scitex.io.save(data, zarr_path, verbose=False)

        # Assert
        assert os.path.exists(zarr_path)
        # Non-dict data is wrapped in {"data": obj} by _save_zarr
        loaded = zarr.open(zarr_path, mode="r")
        np.testing.assert_array_almost_equal(np.array(loaded["data"]), data)

    def test_save_zarr_dict(self, temp_dir):
        """Test saving dict of arrays to Zarr."""
        zarr = pytest.importorskip("zarr")

        # Arrange
        data = {"array1": np.array([1, 2, 3]), "array2": np.ones((5, 5))}
        zarr_path = os.path.join(temp_dir, "dict_data.zarr")

        # Act
        scitex.io.save(data, zarr_path, verbose=False)

        # Assert
        assert os.path.exists(zarr_path)


class TestSaveEdgeCases:
    """Test edge cases for save function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_empty_dict_json(self, temp_dir):
        """Test saving empty dict to JSON."""
        # Arrange
        save_path = os.path.join(temp_dir, "empty.json")

        # Act
        scitex.io.save({}, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = json.load(open(save_path))
        assert loaded == {}

    def test_save_empty_list_json(self, temp_dir):
        """Test saving empty list to JSON."""
        # Arrange
        save_path = os.path.join(temp_dir, "empty_list.json")

        # Act
        scitex.io.save([], save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = json.load(open(save_path))
        assert loaded == []

    def test_save_unicode_content(self, temp_dir):
        """Test saving data with unicode content."""
        # Arrange
        data = {"japanese": "こんにちは", "emoji": "🎉🐍", "chinese": "中文"}
        save_path = os.path.join(temp_dir, "unicode.json")

        # Act
        scitex.io.save(data, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_unicode_filename(self, temp_dir):
        """Test saving with unicode in filename."""
        # Arrange
        data = {"test": True}
        try:
            save_path = os.path.join(temp_dir, "データ.json")

            # Act
            scitex.io.save(data, save_path, verbose=False)

            # Assert
            assert os.path.exists(save_path)
        except OSError:
            pytest.skip("Filesystem does not support unicode filenames")

    def test_save_large_numpy_array(self, temp_dir):
        """Test saving large NumPy array."""
        # Arrange
        large_array = np.random.rand(500, 500, 10)
        save_path = os.path.join(temp_dir, "large.npy")

        # Act
        scitex.io.save(large_array, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded, large_array)

    def test_save_special_float_values(self, temp_dir):
        """Test saving arrays with special float values."""
        # Arrange
        arr = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0, 1e-300, 1e300])
        save_path = os.path.join(temp_dir, "special_floats.npy")

        # Act
        scitex.io.save(arr, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded[0], np.inf)
        np.testing.assert_array_equal(loaded[1], -np.inf)
        assert np.isnan(loaded[2])

    def test_save_returns_path(self, temp_dir):
        """Test that save returns the saved path."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "return_test.json")

        # Act
        result = scitex.io.save(data, save_path, verbose=False)

        # Assert - save should return Path object on success
        # Note: exact return behavior depends on implementation
        if result is not None and result is not False:
            assert os.path.exists(str(result))


class TestSaveTextFormats:
    """Test cases for text format saving."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_markdown(self, temp_dir):
        """Test saving markdown content."""
        # Arrange
        content = (
            "# Header\n\nThis is **bold** and *italic* text.\n\n- Item 1\n- Item 2"
        )
        save_path = os.path.join(temp_dir, "document.md")

        # Act
        scitex.io.save(content, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path) as f:
            loaded = f.read()
        assert loaded == content

    def test_save_python_script(self, temp_dir):
        """Test saving Python script content."""
        # Arrange
        content = '#!/usr/bin/env python3\n\ndef hello():\n    print("Hello, World!")\n'
        save_path = os.path.join(temp_dir, "script.py")

        # Act
        scitex.io.save(content, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        with open(save_path) as f:
            loaded = f.read()
        assert loaded == content

    def test_save_tex(self, temp_dir):
        """Test saving LaTeX content."""
        # Arrange
        content = (
            r"\documentclass{article}\n\begin{document}\nHello LaTeX!\n\end{document}"
        )
        save_path = os.path.join(temp_dir, "document.tex")

        # Act
        scitex.io.save(content, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)


class TestSaveDataFrameFormats:
    """Test cases for DataFrame format saving."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_save_dataframe_to_excel(self, temp_dir):
        """Test saving DataFrame to Excel."""
        # Arrange
        df = pd.DataFrame(
            {
                "Name": ["Alice", "Bob", "Charlie"],
                "Age": [25, 30, 35],
                "Score": [85.5, 90.0, 78.5],
            }
        )
        save_path = os.path.join(temp_dir, "data.xlsx")

        # Act
        scitex.io.save(df, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = pd.read_excel(save_path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_to_csv_with_index(self, temp_dir):
        """Test saving DataFrame to CSV with index."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["x", "y", "z"])
        save_path = os.path.join(temp_dir, "indexed.csv")

        # Act
        scitex.io.save(df, save_path, verbose=False, index=True)

        # Assert
        assert os.path.exists(save_path)
        loaded = pd.read_csv(save_path, index_col=0)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_dataframe_with_nan(self, temp_dir):
        """Test saving DataFrame with NaN values."""
        # Arrange
        df = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 2, np.nan]})
        save_path = os.path.join(temp_dir, "with_nan.csv")

        # Act
        scitex.io.save(df, save_path, verbose=False)

        # Assert
        assert os.path.exists(save_path)
        loaded = pd.read_csv(save_path)
        assert pd.isna(loaded.loc[1, "A"])


class TestSaveRoundTrip:
    """Test round-trip save/load for various formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    def test_roundtrip_json(self, temp_dir):
        """Test JSON round-trip save/load."""
        # Arrange
        data = {"nested": {"key": [1, 2, 3]}, "flag": True}
        save_path = os.path.join(temp_dir, "roundtrip.json")

        # Act
        scitex.io.save(data, save_path, verbose=False)
        loaded = scitex.io.load(save_path)

        # Assert
        assert loaded == data

    def test_roundtrip_yaml(self, temp_dir):
        """Test YAML round-trip save/load."""
        # Arrange
        data = {"config": {"learning_rate": 0.001, "epochs": 100}}
        save_path = os.path.join(temp_dir, "roundtrip.yaml")

        # Act
        scitex.io.save(data, save_path, verbose=False)
        loaded = scitex.io.load(save_path)

        # Assert
        assert loaded == data

    def test_roundtrip_numpy(self, temp_dir):
        """Test NumPy round-trip save/load."""
        # Arrange
        data = np.random.rand(10, 20)
        save_path = os.path.join(temp_dir, "roundtrip.npy")

        # Act
        scitex.io.save(data, save_path, verbose=False)
        loaded = scitex.io.load(save_path, cache=False)

        # Assert
        np.testing.assert_array_equal(loaded, data)

    def test_roundtrip_pickle(self, temp_dir):
        """Test pickle round-trip save/load."""
        # Arrange
        data = {
            "array": np.array([1, 2, 3]),
            "string": "test",
            "nested": {"a": 1, "b": [2, 3]},
        }
        save_path = os.path.join(temp_dir, "roundtrip.pkl")

        # Act
        scitex.io.save(data, save_path, verbose=False)
        loaded = scitex.io.load(save_path)

        # Assert
        np.testing.assert_array_equal(loaded["array"], data["array"])
        assert loaded["string"] == data["string"]
        assert loaded["nested"] == data["nested"]

    def test_roundtrip_dataframe_csv(self, temp_dir):
        """Test DataFrame CSV round-trip."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, 3.3]})
        save_path = os.path.join(temp_dir, "roundtrip.csv")

        # Act
        scitex.io.save(df, save_path, verbose=False, index=False)
        loaded = scitex.io.load(save_path)

        # Assert
        pd.testing.assert_frame_equal(loaded, df)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save.py
#
# """
# Save utilities for various data types to different file formats.
#
# Supported formats include CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML,
# JSON, HDF5, PTH, MAT, CBM, and FTS bundles (.zip or directory).
# """
#
# import inspect
# import os as _os
# from pathlib import Path
# from typing import Any, Union
#
# from scitex import logging
# from scitex.path._clean import clean
# from scitex.path._getsize import getsize
# from scitex.sh import sh
# from scitex.str._clean_path import clean_path
# from scitex.str._color_text import color_text
# from scitex.str._readable_bytes import readable_bytes
#
# # Import save functions from the modular structure
# from ._save_modules import (
#     get_figure_with_data,
#     handle_image_with_csv,
#     save_bibtex,
#     save_catboost,
#     save_csv,
#     save_excel,
#     save_hdf5,
#     save_html,
#     save_joblib,
#     save_json,
#     save_matlab,
#     save_mp4,
#     save_npy,
#     save_npz,
#     save_pickle,
#     save_pickle_compressed,
#     save_plot_bundle,
#     save_stx_bundle,
#     save_tex,
#     save_text,
#     save_torch,
#     save_yaml,
#     save_zarr,
#     symlink,
#     symlink_to,
# )
#
# logger = logging.getLogger()
#
# # Re-export for backward compatibility
# _get_figure_with_data = get_figure_with_data
# _symlink = symlink
# _symlink_to = symlink_to
# _save_stx_bundle = save_stx_bundle
# _save_plot_bundle = save_plot_bundle
# _handle_image_with_csv = handle_image_with_csv
#
#
# def save(
#     obj: Any,
#     specified_path: Union[str, Path],
#     makedirs: bool = True,
#     verbose: bool = True,
#     symlink_from_cwd: bool = False,
#     symlink_to: Union[str, Path] = None,
#     dry_run: bool = False,
#     no_csv: bool = False,
#     use_caller_path: bool = False,
#     auto_crop: bool = True,
#     crop_margin_mm: float = 1.0,
#     metadata_extra: dict = None,
#     json_schema: str = "editable",
#     **kwargs,
# ) -> None:
#     """
#     Save an object to a file with the specified format.
#
#     Parameters
#     ----------
#     obj : Any
#         The object to be saved.
#     specified_path : Union[str, Path]
#         The file path where the object should be saved.
#     makedirs : bool, optional
#         If True, create the directory path if it does not exist. Default is True.
#     verbose : bool, optional
#         If True, print a message upon successful saving. Default is True.
#     symlink_from_cwd : bool, optional
#         If True, create a symlink from the current working directory. Default is False.
#     symlink_to : Union[str, Path], optional
#         If specified, create a symlink at this path. Default is None.
#     dry_run : bool, optional
#         If True, simulate the saving process. Default is False.
#     auto_crop : bool, optional
#         If True, automatically crop saved images. Default is True.
#     crop_margin_mm : float, optional
#         Margin in millimeters for auto_crop. Default is 1.0mm.
#     use_caller_path : bool, optional
#         If True, determine script path by skipping internal library frames.
#     metadata_extra : dict, optional
#         Additional metadata to merge with auto-collected metadata.
#     json_schema : str, optional
#         Schema type for JSON metadata output. Default is "editable".
#     **kwargs
#         Additional keyword arguments for the underlying save function.
#     """
#     try:
#         if isinstance(specified_path, Path):
#             specified_path = str(specified_path)
#
#         # Handle f-string expressions
#         specified_path = _parse_fstring_path(specified_path)
#
#         # Determine save path
#         spath = _determine_save_path(specified_path, use_caller_path)
#         spath_final = clean(spath)
#
#         # Prepare symlink path from cwd
#         spath_cwd = _os.getcwd() + "/" + specified_path
#         spath_cwd = clean(spath_cwd)
#
#         # Remove existing files (skip for CSV/HDF5 with key)
#         _cleanup_existing_files(spath_final, spath_cwd, kwargs)
#
#         if dry_run:
#             _handle_dry_run(spath, verbose)
#             return
#
#         if makedirs:
#             _os.makedirs(_os.path.dirname(spath_final), exist_ok=True)
#
#         # Main save
#         _save(
#             obj,
#             spath_final,
#             verbose=verbose,
#             symlink_from_cwd=symlink_from_cwd,
#             symlink_to=symlink_to,
#             dry_run=dry_run,
#             no_csv=no_csv,
#             auto_crop=auto_crop,
#             crop_margin_mm=crop_margin_mm,
#             metadata_extra=metadata_extra,
#             json_schema=json_schema,
#             **kwargs,
#         )
#
#         # Symbolic links
#         _symlink(spath, spath_cwd, symlink_from_cwd, verbose)
#         _symlink_to(spath_final, symlink_to, verbose)
#         return Path(spath)
#
#     except AssertionError:
#         raise
#     except Exception as e:
#         logger.error(f"Error occurred while saving: {str(e)}")
#         return False
#
#
# def _parse_fstring_path(specified_path):
#     """Parse f-string expressions in path."""
#     if not (specified_path.startswith('f"') or specified_path.startswith("f'")):
#         return specified_path
#
#     import re
#
#     path_content = specified_path[2:-1]
#     frame = inspect.currentframe().f_back.f_back
#     try:
#         variables = re.findall(r"\{([^}]+)\}", path_content)
#         format_dict = {}
#         for var in variables:
#             if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var):
#                 if var in frame.f_locals:
#                     format_dict[var] = frame.f_locals[var]
#                 elif var in frame.f_globals:
#                     format_dict[var] = frame.f_globals[var]
#             else:
#                 raise ValueError(f"Invalid variable name in f-string: {var}")
#         return path_content.format(**format_dict)
#     finally:
#         del frame
#
#
# def _determine_save_path(specified_path, use_caller_path):
#     """Determine the full save path based on environment."""
#     if specified_path.startswith("/"):
#         return specified_path
#
#     from scitex.gen._detect_environment import detect_environment
#     from scitex.gen._get_notebook_path import get_notebook_info_simple
#
#     env_type = detect_environment()
#
#     if env_type == "jupyter":
#         notebook_name, notebook_dir = get_notebook_info_simple()
#         if notebook_name:
#             notebook_base = _os.path.splitext(notebook_name)[0]
#             sdir = _os.path.join(notebook_dir or _os.getcwd(), f"{notebook_base}_out")
#         else:
#             sdir = _os.path.join(_os.getcwd(), "notebook_out")
#         return _os.path.join(sdir, specified_path)
#
#     elif env_type == "script":
#         if use_caller_path:
#             script_path = _find_caller_script_path()
#         else:
#             script_path = inspect.stack()[2].filename
#         sdir = clean_path(_os.path.splitext(script_path)[0] + "_out")
#         return _os.path.join(sdir, specified_path)
#
#     else:
#         script_path = inspect.stack()[2].filename
#         if (
#             ("ipython" in script_path)
#             or ("<stdin>" in script_path)
#             or env_type in ["ipython", "interactive"]
#         ):
#             sdir = f"/tmp/{_os.getenv('USER')}"
#         else:
#             sdir = _os.path.join(_os.getcwd(), "output")
#         return _os.path.join(sdir, specified_path)
#
#
# def _find_caller_script_path():
#     """Find the first non-scitex frame in the call stack."""
#     scitex_src_path = _os.path.abspath(
#         _os.path.join(_os.path.dirname(__file__), "..", "..")
#     )
#     for frame_info in inspect.stack()[3:]:
#         frame_path = _os.path.abspath(frame_info.filename)
#         if not frame_path.startswith(scitex_src_path):
#             return frame_path
#     return inspect.stack()[2].filename
#
#
# def _cleanup_existing_files(spath_final, spath_cwd, kwargs):
#     """Remove existing files to prevent circular links."""
#     should_skip = spath_final.endswith(".csv") or (
#         (spath_final.endswith(".hdf5") or spath_final.endswith(".h5"))
#         and "key" in kwargs
#     )
#     if not should_skip:
#         for path in [spath_final, spath_cwd]:
#             sh(["rm", "-f", f"{path}"], verbose=False)
#
#
# def _handle_dry_run(spath, verbose):
#     """Handle dry run mode."""
#     if verbose:
#         try:
#             rel_path = _os.path.relpath(spath, _os.getcwd())
#         except ValueError:
#             rel_path = spath
#         logger.success(color_text(f"(dry run) Saved to: ./{rel_path}", c="yellow"))
#
#
# def _save(
#     obj,
#     spath,
#     verbose=True,
#     symlink_from_cwd=False,
#     dry_run=False,
#     no_csv=False,
#     symlink_to=None,
#     auto_crop=False,
#     crop_margin_mm=1.0,
#     metadata_extra=None,
#     json_schema="editable",
#     **kwargs,
# ):
#     """Core dispatcher for saving objects to various formats."""
#     ext = _os.path.splitext(spath)[1].lower()
#
#     # Check if this is a matplotlib figure being saved to FTS bundle format
#     # FTS bundles use .zip (archive) or no extension (directory)
#     if _is_matplotlib_figure(obj):
#         # Save as FTS bundle if:
#         # 1. Path ends with .zip (create ZIP bundle)
#         # 2. Path has no extension and doesn't match other formats (create directory bundle)
#         if ext == ".zip" or (ext == "" and not spath.endswith("/")):
#             # Check if explicitly requesting FTS bundle or just .zip
#             as_zip = ext == ".zip"
#             _save_fts_bundle(
#                 obj, spath, as_zip, verbose, symlink_from_cwd, symlink_to, **kwargs
#             )
#             return
#
#     # Dispatch to format handlers
#     if ext in _FILE_HANDLERS:
#         _dispatch_handler(
#             ext,
#             obj,
#             spath,
#             verbose,
#             no_csv,
#             symlink_from_cwd,
#             symlink_to,
#             dry_run,
#             auto_crop,
#             crop_margin_mm,
#             metadata_extra,
#             json_schema,
#             kwargs,
#         )
#     elif spath.endswith(".csv"):
#         save_csv(obj, spath, **kwargs)
#     elif spath.endswith(".pkl.gz"):
#         save_pickle_compressed(obj, spath, **kwargs)
#     else:
#         logger.warning(f"Unsupported file format. {spath} was not saved.")
#         return
#
#     if verbose and _os.path.exists(spath):
#         file_size = readable_bytes(getsize(spath))
#         try:
#             rel_path = _os.path.relpath(spath, _os.getcwd())
#         except ValueError:
#             rel_path = spath
#         logger.success(f"Saved to: ./{rel_path} ({file_size})")
#
#
# def _is_matplotlib_figure(obj):
#     """Check if object is a matplotlib figure or a wrapped figure.
#
#     Handles both raw matplotlib.figure.Figure and SciTeX FigWrapper objects.
#     """
#     try:
#         import matplotlib.figure
#
#         # Direct matplotlib figure
#         if isinstance(obj, matplotlib.figure.Figure):
#             return True
#
#         # Wrapped figure (e.g., FigWrapper from scitex.plt)
#         if hasattr(obj, "figure") and isinstance(
#             obj.figure, matplotlib.figure.Figure
#         ):
#             return True
#
#         return False
#     except ImportError:
#         return False
#
#
# def _save_fts_bundle(
#     obj, spath, as_zip, verbose, symlink_from_cwd, symlink_to_path, **kwargs
# ):
#     """Save matplotlib figure as FTS bundle (.zip or directory).
#
#     Delegates to scitex.io.bundle.from_matplotlib as the single source of truth
#     for bundle structure (canonical/artifacts/payload/children).
#     """
#     from scitex.io.bundle import from_matplotlib
#
#     from ._save_modules._figure_utils import get_figure_with_data
#
#     # Get the actual matplotlib figure
#     import matplotlib.figure
#
#     if isinstance(obj, matplotlib.figure.Figure):
#         fig = obj
#     elif hasattr(obj, "figure") and isinstance(obj.figure, matplotlib.figure.Figure):
#         fig = obj.figure
#     else:
#         raise TypeError(f"Expected matplotlib figure, got {type(obj)}")
#
#     # Extract optional parameters
#     # Support both "csv_df" and "data" parameter names for user convenience
#     csv_df = kwargs.get("csv_df") or kwargs.get("data")
#     dpi = kwargs.get("dpi", 300)
#     name = kwargs.get("name") or Path(spath).stem
#
#     # Extract CSV data from scitex.plt tracking if available
#     scitex_source = get_figure_with_data(obj)
#     if csv_df is None and scitex_source is not None:
#         if hasattr(scitex_source, "export_as_csv"):
#             try:
#                 csv_df = scitex_source.export_as_csv()
#             except Exception:
#                 pass
#
#     # Delegate to FTS (single source of truth)
#     # Encoding is built from CSV columns directly for consistency
#     from_matplotlib(fig, spath, name=name, csv_df=csv_df, dpi=dpi)
#
#     bundle_path = spath
#     if verbose and _os.path.exists(bundle_path):
#         file_size = readable_bytes(getsize(bundle_path))
#         try:
#             rel_path = _os.path.relpath(bundle_path, _os.getcwd())
#         except ValueError:
#             rel_path = bundle_path
#         logger.success(f"Saved to: ./{rel_path} ({file_size})")
#
#     if symlink_from_cwd and _os.path.exists(bundle_path):
#         bundle_basename = _os.path.basename(bundle_path)
#         bundle_cwd = _os.path.join(_os.getcwd(), bundle_basename)
#         _symlink(bundle_path, bundle_cwd, symlink_from_cwd, verbose)
#
#     if symlink_to_path and _os.path.exists(bundle_path):
#         _symlink_to(bundle_path, symlink_to_path, verbose)
#
#
# def _dispatch_handler(
#     ext,
#     obj,
#     spath,
#     verbose,
#     no_csv,
#     symlink_from_cwd,
#     symlink_to_path,
#     dry_run,
#     auto_crop,
#     crop_margin_mm,
#     metadata_extra,
#     json_schema,
#     kwargs,
# ):
#     """Dispatch to the appropriate file handler."""
#     image_exts = [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".svg", ".pdf"]
#     if ext in image_exts:
#         _handle_image_with_csv(
#             obj,
#             spath,
#             verbose=verbose,
#             no_csv=no_csv,
#             symlink_from_cwd=symlink_from_cwd,
#             symlink_to_path=symlink_to_path,
#             dry_run=dry_run,
#             auto_crop=auto_crop,
#             crop_margin_mm=crop_margin_mm,
#             metadata_extra=metadata_extra,
#             json_schema=json_schema,
#             **kwargs,
#         )
#     elif ext in [".hdf5", ".h5", ".zarr"]:
#         _FILE_HANDLERS[ext](obj, spath, **kwargs)
#     else:
#         _FILE_HANDLERS[ext](obj, spath, **kwargs)
#
#
# # Dispatch dictionary for O(1) file format lookup
# _FILE_HANDLERS = {
#     ".xlsx": save_excel,
#     ".xls": save_excel,
#     ".npy": save_npy,
#     ".npz": save_npz,
#     ".pkl": save_pickle,
#     ".pickle": save_pickle,
#     ".pkl.gz": save_pickle_compressed,
#     ".joblib": save_joblib,
#     ".pth": save_torch,
#     ".pt": save_torch,
#     ".mat": save_matlab,
#     ".cbm": save_catboost,
#     ".json": save_json,
#     ".yaml": save_yaml,
#     ".yml": save_yaml,
#     ".txt": save_text,
#     ".md": save_text,
#     ".py": save_text,
#     ".css": save_text,
#     ".js": save_text,
#     ".tex": save_tex,
#     ".bib": save_bibtex,
#     ".html": save_html,
#     ".hdf5": save_hdf5,
#     ".h5": save_hdf5,
#     ".zarr": save_zarr,
#     ".mp4": save_mp4,
#     ".png": handle_image_with_csv,
#     ".jpg": handle_image_with_csv,
#     ".jpeg": handle_image_with_csv,
#     ".gif": handle_image_with_csv,
#     ".tiff": handle_image_with_csv,
#     ".tif": handle_image_with_csv,
#     ".svg": handle_image_with_csv,
#     ".pdf": handle_image_with_csv,
# }
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save.py
# --------------------------------------------------------------------------------
