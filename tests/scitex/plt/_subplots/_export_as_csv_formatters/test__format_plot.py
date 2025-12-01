#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 13:20:00 (ywatanabe)"
# File: tests/scitex/plt/_subplots/_export_as_csv_formatters/test__format_plot.py

"""Tests for _format_plot CSV formatter."""

import numpy as np
import pandas as pd
import pytest

from scitex.plt._subplots._export_as_csv_formatters._format_plot import _format_plot


class TestFormatPlot:
    """Tests for _format_plot function."""

    def test_empty_tracked_dict_returns_empty_df(self):
        """Empty tracked_dict should return empty DataFrame."""
        result = _format_plot("test", {}, {})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_none_tracked_dict_returns_empty_df(self):
        """None tracked_dict should return empty DataFrame."""
        result = _format_plot("test", None, {})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_plot_df_key_returns_prefixed_df(self):
        """When plot_df key exists, should return prefixed DataFrame."""
        plot_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        tracked_dict = {"plot_df": plot_df}

        result = _format_plot("ax_00", tracked_dict, {})

        assert "ax_00_x" in result.columns
        assert "ax_00_y" in result.columns
        assert list(result["ax_00_x"]) == [1, 2, 3]

    def test_args_single_1d_array(self):
        """Single 1D array arg should generate x from indices."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        tracked_dict = {"args": (y,)}

        result = _format_plot("ax_00", tracked_dict, {})

        assert "ax_00_plot_x" in result.columns
        assert "ax_00_plot_y" in result.columns
        assert list(result["ax_00_plot_x"]) == [0, 1, 2, 3]
        assert list(result["ax_00_plot_y"]) == [1.0, 2.0, 3.0, 4.0]

    def test_args_single_2d_array(self):
        """Single 2D array arg should extract x and y columns."""
        data = np.array([[0, 1], [1, 4], [2, 9]])
        tracked_dict = {"args": (data,)}

        result = _format_plot("ax_00", tracked_dict, {})

        assert "ax_00_plot_x" in result.columns
        assert "ax_00_plot_y" in result.columns
        assert list(result["ax_00_plot_x"]) == [0, 1, 2]
        assert list(result["ax_00_plot_y"]) == [1, 4, 9]

    def test_args_two_1d_arrays(self):
        """Two 1D array args should use first as x, second as y."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        tracked_dict = {"args": (x, y)}

        result = _format_plot("ax_00", tracked_dict, {})

        assert "ax_00_plot_x" in result.columns
        assert "ax_00_plot_y" in result.columns
        np.testing.assert_array_equal(result["ax_00_plot_x"], x)
        np.testing.assert_array_equal(result["ax_00_plot_y"], y)

    def test_args_x_and_2d_y(self):
        """X and 2D Y arrays should create multiple y columns."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([[1, 2], [3, 4], [5, 6]])
        tracked_dict = {"args": (x, y)}

        result = _format_plot("ax_00", tracked_dict, {})

        assert "ax_00_plot_x00" in result.columns
        assert "ax_00_plot_y00" in result.columns
        assert "ax_00_plot_x01" in result.columns
        assert "ax_00_plot_y01" in result.columns

    def test_args_x_and_dataframe_y(self):
        """X and DataFrame Y should handle column iteration with indexed columns."""
        x = np.array([0.0, 1.0, 2.0])
        y = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        tracked_dict = {"args": (x, y)}

        result = _format_plot("ax_00", tracked_dict, {})

        # DataFrame Y creates indexed x and y columns for each DataFrame column
        assert "ax_00_plot_x00" in result.columns
        assert "ax_00_plot_y00" in result.columns
        assert "ax_00_plot_x01" in result.columns
        assert "ax_00_plot_y01" in result.columns

    def test_id_prefix_applied_correctly(self):
        """ID prefix should be correctly applied to all columns."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        tracked_dict = {"args": (x, y)}

        result = _format_plot("custom_prefix", tracked_dict, {})

        for col in result.columns:
            assert col.startswith("custom_prefix_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
