#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 13:25:00 (ywatanabe)"
# File: tests/scitex/plt/_subplots/_export_as_csv_formatters/test__format_imshow.py

"""Tests for _format_imshow CSV formatter."""

import numpy as np
import pandas as pd
import pytest

from scitex.plt._subplots._export_as_csv_formatters._format_imshow import _format_imshow


class TestFormatImshow:
    """Tests for _format_imshow function."""

    def test_empty_tracked_dict_returns_empty_df(self):
        """Empty tracked_dict should return empty DataFrame."""
        result = _format_imshow("test", {}, {})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_none_tracked_dict_returns_empty_df(self):
        """None tracked_dict should return empty DataFrame."""
        result = _format_imshow("test", None, {})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_image_df_key_returns_directly(self):
        """When image_df key exists, should return it directly."""
        image_df = pd.DataFrame({"row": [0, 1], "col": [0, 1], "value": [0.5, 0.8]})
        tracked_dict = {"image_df": image_df}

        result = _format_imshow("ax_00", tracked_dict, {})

        pd.testing.assert_frame_equal(result, image_df)

    def test_args_2d_grayscale_image(self):
        """2D grayscale image should be flattened with row/col indices."""
        img = np.array([[0.1, 0.2], [0.3, 0.4]])
        tracked_dict = {"args": (img,)}

        result = _format_imshow("ax_00", tracked_dict, {})

        assert "ax_00_imshow_row" in result.columns
        assert "ax_00_imshow_col" in result.columns
        assert "ax_00_imshow_value" in result.columns
        assert len(result) == 4  # 2x2 = 4 pixels

        # Check values are correct
        expected_values = [0.1, 0.2, 0.3, 0.4]
        np.testing.assert_array_almost_equal(
            result["ax_00_imshow_value"].values, expected_values
        )

    def test_args_3d_rgb_image(self):
        """3D RGB image should have R, G, B columns."""
        img = np.zeros((2, 2, 3))
        img[:, :, 0] = [[255, 0], [0, 255]]  # R
        img[:, :, 1] = [[0, 255], [255, 0]]  # G
        img[:, :, 2] = [[128, 128], [128, 128]]  # B
        tracked_dict = {"args": (img,)}

        result = _format_imshow("ax_00", tracked_dict, {})

        assert "ax_00_imshow_row" in result.columns
        assert "ax_00_imshow_col" in result.columns
        assert "ax_00_imshow_R" in result.columns
        assert "ax_00_imshow_G" in result.columns
        assert "ax_00_imshow_B" in result.columns
        assert len(result) == 4  # 2x2 = 4 pixels

    def test_args_3d_rgba_image(self):
        """3D RGBA image should have R, G, B, A columns."""
        img = np.zeros((2, 2, 4))
        img[:, :, 0] = 1.0  # R
        img[:, :, 1] = 0.5  # G
        img[:, :, 2] = 0.0  # B
        img[:, :, 3] = 0.8  # A
        tracked_dict = {"args": (img,)}

        result = _format_imshow("ax_00", tracked_dict, {})

        assert "ax_00_imshow_R" in result.columns
        assert "ax_00_imshow_G" in result.columns
        assert "ax_00_imshow_B" in result.columns
        assert "ax_00_imshow_A" in result.columns

    def test_row_col_indices_correct(self):
        """Row and column indices should be meshgrid-like."""
        img = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows, 3 cols
        tracked_dict = {"args": (img,)}

        result = _format_imshow("ax_00", tracked_dict, {})

        # Check row indices
        expected_rows = [0, 0, 0, 1, 1, 1]
        expected_cols = [0, 1, 2, 0, 1, 2]
        np.testing.assert_array_equal(
            result["ax_00_imshow_row"].values, expected_rows
        )
        np.testing.assert_array_equal(
            result["ax_00_imshow_col"].values, expected_cols
        )

    def test_id_prefix_applied_correctly(self):
        """ID prefix should be correctly applied to all columns."""
        img = np.array([[1, 2], [3, 4]])
        tracked_dict = {"args": (img,)}

        result = _format_imshow("custom_prefix", tracked_dict, {})

        for col in result.columns:
            assert col.startswith("custom_prefix_")

    def test_large_image_handling(self):
        """Should handle larger images without issues."""
        img = np.random.rand(100, 100)
        tracked_dict = {"args": (img,)}

        result = _format_imshow("ax_00", tracked_dict, {})

        assert len(result) == 10000  # 100x100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
