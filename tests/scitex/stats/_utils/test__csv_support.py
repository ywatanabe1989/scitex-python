#!/usr/bin/env python3
"""Tests for seaborn-style data= parameter support in scitex.stats."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scitex.stats._utils._csv_support import resolve_columns, resolve_groups

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """DataFrame with two numeric columns and a group column."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "before": np.random.normal(100, 10, 30),
            "after": np.random.normal(105, 10, 30),
            "score": np.concatenate(
                [
                    np.random.normal(100, 10, 10),
                    np.random.normal(110, 10, 10),
                    np.random.normal(120, 10, 10),
                ]
            ),
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        }
    )


@pytest.fixture
def csv_path(sample_df):
    """Write sample_df to a temporary CSV and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def df_with_nan():
    """DataFrame containing NaN values."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
            "y": [10.0, np.nan, 30.0, 40.0, 50.0],
        }
    )


# ---------------------------------------------------------------------------
# resolve_columns tests
# ---------------------------------------------------------------------------


class TestResolveColumns:
    def test_two_columns(self, sample_df):
        resolved = resolve_columns(sample_df, x="before", y="after")
        assert "x" in resolved
        assert "y" in resolved
        assert isinstance(resolved["x"], np.ndarray)
        assert len(resolved["x"]) == 30

    def test_single_column(self, sample_df):
        resolved = resolve_columns(sample_df, x="before")
        assert "x" in resolved
        assert len(resolved["x"]) == 30

    def test_csv_path(self, csv_path):
        resolved = resolve_columns(csv_path, x="before", y="after")
        assert isinstance(resolved["x"], np.ndarray)
        assert len(resolved["x"]) == 30

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            resolve_columns(sample_df, x="nonexistent")

    def test_nan_dropped(self, df_with_nan):
        resolved = resolve_columns(df_with_nan, x="x")
        assert len(resolved["x"]) == 4
        assert not np.isnan(resolved["x"]).any()

    def test_passthrough_non_string(self, sample_df):
        arr = np.array([1, 2, 3])
        resolved = resolve_columns(sample_df, x=arr, y="after")
        assert np.array_equal(resolved["x"], arr)
        assert isinstance(resolved["y"], np.ndarray)


# ---------------------------------------------------------------------------
# resolve_groups tests
# ---------------------------------------------------------------------------


class TestResolveGroups:
    def test_basic_split(self, sample_df):
        groups, names = resolve_groups(sample_df, "score", "group")
        assert len(groups) == 3
        assert names == ["A", "B", "C"]
        assert all(isinstance(g, np.ndarray) for g in groups)

    def test_two_groups(self, sample_df):
        df2 = sample_df[sample_df["group"].isin(["A", "B"])]
        groups, names = resolve_groups(df2, "score", "group")
        assert len(groups) == 2
        assert names == ["A", "B"]

    def test_csv_path(self, csv_path):
        groups, names = resolve_groups(csv_path, "score", "group")
        assert len(groups) == 3

    def test_missing_value_col_raises(self, sample_df):
        with pytest.raises(ValueError, match="value_col 'bad'"):
            resolve_groups(sample_df, "bad", "group")

    def test_missing_group_col_raises(self, sample_df):
        with pytest.raises(ValueError, match="group_col 'bad'"):
            resolve_groups(sample_df, "score", "bad")


# ---------------------------------------------------------------------------
# Integration tests: test functions with data= parameter
# ---------------------------------------------------------------------------


class TestTwoSampleDataParam:
    """Test data= parameter on two-sample (x, y) test functions."""

    def test_ttest_ind_with_data(self, sample_df):
        from scitex.stats import test_ttest_ind

        result = test_ttest_ind(x="before", y="after", data=sample_df)
        assert "pvalue" in result

    def test_ttest_rel_with_data(self, sample_df):
        from scitex.stats import test_ttest_rel

        result = test_ttest_rel(x="before", y="after", data=sample_df)
        assert "pvalue" in result

    def test_pearson_with_data(self, sample_df):
        from scitex.stats import test_pearson

        result = test_pearson(x="before", y="after", data=sample_df)
        assert "pvalue" in result

    def test_mannwhitneyu_with_data(self, sample_df):
        from scitex.stats import test_mannwhitneyu

        result = test_mannwhitneyu(x="before", y="after", data=sample_df)
        assert "pvalue" in result

    def test_csv_path_as_data(self, csv_path):
        from scitex.stats import test_ttest_ind

        result = test_ttest_ind(x="before", y="after", data=csv_path)
        assert "pvalue" in result

    def test_backward_compat_arrays(self, sample_df):
        from scitex.stats import test_ttest_ind

        result = test_ttest_ind(sample_df["before"].values, sample_df["after"].values)
        assert "pvalue" in result


class TestOneSampleDataParam:
    """Test data= parameter on one-sample (x) test functions."""

    def test_shapiro_with_data(self, sample_df):
        from scitex.stats import test_shapiro

        result = test_shapiro(x="before", data=sample_df)
        assert "pvalue" in result

    def test_ttest_1samp_with_data(self, sample_df):
        from scitex.stats import test_ttest_1samp

        result = test_ttest_1samp(x="before", popmean=100, data=sample_df)
        assert "pvalue" in result

    def test_backward_compat_array(self, sample_df):
        from scitex.stats import test_shapiro

        result = test_shapiro(sample_df["before"].values)
        assert "pvalue" in result


class TestMultiGroupDataParam:
    """Test data= parameter on multi-group (groups) test functions."""

    def test_anova_with_data(self, sample_df):
        from scitex.stats import test_anova

        result = test_anova(data=sample_df, value_col="score", group_col="group")
        assert "pvalue" in result

    def test_kruskal_with_data(self, sample_df):
        from scitex.stats import test_kruskal

        result = test_kruskal(data=sample_df, value_col="score", group_col="group")
        assert "pvalue" in result

    def test_anova_backward_compat(self, sample_df):
        from scitex.stats import test_anova

        groups = [
            sample_df[sample_df["group"] == g]["score"].values for g in ["A", "B", "C"]
        ]
        result = test_anova(groups)
        assert "pvalue" in result


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
