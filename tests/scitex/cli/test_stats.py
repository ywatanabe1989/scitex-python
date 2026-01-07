#!/usr/bin/env python3
"""Tests for scitex.cli.stats - Statistical analysis CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.stats import stats


class TestStatsGroup:
    """Tests for the stats command group."""

    def test_stats_help(self):
        """Test that stats help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(stats, ["--help"])
        assert result.exit_code == 0
        assert "Statistical analysis" in result.output

    def test_stats_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(stats, ["--help"])
        expected_commands = ["recommend", "describe", "save", "load", "tests"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in stats help"


class TestStatsRecommend:
    """Tests for the stats recommend command."""

    def test_recommend_basic(self):
        """Test recommend command with basic options."""
        runner = CliRunner()
        with patch("scitex.stats.StatContext") as mock_ctx:
            with patch("scitex.stats.recommend_tests") as mock_recommend:
                mock_recommend.return_value = ["t-test", "Mann-Whitney U"]
                result = runner.invoke(
                    stats,
                    [
                        "recommend",
                        "--n-groups",
                        "2",
                        "--outcome-type",
                        "continuous",
                    ],
                )
                assert result.exit_code == 0
                assert "Recommended" in result.output
                mock_recommend.assert_called_once()

    def test_recommend_json(self):
        """Test recommend command with --json flag."""
        runner = CliRunner()
        with patch("scitex.stats.StatContext") as mock_ctx:
            with patch("scitex.stats.recommend_tests") as mock_recommend:
                mock_recommend.return_value = ["ANOVA", "Kruskal-Wallis"]
                result = runner.invoke(
                    stats,
                    [
                        "recommend",
                        "--n-groups",
                        "3",
                        "--outcome-type",
                        "continuous",
                        "--json",
                    ],
                )
                assert result.exit_code == 0
                output = json.loads(result.output)
                assert "recommended_tests" in output

    def test_recommend_with_paired(self):
        """Test recommend command with paired flag."""
        runner = CliRunner()
        with patch("scitex.stats.StatContext") as mock_ctx:
            with patch("scitex.stats.recommend_tests") as mock_recommend:
                mock_recommend.return_value = ["paired t-test"]
                result = runner.invoke(
                    stats,
                    [
                        "recommend",
                        "--n-groups",
                        "2",
                        "--outcome-type",
                        "continuous",
                        "--design",
                        "within",
                        "--paired",
                    ],
                )
                assert result.exit_code == 0

    def test_recommend_categorical(self):
        """Test recommend command with categorical outcome."""
        runner = CliRunner()
        with patch("scitex.stats.StatContext") as mock_ctx:
            with patch("scitex.stats.recommend_tests") as mock_recommend:
                mock_recommend.return_value = ["chi-square"]
                result = runner.invoke(
                    stats,
                    [
                        "recommend",
                        "--n-groups",
                        "2",
                        "--outcome-type",
                        "binary",
                    ],
                )
                assert result.exit_code == 0


class TestStatsDescribe:
    """Tests for the stats describe command."""

    def test_describe_with_file(self):
        """Test describe command with data file."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
            temp_path = f.name

        try:
            with patch("scitex.stats.describe") as mock_describe:
                import pandas as pd

                mock_describe.return_value = pd.DataFrame(
                    {"count": [3, 3, 3], "mean": [4.0, 5.0, 6.0]}
                )
                result = runner.invoke(stats, ["describe", temp_path])
                assert result.exit_code == 0
                assert "Descriptive Statistics" in result.output
        finally:
            os.unlink(temp_path)

    def test_describe_json(self):
        """Test describe command with --json flag."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n3,4\n")
            temp_path = f.name

        try:
            with patch("scitex.stats.describe") as mock_describe:
                mock_describe.return_value = {"count": 2, "mean": 2.5}
                result = runner.invoke(stats, ["describe", temp_path, "--json"])
                assert result.exit_code == 0
                output = json.loads(result.output)
                assert "count" in output
        finally:
            os.unlink(temp_path)

    def test_describe_with_column(self):
        """Test describe command with specific column."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b\n1,2\n3,4\n")
            temp_path = f.name

        try:
            with patch("scitex.stats.describe") as mock_describe:
                mock_describe.return_value = {"count": 2}
                result = runner.invoke(stats, ["describe", temp_path, "--column", "a"])
                assert result.exit_code == 0
        finally:
            os.unlink(temp_path)


class TestStatsSave:
    """Tests for the stats save command."""

    def test_save_basic(self):
        """Test save command."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input JSON file
            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump({"test": "t-test", "p_value": 0.05}, f)

            output_path = os.path.join(tmpdir, "stats.stats")
            with patch("scitex.stats.save_stats") as mock_save:
                mock_save.return_value = output_path
                result = runner.invoke(
                    stats,
                    ["save", input_path, "--output", output_path],
                )
                assert result.exit_code == 0
                assert "saved" in result.output.lower()

    def test_save_with_as_zip(self):
        """Test save command with --as-zip flag."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump([{"test": "t-test", "p_value": 0.05}], f)

            output_path = os.path.join(tmpdir, "stats.stats")
            with patch("scitex.stats.save_stats") as mock_save:
                mock_save.return_value = output_path
                result = runner.invoke(
                    stats,
                    ["save", input_path, "--output", output_path, "--as-zip"],
                )
                assert result.exit_code == 0


class TestStatsLoad:
    """Tests for the stats load command."""

    def test_load_basic(self):
        """Test load command."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stats", delete=False) as f:
            f.write('{"comparisons": [{"name": "test", "p_value": 0.05}]}')
            temp_path = f.name

        try:
            with patch("scitex.stats.load_stats") as mock_load:
                mock_load.return_value = {
                    "comparisons": [
                        {"name": "test", "method": "t-test", "p_value": 0.05}
                    ]
                }
                result = runner.invoke(stats, ["load", temp_path])
                assert result.exit_code == 0
                assert "Statistics Bundle" in result.output
        finally:
            os.unlink(temp_path)

    def test_load_json(self):
        """Test load command with --json flag."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stats", delete=False) as f:
            f.write('{"p_value": 0.05}')
            temp_path = f.name

        try:
            with patch("scitex.stats.load_stats") as mock_load:
                mock_load.return_value = {"comparisons": [], "p_value": 0.05}
                result = runner.invoke(stats, ["load", temp_path, "--json"])
                assert result.exit_code == 0
                output = json.loads(result.output)
                assert "p_value" in output or "comparisons" in output
        finally:
            os.unlink(temp_path)


class TestStatsTests:
    """Tests for the stats tests command."""

    def test_tests_list(self):
        """Test tests list command."""
        runner = CliRunner()
        with patch("scitex.stats.TEST_RULES") as mock_rules:
            mock_rule = MagicMock()
            mock_rule.category = "parametric"
            mock_rule.name = "t-test"
            mock_rules.__iter__ = lambda self: iter([mock_rule])
            result = runner.invoke(stats, ["tests"])
            assert result.exit_code == 0
            assert "Available Statistical Tests" in result.output


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
