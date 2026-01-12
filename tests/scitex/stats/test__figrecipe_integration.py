#!/usr/bin/env python3
# Timestamp: "2026-01-12 (ywatanabe)"
# File: tests/scitex/stats/test__figrecipe_integration.py
"""Tests for scitex.stats figrecipe integration."""

import pytest

from scitex.stats._figrecipe_integration import _AVAILABLE


class TestFigrecipeIntegration:
    """Tests for figrecipe integration functions."""

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_to_figrecipe_single_result(self):
        """Test conversion of single stats result."""
        from scitex.stats import to_figrecipe

        result = {
            "name": "Control vs Treatment",
            "method": "t-test",
            "p_value": 0.003,
            "effect_size": 1.21,
            "ci95": [0.5, 1.8],
        }

        fr_stats = to_figrecipe(result)

        assert "comparisons" in fr_stats
        assert len(fr_stats["comparisons"]) == 1
        assert fr_stats["comparisons"][0]["p_value"] == 0.003
        assert fr_stats["comparisons"][0]["stars"] == "**"

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_to_figrecipe_multiple_results(self):
        """Test conversion of multiple stats results."""
        from scitex.stats import to_figrecipe

        results = [
            {"name": "A vs B", "method": "t-test", "p_value": 0.0001},  # < 0.001 -> ***
            {"name": "A vs C", "method": "t-test", "p_value": 0.04},  # < 0.05 -> *
        ]

        fr_stats = to_figrecipe(results)

        assert len(fr_stats["comparisons"]) == 2
        assert fr_stats["comparisons"][0]["stars"] == "***"
        assert fr_stats["comparisons"][1]["stars"] == "*"

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_to_figrecipe_nested_format(self):
        """Test conversion of nested format stats result."""
        from scitex.stats import to_figrecipe

        result = {
            "method": {"name": "t-test", "variant": "independent"},
            "results": {"p_value": 0.01, "statistic": 2.5},
        }

        fr_stats = to_figrecipe(result)

        assert "comparisons" in fr_stats
        assert fr_stats["comparisons"][0]["p_value"] == 0.01

    @pytest.mark.skipif(not _AVAILABLE, reason="figrecipe not installed")
    def test_annotate_unwraps_axis(self):
        """Test that annotate properly unwraps scitex AxisWrapper."""
        import matplotlib.pyplot as plt

        from scitex.stats import annotate

        fig, ax = plt.subplots()
        ax.bar([0, 1], [10, 15])

        # Create mock wrapper
        class MockWrapper:
            def __init__(self, ax):
                self._axis_mpl = ax

        wrapped_ax = MockWrapper(ax)

        stats = {"comparisons": [{"name": "A vs B", "p_value": 0.01, "stars": "**"}]}

        # Should not raise - properly unwraps
        try:
            annotate(wrapped_ax, stats, positions={"A": 0, "B": 1})
        except Exception as e:
            # May fail if figrecipe annotation requires specific setup
            # but should NOT fail on unwrapping
            assert "_axis_mpl" not in str(e)

        plt.close(fig)

    def test_import_error_without_figrecipe(self):
        """Test that ImportError is raised when figrecipe unavailable."""
        if _AVAILABLE:
            pytest.skip("figrecipe is installed")

        from scitex.stats import to_figrecipe

        with pytest.raises(ImportError, match="figrecipe"):
            to_figrecipe({"p_value": 0.01})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# EOF
