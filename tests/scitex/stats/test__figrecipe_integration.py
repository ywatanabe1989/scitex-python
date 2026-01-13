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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_figrecipe_integration.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2026-01-12 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_figrecipe_integration.py
# """Figrecipe integration for statistical annotations on plots."""
# 
# from typing import Any, Dict, List, Optional, Union
# 
# try:
#     from figrecipe._integrations._scitex_stats import (
#         annotate_from_stats as _fr_annotate,
#     )
#     from figrecipe._integrations._scitex_stats import (
#         from_scitex_stats as _fr_convert,
#     )
#     from figrecipe._integrations._scitex_stats import (
#         load_stats_bundle as _fr_load_bundle,
#     )
# 
#     _AVAILABLE = True
# except ImportError:
#     _AVAILABLE = False
# 
# 
# def to_figrecipe(
#     stats_result: Union[Dict[str, Any], List[Dict[str, Any]]],
# ) -> Dict[str, Any]:
#     """Convert scitex.stats result(s) to figrecipe format.
# 
#     Parameters
#     ----------
#     stats_result : dict or list of dict
#         Statistical result(s) from scitex.stats functions.
# 
#     Returns
#     -------
#     dict
#         Figrecipe-compatible format with 'comparisons' list.
#     """
#     if not _AVAILABLE:
#         raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")
#     return _fr_convert(stats_result)
# 
# 
# def annotate(
#     ax,
#     stats: Union[Dict[str, Any], List[Dict[str, Any]]],
#     positions: Optional[Dict[str, float]] = None,
#     style: str = "stars",
#     **kwargs,
# ) -> List[Any]:
#     """Add statistical annotations to a plot.
# 
#     Parameters
#     ----------
#     ax : Axes or AxisWrapper
#         The axes to annotate.
#     stats : dict or list of dict
#         Statistical results (auto-converted if needed).
#     positions : dict, optional
#         Group name to x position mapping.
#     style : str
#         'stars', 'p_value', or 'both'.
# 
#     Returns
#     -------
#     list
#         Created matplotlib artist objects.
#     """
#     if not _AVAILABLE:
#         raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")
# 
#     # Convert if needed
#     if isinstance(stats, list) or "comparisons" not in stats:
#         stats = _fr_convert(stats)
# 
#     # Unwrap scitex AxisWrapper
#     ax_mpl = getattr(ax, "_axis_mpl", getattr(ax, "_ax", ax))
# 
#     return _fr_annotate(ax_mpl, stats, positions=positions, style=style, **kwargs)
# 
# 
# def load_and_annotate(
#     ax,
#     path: str,
#     positions: Optional[Dict[str, float]] = None,
#     style: str = "stars",
#     **kwargs,
# ) -> List[Any]:
#     """Load stats from bundle file and annotate plot.
# 
#     Parameters
#     ----------
#     ax : Axes or AxisWrapper
#         The axes to annotate.
#     path : str
#         Path to .statsz or .zip bundle.
#     positions : dict, optional
#         Group name to x position mapping.
#     style : str
#         'stars', 'p_value', or 'both'.
# 
#     Returns
#     -------
#     list
#         Created matplotlib artist objects.
#     """
#     if not _AVAILABLE:
#         raise ImportError("figrecipe >= 0.13.0 required: pip install figrecipe")
# 
#     fr_stats = _fr_load_bundle(path)
#     return annotate(ax, fr_stats, positions=positions, style=style, **kwargs)
# 
# 
# __all__ = ["to_figrecipe", "annotate", "load_and_annotate"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_figrecipe_integration.py
# --------------------------------------------------------------------------------
