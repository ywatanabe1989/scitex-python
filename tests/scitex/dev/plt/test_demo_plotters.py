#!/usr/bin/env python3
"""Tests for scitex.dev.plt demo_plotters module.

These tests verify the plotter registry and basic function properties.
The actual plotting functionality requires the full scitex session system.
"""

import inspect
import os

import pytest


class TestPlottersRegistry:
    """Tests for the PLOTTERS registry."""

    def test_plotters_stx_not_empty(self):
        """Test that PLOTTERS_STX contains entries."""
        from scitex.dev.plt import PLOTTERS_STX

        assert len(PLOTTERS_STX) > 0

    def test_plotters_sns_not_empty(self):
        """Test that PLOTTERS_SNS contains entries."""
        from scitex.dev.plt import PLOTTERS_SNS

        assert len(PLOTTERS_SNS) > 0

    def test_plotters_mpl_not_empty(self):
        """Test that PLOTTERS_MPL contains entries."""
        from scitex.dev.plt import PLOTTERS_MPL

        assert len(PLOTTERS_MPL) > 0

    def test_combined_plotters_contains_all(self):
        """Test that PLOTTERS contains all individual registries."""
        from scitex.dev.plt import PLOTTERS, PLOTTERS_MPL, PLOTTERS_SNS, PLOTTERS_STX

        assert len(PLOTTERS) == len(PLOTTERS_STX) + len(PLOTTERS_SNS) + len(
            PLOTTERS_MPL
        )

    def test_all_plotters_are_callable(self):
        """Test that all entries in PLOTTERS are callable."""
        from scitex.dev.plt import PLOTTERS

        for name, plotter in PLOTTERS.items():
            assert callable(plotter), f"Plotter {name} is not callable"

    def test_stx_plotters_naming_convention(self):
        """Test that STX plotters follow naming convention."""
        from scitex.dev.plt import PLOTTERS_STX

        for name in PLOTTERS_STX.keys():
            assert name.startswith("stx_"), (
                f"STX plotter {name} doesn't start with stx_"
            )

    def test_sns_plotters_naming_convention(self):
        """Test that SNS plotters follow naming convention."""
        from scitex.dev.plt import PLOTTERS_SNS

        for name in PLOTTERS_SNS.keys():
            assert name.startswith("sns_"), (
                f"SNS plotter {name} doesn't start with sns_"
            )

    def test_mpl_plotters_naming_convention(self):
        """Test that MPL plotters follow naming convention."""
        from scitex.dev.plt import PLOTTERS_MPL

        for name in PLOTTERS_MPL.keys():
            assert name.startswith("mpl_"), (
                f"MPL plotter {name} doesn't start with mpl_"
            )


class TestPlotterSignatures:
    """Tests for plotter function signatures."""

    def test_all_plotters_have_plt_parameter(self):
        """Test that all plotters have 'plt' as first parameter."""
        from scitex.dev.plt import PLOTTERS

        for name, plotter in PLOTTERS.items():
            sig = inspect.signature(plotter)
            params = list(sig.parameters.keys())
            assert len(params) >= 2, f"Plotter {name} has too few parameters"
            assert params[0] == "plt", f"Plotter {name}'s first param is not 'plt'"

    def test_all_plotters_have_rng_parameter(self):
        """Test that all plotters have 'rng' as second parameter."""
        from scitex.dev.plt import PLOTTERS

        for name, plotter in PLOTTERS.items():
            sig = inspect.signature(plotter)
            params = list(sig.parameters.keys())
            assert params[1] == "rng", f"Plotter {name}'s second param is not 'rng'"

    def test_all_plotters_have_optional_ax(self):
        """Test that all plotters have optional 'ax' parameter."""
        from scitex.dev.plt import PLOTTERS

        for name, plotter in PLOTTERS.items():
            sig = inspect.signature(plotter)
            params = sig.parameters
            assert "ax" in params, f"Plotter {name} doesn't have 'ax' parameter"
            # ax should have a default value (None)
            ax_param = params["ax"]
            assert ax_param.default is None, f"Plotter {name}'s ax default is not None"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_plotter_returns_function(self):
        """Test that get_plotter returns a callable."""
        from scitex.dev.plt import get_plotter

        plotter = get_plotter("stx_line")
        assert callable(plotter)

    def test_get_plotter_raises_on_unknown(self):
        """Test that get_plotter raises KeyError for unknown plotter."""
        from scitex.dev.plt import get_plotter

        with pytest.raises(KeyError):
            get_plotter("unknown_plotter")

    def test_list_plotters_returns_list(self):
        """Test that list_plotters returns a list."""
        from scitex.dev.plt import list_plotters

        result = list_plotters()
        assert isinstance(result, list)
        assert len(result) > 0


class TestIndividualPlotterImports:
    """Tests for individual plotter imports."""

    @pytest.mark.parametrize(
        "plotter_name",
        [
            "plot_stx_line",
            "plot_stx_scatter",
            "plot_stx_bar",
            "plot_sns_boxplot",
            "plot_sns_heatmap",
            "plot_mpl_plot",
            "plot_mpl_scatter",
        ],
    )
    def test_plotter_importable(self, plotter_name):
        """Test that individual plotters can be imported."""
        from scitex.dev import plt as dev_plt

        assert hasattr(dev_plt, plotter_name), f"{plotter_name} not in dev.plt"
        plotter = getattr(dev_plt, plotter_name)
        assert callable(plotter)


class TestPlotterDocstrings:
    """Tests for plotter documentation."""

    def test_all_plotters_have_docstrings(self):
        """Test that all plotters have docstrings."""
        from scitex.dev.plt import PLOTTERS

        for name, plotter in PLOTTERS.items():
            assert plotter.__doc__ is not None, f"Plotter {name} has no docstring"
            assert len(plotter.__doc__) > 0, f"Plotter {name} has empty docstring"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
