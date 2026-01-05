#!/usr/bin/env python3
"""SubplotsWrapper: Monitor data plotted using matplotlib for CSV export."""

import os
from collections import OrderedDict

import matplotlib.pyplot as plt

__FILE__ = "./src/scitex/plt/_subplots/_SubplotsWrapper.py"
__DIR__ = os.path.dirname(__FILE__)

# Configure fonts at import
from ._fonts import _arial_enabled  # noqa: F401
from ._mm_layout import create_with_mm_control

# Register Arial fonts at module import
import matplotlib.font_manager as fm
import matplotlib as mpl
import os

_arial_enabled = False

# Try to find Arial
try:
    fm.findfont("Arial", fallback_to_default=False)
    _arial_enabled = True
except Exception:
    # Search for Arial font files and register them
    arial_paths = [
        f
        for f in fm.findSystemFonts()
        if os.path.basename(f).lower().startswith("arial")
    ]

    if arial_paths:
        for path in arial_paths:
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass

        # Verify Arial is now available
        try:
            fm.findfont("Arial", fallback_to_default=False)
            _arial_enabled = True
        except Exception:
            pass

# Configure matplotlib to use Arial if available
if _arial_enabled:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
    ]
else:
    # Warn about missing Arial
    from scitex import logging as _logging

    _logger = _logging.getLogger(__name__)
    _logger.warning(
        "Arial font not found. Using fallback fonts (Helvetica/DejaVu Sans). "
        "For publication figures with Arial: sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv"
    )


class SubplotsWrapper:
    """
    A wrapper class monitors data plotted using the ax methods from matplotlib.pyplot.
    This data can be converted into a CSV file formatted for SigmaPlot compatibility.

    Supports optional figrecipe integration for reproducible figures.
    When figrecipe is available and `use_figrecipe=True`, figures are created
    with recipe recording capability for later reproduction.
    """

    def __init__(self):
        self._subplots_wrapper_history = OrderedDict()
        self._fig_scitex = None
        self._counter_part = plt.subplots
        self._figrecipe_available = None  # Lazy check

    def _check_figrecipe(self):
        """Check if figrecipe is available (lazy, cached)."""
        if self._figrecipe_available is None:
            try:
                import figrecipe  # noqa: F401

                self._figrecipe_available = True
            except ImportError:
                self._figrecipe_available = False
        return self._figrecipe_available

    def __call__(
        self,
        *args,
        track=True,
        sharex=False,
        sharey=False,
        constrained_layout=None,
        use_figrecipe=None,  # NEW: Enable figrecipe recording
        # MM-control parameters (unified style system)
        axes_width_mm=None,
        axes_height_mm=None,
        margin_left_mm=None,
        margin_right_mm=None,
        margin_bottom_mm=None,
        margin_top_mm=None,
        space_w_mm=None,
        space_h_mm=None,
        axes_thickness_mm=None,
        tick_length_mm=None,
        tick_thickness_mm=None,
        trace_thickness_mm=None,
        marker_size_mm=None,
        axis_font_size_pt=None,
        tick_font_size_pt=None,
        title_font_size_pt=None,
        legend_font_size_pt=None,
        suptitle_font_size_pt=None,
        n_ticks=None,
        mode=None,
        dpi=None,
        styles=None,
        transparent=None,
        theme=None,
        **kwargs,
    ):
        """
        Create figure and axes with optional millimeter-based control.

        Parameters
        ----------
        *args : int
            nrows, ncols passed to matplotlib.pyplot.subplots
        track : bool, optional
            Track plotting operations for CSV export (default: True)
        use_figrecipe : bool or None, optional
            If True, use figrecipe for recipe recording.
            If None (default), auto-detect figrecipe availability.
            If False, disable figrecipe even if available.

        MM-Control Parameters
        ---------------------
        axes_width_mm, axes_height_mm : float or list
            Axes dimensions in mm
        margin_*_mm : float
            Figure margins in mm
        space_w_mm, space_h_mm : float
            Spacing between axes in mm
        mode : str
            'publication' or 'display'

        Returns
        -------
        fig : FigWrapper
            Wrapped matplotlib Figure (with optional RecordingFigure)
        ax or axes : AxisWrapper or AxesWrapper
            Wrapped matplotlib Axes
        """
        # Resolve style values
        from scitex.plt.styles import SCITEX_STYLE as _S
        from scitex.plt.styles import resolve_style_value as _resolve

        axes_width_mm = _resolve(
            "axes.width_mm", axes_width_mm, _S.get("axes_width_mm")
        )
        axes_height_mm = _resolve(
            "axes.height_mm", axes_height_mm, _S.get("axes_height_mm")
        )
        margin_left_mm = _resolve(
            "margins.left_mm", margin_left_mm, _S.get("margin_left_mm")
        )
        margin_right_mm = _resolve(
            "margins.right_mm", margin_right_mm, _S.get("margin_right_mm")
        )
        margin_bottom_mm = _resolve(
            "margins.bottom_mm", margin_bottom_mm, _S.get("margin_bottom_mm")
        )
        margin_top_mm = _resolve(
            "margins.top_mm", margin_top_mm, _S.get("margin_top_mm")
        )
        space_w_mm = _resolve("spacing.horizontal_mm", space_w_mm, _S.get("space_w_mm"))
        space_h_mm = _resolve("spacing.vertical_mm", space_h_mm, _S.get("space_h_mm"))
        axes_thickness_mm = _resolve(
            "axes.thickness_mm", axes_thickness_mm, _S.get("axes_thickness_mm")
        )
        tick_length_mm = _resolve(
            "ticks.length_mm", tick_length_mm, _S.get("tick_length_mm")
        )
        tick_thickness_mm = _resolve(
            "ticks.thickness_mm", tick_thickness_mm, _S.get("tick_thickness_mm")
        )
        trace_thickness_mm = _resolve(
            "lines.trace_mm", trace_thickness_mm, _S.get("trace_thickness_mm")
        )
        marker_size_mm = _resolve(
            "markers.size_mm", marker_size_mm, _S.get("marker_size_mm")
        )
        axis_font_size_pt = _resolve(
            "fonts.axis_label_pt", axis_font_size_pt, _S.get("axis_font_size_pt")
        )
        tick_font_size_pt = _resolve(
            "fonts.tick_label_pt", tick_font_size_pt, _S.get("tick_font_size_pt")
        )
        title_font_size_pt = _resolve(
            "fonts.title_pt", title_font_size_pt, _S.get("title_font_size_pt")
        )
        legend_font_size_pt = _resolve(
            "fonts.legend_pt", legend_font_size_pt, _S.get("legend_font_size_pt")
        )
        suptitle_font_size_pt = _resolve(
            "fonts.suptitle_pt", suptitle_font_size_pt, _S.get("suptitle_font_size_pt")
        )
        n_ticks = _resolve("ticks.n_ticks", n_ticks, _S.get("n_ticks"), int)
        dpi = _resolve("output.dpi", dpi, _S.get("dpi"), int)

        if transparent is None:
            transparent = _S.get("transparent", True)
        if mode is None:
            mode = _S.get("mode", "publication")
        if theme is None:
            theme = _resolve("theme.mode", None, "light", str)

        # Determine figrecipe usage
        if use_figrecipe is None:
            use_figrecipe = self._check_figrecipe()

        # Create figure with mm-control
        fig, axes = create_with_mm_control(
            *args,
            track=track,
            sharex=sharex,
            sharey=sharey,
            axes_width_mm=axes_width_mm,
            axes_height_mm=axes_height_mm,
            margin_left_mm=margin_left_mm,
            margin_right_mm=margin_right_mm,
            margin_bottom_mm=margin_bottom_mm,
            margin_top_mm=margin_top_mm,
            space_w_mm=space_w_mm,
            space_h_mm=space_h_mm,
            axes_thickness_mm=axes_thickness_mm,
            tick_length_mm=tick_length_mm,
            tick_thickness_mm=tick_thickness_mm,
            trace_thickness_mm=trace_thickness_mm,
            marker_size_mm=marker_size_mm,
            axis_font_size_pt=axis_font_size_pt,
            tick_font_size_pt=tick_font_size_pt,
            title_font_size_pt=title_font_size_pt,
            legend_font_size_pt=legend_font_size_pt,
            suptitle_font_size_pt=suptitle_font_size_pt,
            n_ticks=n_ticks,
            mode=mode,
            dpi=dpi,
            styles=styles,
            transparent=transparent,
            theme=theme,
            **kwargs,
        )

        # If figrecipe enabled, create recording layer
        if use_figrecipe:
            self._attach_figrecipe_recorder(fig)

        self._fig_scitex = fig
        return fig, axes

    def _attach_figrecipe_recorder(self, fig_wrapper):
        """Attach figrecipe recorder to FigWrapper for recipe export.

        This creates a RecordingFigure layer that wraps the underlying
        matplotlib figure, enabling save_recipe() on the FigWrapper.
        """
        try:
            from figrecipe._recorder import Recorder

            # Get the underlying matplotlib figure
            mpl_fig = fig_wrapper._fig_mpl

            # Create recorder
            recorder = Recorder()
            figsize = mpl_fig.get_size_inches()
            dpi_val = mpl_fig.dpi
            recorder.start_figure(figsize=tuple(figsize), dpi=int(dpi_val))

            # Store recorder on FigWrapper for later recipe export
            fig_wrapper._figrecipe_recorder = recorder
            fig_wrapper._figrecipe_enabled = True

            # Store style info from scitex in the recipe
            if hasattr(mpl_fig, "_scitex_theme"):
                recorder.figure_record.style = {"theme": mpl_fig._scitex_theme}

        except Exception:
            # Silently fail - figrecipe is optional
            fig_wrapper._figrecipe_enabled = False

    def __dir__(self):
        """Provide combined directory for tab completion."""
        local_attrs = set(super().__dir__())
        try:
            counterpart_attrs = set(dir(self._counter_part))
        except Exception:
            counterpart_attrs = set()
        return sorted(local_attrs.union(counterpart_attrs))


# Instantiate the wrapper
subplots = SubplotsWrapper()


if __name__ == "__main__":
    import matplotlib

    import scitex

    matplotlib.use("TkAgg")

    fig, ax = subplots()
    ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
    scitex.io.save(fig, "/tmp/subplots_demo/plots.png")

    print(ax.export_as_csv())

# EOF
