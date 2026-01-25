#!/usr/bin/env python3
# Timestamp: "2026-01-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/__init__.py
# ----------------------------------------
"""
SciTeX plt module - Publication-quality plotting via figrecipe.

This module provides a thin wrapper around figrecipe with scitex branding.
Simply importing this module automatically configures matplotlib with
SciTeX publication defaults.

Usage
-----
>>> import scitex.plt as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [1, 4, 9])
>>> plt.save(fig, "figure.png")

Style Management
----------------
>>> plt.load_style("SCITEX")  # Load publication style
>>> plt.STYLE  # Access current style configuration
>>> plt.list_presets()  # Show available presets

The module delegates to figrecipe for:
- Recording and reproducing figures
- Style management (mm-based layouts)
- Figure composition
- Graph visualization

SciTeX-specific features (kept locally):
- AxisWrapper/FigWrapper compatibility
- Color palettes (scitex.plt.color)
- Gallery utilities (scitex.plt.gallery)
"""

import os

# ============================================================================
# Set branding environment variables BEFORE importing figrecipe
# This enables automatic docstring replacement: figrecipe -> scitex.plt, fr -> plt
# ============================================================================
os.environ.setdefault("FIGRECIPE_BRAND", "scitex.plt")
os.environ.setdefault("FIGRECIPE_ALIAS", "plt")

# ============================================================================
# Now import figrecipe (branding will be applied)
# ============================================================================
try:
    import figrecipe as _fr

    _FIGRECIPE_AVAILABLE = True
except ImportError:
    _FIGRECIPE_AVAILABLE = False
    _fr = None

# Standard library and matplotlib imports
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as _plt

from scitex import logging as _logging

_logger = _logging.getLogger(__name__)

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

# ============================================================================
# Re-export figrecipe public API with scitex branding
# ============================================================================
if _FIGRECIPE_AVAILABLE:
    # Core public API
    from figrecipe import __version__ as _figrecipe_version
    from figrecipe import (
        compose,
        crop,
        edit,
        extract_data,
        info,
        list_presets,
        load_style,
        reproduce,
        save,
        subplots,
        unload_style,
        validate,
    )

    # Internal imports (not part of figrecipe public API)
    from figrecipe._api._notebook import enable_svg
    from figrecipe._api._seaborn_proxy import sns
    from figrecipe._api._style_manager import STYLE, apply_style
    from figrecipe._composition import align_panels, distribute_panels, smart_align
    from figrecipe._graph_presets import get_preset as get_graph_preset
    from figrecipe._graph_presets import list_presets as list_graph_presets
    from figrecipe._graph_presets import register_preset as register_graph_preset

    # Also export load as alias for reproduce
    load = reproduce
else:
    # Provide stub versions when figrecipe is not available
    _figrecipe_version = "0.0.0"

    def _not_available(*args, **kwargs):
        raise ImportError(
            "figrecipe is required for this feature. Install with: pip install figrecipe"
        )

    STYLE = None
    load_style = _not_available
    unload_style = _not_available
    list_presets = _not_available
    apply_style = _not_available
    subplots = _not_available
    save = _not_available
    reproduce = _not_available
    load = _not_available
    crop = _not_available
    validate = _not_available
    extract_data = _not_available
    info = _not_available
    edit = _not_available
    compose = _not_available
    align_panels = _not_available
    distribute_panels = _not_available
    smart_align = _not_available
    sns = None
    enable_svg = _not_available
    get_graph_preset = _not_available
    list_graph_presets = _not_available
    register_graph_preset = _not_available

# ============================================================================
# Local scitex submodules (kept for compatibility)
# ============================================================================
try:
    from ._tpl import termplot
except ImportError:
    termplot = None

# Backward compatibility: expose styles submodule (deprecated, use figrecipe)
from . import ax, color, gallery, styles, utils

# Import draw_graph from figrecipe integration (handles AxisWrapper)
from ._figrecipe_integration import draw_graph
from .styles import presets

# ============================================================================
# Auto-configure matplotlib with SciTeX defaults on import
# ============================================================================


def _register_arial_fonts():
    """Register Arial fonts if available."""
    try:
        fm.findfont("Arial", fallback_to_default=False)
        return True
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
                return True
            except Exception:
                pass
        return False


def _auto_configure_mpl():
    """Apply SciTeX style configuration automatically on import."""
    # Try to use figrecipe's style system first
    if _FIGRECIPE_AVAILABLE:
        try:
            # Load SCITEX style preset from figrecipe
            load_style("SCITEX")
            return
        except Exception:
            pass

    # Fallback: use local style loader
    from .styles import resolve_style_value

    # mm to pt conversion factor
    mm_to_pt = 2.83465

    # Load all style values from YAML (with env override support)
    font_size = resolve_style_value("fonts.axis_label_pt", None, 7)
    title_size = resolve_style_value("fonts.title_pt", None, 8)
    tick_size = resolve_style_value("fonts.tick_label_pt", None, 7)
    legend_size = resolve_style_value("fonts.legend_pt", None, 6)

    trace_mm = resolve_style_value("lines.trace_mm", None, 0.2)
    line_width = trace_mm * mm_to_pt

    axes_thickness_mm = resolve_style_value("axes.thickness_mm", None, 0.2)
    axes_linewidth = axes_thickness_mm * mm_to_pt

    hide_top = resolve_style_value("behavior.hide_top_spine", None, True, bool)
    hide_right = resolve_style_value("behavior.hide_right_spine", None, True, bool)

    dpi = int(resolve_style_value("output.dpi", None, 300))

    # Calculate figure size from axes + margins
    axes_w = resolve_style_value("axes.width_mm", None, 40)
    axes_h = resolve_style_value("axes.height_mm", None, 28)
    margin_l = resolve_style_value("margins.left_mm", None, 20)
    margin_r = resolve_style_value("margins.right_mm", None, 20)
    margin_b = resolve_style_value("margins.bottom_mm", None, 20)
    margin_t = resolve_style_value("margins.top_mm", None, 20)
    fig_w_mm = axes_w + margin_l + margin_r
    fig_h_mm = axes_h + margin_b + margin_t
    figsize_inch = (fig_w_mm / 25.4, fig_h_mm / 25.4)

    # Apply rcParams
    mpl_config = {
        # Resolution
        "figure.dpi": max(100, dpi // 3),
        "savefig.dpi": dpi,
        # Figure Size
        "figure.figsize": figsize_inch,
        # Font Sizes
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        # Legend
        "legend.fontsize": legend_size,
        "legend.frameon": False,
        "legend.loc": "best",
        # Auto Layout
        "figure.autolayout": True,
        # Spines
        "axes.spines.top": not hide_top,
        "axes.spines.right": not hide_right,
        # Line widths
        "axes.linewidth": axes_linewidth,
        "lines.linewidth": line_width,
        "lines.markersize": 6.0,
        # Grid
        "grid.linewidth": axes_linewidth,
        "grid.alpha": 0.3,
        # Math text
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
    }

    mpl.rcParams.update(mpl_config)


# Register Arial fonts eagerly
_arial_enabled = _register_arial_fonts()

# Configure font family
if _arial_enabled:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
    ]
else:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ]
    # Suppress font warnings
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Apply SciTeX style configuration automatically
_auto_configure_mpl()

# Set up color cycle from scitex colors
try:
    _rgba_norm_cycle = {
        k: tuple(color.update_alpha(v, 1.0))
        for k, v in color.PARAMS.get("RGBA_NORM_FOR_CYCLE", {}).items()
    }
    if _rgba_norm_cycle:
        mpl.rcParams["axes.prop_cycle"] = _plt.cycler(
            color=list(_rgba_norm_cycle.values())
        )
except Exception:
    pass  # Use matplotlib default colors if color module fails


# ============================================================================
# SciTeX-specific wrapper functions (for AxisWrapper/FigWrapper compatibility)
# ============================================================================


def figure(*args, **kwargs):
    """Create a figure that returns a FigWrapper.

    This is the scitex-specific figure function that creates FigWrapper
    objects for compatibility with scitex.plt.ax utilities.

    For figrecipe-style recording figures, use subplots() instead.
    """
    from ._subplots._FigWrapper import FigWrapper

    fig_mpl = _plt.figure(*args, **kwargs)
    return FigWrapper(fig_mpl)


def tight_layout(**kwargs):
    """
    Wrapper for matplotlib.pyplot.tight_layout that handles colorbar layout compatibility.

    This function calls tight_layout on the current figure and gracefully handles:
    1. UserWarning: "The figure layout has changed to tight" - informational only
    2. RuntimeError: Colorbar layout incompatibility - occurs when colorbars exist with old engine

    Parameters
    ----------
    **kwargs
        All keyword arguments are passed to matplotlib.pyplot.tight_layout()
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The figure layout has changed to tight"
        )
        try:
            _plt.tight_layout(**kwargs)
        except RuntimeError as e:
            # Silently handle colorbar layout engine incompatibility
            if "Colorbar layout" not in str(e):
                raise


def colorbar(mappable=None, cax=None, ax=None, **kwargs):
    """
    Create a colorbar, automatically unwrapping SciTeX AxisWrapper objects.

    This function handles both regular matplotlib axes and SciTeX AxisWrapper
    objects transparently, making it a drop-in replacement for plt.colorbar().

    Parameters
    ----------
    mappable : ScalarMappable, optional
        The image, contour set, etc. to which the colorbar applies.
    cax : Axes, optional
        Axes into which the colorbar will be drawn.
    ax : Axes or AxisWrapper or list thereof, optional
        Parent axes from which space for the colorbar will be stolen.
    **kwargs
        Additional keyword arguments passed to matplotlib.pyplot.colorbar()

    Returns
    -------
    Colorbar
        The created colorbar object
    """
    # Unwrap ax if it's a SciTeX AxisWrapper
    if ax is not None:
        if hasattr(ax, "__iter__") and not isinstance(ax, str):
            # Handle list/array of axes
            ax = [a._axis_mpl if hasattr(a, "_axis_mpl") else a for a in ax]
        else:
            # Single axis
            ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax

    # Unwrap cax if provided
    if cax is not None:
        cax = cax._axis_mpl if hasattr(cax, "_axis_mpl") else cax

    # Call matplotlib's colorbar with unwrapped axes
    return _plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)


def close(fig=None):
    """
    Close a figure, automatically unwrapping SciTeX FigWrapper objects.

    This function is a drop-in replacement for matplotlib.pyplot.close() that
    handles both regular matplotlib Figure objects and SciTeX FigWrapper objects.

    Parameters
    ----------
    fig : Figure, FigWrapper, int, str, or None
        The figure to close. Can be:
        - None: close the current figure
        - Figure or FigWrapper: close the specified figure
        - int: close figure with that number
        - str: close figure with that label, or 'all' to close all figures
    """
    if fig is None:
        _plt.close()
    elif isinstance(fig, (int, str)):
        _plt.close(fig)
    elif hasattr(fig, "_fig_mpl"):
        # FigWrapper object - unwrap and close
        _plt.close(fig._fig_mpl)
    elif hasattr(fig, "figure"):
        # Alternative attribute name (backward compatibility)
        _plt.close(fig.figure)
    elif hasattr(fig, "fig"):
        # figrecipe RecordingFigure - unwrap and close
        _plt.close(fig.fig)
    else:
        # Assume it's a matplotlib Figure
        _plt.close(fig)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Figrecipe core (re-exported with branding)
    "subplots",
    "save",
    "reproduce",
    "load",  # Alias for reproduce
    "crop",
    "validate",
    "extract_data",
    "info",
    "edit",
    # Style management
    "STYLE",
    "load_style",
    "unload_style",
    "list_presets",
    "apply_style",
    # Composition
    "compose",
    "align_panels",
    "distribute_panels",
    "smart_align",
    # Graph visualization
    "draw_graph",
    "get_graph_preset",
    "list_graph_presets",
    "register_graph_preset",
    # Extensions
    "sns",
    "enable_svg",
    # SciTeX-specific wrappers
    "figure",
    "colorbar",
    "close",
    "tight_layout",
    # Local submodules
    "ax",
    "color",
    "gallery",
    "utils",
    "styles",
    "presets",
    "termplot",
]


def __getattr__(name):
    """
    Fallback to matplotlib.pyplot for any missing attributes.
    This makes scitex.plt a complete drop-in replacement for matplotlib.pyplot.
    """
    if hasattr(_plt, name):
        return getattr(_plt, name)
    raise AttributeError(f"module 'scitex.plt' has no attribute '{name}'")


def __dir__():
    """
    Provide comprehensive directory listing including matplotlib.pyplot functions.
    """
    local_attrs = list(__all__)
    # Add matplotlib.pyplot attributes
    mpl_attrs = [attr for attr in dir(_plt) if not attr.startswith("_")]
    local_attrs.extend(mpl_attrs)
    return sorted(set(local_attrs))


# EOF
