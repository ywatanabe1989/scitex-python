#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-02 12:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/__init__.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""
SciTeX plt module - Publication-quality plotting.

Simply importing this module automatically configures matplotlib with
SciTeX publication defaults from SCITEX_STYLE.yaml:

    import scitex.plt as splt
    fig, ax = splt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    fig.savefig("figure.png")

No need to call scitex.session() or configure_mpl() - it's automatic.

Style values can be customized by:
1. Editing SCITEX_STYLE.yaml
2. Setting environment variables (SCITEX_PLT_FONTS_AXIS_LABEL_PT=8)
3. Passing parameters directly to subplots()
"""

import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt

from scitex import logging as _logging

_logger = _logging.getLogger(__name__)

# =============================================================================
# Auto-configure matplotlib with SciTeX style on import
# =============================================================================


def _auto_configure_mpl():
    """Apply SciTeX style configuration automatically on import."""
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


# Register Arial fonts eagerly (before style configuration)
_arial_enabled = False
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
    from . import color as _color_module

    _rgba_norm_cycle = {
        k: tuple(_color_module.update_alpha(v, 1.0))
        for k, v in _color_module.PARAMS.get("RGBA_NORM_FOR_CYCLE", {}).items()
    }
    if _rgba_norm_cycle:
        mpl.rcParams["axes.prop_cycle"] = plt.cycler(
            color=list(_rgba_norm_cycle.values())
        )
except Exception:
    pass  # Use matplotlib default colors if color module fails

from ._tpl import termplot
from . import color
from . import utils
from . import ax
from .styles import presets
from . import styles
from . import gallery

# Lazy import for subplots to avoid circular dependencies
# Note: Use names that don't conflict with submodule names like _subplots
_subplots_func_cached = None
_figure_func_cached = None
_crop_func_cached = None


def subplots(*args, **kwargs):
    """Lazy-loaded subplots function."""
    global _subplots_func_cached
    if _subplots_func_cached is None:
        from ._subplots._SubplotsWrapper import subplots as _subplots_impl

        _subplots_func_cached = _subplots_impl
    return _subplots_func_cached(*args, **kwargs)


def figure(*args, **kwargs):
    """Lazy-loaded figure function that returns a FigWrapper."""
    global _figure_func_cached
    if _figure_func_cached is None:
        import matplotlib.pyplot as plt
        from ._subplots._FigWrapper import FigWrapper

        def _figure_impl(*args, **kwargs):
            fig_mpl = plt.figure(*args, **kwargs)
            return FigWrapper(fig_mpl)

        _figure_func_cached = _figure_impl
    return _figure_func_cached(*args, **kwargs)


def crop(input_path, output_path=None, margin=12, overwrite=False, verbose=False):
    """
    Auto-crop a figure to its content area.

    This function automatically detects the content area of a saved figure
    and crops it, removing excess whitespace. Designed for publication figures
    created with large margins.

    Parameters
    ----------
    input_path : str
        Path to the input image
    output_path : str, optional
        Path to save cropped image. If None and overwrite=True, overwrites input.
        If None and overwrite=False, adds '_cropped' suffix.
    margin : int, optional
        Margin in pixels around content (default: 12, ~1mm at 300 DPI)
    overwrite : bool, optional
        Overwrite input file (default: False)
    verbose : bool, optional
        Print detailed information (default: False)

    Returns
    -------
    str
        Path to the saved cropped image

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> stx.io.save(fig, "figure.png")
    >>> stx.plt.crop("figure.png", "figure_cropped.png")  # 1mm margin
    """
    global _crop_func_cached
    if _crop_func_cached is None:
        from .utils._crop import crop as _crop_impl

        _crop_func_cached = _crop_impl
    return _crop_func_cached(input_path, output_path, margin, overwrite, verbose)


def load(path, apply_manual=True):
    """
    Load a figure from saved JSON + CSV files.

    Parameters
    ----------
    path : str or Path
        Path to JSON file, PNG file, or CSV file.
        Will auto-detect sibling files in same directory or organized subdirectories.
    apply_manual : bool, optional
        If True, apply .manual.json overrides if exists (default: True)

    Returns
    -------
    tuple
        (fig, axes) where fig is FigWrapper and axes is AxisWrapper or array

    Raises
    ------
    FileNotFoundError
        If required JSON file is not found
    ValueError
        If manual.json hash doesn't match (stale manual edits)

    Examples
    --------
    >>> # Load from JSON (sibling pattern)
    >>> fig, axes = stx.plt.load("output/figure.json")

    >>> # Load from PNG (finds sibling JSON + CSV)
    >>> fig, axes = stx.plt.load("output/figure.png")

    >>> # Load from organized directory pattern
    >>> fig, axes = stx.plt.load("output/json/figure.json")

    >>> # Skip manual overrides
    >>> fig, axes = stx.plt.load("figure.json", apply_manual=False)

    Notes
    -----
    Supports two directory patterns:

    Pattern 1 (flat/sibling):
        output/figure.png
        output/figure.json
        output/figure.csv

    Pattern 2 (organized):
        output/png/figure.png
        output/json/figure.json
        output/csv/figure.csv

    Manual overrides (.manual.json) are applied if:
    - apply_manual=True
    - figure.manual.json exists alongside figure.json
    - Hash validation passes (warns if stale)
    """
    from pathlib import Path
    import hashlib
    import scitex as stx

    path = Path(path)

    # Resolve JSON path from any input (png, csv, or json)
    json_path, csv_path = _resolve_figure_paths(path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load JSON metadata
    metadata = stx.io.load(json_path)

    # Load CSV data if exists
    csv_data = None
    if csv_path and csv_path.exists():
        csv_data = stx.io.load(csv_path)

    # Check for manual overrides
    manual_path = json_path.with_suffix(".manual.json")
    manual_overrides = None
    if apply_manual and manual_path.exists():
        manual_data = stx.io.load(manual_path)

        # Validate hash
        if "base_hash" in manual_data:
            current_hash = _compute_file_hash(json_path)
            if manual_data["base_hash"] != current_hash:
                _logger.warning(
                    f"Manual overrides may be stale: base data changed since manual edits.\n"
                    f"  Expected hash: {manual_data['base_hash'][:16]}...\n"
                    f"  Current hash:  {current_hash[:16]}...\n"
                    f"  Review: {manual_path}"
                )
        manual_overrides = manual_data.get("overrides", {})

    # Reconstruct figure
    fig, axes = _reconstruct_figure(metadata, csv_data, manual_overrides)

    return fig, axes


def _resolve_figure_paths(path):
    """
    Resolve JSON and CSV paths from any input file path.

    Supports both flat (sibling) and organized (subdirectory) patterns.
    """
    from pathlib import Path

    path = Path(path)
    stem = path.stem
    suffix = path.suffix.lower()
    parent = path.parent

    # Determine base name (remove .manual if present)
    if stem.endswith(".manual"):
        stem = stem[:-7]

    json_path = None
    csv_path = None

    if suffix == ".json":
        json_path = path
        # Try sibling CSV first
        csv_sibling = parent / f"{stem}.csv"
        if csv_sibling.exists():
            csv_path = csv_sibling
        # Try organized pattern (../csv/)
        elif parent.name == "json":
            csv_organized = parent.parent / "csv" / f"{stem}.csv"
            if csv_organized.exists():
                csv_path = csv_organized

    elif suffix in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
        # Look for sibling JSON
        json_sibling = parent / f"{stem}.json"
        csv_sibling = parent / f"{stem}.csv"

        if json_sibling.exists():
            json_path = json_sibling
            if csv_sibling.exists():
                csv_path = csv_sibling
        # Try organized pattern (parent has png/, look for json/)
        elif parent.name in ("png", "jpg", "jpeg", "pdf", "svg"):
            json_organized = parent.parent / "json" / f"{stem}.json"
            csv_organized = parent.parent / "csv" / f"{stem}.csv"
            if json_organized.exists():
                json_path = json_organized
            if csv_organized.exists():
                csv_path = csv_organized

    elif suffix == ".csv":
        csv_path = path
        # Try sibling JSON
        json_sibling = parent / f"{stem}.json"
        if json_sibling.exists():
            json_path = json_sibling
        # Try organized pattern (../json/)
        elif parent.name == "csv":
            json_organized = parent.parent / "json" / f"{stem}.json"
            if json_organized.exists():
                json_path = json_organized

    # Fallback: assume it's the JSON path
    if json_path is None:
        json_path = path if suffix == ".json" else path.with_suffix(".json")

    return json_path, csv_path


def _compute_file_hash(path):
    """Compute SHA256 hash of a file."""
    import hashlib
    from pathlib import Path

    path = Path(path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def _reconstruct_figure(metadata, csv_data, manual_overrides=None):
    """
    Reconstruct figure from metadata and CSV data.

    Parameters
    ----------
    metadata : dict
        JSON metadata from stx.io.save()
    csv_data : DataFrame or None
        CSV data with plot values
    manual_overrides : dict or None
        Manual style/annotation overrides

    Returns
    -------
    tuple
        (fig, axes)
    """
    import numpy as np

    # Extract dimensions from metadata
    dims = metadata.get("dimensions", {})
    fig_size_mm = dims.get("figure_size_mm", [80, 68])
    dpi = dims.get("dpi", 300)

    # Get style from metadata
    scitex_meta = metadata.get("scitex", {})
    style_mm = scitex_meta.get("style_mm", {})

    # Create figure with same dimensions
    fig, axes = subplots(
        axes_width_mm=style_mm.get(
            "axes_width_mm", dims.get("axes_size_mm", [40, 28])[0]
        ),
        axes_height_mm=style_mm.get(
            "axes_height_mm", dims.get("axes_size_mm", [40, 28])[1]
        ),
        dpi=dpi,
    )

    # Handle single vs multiple axes
    ax = axes if not hasattr(axes, "flat") else list(axes.flat)[0]

    # Set axis labels from metadata
    axes_meta = metadata.get("axes", {})
    x_meta = axes_meta.get("x", {})
    y_meta = axes_meta.get("y", {})

    xlabel = x_meta.get("label", "")
    ylabel = y_meta.get("label", "")
    x_unit = x_meta.get("unit", "")
    y_unit = y_meta.get("unit", "")

    if xlabel:
        full_xlabel = f"{xlabel} [{x_unit}]" if x_unit else xlabel
        ax.set_xlabel(full_xlabel)
    if ylabel:
        full_ylabel = f"{ylabel} [{y_unit}]" if y_unit else ylabel
        ax.set_ylabel(full_ylabel)

    # Reconstruct plots from CSV data
    if csv_data is not None and not csv_data.empty:
        _reconstruct_plots_from_csv(ax, csv_data, metadata)

    # Apply manual overrides
    if manual_overrides:
        _apply_manual_overrides(fig, axes, manual_overrides)

    return fig, axes


def _reconstruct_plots_from_csv(ax, csv_data, metadata):
    """
    Reconstruct plot elements from CSV data.

    CSV columns follow pattern: ax_00_<type>_<name>
    """
    import pandas as pd
    import numpy as np

    # Group columns by plot type
    plot_type = metadata.get("plot_type", metadata.get("method", "line"))

    # Parse column names to find plot data
    columns = csv_data.columns.tolist()

    # Find x/y data columns
    x_cols = [c for c in columns if "_x" in c.lower() and "text" not in c.lower()]
    y_cols = [c for c in columns if "_y" in c.lower() and "text" not in c.lower()]

    if plot_type == "line" or plot_type == "plot":
        # For line plots, look for paired x/y or just y with index
        if x_cols and y_cols:
            for x_col, y_col in zip(x_cols, y_cols):
                x = csv_data[x_col].dropna().values
                y = csv_data[y_col].dropna().values
                if len(x) > 0 and len(y) > 0:
                    ax.plot(x[: len(y)], y[: len(x)])
        elif y_cols:
            for y_col in y_cols:
                y = csv_data[y_col].dropna().values
                if len(y) > 0:
                    ax.plot(y)

    elif plot_type == "scatter":
        if x_cols and y_cols:
            x = csv_data[x_cols[0]].dropna().values
            y = csv_data[y_cols[0]].dropna().values
            ax.scatter(x[: min(len(x), len(y))], y[: min(len(x), len(y))])

    # Add text annotations
    text_cols = [c for c in columns if "text" in c.lower() and "content" in c.lower()]
    for text_col in text_cols:
        # Find corresponding x, y columns
        prefix = text_col.rsplit("_content", 1)[0]
        x_col = f"{prefix}_x"
        y_col = f"{prefix}_y"

        if x_col in columns and y_col in columns:
            x_vals = csv_data[x_col].dropna()
            y_vals = csv_data[y_col].dropna()
            text_vals = csv_data[text_col].dropna()

            for i in range(min(len(x_vals), len(y_vals), len(text_vals))):
                ax.text(
                    x_vals.iloc[i],
                    y_vals.iloc[i],
                    str(text_vals.iloc[i]),
                    transform=ax.transAxes,
                    fontsize=6,
                    verticalalignment="top",
                    horizontalalignment="right",
                )


def _apply_manual_overrides(fig, axes, overrides):
    """
    Apply manual style/annotation overrides to figure.

    Parameters
    ----------
    fig : FigWrapper
        Figure to modify
    axes : AxisWrapper or array
        Axes to modify
    overrides : dict
        Override specifications like {"axes[0].style.linewidth": 0.5}
    """
    # Simple override application - can be extended
    for key, value in overrides.items():
        # Parse key like "axes[0].title" or "style.linewidth"
        parts = key.split(".")

        if parts[0].startswith("axes["):
            # Extract axis index
            import re

            match = re.match(r"axes\[(\d+)\]", parts[0])
            if match:
                idx = int(match.group(1))
                ax = axes if not hasattr(axes, "flat") else list(axes.flat)[idx]

                if len(parts) > 1:
                    attr = parts[1]
                    if attr == "title":
                        ax.set_title(value)
                    elif attr == "xlabel":
                        ax.set_xlabel(value)
                    elif attr == "ylabel":
                        ax.set_ylabel(value)


def tight_layout(**kwargs):
    """
    Wrapper for matplotlib.pyplot.tight_layout that handles colorbar layout compatibility.

    This function calls tight_layout on the current figure and gracefully handles:
    1. UserWarning: "The figure layout has changed to tight" - informational only
    2. RuntimeError: Colorbar layout incompatibility - occurs when colorbars exist with old engine

    When a colorbar layout error occurs, the function silently continues as the layout
    is still functional even if the engine cannot be changed.

    Parameters
    ----------
    **kwargs
        All keyword arguments are passed to matplotlib.pyplot.tight_layout()
    """
    import warnings
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The figure layout has changed to tight"
        )
        try:
            plt.tight_layout(**kwargs)
        except RuntimeError as e:
            # Silently handle colorbar layout engine incompatibility
            # This occurs when colorbars were created before tight_layout is called
            # The layout is still usable, so we can safely ignore this error
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
        If None, uses the current image.
    cax : Axes, optional
        Axes into which the colorbar will be drawn.
    ax : Axes or AxisWrapper or list thereof, optional
        Parent axes from which space for the colorbar will be stolen.
        If None, uses current axes.
    **kwargs
        Additional keyword arguments passed to matplotlib.pyplot.colorbar()

    Returns
    -------
    Colorbar
        The created colorbar object
    """
    import matplotlib.pyplot as plt

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
    return plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)


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

    Examples
    --------
    >>> import scitex.plt as splt
    >>> fig, ax = splt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> splt.close(fig)  # Works with FigWrapper

    >>> splt.close('all')  # Close all figures
    >>> splt.close()  # Close current figure

    See Also
    --------
    matplotlib.pyplot.close : Standard matplotlib close function
    """
    import matplotlib.pyplot as plt

    if fig is None:
        # Close current figure
        plt.close()
    elif isinstance(fig, (int, str)):
        # Close by figure number or label (including 'all')
        plt.close(fig)
    elif hasattr(fig, "_fig_mpl"):
        # FigWrapper object - unwrap and close
        plt.close(fig._fig_mpl)
    elif hasattr(fig, "figure"):
        # Alternative attribute name (backward compatibility)
        plt.close(fig.figure)
    else:
        # Assume it's a matplotlib Figure
        plt.close(fig)


__all__ = [
    "close",
    "color",
    "colorbar",
    "figure",
    "gallery",
    "load",
    "presets",
    "subplots",
    "termplot",
    "tight_layout",
    "utils",
]


def __getattr__(name):
    """
    Fallback to matplotlib.pyplot for any missing attributes.
    This makes scitex.plt a complete drop-in replacement for matplotlib.pyplot.
    """
    try:
        import matplotlib.pyplot as plt

        if hasattr(plt, name):
            return getattr(plt, name)
        else:
            raise AttributeError(f"module 'scitex.plt' has no attribute '{name}'")
    except ImportError:
        raise AttributeError(
            f"module 'scitex.plt' has no attribute '{name}' (matplotlib not available)"
        )


def __dir__():
    """
    Provide comprehensive directory listing including matplotlib.pyplot functions.
    """
    # Get local attributes
    local_attrs = __all__.copy()

    # Add matplotlib.pyplot attributes
    try:
        import matplotlib.pyplot as plt

        mpl_attrs = [attr for attr in dir(plt) if not attr.startswith("_")]
        local_attrs.extend(mpl_attrs)
    except ImportError:
        pass

    return sorted(set(local_attrs))


# EOF
