#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 13:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_collect_figure_metadata.py

"""
Collect metadata from matplotlib figures for embedding in saved images.

This module provides utilities to automatically extract dimension, styling,
and configuration information from matplotlib figures and axes, making saved
figures self-documenting and reproducible.
"""

__FILE__ = __file__

from typing import Dict, Optional


def collect_figure_metadata(fig, ax=None, plot_id=None) -> Dict:
    """
    Collect all metadata from figure and axes for embedding in saved images.

    This function automatically extracts:
    - Software versions (scitex, matplotlib)
    - Timestamp
    - Figure/axes dimensions (mm, inch, px)
    - DPI settings
    - Margins
    - Styling parameters (if available)
    - Mode (display/publication)
    - Creation method
    - Plot type and axes information (Phase 1)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to collect metadata from
    ax : matplotlib.axes.Axes, optional
        Primary axes to collect dimension info from.
        If not provided, uses first axes in figure.
    plot_id : str, optional
        Identifier for this plot (e.g., "01_plot"). If not provided,
        will be extracted from filename if available.

    Returns
    -------
    dict
        Complete metadata dictionary ready for embedding via scitex.io.embed_metadata()

    Examples
    --------
    >>> from scitex.plt.utils import create_axes_with_size_mm, collect_figure_metadata
    >>> fig, ax = create_axes_with_size_mm(30, 21, mode='publication')
    >>> ax.plot(x, y)
    >>> metadata = collect_figure_metadata(fig, ax)
    >>> print(metadata['dimensions']['axes_size_mm'])
    (30.0, 21.0)

    Notes
    -----
    This function is automatically called by FigWrapper.savefig() when
    embed_metadata=True (the default). You typically don't need to call it manually.

    The collected metadata enables:
    - Reproducing exact figure dimensions later
    - Matching styling across multiple figures
    - Documenting figure provenance
    - Debugging dimension/DPI issues
    """
    import datetime

    import matplotlib
    import scitex

    # Base metadata
    metadata = {
        "metadata_version": "1.1.0",  # Version of the metadata schema itself (updated for Phase 1)
        "scitex": {
            "version": scitex.__version__,
            "created_at": datetime.datetime.now().isoformat(),
        },
        "matplotlib": {
            "version": matplotlib.__version__,
        },
    }

    # Add plot ID if provided
    if plot_id:
        metadata["id"] = plot_id

    # If no axes provided, try to get first axes from figure
    if ax is None and hasattr(fig, "axes") and len(fig.axes) > 0:
        ax = fig.axes[0]

    # Add dimension info if axes available
    if ax is not None:
        try:
            from ._figure_from_axes_mm import get_dimension_info

            dim_info = get_dimension_info(fig, ax)

            metadata["dimensions"] = {
                "figure_size_mm": dim_info["figure_size_mm"],
                "figure_size_inch": dim_info["figure_size_inch"],
                "figure_size_px": dim_info["figure_size_px"],
                "axes_size_mm": dim_info["axes_size_mm"],
                "axes_size_inch": dim_info["axes_size_inch"],
                "axes_size_px": dim_info["axes_size_px"],
                "axes_position": dim_info["axes_position"],
                "dpi": dim_info["dpi"],
            }

            # Calculate margins from dimension info
            fig_w_mm, fig_h_mm = dim_info["figure_size_mm"]
            axes_w_mm, axes_h_mm = dim_info["axes_size_mm"]
            axes_pos = dim_info["axes_position"]
            fig_w_px, fig_h_px = dim_info["figure_size_px"]
            axes_w_px, axes_h_px = dim_info["axes_size_px"]
            dpi = dim_info["dpi"]

            metadata["margins_mm"] = {
                "left": axes_pos[0] * fig_w_mm,
                "bottom": axes_pos[1] * fig_h_mm,
                "right": fig_w_mm - (axes_pos[0] * fig_w_mm + axes_w_mm),
                "top": fig_h_mm - (axes_pos[1] * fig_h_mm + axes_h_mm),
            }

            # Calculate axes bounding box in pixels and millimeters
            # axes_position is (left, bottom, width, height) in figure coordinates (0-1)
            # Convert to absolute coordinates
            x0_px = int(axes_pos[0] * fig_w_px)
            y0_px = int((1 - axes_pos[1] - axes_pos[3]) * fig_h_px)  # Flip Y (matplotlib origin is bottom-left)
            x1_px = x0_px + axes_w_px
            y1_px = y0_px + axes_h_px

            x0_mm = axes_pos[0] * fig_w_mm
            y0_mm = (1 - axes_pos[1] - axes_pos[3]) * fig_h_mm  # Flip Y
            x1_mm = x0_mm + axes_w_mm
            y1_mm = y0_mm + axes_h_mm

            metadata["axes_bbox_px"] = {
                "x0": x0_px,
                "y0": y0_px,
                "x1": x1_px,
                "y1": y1_px,
                "width": axes_w_px,
                "height": axes_h_px,
            }

            metadata["axes_bbox_mm"] = {
                "x0": x0_mm,
                "y0": y0_mm,
                "x1": x1_mm,
                "y1": y1_mm,
                "width": axes_w_mm,
                "height": axes_h_mm,
            }

        except Exception as e:
            # If dimension extraction fails, continue without it
            import warnings

            warnings.warn(
                f"Could not extract dimension info for metadata: {e}"
            )

    # Add scitex-specific metadata if axes was tagged
    if ax is not None and hasattr(ax, "_scitex_metadata"):
        scitex_meta = ax._scitex_metadata

        # Extract stats separately for top-level access
        if 'stats' in scitex_meta:
            metadata['stats'] = scitex_meta['stats']

        # Merge into scitex section
        for key, value in scitex_meta.items():
            if key not in metadata["scitex"] and key != 'stats':  # Don't duplicate stats
                metadata["scitex"][key] = value

    # Alternative: check figure for metadata (for multi-axes cases)
    elif hasattr(fig, "_scitex_metadata"):
        scitex_meta = fig._scitex_metadata

        # Extract stats separately for top-level access
        if 'stats' in scitex_meta:
            metadata['stats'] = scitex_meta['stats']

        for key, value in scitex_meta.items():
            if key not in metadata["scitex"] and key != 'stats':  # Don't duplicate stats
                metadata["scitex"][key] = value

    # Add actual font information
    try:
        from ._get_actual_font import get_actual_font_name
        actual_font = get_actual_font_name()

        # Store both requested and actual font
        if "style_mm" in metadata.get("scitex", {}):
            requested_font = metadata["scitex"]["style_mm"].get("font_family", "Arial")
            metadata["scitex"]["style_mm"]["font_family_requested"] = requested_font
            metadata["scitex"]["style_mm"]["font_family_actual"] = actual_font

            # Warn if requested and actual fonts differ
            if requested_font != actual_font:
                try:
                    from scitex.logging import getLogger
                    logger = getLogger(__name__)
                    logger.warning(
                        f"Font mismatch: Requested '{requested_font}' but using '{actual_font}'. "
                        f"For {requested_font}: sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv"
                    )
                except ImportError:
                    # Fallback to warnings if scitex.logging not available
                    import warnings
                    warnings.warn(
                        f"Font mismatch: Requested '{requested_font}' but using '{actual_font}'",
                        UserWarning
                    )
        else:
            # If no style_mm, add font info to scitex section
            if "scitex" in metadata:
                metadata["scitex"]["font_family_actual"] = actual_font
    except Exception:
        # If font detection fails, continue without it
        pass

    # Phase 1: Add plot_type, axes, and style_preset
    if ax is not None:
        try:
            # Extract axes labels and units
            axes_info = {}

            # X-axis
            xlabel = ax.get_xlabel()
            x_label, x_unit = _parse_label_unit(xlabel)
            axes_info["x"] = {
                "label": x_label,
                "unit": x_unit,
                "scale": ax.get_xscale(),
                "lim": list(ax.get_xlim()),
            }

            # Y-axis
            ylabel = ax.get_ylabel()
            y_label, y_unit = _parse_label_unit(ylabel)
            axes_info["y"] = {
                "label": y_label,
                "unit": y_unit,
                "scale": ax.get_yscale(),
                "lim": list(ax.get_ylim()),
            }

            # Add n_ticks if available from style
            if "scitex" in metadata and "style_mm" in metadata["scitex"]:
                if "n_ticks" in metadata["scitex"]["style_mm"]:
                    n_ticks = metadata["scitex"]["style_mm"]["n_ticks"]
                    axes_info["x"]["n_ticks"] = n_ticks
                    axes_info["y"]["n_ticks"] = n_ticks

            metadata["axes"] = axes_info

            # Extract title
            title = ax.get_title()
            if title:
                metadata["title"] = title

            # Detect plot type and method from axes history or lines
            plot_type, method = _detect_plot_type(ax)
            if plot_type:
                metadata["plot_type"] = plot_type
            if method:
                metadata["method"] = method

            # Extract style preset if available
            if hasattr(ax, "_scitex_metadata") and "style_preset" in ax._scitex_metadata:
                metadata["style_preset"] = ax._scitex_metadata["style_preset"]
            elif hasattr(fig, "_scitex_metadata") and "style_preset" in fig._scitex_metadata:
                metadata["style_preset"] = fig._scitex_metadata["style_preset"]

            # Phase 2: Extract traces (lines) with their properties and CSV column mapping
            traces = _extract_traces(ax)
            if traces:
                metadata["traces"] = traces

            # Phase 2: Extract legend info
            legend_info = _extract_legend_info(ax)
            if legend_info:
                metadata["legend"] = legend_info

        except Exception as e:
            # If Phase 1 extraction fails, continue without it
            import warnings
            warnings.warn(f"Could not extract Phase 1 metadata: {e}")

    return metadata


def _parse_label_unit(label_text: str) -> tuple:
    """
    Parse label text to extract label and unit.

    Handles formats like:
    - "Time [s]" -> ("Time", "s")
    - "Amplitude (a.u.)" -> ("Amplitude", "a.u.")
    - "Value" -> ("Value", "")

    Parameters
    ----------
    label_text : str
        The full label text from axes

    Returns
    -------
    tuple
        (label, unit) where unit is empty string if not found
    """
    import re

    if not label_text:
        return "", ""

    # Try to match [...] pattern first (preferred format)
    match = re.match(r'^(.+?)\s*\[([^\]]+)\]$', label_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Try to match (...) pattern
    match = re.match(r'^(.+?)\s*\(([^\)]+)\)$', label_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # No unit found
    return label_text.strip(), ""


def _extract_traces(ax) -> list:
    """
    Extract trace (line) information including properties and CSV column mapping.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to extract traces from

    Returns
    -------
    list
        List of trace dictionaries with id, label, color, linestyle, linewidth,
        and csv_columns mapping
    """
    import matplotlib.colors as mcolors

    traces = []

    # Get axes position for CSV column naming
    ax_pos = "00"  # Default for single axes
    if hasattr(ax, '_scitex_metadata') and 'position_in_grid' in ax._scitex_metadata:
        pos = ax._scitex_metadata['position_in_grid']
        ax_pos = f"{pos[0]}{pos[1]}"

    for i, line in enumerate(ax.lines):
        trace = {}

        # Get ID from _scitex_id attribute (set by scitex plotting functions)
        # This matches the id= kwarg passed to ax.plot()
        scitex_id = getattr(line, '_scitex_id', None)

        # Get label for legend
        label = line.get_label()

        # Determine trace_id for CSV column matching
        if scitex_id:
            trace_id = scitex_id
        elif not label.startswith('_'):
            trace_id = label
        else:
            trace_id = f"line_{i}"

        trace["id"] = trace_id

        # Label (for legend) - use label if not internal
        if not label.startswith('_'):
            trace["label"] = label

        # Color - always convert to hex for consistent JSON storage
        color = line.get_color()
        try:
            # mcolors.to_hex handles strings, RGB tuples, RGBA tuples
            color_hex = mcolors.to_hex(color, keep_alpha=False)
            trace["color"] = color_hex
        except (ValueError, TypeError):
            # Fallback: store as-is
            trace["color"] = color

        # Line style
        trace["linestyle"] = line.get_linestyle()

        # Line width
        trace["linewidth"] = line.get_linewidth()

        # Marker
        marker = line.get_marker()
        if marker and marker != 'None':
            trace["marker"] = marker
            trace["markersize"] = line.get_markersize()

        # CSV column mapping - this is how we'll reconstruct from CSV
        # Format matches what _export_as_csv generates: ax_{row}{col}_{id}_plot_x/y
        # The id should match the id= kwarg passed to ax.plot()
        trace["csv_columns"] = {
            "x": f"ax_{ax_pos}_{trace_id}_plot_x",
            "y": f"ax_{ax_pos}_{trace_id}_plot_y",
        }

        traces.append(trace)

    return traces


def _extract_legend_info(ax) -> Optional[dict]:
    """
    Extract legend information from axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to extract legend from

    Returns
    -------
    dict or None
        Legend info dictionary or None if no legend
    """
    legend = ax.get_legend()
    if legend is None:
        return None

    legend_info = {
        "visible": legend.get_visible(),
        "loc": legend._loc if hasattr(legend, '_loc') else "best",
        "frameon": legend.get_frame_on() if hasattr(legend, 'get_frame_on') else True,
    }

    # Extract legend entries (labels)
    texts = legend.get_texts()
    if texts:
        legend_info["labels"] = [t.get_text() for t in texts]

    return legend_info


def _detect_plot_type(ax) -> tuple:
    """
    Detect the primary plot type and method from axes content.

    Checks for:
    - Lines -> "line"
    - Scatter collections -> "scatter"
    - Bar containers -> "bar"
    - Patches (histogram) -> "hist"
    - Box plot -> "boxplot"
    - Violin plot -> "violin"
    - Image -> "image"
    - Contour -> "contour"
    - KDE -> "kde"

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to analyze

    Returns
    -------
    tuple
        (plot_type, method) where method is the actual plotting function used,
        or (None, None) if unclear
    """
    # Check scitex history FIRST (most reliable for scitex plots)
    if hasattr(ax, 'history') and len(ax.history) > 0:
        # Get the first plotting command
        first_cmd = ax.history[0].get('command', '')
        if 'stx_heatmap' in first_cmd:
            return "heatmap", "stx_heatmap"
        elif 'stx_kde' in first_cmd:
            return "kde", "stx_kde"
        elif 'stx_ecdf' in first_cmd:
            return "ecdf", "stx_ecdf"
        elif 'stx_violin' in first_cmd:
            return "violin", "stx_violin"
        elif 'stx_box' in first_cmd or 'boxplot' in first_cmd:
            return "boxplot", "stx_box"
        elif 'stx_line' in first_cmd:
            return "line", "stx_line"
        elif 'plot_scatter' in first_cmd:
            return "scatter", "plot_scatter"
        elif 'stx_mean_std' in first_cmd:
            return "line", "stx_mean_std"
        elif 'stx_shaded_line' in first_cmd:
            return "line", "stx_shaded_line"
        elif 'sns_boxplot' in first_cmd:
            return "boxplot", "sns_boxplot"
        elif 'sns_violinplot' in first_cmd:
            return "violin", "sns_violinplot"
        elif 'sns_scatterplot' in first_cmd:
            return "scatter", "sns_scatterplot"
        elif 'sns_lineplot' in first_cmd:
            return "line", "sns_lineplot"
        elif 'sns_histplot' in first_cmd:
            return "hist", "sns_histplot"
        elif 'sns_barplot' in first_cmd:
            return "bar", "sns_barplot"
        elif 'sns_stripplot' in first_cmd:
            return "scatter", "sns_stripplot"
        elif 'sns_kdeplot' in first_cmd:
            return "kde", "sns_kdeplot"
        elif 'scatter' in first_cmd:
            return "scatter", "scatter"
        elif 'bar' in first_cmd:
            return "bar", "bar"
        elif 'hist' in first_cmd:
            return "hist", "hist"

    # Check for images (takes priority)
    if len(ax.images) > 0:
        return "image", "imshow"

    # Check for contours
    if hasattr(ax, 'collections'):
        for coll in ax.collections:
            if 'Contour' in type(coll).__name__:
                return "contour", "contour"

    # Check for bar plots
    if len(ax.containers) > 0:
        # Check if it's a boxplot (has multiple containers with specific structure)
        if any('boxplot' in str(type(c)).lower() for c in ax.containers):
            return "boxplot", "boxplot"
        # Otherwise assume bar plot
        return "bar", "bar"

    # Check for patches (could be histogram, violin, etc.)
    if len(ax.patches) > 0:
        # If there are many rectangular patches, likely histogram
        if len(ax.patches) > 5:
            return "hist", "hist"
        # Check for violin plot
        if any('Poly' in type(p).__name__ for p in ax.patches):
            return "violin", "violinplot"

    # Check for scatter plots (PathCollection)
    if hasattr(ax, 'collections') and len(ax.collections) > 0:
        for coll in ax.collections:
            if 'PathCollection' in type(coll).__name__:
                return "scatter", "scatter"

    # Check for line plots
    if len(ax.lines) > 0:
        # If there are error bars, it might be errorbar plot
        if any(hasattr(line, '_mpl_error') for line in ax.lines):
            return "errorbar", "errorbar"
        return "line", "plot"

    return None, None


if __name__ == "__main__":
    import numpy as np

    from ._figure_from_axes_mm import create_axes_with_size_mm

    print("=" * 60)
    print("METADATA COLLECTION DEMO")
    print("=" * 60)

    # Create a figure with mm control
    print("\n1. Creating figure with mm control...")
    fig, ax = create_axes_with_size_mm(
        axes_width_mm=30,
        axes_height_mm=21,
        mode="publication",
        style_mm={
            "axis_thickness_mm": 0.2,
            "trace_thickness_mm": 0.12,
            "tick_length_mm": 0.8,
        },
    )

    # Plot something
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), "b-")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    # Collect metadata
    print("\n2. Collecting metadata...")
    metadata = collect_figure_metadata(fig, ax)

    # Display metadata
    print("\n3. Collected metadata:")
    print("-" * 60)
    import json

    print(json.dumps(metadata, indent=2))
    print("-" * 60)

    print("\n✅ Metadata collection complete!")
    print("\nKey fields collected:")
    print(f"  • Software version: {metadata['scitex']['version']}")
    print(f"  • Timestamp: {metadata['scitex']['created_at']}")
    if "dimensions" in metadata:
        print(
            f"  • Axes size: {metadata['dimensions']['axes_size_mm']} mm"
        )
        print(f"  • DPI: {metadata['dimensions']['dpi']}")
    if "scitex" in metadata and "mode" in metadata["scitex"]:
        print(f"  • Mode: {metadata['scitex']['mode']}")
    if "scitex" in metadata and "style_mm" in metadata["scitex"]:
        print("  • Style: Embedded ✓")

# EOF
