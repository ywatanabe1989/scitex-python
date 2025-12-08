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
            y0_px = int(
                (1 - axes_pos[1] - axes_pos[3]) * fig_h_px
            )  # Flip Y (matplotlib origin is bottom-left)
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

            warnings.warn(f"Could not extract dimension info for metadata: {e}")

    # Add scitex-specific metadata if axes was tagged
    if ax is not None and hasattr(ax, "_scitex_metadata"):
        scitex_meta = ax._scitex_metadata

        # Extract stats separately for top-level access
        if "stats" in scitex_meta:
            metadata["stats"] = scitex_meta["stats"]

        # Merge into scitex section
        for key, value in scitex_meta.items():
            if (
                key not in metadata["scitex"] and key != "stats"
            ):  # Don't duplicate stats
                metadata["scitex"][key] = value

    # Alternative: check figure for metadata (for multi-axes cases)
    elif hasattr(fig, "_scitex_metadata"):
        scitex_meta = fig._scitex_metadata

        # Extract stats separately for top-level access
        if "stats" in scitex_meta:
            metadata["stats"] = scitex_meta["stats"]

        for key, value in scitex_meta.items():
            if (
                key not in metadata["scitex"] and key != "stats"
            ):  # Don't duplicate stats
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
                        UserWarning,
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
            # Try to get scitex AxisWrapper for history access
            # This is needed because matplotlib axes don't have the tracking history
            ax_for_history = ax

            # If ax is a raw matplotlib axes, try to find the scitex wrapper
            if not hasattr(ax, 'history'):
                # Check if ax has a scitex wrapper stored on it
                if hasattr(ax, '_scitex_wrapper'):
                    ax_for_history = ax._scitex_wrapper
                # Check if figure has scitex axes reference
                elif hasattr(fig, 'axes') and hasattr(fig.axes, 'history'):
                    ax_for_history = fig.axes
                # Check for FigWrapper's axes attribute
                elif hasattr(fig, '_fig_scitex') and hasattr(fig._fig_scitex, 'axes'):
                    ax_for_history = fig._fig_scitex.axes
                # Check if the figure object itself has scitex_axes
                elif hasattr(fig, '_scitex_axes'):
                    ax_for_history = fig._scitex_axes

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
            # Use ax_for_history which has the scitex history if available
            plot_type, method = _detect_plot_type(ax_for_history)
            if plot_type:
                metadata["plot_type"] = plot_type
            if method:
                metadata["method"] = method

            # Extract style preset if available
            if (
                hasattr(ax, "_scitex_metadata")
                and "style_preset" in ax._scitex_metadata
            ):
                metadata["style_preset"] = ax._scitex_metadata["style_preset"]
            elif (
                hasattr(fig, "_scitex_metadata")
                and "style_preset" in fig._scitex_metadata
            ):
                metadata["style_preset"] = fig._scitex_metadata["style_preset"]

            # Phase 2: Extract traces (lines) with their properties and CSV column mapping
            traces = _extract_traces(ax)
            if traces:
                metadata["traces"] = traces

            # Phase 2: Extract legend info
            legend_info = _extract_legend_info(ax)
            if legend_info:
                metadata["legend"] = legend_info

            # Phase 3: Extract csv_columns for ALL plot types (from scitex history)
            # This provides a mapping of JSON metadata to CSV columns for reproducibility
            csv_columns_info = _extract_csv_columns_from_history(ax_for_history)
            if csv_columns_info:
                metadata["csv_columns"] = csv_columns_info

            # Phase 4: Compute CSV data hash for reproducibility verification
            csv_hash = _compute_csv_hash(ax_for_history)
            if csv_hash:
                metadata["csv_hash"] = csv_hash

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
    match = re.match(r"^(.+?)\s*\[([^\]]+)\]$", label_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Try to match (...) pattern
    match = re.match(r"^(.+?)\s*\(([^\)]+)\)$", label_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # No unit found
    return label_text.strip(), ""


def _extract_traces(ax) -> list:
    """
    Extract trace (line) information including properties and CSV column mapping.

    Only includes lines that were explicitly created via scitex tracking (top-level calls),
    not internal lines created by matplotlib functions like boxplot() which internally
    call plot() multiple times.

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
    from ._csv_column_naming import get_csv_column_name, sanitize_trace_id

    traces = []

    # Get axes position for CSV column naming
    ax_row, ax_col = 0, 0  # Default for single axes
    if hasattr(ax, "_scitex_metadata") and "position_in_grid" in ax._scitex_metadata:
        pos = ax._scitex_metadata["position_in_grid"]
        ax_row, ax_col = pos[0], pos[1]

    # Get the raw matplotlib axes for accessing lines
    mpl_ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax

    # Try to find scitex wrapper for plot type detection
    ax_for_detection = ax
    if not hasattr(ax, 'history') and hasattr(mpl_ax, '_scitex_wrapper'):
        ax_for_detection = mpl_ax._scitex_wrapper

    # Check if we should filter to only tracked lines
    # For plot types that internally call plot (boxplot, errorbar, etc.),
    # we don't export the internal lines as traces
    plot_type, _ = _detect_plot_type(ax_for_detection)
    internal_plot_types = {"boxplot", "violin", "errorbar", "hist", "bar", "fill", "image", "heatmap", "kde", "ecdf"}

    # If this is a plot type that internally creates lines, skip trace export
    if plot_type in internal_plot_types:
        return []  # No line traces for these plot types

    for i, line in enumerate(mpl_ax.lines):
        trace = {}

        # Get ID from _scitex_id attribute (set by scitex plotting functions)
        # This matches the id= kwarg passed to ax.plot()
        scitex_id = getattr(line, "_scitex_id", None)

        # Get label for legend
        label = line.get_label()

        # Determine trace_id for CSV column matching
        # Use index-based ID to match CSV export (single source of truth)
        trace_id_for_csv = None  # Will use trace_index in get_csv_column_name

        # Store display id/label separately
        if scitex_id:
            trace["id"] = scitex_id
        elif not label.startswith("_"):
            trace["id"] = label
        else:
            trace["id"] = f"line_{i}"

        # Label (for legend) - use label if not internal
        if not label.startswith("_"):
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
        if marker and marker != "None":
            trace["marker"] = marker
            trace["markersize"] = line.get_markersize()

        # CSV column mapping - use single source of truth
        # Uses trace_index to match what _export_as_csv generates
        trace["csv_columns"] = {
            "x": get_csv_column_name("plot_x", ax_row, ax_col, trace_index=i),
            "y": get_csv_column_name("plot_y", ax_row, ax_col, trace_index=i),
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
        "loc": legend._loc if hasattr(legend, "_loc") else "best",
        "frameon": legend.get_frame_on() if hasattr(legend, "get_frame_on") else True,
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
    # History format: dict with keys as IDs and values as tuples (id, method, tracked_dict, kwargs)
    if hasattr(ax, "history") and len(ax.history) > 0:
        # Get all methods from history
        methods = []
        for record in ax.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                methods.append(record[1])  # record[1] is the method name

        # Check methods in priority order (more specific first)
        for method in methods:
            if method == "stx_heatmap":
                return "heatmap", "stx_heatmap"
            elif method == "stx_kde":
                return "kde", "stx_kde"
            elif method == "stx_ecdf":
                return "ecdf", "stx_ecdf"
            elif method == "stx_violin":
                return "violin", "stx_violin"
            elif method in ("stx_box", "boxplot"):
                return "boxplot", method
            elif method == "stx_line":
                return "line", "stx_line"
            elif method == "plot_scatter":
                return "scatter", "plot_scatter"
            elif method == "stx_mean_std":
                return "line", "stx_mean_std"
            elif method == "stx_mean_ci":
                return "line", "stx_mean_ci"
            elif method == "stx_median_iqr":
                return "line", "stx_median_iqr"
            elif method == "stx_shaded_line":
                return "line", "stx_shaded_line"
            elif method == "sns_boxplot":
                return "boxplot", "sns_boxplot"
            elif method == "sns_violinplot":
                return "violin", "sns_violinplot"
            elif method == "sns_scatterplot":
                return "scatter", "sns_scatterplot"
            elif method == "sns_lineplot":
                return "line", "sns_lineplot"
            elif method == "sns_histplot":
                return "hist", "sns_histplot"
            elif method == "sns_barplot":
                return "bar", "sns_barplot"
            elif method == "sns_stripplot":
                return "scatter", "sns_stripplot"
            elif method == "sns_kdeplot":
                return "kde", "sns_kdeplot"
            elif method == "scatter":
                return "scatter", "scatter"
            elif method == "bar":
                return "bar", "bar"
            elif method == "hist":
                return "hist", "hist"
            elif method == "violinplot":
                return "violin", "violinplot"
            elif method == "errorbar":
                return "errorbar", "errorbar"
            elif method == "fill_between":
                return "fill", "fill_between"
            elif method == "imshow":
                return "image", "imshow"
            elif method == "contour":
                return "contour", "contour"
            elif method == "contourf":
                return "contour", "contourf"
            # Note: "plot" method is handled last as a fallback since boxplot uses it internally

    # Check for images (takes priority)
    if len(ax.images) > 0:
        return "image", "imshow"

    # Check for contours
    if hasattr(ax, "collections"):
        for coll in ax.collections:
            if "Contour" in type(coll).__name__:
                return "contour", "contour"

    # Check for bar plots
    if len(ax.containers) > 0:
        # Check if it's a boxplot (has multiple containers with specific structure)
        if any("boxplot" in str(type(c)).lower() for c in ax.containers):
            return "boxplot", "boxplot"
        # Otherwise assume bar plot
        return "bar", "bar"

    # Check for patches (could be histogram, violin, etc.)
    if len(ax.patches) > 0:
        # If there are many rectangular patches, likely histogram
        if len(ax.patches) > 5:
            return "hist", "hist"
        # Check for violin plot
        if any("Poly" in type(p).__name__ for p in ax.patches):
            return "violin", "violinplot"

    # Check for scatter plots (PathCollection)
    if hasattr(ax, "collections") and len(ax.collections) > 0:
        for coll in ax.collections:
            if "PathCollection" in type(coll).__name__:
                return "scatter", "scatter"

    # Check for line plots
    if len(ax.lines) > 0:
        # If there are error bars, it might be errorbar plot
        if any(hasattr(line, "_mpl_error") for line in ax.lines):
            return "errorbar", "errorbar"
        return "line", "plot"

    return None, None


def _extract_csv_columns_from_history(ax) -> list:
    """
    Extract CSV column names from scitex history for all plot types.

    This function generates the exact column names that will be produced
    by export_as_csv(), providing a mapping between JSON metadata and CSV data.

    Parameters
    ----------
    ax : AxisWrapper or matplotlib.axes.Axes
        The axes to extract CSV column info from

    Returns
    -------
    list
        List of dictionaries containing CSV column mappings for each tracked plot,
        e.g., [{"id": "boxplot_0", "method": "boxplot", "columns": ["ax_00_boxplot_0_boxplot_0", "ax_00_boxplot_0_boxplot_1"]}]
    """
    # Get axes index for CSV column naming
    # FigWrapper.export_as_csv() uses flattened index (ii:02d) from _traverse_axes()
    # For single axes figures, this is always 0
    # For multi-axes, we'd need the traversal index
    #
    # Note: position_in_grid stores (row, col) but we need flattened index
    # For single axes (most common case), ax_index = 0
    ax_index = 0

    csv_columns_list = []

    # Check if we have scitex history
    if not hasattr(ax, "history") or len(ax.history) == 0:
        return csv_columns_list

    # Iterate through history to extract column names
    for record_id, record in ax.history.items():
        if not isinstance(record, tuple) or len(record) < 4:
            continue

        id_val, method, tracked_dict, kwargs = record

        # Determine column names based on method type
        columns = _get_csv_columns_for_method(
            id_val, method, tracked_dict, kwargs, ax_index
        )

        if columns:
            csv_columns_list.append({
                "id": id_val,
                "method": method,
                "columns": columns,
            })

    return csv_columns_list


def _compute_csv_hash_from_df(df) -> Optional[str]:
    """
    Compute a hash of CSV data from a DataFrame.

    This is used after actual CSV export to compute the hash from the
    exact data that was written.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to compute hash from

    Returns
    -------
    str or None
        SHA256 hash of the CSV data (first 16 chars), or None if unable to compute
    """
    import hashlib

    try:
        if df is None or df.empty:
            return None

        # Convert to CSV string for hashing
        csv_string = df.to_csv(index=False)

        # Compute SHA256 hash
        hash_obj = hashlib.sha256(csv_string.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        # Return first 16 characters for readability
        return hash_hex[:16]

    except Exception:
        return None


def _compute_csv_hash(ax_or_df) -> Optional[str]:
    """
    Compute a hash of the CSV data for reproducibility verification.

    The hash is computed from the actual data that would be exported to CSV,
    allowing verification that JSON and CSV files are in sync.

    Note: The hash is computed from the AxisWrapper's export_as_csv(), which
    does NOT include the ax_{index:02d}_ prefix. The FigWrapper.export_as_csv()
    adds this prefix. We replicate this prefix addition here.

    Parameters
    ----------
    ax_or_df : AxisWrapper, matplotlib.axes.Axes, or pandas.DataFrame
        The axes to compute CSV hash from, or a pre-exported DataFrame

    Returns
    -------
    str or None
        SHA256 hash of the CSV data (first 16 chars), or None if unable to compute
    """
    import hashlib

    import pandas as pd

    # If it's already a DataFrame, use the direct hash function
    if isinstance(ax_or_df, pd.DataFrame):
        return _compute_csv_hash_from_df(ax_or_df)

    ax = ax_or_df

    # Check if we have scitex history with export capability
    if not hasattr(ax, "export_as_csv"):
        return None

    try:
        # For single axes figures (most common case), ax_index = 0
        ax_index = 0

        # Export the data as CSV from the AxisWrapper
        df = ax.export_as_csv()

        if df is None or df.empty:
            return None

        # Add axis prefix to match what FigWrapper.export_as_csv produces
        # Uses zero-padded index: ax_00_, ax_01_, etc.
        prefix = f"ax_{ax_index:02d}_"
        new_cols = []
        for col in df.columns:
            col_str = str(col)
            if not col_str.startswith(prefix):
                col_str = f"{prefix}{col_str}"
            new_cols.append(col_str)
        df.columns = new_cols

        # Convert to CSV string for hashing
        csv_string = df.to_csv(index=False)

        # Compute SHA256 hash
        hash_obj = hashlib.sha256(csv_string.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        # Return first 16 characters for readability
        return hash_hex[:16]

    except Exception:
        return None


def _get_csv_columns_for_method(id_val, method, tracked_dict, kwargs, ax_index: int) -> list:
    """
    Get CSV column names for a specific plotting method.

    This simulates the actual CSV export to get exact column names.
    It uses the same formatters that generate the CSV to ensure consistency.

    Architecture note:
    - CSV formatters (e.g., _format_boxplot) generate columns WITHOUT ax_ prefix
    - FigWrapper.export_as_csv() adds the ax_{index:02d}_ prefix
    - This function simulates that process to get the final column names

    Parameters
    ----------
    id_val : str
        The plot ID (e.g., "boxplot_0", "plot_0")
    method : str
        The plotting method name (e.g., "boxplot", "plot", "scatter")
    tracked_dict : dict
        The tracked data dictionary
    kwargs : dict
        The keyword arguments passed to the plot
    ax_index : int
        Flattened index of axes (0 for single axes, 0-N for multi-axes)

    Returns
    -------
    list
        List of column names that will be in the CSV (exact match)
    """
    # Import the actual formatters to ensure consistency
    # This is the single source of truth - we use the same code path as CSV export
    try:
        from scitex.plt._subplots._export_as_csv import format_record
        import pandas as pd

        # Construct the record tuple as used in tracking
        record = (id_val, method, tracked_dict, kwargs)

        # Call the actual formatter to get the DataFrame
        df = format_record(record)

        if df is not None and not df.empty:
            # Add the axis prefix (this is what FigWrapper.export_as_csv does)
            # Uses zero-padded index: ax_00_, ax_01_, etc.
            prefix = f"ax_{ax_index:02d}_"
            columns = []
            for col in df.columns:
                col_str = str(col)
                if not col_str.startswith(prefix):
                    col_str = f"{prefix}{col_str}"
                columns.append(col_str)
            return columns

    except Exception:
        # If formatters fail, fall back to pattern-based generation
        pass

    # Fallback: Pattern-based column name generation
    # This should rarely be used since we prefer the actual formatter
    import numpy as np

    prefix = f"ax_{ax_index:02d}_"
    columns = []

    # Get args from tracked_dict
    args = tracked_dict.get("args", []) if tracked_dict else []

    if method in ("boxplot", "stx_box"):
        # Boxplot: one column per box (mirrors _format_boxplot)
        if len(args) >= 1:
            data = args[0]
            labels = kwargs.get("labels", None) if kwargs else None

            from scitex.types import is_listed_X as scitex_types_is_listed_X

            if isinstance(data, np.ndarray) or scitex_types_is_listed_X(data, [float, int]):
                # Single box
                if labels and len(labels) == 1:
                    columns.append(f"{prefix}{id_val}_{labels[0]}")
                else:
                    columns.append(f"{prefix}{id_val}_boxplot_0")
            else:
                # Multiple boxes
                try:
                    num_boxes = len(data)
                    if labels and len(labels) == num_boxes:
                        for label in labels:
                            columns.append(f"{prefix}{id_val}_{label}")
                    else:
                        for i in range(num_boxes):
                            columns.append(f"{prefix}{id_val}_boxplot_{i}")
                except TypeError:
                    columns.append(f"{prefix}{id_val}_boxplot_0")

    elif method in ("plot", "stx_line"):
        # Line plot: x and y columns
        # For single axes (ax_index=0), use simple prefix
        columns.append(f"{prefix}{id_val}_plot_x")
        columns.append(f"{prefix}{id_val}_plot_y")

    elif method in ("scatter", "plot_scatter"):
        columns.append(f"{prefix}{id_val}_scatter_x")
        columns.append(f"{prefix}{id_val}_scatter_y")

    elif method in ("bar", "barh"):
        columns.append(f"{prefix}{id_val}_bar_x")
        columns.append(f"{prefix}{id_val}_bar_height")

    elif method == "hist":
        columns.append(f"{prefix}{id_val}_hist_bins")
        columns.append(f"{prefix}{id_val}_hist_counts")

    elif method in ("violinplot", "stx_violin"):
        if len(args) >= 1:
            data = args[0]
            try:
                num_violins = len(data)
                for i in range(num_violins):
                    columns.append(f"{prefix}{id_val}_violin_{i}")
            except TypeError:
                columns.append(f"{prefix}{id_val}_violin_0")

    elif method == "errorbar":
        columns.append(f"{prefix}{id_val}_errorbar_x")
        columns.append(f"{prefix}{id_val}_errorbar_y")
        columns.append(f"{prefix}{id_val}_errorbar_yerr")

    elif method == "fill_between":
        columns.append(f"{prefix}{id_val}_fill_x")
        columns.append(f"{prefix}{id_val}_fill_y1")
        columns.append(f"{prefix}{id_val}_fill_y2")

    elif method in ("imshow", "stx_heatmap", "stx_image"):
        if len(args) >= 1:
            data = args[0]
            try:
                if hasattr(data, "shape") and len(data.shape) >= 2:
                    columns.append(f"{prefix}{id_val}_image_data")
            except (TypeError, AttributeError):
                pass

    elif method in ("stx_kde", "stx_ecdf"):
        suffix = method.replace("stx_", "")
        columns.append(f"{prefix}{id_val}_{suffix}_x")
        columns.append(f"{prefix}{id_val}_{suffix}_y")

    elif method in ("stx_mean_std", "stx_mean_ci", "stx_median_iqr", "stx_shaded_line"):
        suffix = method.replace("stx_", "")
        columns.append(f"{prefix}{id_val}_{suffix}_x")
        columns.append(f"{prefix}{id_val}_{suffix}_y")
        columns.append(f"{prefix}{id_val}_{suffix}_lower")
        columns.append(f"{prefix}{id_val}_{suffix}_upper")

    elif method.startswith("sns_"):
        sns_type = method.replace("sns_", "")
        if sns_type in ("boxplot", "violinplot"):
            columns.append(f"{prefix}{id_val}_{sns_type}_data")
        elif sns_type in ("scatterplot", "lineplot"):
            columns.append(f"{prefix}{id_val}_{sns_type}_x")
            columns.append(f"{prefix}{id_val}_{sns_type}_y")
        elif sns_type == "barplot":
            columns.append(f"{prefix}{id_val}_barplot_x")
            columns.append(f"{prefix}{id_val}_barplot_y")
        elif sns_type == "histplot":
            columns.append(f"{prefix}{id_val}_histplot_bins")
            columns.append(f"{prefix}{id_val}_histplot_counts")
        elif sns_type == "kdeplot":
            columns.append(f"{prefix}{id_val}_kdeplot_x")
            columns.append(f"{prefix}{id_val}_kdeplot_y")

    return columns


def assert_csv_json_consistency(csv_path: str, json_path: str = None) -> None:
    """
    Assert that CSV data file and its JSON metadata are consistent.

    Raises AssertionError if the column names don't match.

    Parameters
    ----------
    csv_path : str
        Path to the CSV data file
    json_path : str, optional
        Path to the JSON metadata file. If not provided, assumes
        the JSON is at the same location with .json extension.

    Raises
    ------
    AssertionError
        If CSV and JSON column names don't match
    FileNotFoundError
        If CSV or JSON files don't exist

    Examples
    --------
    >>> assert_csv_json_consistency('/tmp/plot.csv')  # Passes silently if valid
    >>> # Or use in tests:
    >>> try:
    ...     assert_csv_json_consistency('/tmp/plot.csv')
    ... except AssertionError as e:
    ...     print(f"Validation failed: {e}")
    """
    result = verify_csv_json_consistency(csv_path, json_path)

    if result['errors']:
        raise FileNotFoundError('\n'.join(result['errors']))

    if not result['valid']:
        msg_parts = ["CSV/JSON column mismatch:"]
        if result['missing_in_csv']:
            msg_parts.append(f"  Missing in CSV: {result['missing_in_csv']}")
        if result['extra_in_csv']:
            msg_parts.append(f"  Extra in CSV: {result['extra_in_csv']}")
        raise AssertionError('\n'.join(msg_parts))


def verify_csv_json_consistency(csv_path: str, json_path: str = None) -> dict:
    """
    Verify consistency between CSV data file and its JSON metadata.

    This function checks that the column names in the CSV file match
    those declared in the JSON metadata's csv_columns field.

    Parameters
    ----------
    csv_path : str
        Path to the CSV data file
    json_path : str, optional
        Path to the JSON metadata file. If not provided, assumes
        the JSON is at the same location with .json extension.

    Returns
    -------
    dict
        Verification result with keys:
        - 'valid': bool - True if CSV and JSON are consistent
        - 'csv_columns': list - Column names found in CSV
        - 'json_columns': list - Column names declared in JSON
        - 'missing_in_csv': list - Columns in JSON but not in CSV
        - 'extra_in_csv': list - Columns in CSV but not in JSON
        - 'errors': list - Any error messages

    Examples
    --------
    >>> result = verify_csv_json_consistency('/tmp/plot.csv')
    >>> print(result['valid'])
    True
    >>> print(result['missing_in_csv'])
    []
    """
    import json
    import os
    import pandas as pd

    result = {
        'valid': False,
        'csv_columns': [],
        'json_columns': [],
        'missing_in_csv': [],
        'extra_in_csv': [],
        'errors': [],
    }

    # Determine JSON path
    if json_path is None:
        base, _ = os.path.splitext(csv_path)
        json_path = base + '.json'

    # Check files exist
    if not os.path.exists(csv_path):
        result['errors'].append(f"CSV file not found: {csv_path}")
        return result
    if not os.path.exists(json_path):
        result['errors'].append(f"JSON file not found: {json_path}")
        return result

    try:
        # Read CSV columns
        df = pd.read_csv(csv_path, nrows=0)  # Just read header
        csv_columns = list(df.columns)
        result['csv_columns'] = csv_columns
    except Exception as e:
        result['errors'].append(f"Error reading CSV: {e}")
        return result

    try:
        # Read JSON metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Prefer csv_columns_actual (exact match) over csv_columns (predicted)
        json_columns = []
        if 'csv_columns_actual' in metadata:
            # Use the exact columns from actual CSV export
            json_columns = metadata['csv_columns_actual']
        elif 'csv_columns' in metadata:
            # Fallback to predicted columns (may not match due to deduplication)
            for entry in metadata['csv_columns']:
                if 'columns' in entry:
                    json_columns.extend(entry['columns'])
        result['json_columns'] = json_columns
    except Exception as e:
        result['errors'].append(f"Error reading JSON: {e}")
        return result

    # Compare columns
    csv_set = set(csv_columns)
    json_set = set(json_columns)

    result['missing_in_csv'] = list(json_set - csv_set)
    result['extra_in_csv'] = list(csv_set - json_set)
    result['valid'] = len(result['missing_in_csv']) == 0 and len(result['extra_in_csv']) == 0

    return result


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
        print(f"  • Axes size: {metadata['dimensions']['axes_size_mm']} mm")
        print(f"  • DPI: {metadata['dimensions']['dpi']}")
    if "scitex" in metadata and "mode" in metadata["scitex"]:
        print(f"  • Mode: {metadata['scitex']['mode']}")
    if "scitex" in metadata and "style_mm" in metadata["scitex"]:
        print("  • Style: Embedded ✓")

# EOF
