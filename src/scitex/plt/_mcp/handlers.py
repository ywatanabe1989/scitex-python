#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/plt/_mcp.handlers.py
# ----------------------------------------

"""
MCP Handler implementations for SciTeX plt module.

Provides async handlers for publication-quality plotting operations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

# Figure registry for tracking active figures across MCP calls
_FIGURE_REGISTRY: dict[str, dict[str, Any]] = {}


async def get_style_handler() -> dict:
    """
    Get current SciTeX publication style configuration.

    Returns
    -------
    dict
        Success status and current style parameters
    """
    try:
        from scitex.plt.styles.presets import get_style

        style = get_style()

        return {
            "success": True,
            "style": style,
            "description": "Current SciTeX publication style configuration",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def set_style_handler(
    axes_width_mm: Optional[float] = None,
    axes_height_mm: Optional[float] = None,
    margin_left_mm: Optional[float] = None,
    margin_right_mm: Optional[float] = None,
    margin_top_mm: Optional[float] = None,
    margin_bottom_mm: Optional[float] = None,
    dpi: Optional[int] = None,
    axis_font_size_pt: Optional[float] = None,
    tick_font_size_pt: Optional[float] = None,
    title_font_size_pt: Optional[float] = None,
    legend_font_size_pt: Optional[float] = None,
    trace_thickness_mm: Optional[float] = None,
    reset: bool = False,
) -> dict:
    """
    Set global style overrides for publication figures.

    Parameters
    ----------
    axes_width_mm : float, optional
        Axes width in millimeters
    axes_height_mm : float, optional
        Axes height in millimeters
    margin_*_mm : float, optional
        Margins in millimeters
    dpi : int, optional
        Output resolution
    *_font_size_pt : float, optional
        Font sizes in points
    trace_thickness_mm : float, optional
        Line thickness in mm
    reset : bool, optional
        Reset to defaults before applying

    Returns
    -------
    dict
        Success status and updated style
    """
    try:
        from scitex.plt.styles.presets import get_style, set_style

        # Reset if requested
        if reset:
            set_style(None)

        # Build style dict from provided parameters
        style_updates = {}

        if axes_width_mm is not None:
            style_updates["axes_width_mm"] = axes_width_mm
        if axes_height_mm is not None:
            style_updates["axes_height_mm"] = axes_height_mm
        if margin_left_mm is not None:
            style_updates["margin_left_mm"] = margin_left_mm
        if margin_right_mm is not None:
            style_updates["margin_right_mm"] = margin_right_mm
        if margin_top_mm is not None:
            style_updates["margin_top_mm"] = margin_top_mm
        if margin_bottom_mm is not None:
            style_updates["margin_bottom_mm"] = margin_bottom_mm
        if dpi is not None:
            style_updates["dpi"] = dpi
        if axis_font_size_pt is not None:
            style_updates["axis_font_size_pt"] = axis_font_size_pt
        if tick_font_size_pt is not None:
            style_updates["tick_font_size_pt"] = tick_font_size_pt
        if title_font_size_pt is not None:
            style_updates["title_font_size_pt"] = title_font_size_pt
        if legend_font_size_pt is not None:
            style_updates["legend_font_size_pt"] = legend_font_size_pt
        if trace_thickness_mm is not None:
            style_updates["trace_thickness_mm"] = trace_thickness_mm

        # Apply style updates
        if style_updates:
            set_style(style_updates)

        # Get final style
        final_style = get_style()

        return {
            "success": True,
            "updated_parameters": list(style_updates.keys()),
            "style": final_style,
            "message": f"Updated {len(style_updates)} style parameters",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def list_presets_handler() -> dict:
    """
    List available publication style presets.

    Returns
    -------
    dict
        Success status and list of presets
    """
    try:
        # Define available presets with descriptions
        presets = [
            {
                "name": "SCITEX_STYLE",
                "description": "Default SciTeX publication style",
                "axes_size_mm": "40x28",
                "dpi": 300,
                "font_sizes_pt": {"axis": 7, "tick": 7, "title": 8, "legend": 6},
            },
            {
                "name": "nature",
                "description": "Nature journal style (single column)",
                "axes_size_mm": "89x60",
                "dpi": 300,
                "notes": "Single column width: 89mm",
            },
            {
                "name": "science",
                "description": "Science journal style",
                "axes_size_mm": "85x60",
                "dpi": 300,
                "notes": "Single column width: 85mm",
            },
            {
                "name": "pnas",
                "description": "PNAS journal style",
                "axes_size_mm": "87x60",
                "dpi": 300,
                "notes": "Single column width: 8.7cm",
            },
            {
                "name": "ieee",
                "description": "IEEE journal style",
                "axes_size_mm": "88x60",
                "dpi": 300,
                "notes": "Single column width: 3.5 inches",
            },
        ]

        return {
            "success": True,
            "count": len(presets),
            "presets": presets,
            "usage": "Use set_style() to apply custom dimensions",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def crop_figure_handler(
    input_path: str,
    output_path: Optional[str] = None,
    margin: int = 12,
    overwrite: bool = False,
) -> dict:
    """
    Auto-crop whitespace from a saved figure image.

    Parameters
    ----------
    input_path : str
        Path to input figure
    output_path : str, optional
        Output path (adds '_cropped' suffix if not provided)
    margin : int, optional
        Margin in pixels (default: 12)
    overwrite : bool, optional
        Overwrite input file

    Returns
    -------
    dict
        Success status and output path
    """
    try:
        from scitex.plt import crop

        loop = asyncio.get_event_loop()
        result_path = await loop.run_in_executor(
            None,
            lambda: crop(
                input_path=input_path,
                output_path=output_path,
                margin=margin,
                overwrite=overwrite,
            ),
        )

        return {
            "success": True,
            "input_path": input_path,
            "output_path": str(result_path),
            "margin_pixels": margin,
            "message": f"Cropped figure saved to {result_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_dpi_settings_handler() -> dict:
    """
    Get DPI settings for different output contexts.

    Returns
    -------
    dict
        Success status and DPI settings
    """
    try:
        from scitex.plt.styles.presets import (
            get_default_dpi,
            get_display_dpi,
            get_preview_dpi,
        )

        return {
            "success": True,
            "dpi_settings": {
                "save": {
                    "value": get_default_dpi(),
                    "description": "Publication-quality output (high resolution)",
                },
                "display": {
                    "value": get_display_dpi(),
                    "description": "Screen display (lower resolution for speed)",
                },
                "preview": {
                    "value": get_preview_dpi(),
                    "description": "Editor previews and thumbnails",
                },
            },
            "recommendation": "Use 'save' DPI for final figures, 'display' for iterating",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_color_palette_handler(format: str = "hex") -> dict:
    """
    Get the SciTeX color palette.

    Parameters
    ----------
    format : str
        Color format: 'hex', 'rgb', or 'rgba'

    Returns
    -------
    dict
        Success status and color palette
    """
    try:
        from scitex.plt import color as color_module

        # Get color parameters
        params = getattr(color_module, "PARAMS", {})

        # Get cycle colors
        rgba_cycle = params.get("RGBA_NORM_FOR_CYCLE", {})

        colors = {}
        for name, rgba in rgba_cycle.items():
            if format == "hex":
                # Convert RGBA to hex
                r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                colors[name] = f"#{r:02x}{g:02x}{b:02x}"
            elif format == "rgb":
                colors[name] = {
                    "r": int(rgba[0] * 255),
                    "g": int(rgba[1] * 255),
                    "b": int(rgba[2] * 255),
                }
            else:  # rgba
                colors[name] = {
                    "r": rgba[0],
                    "g": rgba[1],
                    "b": rgba[2],
                    "a": rgba[3] if len(rgba) > 3 else 1.0,
                }

        return {
            "success": True,
            "format": format,
            "count": len(colors),
            "colors": colors,
            "usage": "Colors are used in matplotlib's default color cycle",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def create_figure_handler(
    nrows: int = 1,
    ncols: int = 1,
    axes_width_mm: Optional[float] = None,
    axes_height_mm: Optional[float] = None,
    space_w_mm: Optional[float] = None,
    space_h_mm: Optional[float] = None,
) -> dict:
    """
    Create a multi-panel figure canvas with SciTeX style.

    Parameters
    ----------
    nrows : int
        Number of rows (default: 1)
    ncols : int
        Number of columns (default: 1)
    axes_width_mm : float, optional
        Width of each axes in mm (default: 40 from style)
    axes_height_mm : float, optional
        Height of each axes in mm (default: 28 from style)
    space_w_mm : float, optional
        Horizontal spacing between panels (default: 8 from style)
    space_h_mm : float, optional
        Vertical spacing between panels (default: 10 from style)

    Returns
    -------
    dict
        Success status and figure_id for subsequent operations
    """
    import uuid

    try:
        import scitex.plt as splt

        # Build kwargs from provided parameters
        kwargs = {"nrows": nrows, "ncols": ncols}
        if axes_width_mm is not None:
            kwargs["axes_width_mm"] = axes_width_mm
        if axes_height_mm is not None:
            kwargs["axes_height_mm"] = axes_height_mm
        if space_w_mm is not None:
            kwargs["space_w_mm"] = space_w_mm
        if space_h_mm is not None:
            kwargs["space_h_mm"] = space_h_mm

        # Create figure with SciTeX style
        fig, axes = splt.subplots(**kwargs)

        # Generate unique figure ID
        figure_id = str(uuid.uuid4())[:8]

        # Store in registry
        _FIGURE_REGISTRY[figure_id] = {
            "fig": fig,
            "axes": axes,
            "nrows": nrows,
            "ncols": ncols,
        }

        return {
            "success": True,
            "figure_id": figure_id,
            "nrows": nrows,
            "ncols": ncols,
            "message": f"Created {nrows}x{ncols} figure with SciTeX style",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_axes(figure_id: Optional[str], panel: str):
    """Helper to get axes from figure registry."""
    if figure_id is None:
        # Use most recent figure
        if not _FIGURE_REGISTRY:
            raise ValueError("No active figures. Call create_figure first.")
        figure_id = list(_FIGURE_REGISTRY.keys())[-1]

    if figure_id not in _FIGURE_REGISTRY:
        raise ValueError(f"Figure '{figure_id}' not found")

    fig_data = _FIGURE_REGISTRY[figure_id]
    axes = fig_data["axes"]

    # Parse panel specification
    if "," in panel:
        row, col = map(int, panel.split(","))
        # AxesWrapper supports [row, col] indexing
        try:
            ax = axes[row, col]
        except (TypeError, IndexError):
            # Fallback for single axes
            ax = axes
    else:
        # Panel label like 'A', 'B', etc.
        idx = ord(panel.upper()) - ord("A")
        # Use flat iterator if available
        if hasattr(axes, "flat"):
            ax = list(axes.flat)[idx]
        elif hasattr(axes, "__getitem__"):
            ax = axes[idx]
        else:
            ax = axes

    return fig_data["fig"], ax, figure_id


async def plot_bar_handler(
    x: list[str],
    y: list[float],
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    yerr: Optional[list[float]] = None,
    colors: Optional[list[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Create a bar plot on specified panel."""
    try:
        fig, ax, fid = _get_axes(figure_id, panel)

        # Get colors from palette if not specified
        if colors is None:
            from scitex.plt import color as color_module

            params = getattr(color_module, "PARAMS", {})
            rgba_cycle = params.get("RGBA_NORM_FOR_CYCLE", {})
            color_list = list(rgba_cycle.values())
            colors = color_list[: len(x)] if color_list else None

        # Create bar plot
        bars = ax.bar(x, y, yerr=yerr, capsize=3, color=colors)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "plot_type": "bar",
            "n_bars": len(x),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def plot_scatter_handler(
    x: list[float],
    y: list[float],
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    color: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.7,
    add_regression: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Create a scatter plot on specified panel."""
    try:
        import numpy as np

        fig, ax, fid = _get_axes(figure_id, panel)

        # Default color from palette
        if color is None:
            color = "#c633ff"  # SciTeX purple

        # Convert size from mm to points^2 if specified
        s = (size * 2.83465) ** 2 if size else 15

        ax.scatter(x, y, c=color, s=s, alpha=alpha)

        # Add regression line if requested
        if add_regression:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p(x_line), color="#e25e33", linestyle="--", linewidth=1)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "plot_type": "scatter",
            "n_points": len(x),
            "regression_added": add_regression,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def plot_line_handler(
    x: list[float],
    y: list[float],
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    yerr: Optional[list[float]] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    linestyle: str = "-",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Create a line plot on specified panel."""
    try:
        fig, ax, fid = _get_axes(figure_id, panel)

        # Default color
        if color is None:
            color = "#007fbf"  # SciTeX blue

        ax.plot(x, y, color=color, label=label, linestyle=linestyle)

        # Add shaded error region if yerr provided
        if yerr:
            import numpy as np

            y_arr = np.array(y)
            yerr_arr = np.array(yerr)
            ax.fill_between(
                x, y_arr - yerr_arr, y_arr + yerr_arr, alpha=0.3, color=color
            )

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if label:
            ax.legend(loc="upper right", frameon=False)

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "plot_type": "line",
            "n_points": len(x),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def plot_box_handler(
    data: list[list[float]],
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Create a box plot on specified panel."""
    try:
        fig, ax, fid = _get_axes(figure_id, panel)

        bp = ax.boxplot(data, patch_artist=True, widths=0.6)

        # Apply colors
        if colors is None:
            colors = ["#007fbf", "#ff4433", "#14b514", "#c633ff", "#e25e33"]
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[i % len(colors)])

        if labels:
            ax.set_xticklabels(labels)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "plot_type": "box",
            "n_groups": len(data),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def plot_violin_handler(
    data: list[list[float]],
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    labels: Optional[list[str]] = None,
    colors: Optional[list[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Create a violin plot on specified panel."""
    try:
        fig, ax, fid = _get_axes(figure_id, panel)

        positions = list(range(1, len(data) + 1))
        vp = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)

        # Apply colors
        if colors is None:
            colors = ["#007fbf", "#ff4433", "#14b514", "#c633ff", "#e25e33"]
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i % len(colors)])
            body.set_alpha(0.6)

        if labels:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "plot_type": "violin",
            "n_groups": len(data),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def add_significance_handler(
    x1: float,
    x2: float,
    y: float,
    text: str,
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    height: Optional[float] = None,
) -> dict:
    """Add significance bracket between two groups."""
    try:
        fig, ax, fid = _get_axes(figure_id, panel)

        # Get ylim as floats
        ylim = ax.get_ylim()
        ylim_range = float(ylim[1]) - float(ylim[0])
        h = height if height else 0.1 * ylim_range

        # Draw bracket
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8)

        # Add text
        ax.text(
            (x1 + x2) / 2,
            y + h + 0.02 * ylim_range,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
        )

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "bracket": {"x1": x1, "x2": x2, "y": y, "text": text},
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def add_panel_label_handler(
    label: str,
    figure_id: Optional[str] = None,
    panel: str = "0,0",
    x: float = -0.15,
    y: float = 1.1,
    fontsize: float = 10,
    fontweight: str = "bold",
) -> dict:
    """Add panel label (A, B, C, etc.) to a panel."""
    try:
        fig, ax, fid = _get_axes(figure_id, panel)

        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
        )

        return {
            "success": True,
            "figure_id": fid,
            "panel": panel,
            "label": label,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def save_figure_handler(
    output_path: str,
    figure_id: Optional[str] = None,
    dpi: int = 300,
    crop: bool = True,
) -> dict:
    """Save the figure to file."""
    try:
        if figure_id is None:
            if not _FIGURE_REGISTRY:
                raise ValueError("No active figures")
            figure_id = list(_FIGURE_REGISTRY.keys())[-1]

        if figure_id not in _FIGURE_REGISTRY:
            raise ValueError(f"Figure '{figure_id}' not found")

        fig = _FIGURE_REGISTRY[figure_id]["fig"]

        # Save figure
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

        # Auto-crop if requested
        final_path = output_path
        if crop and output_path.endswith(".png"):
            from scitex.plt import crop as crop_fn

            final_path = crop_fn(output_path, overwrite=True)

        return {
            "success": True,
            "figure_id": figure_id,
            "output_path": str(final_path),
            "dpi": dpi,
            "cropped": crop,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def close_figure_handler(figure_id: Optional[str] = None) -> dict:
    """Close a figure and free memory."""
    try:
        import scitex.plt as splt

        if figure_id is None:
            if not _FIGURE_REGISTRY:
                return {"success": True, "message": "No figures to close"}
            figure_id = list(_FIGURE_REGISTRY.keys())[-1]

        if figure_id in _FIGURE_REGISTRY:
            fig = _FIGURE_REGISTRY[figure_id]["fig"]
            splt.close(fig)
            del _FIGURE_REGISTRY[figure_id]

        return {
            "success": True,
            "figure_id": figure_id,
            "message": f"Closed figure {figure_id}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


__all__ = [
    "get_style_handler",
    "set_style_handler",
    "list_presets_handler",
    "crop_figure_handler",
    "get_dpi_settings_handler",
    "get_color_palette_handler",
    "create_figure_handler",
    "plot_bar_handler",
    "plot_scatter_handler",
    "plot_line_handler",
    "plot_box_handler",
    "plot_violin_handler",
    "add_significance_handler",
    "add_panel_label_handler",
    "save_figure_handler",
    "close_figure_handler",
]

# EOF
