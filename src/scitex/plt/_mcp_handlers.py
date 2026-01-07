#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/plt/_mcp_handlers.py
# ----------------------------------------

"""
MCP Handler implementations for SciTeX plt module.

Provides async handlers for publication-quality plotting operations.
"""

from __future__ import annotations

import asyncio
from typing import Optional


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


__all__ = [
    "get_style_handler",
    "set_style_handler",
    "list_presets_handler",
    "crop_figure_handler",
    "get_dpi_settings_handler",
    "get_color_palette_handler",
]

# EOF
