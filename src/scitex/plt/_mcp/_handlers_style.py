#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: src/scitex/plt/_mcp/_handlers_style.py

"""Style-related MCP handlers for SciTeX plt module.

Delegates to figrecipe's style system for consistency.
"""

from __future__ import annotations

from typing import Optional


async def get_style_handler() -> dict:
    """Get current SciTeX publication style configuration."""
    try:
        # Use figrecipe's STYLE object (re-exported via scitex.plt)
        from scitex.plt import STYLE

        if STYLE is None:
            return {
                "success": True,
                "style": None,
                "description": "No style loaded. Use load_style('SCITEX') to load.",
            }

        # Convert STYLE object to dict for JSON serialization
        style_dict = {}
        for attr in ["axes", "margins", "spacing", "fonts", "lines", "output"]:
            if hasattr(STYLE, attr):
                val = getattr(STYLE, attr)
                if hasattr(val, "__dict__"):
                    style_dict[attr] = {
                        k: v for k, v in val.__dict__.items() if not k.startswith("_")
                    }
                else:
                    style_dict[attr] = val

        return {
            "success": True,
            "style": style_dict,
            "description": "Current SciTeX publication style configuration (via figrecipe)",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    reset: Optional[bool] = None,
) -> dict:
    """Set global style overrides for publication figures."""
    try:
        # Use figrecipe's style management (re-exported via scitex.plt)
        from scitex.plt import STYLE, load_style, unload_style

        if reset:
            unload_style()
            load_style("SCITEX")

        # Build style updates dict
        style_updates = {}
        if axes_width_mm is not None:
            style_updates["axes.width_mm"] = axes_width_mm
        if axes_height_mm is not None:
            style_updates["axes.height_mm"] = axes_height_mm
        if margin_left_mm is not None:
            style_updates["margins.left_mm"] = margin_left_mm
        if margin_right_mm is not None:
            style_updates["margins.right_mm"] = margin_right_mm
        if margin_top_mm is not None:
            style_updates["margins.top_mm"] = margin_top_mm
        if margin_bottom_mm is not None:
            style_updates["margins.bottom_mm"] = margin_bottom_mm
        if dpi is not None:
            style_updates["output.dpi"] = dpi
        if axis_font_size_pt is not None:
            style_updates["fonts.axis_label_pt"] = axis_font_size_pt
        if tick_font_size_pt is not None:
            style_updates["fonts.tick_label_pt"] = tick_font_size_pt
        if title_font_size_pt is not None:
            style_updates["fonts.title_pt"] = title_font_size_pt
        if legend_font_size_pt is not None:
            style_updates["fonts.legend_pt"] = legend_font_size_pt
        if trace_thickness_mm is not None:
            style_updates["lines.trace_mm"] = trace_thickness_mm

        # Apply updates to STYLE object
        if style_updates and STYLE is not None:
            for key, value in style_updates.items():
                parts = key.split(".")
                if len(parts) == 2:
                    section, attr = parts
                    if hasattr(STYLE, section):
                        section_obj = getattr(STYLE, section)
                        if hasattr(section_obj, attr):
                            setattr(section_obj, attr, value)

        # Get final style for response
        final_style = await get_style_handler()

        return {
            "success": True,
            "updated_parameters": list(style_updates.keys()),
            "style": final_style.get("style"),
            "message": f"Updated {len(style_updates)} style parameters",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def list_presets_handler() -> dict:
    """List available publication style presets."""
    try:
        # Use figrecipe's list_presets (re-exported via scitex.plt)
        from scitex.plt import list_presets

        presets = list_presets()

        # Format presets for MCP response
        preset_info = []
        for name in presets:
            info = {"name": name}
            # Add descriptions for known presets
            if name == "SCITEX":
                info["description"] = "Default SciTeX publication style"
                info["axes_size_mm"] = "40x28"
            elif name == "NATURE":
                info["description"] = "Nature journal style (single column)"
                info["axes_size_mm"] = "89x60"
            elif name == "SCIENCE":
                info["description"] = "Science journal style"
                info["axes_size_mm"] = "85x60"
            preset_info.append(info)

        return {
            "success": True,
            "count": len(preset_info),
            "presets": preset_info,
            "usage": "Use load_style('PRESET_NAME') to apply a preset",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_dpi_settings_handler() -> dict:
    """Get DPI settings for different output contexts."""
    try:
        # Use figrecipe's STYLE object for DPI settings
        from scitex.plt import STYLE

        # Get DPI from STYLE if available
        save_dpi = 300  # default
        if STYLE is not None and hasattr(STYLE, "output"):
            save_dpi = getattr(STYLE.output, "dpi", 300)

        return {
            "success": True,
            "dpi_settings": {
                "save": {
                    "value": save_dpi,
                    "description": "Publication-quality output (high resolution)",
                },
                "display": {
                    "value": max(100, save_dpi // 3),
                    "description": "Screen display (lower resolution for speed)",
                },
                "preview": {
                    "value": max(72, save_dpi // 4),
                    "description": "Editor previews and thumbnails",
                },
            },
            "recommendation": "Use 'save' DPI for final figures, 'display' for iterating",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_color_palette_handler(format: str = "hex") -> dict:
    """Get the SciTeX color palette."""
    try:
        from scitex.plt import color as color_module

        params = getattr(color_module, "PARAMS", {})
        rgba_cycle = params.get("RGBA_NORM_FOR_CYCLE", {})

        colors = {}
        for name, rgba in rgba_cycle.items():
            if format == "hex":
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
        return {"success": False, "error": str(e)}


__all__ = [
    "get_style_handler",
    "set_style_handler",
    "list_presets_handler",
    "get_dpi_settings_handler",
    "get_color_palette_handler",
]

# EOF
