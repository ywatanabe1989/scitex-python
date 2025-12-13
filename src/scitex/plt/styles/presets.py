#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 21:30:00 (ywatanabe)"
# File: ./src/scitex/plt/styles/presets.py

"""
SciTeX style configuration.

Style Control Levels:
    1. Global level (subplots): axes dimensions, fonts, default line thickness
    2. Plot level (per-call): override for individual ax.plot(), ax.scatter(), etc.

Priority cascade: direct → env → yaml → default

Style Management:
    - load_style(path): Load style from YAML/JSON file
    - save_style(path): Export current style to file
    - set_style(style_dict): Change active style globally
    - get_style(): Get current active style

Usage:
    from scitex.plt.styles import load_style, save_style, set_style

    # Load and use custom style
    style = load_style("my_style.yaml")
    fig, ax = stx.plt.subplots(**style)

    # Export current style
    save_style("exported_style.yaml")

    # Change global style
    set_style({"axes_width_mm": 60, "trace_thickness_mm": 0.3})
"""

__all__ = [
    "SCITEX_STYLE",
    "STYLE",
    "load_style",
    "save_style",
    "set_style",
    "get_style",
    "resolve_style_value",
    # DPI utilities
    "get_default_dpi",
    "get_display_dpi",
    "get_preview_dpi",
    "DPI_SAVE",
    "DPI_DISPLAY",
    "DPI_PREVIEW",
]

from pathlib import Path
from typing import Any, Dict, Optional, Union

import scitex.io
from scitex.config import PriorityConfig

_STYLE_FILE = Path(__file__).parent / "SCITEX_STYLE.yaml"
_config: Optional[PriorityConfig] = None
_active_style: Optional[Dict[str, Any]] = None  # User-set style override


def _get_config(yaml_path: Optional[Path] = None) -> PriorityConfig:
    """Get or create PriorityConfig from YAML."""
    global _config
    if _config is None or yaml_path:
        yaml = scitex.io.load(yaml_path or _STYLE_FILE)
        flat = {
            f"{k}.{k2}": v2
            for k, v in yaml.items()
            if isinstance(v, dict) and k != "presets"
            for k2, v2 in v.items()
        }
        _config = PriorityConfig(flat, env_prefix="SCITEX_PLT_", auto_uppercase=True)
    return _config


def resolve_style_value(
    key: str, direct_val: Any = None, default: Any = None, type: type = float
) -> Any:
    """Resolve value with priority: direct → env → yaml → default.

    Key format: 'axes.width_mm' - dots for YAML hierarchy, underscores for env vars.
    Env var: SCITEX_PLT_AXES_WIDTH_MM (prefix + key with dots→underscores, uppercased)
    """
    return _get_config().resolve(key, direct_val, default, type)


# =============================================================================
# DPI Resolution - Central Source of Truth
# =============================================================================
#
# DPI Priority Chain:
#   1. Bundle's geometry_px.json (highest - existing figure's actual DPI)
#   2. User override / session setting
#   3. SCITEX_STYLE.yaml output.dpi (project default)
#   4. Hardcoded fallback (only if config missing)
#
# Usage:
#   from scitex.plt.styles import get_default_dpi, DPI_SAVE
#
#   # For saving figures (publication quality)
#   fig.savefig("out.png", dpi=get_default_dpi())
#
#   # For display/preview (lower resolution)
#   fig.savefig("preview.png", dpi=get_display_dpi())
#
#   # When loading from bundle, use bundle's DPI:
#   dpi = bundle.get("geometry", {}).get("dpi") or get_default_dpi()
#

# Fallback values (only used if config unavailable)
_FALLBACK_DPI_SAVE = 300
_FALLBACK_DPI_DISPLAY = 100
_FALLBACK_DPI_PREVIEW = 150


def get_default_dpi() -> int:
    """Get default DPI for saving/publication from config.

    Priority: SCITEX_STYLE.yaml → env var → fallback 300

    Returns:
        int: DPI value for publication-quality output
    """
    return int(resolve_style_value("output.dpi", None, _FALLBACK_DPI_SAVE, int))


def get_display_dpi() -> int:
    """Get DPI for screen display (lower resolution for speed).

    Returns approximately 1/3 of save DPI, minimum 100.

    Returns:
        int: DPI value for screen display
    """
    save_dpi = get_default_dpi()
    return max(_FALLBACK_DPI_DISPLAY, save_dpi // 3)


def get_preview_dpi() -> int:
    """Get DPI for editor previews and thumbnails.

    Returns 1/2 of save DPI, clamped between 100-200.

    Returns:
        int: DPI value for previews
    """
    save_dpi = get_default_dpi()
    return max(_FALLBACK_DPI_DISPLAY, min(200, save_dpi // 2))


# Module-level constants (evaluated at import time)
# Use functions for dynamic resolution, constants for static defaults
DPI_SAVE = _FALLBACK_DPI_SAVE      # Use get_default_dpi() for dynamic
DPI_DISPLAY = _FALLBACK_DPI_DISPLAY
DPI_PREVIEW = _FALLBACK_DPI_PREVIEW


def load_style(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load style from YAML as subplots kwargs."""
    cfg = _get_config(Path(path) if path else None)
    g = lambda k, d, t=float: cfg.resolve(k, None, d, t)
    return {
        "axes_width_mm": g("axes.width_mm", 40),
        "axes_height_mm": g("axes.height_mm", 28),
        "axes_thickness_mm": g("axes.thickness_mm", 0.2),
        "margin_left_mm": g("margins.left_mm", 20),
        "margin_right_mm": g("margins.right_mm", 20),
        "margin_bottom_mm": g("margins.bottom_mm", 20),
        "margin_top_mm": g("margins.top_mm", 20),
        "space_w_mm": g("spacing.horizontal_mm", 8),
        "space_h_mm": g("spacing.vertical_mm", 10),
        "tick_length_mm": g("ticks.length_mm", 0.8),
        "tick_thickness_mm": g("ticks.thickness_mm", 0.2),
        "n_ticks": g("ticks.n_ticks", 4, int),
        "trace_thickness_mm": g("lines.trace_mm", 0.2),
        "errorbar_thickness_mm": g("lines.errorbar_mm", 0.2),
        "errorbar_cap_width_mm": g("lines.errorbar_cap_mm", 0.8),
        "bar_edge_thickness_mm": g("lines.bar_edge_mm", 0.2),
        "kde_line_thickness_mm": g("lines.kde_mm", 0.2),
        "scatter_size_mm": g("markers.scatter_mm", 0.8),
        "marker_size_mm": g("markers.size_mm", 0.8),
        "font_family": g("fonts.family", "Arial", str),
        "axis_font_size_pt": g("fonts.axis_label_pt", 7),
        "tick_font_size_pt": g("fonts.tick_label_pt", 7),
        "title_font_size_pt": g("fonts.title_pt", 8),
        "suptitle_font_size_pt": g("fonts.suptitle_pt", 8),
        "legend_font_size_pt": g("fonts.legend_pt", 6),
        "annotation_font_size_pt": g("fonts.annotation_pt", 6),
        "label_pad_pt": g("padding.label_pt", 0.5),
        "tick_pad_pt": g("padding.tick_pt", 2.0),
        "title_pad_pt": g("padding.title_pt", 1.0),
        "dpi": g("output.dpi", 300, int),
        "transparent": g("output.transparent", True, bool),
        "auto_scale_axes": g("behavior.auto_scale_axes", True, bool),
        "mode": "publication",
    }


def get_style() -> Dict[str, Any]:
    """Get current active style configuration.

    Returns style with priority: active_style → env → yaml → default
    """
    global _active_style
    base = load_style()
    if _active_style:
        base.update(_active_style)
    return base


def set_style(style_dict: Optional[Dict[str, Any]] = None) -> None:
    """Set active style globally.

    Args:
        style_dict: Style values to override. Pass None to reset to defaults.

    Example:
        set_style({"axes_width_mm": 60, "trace_thickness_mm": 0.3})
        set_style(None)  # Reset to defaults
    """
    global _active_style, SCITEX_STYLE, STYLE
    _active_style = style_dict
    SCITEX_STYLE = get_style()
    STYLE = SCITEX_STYLE


def save_style(path: Union[str, Path], style: Optional[Dict[str, Any]] = None) -> Path:
    """Export style to YAML or JSON file.

    Args:
        path: Output file path (.yaml, .yml, or .json)
        style: Style dict to export. If None, exports current active style.

    Returns:
        Path to saved file.

    Example:
        save_style("my_style.yaml")
        save_style("custom.json", {"axes_width_mm": 60})
    """
    path = Path(path)
    style = style or get_style()

    # Convert flat style dict to hierarchical YAML structure
    hierarchical = _flat_to_hierarchical(style)

    scitex.io.save(hierarchical, path)
    return path


def _flat_to_hierarchical(style: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat style dict to hierarchical YAML structure.

    Example:
        {"axes_width_mm": 40} -> {"axes": {"width_mm": 40}}
    """
    # Mapping from flat keys to hierarchical paths
    mapping = {
        "axes_width_mm": ("axes", "width_mm"),
        "axes_height_mm": ("axes", "height_mm"),
        "axes_thickness_mm": ("axes", "thickness_mm"),
        "margin_left_mm": ("margins", "left_mm"),
        "margin_right_mm": ("margins", "right_mm"),
        "margin_bottom_mm": ("margins", "bottom_mm"),
        "margin_top_mm": ("margins", "top_mm"),
        "space_w_mm": ("spacing", "horizontal_mm"),
        "space_h_mm": ("spacing", "vertical_mm"),
        "tick_length_mm": ("ticks", "length_mm"),
        "tick_thickness_mm": ("ticks", "thickness_mm"),
        "n_ticks": ("ticks", "n_ticks"),
        "trace_thickness_mm": ("lines", "trace_mm"),
        "errorbar_thickness_mm": ("lines", "errorbar_mm"),
        "errorbar_cap_width_mm": ("lines", "errorbar_cap_mm"),
        "bar_edge_thickness_mm": ("lines", "bar_edge_mm"),
        "kde_line_thickness_mm": ("lines", "kde_mm"),
        "scatter_size_mm": ("markers", "scatter_mm"),
        "marker_size_mm": ("markers", "size_mm"),
        "font_family": ("fonts", "family"),
        "axis_font_size_pt": ("fonts", "axis_label_pt"),
        "tick_font_size_pt": ("fonts", "tick_label_pt"),
        "title_font_size_pt": ("fonts", "title_pt"),
        "suptitle_font_size_pt": ("fonts", "suptitle_pt"),
        "legend_font_size_pt": ("fonts", "legend_pt"),
        "annotation_font_size_pt": ("fonts", "annotation_pt"),
        "label_pad_pt": ("padding", "label_pt"),
        "tick_pad_pt": ("padding", "tick_pt"),
        "title_pad_pt": ("padding", "title_pt"),
        "dpi": ("output", "dpi"),
        "transparent": ("output", "transparent"),
        "auto_scale_axes": ("behavior", "auto_scale_axes"),
    }

    result: Dict[str, Any] = {}
    for flat_key, value in style.items():
        if flat_key in mapping:
            section, key = mapping[flat_key]
            if section not in result:
                result[section] = {}
            result[section][key] = value
        # Skip keys not in mapping (like 'mode')

    return result


SCITEX_STYLE = load_style()
STYLE = SCITEX_STYLE


# EOF
