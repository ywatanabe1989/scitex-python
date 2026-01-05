# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_style_loader.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 20:00:00 (ywatanabe)"
# # File: ./src/scitex/plt/styles/_style_loader.py
# 
# """
# Style loader for SciTeX plotting.
# 
# Loads style configuration from YAML file and provides centralized access
# to all style parameters. This ensures consistency across all plotting functions.
# 
# Usage:
#     from scitex.plt.styles import load_style, get_style, STYLE
# 
#     # Load default style
#     style = load_style()
#     fig, ax = stx.plt.subplots(**style)
# 
#     # Load with journal preset
#     style = load_style(preset="nature")
# 
#     # Load custom YAML file
#     style = load_style("path/to/my_style.yaml")
# 
#     # Access individual style parameters
#     from scitex.plt.styles._style_loader import STYLE
#     line_width = STYLE.lines.trace_mm
# """
# 
# __all__ = ["load_style", "get_style", "STYLE", "reload_style"]
# 
# import os
# from pathlib import Path
# from typing import Any, Dict, Optional, Union
# 
# import yaml
# 
# from scitex.dict import DotDict
# 
# 
# # Path to default style file
# _DEFAULT_STYLE_PATH = Path(__file__).parent / "SCITEX_STYLE.yaml"
# 
# # Global style cache
# _STYLE_CACHE: Optional[DotDict] = None
# 
# 
# def _deep_merge(base: Dict, override: Dict) -> Dict:
#     """Deep merge two dictionaries, with override taking precedence."""
#     result = base.copy()
#     for key, value in override.items():
#         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
#             result[key] = _deep_merge(result[key], value)
#         else:
#             result[key] = value
#     return result
# 
# 
# def _load_yaml(path: Union[str, Path]) -> Dict:
#     """Load YAML file and return as dictionary."""
#     with open(path, "r") as f:
#         return yaml.safe_load(f)
# 
# 
# def reload_style(
#     path: Optional[Union[str, Path]] = None,
#     preset: Optional[str] = None,
# ) -> DotDict:
#     """
#     Reload style from YAML file (clears cache).
# 
#     Parameters
#     ----------
#     path : str or Path, optional
#         Path to YAML style file. If None, uses default SCITEX_STYLE.yaml
#     preset : str, optional
#         Journal preset to apply: "nature", "science", "cell", "pnas"
# 
#     Returns
#     -------
#     DotDict
#         Style configuration as DotDict for dot-access
#     """
#     global _STYLE_CACHE
#     _STYLE_CACHE = None
#     return load_style(path, preset)
# 
# 
# def load_style(
#     path: Optional[Union[str, Path]] = None,
#     preset: Optional[str] = None,
# ) -> DotDict:
#     """
#     Load style configuration from YAML file.
# 
#     Parameters
#     ----------
#     path : str or Path, optional
#         Path to YAML style file. If None, uses default SCITEX_STYLE.yaml
#     preset : str, optional
#         Journal preset to apply: "nature", "science", "cell", "pnas"
# 
#     Returns
#     -------
#     DotDict
#         Style configuration as DotDict for dot-access
# 
#     Examples
#     --------
#     >>> style = load_style()
#     >>> style.fonts.axis_label_pt
#     7
#     >>> style.lines.trace_mm
#     0.2
#     """
#     global _STYLE_CACHE
# 
#     # Use cache if available and no custom path/preset
#     if _STYLE_CACHE is not None and path is None and preset is None:
#         return _STYLE_CACHE
# 
#     # Load from file
#     style_path = Path(path) if path else _DEFAULT_STYLE_PATH
#     if not style_path.exists():
#         raise FileNotFoundError(f"Style file not found: {style_path}")
# 
#     style_dict = _load_yaml(style_path)
# 
#     # Apply preset if specified
#     if preset:
#         presets = style_dict.get("presets", {})
#         if preset not in presets:
#             available = list(presets.keys())
#             raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
# 
#         # Deep merge preset over base style
#         preset_overrides = presets[preset]
#         for section, values in preset_overrides.items():
#             if section in style_dict and isinstance(style_dict[section], dict):
#                 style_dict[section] = _deep_merge(style_dict[section], values)
#             else:
#                 style_dict[section] = values
# 
#     # Remove presets section from final style
#     style_dict.pop("presets", None)
# 
#     # Convert to DotDict for convenient access
#     style = DotDict(style_dict)
# 
#     # Cache if using default
#     if path is None and preset is None:
#         _STYLE_CACHE = style
# 
#     return style
# 
# 
# def get_style() -> DotDict:
#     """
#     Get the current loaded style (loads default if not yet loaded).
# 
#     Returns
#     -------
#     DotDict
#         Current style configuration
#     """
#     global _STYLE_CACHE
#     if _STYLE_CACHE is None:
#         return load_style()
#     return _STYLE_CACHE
# 
# 
# def to_subplots_kwargs(style: Optional[DotDict] = None) -> Dict[str, Any]:
#     """
#     Convert style DotDict to kwargs for stx.plt.subplots().
# 
#     Parameters
#     ----------
#     style : DotDict, optional
#         Style configuration. If None, uses current loaded style.
# 
#     Returns
#     -------
#     dict
#         Keyword arguments for stx.plt.subplots()
# 
#     Examples
#     --------
#     >>> style = load_style(preset="nature")
#     >>> kwargs = to_subplots_kwargs(style)
#     >>> fig, ax = stx.plt.subplots(**kwargs)
#     """
#     if style is None:
#         style = get_style()
# 
#     return {
#         # Axes dimensions
#         "ax_width_mm": style.axes.width_mm,
#         "ax_height_mm": style.axes.height_mm,
#         "ax_thickness_mm": style.axes.thickness_mm,
#         # Margins
#         "margin_left_mm": style.margins.left_mm,
#         "margin_right_mm": style.margins.right_mm,
#         "margin_bottom_mm": style.margins.bottom_mm,
#         "margin_top_mm": style.margins.top_mm,
#         # Spacing
#         "space_w_mm": style.spacing.horizontal_mm,
#         "space_h_mm": style.spacing.vertical_mm,
#         # Ticks
#         "tick_length_mm": style.ticks.length_mm,
#         "tick_thickness_mm": style.ticks.thickness_mm,
#         "n_ticks": style.ticks.n_ticks,
#         # Lines
#         "trace_thickness_mm": style.lines.trace_mm,
#         # Markers
#         "marker_size_mm": style.markers.size_mm,
#         # Fonts
#         "axis_font_size_pt": style.fonts.axis_label_pt,
#         "tick_font_size_pt": style.fonts.tick_label_pt,
#         "title_font_size_pt": style.fonts.title_pt,
#         "suptitle_font_size_pt": style.fonts.suptitle_pt,
#         "legend_font_size_pt": style.fonts.legend_pt,
#         # Padding
#         "label_pad_pt": style.padding.label_pt,
#         "tick_pad_pt": style.padding.tick_pt,
#         "title_pad_pt": style.padding.title_pt,
#         # Output
#         "dpi": style.output.dpi,
#         "transparent": style.output.transparent,
#         # Mode
#         "mode": "publication",
#     }
# 
# 
# # Lazy-loaded global STYLE object
# class _StyleProxy:
#     """Proxy object that loads style on first access."""
# 
#     def __getattr__(self, name: str) -> Any:
#         return getattr(get_style(), name)
# 
#     def __repr__(self) -> str:
#         return repr(get_style())
# 
# 
# STYLE = _StyleProxy()
# 
# 
# if __name__ == "__main__":
#     # Test loading
#     print("Loading default style...")
#     style = load_style()
#     print(f"  axes.width_mm: {style.axes.width_mm}")
#     print(f"  fonts.axis_label_pt: {style.fonts.axis_label_pt}")
#     print(f"  lines.trace_mm: {style.lines.trace_mm}")
# 
#     print("\nLoading Nature preset...")
#     nature_style = load_style(preset="nature")
#     print(f"  axes.width_mm: {nature_style.axes.width_mm}")
# 
#     print("\nConverting to subplots kwargs...")
#     kwargs = to_subplots_kwargs()
#     for k, v in list(kwargs.items())[:5]:
#         print(f"  {k}: {v}")
# 
#     print("\nUsing STYLE proxy...")
#     print(f"  STYLE.fonts.family: {STYLE.fonts.family}")
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_style_loader.py
# --------------------------------------------------------------------------------
