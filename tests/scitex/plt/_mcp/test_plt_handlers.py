# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/handlers.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-13
# # File: src/scitex/plt/_mcp/handlers.py
# 
# """MCP Handler implementations for SciTeX plt module.
# 
# Provides async handlers for publication-quality plotting operations.
# This module re-exports handlers from submodules for backward compatibility.
# """
# 
# from __future__ import annotations
# 
# # Re-export annotation handlers
# from ._handlers_annotation import (
#     add_panel_label_handler,
#     add_significance_handler,
# )
# 
# # Re-export figure registry and helper
# # Re-export figure management handlers
# from ._handlers_figure import (
#     _FIGURE_REGISTRY,
#     _get_axes,
#     close_figure_handler,
#     create_figure_handler,
#     crop_figure_handler,
#     save_figure_handler,
# )
# 
# # Re-export plot handlers
# from ._handlers_plot import (
#     plot_bar_handler,
#     plot_box_handler,
#     plot_line_handler,
#     plot_scatter_handler,
#     plot_violin_handler,
# )
# 
# # Re-export style handlers
# from ._handlers_style import (
#     get_color_palette_handler,
#     get_dpi_settings_handler,
#     get_style_handler,
#     list_presets_handler,
#     set_style_handler,
# )
# 
# __all__ = [
#     # Style handlers
#     "get_style_handler",
#     "set_style_handler",
#     "list_presets_handler",
#     "get_dpi_settings_handler",
#     "get_color_palette_handler",
#     # Figure management
#     "create_figure_handler",
#     "crop_figure_handler",
#     "save_figure_handler",
#     "close_figure_handler",
#     # Plot handlers
#     "plot_bar_handler",
#     "plot_scatter_handler",
#     "plot_line_handler",
#     "plot_box_handler",
#     "plot_violin_handler",
#     # Annotation handlers
#     "add_significance_handler",
#     "add_panel_label_handler",
#     # Internal (for submodules)
#     "_FIGURE_REGISTRY",
#     "_get_axes",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_mcp/handlers.py
# --------------------------------------------------------------------------------
