#!/usr/bin/env python3
# File: ./src/scitex/bridge/__init__.py
# Time-stamp: "2024-12-09 09:30:00 (ywatanabe)"
"""
SciTeX Bridge Module - Cross-module adapters and transformations.

This module provides the official API for connecting SciTeX modules:
- stats ↔ plt: Statistical results to plot annotations
- stats ↔ vis: Statistical results to vis annotations
- plt ↔ vis: Matplotlib figures to vis FigureModel

Design Principles:
- Bridge functions use only public APIs of each module
- All transformations go through schema validation
- Single source of truth for cross-module conventions
- Protocol versioning for forward/backward compatibility

Protocol Version: 1.0.0

Coordinate Conventions:
- plt bridge: Uses axes coordinates (0-1 normalized)
- vis bridge: Uses data coordinates (actual x/y values)
- See COORDINATE_SYSTEMS for full definitions

Usage:
    from scitex.bridge import (
        # Protocol version
        BRIDGE_PROTOCOL_VERSION,
        check_protocol_compatibility,

        # Stats to Plt
        add_stat_to_axes,
        extract_stats_from_axes,

        # Stats to Vis
        stat_result_to_annotation,
        add_stats_to_figure_model,

        # Plt to Vis
        figure_to_vis_model,
        axes_to_vis_axes,
        tracking_to_plot_configs,
    )
"""

# Stats ↔ Plt bridges
# FigRecipe integration (optional)
from scitex.bridge._figrecipe import (
    FIGRECIPE_AVAILABLE,
    has_figrecipe,
    load_recipe,
    save_with_recipe,
)

# High-level helpers
from scitex.bridge._helpers import (
    add_stats_from_results,
)

# Plt ↔ Vis bridges
from scitex.bridge._plt_vis import (
    axes_to_vis_axes,
    collect_figure_data,
    figure_to_vis_model,
    tracking_to_plot_configs,
)

# Protocol versioning
from scitex.bridge._protocol import (
    BRIDGE_PROTOCOL_VERSION,
    COORDINATE_SYSTEMS,
    ProtocolInfo,
    add_protocol_metadata,
    check_protocol_compatibility,
    extract_protocol_metadata,
)
from scitex.bridge._stats_plt import (
    add_stat_to_axes,
    extract_stats_from_axes,
    format_stat_for_plot,
)

# Stats ↔ Vis bridges
from scitex.bridge._stats_vis import (
    add_stats_to_figure_model,
    position_stat_annotation,
    stat_result_to_annotation,
)

__all__ = [
    # Protocol
    "BRIDGE_PROTOCOL_VERSION",
    "ProtocolInfo",
    "check_protocol_compatibility",
    "add_protocol_metadata",
    "extract_protocol_metadata",
    "COORDINATE_SYSTEMS",
    # Stats ↔ Plt
    "add_stat_to_axes",
    "extract_stats_from_axes",
    "format_stat_for_plot",
    # Stats ↔ Vis
    "stat_result_to_annotation",
    "add_stats_to_figure_model",
    "position_stat_annotation",
    # Plt ↔ Vis
    "figure_to_vis_model",
    "axes_to_vis_axes",
    "tracking_to_plot_configs",
    "collect_figure_data",
    # High-level helpers
    "add_stats_from_results",
    # FigRecipe integration
    "save_with_recipe",
    "load_recipe",
    "has_figrecipe",
    "FIGRECIPE_AVAILABLE",
]


# EOF
