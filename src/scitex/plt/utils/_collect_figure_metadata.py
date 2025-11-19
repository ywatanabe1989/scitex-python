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


def collect_figure_metadata(fig, ax=None) -> Dict:
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

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to collect metadata from
    ax : matplotlib.axes.Axes, optional
        Primary axes to collect dimension info from.
        If not provided, uses first axes in figure.

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
        "metadata_version": "1.0.0",  # Version of the metadata schema itself
        "scitex": {
            "version": scitex.__version__,
            "created_at": datetime.datetime.now().isoformat(),
        },
        "matplotlib": {
            "version": matplotlib.__version__,
        },
    }

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

        # Merge into scitex section
        for key, value in scitex_meta.items():
            if key not in metadata["scitex"]:
                metadata["scitex"][key] = value

    # Alternative: check figure for metadata (for multi-axes cases)
    elif hasattr(fig, "_scitex_metadata"):
        scitex_meta = fig._scitex_metadata
        for key, value in scitex_meta.items():
            if key not in metadata["scitex"]:
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

    return metadata


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
