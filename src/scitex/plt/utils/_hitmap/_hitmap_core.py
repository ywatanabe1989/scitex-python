#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/_hitmap_core.py

"""
Core hitmap generation functions.

This module provides the main hitmap generation functions using unique ID colors
for pixel-perfect element selection.
"""

import io
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np

from ._artist_extraction import get_all_artists
from ._color_application import apply_id_color
from ._color_conversion import id_to_rgb
from ._constants import HITMAP_AXES_COLOR, HITMAP_BACKGROUND_COLOR

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "generate_hitmap_id_colors",
    "generate_hitmap_with_bbox_tight",
]


def generate_hitmap_id_colors(
    fig,
    dpi: int = 100,
    include_text: bool = False,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Generate a hit map using unique ID colors (fastest method).

    Assigns unique RGB colors to each element, renders once, and creates
    a pixel-perfect hit map where each pixel's RGB values encode the
    element ID using 24-bit color space (~16.7M unique IDs).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to generate hit map for.
    dpi : int
        Resolution for hit map rendering.
    include_text : bool
        Whether to include text elements in hit map.

    Returns
    -------
    tuple
        (hitmap_array, color_map) where:
        - hitmap_array: uint32 array with element IDs (0 = background)
        - color_map: dict mapping ID to element info

    Notes
    -----
    Performance: ~89ms for complex figures (33x faster than sequential)
    """
    artists = get_all_artists(fig, include_text)

    if not artists:
        h = int(fig.get_figheight() * dpi)
        w = int(fig.get_figwidth() * dpi)
        return np.zeros((h, w), dtype=np.uint32), {}

    original_props = []
    color_map = {}

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        element_id = i + 1
        r, g, b = id_to_rgb(element_id)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # Store original properties
        props = {"artist": artist, "type": artist_type}
        try:
            if hasattr(artist, "get_color"):
                props["color"] = artist.get_color()
            if hasattr(artist, "get_facecolor"):
                props["facecolor"] = artist.get_facecolor()
            if hasattr(artist, "get_edgecolor"):
                props["edgecolor"] = artist.get_edgecolor()
            if hasattr(artist, "get_alpha"):
                props["alpha"] = artist.get_alpha()
            if hasattr(artist, "get_antialiased"):
                props["antialiased"] = artist.get_antialiased()
        except Exception:
            pass
        original_props.append(props)

        # Build color map entry
        label = ""
        if hasattr(artist, "get_label"):
            label = artist.get_label()
            if label.startswith("_"):
                label = f"{artist_type}_{i}"

        color_map[element_id] = {
            "id": element_id,
            "type": artist_type,
            "label": label,
            "axes_index": ax_idx,
            "rgb": [r, g, b],
        }

        # Apply ID color and disable anti-aliasing
        try:
            apply_id_color(artist, hex_color)
        except Exception:
            pass

    # Make non-artist elements the reserved axes color
    axes_color = HITMAP_AXES_COLOR
    for ax in fig.axes:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(axes_color)
        ax.set_facecolor(HITMAP_BACKGROUND_COLOR)
        ax.tick_params(colors=axes_color, labelcolor=axes_color)
        ax.xaxis.label.set_color(axes_color)
        ax.yaxis.label.set_color(axes_color)
        ax.title.set_color(axes_color)
        if ax.get_legend():
            ax.get_legend().set_visible(False)

    fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

    # Render
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())
    # Convert RGB to element ID using 24-bit encoding
    hitmap = (
        (img[:, :, 0].astype(np.uint32) << 16)
        | (img[:, :, 1].astype(np.uint32) << 8)
        | img[:, :, 2].astype(np.uint32)
    )

    # Restore original properties
    for props in original_props:
        artist = props["artist"]
        try:
            if "color" in props and hasattr(artist, "set_color"):
                artist.set_color(props["color"])
            if "facecolor" in props and hasattr(artist, "set_facecolor"):
                artist.set_facecolor(props["facecolor"])
            if "edgecolor" in props and hasattr(artist, "set_edgecolor"):
                artist.set_edgecolor(props["edgecolor"])
            if "alpha" in props and hasattr(artist, "set_alpha"):
                artist.set_alpha(props["alpha"])
            if "antialiased" in props and hasattr(artist, "set_antialiased"):
                artist.set_antialiased(props["antialiased"])
        except Exception:
            pass

    return hitmap, color_map


def generate_hitmap_with_bbox_tight(
    fig,
    dpi: int = 150,
    include_text: bool = False,
) -> Tuple["Image.Image", Dict[int, Dict[str, Any]]]:
    """
    Generate a hitmap image with bbox_inches='tight' to match PNG output.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to generate hit map for.
    dpi : int
        Resolution for hit map rendering.
    include_text : bool
        Whether to include text elements in hit map.

    Returns
    -------
    tuple
        (hitmap_image, color_map) where:
        - hitmap_image: PIL.Image.Image with RGB-encoded element IDs
        - color_map: dict mapping ID to element info
    """
    from PIL import Image

    artists = get_all_artists(fig, include_text)

    if not artists:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        black_img = Image.new("RGB", img.size, (0, 0, 0))
        return black_img, {}

    original_props = []
    original_ax_props = []

    # Store original axes properties
    for ax in fig.axes:
        ax_props = {
            "ax": ax,
            "facecolor": ax.get_facecolor(),
            "grid_visible": (
                ax.xaxis.get_gridlines()[0].get_visible()
                if ax.xaxis.get_gridlines()
                else False
            ),
            "spines": {name: spine.get_visible() for name, spine in ax.spines.items()},
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "title": ax.get_title(),
            "tick_params": {},
        }
        if ax.get_legend():
            ax_props["legend_visible"] = ax.get_legend().get_visible()
        original_ax_props.append(ax_props)

    original_fig_facecolor = fig.patch.get_facecolor()

    # Build color map
    color_map = {}

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        element_id = i + 1
        r, g, b = id_to_rgb(element_id)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # Store original properties
        props = {"artist": artist, "type": artist_type}
        try:
            if hasattr(artist, "get_color"):
                props["color"] = artist.get_color()
            if hasattr(artist, "get_facecolor"):
                props["facecolor"] = artist.get_facecolor()
            if hasattr(artist, "get_edgecolor"):
                props["edgecolor"] = artist.get_edgecolor()
            if hasattr(artist, "get_alpha"):
                props["alpha"] = artist.get_alpha()
            if hasattr(artist, "get_antialiased"):
                props["antialiased"] = artist.get_antialiased()
        except Exception:
            pass
        original_props.append(props)

        # Build color map entry
        label = ""
        if hasattr(artist, "get_label"):
            label = artist.get_label()
            if label.startswith("_"):
                label = f"{artist_type}_{i}"

        color_map[element_id] = {
            "id": element_id,
            "type": artist_type,
            "label": label,
            "axes_index": ax_idx,
            "rgb": [r, g, b],
        }

        # Apply ID color
        try:
            apply_id_color(artist, hex_color)
        except Exception:
            pass

    # Make non-artist elements reserved axes color
    axes_color = HITMAP_AXES_COLOR
    for ax in fig.axes:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(axes_color)
        ax.set_facecolor(HITMAP_BACKGROUND_COLOR)
        ax.tick_params(colors=axes_color, labelcolor=axes_color)
        ax.xaxis.label.set_color(axes_color)
        ax.yaxis.label.set_color(axes_color)
        ax.title.set_color(axes_color)
        if ax.get_legend():
            ax.get_legend().set_visible(False)

    fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

    # Save hitmap with bbox_inches='tight'
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=HITMAP_BACKGROUND_COLOR,
    )
    buf.seek(0)
    hitmap_img = Image.open(buf).convert("RGB")

    # Restore original properties
    for props in original_props:
        artist = props["artist"]
        try:
            if "color" in props and hasattr(artist, "set_color"):
                artist.set_color(props["color"])
            if "facecolor" in props and hasattr(artist, "set_facecolor"):
                artist.set_facecolor(props["facecolor"])
            if "edgecolor" in props and hasattr(artist, "set_edgecolor"):
                artist.set_edgecolor(props["edgecolor"])
            if "alpha" in props and hasattr(artist, "set_alpha"):
                artist.set_alpha(props["alpha"])
            if "antialiased" in props and hasattr(artist, "set_antialiased"):
                artist.set_antialiased(props["antialiased"])
        except Exception:
            pass

    # Restore axes properties
    for ax_props in original_ax_props:
        ax = ax_props["ax"]
        try:
            ax.set_facecolor(ax_props["facecolor"])
            for name, visible in ax_props["spines"].items():
                ax.spines[name].set_visible(visible)
            ax.set_xlabel(ax_props["xlabel"])
            ax.set_ylabel(ax_props["ylabel"])
            ax.set_title(ax_props["title"])
            if "legend_visible" in ax_props and ax.get_legend():
                ax.get_legend().set_visible(ax_props["legend_visible"])
        except Exception:
            pass

    fig.patch.set_facecolor(original_fig_facecolor)

    return hitmap_img, color_map


# EOF
