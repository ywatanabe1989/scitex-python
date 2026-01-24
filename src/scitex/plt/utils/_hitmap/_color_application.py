#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/_color_application.py

"""
Color application and restoration functions for hitmap generation.

This module provides functions to apply unique ID colors to matplotlib artists
and restore original colors after hitmap rendering.
"""

from typing import Any, Dict, List, Tuple

from ._artist_extraction import get_all_artists, get_all_artists_with_groups
from ._color_conversion import id_to_rgb
from ._constants import HITMAP_AXES_COLOR, HITMAP_BACKGROUND_COLOR

__all__ = [
    "apply_id_color",
    "apply_hitmap_colors",
    "restore_original_colors",
    "prepare_hitmap_figure",
    "restore_figure_props",
]


def apply_id_color(artist, hex_color: str):
    """Apply ID color to an artist, handling different artist types."""
    if hasattr(artist, "set_color"):
        artist.set_color(hex_color)
        if hasattr(artist, "set_antialiased"):
            artist.set_antialiased(False)

    elif hasattr(artist, "set_facecolor"):
        artist.set_facecolor(hex_color)
        if hasattr(artist, "set_edgecolor"):
            artist.set_edgecolor(hex_color)
        if hasattr(artist, "set_alpha"):
            artist.set_alpha(1.0)
        if hasattr(artist, "set_antialiased"):
            artist.set_antialiased(False)

    # Handle BarContainer
    if hasattr(artist, "patches"):
        for patch in artist.patches:
            patch.set_facecolor(hex_color)
            patch.set_edgecolor(hex_color)
            if hasattr(patch, "set_antialiased"):
                patch.set_antialiased(False)


def apply_hitmap_colors(
    fig,
    include_text: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Apply unique ID colors to data elements in a figure.

    Also detects logical groups (histogram, bar_series, etc.) and assigns
    group_id to each element for hierarchical selection.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to modify.
    include_text : bool
        Whether to include text elements.

    Returns
    -------
    tuple
        (original_props, color_map, groups) where:
        - original_props: list of dicts with original artist properties
        - color_map: dict mapping ID to element info
        - groups: dict mapping group_id to logical group info
    """
    artists_with_groups, groups = get_all_artists_with_groups(fig, include_text)

    original_props = []
    color_map = {}

    for i, (artist, ax_idx, artist_type, group_id) in enumerate(artists_with_groups):
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
            if hasattr(artist, "get_linewidth"):
                props["linewidth"] = artist.get_linewidth()
        except Exception:
            pass
        original_props.append(props)

        # Build color map entry with group information
        label = ""
        if hasattr(artist, "get_label"):
            label = artist.get_label()
            if label.startswith("_"):
                label = f"{artist_type}_{i}"

        role = "physical" if group_id else "standalone"

        color_map[element_id] = {
            "id": element_id,
            "type": artist_type,
            "label": label,
            "axes_index": ax_idx,
            "rgb": [r, g, b],
            "group_id": group_id,
            "role": role,
        }

        # Apply ID color
        try:
            apply_id_color(artist, hex_color)
        except Exception:
            pass

    # Add RGB color to groups for logical selection
    group_id_start = len(artists_with_groups) + 1
    groups_with_colors = {}
    for i, (gid, ginfo) in enumerate(groups.items()):
        logical_id = group_id_start + i
        r, g, b = id_to_rgb(logical_id)

        # Find member element IDs
        member_ids = []
        for elem_id, elem_info in color_map.items():
            if elem_info.get("group_id") == gid:
                member_ids.append(elem_id)

        groups_with_colors[gid] = {
            "id": logical_id,
            "type": ginfo["type"],
            "label": ginfo["label"],
            "axes_index": ginfo["axes_index"],
            "rgb": [r, g, b],
            "role": "logical",
            "member_ids": member_ids,
            "member_count": ginfo["member_count"],
        }

    return original_props, color_map, groups_with_colors


def restore_original_colors(original_props: List[Dict[str, Any]]):
    """
    Restore original colors to artists after hitmap generation.

    Parameters
    ----------
    original_props : list
        List of dicts with original artist properties.
    """
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
            if "linewidth" in props and hasattr(artist, "set_linewidth"):
                artist.set_linewidth(props["linewidth"])
        except Exception:
            pass


def prepare_hitmap_figure(
    fig,
    include_text: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare a figure for hitmap rendering by coloring elements with unique IDs.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to prepare for hitmap rendering.
    include_text : bool
        Whether to include text elements.

    Returns
    -------
    tuple
        (color_map, original_props) where:
        - color_map: dict mapping ID to element info
        - original_props: list of dicts with original properties
    """
    artists = get_all_artists(fig, include_text)

    if not artists:
        return {}, []

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

        # Apply ID color
        try:
            apply_id_color(artist, hex_color)
        except Exception:
            pass

    # Hide non-artist elements
    axes_props = []
    for ax in fig.axes:
        ax_props = {
            "ax": ax,
            "grid_visible": (
                ax.xaxis.get_gridlines()[0].get_visible()
                if ax.xaxis.get_gridlines()
                else False
            ),
            "facecolor": ax.get_facecolor(),
            "spines_visible": {k: v.get_visible() for k, v in ax.spines.items()},
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "title": ax.get_title(),
            "legend_visible": (
                ax.get_legend().get_visible() if ax.get_legend() else None
            ),
            "tick_params": {},
        }
        axes_props.append(ax_props)

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(HITMAP_AXES_COLOR)
        ax.set_facecolor(HITMAP_BACKGROUND_COLOR)
        ax.tick_params(colors=HITMAP_AXES_COLOR, labelcolor=HITMAP_AXES_COLOR)
        ax.xaxis.label.set_color(HITMAP_AXES_COLOR)
        ax.yaxis.label.set_color(HITMAP_AXES_COLOR)
        ax.title.set_color(HITMAP_AXES_COLOR)
        if ax.get_legend():
            ax.get_legend().set_visible(False)

    original_props.append(
        {
            "type": "_figure_patch",
            "facecolor": fig.patch.get_facecolor(),
            "axes_props": axes_props,
        }
    )
    fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

    return color_map, original_props


def restore_figure_props(original_props: List[Dict[str, Any]]):
    """
    Restore figure properties after hitmap rendering.

    Parameters
    ----------
    original_props : list
        List of property dicts from prepare_hitmap_figure().
    """
    for props in original_props:
        if props.get("type") == "_figure_patch":
            if "axes_props" in props:
                for ax_props in props["axes_props"]:
                    ax = ax_props["ax"]
                    ax.set_facecolor(ax_props["facecolor"])
                    for spine_name, visible in ax_props["spines_visible"].items():
                        ax.spines[spine_name].set_visible(visible)
                    ax.set_xlabel(ax_props["xlabel"])
                    ax.set_ylabel(ax_props["ylabel"])
                    ax.set_title(ax_props["title"])
                    if ax_props["legend_visible"] is not None and ax.get_legend():
                        ax.get_legend().set_visible(ax_props["legend_visible"])
            continue

        artist = props.get("artist")
        if not artist:
            continue

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


# EOF
