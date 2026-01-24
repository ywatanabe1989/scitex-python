#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/_path_extraction.py

"""
Path data and selectable regions extraction for hitmap generation.

This module provides functions to extract path/geometry data and selectable
regions for client-side hit testing.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ._artist_extraction import get_all_artists
from ._constants import to_native

__all__ = [
    "extract_path_data",
    "extract_selectable_regions",
]


def extract_path_data(
    fig,
    include_text: bool = False,
) -> Dict[str, Any]:
    """
    Extract path/geometry data for client-side hit testing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to extract data from.
    include_text : bool
        Whether to include text elements.

    Returns
    -------
    dict
        Exported data structure with figure info and artist geometries.

    Notes
    -----
    Performance: ~192ms extraction, ~0.01ms client-side queries
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.canvas.draw()

    artists = get_all_artists(fig, include_text)

    dpi = fig.dpi
    fig_width_px = int(fig.get_figwidth() * dpi)
    fig_height_px = int(fig.get_figheight() * dpi)

    export = {
        "figure": {
            "width_px": fig_width_px,
            "height_px": fig_height_px,
            "dpi": dpi,
        },
        "axes": [],
        "artists": [],
    }

    # Export axes info
    for ax in fig.axes:
        bbox = ax.get_position()
        export["axes"].append(
            {
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "bbox_norm": {
                    "x0": bbox.x0,
                    "y0": bbox.y0,
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                },
                "bbox_px": {
                    "x0": int(bbox.x0 * fig_width_px),
                    "y0": int((1 - bbox.y1) * fig_height_px),
                    "x1": int(bbox.x1 * fig_width_px),
                    "y1": int((1 - bbox.y0) * fig_height_px),
                },
            }
        )

    # Export artist geometries
    renderer = fig.canvas.get_renderer()

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        artist_data = {
            "id": i,
            "type": artist_type,
            "axes_index": ax_idx,
            "label": "",
        }

        # Get label
        if hasattr(artist, "get_label"):
            label = artist.get_label()
            artist_data["label"] = (
                label if not label.startswith("_") else f"{artist_type}_{i}"
            )

        # Get bounding box
        try:
            bbox = artist.get_window_extent(renderer)
            artist_data["bbox_px"] = {
                "x0": float(bbox.x0),
                "y0": float(fig_height_px - bbox.y1),
                "x1": float(bbox.x1),
                "y1": float(fig_height_px - bbox.y0),
            }
        except Exception:
            artist_data["bbox_px"] = None

        # Extract type-specific geometry
        try:
            if artist_type == "line" and hasattr(artist, "get_xydata"):
                xy = artist.get_xydata()
                transform = artist.get_transform()
                xy_px = transform.transform(xy)
                xy_px[:, 1] = fig_height_px - xy_px[:, 1]
                if len(xy_px) > 100:
                    indices = np.linspace(0, len(xy_px) - 1, 100, dtype=int)
                    xy_px = xy_px[indices]
                artist_data["path_px"] = xy_px.tolist()
                artist_data["linewidth"] = artist.get_linewidth()

            elif artist_type == "scatter" and hasattr(artist, "get_offsets"):
                offsets = artist.get_offsets()
                transform = artist.get_transform()
                offsets_px = transform.transform(offsets)
                offsets_px[:, 1] = fig_height_px - offsets_px[:, 1]
                artist_data["points_px"] = offsets_px.tolist()
                sizes = artist.get_sizes()
                artist_data["sizes"] = sizes.tolist() if len(sizes) > 0 else [36]

            elif artist_type == "fill" and hasattr(artist, "get_paths"):
                paths = artist.get_paths()
                if paths:
                    transform = artist.get_transform()
                    vertices = paths[0].vertices
                    vertices_px = transform.transform(vertices)
                    vertices_px[:, 1] = fig_height_px - vertices_px[:, 1]
                    if len(vertices_px) > 100:
                        indices = np.linspace(0, len(vertices_px) - 1, 100, dtype=int)
                        vertices_px = vertices_px[indices]
                    artist_data["polygon_px"] = vertices_px.tolist()

            elif artist_type == "bar" and hasattr(artist, "patches"):
                bars = []
                for patch in artist.patches:
                    x_data = patch.get_x()
                    y_data = patch.get_y()
                    w_data = patch.get_width()
                    h_data = patch.get_height()
                    bars.append(
                        {
                            "x": x_data,
                            "y": y_data,
                            "width": w_data,
                            "height": h_data,
                        }
                    )
                artist_data["bars_data"] = bars

            elif artist_type == "rectangle":
                artist_data["rectangle"] = {
                    "x": artist.get_x(),
                    "y": artist.get_y(),
                    "width": artist.get_width(),
                    "height": artist.get_height(),
                }

        except Exception as e:
            artist_data["error"] = str(e)

        export["artists"].append(artist_data)

    return to_native(export)


def extract_selectable_regions(fig) -> Dict[str, Any]:
    """
    Extract bounding boxes for axis/annotation elements (non-data elements).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to extract regions from.

    Returns
    -------
    dict
        Dictionary with selectable regions.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.canvas.draw()

    dpi = fig.dpi
    fig_width_px = int(fig.get_figwidth() * dpi)
    fig_height_px = int(fig.get_figheight() * dpi)

    renderer = fig.canvas.get_renderer()

    def get_bbox_px(artist) -> Optional[List[float]]:
        """Get bounding box in pixels (y-flipped for image coordinates)."""
        try:
            bbox = artist.get_window_extent(renderer)
            if bbox.width > 0 and bbox.height > 0:
                return [
                    float(bbox.x0),
                    float(fig_height_px - bbox.y1),
                    float(bbox.x1),
                    float(fig_height_px - bbox.y0),
                ]
        except Exception:
            pass
        return None

    def get_text_info(text_artist) -> Optional[Dict[str, Any]]:
        """Extract text element info with bounding box."""
        if text_artist is None:
            return None
        text = text_artist.get_text()
        if not text or not text.strip():
            return None
        bbox = get_bbox_px(text_artist)
        if bbox is None:
            return None
        return {
            "bbox_px": bbox,
            "text": text,
            "fontsize": text_artist.get_fontsize(),
            "color": text_artist.get_color(),
        }

    regions = {"axes": []}

    for ax_idx, ax in enumerate(fig.axes):
        ax_region = {"index": ax_idx}

        # Title
        title_info = get_text_info(ax.title)
        if title_info:
            ax_region["title"] = title_info

        # X label
        xlabel_info = get_text_info(ax.xaxis.label)
        if xlabel_info:
            ax_region["xlabel"] = xlabel_info

        # Y label
        ylabel_info = get_text_info(ax.yaxis.label)
        if ylabel_info:
            ax_region["ylabel"] = ylabel_info

        # X axis elements
        xaxis_info = {"spine": None, "ticks": [], "ticklabels": []}

        if ax.spines["bottom"].get_visible():
            spine_bbox = get_bbox_px(ax.spines["bottom"])
            if spine_bbox:
                xaxis_info["spine"] = {"bbox_px": spine_bbox}

        for tick in ax.xaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                tick_bbox = get_bbox_px(tick.tick1line)
                if tick_bbox:
                    xaxis_info["ticks"].append(
                        {
                            "bbox_px": tick_bbox,
                            "position": (
                                float(tick.get_loc())
                                if hasattr(tick, "get_loc")
                                else None
                            ),
                        }
                    )

            if tick.label1.get_visible():
                label_info = get_text_info(tick.label1)
                if label_info:
                    xaxis_info["ticklabels"].append(label_info)

        if xaxis_info["spine"] or xaxis_info["ticks"] or xaxis_info["ticklabels"]:
            ax_region["xaxis"] = xaxis_info

        # Y axis elements
        yaxis_info = {"spine": None, "ticks": [], "ticklabels": []}

        if ax.spines["left"].get_visible():
            spine_bbox = get_bbox_px(ax.spines["left"])
            if spine_bbox:
                yaxis_info["spine"] = {"bbox_px": spine_bbox}

        for tick in ax.yaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                tick_bbox = get_bbox_px(tick.tick1line)
                if tick_bbox:
                    yaxis_info["ticks"].append(
                        {
                            "bbox_px": tick_bbox,
                            "position": (
                                float(tick.get_loc())
                                if hasattr(tick, "get_loc")
                                else None
                            ),
                        }
                    )

            if tick.label1.get_visible():
                label_info = get_text_info(tick.label1)
                if label_info:
                    yaxis_info["ticklabels"].append(label_info)

        if yaxis_info["spine"] or yaxis_info["ticks"] or yaxis_info["ticklabels"]:
            ax_region["yaxis"] = yaxis_info

        # Legend
        legend = ax.get_legend()
        if legend and legend.get_visible():
            legend_info = {"bbox_px": None, "entries": []}

            legend_bbox = get_bbox_px(legend)
            if legend_bbox:
                legend_info["bbox_px"] = legend_bbox

            for text in legend.get_texts():
                entry_info = get_text_info(text)
                if entry_info:
                    legend_info["entries"].append(entry_info)

            try:
                handles = legend.legendHandles
                for i, handle in enumerate(handles):
                    handle_bbox = get_bbox_px(handle)
                    if handle_bbox and i < len(legend_info["entries"]):
                        legend_info["entries"][i]["handle_bbox_px"] = handle_bbox
            except Exception:
                pass

            if legend_info["bbox_px"] or legend_info["entries"]:
                ax_region["legend"] = legend_info

        regions["axes"].append(ax_region)

    return to_native(regions)


# EOF
