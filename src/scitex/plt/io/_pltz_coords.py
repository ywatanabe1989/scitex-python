#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/_pltz_coords.py

"""Coordinate adjustment utilities for pltz bundles."""

import copy
from typing import Any, Dict, List

__all__ = [
    "adjust_coords_for_offset",
    "adjust_path_data_for_offset",
    "adjust_path_data_for_crop",
]


def adjust_coords_for_offset(
    selectable_regions: Dict[str, Any],
    offset_left: float,
    offset_upper: float,
) -> Dict[str, Any]:
    """Adjust bbox_px coordinates by subtracting offset.

    Used when coordinates are already in PNG space (extracted at export DPI).

    Parameters
    ----------
    selectable_regions : dict
        The selectable_regions dict (already in PNG coords).
    offset_left : float
        Total offset from left edge to subtract.
    offset_upper : float
        Total offset from top edge to subtract.

    Returns
    -------
    dict
        selectable_regions with adjusted coordinates.
    """
    result = copy.deepcopy(selectable_regions)

    def adjust_bbox(bbox: List[float]) -> List[float]:
        """Subtract offset from [x0, y0, x1, y1] bbox."""
        return [
            bbox[0] - offset_left,
            bbox[1] - offset_upper,
            bbox[2] - offset_left,
            bbox[3] - offset_upper,
        ]

    for ax_region in result.get("axes", []):
        # Adjust title, xlabel, ylabel
        for key in ["title", "xlabel", "ylabel"]:
            if key in ax_region and "bbox_px" in ax_region[key]:
                ax_region[key]["bbox_px"] = adjust_bbox(ax_region[key]["bbox_px"])

        # Adjust xaxis elements
        if "xaxis" in ax_region:
            xaxis = ax_region["xaxis"]
            if xaxis.get("spine") and "bbox_px" in xaxis["spine"]:
                xaxis["spine"]["bbox_px"] = adjust_bbox(xaxis["spine"]["bbox_px"])
            for tick in xaxis.get("ticks", []):
                if "bbox_px" in tick:
                    tick["bbox_px"] = adjust_bbox(tick["bbox_px"])
            for label in xaxis.get("ticklabels", []):
                if "bbox_px" in label:
                    label["bbox_px"] = adjust_bbox(label["bbox_px"])

        # Adjust yaxis elements
        if "yaxis" in ax_region:
            yaxis = ax_region["yaxis"]
            if yaxis.get("spine") and "bbox_px" in yaxis["spine"]:
                yaxis["spine"]["bbox_px"] = adjust_bbox(yaxis["spine"]["bbox_px"])
            for tick in yaxis.get("ticks", []):
                if "bbox_px" in tick:
                    tick["bbox_px"] = adjust_bbox(tick["bbox_px"])
            for label in yaxis.get("ticklabels", []):
                if "bbox_px" in label:
                    label["bbox_px"] = adjust_bbox(label["bbox_px"])

        # Adjust legend
        if "legend" in ax_region:
            legend = ax_region["legend"]
            if "bbox_px" in legend:
                legend["bbox_px"] = adjust_bbox(legend["bbox_px"])
            for entry in legend.get("entries", []):
                if "bbox_px" in entry:
                    entry["bbox_px"] = adjust_bbox(entry["bbox_px"])

    return result


def adjust_path_data_for_offset(
    path_data: Dict[str, Any],
    offset_left: float,
    offset_upper: float,
) -> Dict[str, Any]:
    """Adjust path_data coordinates by subtracting offset.

    Used when coordinates are already in PNG space (extracted at export DPI).

    Parameters
    ----------
    path_data : dict
        The path_data dict (already in PNG coords).
    offset_left : float
        Total offset from left edge to subtract.
    offset_upper : float
        Total offset from top edge to subtract.

    Returns
    -------
    dict
        path_data with adjusted coordinates.
    """
    result = copy.deepcopy(path_data)

    # Adjust axes bbox_px
    for ax in result.get("axes", []):
        if "bbox_px" in ax:
            bbox = ax["bbox_px"]
            if isinstance(bbox, dict):
                bbox["x0"] = bbox.get("x0", 0) - offset_left
                bbox["y0"] = bbox.get("y0", 0) - offset_upper
                if "x1" in bbox:
                    bbox["x1"] = bbox["x1"] - offset_left
                if "y1" in bbox:
                    bbox["y1"] = bbox["y1"] - offset_upper

    # Adjust artists
    for artist in result.get("artists", []):
        if "bbox_px" in artist and artist["bbox_px"]:
            bbox = artist["bbox_px"]
            if isinstance(bbox, dict):
                bbox["x0"] = bbox.get("x0", 0) - offset_left
                bbox["y0"] = bbox.get("y0", 0) - offset_upper
                if "x1" in bbox:
                    bbox["x1"] = bbox["x1"] - offset_left
                if "y1" in bbox:
                    bbox["y1"] = bbox["y1"] - offset_upper

        # Adjust path_px points
        if "path_px" in artist and artist["path_px"]:
            artist["path_px"] = [
                [pt[0] - offset_left, pt[1] - offset_upper]
                for pt in artist["path_px"]
                if len(pt) >= 2
            ]

    return result


def adjust_path_data_for_crop(
    path_data: Dict[str, Any],
    offset_left: float,
    offset_upper: float,
) -> Dict[str, Any]:
    """Adjust path_data coordinates by subtracting crop offset.

    Parameters
    ----------
    path_data : dict
        The path_data dict from extract_path_data.
    offset_left : float
        Total offset from left edge.
    offset_upper : float
        Total offset from top edge.

    Returns
    -------
    dict
        path_data with adjusted coordinates.
    """
    result = copy.deepcopy(path_data)

    # Adjust axes bbox_px
    for ax in result.get("axes", []):
        if "bbox_px" in ax:
            bbox = ax["bbox_px"]
            if isinstance(bbox, dict):
                bbox["x0"] = bbox.get("x0", 0) - offset_left
                bbox["y0"] = bbox.get("y0", 0) - offset_upper
                if "x1" in bbox:
                    bbox["x1"] = bbox["x1"] - offset_left
                if "y1" in bbox:
                    bbox["y1"] = bbox["y1"] - offset_upper

    # Adjust artists
    for artist in result.get("artists", []):
        if "bbox_px" in artist and artist["bbox_px"]:
            bbox = artist["bbox_px"]
            if isinstance(bbox, dict):
                bbox["x0"] = bbox.get("x0", 0) - offset_left
                bbox["y0"] = bbox.get("y0", 0) - offset_upper
                if "x1" in bbox:
                    bbox["x1"] = bbox["x1"] - offset_left
                if "y1" in bbox:
                    bbox["y1"] = bbox["y1"] - offset_upper

        # Adjust path_px points
        if "path_px" in artist and artist["path_px"]:
            artist["path_px"] = [
                [pt[0] - offset_left, pt[1] - offset_upper]
                for pt in artist["path_px"]
                if len(pt) >= 2
            ]

    return result


# EOF
