#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_bbox_extraction.py

"""Bounding box extraction from pltz metadata."""

from typing import Any, Dict, Optional

__all__ = ["extract_bboxes_from_metadata"]


def extract_bboxes_from_metadata(
    metadata: Dict[str, Any],
    display_width: Optional[float] = None,
    display_height: Optional[float] = None,
) -> Dict[str, Any]:
    """Extract element bounding boxes from pltz metadata.

    Builds bboxes from selectable_regions in the metadata for click detection.
    This allows the editor to highlight elements when clicked.

    Coordinate system (new layered format):
    - selectable_regions bbox_px: Already in final image space (figure_px)
    - Display size: Actual displayed image size (PNG pixels or SVG viewBox)
    - Scale = display_size / figure_px (usually 1:1, but may differ for scaled display)

    Parameters
    ----------
    metadata : dict
        The pltz JSON metadata containing selectable_regions
    display_width : float, optional
        Actual display image width (from PNG size or SVG viewBox)
    display_height : float, optional
        Actual display image height (from PNG size or SVG viewBox)

    Returns
    -------
    dict
        Mapping of element IDs to their bounding box coordinates (in display pixels)
    """
    bboxes = {}
    selectable = metadata.get("selectable_regions", {})

    # Figure dimensions from new layered format (bbox_px are in this space)
    figure_px = metadata.get("figure_px", [])
    if isinstance(figure_px, list) and len(figure_px) >= 2:
        fig_width = figure_px[0]
        fig_height = figure_px[1]
    else:
        # Fallback for old format: try hit_regions.path_data.figure
        hit_regions = metadata.get("hit_regions", {})
        path_data = hit_regions.get("path_data", {})
        orig_fig = path_data.get("figure", {})
        fig_width = orig_fig.get("width_px", 944)
        fig_height = orig_fig.get("height_px", 803)

    # Use actual display dimensions if provided, else use figure_px
    if display_width is None:
        display_width = fig_width
    if display_height is None:
        display_height = fig_height

    # Scale factor: display / figure_px
    scale_x = display_width / fig_width if fig_width > 0 else 1
    scale_y = display_height / fig_height if fig_height > 0 else 1

    def to_display_bbox(bbox, is_list=True):
        """Convert bbox to display pixels."""
        if is_list:
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            x0 = bbox.get("x0", 0)
            y0 = bbox.get("y0", 0)
            x1 = bbox.get("x1", bbox.get("x0", 0) + bbox.get("width", 0))
            y1 = bbox.get("y1", bbox.get("y0", 0) + bbox.get("height", 0))

        disp_x0 = x0 * scale_x
        disp_x1 = x1 * scale_x
        disp_y0 = y0 * scale_y
        disp_y1 = y1 * scale_y

        return {
            "x0": disp_x0,
            "y0": disp_y0,
            "x1": disp_x1,
            "y1": disp_y1,
            "x": disp_x0,
            "y": disp_y0,
            "width": disp_x1 - disp_x0,
            "height": disp_y1 - disp_y0,
        }

    # Extract from selectable_regions.axes
    axes_regions = selectable.get("axes", [])
    for ax_idx, ax in enumerate(axes_regions):
        ax_key = f"ax_{ax_idx:02d}"

        # Title
        title = ax.get("title", {})
        if title and "bbox_px" in title:
            bbox_disp = to_display_bbox(title["bbox_px"])
            bboxes[f"{ax_key}_title"] = {
                **bbox_disp,
                "type": "title",
                "text": title.get("text", ""),
            }

        # X label
        xlabel = ax.get("xlabel", {})
        if xlabel and "bbox_px" in xlabel:
            bbox_disp = to_display_bbox(xlabel["bbox_px"])
            bboxes[f"{ax_key}_xlabel"] = {
                **bbox_disp,
                "type": "xlabel",
                "text": xlabel.get("text", ""),
            }

        # Y label
        ylabel = ax.get("ylabel", {})
        if ylabel and "bbox_px" in ylabel:
            bbox_disp = to_display_bbox(ylabel["bbox_px"])
            bboxes[f"{ax_key}_ylabel"] = {
                **bbox_disp,
                "type": "ylabel",
                "text": ylabel.get("text", ""),
            }

        # Legend
        legend = ax.get("legend", {})
        if legend and "bbox_px" in legend:
            bbox_disp = to_display_bbox(legend["bbox_px"])
            bboxes[f"{ax_key}_legend"] = {
                **bbox_disp,
                "type": "legend",
            }

        # X-axis spine
        xaxis = ax.get("xaxis", {})
        if xaxis:
            spine = xaxis.get("spine", {})
            if spine and "bbox_px" in spine:
                bbox_disp = to_display_bbox(spine["bbox_px"])
                bboxes[f"{ax_key}_xaxis_spine"] = {
                    **bbox_disp,
                    "type": "xaxis",
                }

        # Y-axis spine
        yaxis = ax.get("yaxis", {})
        if yaxis:
            spine = yaxis.get("spine", {})
            if spine and "bbox_px" in spine:
                bbox_disp = to_display_bbox(spine["bbox_px"])
                bboxes[f"{ax_key}_yaxis_spine"] = {
                    **bbox_disp,
                    "type": "yaxis",
                }

    # Extract traces from artists
    artists = metadata.get("artists", [])
    if not artists:
        hit_regions = metadata.get("hit_regions", {})
        path_data = hit_regions.get("path_data", {})
        artists = path_data.get("artists", [])

    for artist in artists:
        artist_id = artist.get("id", 0)
        artist_type = artist.get("type", "line")
        bbox_px = artist.get("bbox_px", {})
        if bbox_px:
            bbox_disp = to_display_bbox(bbox_px, is_list=False)
            trace_entry = {
                **bbox_disp,
                "type": artist_type,
                "label": artist.get("label", f"Trace {artist_id}"),
                "element_type": artist_type,
            }

            path_px = artist.get("path_px", [])
            if path_px:
                scaled_points = [
                    [pt[0] * scale_x, pt[1] * scale_y] for pt in path_px if len(pt) >= 2
                ]
                trace_entry["points"] = scaled_points

            bboxes[f"trace_{artist_id}"] = trace_entry

    bboxes["_meta"] = {
        "display_width": display_width,
        "display_height": display_height,
        "figure_px_width": fig_width,
        "figure_px_height": fig_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }

    return bboxes


# EOF
