#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/bbox.py
"""Bounding box extraction for figure elements."""

from typing import Dict, Any


def extract_bboxes(fig, ax, renderer, img_width: int, img_height: int) -> Dict[str, Any]:
    """Extract bounding boxes for all figure elements."""
    from matplotlib.transforms import Bbox

    # Get figure tight bbox in inches
    fig_bbox = fig.get_tightbbox(renderer)
    tight_x0 = fig_bbox.x0
    tight_y0 = fig_bbox.y0
    tight_width = fig_bbox.width
    tight_height = fig_bbox.height

    # bbox_inches='tight' adds pad_inches (default 0.1) around the tight bbox
    pad_inches = 0.1
    saved_width_inches = tight_width + 2 * pad_inches
    saved_height_inches = tight_height + 2 * pad_inches

    # Scale factors for converting inches to pixels
    scale_x = img_width / saved_width_inches
    scale_y = img_height / saved_height_inches

    bboxes = {}

    def get_element_bbox(element, name):
        """Get element bbox in image pixel coordinates."""
        try:
            bbox = element.get_window_extent(renderer)

            elem_x0_inches = bbox.x0 / fig.dpi
            elem_x1_inches = bbox.x1 / fig.dpi
            elem_y0_inches = bbox.y0 / fig.dpi
            elem_y1_inches = bbox.y1 / fig.dpi

            x0_rel = elem_x0_inches - tight_x0 + pad_inches
            x1_rel = elem_x1_inches - tight_x0 + pad_inches
            y0_rel = saved_height_inches - (elem_y1_inches - tight_y0 + pad_inches)
            y1_rel = saved_height_inches - (elem_y0_inches - tight_y0 + pad_inches)

            bboxes[name] = {
                'x0': max(0, int(x0_rel * scale_x)),
                'y0': max(0, int(y0_rel * scale_y)),
                'x1': min(img_width, int(x1_rel * scale_x)),
                'y1': min(img_height, int(y1_rel * scale_y)),
                'label': name.replace('_', ' ').title()
            }
        except Exception as e:
            print(f"Error getting bbox for {name}: {e}")

    def bbox_to_img_coords(bbox):
        """Convert matplotlib bbox to image pixel coordinates."""
        x0_inches = bbox.x0 / fig.dpi
        y0_inches = bbox.y0 / fig.dpi
        x1_inches = bbox.x1 / fig.dpi
        y1_inches = bbox.y1 / fig.dpi
        x0_rel = x0_inches - tight_x0 + pad_inches
        y0_rel = y0_inches - tight_y0 + pad_inches
        x1_rel = x1_inches - tight_x0 + pad_inches
        y1_rel = y1_inches - tight_y0 + pad_inches
        return {
            'x0': int(x0_rel * scale_x),
            'y0': int((saved_height_inches - y1_rel) * scale_y),
            'x1': int(x1_rel * scale_x),
            'y1': int((saved_height_inches - y0_rel) * scale_y),
        }

    # Get bboxes for title, labels
    if ax.title.get_text():
        get_element_bbox(ax.title, 'title')
    if ax.xaxis.label.get_text():
        get_element_bbox(ax.xaxis.label, 'xlabel')
    if ax.yaxis.label.get_text():
        get_element_bbox(ax.yaxis.label, 'ylabel')

    # Get axis bboxes
    _extract_axis_bboxes(ax, renderer, bboxes, bbox_to_img_coords, Bbox)

    # Get legend bbox
    legend = ax.get_legend()
    if legend:
        get_element_bbox(legend, 'legend')

    # Get trace (line) bboxes
    _extract_trace_bboxes(ax, fig, renderer, bboxes, get_element_bbox,
                          tight_x0, tight_y0, saved_height_inches, scale_x, scale_y, pad_inches)

    return bboxes


def _extract_axis_bboxes(ax, renderer, bboxes, bbox_to_img_coords, Bbox):
    """Extract bboxes for X and Y axis elements."""
    try:
        # X-axis: combine spine and tick labels into one bbox
        x_axis_bboxes = []
        for ticklabel in ax.xaxis.get_ticklabels():
            if ticklabel.get_visible():
                try:
                    tb = ticklabel.get_window_extent(renderer)
                    if tb.width > 0:
                        x_axis_bboxes.append(tb)
                except Exception:
                    pass
        for tick in ax.xaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                try:
                    tb = tick.tick1line.get_window_extent(renderer)
                    if tb.width > 0 or tb.height > 0:
                        x_axis_bboxes.append(tb)
                except Exception:
                    pass
        spine_bbox = ax.spines['bottom'].get_window_extent(renderer)
        if spine_bbox.width > 0:
            if x_axis_bboxes:
                tick_union = Bbox.union(x_axis_bboxes)
                constrained_spine = Bbox.from_extents(
                    tick_union.x0, spine_bbox.y0,
                    tick_union.x1, spine_bbox.y1
                )
                x_axis_bboxes.append(constrained_spine)
            else:
                x_axis_bboxes.append(spine_bbox)
        if x_axis_bboxes:
            combined = Bbox.union(x_axis_bboxes)
            bboxes['xaxis_ticks'] = bbox_to_img_coords(combined)
            bboxes['xaxis_ticks']['label'] = 'X Spine & Ticks'

        # Y-axis: combine spine and tick labels into one bbox
        y_axis_bboxes = []
        for ticklabel in ax.yaxis.get_ticklabels():
            if ticklabel.get_visible():
                try:
                    tb = ticklabel.get_window_extent(renderer)
                    if tb.width > 0:
                        y_axis_bboxes.append(tb)
                except Exception:
                    pass
        for tick in ax.yaxis.get_major_ticks():
            if tick.tick1line.get_visible():
                try:
                    tb = tick.tick1line.get_window_extent(renderer)
                    if tb.width > 0 or tb.height > 0:
                        y_axis_bboxes.append(tb)
                except Exception:
                    pass
        spine_bbox = ax.spines['left'].get_window_extent(renderer)
        if spine_bbox.height > 0:
            if y_axis_bboxes:
                tick_union = Bbox.union(y_axis_bboxes)
                constrained_spine = Bbox.from_extents(
                    spine_bbox.x0, tick_union.y0,
                    spine_bbox.x1, tick_union.y1
                )
                y_axis_bboxes.append(constrained_spine)
            else:
                y_axis_bboxes.append(spine_bbox)
        if y_axis_bboxes:
            combined = Bbox.union(y_axis_bboxes)
            padded = Bbox.from_extents(
                combined.x0 - 10,
                combined.y0 - 5,
                combined.x1 + 5,
                combined.y1 + 5
            )
            bboxes['yaxis_ticks'] = bbox_to_img_coords(padded)
            bboxes['yaxis_ticks']['label'] = 'Y Spine & Ticks'

    except Exception as e:
        print(f"Error getting axis bboxes: {e}")


def _extract_trace_bboxes(ax, fig, renderer, bboxes, get_element_bbox,
                          tight_x0, tight_y0, saved_height_inches, scale_x, scale_y, pad_inches):
    """Extract bboxes for trace (line) elements with proximity detection points."""
    for idx, line in enumerate(ax.get_lines()):
        try:
            label = line.get_label()
            if label.startswith('_'):
                continue
            get_element_bbox(line, f'trace_{idx}')
            if f'trace_{idx}' in bboxes:
                bboxes[f'trace_{idx}']['label'] = label or f'Trace {idx}'
                bboxes[f'trace_{idx}']['trace_idx'] = idx

                # Get line data points in pixel coordinates for proximity detection
                xdata, ydata = line.get_xdata(), line.get_ydata()
                if len(xdata) > 0:
                    transform = ax.transData
                    points_display = transform.transform(list(zip(xdata, ydata)))

                    points_img = []
                    for px, py in points_display:
                        px_inches = px / fig.dpi
                        py_inches = py / fig.dpi
                        x_rel = px_inches - tight_x0 + pad_inches
                        y_rel = saved_height_inches - (py_inches - tight_y0 + pad_inches)
                        x_img = int(x_rel * scale_x)
                        y_img = int(y_rel * scale_y)
                        points_img.append([x_img, y_img])

                    # Downsample points if too many
                    if len(points_img) > 100:
                        step = len(points_img) // 100
                        points_img = points_img[::step]

                    bboxes[f'trace_{idx}']['points'] = points_img
        except Exception as e:
            print(f"Error getting trace bbox: {e}")


# EOF
