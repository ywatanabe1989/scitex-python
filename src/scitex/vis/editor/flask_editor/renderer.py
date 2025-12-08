#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/renderer.py
"""Figure rendering for Flask editor."""

from typing import Dict, Any, Tuple
import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

from .plotter import plot_from_csv
from .bbox import extract_bboxes

# mm to pt conversion factor
MM_TO_PT = 2.83465


def render_preview_with_bboxes(csv_data, overrides: Dict[str, Any], axis_fontsize: int = 7
                               ) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render figure and return base64 PNG along with element bounding boxes.

    Args:
        csv_data: DataFrame containing CSV data
        overrides: Dictionary with override settings
        axis_fontsize: Default font size for axis labels

    Returns:
        tuple: (base64_image_data, bboxes_dict, image_size)
    """
    o = overrides

    # Dimensions
    dpi = o.get('dpi', 300)
    fig_size = o.get('fig_size', [3.15, 2.68])

    # Font sizes
    axis_fontsize = o.get('axis_fontsize', 7)
    tick_fontsize = o.get('tick_fontsize', 7)
    title_fontsize = o.get('title_fontsize', 8)

    # Line/axis thickness
    linewidth_pt = o.get('linewidth', 0.57)
    axis_width_pt = o.get('axis_width', 0.2) * MM_TO_PT
    tick_length_pt = o.get('tick_length', 0.8) * MM_TO_PT
    tick_width_pt = o.get('tick_width', 0.2) * MM_TO_PT
    tick_direction = o.get('tick_direction', 'out')
    x_n_ticks = o.get('x_n_ticks', o.get('n_ticks', 4))
    y_n_ticks = o.get('y_n_ticks', o.get('n_ticks', 4))
    hide_x_ticks = o.get('hide_x_ticks', False)
    hide_y_ticks = o.get('hide_y_ticks', False)

    transparent = o.get('transparent', True)

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    _apply_background(fig, ax, o, transparent)

    # Plot from CSV data
    if csv_data is not None:
        plot_from_csv(ax, csv_data, overrides, linewidth=linewidth_pt)
    else:
        ax.text(0.5, 0.5, "No plot data available\n(CSV not found)",
               ha='center', va='center', transform=ax.transAxes,
               fontsize=axis_fontsize)

    # Apply labels
    _apply_labels(ax, o, title_fontsize, axis_fontsize)

    # Tick styling
    _apply_tick_styling(ax, tick_fontsize, tick_length_pt, tick_width_pt, tick_direction,
                        x_n_ticks, y_n_ticks, hide_x_ticks, hide_y_ticks)

    # Apply grid, limits, spines
    _apply_style(ax, o, axis_width_pt)

    # Apply annotations
    _apply_annotations(ax, o, axis_fontsize)

    fig.tight_layout()

    # Get element bounding boxes BEFORE saving (need renderer)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Save to buffer first to get actual image size
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=transparent)
    buf.seek(0)

    # Get actual saved image dimensions
    img = Image.open(buf)
    img_width, img_height = img.size
    buf.seek(0)

    # Get bboxes
    bboxes = extract_bboxes(fig, ax, renderer, img_width, img_height)

    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_data, bboxes, {'width': img_width, 'height': img_height}


def _apply_background(fig, ax, o, transparent):
    """Apply background settings to figure."""
    if transparent:
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
    elif o.get('facecolor'):
        fig.patch.set_facecolor(o['facecolor'])
        ax.patch.set_facecolor(o['facecolor'])


def _apply_labels(ax, o, title_fontsize, axis_fontsize):
    """Apply title and axis labels."""
    if o.get('title'):
        ax.set_title(o['title'], fontsize=title_fontsize)
    if o.get('xlabel'):
        ax.set_xlabel(o['xlabel'], fontsize=axis_fontsize)
    if o.get('ylabel'):
        ax.set_ylabel(o['ylabel'], fontsize=axis_fontsize)


def _apply_tick_styling(ax, tick_fontsize, tick_length_pt, tick_width_pt, tick_direction,
                        x_n_ticks, y_n_ticks, hide_x_ticks, hide_y_ticks):
    """Apply tick styling to axes."""
    ax.tick_params(
        axis='both',
        labelsize=tick_fontsize,
        length=tick_length_pt,
        width=tick_width_pt,
        direction=tick_direction,
    )

    if hide_x_ticks:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_n_ticks))
    if hide_y_ticks:
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_n_ticks))


def _apply_style(ax, o, axis_width_pt):
    """Apply grid, axis limits, and spine settings."""
    if o.get('grid'):
        ax.grid(True, linewidth=axis_width_pt, alpha=0.3)

    if o.get('xlim'):
        ax.set_xlim(o['xlim'])
    if o.get('ylim'):
        ax.set_ylim(o['ylim'])

    if o.get('hide_top_spine', True):
        ax.spines['top'].set_visible(False)
    if o.get('hide_right_spine', True):
        ax.spines['right'].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_width_pt)


def _apply_annotations(ax, o, axis_fontsize):
    """Apply text annotations to figure."""
    for annot in o.get('annotations', []):
        if annot.get('type') == 'text':
            ax.text(
                annot.get('x', 0.5),
                annot.get('y', 0.5),
                annot.get('text', ''),
                transform=ax.transAxes,
                fontsize=annot.get('fontsize', axis_fontsize),
            )


# EOF
