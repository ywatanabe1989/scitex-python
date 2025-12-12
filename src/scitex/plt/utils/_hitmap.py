#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-12 (ywatanabe)"
# File: scitex/plt/utils/_hitmap.py

"""
Hit map generation utilities for interactive element selection.

This module provides functions to generate hit maps for matplotlib figures,
enabling pixel-perfect element selection in web editors and interactive tools.

Supported methods:
1. ID Colors: Single render with unique colors per element (~89ms)
2. Export Path Data: Extract geometry for client-side hit testing (~192ms)

Based on experimental results (see FIGZ_PLTZ_STATSZ.md):
- ID Colors is 33x faster than sequential per-element rendering
- Export Path Data supports reshape/zoom operations
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


def _to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(v) for v in obj]
    return obj


def get_all_artists(fig, include_text: bool = False) -> List[Tuple[Any, int, str]]:
    """
    Extract all selectable artists from a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to extract artists from.
    include_text : bool
        Whether to include text elements.

    Returns
    -------
    list of tuple
        List of (artist, axes_index, artist_type) tuples.
    """
    artists = []

    for ax_idx, ax in enumerate(fig.axes):
        # Lines (Line2D)
        for line in ax.get_lines():
            label = line.get_label()
            if not label.startswith('_'):  # Skip internal lines
                artists.append((line, ax_idx, 'line'))

        # Scatter plots (PathCollection)
        for coll in ax.collections:
            coll_type = type(coll).__name__
            if 'PathCollection' in coll_type:
                artists.append((coll, ax_idx, 'scatter'))
            elif 'PolyCollection' in coll_type or 'FillBetween' in coll_type:
                artists.append((coll, ax_idx, 'fill'))
            elif 'QuadMesh' in coll_type:
                artists.append((coll, ax_idx, 'mesh'))

        # Bars (Rectangle patches in containers)
        for container in ax.containers:
            if hasattr(container, 'patches') and container.patches:
                artists.append((container, ax_idx, 'bar'))

        # Individual patches (rectangles, circles, etc.)
        for patch in ax.patches:
            patch_type = type(patch).__name__
            if patch_type == 'Rectangle':
                artists.append((patch, ax_idx, 'rectangle'))
            elif patch_type in ('Circle', 'Ellipse'):
                artists.append((patch, ax_idx, 'circle'))
            elif patch_type == 'Polygon':
                artists.append((patch, ax_idx, 'polygon'))

        # Images
        for img in ax.images:
            artists.append((img, ax_idx, 'image'))

        # Text (optional)
        if include_text:
            for text in ax.texts:
                if text.get_text():
                    artists.append((text, ax_idx, 'text'))

    return artists


def _id_to_rgb(element_id: int) -> Tuple[int, int, int]:
    """Convert element ID to unique RGB color (24-bit, ~16.7M unique IDs)."""
    # Use all 24 bits for unique ID encoding
    # ID 0 is reserved for background (black)
    r = (element_id >> 16) & 0xFF
    g = (element_id >> 8) & 0xFF
    b = element_id & 0xFF
    return (r, g, b)


def _rgb_to_id(r: int, g: int, b: int) -> int:
    """Convert RGB color back to element ID."""
    return (r << 16) | (g << 8) | b


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
    Uses RGB 24-bit encoding for up to ~16.7 million unique element IDs.
    """
    import matplotlib.pyplot as plt
    import copy

    # Get all artists
    artists = get_all_artists(fig, include_text)

    if not artists:
        h = int(fig.get_figheight() * dpi)
        w = int(fig.get_figwidth() * dpi)
        return np.zeros((h, w), dtype=np.uint32), {}

    # Store original properties for restoration
    original_props = []

    # Build color map
    color_map = {}

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        element_id = i + 1
        # Use full RGB 24-bit encoding for unique ID colors
        r, g, b = _id_to_rgb(element_id)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # Store original properties
        props = {'artist': artist, 'type': artist_type}
        try:
            if hasattr(artist, 'get_color'):
                props['color'] = artist.get_color()
            if hasattr(artist, 'get_facecolor'):
                props['facecolor'] = artist.get_facecolor()
            if hasattr(artist, 'get_edgecolor'):
                props['edgecolor'] = artist.get_edgecolor()
            if hasattr(artist, 'get_alpha'):
                props['alpha'] = artist.get_alpha()
            if hasattr(artist, 'get_antialiased'):
                props['antialiased'] = artist.get_antialiased()
        except Exception:
            pass
        original_props.append(props)

        # Build color map entry (use element_id as key)
        label = ''
        if hasattr(artist, 'get_label'):
            label = artist.get_label()
            if label.startswith('_'):
                label = f'{artist_type}_{i}'

        color_map[element_id] = {
            'id': element_id,
            'type': artist_type,
            'label': label,
            'axes_index': ax_idx,
            'rgb': [r, g, b],
        }

        # Apply ID color and disable anti-aliasing
        try:
            _apply_id_color(artist, hex_color)
        except Exception:
            pass

    # Hide non-artist elements
    for ax in fig.axes:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('black')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        if ax.get_legend():
            ax.get_legend().set_visible(False)

    fig.patch.set_facecolor('black')

    # Render
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())
    # Convert RGB to element ID using 24-bit encoding
    hitmap = (img[:, :, 0].astype(np.uint32) << 16) | \
             (img[:, :, 1].astype(np.uint32) << 8) | \
             img[:, :, 2].astype(np.uint32)

    # Restore original properties
    for props in original_props:
        artist = props['artist']
        try:
            if 'color' in props and hasattr(artist, 'set_color'):
                artist.set_color(props['color'])
            if 'facecolor' in props and hasattr(artist, 'set_facecolor'):
                artist.set_facecolor(props['facecolor'])
            if 'edgecolor' in props and hasattr(artist, 'set_edgecolor'):
                artist.set_edgecolor(props['edgecolor'])
            if 'alpha' in props and hasattr(artist, 'set_alpha'):
                artist.set_alpha(props['alpha'])
            if 'antialiased' in props and hasattr(artist, 'set_antialiased'):
                artist.set_antialiased(props['antialiased'])
        except Exception:
            pass

    return hitmap, color_map


def _apply_id_color(artist, hex_color: str):
    """Apply ID color to an artist, handling different artist types."""
    artist_type = type(artist).__name__

    if hasattr(artist, 'set_color'):
        artist.set_color(hex_color)
        if hasattr(artist, 'set_antialiased'):
            artist.set_antialiased(False)

    elif hasattr(artist, 'set_facecolor'):
        artist.set_facecolor(hex_color)
        if hasattr(artist, 'set_edgecolor'):
            artist.set_edgecolor(hex_color)
        if hasattr(artist, 'set_alpha'):
            artist.set_alpha(1.0)
        if hasattr(artist, 'set_antialiased'):
            artist.set_antialiased(False)

    # Handle BarContainer
    if hasattr(artist, 'patches'):
        for patch in artist.patches:
            patch.set_facecolor(hex_color)
            patch.set_edgecolor(hex_color)
            if hasattr(patch, 'set_antialiased'):
                patch.set_antialiased(False)


def extract_path_data(
    fig,
    include_text: bool = False,
) -> Dict[str, Any]:
    """
    Extract path/geometry data for client-side hit testing.

    Extracts bounding boxes and path coordinates for all selectable elements,
    enabling JavaScript-based hit testing in web editors.

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
    Supports: resize/zoom (transform coordinates client-side)
    """
    fig.canvas.draw()  # Ensure transforms are computed

    artists = get_all_artists(fig, include_text)

    dpi = fig.dpi
    fig_width_px = int(fig.get_figwidth() * dpi)
    fig_height_px = int(fig.get_figheight() * dpi)

    export = {
        'figure': {
            'width_px': fig_width_px,
            'height_px': fig_height_px,
            'dpi': dpi,
        },
        'axes': [],
        'artists': [],
    }

    # Export axes info
    for ax in fig.axes:
        bbox = ax.get_position()
        export['axes'].append({
            'xlim': list(ax.get_xlim()),
            'ylim': list(ax.get_ylim()),
            'bbox_norm': {
                'x0': bbox.x0,
                'y0': bbox.y0,
                'x1': bbox.x1,
                'y1': bbox.y1,
            },
            'bbox_px': {
                'x0': int(bbox.x0 * fig_width_px),
                'y0': int((1 - bbox.y1) * fig_height_px),
                'x1': int(bbox.x1 * fig_width_px),
                'y1': int((1 - bbox.y0) * fig_height_px),
            },
        })

    # Export artist geometries
    renderer = fig.canvas.get_renderer()

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        artist_data = {
            'id': i,
            'type': artist_type,
            'axes_index': ax_idx,
            'label': '',
        }

        # Get label
        if hasattr(artist, 'get_label'):
            label = artist.get_label()
            artist_data['label'] = label if not label.startswith('_') else f'{artist_type}_{i}'

        # Get bounding box
        try:
            bbox = artist.get_window_extent(renderer)
            artist_data['bbox_px'] = {
                'x0': float(bbox.x0),
                'y0': float(fig_height_px - bbox.y1),  # Flip Y
                'x1': float(bbox.x1),
                'y1': float(fig_height_px - bbox.y0),
            }
        except Exception:
            artist_data['bbox_px'] = None

        # Extract type-specific geometry
        try:
            if artist_type == 'line' and hasattr(artist, 'get_xydata'):
                xy = artist.get_xydata()
                transform = artist.get_transform()
                xy_px = transform.transform(xy)
                # Flip Y and limit points for JSON size
                xy_px[:, 1] = fig_height_px - xy_px[:, 1]
                # Sample if too many points
                if len(xy_px) > 100:
                    indices = np.linspace(0, len(xy_px) - 1, 100, dtype=int)
                    xy_px = xy_px[indices]
                artist_data['path_px'] = xy_px.tolist()
                artist_data['linewidth'] = artist.get_linewidth()

            elif artist_type == 'scatter' and hasattr(artist, 'get_offsets'):
                offsets = artist.get_offsets()
                transform = artist.get_transform()
                offsets_px = transform.transform(offsets)
                offsets_px[:, 1] = fig_height_px - offsets_px[:, 1]
                artist_data['points_px'] = offsets_px.tolist()
                sizes = artist.get_sizes()
                artist_data['sizes'] = sizes.tolist() if len(sizes) > 0 else [36]

            elif artist_type == 'fill' and hasattr(artist, 'get_paths'):
                paths = artist.get_paths()
                if paths:
                    transform = artist.get_transform()
                    vertices = paths[0].vertices
                    vertices_px = transform.transform(vertices)
                    vertices_px[:, 1] = fig_height_px - vertices_px[:, 1]
                    # Sample if too many vertices
                    if len(vertices_px) > 100:
                        indices = np.linspace(0, len(vertices_px) - 1, 100, dtype=int)
                        vertices_px = vertices_px[indices]
                    artist_data['polygon_px'] = vertices_px.tolist()

            elif artist_type == 'bar' and hasattr(artist, 'patches'):
                bars = []
                ax = fig.axes[ax_idx]
                for patch in artist.patches:
                    # Get data coordinates
                    x_data = patch.get_x()
                    y_data = patch.get_y()
                    w_data = patch.get_width()
                    h_data = patch.get_height()
                    bars.append({
                        'x': x_data,
                        'y': y_data,
                        'width': w_data,
                        'height': h_data,
                    })
                artist_data['bars_data'] = bars

            elif artist_type == 'rectangle':
                artist_data['rectangle'] = {
                    'x': artist.get_x(),
                    'y': artist.get_y(),
                    'width': artist.get_width(),
                    'height': artist.get_height(),
                }

        except Exception as e:
            artist_data['error'] = str(e)

        export['artists'].append(artist_data)

    # Convert all numpy types to native Python for JSON serialization
    return _to_native(export)


def query_hitmap_neighborhood(
    hitmap: np.ndarray,
    x: int,
    y: int,
    color_map: Dict[int, Dict[str, Any]],
    radius: int = 2,
) -> List[Dict[str, Any]]:
    """
    Query hit map with neighborhood sampling for smart selection.

    Finds all element IDs in a neighborhood around the click point,
    enabling selection of overlapping elements and thin lines.

    Parameters
    ----------
    hitmap : np.ndarray
        Hit map array (uint32, element IDs from RGB encoding).
    x : int
        X coordinate (column) of click point.
    y : int
        Y coordinate (row) of click point.
    color_map : dict
        Mapping from element ID to element info.
    radius : int
        Sampling radius (e.g., 2 = 5×5 neighborhood).

    Returns
    -------
    list of dict
        List of element info dicts for all elements found in neighborhood,
        sorted by distance from click point (closest first).

    Notes
    -----
    Use cases:
    - Alt+Click to select objects underneath (lower z-order)
    - Click on thin lines that might be missed with exact pixel
    - Show candidate list when multiple elements overlap
    """
    h, w = hitmap.shape
    found_ids = set()
    id_distances = {}

    # Sample neighborhood
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                element_id = int(hitmap[ny, nx])
                if element_id > 0 and element_id in color_map:
                    found_ids.add(element_id)
                    # Track minimum distance for each ID
                    dist = abs(dx) + abs(dy)  # Manhattan distance
                    if element_id not in id_distances or dist < id_distances[element_id]:
                        id_distances[element_id] = dist

    # Sort by distance (closest first), then by ID for stability
    sorted_ids = sorted(found_ids, key=lambda eid: (id_distances[eid], eid))

    return [color_map[eid] for eid in sorted_ids]


def save_hitmap_png(hitmap: np.ndarray, path: str, color_map: Dict = None):
    """
    Save hit map as a PNG file (RGB encoding for 24-bit IDs).

    Parameters
    ----------
    hitmap : np.ndarray
        Hit map array (uint32, element IDs from 24-bit RGB encoding).
    path : str
        Output path for PNG file.
    color_map : dict, optional
        Color map for visualization.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # Convert 24-bit IDs back to RGB for PNG storage
    h, w = hitmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (hitmap >> 16) & 0xFF  # R
    rgb[:, :, 1] = (hitmap >> 8) & 0xFF   # G
    rgb[:, :, 2] = hitmap & 0xFF          # B

    # Save as RGB PNG (preserves exact ID values)
    img = Image.fromarray(rgb, mode='RGB')
    img.save(path)


# EOF
