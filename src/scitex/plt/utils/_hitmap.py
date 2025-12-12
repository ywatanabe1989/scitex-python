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

Reserved colors:
- Black (#000000, ID=0): Background/no element
- Dark gray (#010101, ID=65793): Non-selectable axes elements (spines, labels, ticks)
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

__all__ = [
    'get_all_artists',
    'generate_hitmap_id_colors',
    'extract_path_data',
    'query_hitmap_neighborhood',
    'save_hitmap_png',
    'apply_hitmap_colors',
    'restore_original_colors',
    'generate_hitmap_with_bbox_tight',
    'HITMAP_BACKGROUND_COLOR',
    'HITMAP_AXES_COLOR',
    '_rgb_to_id_lookup',
]


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
    """
    Convert element ID to unique, human-readable RGB color using hash-based generation.

    Uses a hash function to generate visually distinct colors that are:
    1. Deterministic (same ID always gives same color)
    2. Visually distinct (spread across the color space)
    3. Bright and saturated (easy to see)

    The first 12 elements use a hand-picked palette for maximum distinctness.
    Beyond that, uses hash-based HSV generation with high saturation.

    Parameters
    ----------
    element_id : int
        Element ID (1-based). ID 0 is reserved for background.

    Returns
    -------
    tuple
        (R, G, B) values (0-255)

    Notes
    -----
    The hash ensures:
    - Same element_id always maps to the same color
    - Colors are well-distributed across the spectrum
    - Avoids dark colors (reserved for background/axes)
    """
    import colorsys
    import hashlib

    if element_id <= 0:
        return (0, 0, 0)  # Background

    # Hand-picked palette for first 12 elements (most common case)
    # These are maximally distinct primary/secondary colors
    DISTINCT_COLORS = [
        (255, 0, 0),      # 1: Red
        (0, 200, 0),      # 2: Green (slightly darker for visibility)
        (0, 100, 255),    # 3: Blue (lighter for visibility)
        (255, 200, 0),    # 4: Yellow/Gold
        (255, 0, 200),    # 5: Magenta/Pink
        (0, 220, 220),    # 6: Cyan
        (255, 100, 0),    # 7: Orange
        (150, 0, 255),    # 8: Purple
        (0, 255, 100),    # 9: Spring Green
        (255, 100, 150),  # 10: Salmon/Rose
        (100, 255, 0),    # 11: Lime
        (100, 150, 255),  # 12: Sky Blue
    ]

    if element_id <= len(DISTINCT_COLORS):
        return DISTINCT_COLORS[element_id - 1]

    # For IDs > 12, use hash-based color generation
    # Hash the ID to get a pseudo-random but deterministic value
    hash_bytes = hashlib.md5(str(element_id).encode()).digest()

    # Use hash bytes to generate HSV values
    # Hue: full range (0-1) for variety
    hue = int.from_bytes(hash_bytes[0:2], 'big') / 65535.0

    # Saturation: high (0.7-1.0) for vivid colors
    saturation = 0.7 + (int.from_bytes(hash_bytes[2:3], 'big') / 255.0) * 0.3

    # Value: high (0.75-1.0) to avoid dark colors
    value = 0.75 + (int.from_bytes(hash_bytes[3:4], 'big') / 255.0) * 0.25

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _rgb_to_id_lookup(r: int, g: int, b: int, color_map: dict) -> int:
    """
    Convert RGB color back to element ID using the color map.

    Since we use human-readable colors, we need to look up in the map.

    Parameters
    ----------
    r, g, b : int
        RGB values (0-255)
    color_map : dict
        Color map from generate_hitmap_id_colors (maps ID -> info with 'rgb' key)

    Returns
    -------
    int
        Element ID, or 0 if not found
    """
    rgb = [r, g, b]
    for element_id, info in color_map.items():
        if info.get('rgb') == rgb:
            return element_id
    return 0


# Reserved colors for hitmap (human-readable)
HITMAP_BACKGROUND_COLOR = '#1a1a1a'  # Dark gray (not pure black, easier to see)
HITMAP_AXES_COLOR = '#404040'  # Medium gray (non-selectable axes elements)


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

    # Make non-artist elements the reserved axes color (NOT black)
    # This distinguishes axes from background while making them non-selectable
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


def _prepare_hitmap_figure(
    fig,
    include_text: bool = False,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare a figure for hitmap rendering by coloring elements with unique IDs.

    This function modifies the figure in-place by:
    1. Assigning unique RGB colors to each artist (24-bit ID encoding)
    2. Hiding non-selectable elements (axes, spines, grid, etc.)
    3. Setting background to black (ID = 0)

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
        - original_props: list of dicts with original artist properties (for restoration)

    Notes
    -----
    After calling this function, you can render the figure using savefig()
    with bbox_inches='tight' to get a pixel-perfect hitmap.
    Call _restore_figure_props(original_props) to restore the figure.
    """
    artists = get_all_artists(fig, include_text)

    if not artists:
        return {}, []

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

        # Build color map entry
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

    # Hide non-artist elements (we need to save these for restoration too)
    axes_props = []
    for ax in fig.axes:
        ax_props = {
            'ax': ax,
            'grid_visible': ax.xaxis.get_gridlines()[0].get_visible() if ax.xaxis.get_gridlines() else False,
            'facecolor': ax.get_facecolor(),
            'spines_visible': {k: v.get_visible() for k, v in ax.spines.items()},
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'title': ax.get_title(),
            'legend_visible': ax.get_legend().get_visible() if ax.get_legend() else None,
            'tick_params': {},  # Complex to save/restore, skip for now
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

    # Save figure background
    original_props.append({
        'type': '_figure_patch',
        'facecolor': fig.patch.get_facecolor(),
        'axes_props': axes_props,
    })
    fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

    return color_map, original_props


def _restore_figure_props(original_props: List[Dict[str, Any]]):
    """
    Restore figure properties after hitmap rendering.

    Parameters
    ----------
    original_props : list
        List of property dicts from _prepare_hitmap_figure().
    """
    for props in original_props:
        if props.get('type') == '_figure_patch':
            # Restore axes and figure background
            if 'axes_props' in props:
                for ax_props in props['axes_props']:
                    ax = ax_props['ax']
                    ax.set_facecolor(ax_props['facecolor'])
                    for spine_name, visible in ax_props['spines_visible'].items():
                        ax.spines[spine_name].set_visible(visible)
                    ax.set_xlabel(ax_props['xlabel'])
                    ax.set_ylabel(ax_props['ylabel'])
                    ax.set_title(ax_props['title'])
                    if ax_props['legend_visible'] is not None and ax.get_legend():
                        ax.get_legend().set_visible(ax_props['legend_visible'])
            continue

        artist = props.get('artist')
        if not artist:
            continue

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


def apply_hitmap_colors(
    fig,
    include_text: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Apply unique ID colors to data elements in a figure.

    This function modifies data elements (lines, patches, etc.) to have unique
    RGB colors for hit testing, while keeping axes/spines/labels unchanged.
    This preserves the bbox_inches='tight' bounding box calculation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to modify.
    include_text : bool
        Whether to include text elements.

    Returns
    -------
    tuple
        (original_props, color_map) where:
        - original_props: list of dicts with original artist properties for restoration
        - color_map: dict mapping ID to element info
    """
    artists = get_all_artists(fig, include_text)

    original_props = []
    color_map = {}

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        element_id = i + 1
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
            if hasattr(artist, 'get_linewidth'):
                props['linewidth'] = artist.get_linewidth()
        except Exception:
            pass
        original_props.append(props)

        # Build color map entry
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

        # Apply ID color
        try:
            _apply_id_color(artist, hex_color)
        except Exception:
            pass

    return original_props, color_map


def restore_original_colors(original_props: List[Dict[str, Any]]):
    """
    Restore original colors to artists after hitmap generation.

    Parameters
    ----------
    original_props : list
        List of dicts with original artist properties (from apply_hitmap_colors).
    """
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
            if 'linewidth' in props and hasattr(artist, 'set_linewidth'):
                artist.set_linewidth(props['linewidth'])
        except Exception:
            pass


def generate_hitmap_with_bbox_tight(
    fig,
    dpi: int = 150,
    include_text: bool = False,
) -> Tuple['Image.Image', Dict[int, Dict[str, Any]]]:
    """
    Generate a hitmap image with bbox_inches='tight' to match PNG output.

    This function generates a hitmap that exactly matches the PNG saved with
    bbox_inches='tight'. The key insight is that both PNG and hitmap must use
    the same savefig parameters to have identical cropping.

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
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    import tempfile

    # Get all artists
    artists = get_all_artists(fig, include_text)

    if not artists:
        # Return empty black image with same size as PNG would have
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # Create black image of same size
        black_img = Image.new('RGB', img.size, (0, 0, 0))
        return black_img, {}

    # Store original properties for restoration
    original_props = []
    original_ax_props = []

    # Store original axes properties
    for ax in fig.axes:
        ax_props = {
            'ax': ax,
            'facecolor': ax.get_facecolor(),
            'grid_visible': ax.xaxis.get_gridlines()[0].get_visible() if ax.xaxis.get_gridlines() else False,
            'spines': {name: spine.get_visible() for name, spine in ax.spines.items()},
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'title': ax.get_title(),
            'tick_params': {
                'left': ax.yaxis.get_tick_params()['left'] if hasattr(ax.yaxis.get_tick_params(), '__getitem__') else True,
                'bottom': ax.xaxis.get_tick_params()['bottom'] if hasattr(ax.xaxis.get_tick_params(), '__getitem__') else True,
            },
        }
        if ax.get_legend():
            ax_props['legend_visible'] = ax.get_legend().get_visible()
        original_ax_props.append(ax_props)

    original_fig_facecolor = fig.patch.get_facecolor()

    # Build color map
    color_map = {}

    for i, (artist, ax_idx, artist_type) in enumerate(artists):
        element_id = i + 1
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

        # Build color map entry
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

        # Apply ID color
        try:
            _apply_id_color(artist, hex_color)
        except Exception:
            pass

    # Make non-artist elements a reserved "axes" color (NOT black/invisible)
    # This preserves bbox_inches='tight' bounds while distinguishing from background
    # Use HITMAP_AXES_COLOR (#010101) which maps to ID 65793 (non-selectable)
    axes_color = HITMAP_AXES_COLOR
    for ax in fig.axes:
        ax.grid(False)
        # Make spines the reserved axes color instead of black
        for spine in ax.spines.values():
            spine.set_color(axes_color)
        ax.set_facecolor(HITMAP_BACKGROUND_COLOR)  # Keep facecolor as background
        # Make tick labels the reserved axes color
        ax.tick_params(colors=axes_color, labelcolor=axes_color)
        # Make axis labels the reserved axes color
        ax.xaxis.label.set_color(axes_color)
        ax.yaxis.label.set_color(axes_color)
        # Make title the reserved axes color
        ax.title.set_color(axes_color)
        if ax.get_legend():
            ax.get_legend().set_visible(False)

    fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

    # Save hitmap with bbox_inches='tight' - SAME as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=HITMAP_BACKGROUND_COLOR)
    buf.seek(0)
    hitmap_img = Image.open(buf).convert('RGB')

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

    # Restore axes properties
    for ax_props in original_ax_props:
        ax = ax_props['ax']
        try:
            ax.set_facecolor(ax_props['facecolor'])
            for name, visible in ax_props['spines'].items():
                ax.spines[name].set_visible(visible)
            ax.set_xlabel(ax_props['xlabel'])
            ax.set_ylabel(ax_props['ylabel'])
            ax.set_title(ax_props['title'])
            if 'legend_visible' in ax_props and ax.get_legend():
                ax.get_legend().set_visible(ax_props['legend_visible'])
        except Exception:
            pass

    fig.patch.set_facecolor(original_fig_facecolor)

    return hitmap_img, color_map


# EOF
