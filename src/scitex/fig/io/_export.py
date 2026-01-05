#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/io/export.py
"""
Export operations for scitex.fig.

Handles composing and exporting canvas to various formats (PNG, PDF, SVG).
"""

from pathlib import Path
from typing import List, Union, Optional, Dict, Any


def export_canvas_to_file(
    project_dir: Union[str, Path],
    canvas_name: str,
    output_format: str = "png",
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """
    Export canvas to specified format.

    Composes all panels according to canvas.json and exports to a single image.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    output_format : str, optional
        Output format: "png", "pdf", "svg", "eps" (default: "png")
    dpi : int, optional
        Resolution for raster formats (default: 300)
    transparent : bool, optional
        Use transparent background (default: False)

    Returns
    -------
    Path
        Path to exported file in exports/ directory
    """
    from ._directory import get_canvas_directory_path
    from ._canvas import load_canvas_json

    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    canvas_json = load_canvas_json(project_dir, canvas_name, verify_data_hashes=False)

    exports_dir = canvas_dir / "exports"
    exports_dir.mkdir(exist_ok=True)

    output_path = exports_dir / f"canvas.{output_format}"

    # Compose canvas
    _compose_and_export(
        canvas_dir=canvas_dir,
        canvas_json=canvas_json,
        output_path=output_path,
        output_format=output_format,
        dpi=dpi,
        transparent=transparent,
    )

    return output_path


def export_canvas_to_multiple_formats(
    project_dir: Union[str, Path],
    canvas_name: str,
    formats: List[str] = None,
    dpi: int = 300,
    transparent: bool = False,
) -> List[Path]:
    """
    Export canvas to multiple formats.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name
    formats : List[str], optional
        List of formats (default: ["png", "pdf", "svg"])
    dpi : int, optional
        Resolution for raster formats (default: 300)
    transparent : bool, optional
        Use transparent background (default: False)

    Returns
    -------
    List[Path]
        List of paths to exported files
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    paths = []
    for fmt in formats:
        path = export_canvas_to_file(
            project_dir=project_dir,
            canvas_name=canvas_name,
            output_format=fmt,
            dpi=dpi,
            transparent=transparent,
        )
        paths.append(path)

    return paths


def list_canvas_exports(
    project_dir: Union[str, Path],
    canvas_name: str,
) -> List[Path]:
    """
    List all exported files for a canvas.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    canvas_name : str
        Canvas name

    Returns
    -------
    List[Path]
        List of paths in exports/ directory
    """
    from ._directory import get_canvas_directory_path

    canvas_dir = get_canvas_directory_path(project_dir, canvas_name)
    exports_dir = canvas_dir / "exports"

    if not exports_dir.exists():
        return []

    return sorted([p for p in exports_dir.iterdir() if p.is_file()])


def _compose_and_export(
    canvas_dir: Path,
    canvas_json: Dict[str, Any],
    output_path: Path,
    output_format: str,
    dpi: int,
    transparent: bool,
) -> None:
    """
    Compose all panels and export to file.

    Uses matplotlib for composition.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from PIL import Image
    import numpy as np

    # Canvas size in mm
    width_mm = canvas_json["size"]["width_mm"]
    height_mm = canvas_json["size"]["height_mm"]

    # Convert to inches for matplotlib
    mm_to_inch = 1 / 25.4
    width_inch = width_mm * mm_to_inch
    height_inch = height_mm * mm_to_inch

    # Create figure
    fig = plt.figure(figsize=(width_inch, height_inch), dpi=dpi)

    # Set background
    bg_color = canvas_json.get("background", {}).get("color", "#ffffff")
    if transparent:
        fig.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor(bg_color)

    # Sort panels by z_index
    panels = sorted(canvas_json.get("panels", []), key=lambda p: p.get("z_index", 0))

    # Place each panel
    for panel in panels:
        if not panel.get("visible", True):
            continue

        _place_panel(
            fig=fig,
            canvas_dir=canvas_dir,
            panel=panel,
            canvas_width_mm=width_mm,
            canvas_height_mm=height_mm,
        )

    # Add annotations
    _add_annotations(
        fig=fig,
        annotations=canvas_json.get("annotations", []),
        canvas_width_mm=width_mm,
        canvas_height_mm=height_mm,
    )

    # Add title
    title_config = canvas_json.get("title", {})
    if title_config.get("text"):
        _add_title(
            fig=fig,
            title_config=title_config,
            canvas_width_mm=width_mm,
            canvas_height_mm=height_mm,
        )

    # Add caption (figure legend in scientific sense) - only if render=True
    caption_config = canvas_json.get("caption", {})
    if caption_config.get("text") and caption_config.get("render", False):
        _add_caption(
            fig=fig,
            caption_config=caption_config,
            canvas_width_mm=width_mm,
            canvas_height_mm=height_mm,
        )

    # Save
    fig.savefig(
        output_path,
        format=output_format,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=transparent,
    )

    plt.close(fig)


def _place_panel(
    fig: "Figure",
    canvas_dir: Path,
    panel: Dict[str, Any],
    canvas_width_mm: float,
    canvas_height_mm: float,
) -> None:
    """Place a single panel on the figure."""
    from PIL import Image
    import numpy as np

    panel_name = panel.get("name", "")
    panel_type = panel.get("type", "image")

    # Get panel image path
    panel_dir = canvas_dir / "panels" / panel_name

    if panel_type == "scitex":
        img_path = panel_dir / "panel.png"
    else:
        # Image type - use source filename
        source = panel.get("source", "panel.png")
        img_path = panel_dir / source

    # Check if path exists (resolve symlinks)
    if not img_path.exists():
        # Try resolving symlink
        try:
            resolved_path = img_path.resolve()
            if not resolved_path.exists():
                return
            img_path = resolved_path
        except (OSError, ValueError):
            return

    # Load image
    img = Image.open(img_path)

    # Apply transforms
    opacity = panel.get("opacity", 1.0)
    flip_h = panel.get("flip_h", False)
    flip_v = panel.get("flip_v", False)
    rotation_deg = panel.get("rotation_deg", 0)

    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotation_deg != 0:
        img = img.rotate(-rotation_deg, expand=True, resample=Image.BICUBIC)

    # Apply clip
    clip = panel.get("clip", {})
    if clip.get("enabled", False):
        # Clip is in mm relative to panel, need to convert to pixels
        # For now, simple implementation using PIL crop
        pass  # TODO: Implement clipping

    # Convert to numpy array
    img_array = np.array(img)

    # Handle opacity
    if opacity < 1.0:
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            # Has alpha channel
            img_array[:, :, 3] = (img_array[:, :, 3] * opacity).astype(np.uint8)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            # Add alpha channel
            alpha = np.full(img_array.shape[:2], int(255 * opacity), dtype=np.uint8)
            img_array = np.dstack([img_array, alpha])

    # Position in normalized coordinates (0-1)
    pos = panel.get("position", {})
    size = panel.get("size", {})

    x_mm = pos.get("x_mm", 0)
    y_mm = pos.get("y_mm", 0)
    w_mm = size.get("width_mm", 50)
    h_mm = size.get("height_mm", 50)

    # Convert to figure coordinates (origin bottom-left)
    left = x_mm / canvas_width_mm
    bottom = 1 - (y_mm + h_mm) / canvas_height_mm
    width = w_mm / canvas_width_mm
    height = h_mm / canvas_height_mm

    # Create axes and place image
    ax = fig.add_axes([left, bottom, width, height])
    ax.imshow(img_array)
    ax.axis("off")

    # Add label
    label = panel.get("label", {})
    if label.get("text"):
        _add_panel_label(ax, label)

    # Add border
    border = panel.get("border", {})
    if border.get("visible", False):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border.get("color", "#000000"))
            spine.set_linewidth(border.get("width_mm", 0.2) * 72 / 25.4)  # mm to points


def _add_panel_label(ax, label: Dict[str, Any]) -> None:
    """Add label (A, B, C...) to panel."""
    text = label.get("text", "")
    position = label.get("position", "top-left")
    fontsize = label.get("fontsize", 12)
    fontweight = label.get("fontweight", "bold")

    # Position mapping
    pos_map = {
        "top-left": (0.02, 0.98, "left", "top"),
        "top-right": (0.98, 0.98, "right", "top"),
        "bottom-left": (0.02, 0.02, "left", "bottom"),
        "bottom-right": (0.98, 0.02, "right", "bottom"),
    }

    x, y, ha, va = pos_map.get(position, pos_map["top-left"])

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha,
        va=va,
    )


def _add_annotations(
    fig: "Figure",
    annotations: List[Dict[str, Any]],
    canvas_width_mm: float,
    canvas_height_mm: float,
) -> None:
    """Add annotations to figure."""
    for ann in annotations:
        ann_type = ann.get("type", "")

        if ann_type == "text":
            pos = ann.get("position", {})
            x = pos.get("x_mm", 0) / canvas_width_mm
            y = 1 - pos.get("y_mm", 0) / canvas_height_mm

            fig.text(
                x,
                y,
                ann.get("content", ""),
                fontsize=ann.get("fontsize", 10),
                color=ann.get("color", "#000000"),
            )

        # TODO: Implement arrow, bracket, line, rectangle annotations


def _add_title(
    fig: "Figure",
    title_config: Dict[str, Any],
    canvas_width_mm: float,
    canvas_height_mm: float,
) -> None:
    """Add title to figure."""
    pos = title_config.get("position", {})
    x = pos.get("x_mm", canvas_width_mm / 2) / canvas_width_mm
    y = 1 - pos.get("y_mm", 5) / canvas_height_mm

    fig.text(
        x,
        y,
        title_config.get("text", ""),
        fontsize=title_config.get("fontsize", 14),
        ha="center",
        va="top",
    )


def _add_caption(
    fig: "Figure",
    caption_config: Dict[str, Any],
    canvas_width_mm: float,
    canvas_height_mm: float,
) -> None:
    """
    Add figure caption (legend in scientific sense).

    Caption is placed below the figure by default.
    Text is wrapped to fit within specified width.
    """
    import textwrap

    text = caption_config.get("text", "")
    fontsize = caption_config.get("fontsize", 10)
    width_mm = caption_config.get("width_mm") or (canvas_width_mm - 20)

    pos = caption_config.get("position", {})
    x = pos.get("x_mm", 10) / canvas_width_mm
    y = pos.get("y_mm", canvas_height_mm + 5)  # Below canvas by default

    # Convert to figure coordinates (y below canvas is negative in bbox_inches="tight")
    y_norm = 1 - y / canvas_height_mm

    # Estimate characters per line based on width and fontsize
    # Approximate: 1 character ~ 0.6 * fontsize in points, 1 point ~ 0.35mm
    chars_per_mm = 1 / (0.6 * fontsize * 0.35)
    wrap_width = int(width_mm * chars_per_mm)

    # Wrap text
    wrapped_text = textwrap.fill(text, width=wrap_width)

    fig.text(
        x,
        y_norm,
        wrapped_text,
        fontsize=fontsize,
        ha="left",
        va="top",
        wrap=True,
    )


# EOF
