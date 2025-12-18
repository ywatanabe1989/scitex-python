#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_render.py

"""Rendering functions for Figz bundles."""

import io
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List


def render_preview_internal(
    elements: List[Dict[str, Any]],
    size_mm: Dict[str, float],
    get_element_content: Callable[[str], bytes],
    fmt: str = "png",
    dpi: int = 150,
) -> bytes:
    """Render composed preview in specified format.

    Args:
        elements: List of element specifications
        size_mm: Canvas size {"width": mm, "height": mm}
        get_element_content: Function to get element content bytes
        fmt: Output format ("png", "svg", "pdf")
        dpi: Resolution for raster formats

    Returns:
        bytes: Rendered image data
    """
    import matplotlib.pyplot as plt


    width_in = size_mm.get("width", 170) / 25.4
    height_in = size_mm.get("height", 120) / 25.4

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, size_mm.get("width", 170))
    ax.set_ylim(size_mm.get("height", 120), 0)  # Flip Y for top-left origin
    ax.axis("off")

    for elem in elements:
        _render_element(ax, elem, get_element_content, dpi)

    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer.getvalue()


def _render_element(ax, elem: Dict[str, Any], get_element_content: Callable, dpi: int):
    """Render a single element onto the axes."""
    elem_type = elem.get("type")
    pos = elem.get("position", {})
    sz = elem.get("size", {})

    if elem_type == "plot":
        _render_plot_element(ax, elem, pos, sz, get_element_content, dpi)
    elif elem_type == "text":
        _render_text_element(ax, elem, pos)
    elif elem_type == "shape":
        _render_shape_element(ax, elem)
    elif elem_type == "image":
        _render_image_element(ax, elem, pos, sz, get_element_content)


def _render_plot_element(ax, elem, pos, sz, get_element_content, dpi):
    """Render a plot element."""
    from PIL import Image

    from scitex.plt import Pltz

    content = get_element_content(elem["id"])
    if not content:
        return

    with tempfile.NamedTemporaryFile(suffix=".stx", delete=False) as f:
        f.write(content)
        temp_path = f.name
    try:
        pltz = Pltz(temp_path)
        preview = pltz.get_preview() or pltz.render_preview(dpi=dpi)
        img = Image.open(io.BytesIO(preview))
        _place_image(ax, img, pos, sz)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _render_text_element(ax, elem, pos):
    """Render a text element."""
    content = elem.get("content", "")
    ax.text(
        pos.get("x_mm", 0),
        pos.get("y_mm", 0),
        content,
        fontsize=elem.get("fontsize", 10),
        ha=elem.get("ha", "left"),
        va=elem.get("va", "top"),
    )


def _render_shape_element(ax, elem):
    """Render a shape element (arrow, bracket, line)."""
    shape_type = elem.get("shape_type", "")
    start = elem.get("start", {})
    end = elem.get("end", {})
    start_x = start.get("x_mm", 0)
    start_y = start.get("y_mm", 0)
    end_x = end.get("x_mm", 0)
    end_y = end.get("y_mm", 0)

    if shape_type == "arrow":
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
    elif shape_type == "bracket":
        bracket_height = 3
        ax.plot([start_x, start_x], [start_y, start_y - bracket_height], "k-", lw=1.5)
        ax.plot([end_x, end_x], [end_y, end_y - bracket_height], "k-", lw=1.5)
        ax.plot(
            [start_x, end_x],
            [start_y - bracket_height, end_y - bracket_height],
            "k-",
            lw=1.5,
        )
    elif shape_type == "line":
        ax.plot([start_x, end_x], [start_y, end_y], "k-", lw=1.5)


def _render_image_element(ax, elem, pos, sz, get_element_content):
    """Render an image element."""
    from PIL import Image

    content = get_element_content(elem["id"])
    if not content:
        return
    try:
        img = Image.open(io.BytesIO(content))
        _place_image(ax, img, pos, sz)
    except Exception:
        pass


def _place_image(ax, img, pos, sz):
    """Place an image on the axes with proper aspect ratio."""
    target_width = sz.get("width_mm", 80)
    target_height = sz.get("height_mm", 60)

    img_width, img_height = img.size
    img_aspect = img_width / img_height
    target_aspect = target_width / target_height

    if img_aspect > target_aspect:
        render_width = target_width
        render_height = target_width / img_aspect
    else:
        render_height = target_height
        render_width = target_height * img_aspect

    x_offset = (target_width - render_width) / 2
    y_offset = (target_height - render_height) / 2

    x_start = pos.get("x_mm", 0) + x_offset
    y_start = pos.get("y_mm", 0) + y_offset

    ax.imshow(
        img,
        extent=[x_start, x_start + render_width, y_start + render_height, y_start],
        aspect="auto",
    )


# EOF
