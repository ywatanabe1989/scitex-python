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
    for_export: bool = False,
    panel_labels_config: Dict[str, Any] = None,
    bundle_path: Path = None,
) -> bytes:
    """Render composed preview in specified format.

    Args:
        elements: List of element specifications
        size_mm: Canvas size {"width": mm, "height": mm}
        get_element_content: Function to get element content bytes
        fmt: Output format ("png", "svg", "pdf")
        dpi: Resolution for raster formats
        for_export: If True, hide editor-only elements (comments)
        panel_labels_config: Panel label settings from theme.json

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
        _render_element(ax, elem, get_element_content, dpi, for_export)

    # Render panel labels
    if panel_labels_config is None:
        panel_labels_config = _load_panel_labels_config(bundle_path)
    _render_panel_labels(ax, elements, panel_labels_config)

    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer.getvalue()


def _render_element(
    ax,
    elem: Dict[str, Any],
    get_element_content: Callable,
    dpi: int,
    for_export: bool = False,
):
    """Render a single element onto the axes.

    Args:
        for_export: If True, skip editor-only elements like comments
    """
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
    elif elem_type == "symbol":
        _render_symbol_element(ax, elem, pos)
    elif elem_type == "equation":
        _render_equation_element(ax, elem, pos)
    elif elem_type == "comment":
        # Comments are only visible in editor, hidden in exports
        if not for_export:
            _render_comment_element(ax, elem, pos)


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


# Predefined symbol mappings
SYMBOL_MAP = {
    "star": "★",
    "asterisk": "*",
    "dagger": "†",
    "double_dagger": "‡",
    "bullet": "•",
    "checkmark": "✓",
    "cross": "✗",
    "arrow_right": "→",
    "arrow_left": "←",
    "arrow_up": "↑",
    "arrow_down": "↓",
    "plus_minus": "±",
    "degree": "°",
    "infinity": "∞",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "mu": "μ",
    "sigma": "σ",
    "omega": "ω",
    "pi": "π",
}


def _render_symbol_element(ax, elem, pos):
    """Render a symbol element.

    Schema:
        symbol_type: str - predefined type (e.g., "star", "dagger") or "custom"
        content: str - actual symbol character (required if symbol_type="custom")
        fontsize: float - font size in points (default: 12)
        color: str - symbol color (default: "black")
        ha: str - horizontal alignment (default: "center")
        va: str - vertical alignment (default: "center")
    """
    symbol_type = elem.get("symbol_type", "asterisk")
    content = elem.get("content")

    # Get symbol from map or use content directly
    if content is None:
        content = SYMBOL_MAP.get(symbol_type, symbol_type)

    ax.text(
        pos.get("x_mm", 0),
        pos.get("y_mm", 0),
        content,
        fontsize=elem.get("fontsize", 12),
        color=elem.get("color", "black"),
        ha=elem.get("ha", "center"),
        va=elem.get("va", "center"),
        fontweight=elem.get("fontweight", "normal"),
    )


def _render_equation_element(ax, elem, pos):
    """Render an equation element using matplotlib's mathtext.

    Schema:
        latex: str - LaTeX equation string (e.g., "$E = mc^2$")
        fontsize: float - font size in points (default: 12)
        color: str - equation color (default: "black")
        ha: str - horizontal alignment (default: "center")
        va: str - vertical alignment (default: "center")
    """
    latex = elem.get("latex", elem.get("content", ""))

    # Ensure LaTeX is wrapped in $ signs for mathtext
    if latex and not latex.startswith("$"):
        latex = f"${latex}$"

    ax.text(
        pos.get("x_mm", 0),
        pos.get("y_mm", 0),
        latex,
        fontsize=elem.get("fontsize", 12),
        color=elem.get("color", "black"),
        ha=elem.get("ha", "center"),
        va=elem.get("va", "center"),
        usetex=False,  # Use matplotlib's built-in mathtext
    )


def _render_comment_element(ax, elem, pos):
    """Render a comment element (annotation marker).

    Schema:
        content: str - comment text
        author: str - comment author (optional)
        resolved: bool - whether comment is resolved (default: False)
        visible: bool - whether to render in preview (default: True for editing)
        marker_color: str - color of comment marker (default: "orange")

    Comments are rendered as small markers in edit mode but hidden in final exports.
    The visible flag controls whether to show in preview rendering.
    """
    visible = elem.get("visible", True)
    if not visible:
        return

    resolved = elem.get("resolved", False)
    marker_color = elem.get("marker_color", "green" if resolved else "orange")

    # Draw comment marker (small filled circle with "C" inside)
    from matplotlib.patches import Circle

    marker_radius = 3  # mm
    circle = Circle(
        (pos.get("x_mm", 0), pos.get("y_mm", 0)),
        marker_radius,
        facecolor=marker_color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    ax.add_patch(circle)

    # Add "C" label inside marker
    ax.text(
        pos.get("x_mm", 0),
        pos.get("y_mm", 0),
        "C",
        fontsize=6,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
    )


def _load_panel_labels_config(bundle_path: Path) -> dict:
    """Load panel labels config from theme.json or return defaults."""
    import json

    if bundle_path and bundle_path.exists():
        theme_path = bundle_path / "theme.json"
        if theme_path.exists():
            try:
                with open(theme_path, encoding="utf-8") as f:
                    theme = json.load(f)
                    return theme.get("panel_labels", {})
            except Exception:
                pass
    # Return defaults
    return {
        "visible": True,
        "style": "uppercase",
        "format": "({letter})",
        "font_size_pt": 12,
        "font_weight": "bold",
        "position": "top-left",
        "offset_mm": {"x": 2.0, "y": 2.0},
    }


def _render_panel_labels(ax, elements: List[Dict[str, Any]], config: Dict[str, Any]):
    """Render panel labels (A, B, C, ...) on plot elements.

    Args:
        ax: Matplotlib axes
        elements: List of element specifications
        config: Panel label configuration from theme.json:
            - style: "uppercase", "lowercase", "roman", "Roman"
            - format: e.g., "({letter})", "{letter}.", etc.
            - font_size_pt: Font size
            - font_weight: "bold", "normal"
            - position: "top-left", "top-right", "bottom-left", "bottom-right"
            - offset_mm: {"x": mm, "y": mm}
            - visible: bool
    """
    if not config.get("visible", True):
        return

    # Get config with defaults
    font_size = config.get("font_size_pt", 12)
    font_weight = config.get("font_weight", "bold")
    position = config.get("position", "top-left")
    offset = config.get("offset_mm", {"x": 2.0, "y": 2.0})
    label_format = config.get("format", "({letter})")

    # Get plot elements with panel letters
    plot_elements = [e for e in elements if e.get("type") == "plot"]

    for elem in plot_elements:
        panel_letter = elem.get("panel_letter")
        if not panel_letter:
            continue

        pos = elem.get("position", {})
        sz = elem.get("size", {})

        # Calculate label position based on position setting
        x_mm = pos.get("x_mm", 0)
        y_mm = pos.get("y_mm", 0)
        width_mm = sz.get("width_mm", 80)
        height_mm = sz.get("height_mm", 60)

        if "left" in position:
            label_x = x_mm + offset.get("x", 2)
            ha = "left"
        else:
            label_x = x_mm + width_mm - offset.get("x", 2)
            ha = "right"

        if "top" in position:
            label_y = y_mm + offset.get("y", 2)
            va = "top"
        else:
            label_y = y_mm + height_mm - offset.get("y", 2)
            va = "bottom"

        # Format the label
        label_text = label_format.replace("{letter}", panel_letter)

        # Draw the label
        ax.text(
            label_x,
            label_y,
            label_text,
            fontsize=font_size,
            fontweight=font_weight,
            ha=ha,
            va=va,
            color="black",
            zorder=100,  # Ensure labels are on top
        )


# EOF
