#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_kinds/_text/__init__.py

"""Text kind - Text annotation elements.

A text bundle contains styled text (no payload data).
Used for titles, labels, annotations in figures.

Structure:
- canonical/node.json: Text content and styling
- No payload (text is in node specification)
"""

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def render_text(
    ax: "Axes",
    text: str,
    x: float,
    y: float,
    fontsize: Optional[float] = None,
    fontweight: Optional[str] = None,
    color: Optional[str] = None,
    ha: str = "center",
    va: str = "center",
    rotation: float = 0,
    **kwargs: Any,
) -> None:
    """Render text annotation on axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render on
    text : str
        Text content to display
    x, y : float
        Position in axes coordinates
    fontsize : float, optional
        Font size in points
    fontweight : str, optional
        Font weight (normal, bold, etc.)
    color : str, optional
        Text color
    ha : str
        Horizontal alignment (left, center, right)
    va : str
        Vertical alignment (top, center, bottom, baseline)
    rotation : float
        Rotation angle in degrees
    **kwargs : Any
        Additional kwargs passed to ax.text()
    """
    text_kwargs = {
        "ha": ha,
        "va": va,
        "rotation": rotation,
    }
    if fontsize is not None:
        text_kwargs["fontsize"] = fontsize
    if fontweight is not None:
        text_kwargs["fontweight"] = fontweight
    if color is not None:
        text_kwargs["color"] = color
    text_kwargs.update(kwargs)

    ax.text(x, y, text, transform=ax.transAxes, **text_kwargs)


__all__ = ["render_text"]

# EOF
