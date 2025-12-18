#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_element_ops.py

"""Element processing operations for Figz bundles."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def process_content(
    element_type: str,
    content: Any,
    element_id: str,
    element: Dict[str, Any],
    figure_to_stx_bytes_fn,
) -> Optional[bytes]:
    """Process content for an element.

    Args:
        element_type: Type of element ("plot", "image", etc.)
        content: The content to process
        element_id: ID of the element
        element: Element dictionary (may be modified)
        figure_to_stx_bytes_fn: Function to convert matplotlib figure to bytes

    Returns:
        Content bytes or None
    """
    if element_type == "plot":
        import matplotlib.figure

        if isinstance(content, matplotlib.figure.Figure) or hasattr(
            content, "figure"
        ):
            return figure_to_stx_bytes_fn(content, element_id)
        elif isinstance(content, bytes):
            return content
    elif element_type == "image":
        return process_image_content(content, element)
    elif isinstance(content, bytes):
        return content
    return None


def process_image_content(content: Any, element: Dict[str, Any]) -> Optional[bytes]:
    """Process image content and detect format.

    Args:
        content: Image content (path, bytes, etc.)
        element: Element dictionary (image_format will be set)

    Returns:
        Image bytes or None
    """
    img_ext = ".png"

    if isinstance(content, (str, Path)):
        img_path = Path(content)
        if img_path.exists():
            with open(img_path, "rb") as f:
                content_bytes = f.read()
            img_ext = img_path.suffix.lower() or ".png"
        else:
            raise FileNotFoundError(f"Image not found: {content}")
    elif isinstance(content, bytes):
        content_bytes = content
        if content_bytes[:4] == b"\x89PNG":
            img_ext = ".png"
        elif content_bytes[:4] == b"<svg" or b"<svg" in content_bytes[:100]:
            img_ext = ".svg"
        elif content_bytes[:2] == b"\xff\xd8":
            img_ext = ".jpg"
        elif content_bytes[:4] == b"%PDF":
            img_ext = ".pdf"
    else:
        return None

    element["image_format"] = img_ext.lstrip(".")
    return content_bytes


def get_content_extension(element_type: str, element: Dict[str, Any]) -> str:
    """Get file extension for element content.

    Args:
        element_type: Type of element
        element: Element dictionary

    Returns:
        File extension including dot (e.g., ".stx")
    """
    if element_type == "image":
        ext = element.get("image_format", "png")
        return f".{ext}" if not ext.startswith(".") else ext
    return {
        "plot": ".stx",
        "figure": ".stx",
        "stats": ".stx",
    }.get(element_type, ".bin")


def figure_to_stx_bytes(fig, basename: str = "plot") -> bytes:
    """Convert matplotlib figure to STX bundle bytes.

    Args:
        fig: Matplotlib figure or wrapper with .figure attribute
        basename: Base filename for the bundle

    Returns:
        Bundle bytes
    """
    import matplotlib.figure

    if hasattr(fig, "figure"):
        fig = fig.figure
    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(f"Expected matplotlib Figure, got {type(fig).__name__}")

    with tempfile.TemporaryDirectory() as tmpdir:
        stx_path = Path(tmpdir) / f"{basename}.stx"
        from scitex.io import save as io_save

        io_save(fig, stx_path, verbose=False, basename=basename)
        with open(stx_path, "rb") as f:
            return f.read()


# EOF
