#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/utils/defaults.py
"""Default figure templates for common publication formats."""

from typing import Dict, Any


# Common journal figure widths (mm)
NATURE_SINGLE_COLUMN_MM = 89  # Nature single column
NATURE_DOUBLE_COLUMN_MM = 183  # Nature double column
NATURE_FULL_PAGE_MM = 247  # Nature full page

SCIENCE_SINGLE_COLUMN_MM = 84  # Science single column
SCIENCE_DOUBLE_COLUMN_MM = 174  # Science double column

CELL_SINGLE_COLUMN_MM = 85  # Cell single column
CELL_DOUBLE_COLUMN_MM = 174  # Cell double column

PNAS_SINGLE_COLUMN_MM = 87  # PNAS single column
PNAS_DOUBLE_COLUMN_MM = 178  # PNAS double column

# A4 page dimensions
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# Default spacing (mm)
DEFAULT_MARGIN_MM = 5
DEFAULT_SPACING_MM = 3


def get_nature_single_column(
    height_mm: float = 89,
    nrows: int = 1,
    ncols: int = 1,
) -> Dict[str, Any]:
    """
    Get Nature single column figure template.

    Parameters
    ----------
    height_mm : float, optional
        Figure height in mm (default: 89, square)
    nrows : int, optional
        Number of subplot rows (default: 1)
    ncols : int, optional
        Number of subplot columns (default: 1)

    Returns
    -------
    Dict[str, Any]
        Figure JSON template
    """
    return {
        "width_mm": NATURE_SINGLE_COLUMN_MM,
        "height_mm": height_mm,
        "nrows": nrows,
        "ncols": ncols,
        "dpi": 300,
        "left_mm": DEFAULT_MARGIN_MM,
        "right_mm": DEFAULT_MARGIN_MM,
        "top_mm": DEFAULT_MARGIN_MM,
        "bottom_mm": DEFAULT_MARGIN_MM,
        "wspace_mm": DEFAULT_SPACING_MM,
        "hspace_mm": DEFAULT_SPACING_MM,
        "metadata": {
            "template": "nature_single_column",
            "journal": "Nature",
        },
    }


def get_nature_double_column(
    height_mm: float = 120,
    nrows: int = 1,
    ncols: int = 1,
) -> Dict[str, Any]:
    """
    Get Nature double column figure template.

    Parameters
    ----------
    height_mm : float, optional
        Figure height in mm (default: 120)
    nrows : int, optional
        Number of subplot rows (default: 1)
    ncols : int, optional
        Number of subplot columns (default: 1)

    Returns
    -------
    Dict[str, Any]
        Figure JSON template
    """
    return {
        "width_mm": NATURE_DOUBLE_COLUMN_MM,
        "height_mm": height_mm,
        "nrows": nrows,
        "ncols": ncols,
        "dpi": 300,
        "left_mm": DEFAULT_MARGIN_MM,
        "right_mm": DEFAULT_MARGIN_MM,
        "top_mm": DEFAULT_MARGIN_MM,
        "bottom_mm": DEFAULT_MARGIN_MM,
        "wspace_mm": DEFAULT_SPACING_MM,
        "hspace_mm": DEFAULT_SPACING_MM,
        "metadata": {
            "template": "nature_double_column",
            "journal": "Nature",
        },
    }


def get_science_single_column(
    height_mm: float = 84,
    nrows: int = 1,
    ncols: int = 1,
) -> Dict[str, Any]:
    """
    Get Science single column figure template.

    Parameters
    ----------
    height_mm : float, optional
        Figure height in mm (default: 84, square)
    nrows : int, optional
        Number of subplot rows (default: 1)
    ncols : int, optional
        Number of subplot columns (default: 1)

    Returns
    -------
    Dict[str, Any]
        Figure JSON template
    """
    return {
        "width_mm": SCIENCE_SINGLE_COLUMN_MM,
        "height_mm": height_mm,
        "nrows": nrows,
        "ncols": ncols,
        "dpi": 300,
        "left_mm": DEFAULT_MARGIN_MM,
        "right_mm": DEFAULT_MARGIN_MM,
        "top_mm": DEFAULT_MARGIN_MM,
        "bottom_mm": DEFAULT_MARGIN_MM,
        "wspace_mm": DEFAULT_SPACING_MM,
        "hspace_mm": DEFAULT_SPACING_MM,
        "metadata": {
            "template": "science_single_column",
            "journal": "Science",
        },
    }


def get_a4_figure(
    width_mm: float = 180,
    height_mm: float = 120,
    nrows: int = 1,
    ncols: int = 1,
) -> Dict[str, Any]:
    """
    Get A4-sized figure template.

    Parameters
    ----------
    width_mm : float, optional
        Figure width in mm (default: 180)
    height_mm : float, optional
        Figure height in mm (default: 120)
    nrows : int, optional
        Number of subplot rows (default: 1)
    ncols : int, optional
        Number of subplot columns (default: 1)

    Returns
    -------
    Dict[str, Any]
        Figure JSON template
    """
    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "nrows": nrows,
        "ncols": ncols,
        "dpi": 300,
        "left_mm": DEFAULT_MARGIN_MM,
        "right_mm": DEFAULT_MARGIN_MM,
        "top_mm": DEFAULT_MARGIN_MM,
        "bottom_mm": DEFAULT_MARGIN_MM,
        "wspace_mm": DEFAULT_SPACING_MM,
        "hspace_mm": DEFAULT_SPACING_MM,
        "metadata": {
            "template": "a4_figure",
        },
    }


def get_square_figure(
    size_mm: float = 120,
    nrows: int = 1,
    ncols: int = 1,
) -> Dict[str, Any]:
    """
    Get square figure template.

    Parameters
    ----------
    size_mm : float, optional
        Figure size (width and height) in mm (default: 120)
    nrows : int, optional
        Number of subplot rows (default: 1)
    ncols : int, optional
        Number of subplot columns (default: 1)

    Returns
    -------
    Dict[str, Any]
        Figure JSON template
    """
    return {
        "width_mm": size_mm,
        "height_mm": size_mm,
        "nrows": nrows,
        "ncols": ncols,
        "dpi": 300,
        "left_mm": DEFAULT_MARGIN_MM,
        "right_mm": DEFAULT_MARGIN_MM,
        "top_mm": DEFAULT_MARGIN_MM,
        "bottom_mm": DEFAULT_MARGIN_MM,
        "wspace_mm": DEFAULT_SPACING_MM,
        "hspace_mm": DEFAULT_SPACING_MM,
        "metadata": {
            "template": "square_figure",
        },
    }


def get_presentation_slide(
    aspect_ratio: str = "16:9",
    width_mm: float = 254,  # 10 inches at 25.4 mm/inch
) -> Dict[str, Any]:
    """
    Get presentation slide figure template.

    Parameters
    ----------
    aspect_ratio : str, optional
        Aspect ratio ("16:9" or "4:3", default: "16:9")
    width_mm : float, optional
        Figure width in mm (default: 254, ~10 inches)

    Returns
    -------
    Dict[str, Any]
        Figure JSON template
    """
    if aspect_ratio == "16:9":
        height_mm = width_mm * 9 / 16
    elif aspect_ratio == "4:3":
        height_mm = width_mm * 3 / 4
    else:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "nrows": 1,
        "ncols": 1,
        "dpi": 150,  # Lower DPI for presentations
        "left_mm": DEFAULT_MARGIN_MM,
        "right_mm": DEFAULT_MARGIN_MM,
        "top_mm": DEFAULT_MARGIN_MM,
        "bottom_mm": DEFAULT_MARGIN_MM,
        "metadata": {
            "template": "presentation_slide",
            "aspect_ratio": aspect_ratio,
        },
    }


# Template registry
TEMPLATES = {
    "nature_single": get_nature_single_column,
    "nature_double": get_nature_double_column,
    "science_single": get_science_single_column,
    "a4": get_a4_figure,
    "square": get_square_figure,
    "presentation": get_presentation_slide,
}


def get_template(name: str, **kwargs) -> Dict[str, Any]:
    """
    Get a figure template by name.

    Parameters
    ----------
    name : str
        Template name
    **kwargs
        Template-specific parameters

    Returns
    -------
    Dict[str, Any]
        Figure JSON template

    Examples
    --------
    >>> template = get_template("nature_single", height_mm=100)
    >>> template["width_mm"]
    89
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}"
        )

    return TEMPLATES[name](**kwargs)


def list_templates() -> list:
    """
    List available template names.

    Returns
    -------
    list
        List of template names
    """
    return list(TEMPLATES.keys())


# EOF
