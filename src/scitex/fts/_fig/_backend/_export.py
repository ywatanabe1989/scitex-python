#!/usr/bin/env python3
# File: ./src/scitex/vis/backend/export.py
"""Export figure models to image files via scitex.io."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ._render import build_figure_from_json


def export_figure(
    fig_json: Dict[str, Any],
    output_path: Union[str, Path],
    fmt: Optional[str] = None,
    dpi: int = 300,
    auto_crop: bool = False,
    **kwargs,
) -> Path:
    """
    Export figure JSON to image file.

    Parameters
    ----------
    fig_json : Dict[str, Any]
        Figure JSON specification
    output_path : str or Path
        Output file path
    fmt : str, optional
        Output format ("png", "pdf", "svg", etc.)
        If None, inferred from output_path extension
    dpi : int, optional
        Resolution in dots per inch (default: 300)
    auto_crop : bool, optional
        Automatically crop whitespace (default: False)
    **kwargs
        Additional keyword arguments passed to scitex.io.save()

    Returns
    -------
    Path
        Path to the saved file

    Examples
    --------
    >>> fig_json = {"width_mm": 180, "height_mm": 120, ...}
    >>> export_figure(fig_json, "output.png", dpi=300)
    PosixPath('output.png')

    >>> # Auto-crop for publication
    >>> export_figure(fig_json, "figure.pdf", auto_crop=True)
    PosixPath('figure.pdf')
    """
    import scitex as stx

    output_path = Path(output_path)

    # Build figure from JSON
    fig, axes = build_figure_from_json(fig_json)

    # Save using scitex.io
    save_kwargs = {"dpi": dpi, "auto_crop": auto_crop}
    save_kwargs.update(kwargs)

    if fmt:
        save_kwargs["fmt"] = fmt

    stx.io.save(fig, output_path, **save_kwargs)

    return output_path


def export_figure_from_file(
    json_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs,
) -> Path:
    """
    Export figure from JSON file to image file.

    Parameters
    ----------
    json_path : str or Path
        Path to figure JSON file
    output_path : str or Path
        Output image file path
    **kwargs
        Additional keyword arguments passed to export_figure()

    Returns
    -------
    Path
        Path to the saved file

    Examples
    --------
    >>> export_figure_from_file("figure.json", "figure.png")
    PosixPath('figure.png')
    """
    import scitex as stx

    json_path = Path(json_path)

    # Load JSON
    fig_json = stx.io.load(json_path)

    # Export
    return export_figure(fig_json, output_path, **kwargs)


def export_multiple_formats(
    fig_json: Dict[str, Any],
    output_dir: Union[str, Path],
    base_name: str,
    formats: list = None,
    **kwargs,
) -> Dict[str, Path]:
    """
    Export figure to multiple formats.

    Parameters
    ----------
    fig_json : Dict[str, Any]
        Figure JSON specification
    output_dir : str or Path
        Output directory
    base_name : str
        Base filename (without extension)
    formats : list, optional
        List of formats (default: ["png", "pdf", "svg"])
    **kwargs
        Additional keyword arguments passed to export_figure()

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping format to output path

    Examples
    --------
    >>> fig_json = {...}
    >>> paths = export_multiple_formats(
    ...     fig_json,
    ...     "output",
    ...     "figure-01",
    ...     formats=["png", "pdf"]
    ... )
    >>> paths
    {'png': PosixPath('output/figure-01.png'),
     'pdf': PosixPath('output/figure-01.pdf')}
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for fmt in formats:
        output_path = output_dir / f"{base_name}.{fmt}"
        results[fmt] = export_figure(fig_json, output_path, fmt=fmt, **kwargs)

    return results


# EOF
