#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/io/save.py
"""Save figure JSON specifications."""

from pathlib import Path
from typing import Dict, Any, Union, Optional


def save_figure_json(
    fig_json: Dict[str, Any],
    output_path: Union[str, Path],
    **kwargs,
) -> Path:
    """
    Save figure JSON to file.

    Parameters
    ----------
    fig_json : Dict[str, Any]
        Figure JSON specification
    output_path : str or Path
        Output JSON file path
    **kwargs
        Additional keyword arguments passed to scitex.io.save()

    Returns
    -------
    Path
        Path to the saved JSON file

    Examples
    --------
    >>> fig_json = {"width_mm": 180, "height_mm": 120, ...}
    >>> save_figure_json(fig_json, "figure.json")
    PosixPath('figure.json')
    """
    import scitex as stx

    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save using scitex.io
    stx.io.save(fig_json, output_path, **kwargs)

    return output_path


def save_figure_json_to_project(
    project_dir: Union[str, Path],
    figure_id: str,
    fig_json: Dict[str, Any],
    subdir: str = "figs",
    **kwargs,
) -> Path:
    """
    Save figure JSON to project directory structure.

    Follows the convention: project_dir/scitex/vis/{subdir}/{figure_id}.json

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    figure_id : str
        Figure identifier
    fig_json : Dict[str, Any]
        Figure JSON specification
    subdir : str, optional
        Subdirectory name (default: "figs")
    **kwargs
        Additional keyword arguments passed to save_figure_json()

    Returns
    -------
    Path
        Path to the saved JSON file

    Examples
    --------
    >>> save_figure_json_to_project(
    ...     "/path/to/project",
    ...     "fig-001",
    ...     fig_json
    ... )
    PosixPath('/path/to/project/scitex/vis/figs/fig-001.json')
    """
    project_dir = Path(project_dir)
    vis_dir = project_dir / "scitex" / "vis" / subdir
    vis_dir.mkdir(parents=True, exist_ok=True)

    output_path = vis_dir / f"{figure_id}.json"
    return save_figure_json(fig_json, output_path, **kwargs)


def save_figure_model(
    fig_model,
    output_path: Union[str, Path],
    **kwargs,
) -> Path:
    """
    Save FigureModel to JSON file.

    Parameters
    ----------
    fig_model : FigureModel
        Figure model to save
    output_path : str or Path
        Output JSON file path
    **kwargs
        Additional keyword arguments passed to save_figure_json()

    Returns
    -------
    Path
        Path to the saved JSON file

    Examples
    --------
    >>> from scitex.canvas.model import FigureModel
    >>> fig_model = FigureModel(width_mm=180, height_mm=120)
    >>> save_figure_model(fig_model, "figure.json")
    PosixPath('figure.json')
    """
    fig_json = fig_model.to_dict()
    return save_figure_json(fig_json, output_path, **kwargs)


# EOF
