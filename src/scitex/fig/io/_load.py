#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/io/load.py
"""Load figure JSON specifications."""

from pathlib import Path
from typing import Dict, Any, Union, Optional


def load_figure_json(
    input_path: Union[str, Path],
    validate: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Load figure JSON from file.

    Parameters
    ----------
    input_path : str or Path
        Input JSON file path
    validate : bool, optional
        Validate JSON after loading (default: True)
    **kwargs
        Additional keyword arguments passed to scitex.io.load()

    Returns
    -------
    Dict[str, Any]
        Figure JSON specification

    Examples
    --------
    >>> fig_json = load_figure_json("figure.json")
    >>> fig_json["width_mm"]
    180
    """
    import scitex as stx

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Figure JSON not found: {input_path}")

    # Load using scitex.io
    fig_json = stx.io.load(input_path, **kwargs)

    # Validate if requested
    if validate:
        from ..backend._parser import validate_figure_json

        validate_figure_json(fig_json)

    return fig_json


def load_figure_json_from_project(
    project_dir: Union[str, Path],
    figure_id: str,
    subdir: str = "figs",
    **kwargs,
) -> Dict[str, Any]:
    """
    Load figure JSON from project directory structure.

    Follows the convention: project_dir/scitex/vis/{subdir}/{figure_id}.json

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    figure_id : str
        Figure identifier
    subdir : str, optional
        Subdirectory name (default: "figs")
    **kwargs
        Additional keyword arguments passed to load_figure_json()

    Returns
    -------
    Dict[str, Any]
        Figure JSON specification

    Examples
    --------
    >>> fig_json = load_figure_json_from_project(
    ...     "/path/to/project",
    ...     "fig-001"
    ... )
    """
    project_dir = Path(project_dir)
    json_path = project_dir / "scitex" / "vis" / subdir / f"{figure_id}.json"

    return load_figure_json(json_path, **kwargs)


def load_figure_model(
    input_path: Union[str, Path],
    **kwargs,
):
    """
    Load FigureModel from JSON file.

    Parameters
    ----------
    input_path : str or Path
        Input JSON file path
    **kwargs
        Additional keyword arguments passed to load_figure_json()

    Returns
    -------
    FigureModel
        Loaded figure model

    Examples
    --------
    >>> from scitex.fig.io import load_figure_model
    >>> fig_model = load_figure_model("figure.json")
    >>> fig_model.width_mm
    180
    """
    from ..model import FigureModel

    fig_json = load_figure_json(input_path, **kwargs)
    return FigureModel.from_dict(fig_json)


def list_figures_in_project(
    project_dir: Union[str, Path],
    subdir: str = "figs",
) -> list:
    """
    List all figure JSONs in a project directory.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory
    subdir : str, optional
        Subdirectory name (default: "figs")

    Returns
    -------
    list
        List of figure IDs

    Examples
    --------
    >>> list_figures_in_project("/path/to/project")
    ['fig-001', 'fig-002', 'fig-003']
    """
    project_dir = Path(project_dir)
    figs_dir = project_dir / "scitex" / "vis" / subdir

    if not figs_dir.exists():
        return []

    # Find all JSON files
    json_files = sorted(figs_dir.glob("*.json"))

    # Extract figure IDs (filename without extension)
    figure_ids = [f.stem for f in json_files]

    return figure_ids


# EOF
