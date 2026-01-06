#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/backend/parser.py
"""Parse figure JSON to Python object models."""

from typing import Dict, Any
from ..model import (
    FigureModel,
    AxesModel,
    PlotModel,
    GuideModel,
    AnnotationModel,
)


def parse_figure_json(fig_json: Dict[str, Any]) -> FigureModel:
    """
    Parse figure JSON into FigureModel.

    Parameters
    ----------
    fig_json : Dict[str, Any]
        Figure JSON specification

    Returns
    -------
    FigureModel
        Parsed figure model

    Examples
    --------
    >>> fig_json = {
    ...     "width_mm": 180,
    ...     "height_mm": 120,
    ...     "nrows": 1,
    ...     "ncols": 2,
    ...     "axes": [...]
    ... }
    >>> fig_model = parse_figure_json(fig_json)
    """
    # Parse nested axes
    axes_data = fig_json.get("axes", [])
    parsed_axes = [parse_axes_json(ax) for ax in axes_data]

    # Create FigureModel with parsed axes
    fig_data = fig_json.copy()
    fig_data["axes"] = [ax.to_dict() for ax in parsed_axes]

    return FigureModel.from_dict(fig_data)


def parse_axes_json(axes_json: Dict[str, Any]) -> AxesModel:
    """
    Parse axes JSON into AxesModel.

    Parameters
    ----------
    axes_json : Dict[str, Any]
        Axes JSON specification

    Returns
    -------
    AxesModel
        Parsed axes model
    """
    # Parse nested plots
    plots_data = axes_json.get("plots", [])
    parsed_plots = [parse_plot_json(plot) for plot in plots_data]

    # Parse nested annotations
    annotations_data = axes_json.get("annotations", [])
    parsed_annotations = [parse_annotation_json(ann) for ann in annotations_data]

    # Parse nested guides
    guides_data = axes_json.get("guides", [])
    parsed_guides = [parse_guide_json(guide) for guide in guides_data]

    # Create AxesModel with parsed children
    ax_data = axes_json.copy()
    ax_data["plots"] = [p.to_dict() for p in parsed_plots]
    ax_data["annotations"] = [a.to_dict() for a in parsed_annotations]
    ax_data["guides"] = [g.to_dict() for g in parsed_guides]

    return AxesModel.from_dict(ax_data)


def parse_plot_json(plot_json: Dict[str, Any]) -> PlotModel:
    """
    Parse plot JSON into PlotModel.

    Parameters
    ----------
    plot_json : Dict[str, Any]
        Plot JSON specification

    Returns
    -------
    PlotModel
        Parsed plot model
    """
    return PlotModel.from_dict(plot_json)


def parse_guide_json(guide_json: Dict[str, Any]) -> GuideModel:
    """
    Parse guide JSON into GuideModel.

    Parameters
    ----------
    guide_json : Dict[str, Any]
        Guide JSON specification

    Returns
    -------
    GuideModel
        Parsed guide model
    """
    return GuideModel.from_dict(guide_json)


def parse_annotation_json(annotation_json: Dict[str, Any]) -> AnnotationModel:
    """
    Parse annotation JSON into AnnotationModel.

    Parameters
    ----------
    annotation_json : Dict[str, Any]
        Annotation JSON specification

    Returns
    -------
    AnnotationModel
        Parsed annotation model
    """
    return AnnotationModel.from_dict(annotation_json)


def validate_figure_json(fig_json: Dict[str, Any]) -> bool:
    """
    Validate figure JSON structure (single entry point for validation).

    This is the recommended validation function that performs:
    1. Basic JSON structure validation (utils.validate_json_structure)
    2. Model parsing and validation (FigureModel.validate)
    3. Axes layout validation (utils.validate_axes_layout)

    Parameters
    ----------
    fig_json : Dict[str, Any]
        Figure JSON to validate

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise

    Raises
    ------
    ValueError
        If JSON structure is invalid

    Examples
    --------
    >>> from scitex.canvas.backend import validate_figure_json
    >>> fig_json = {"width_mm": 180, "height_mm": 120, "axes": []}
    >>> validate_figure_json(fig_json)
    True
    """
    from ..utils import validate_json_structure, validate_axes_layout

    # Step 1: Validate basic JSON structure
    validate_json_structure(fig_json)

    # Step 2: Parse to models and validate
    fig_model = parse_figure_json(fig_json)
    fig_model.validate()

    # Step 3: Validate axes layout consistency
    validate_axes_layout(
        nrows=fig_model.nrows,
        ncols=fig_model.ncols,
        num_axes=len(fig_model.axes),
    )

    return True


# EOF
