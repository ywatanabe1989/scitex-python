#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_image.py

"""Image and heatmap configurations."""

from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class ImshowConfig:
    """Image display configuration."""

    img: Any
    cmap: Optional[str] = "viridis"
    aspect: str = "auto"
    interpolation: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    origin: str = "upper"
    id: Optional[str] = None


@dataclass
class ContourConfig:
    """Contour plot configuration."""

    x: List[float]
    y: List[float]
    z: Any
    levels: Optional[Union[int, List[float]]] = None
    filled: bool = False
    cmap: Optional[str] = None
    colors: Optional[str] = None
    linewidths: Optional[float] = None
    alpha: Optional[float] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    id: Optional[str] = None


@dataclass
class HeatmapConfig:
    """Heatmap configuration (scitex.plt.ax.stx_heatmap)."""

    data: Any
    x_labels: Optional[List[str]] = None
    y_labels: Optional[List[str]] = None
    cbar_label: Optional[str] = None
    cmap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    show_annot: bool = False
    value_format: str = "{x:.2f}"
    id: Optional[str] = None


__all__ = ["ImshowConfig", "ContourConfig", "HeatmapConfig"]

# EOF
