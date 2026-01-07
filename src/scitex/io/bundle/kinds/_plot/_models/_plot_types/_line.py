#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_line.py

"""Line plot configurations."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LinePlotConfig:
    """Line plot configuration."""

    x: List[float]
    y: List[float]
    color: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = "-"
    marker: Optional[str] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None
    label: Optional[str] = None
    id: Optional[str] = None


@dataclass
class PlotLineConfig:
    """Plot line configuration (scitex.plt.ax.stx_line)."""

    y: List[float]
    x: Optional[List[float]] = None
    color: Optional[str] = None
    linewidth_mm: Optional[float] = None
    linestyle: Optional[str] = "-"
    label: Optional[str] = None
    id: Optional[str] = None


@dataclass
class ShadedLineConfig:
    """Shaded line configuration (scitex.plt.ax.stx_shaded_line)."""

    x: List[float]
    y_lower: List[float]
    y_middle: List[float]
    y_upper: List[float]
    color: Optional[str] = None
    alpha: Optional[float] = 0.3
    linewidth_mm: Optional[float] = None
    label: Optional[str] = None
    id: Optional[str] = None


__all__ = ["LinePlotConfig", "PlotLineConfig", "ShadedLineConfig"]

# EOF
