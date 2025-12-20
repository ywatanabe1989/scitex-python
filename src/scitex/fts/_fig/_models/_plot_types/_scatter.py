#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_scatter.py

"""Scatter plot configurations."""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class ScatterPlotConfig:
    """Scatter plot configuration."""

    x: List[float]
    y: List[float]
    color: Optional[str] = None
    size_mm: Optional[float] = None
    s: Optional[float] = None
    marker: Optional[str] = "o"
    alpha: Optional[float] = None
    cmap: Optional[str] = None
    c: Optional[Union[str, List[float]]] = None
    label: Optional[str] = None
    id: Optional[str] = None


__all__ = ["ScatterPlotConfig"]

# EOF
