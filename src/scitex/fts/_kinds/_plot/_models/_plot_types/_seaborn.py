#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_seaborn.py

"""Seaborn plot configurations."""

from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class SeabornBoxplotConfig:
    """Seaborn boxplot configuration."""

    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    palette: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SeabornViolinplotConfig:
    """Seaborn violinplot configuration."""

    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    palette: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SeabornScatterplotConfig:
    """Seaborn scatterplot configuration."""

    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None
    palette: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SeabornLineplotConfig:
    """Seaborn lineplot configuration."""

    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    style: Optional[str] = None
    palette: Optional[str] = None
    id: Optional[str] = None


@dataclass
class SeabornHistplotConfig:
    """Seaborn histplot configuration."""

    x: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    bins: Union[int, str] = "auto"
    kde: bool = False
    alpha: Optional[float] = None
    id: Optional[str] = None


@dataclass
class SeabornBarplotConfig:
    """Seaborn barplot configuration."""

    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    palette: Optional[str] = None
    estimator: str = "mean"
    id: Optional[str] = None


@dataclass
class SeabornStripplotConfig:
    """Seaborn stripplot configuration."""

    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    alpha: Optional[float] = None
    id: Optional[str] = None


@dataclass
class SeabornKDEplotConfig:
    """Seaborn KDE plot configuration."""

    x: Optional[str] = None
    data: Optional[Any] = None
    hue: Optional[str] = None
    fill: bool = False


__all__ = [
    "SeabornBoxplotConfig",
    "SeabornViolinplotConfig",
    "SeabornScatterplotConfig",
    "SeabornLineplotConfig",
    "SeabornHistplotConfig",
    "SeabornBarplotConfig",
    "SeabornStripplotConfig",
    "SeabornKDEplotConfig",
]

# EOF
