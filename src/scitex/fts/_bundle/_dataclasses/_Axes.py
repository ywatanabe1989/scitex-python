#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/Axes.py

"""Axes - Axis configuration for plot nodes."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class Axes:
    """Axis configuration for plot nodes.

    Defines axis limits, scales, and labels.
    """

    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    xscale: str = "linear"  # linear, log, symlog
    yscale: str = "linear"
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.xlim is not None:
            result["xlim"] = list(self.xlim)
        if self.ylim is not None:
            result["ylim"] = list(self.ylim)
        if self.xscale != "linear":
            result["xscale"] = self.xscale
        if self.yscale != "linear":
            result["yscale"] = self.yscale
        if self.xlabel:
            result["xlabel"] = self.xlabel
        if self.ylabel:
            result["ylabel"] = self.ylabel
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Axes":
        xlim = data.get("xlim")
        ylim = data.get("ylim")
        return cls(
            xlim=tuple(xlim) if xlim else None,
            ylim=tuple(ylim) if ylim else None,
            xscale=data.get("xscale", "linear"),
            yscale=data.get("yscale", "linear"),
            xlabel=data.get("xlabel"),
            ylabel=data.get("ylabel"),
        )


__all__ = ["Axes"]

# EOF
