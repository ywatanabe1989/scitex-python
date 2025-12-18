#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-18 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_bundle.py

"""
Pltz - Object-oriented API for .pltz bundles.

Usage:
    from scitex.plt import Pltz

    pltz = Pltz.create("plot.pltz", plot_type="line", data=df)
    pltz = Pltz("plot.pltz")
    pltz.spec["axes"]["xlabel"] = "Time (s)"
    pltz.save()
    png_bytes = pltz.render_preview()
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from scitex.io.bundle import ZipBundle

__all__ = ["Pltz"]


class Pltz:
    """High-level API for .pltz bundles (zipped format)."""

    SCHEMA = {"name": "scitex.plt.plot", "version": "1.0.0"}

    def __init__(self, path: Union[str, Path]):
        """Load an existing .pltz bundle."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.path}")
        self._spec: Optional[Dict[str, Any]] = None
        self._style: Optional[Dict[str, Any]] = None
        self._data: Optional[pd.DataFrame] = None
        self._modified = False
        self._load()

    def _load(self) -> None:
        """Load bundle contents into memory."""
        with ZipBundle(self.path, mode="r") as zb:
            try:
                self._spec = zb.read_json("spec.json")
            except FileNotFoundError:
                self._spec = {"schema": self.SCHEMA}
            try:
                self._style = zb.read_json("style.json")
            except FileNotFoundError:
                self._style = None
            try:
                self._data = zb.read_csv("data.csv")
            except FileNotFoundError:
                self._data = None

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        plot_type: str,
        data: Optional[pd.DataFrame] = None,
        spec_overrides: Optional[Dict[str, Any]] = None,
    ) -> "Pltz":
        """Create a new .pltz bundle."""
        path = Path(path)
        if path.suffix != ".pltz":
            path = path.with_suffix(".pltz")
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = cls._build_spec(plot_type, spec_overrides)
        with ZipBundle(path, mode="w") as zb:
            zb.write_json("spec.json", spec)
            if data is not None:
                zb.write_csv("data.csv", data)
        return cls(path)

    @classmethod
    def create_from_gallery(
        cls, path: Union[str, Path], category: str, plot_name: str
    ) -> "Pltz":
        """Create a .pltz bundle from gallery template."""
        from scitex.plt.gallery import get_plot_spec, get_plot_data

        path = Path(path)
        if path.suffix != ".pltz":
            path = path.with_suffix(".pltz")
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = get_plot_spec(category, plot_name)
        data = get_plot_data(category, plot_name)
        with ZipBundle(path, mode="w") as zb:
            zb.write_json("spec.json", spec)
            if data is not None:
                zb.write_csv("data.csv", data)
        return cls(path)

    @classmethod
    def _build_spec(
        cls, plot_type: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build spec dictionary for plot type."""
        spec = {
            "schema": cls.SCHEMA,
            "plot_type": plot_type,
            "axes": {"xlabel": "", "ylabel": ""},
        }
        if overrides:
            spec.update(overrides)
        return spec

    @property
    def spec(self) -> Dict[str, Any]:
        """Plot specification dictionary."""
        return self._spec or {}

    @spec.setter
    def spec(self, value: Dict[str, Any]) -> None:
        self._spec = value
        self._modified = True

    @property
    def style(self) -> Optional[Dict[str, Any]]:
        """Style dictionary (appearance)."""
        return self._style

    @style.setter
    def style(self, value: Dict[str, Any]) -> None:
        self._style = value
        self._modified = True

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Plot data as DataFrame."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        self._data = value
        self._modified = True

    @property
    def plot_type(self) -> str:
        """Plot type from spec."""
        return self.spec.get("plot_type", "unknown")

    def save(self) -> None:
        """Save changes to bundle atomically."""
        with ZipBundle(self.path, mode="a") as zb:
            if self._spec is not None:
                zb.write_json("spec.json", self._spec)
            if self._style is not None:
                zb.write_json("style.json", self._style)
            if self._data is not None:
                zb.write_csv("data.csv", self._data)
        self._modified = False

    def render_preview(self, format: str = "png", dpi: int = 150) -> bytes:
        """Render plot preview and return image bytes."""
        import scitex as stx
        import matplotlib.pyplot as plt_mpl
        from scitex.plt.gallery._plots import PLOT_FUNCTIONS
        from scitex.plt.styles.presets import SCITEX_STYLE

        plot_type = self.spec.get("plot_type", "plot")
        category = self.spec.get("category", "line")

        # Create figure with SciTeX style
        style = SCITEX_STYLE.copy()
        style["figsize"] = (4, 3)
        fig, ax = stx.plt.subplots(**style)

        # Try to render using gallery plot function
        if plot_type in PLOT_FUNCTIONS:
            plot_func = PLOT_FUNCTIONS[plot_type]
            fig, ax = plot_func(fig, ax, stx)
        else:
            # Fallback: basic line plot if data available
            if self._data is not None and not self._data.empty:
                # Find numeric columns for simple plot
                numeric_cols = self._data.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    ax.plot(self._data[numeric_cols[0]], self._data[numeric_cols[1]])
                elif len(numeric_cols) == 1:
                    ax.plot(self._data[numeric_cols[0]])

        # Render to bytes
        buffer = io.BytesIO()
        fig_mpl = fig._fig_mpl if hasattr(fig, "_fig_mpl") else fig
        fig_mpl.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        plt_mpl.close(fig_mpl)
        return buffer.getvalue()

    def get_preview(self) -> Optional[bytes]:
        """Get cached preview from bundle if available."""
        try:
            with ZipBundle(self.path, mode="r") as zb:
                return zb.read_bytes("exports/preview.png")
        except FileNotFoundError:
            return None

    def update_preview(self, dpi: int = 150) -> None:
        """Update cached preview in bundle."""
        preview = self.render_preview(dpi=dpi)
        with ZipBundle(self.path, mode="a") as zb:
            zb.write_bytes("exports/preview.png", preview)

    def __repr__(self) -> str:
        return f"Pltz({self.path.name!r}, plot_type={self.plot_type!r})"


# EOF
