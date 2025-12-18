#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-18 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/_bundle.py

"""
Statsz - Object-oriented API for .statsz bundles.

Usage:
    from scitex.stats import Statsz

    statsz = Statsz.create("results.statsz", comparisons=[...])
    statsz = Statsz("results.statsz")
    statsz.add_comparison("A vs B", method="t-test", p_value=0.03)
    statsz.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from scitex.io.bundle import ZipBundle

__all__ = ["Statsz"]


class Statsz:
    """High-level API for .statsz bundles (zipped format)."""

    SCHEMA = {"name": "scitex.stats.stats", "version": "1.0.0"}

    def __init__(self, path: Union[str, Path]):
        """Load an existing .statsz bundle."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.path}")
        self._spec: Optional[Dict[str, Any]] = None
        self._data: Optional[pd.DataFrame] = None
        self._modified = False
        self._load()

    def _load(self) -> None:
        """Load bundle contents into memory."""
        with ZipBundle(self.path, mode="r") as zb:
            try:
                self._spec = zb.read_json("spec.json")
            except FileNotFoundError:
                self._spec = {"schema": self.SCHEMA, "comparisons": []}
            try:
                self._data = zb.read_csv("data.csv")
            except FileNotFoundError:
                self._data = None

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        comparisons: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "Statsz":
        """Create a new .statsz bundle."""
        path = Path(path)
        if path.suffix != ".statsz":
            path = path.with_suffix(".statsz")
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = cls._build_spec(comparisons or [], metadata)
        with ZipBundle(path, mode="w") as zb:
            zb.write_json("spec.json", spec)
            if data is not None:
                zb.write_csv("data.csv", data)
        return cls(path)

    @classmethod
    def _build_spec(
        cls, comparisons: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build spec dictionary."""
        return {"schema": cls.SCHEMA, "comparisons": comparisons, "metadata": metadata or {}}

    @property
    def spec(self) -> Dict[str, Any]:
        """Statistics specification dictionary."""
        return self._spec or {}

    @spec.setter
    def spec(self, value: Dict[str, Any]) -> None:
        self._spec = value
        self._modified = True

    @property
    def comparisons(self) -> List[Dict[str, Any]]:
        """List of comparison results."""
        return self.spec.get("comparisons", [])

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata dictionary."""
        return self.spec.get("metadata", {})

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        if self._spec is None:
            self._spec = {"schema": self.SCHEMA, "comparisons": []}
        self._spec["metadata"] = value
        self._modified = True

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Raw data as DataFrame."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        self._data = value
        self._modified = True

    def save(self) -> None:
        """Save changes to bundle atomically."""
        with ZipBundle(self.path, mode="a") as zb:
            if self._spec is not None:
                zb.write_json("spec.json", self._spec)
            if self._data is not None:
                zb.write_csv("data.csv", self._data)
        self._modified = False

    def add_comparison(
        self,
        name: str,
        method: str,
        p_value: float,
        effect_size: Optional[float] = None,
        ci95: Optional[List[float]] = None,
        statistic: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Add a comparison result."""
        if self._spec is None:
            self._spec = {"schema": self.SCHEMA, "comparisons": [], "metadata": {}}
        comp = {"name": name, "method": method, "p_value": p_value}
        if effect_size is not None:
            comp["effect_size"] = effect_size
        if ci95 is not None:
            comp["ci95"] = ci95
        if statistic is not None:
            comp["statistic"] = statistic
        comp.update(kwargs)
        self._spec["comparisons"].append(comp)
        self._modified = True

    def remove_comparison(self, name: str) -> None:
        """Remove a comparison by name."""
        if self._spec is None:
            return
        self._spec["comparisons"] = [c for c in self._spec.get("comparisons", []) if c.get("name") != name]
        self._modified = True

    def get_comparison(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a comparison by name."""
        for comp in self.comparisons:
            if comp.get("name") == name:
                return comp
        return None

    def get_significant(self, alpha: float = 0.05) -> List[Dict[str, Any]]:
        """Get comparisons with p < alpha."""
        return [c for c in self.comparisons if c.get("p_value", 1.0) < alpha]

    def format_comparison(self, name: str, style: str = "apa") -> str:
        """Format a comparison for publication."""
        comp = self.get_comparison(name)
        if not comp:
            return ""
        p = comp.get("p_value", 1.0)
        method = comp.get("method", "test")
        stat = comp.get("statistic")
        es = comp.get("effect_size")
        if style == "apa":
            parts = [method]
            if stat is not None:
                parts.append(f"= {stat:.2f}")
            parts.append(f"p = {p:.3f}" if p >= 0.001 else "p < .001")
            if es is not None:
                parts.append(f"d = {es:.2f}")
            return ", ".join(parts)
        return f"{method}: p={p:.3f}"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comparisons to DataFrame."""
        if not self.comparisons:
            return pd.DataFrame()
        return pd.DataFrame(self.comparisons)

    def generate_report(self) -> str:
        """Generate summary report."""
        lines = ["# Statistical Results", ""]
        if self.metadata:
            lines.append("## Metadata")
            for k, v in self.metadata.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        lines.append("## Comparisons")
        for comp in self.comparisons:
            name = comp.get("name", "Unknown")
            p = comp.get("p_value", "N/A")
            sig = "***" if isinstance(p, float) and p < 0.001 else "**" if isinstance(p, float) and p < 0.01 else "*" if isinstance(p, float) and p < 0.05 else ""
            lines.append(f"- **{name}**: p = {p} {sig}")
        return "\n".join(lines)

    def save_report(self, format: str = "md") -> None:
        """Save report to bundle."""
        report = self.generate_report()
        with ZipBundle(self.path, mode="a") as zb:
            zb.write_bytes(f"exports/report.{format}", report.encode("utf-8"))

    def __repr__(self) -> str:
        n = len(self.comparisons)
        return f"Statsz({self.path.name!r}, comparisons={n})"


# EOF
