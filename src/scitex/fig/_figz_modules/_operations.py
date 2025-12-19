#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_operations.py

"""Pack/unpack and layout operations for Figz bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from scitex.fig._bundle import Figz


def auto_crop_figz(
    elements: List[Dict[str, Any]],
    size_mm: Dict[str, float],
    spec: Dict[str, Any],
    margin_mm: float = 5.0,
) -> Dict[str, Any]:
    """Auto-crop figure to content bounds.

    Args:
        elements: List of elements
        size_mm: Current canvas size
        spec: Bundle spec to update
        margin_mm: Margin around content

    Returns:
        Dict with original_size, new_size, offset, bounds
    """
    from scitex.fig.layout import auto_crop_layout, content_bounds

    if not elements:
        return {
            "original_size": size_mm.copy(),
            "new_size": size_mm.copy(),
            "offset": {"x_mm": 0, "y_mm": 0},
            "bounds": None,
        }
    orig = size_mm.copy()
    bounds = content_bounds(elements)
    shifted, new_sz = auto_crop_layout(elements, margin_mm)
    offset = (
        {"x_mm": bounds["x_mm"] - margin_mm, "y_mm": bounds["y_mm"] - margin_mm}
        if bounds
        else {"x_mm": 0, "y_mm": 0}
    )
    spec["elements"] = shifted
    spec["size_mm"] = {"width": new_sz["width_mm"], "height": new_sz["height_mm"]}
    return {
        "original_size": orig,
        "new_size": {"width": new_sz["width_mm"], "height": new_sz["height_mm"]},
        "offset": offset,
        "bounds": bounds,
    }


def pack_bundle(figz: Figz, output_path: Path = None) -> Figz:
    """Pack directory bundle to ZIP.

    Args:
        figz: Figz instance to pack
        output_path: Optional output path

    Returns:
        New Figz instance for packed bundle
    """
    from scitex.fig._bundle import Figz
    from scitex.io.bundle import pack as bundle_pack

    if not figz._is_dir:
        raise ValueError("Bundle is already ZIP")
    output_path = (
        Path(output_path) if output_path else figz.path.parent / figz.path.stem
    )
    figz.save()
    bundle_pack(figz.path, output_path)
    return Figz(output_path)


def unpack_bundle(figz: Figz, output_path: Path = None) -> Figz:
    """Unpack ZIP bundle to directory.

    Args:
        figz: Figz instance to unpack
        output_path: Optional output path

    Returns:
        New Figz instance for unpacked bundle
    """
    from scitex.fig._bundle import Figz
    from scitex.io.bundle import unpack as bundle_unpack

    if figz._is_dir:
        raise ValueError("Bundle is already directory")
    output_path = (
        Path(output_path) if output_path else figz.path.parent / f"{figz.path.name}.d"
    )
    # Don't save() here - it would add non-prefixed entries to ZIP
    # Use bundle_unpack which handles the top-level directory in ZIP correctly
    bundle_unpack(figz.path, output_path)
    return Figz(output_path)


# EOF
