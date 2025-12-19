#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_fsb_bridge.py

"""
FSB Bridge for Figz.

Provides integration between scitex.fig.Figz and fsb.Bundle.
FSB (Figure-Statistics Bundle) is the single source of truth
for bundle specification.

Usage:
    from scitex.fig import Figz
    from scitex.fig._fsb_bridge import to_fsb, from_fsb

    # Convert Figz to FSB Bundle
    figz = Figz("my_figure.zip", name="My Figure")
    bundle = to_fsb(figz, "my_figure_fsb")

    # Create Figz from FSB Bundle
    bundle = fsb.Bundle("plot", create=True, node_type="plot")
    figz = from_fsb(bundle, "output.zip")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from scitex.io.bundle import fsb

if TYPE_CHECKING:
    from ._bundle import Figz

__all__ = ["to_fsb", "from_fsb", "FigzFSBMixin"]


def to_fsb(
    figz: Figz,
    output_path: Optional[Union[str, Path]] = None,
    save: bool = True,
) -> fsb.Bundle:
    """Convert Figz instance to FSB Bundle.

    Args:
        figz: Source Figz instance.
        output_path: Path for FSB bundle. Uses figz path stem if None.
        save: Whether to save the bundle after creation.

    Returns:
        FSB Bundle instance.
    """
    if not fsb.FSB_AVAILABLE:
        raise ImportError("FSB package not available. Install with: pip install fsb")

    # Determine output path
    if output_path is None:
        output_path = figz.path.parent / f"{figz.path.stem}_fsb"
    output_path = Path(output_path)

    # Convert spec to FSB format
    fsb_data = fsb.from_scitex_spec(figz.spec, figz.style)

    # Create bundle
    bundle = fsb.Bundle(
        output_path,
        create=True,
        node_type=figz.bundle_type,
        name=figz.spec.get("title"),
        size_mm=figz.size_mm,
    )

    # Apply encoding and theme
    bundle.encoding = fsb_data["encoding"]
    bundle.theme = fsb_data["theme"]

    # Convert elements to traces
    _convert_elements_to_traces(figz.elements, bundle)

    if save:
        bundle.save()

    return bundle


def from_fsb(
    bundle: fsb.Bundle,
    output_path: Union[str, Path],
    save: bool = True,
) -> Figz:
    """Create Figz instance from FSB Bundle.

    Args:
        bundle: Source FSB Bundle.
        output_path: Path for new Figz bundle.
        save: Whether to save after creation.

    Returns:
        Figz instance.
    """
    from ._bundle import Figz

    if not fsb.FSB_AVAILABLE:
        raise ImportError("FSB package not available. Install with: pip install fsb")

    output_path = Path(output_path)

    # Convert FSB to scitex format
    spec, style = fsb.to_scitex_spec(bundle)

    # Create Figz
    figz = Figz(
        output_path,
        name=spec.get("title", "Untitled"),
        size_mm=spec.get("size_mm"),
        bundle_type=bundle.bundle_type,
    )

    # Update spec and style
    figz._spec.update(spec)
    figz._style = style

    if save:
        figz.save(verbose=False)

    return figz


def _convert_elements_to_traces(elements: list, bundle: fsb.Bundle) -> None:
    """Convert Figz elements to FSB traces.

    Maps scitex element types to FSB trace encodings.
    """
    traces = []

    for elem in elements:
        elem_type = elem.get("type", "")

        if elem_type in ("plot", "trace"):
            trace = {
                "trace_id": elem.get("id", f"trace_{len(traces)}"),
            }

            # Copy relevant fields
            for key in ["data_ref", "x", "y", "color", "size", "group"]:
                if key in elem:
                    trace[key] = elem[key]

            # Handle plot references
            if "ref" in elem:
                trace["data_ref"] = elem["ref"]

            traces.append(trace)

    if traces:
        encoding = bundle.encoding or {}
        encoding["traces"] = traces
        bundle.encoding = encoding


class FigzFSBMixin:
    """Mixin to add FSB support to Figz class.

    Add this mixin to Figz to enable direct FSB Bundle access:

        class Figz(FigzFSBMixin, FigzCaptionMixin, ...):
            ...
    """

    def to_fsb(
        self,
        output_path: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> fsb.Bundle:
        """Convert this Figz to an FSB Bundle.

        Args:
            output_path: Path for FSB bundle.
            save: Whether to save the bundle.

        Returns:
            FSB Bundle instance.
        """
        return to_fsb(self, output_path, save)

    @classmethod
    def from_fsb(
        cls,
        bundle: fsb.Bundle,
        output_path: Union[str, Path],
        save: bool = True,
    ) -> Figz:
        """Create Figz from an FSB Bundle.

        Args:
            bundle: Source FSB Bundle.
            output_path: Path for Figz bundle.
            save: Whether to save after creation.

        Returns:
            Figz instance.
        """
        return from_fsb(bundle, output_path, save)

    @property
    def as_fsb_dict(self) -> Dict[str, Any]:
        """Get this Figz as FSB-compatible dictionary.

        Returns:
            Dictionary with 'node', 'encoding', 'theme' keys.
        """
        if not fsb.FSB_AVAILABLE:
            raise ImportError("FSB package not available")
        return fsb.from_scitex_spec(self.spec, self.style)


# EOF
