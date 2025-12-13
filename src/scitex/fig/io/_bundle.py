#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/io/_bundle.py

"""
SciTeX .figz Bundle I/O - Figure-specific bundle operations.

Handles:
    - Figure specification validation
    - Panel composition and layout
    - Nested .pltz bundle management
    - Export file handling
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

__all__ = [
    "validate_figz_spec",
    "load_figz_bundle",
    "save_figz_bundle",
    "FIGZ_SCHEMA_SPEC",
]

# Schema specification for .figz bundles
FIGZ_SCHEMA_SPEC = {
    "name": "scitex.fig.figure",
    "version": "1.0.0",
    "required_fields": ["schema"],
    "optional_fields": ["figure", "panels", "notations"],
}


def validate_figz_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate .figz-specific fields.

    Args:
        spec: The specification dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if "panels" in spec:
        panels = spec["panels"]
        if not isinstance(panels, list):
            errors.append("'panels' must be a list")
        else:
            for i, panel in enumerate(panels):
                if not isinstance(panel, dict):
                    errors.append(f"panels[{i}] must be a dictionary")
                    continue
                if "id" not in panel:
                    errors.append(f"panels[{i}].id is required")

    if "figure" in spec:
        figure = spec["figure"]
        if not isinstance(figure, dict):
            errors.append("'figure' must be a dictionary")

    return errors


def load_figz_bundle(bundle_dir: Path) -> Dict[str, Any]:
    """Load .figz bundle contents from directory.

    Args:
        bundle_dir: Path to the bundle directory.

    Returns:
        Dictionary with loaded bundle contents.
    """
    result = {}

    # Find the spec file (could be figure.json or {basename}.json)
    spec_file = None
    for f in bundle_dir.glob("*.json"):
        if not f.name.startswith('.'):  # Skip hidden files
            spec_file = f
            break

    if spec_file and spec_file.exists():
        with open(spec_file, "r") as f:
            result["spec"] = json.load(f)
        result["basename"] = spec_file.stem
    else:
        result["spec"] = None
        result["basename"] = "figure"

    # Load nested .pltz bundles
    result["plots"] = {}

    # Load from .pltz.d directories
    for pltz_dir in bundle_dir.glob("*.pltz.d"):
        plot_name = pltz_dir.stem.replace(".pltz", "")
        from scitex.io._bundle import load_bundle
        result["plots"][plot_name] = load_bundle(pltz_dir)

    # Load from .pltz ZIP files
    for pltz_zip in bundle_dir.glob("*.pltz"):
        if pltz_zip.is_file():
            plot_name = pltz_zip.stem
            from scitex.io._bundle import load_bundle
            result["plots"][plot_name] = load_bundle(pltz_zip)

    return result


def save_figz_bundle(data: Dict[str, Any], dir_path: Path) -> None:
    """Save .figz bundle contents to directory.

    Args:
        data: Bundle data dictionary.
        dir_path: Path to the bundle directory.
    """
    # Get basename from directory name (e.g., "Figure1" from "Figure1.figz.d")
    basename = dir_path.stem.replace(".figz", "")

    # Save specification with proper basename
    spec = data.get("spec", {})
    spec_file = dir_path / f"{basename}.json"
    with open(spec_file, "w") as f:
        json.dump(spec, f, indent=2)

    # Save exports (PNG, SVG, PDF) with proper basename
    _save_exports(data, dir_path, spec, basename)

    # Copy nested .pltz bundles directly (preserving all files)
    if "plots" in data:
        _copy_nested_pltz_bundles(data["plots"], dir_path)

    # Generate figz overview
    try:
        _generate_figz_overview(dir_path, spec, data, basename)
    except Exception as e:
        import logging
        logging.getLogger("scitex").debug(f"Could not generate figz overview: {e}")


def _save_exports(data: Dict[str, Any], dir_path: Path, spec: Dict, basename: str = "figure") -> None:
    """Save export files (PNG, SVG, PDF) with embedded metadata."""
    for fmt in ["png", "svg", "pdf"]:
        if fmt not in data:
            continue

        out_file = dir_path / f"{basename}.{fmt}"
        export_data = data[fmt]

        if isinstance(export_data, bytes):
            with open(out_file, "wb") as f:
                f.write(export_data)
        elif isinstance(export_data, (str, Path)) and Path(export_data).exists():
            shutil.copy(export_data, out_file)

        # Embed metadata into PNG and PDF files
        if out_file.exists() and spec:
            try:
                _embed_metadata_in_export(out_file, spec, fmt)
            except Exception as e:
                import logging
                logging.getLogger("scitex").debug(
                    f"Could not embed metadata in {out_file}: {e}"
                )


def _copy_nested_pltz_bundles(plots: Dict[str, Any], dir_path: Path) -> None:
    """Copy nested .pltz bundles directly, preserving all files.

    Args:
        plots: Dict mapping panel IDs to either:
            - source_path: Path to existing .pltz.d directory
            - bundle_data: Dict with spec/data (will use save_bundle)
        dir_path: Target figz directory.
    """
    for panel_id, plot_source in plots.items():
        target_path = dir_path / f"{panel_id}.pltz.d"

        if isinstance(plot_source, (str, Path)):
            # Direct copy from source path
            source_path = Path(plot_source)
            if source_path.exists() and source_path.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
        elif isinstance(plot_source, dict):
            # Check if it has source_path for direct copy
            if "source_path" in plot_source:
                source_path = Path(plot_source["source_path"])
                if source_path.exists() and source_path.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path)
            else:
                # Fallback to save_bundle (will lose images)
                from scitex.io._bundle import save_bundle, BundleType
                save_bundle(plot_source, target_path, bundle_type=BundleType.PLTZ)


def _generate_figz_overview(dir_path: Path, spec: Dict, data: Dict, basename: str) -> None:
    """Generate overview image for figz bundle showing all panels.

    Args:
        dir_path: Bundle directory path.
        spec: Bundle specification.
        data: Bundle data dictionary.
        basename: Base filename for bundle files.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image
    import warnings

    # Find all panel directories
    panel_dirs = sorted(dir_path.glob("*.pltz.d"))
    n_panels = len(panel_dirs)

    if n_panels == 0:
        return

    # Determine grid layout
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    # Create figure
    fig_width = 5 * n_cols
    fig_height = 4 * n_rows + 1
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # Title
    title = spec.get("figure", {}).get("title", basename)
    fig.suptitle(f"Figure Overview: {title}", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    # Add each panel
    for idx, panel_dir in enumerate(panel_dirs):
        panel_id = panel_dir.stem.replace(".pltz", "")
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(gs[row, col])
        ax.set_title(f"Panel {panel_id}", fontweight="bold", fontsize=11)

        # Find PNG in panel directory
        png_files = list(panel_dir.glob("*.png"))
        png_files = [f for f in png_files if "_hitmap" not in f.name and "_overview" not in f.name]

        if png_files:
            img = Image.open(png_files[0])
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax.transAxes)

        ax.axis("off")

    # Save overview
    overview_path = dir_path / f"{basename}_overview.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.savefig(overview_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _embed_metadata_in_export(
    file_path: Path, spec: Dict[str, Any], fmt: str
) -> None:
    """Embed bundle spec metadata into exported image files."""
    from scitex.io._metadata import embed_metadata

    embed_data = {
        "scitex_bundle": True,
        "schema": spec.get("schema", {}),
    }

    for key in ["figure", "panels", "notations"]:
        if key in spec:
            embed_data[key] = spec[key]

    if fmt in ("png", "pdf"):
        embed_metadata(str(file_path), embed_data)


# EOF
