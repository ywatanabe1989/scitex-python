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

    # Generate README.md
    try:
        _generate_figz_readme(dir_path, spec, data, basename)
    except Exception as e:
        import logging
        logging.getLogger("scitex").debug(f"Could not generate figz README: {e}")


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
    """Generate overview image for figz bundle showing all panels with hitmaps and overlays.

    Args:
        dir_path: Bundle directory path.
        spec: Bundle specification.
        data: Bundle data dictionary.
        basename: Base filename for bundle files.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image
    import numpy as np
    import warnings

    # Find all panel directories
    panel_dirs = sorted(dir_path.glob("*.pltz.d"))
    n_panels = len(panel_dirs)

    if n_panels == 0:
        return

    # Determine grid layout - 3 columns per panel (image + hitmap + overlay)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols

    # Create figure with 3 sub-columns per panel
    fig_width = 12 * n_cols
    fig_height = 4 * n_rows + 1
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

    # Title
    title = spec.get("figure", {}).get("title", basename)
    fig.suptitle(f"Figure Overview: {title}", fontsize=14, fontweight="bold", y=0.98)

    # Create nested gridspec - each panel gets 3 sub-columns
    gs = gridspec.GridSpec(n_rows, n_cols * 3, figure=fig, hspace=0.3, wspace=0.1)

    # Add each panel
    for idx, panel_dir in enumerate(panel_dirs):
        panel_id = panel_dir.stem.replace(".pltz", "")
        row = idx // n_cols
        col = (idx % n_cols) * 3  # Triple column index

        # Find PNG in panel directory (check exports/ first for layered format, then root)
        png_files = list(panel_dir.glob("exports/*.png"))
        if not png_files:
            png_files = list(panel_dir.glob("*.png"))
        main_pngs = [f for f in png_files if "_hitmap" not in f.name and "_overview" not in f.name]

        # Find hitmap PNG
        hitmap_files = list(panel_dir.glob("exports/*_hitmap.png"))
        if not hitmap_files:
            hitmap_files = list(panel_dir.glob("*_hitmap.png"))

        # Left subplot: main image
        ax_main = fig.add_subplot(gs[row, col])
        ax_main.set_title(f"Panel {panel_id}", fontweight="bold", fontsize=11)

        main_img = None
        if main_pngs:
            main_img = Image.open(main_pngs[0])
            ax_main.imshow(main_img)
        else:
            ax_main.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax_main.transAxes)
        ax_main.axis("off")

        # Middle subplot: hitmap
        ax_hitmap = fig.add_subplot(gs[row, col + 1])
        ax_hitmap.set_title(f"Hitmap {panel_id}", fontweight="bold", fontsize=11)

        hitmap_img = None
        if hitmap_files:
            hitmap_img = Image.open(hitmap_files[0])
            ax_hitmap.imshow(hitmap_img)
        else:
            ax_hitmap.text(0.5, 0.5, "No hitmap", ha="center", va="center", transform=ax_hitmap.transAxes)
        ax_hitmap.axis("off")

        # Right subplot: overlay
        ax_overlay = fig.add_subplot(gs[row, col + 2])
        ax_overlay.set_title(f"Overlay {panel_id}", fontweight="bold", fontsize=11)

        if main_img is not None:
            ax_overlay.imshow(main_img)
            if hitmap_img is not None:
                hitmap_rgba = hitmap_img.convert("RGBA")
                hitmap_array = np.array(hitmap_rgba)
                # Create semi-transparent overlay
                hitmap_array[:, :, 3] = (hitmap_array[:, :, 3] * 0.5).astype(np.uint8)
                ax_overlay.imshow(hitmap_array, alpha=0.5)
        else:
            ax_overlay.text(0.5, 0.5, "No overlay", ha="center", va="center", transform=ax_overlay.transAxes)
        ax_overlay.axis("off")

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


def _generate_figz_readme(
    dir_path: Path, spec: Dict, data: Dict, basename: str
) -> None:
    """Generate a dynamic README.md for figz bundle.

    Args:
        dir_path: Bundle directory path.
        spec: Bundle specification.
        data: Bundle data dictionary.
        basename: Base filename for bundle files.
    """
    from datetime import datetime

    # Extract figure info
    figure = spec.get("figure", {})
    title = figure.get("title", basename)
    caption = figure.get("caption", "")
    styles = figure.get("styles", {})
    size = styles.get("size", {})
    width_mm = size.get("width_mm", 0)
    height_mm = size.get("height_mm", 0)
    background = styles.get("background", "#ffffff")

    # Count panels
    panels = spec.get("panels", [])
    n_panels = len(panels)

    # Find panel directories
    panel_dirs = sorted(dir_path.glob("*.pltz.d"))

    # Build panel table
    panel_rows = ""
    for panel in panels:
        panel_id = panel.get("id", "?")
        label = panel.get("label", panel_id)
        plot_ref = panel.get("plot", "")
        pos = panel.get("position", {})
        panel_size = panel.get("size", {})
        panel_rows += f"| {label} | {plot_ref} | ({pos.get('x_mm', 0)}, {pos.get('y_mm', 0)}) | {panel_size.get('width_mm', 0)} × {panel_size.get('height_mm', 0)} mm |\n"

    # Build panel directory list
    panel_dir_list = ""
    for pd in panel_dirs:
        panel_dir_list += f"│   ├── {pd.name}/\n"

    readme_content = f"""# {basename}.figz.d

> SciTeX Figure Bundle - Auto-generated README

## Overview

![Figure Overview]({basename}_overview.png)

## Bundle Structure

```
{basename}.figz.d/
├── {basename}.json         # Figure specification (panels, layout)
├── {basename}.png          # Rendered figure (raster)
├── {basename}.svg          # Rendered figure (vector)
├── {basename}_overview.png # Visual summary with hitmaps
{panel_dir_list}└── README.md              # This file
```

## Figure Information

| Property | Value |
|----------|-------|
| Title | {title or '(none)'} |
| Panels | {n_panels} |
| Size | {width_mm:.1f} × {height_mm:.1f} mm |
| Background | `{background}` |

{f"**Caption**: {caption}" if caption else ""}

## Panel Layout

| Label | Plot Bundle | Position (x, y) | Size |
|-------|-------------|-----------------|------|
{panel_rows}

## Nested Bundles

Each panel is stored as a separate `.pltz.d` bundle containing:
- `spec.json` - What to plot (data, axes, traces)
- `style.json` - How it looks (colors, fonts, theme)
- `exports/` - Rendered images (PNG, SVG, hitmap)
- `cache/` - Computed geometry (regenerable)

## Usage

### Python

```python
import scitex as stx

# Load the figure bundle
bundle = stx.load("{dir_path}")

# Access components
spec = bundle["spec"]       # Figure layout
plots = bundle["plots"]     # Dict of panel bundles

# Access specific panel
panel_a = plots["A"]        # Get panel A's pltz bundle
```

### Editing

Edit `{basename}.json` to change:
- Panel positions and sizes
- Figure title and caption
- Background color

Edit individual `*.pltz.d/spec.json` to change:
- Plot data and axes
- Trace specifications

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Schema: {spec.get("schema", {}).get("name", "scitex.fig.figure")} v{spec.get("schema", {}).get("version", "1.0.0")}*
"""

    readme_path = dir_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)


# EOF
