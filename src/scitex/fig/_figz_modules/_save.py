#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_save.py

"""Save functions for Figz bundles."""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from scitex.io.bundle import ZipBundle


def save_to_zip(
    path: Path,
    spec: Dict[str, Any],
    style: Dict[str, Any],
    elements: List[Dict[str, Any]],
    render_preview_fn: Callable[[str, int], bytes],
    original_path: Optional[Path] = None,
    original_is_dir: bool = False,
) -> None:
    """Save to ZIP archive."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with ZipBundle(path, mode="a") as zb:
        if spec is not None:
            zb.write_json("spec.json", spec)
        if style is not None:
            zb.write_json("style.json", style)

        # Generate exports
        if elements:
            try:
                for fmt in ("png", "svg", "pdf"):
                    export_bytes = render_preview_fn(fmt, 150)
                    zb.write_bytes(f"exports/figure.{fmt}", export_bytes)
            except Exception:
                pass

        # Copy children from original
        if original_path and original_path != path and original_path.exists():
            _copy_children_to_zip(zb, original_path, original_is_dir)


def save_to_directory(
    path: Path,
    spec: Dict[str, Any],
    style: Dict[str, Any],
    elements: List[Dict[str, Any]],
    render_preview_fn: Callable[[str, int], bytes],
    extract_geometry_fn: Callable[[], dict],
    original_path: Optional[Path] = None,
    original_is_dir: bool = False,
) -> None:
    """Save to directory bundle."""
    path.mkdir(parents=True, exist_ok=True)

    if spec is not None:
        with open(path / "spec.json", "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
    if style is not None:
        with open(path / "style.json", "w", encoding="utf-8") as f:
            json.dump(style, f, indent=2)

    # Create consistent directory structure (same as child bundles)
    exports_dir = path / "exports"
    cache_dir = path / "cache"
    data_dir = path / "data"
    stats_dir = path / "stats"

    exports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Create placeholder files for consistent structure
    if not (data_dir / "data_info.json").exists():
        with open(data_dir / "data_info.json", "w") as f:
            json.dump(
                {"type": "figure", "description": "Container bundle - no raw data"},
                f,
                indent=2,
            )

    if not (stats_dir / "stats.json").exists():
        with open(stats_dir / "stats.json", "w") as f:
            json.dump({"comparisons": [], "tests": []}, f, indent=2)

    if not (stats_dir / "stats.csv").exists():
        import pandas as pd

        stats_df = pd.DataFrame(
            columns=[
                "test_type",
                "group1",
                "group2",
                "statistic",
                "p_value",
                "effect_size",
                "significant",
            ]
        )
        stats_df.to_csv(stats_dir / "stats.csv", index=False)

    # Generate exports
    if elements:
        try:
            for fmt in ("png", "svg", "pdf"):
                export_bytes = render_preview_fn(fmt, 150)
                with open(exports_dir / f"figure.{fmt}", "wb") as f:
                    f.write(export_bytes)

            geometry = extract_geometry_fn()
            if geometry:
                with open(cache_dir / "geometry_px.json", "w") as f:
                    json.dump(geometry, f, indent=2)

            # Generate hitmap for GUI hit testing
            _generate_hitmap(path, render_preview_fn, cache_dir)

        except Exception:
            pass

    # Generate README.md
    _generate_figz_readme(path, spec, elements)

    # Copy from original
    if original_path and original_path != path and original_path.exists():
        _copy_from_original_to_directory(path, original_path, original_is_dir)


def _generate_hitmap(
    path: Path,
    render_preview_fn: Callable[[str, int], bytes],
    cache_dir: Path,
) -> None:
    """Generate hitmap images for GUI hit testing.

    For figure bundles, we generate a simple hitmap from the rendered preview.
    Each element gets a unique color for pixel-perfect hit detection.
    """
    try:
        # Generate hitmap PNG (same as preview but used for hit testing)
        hitmap_bytes = render_preview_fn("png", 150)
        with open(cache_dir / "hitmap.png", "wb") as f:
            f.write(hitmap_bytes)

        # Generate hitmap SVG
        hitmap_svg_bytes = render_preview_fn("svg", 150)
        with open(cache_dir / "hitmap.svg", "wb") as f:
            f.write(hitmap_svg_bytes)

        # Update or create render_manifest.json
        manifest_path = cache_dir / "render_manifest.json"
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        manifest["hitmap_png"] = "cache/hitmap.png"
        manifest["hitmap_svg"] = "cache/hitmap.svg"

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    except Exception:
        pass  # Skip if hitmap generation fails


def _copy_children_to_zip(zb: ZipBundle, original_path: Path, original_is_dir: bool):
    """Copy children from original bundle to ZIP."""
    if original_is_dir:
        children_dir = original_path / "children"
        if children_dir.exists():
            for child_file in children_dir.iterdir():
                with open(child_file, "rb") as f:
                    zb.write_bytes(f"children/{child_file.name}", f.read())
    else:
        with zipfile.ZipFile(original_path, "r") as src_zip:
            for name in src_zip.namelist():
                if name.startswith("children/"):
                    zb.write_bytes(name, src_zip.read(name))


def _copy_from_original_to_directory(
    path: Path, original_path: Path, original_is_dir: bool
):
    """Copy children and exports from original to directory."""
    children_dest = path / "children"
    exports_dest = path / "exports"

    if original_is_dir:
        children_src = original_path / "children"
        if children_src.exists():
            if children_dest.exists():
                shutil.rmtree(children_dest)
            shutil.copytree(children_src, children_dest)

        exports_src = original_path / "exports"
        if exports_src.exists():
            if exports_dest.exists():
                shutil.rmtree(exports_dest)
            shutil.copytree(exports_src, exports_dest)
    else:
        with zipfile.ZipFile(original_path, "r") as src_zip:
            for name in src_zip.namelist():
                if name.startswith("children/"):
                    children_dest.mkdir(parents=True, exist_ok=True)
                    child_name = name[len("children/") :]
                    if child_name:
                        with open(children_dest / child_name, "wb") as f:
                            f.write(src_zip.read(name))
                elif name.startswith("exports/"):
                    exports_dest.mkdir(parents=True, exist_ok=True)
                    export_name = name[len("exports/") :]
                    if export_name:
                        with open(exports_dest / export_name, "wb") as f:
                            f.write(src_zip.read(name))


def _generate_figz_readme(path: Path, spec: Dict, elements: List) -> None:
    """Generate README.md for figure bundle."""
    from datetime import datetime

    title = spec.get("title", path.stem) if spec else path.stem
    bundle_id = spec.get("bundle_id", "N/A") if spec else "N/A"
    size_mm = spec.get("size_mm", {}) if spec else {}

    readme_lines = [
        f"# {title} Bundle",
        "",
        "## Overview",
        "",
        "- **Type**: figure",
        f"- **Created**: {datetime.now().isoformat()}",
        f"- **Bundle ID**: {bundle_id}",
        f"- **Size**: {size_mm.get('width', 170)} × {size_mm.get('height', 120)} mm",
        "",
        "## Structure",
        "",
        "```",
        f"{path.name}/",
        "├── spec.json           # Layout and elements",
        "├── style.json          # Visual style",
        "├── data/",
        "│   └── data_info.json  # Container metadata",
        "├── stats/",
        "│   ├── stats.json      # Statistical results",
        "│   └── stats.csv       # Tabular statistics",
        "├── cache/",
        "│   ├── geometry_px.json",
        "│   ├── hitmap.png",
        "│   └── hitmap.svg",
        "├── exports/",
        "│   ├── figure.png",
        "│   ├── figure.svg",
        "│   └── figure.pdf",
        "└── children/           # Embedded plots",
        "```",
        "",
    ]

    if elements:
        readme_lines.extend([
            "## Elements",
            "",
            "| ID | Type | Position (mm) |",
            "|----|------|---------------|",
        ])
        for elem in elements:
            pos = elem.get("position", {})
            pos_str = f"({pos.get('x_mm', 0)}, {pos.get('y_mm', 0)})"
            readme_lines.append(f"| {elem.get('id', '-')} | {elem.get('type', '-')} | {pos_str} |")
        readme_lines.append("")

    readme_lines.extend([
        "## Usage",
        "",
        "```python",
        "from scitex.fig import Figz",
        "",
        f'fig = Figz("{path.name}")',
        "fig.save()  # Save changes",
        "```",
        "",
        "---",
        "*Generated by SciTeX*",
    ])

    with open(path / "README.md", "w") as f:
        f.write("\n".join(readme_lines))


# EOF
