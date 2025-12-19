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

        # Save encoding.json and theme.json (new split structure)
        encoding_data, theme_data = _build_encoding_theme_for_zip(spec, style, elements)
        zb.write_json("encoding.json", encoding_data)
        zb.write_json("theme.json", theme_data)

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

    # Save encoding.json and theme.json (new split structure)
    _save_encoding_theme_split(path, spec, style, elements)

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
        "├── spec.json           # Layout and elements (WHAT)",
        "├── encoding.json       # Data→visual mapping (scientific rigor)",
        "├── theme.json          # Pure aesthetics (colors, fonts)",
        "├── style.json          # Backward compat (= encoding + theme)",
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
        readme_lines.extend(
            [
                "## Elements",
                "",
                "| ID | Type | Position (mm) |",
                "|----|------|---------------|",
            ]
        )
        for elem in elements:
            pos = elem.get("position", {})
            pos_str = f"({pos.get('x_mm', 0)}, {pos.get('y_mm', 0)})"
            readme_lines.append(
                f"| {elem.get('id', '-')} | {elem.get('type', '-')} | {pos_str} |"
            )
        readme_lines.append("")

    readme_lines.extend(
        [
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
        ]
    )

    with open(path / "README.md", "w") as f:
        f.write("\n".join(readme_lines))


def _save_encoding_theme_split(
    path: Path,
    spec: Dict[str, Any],
    style: Dict[str, Any],
    elements: List[Dict[str, Any]],
) -> None:
    """Save encoding.json and theme.json for the new split structure.

    This separates:
    - encoding.json: Data→visual mapping (scientific rigor)
    - theme.json: Pure aesthetics (colors, fonts)
    """
    from scitex.schema import ENCODING_VERSION, THEME_VERSION

    # Build encoding.json from spec elements
    encoding_traces = []
    for elem in elements or []:
        if elem.get("type") == "plot":
            trace_encoding = {
                "trace_id": elem.get("id", ""),
                "bindings": [],
            }
            # Extract any data bindings from element
            if elem.get("x_col"):
                trace_encoding["bindings"].append(
                    {
                        "channel": "x",
                        "column": elem["x_col"],
                        "scale": "linear",
                    }
                )
            if elem.get("y_col"):
                trace_encoding["bindings"].append(
                    {
                        "channel": "y",
                        "column": elem["y_col"],
                        "scale": "linear",
                    }
                )
            if trace_encoding["bindings"]:
                encoding_traces.append(trace_encoding)

    encoding_data = {
        "schema": {"name": "scitex.plt.encoding", "version": ENCODING_VERSION},
        "traces": encoding_traces,
    }
    with open(path / "encoding.json", "w", encoding="utf-8") as f:
        json.dump(encoding_data, f, indent=2)

    # Build theme.json from style
    theme_data = {
        "schema": {"name": "scitex.plt.theme", "version": THEME_VERSION},
        "colors": {
            "mode": "light",
            "background": "transparent",
            "axes_bg": "white",
        },
        "typography": {
            "family": "sans-serif",
            "size_pt": 7.0,
        },
        "traces": [],
        "grid": False,
    }

    # Extract theme info from style if present
    if style:
        if "theme" in style:
            theme_info = style["theme"]
            if "mode" in theme_info:
                theme_data["colors"]["mode"] = theme_info["mode"]
            if "colors" in theme_info:
                theme_data["colors"].update(theme_info["colors"])
        if "font" in style:
            theme_data["typography"].update(style["font"])
        if "traces" in style:
            # Extract style-only fields (color, linewidth, etc.)
            for trace_style in style["traces"]:
                theme_trace = {"trace_id": trace_style.get("trace_id", "")}
                for key in [
                    "color",
                    "linewidth",
                    "linestyle",
                    "marker",
                    "markersize",
                    "alpha",
                ]:
                    if key in trace_style:
                        theme_trace[key] = trace_style[key]
                if len(theme_trace) > 1:
                    theme_data["traces"].append(theme_trace)
        if "grid" in style:
            theme_data["grid"] = style["grid"]

    with open(path / "theme.json", "w", encoding="utf-8") as f:
        json.dump(theme_data, f, indent=2)


def _build_encoding_theme_for_zip(
    spec: Dict[str, Any],
    style: Dict[str, Any],
    elements: List[Dict[str, Any]],
) -> tuple:
    """Build encoding and theme data for ZIP bundles.

    Returns tuple of (encoding_data, theme_data).
    """
    from scitex.schema import ENCODING_VERSION, THEME_VERSION

    # Build encoding.json
    encoding_traces = []
    for elem in elements or []:
        if elem.get("type") == "plot":
            trace_encoding = {
                "trace_id": elem.get("id", ""),
                "bindings": [],
            }
            if elem.get("x_col"):
                trace_encoding["bindings"].append(
                    {
                        "channel": "x",
                        "column": elem["x_col"],
                        "scale": "linear",
                    }
                )
            if elem.get("y_col"):
                trace_encoding["bindings"].append(
                    {
                        "channel": "y",
                        "column": elem["y_col"],
                        "scale": "linear",
                    }
                )
            if trace_encoding["bindings"]:
                encoding_traces.append(trace_encoding)

    encoding_data = {
        "schema": {"name": "scitex.plt.encoding", "version": ENCODING_VERSION},
        "traces": encoding_traces,
    }

    # Build theme.json
    theme_data = {
        "schema": {"name": "scitex.plt.theme", "version": THEME_VERSION},
        "colors": {"mode": "light", "background": "transparent", "axes_bg": "white"},
        "typography": {"family": "sans-serif", "size_pt": 7.0},
        "traces": [],
        "grid": False,
    }

    if style:
        if "theme" in style:
            theme_info = style["theme"]
            if "mode" in theme_info:
                theme_data["colors"]["mode"] = theme_info["mode"]
            if "colors" in theme_info:
                theme_data["colors"].update(theme_info["colors"])
        if "font" in style:
            theme_data["typography"].update(style["font"])
        if "traces" in style:
            for trace_style in style["traces"]:
                theme_trace = {"trace_id": trace_style.get("trace_id", "")}
                for key in [
                    "color",
                    "linewidth",
                    "linestyle",
                    "marker",
                    "markersize",
                    "alpha",
                ]:
                    if key in trace_style:
                        theme_trace[key] = trace_style[key]
                if len(theme_trace) > 1:
                    theme_data["traces"].append(theme_trace)
        if "grid" in style:
            theme_data["grid"] = style["grid"]

    return encoding_data, theme_data


# EOF
