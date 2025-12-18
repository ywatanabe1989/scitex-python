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

    exports_dir = path / "exports"
    cache_dir = path / "cache"
    exports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

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
        except Exception:
            pass

    # Copy from original
    if original_path and original_path != path and original_path.exists():
        _copy_from_original_to_directory(path, original_path, original_is_dir)


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


# EOF
