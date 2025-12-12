#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-12 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_bundle.py
# ----------------------------------------

"""
SciTeX Bundle I/O for .figz, .pltz, and .statsz formats.

Bundle formats:
    .figz  - Publication Figure Bundle (panels + layout)
    .pltz  - Reproducible Plot Bundle (data + spec + exports)
    .statsz - Statistical Results Bundle (stats + metadata)

Each bundle can be:
    - Directory form: Figure1.figz.d/
    - ZIP archive form: Figure1.figz
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "is_bundle",
    "load_bundle",
    "save_bundle",
    "pack_bundle",
    "unpack_bundle",
    "validate_bundle",
    "validate_spec",
    "BundleType",
    "BundleValidationError",
]

# Bundle extensions
BUNDLE_EXTENSIONS = (".figz", ".pltz", ".statsz")


class BundleValidationError(ValueError):
    """Error raised when bundle validation fails."""
    pass


class BundleType:
    """Bundle type constants."""
    FIGZ = "figz"
    PLTZ = "pltz"
    STATSZ = "statsz"


def _embed_metadata_in_export(file_path: Path, spec: Dict[str, Any], fmt: str) -> None:
    """Embed bundle spec metadata into exported image files.

    This allows individual PNG/PDF files to carry their metadata
    when extracted from the bundle.

    Args:
        file_path: Path to the exported file.
        spec: The bundle specification to embed.
        fmt: File format ('png', 'svg', 'pdf').
    """
    from ._metadata import embed_metadata

    # Create a simplified metadata dict for embedding
    embed_data = {
        "scitex_bundle": True,
        "schema": spec.get("schema", {}),
    }

    # Add key fields based on what's available
    for key in ["plot_type", "backend", "size", "axes", "figure", "panels"]:
        if key in spec:
            embed_data[key] = spec[key]

    if fmt == 'png':
        embed_metadata(str(file_path), embed_data)
    elif fmt == 'pdf':
        embed_metadata(str(file_path), embed_data)
    # SVG embedding is handled differently (XML comment) - skip for now


# Expected schema names and versions
SCHEMA_SPECS = {
    BundleType.FIGZ: {
        "name": "scitex.fig.figure",
        "version": "1.0.0",
        "required_fields": ["schema"],
        "optional_fields": ["figure", "panels", "notations"],
    },
    BundleType.PLTZ: {
        "name": "scitex.plt.plot",
        "version": "1.0.0",
        "required_fields": ["schema"],
        "optional_fields": ["backend", "plot_type", "data", "axes", "styles", "stats"],
    },
    BundleType.STATSZ: {
        "name": "scitex.stats.stats",
        "version": "1.0.0",
        "required_fields": ["schema"],
        "optional_fields": ["comparisons", "metadata"],
    },
}


def validate_spec(spec: Dict[str, Any], bundle_type: str, strict: bool = False) -> List[str]:
    """Validate a bundle specification against its schema.

    Args:
        spec: The specification dictionary to validate.
        bundle_type: Bundle type ('figz', 'pltz', 'statsz').
        strict: If True, raise BundleValidationError on failure.

    Returns:
        List of validation warning/error messages (empty if valid).

    Raises:
        BundleValidationError: If strict=True and validation fails.
    """
    errors = []

    if bundle_type not in SCHEMA_SPECS:
        errors.append(f"Unknown bundle type: {bundle_type}")
        if strict:
            raise BundleValidationError("; ".join(errors))
        return errors

    schema_spec = SCHEMA_SPECS[bundle_type]

    # Check required fields
    for field in schema_spec["required_fields"]:
        if field not in spec:
            errors.append(f"Missing required field: {field}")

    # Validate schema field
    if "schema" in spec:
        schema = spec["schema"]
        if not isinstance(schema, dict):
            errors.append("'schema' field must be a dictionary")
        else:
            if "name" not in schema:
                errors.append("schema.name is required")
            elif schema["name"] != schema_spec["name"]:
                errors.append(
                    f"Schema name mismatch: expected '{schema_spec['name']}', "
                    f"got '{schema['name']}'"
                )

            if "version" not in schema:
                errors.append("schema.version is required")

    # Type-specific validation
    if bundle_type == BundleType.FIGZ:
        errors.extend(_validate_figz_spec(spec))
    elif bundle_type == BundleType.PLTZ:
        errors.extend(_validate_pltz_spec(spec))
    elif bundle_type == BundleType.STATSZ:
        errors.extend(_validate_statsz_spec(spec))

    if strict and errors:
        raise BundleValidationError("; ".join(errors))

    return errors


def _validate_figz_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate .figz-specific fields."""
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


def _validate_pltz_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate .pltz-specific fields."""
    errors = []

    if "axes" in spec:
        axes = spec["axes"]
        if not isinstance(axes, (dict, list)):
            errors.append("'axes' must be a dictionary or list")

    return errors


def _validate_statsz_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate .statsz-specific fields."""
    errors = []

    if "comparisons" in spec:
        comparisons = spec["comparisons"]
        if not isinstance(comparisons, list):
            errors.append("'comparisons' must be a list")
        else:
            for i, comp in enumerate(comparisons):
                if not isinstance(comp, dict):
                    errors.append(f"comparisons[{i}] must be a dictionary")
                    continue
                # p_value is commonly expected
                if "p_value" in comp:
                    p = comp["p_value"]
                    if not isinstance(p, (int, float)):
                        errors.append(f"comparisons[{i}].p_value must be numeric")
                    elif not (0 <= p <= 1):
                        errors.append(f"comparisons[{i}].p_value must be between 0 and 1")

    return errors


def validate_bundle(path: Union[str, Path], strict: bool = False) -> Dict[str, Any]:
    """Validate a bundle and return validation results.

    Args:
        path: Path to bundle (directory or ZIP).
        strict: If True, raise BundleValidationError on failure.

    Returns:
        Dictionary with:
        - 'valid': bool
        - 'errors': list of error messages
        - 'bundle_type': detected bundle type
        - 'spec': parsed spec (if available)

    Raises:
        BundleValidationError: If strict=True and validation fails.
    """
    result = {
        'valid': True,
        'errors': [],
        'bundle_type': None,
        'spec': None,
    }

    p = Path(path)

    # Check bundle type
    bundle_type = _get_bundle_type(p)
    if bundle_type is None:
        result['valid'] = False
        result['errors'].append(f"Not a valid bundle path: {path}")
        if strict:
            raise BundleValidationError(result['errors'][0])
        return result

    result['bundle_type'] = bundle_type

    # Try to load and validate
    try:
        bundle = load_bundle(path)
        spec = bundle.get('spec')
        result['spec'] = spec

        if spec is not None:
            errors = validate_spec(spec, bundle_type, strict=False)
            if errors:
                result['valid'] = False
                result['errors'].extend(errors)
        else:
            result['valid'] = False
            result['errors'].append("Bundle has no spec")

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to load bundle: {e}")

    if strict and not result['valid']:
        raise BundleValidationError("; ".join(result['errors']))

    return result


def _get_bundle_type(path: Union[str, Path]) -> Optional[str]:
    """Get bundle type from path.

    Args:
        path: Path to bundle (directory or ZIP).

    Returns:
        Bundle type string or None if not a bundle.
    """
    p = Path(path)

    # Directory bundle: ends with .figz.d, .pltz.d, .statsz.d
    if p.is_dir() and p.suffix == ".d":
        stem = p.stem  # e.g., "Figure1.figz"
        for ext in BUNDLE_EXTENSIONS:
            if stem.endswith(ext):
                return ext[1:]  # Remove leading dot
        return None

    # ZIP bundle: ends with .figz, .pltz, .statsz
    if p.suffix in BUNDLE_EXTENSIONS:
        return p.suffix[1:]  # Remove leading dot

    return None


def is_bundle(path: Union[str, Path]) -> bool:
    """Check if path is a SciTeX bundle (directory or ZIP).

    Args:
        path: Path to check.

    Returns:
        True if path is a bundle.
    """
    return _get_bundle_type(path) is not None


def _dir_to_zip_path(dir_path: Path) -> Path:
    """Convert directory path to ZIP path.

    Example: Figure1.figz.d/ -> Figure1.figz
    """
    if dir_path.suffix == ".d":
        return dir_path.with_suffix("")
    return dir_path


def _zip_to_dir_path(zip_path: Path) -> Path:
    """Convert ZIP path to directory path.

    Example: Figure1.figz -> Figure1.figz.d/
    """
    return Path(str(zip_path) + ".d")


def pack_bundle(dir_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """Pack a bundle directory into a ZIP archive.

    Args:
        dir_path: Path to bundle directory (e.g., Figure1.figz.d/).
        output_path: Output ZIP path. Auto-generated if None.

    Returns:
        Path to created ZIP archive.
    """
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    if output_path is None:
        output_path = _dir_to_zip_path(dir_path)
    else:
        output_path = Path(output_path)

    # Create ZIP archive
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(dir_path)
                zf.write(file_path, arcname)

    return output_path


def unpack_bundle(zip_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """Unpack a bundle ZIP archive into a directory.

    Args:
        zip_path: Path to bundle ZIP (e.g., Figure1.figz).
        output_path: Output directory path. Auto-generated if None.

    Returns:
        Path to created directory.
    """
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise ValueError(f"Not a file: {zip_path}")

    if output_path is None:
        output_path = _zip_to_dir_path(zip_path)
    else:
        output_path = Path(output_path)

    # Extract ZIP archive
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_path)

    return output_path


def load_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Load bundle from directory or ZIP transparently.

    Args:
        path: Path to bundle (directory or ZIP).

    Returns:
        Bundle data as dictionary with:
        - 'type': Bundle type ('figz', 'pltz', 'statsz')
        - 'spec': Parsed JSON specification
        - 'data': CSV data (for .pltz)
        - 'path': Original path
        - 'is_zip': Whether loaded from ZIP
    """
    p = Path(path)
    bundle_type = _get_bundle_type(p)

    if bundle_type is None:
        raise ValueError(f"Not a valid bundle: {path}")

    result = {
        'type': bundle_type,
        'path': p,
        'is_zip': False,
    }

    # Handle ZIP vs directory
    if p.is_file() and p.suffix in BUNDLE_EXTENSIONS:
        # ZIP archive - extract to temp and load
        result['is_zip'] = True
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(p, 'r') as zf:
            zf.extractall(temp_dir)
        bundle_dir = temp_dir
    else:
        bundle_dir = p

    # Load specification based on bundle type
    if bundle_type == BundleType.FIGZ:
        spec_file = bundle_dir / "figure.json"
    elif bundle_type == BundleType.PLTZ:
        spec_file = bundle_dir / "plot.json"
    elif bundle_type == BundleType.STATSZ:
        spec_file = bundle_dir / "stats.json"
    else:
        raise ValueError(f"Unknown bundle type: {bundle_type}")

    if spec_file.exists():
        with open(spec_file, 'r') as f:
            result['spec'] = json.load(f)
    else:
        result['spec'] = None

    # Load CSV data for .pltz bundles
    if bundle_type == BundleType.PLTZ:
        csv_file = bundle_dir / "plot.csv"
        if csv_file.exists():
            try:
                import pandas as pd
                result['data'] = pd.read_csv(csv_file)
            except ImportError:
                # Fallback to basic CSV reading
                with open(csv_file, 'r') as f:
                    result['data'] = f.read()

    # Load nested .pltz bundles for .figz
    if bundle_type == BundleType.FIGZ:
        result['plots'] = {}
        for pltz_dir in bundle_dir.glob("*.pltz.d"):
            plot_name = pltz_dir.stem.replace(".pltz", "")
            result['plots'][plot_name] = load_bundle(pltz_dir)
        for pltz_zip in bundle_dir.glob("*.pltz"):
            if pltz_zip.is_file():
                plot_name = pltz_zip.stem
                result['plots'][plot_name] = load_bundle(pltz_zip)

    return result


def _generate_bundle_overview(dir_path: Path, spec: Dict, data: Dict):
    """Generate overview.png showing bundle contents visually.

    Creates a comprehensive overview image with:
    - CSV statistics (columns, rows, dtypes)
    - JSON structure as tree
    - Figures grid (PNG, hitmap, diff overlay)

    Parameters
    ----------
    dir_path : Path
        Bundle directory path.
    spec : dict
        Bundle specification.
    data : dict
        Bundle data dictionary.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image
    import numpy as np
    import io

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3,
                           left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title
    bundle_name = dir_path.name
    fig.suptitle(f'Bundle Overview: {bundle_name}', fontsize=16, fontweight='bold')

    # === Panel 1: CSV Statistics ===
    ax_csv = fig.add_subplot(gs[0, 0])
    ax_csv.set_title('CSV Data', fontweight='bold', fontsize=11)
    ax_csv.axis('off')

    csv_text = []
    if 'data' in data and hasattr(data['data'], 'columns'):
        df = data['data']
        csv_text.append(f"Rows: {len(df)}")
        csv_text.append(f"Columns: {len(df.columns)}")
        csv_text.append("")
        csv_text.append("Columns:")
        for col in df.columns[:10]:  # Show first 10 columns
            dtype = str(df[col].dtype)
            csv_text.append(f"  • {col} ({dtype})")
        if len(df.columns) > 10:
            csv_text.append(f"  ... +{len(df.columns) - 10} more")
    else:
        csv_text.append("No CSV data")

    ax_csv.text(0.05, 0.95, '\n'.join(csv_text), transform=ax_csv.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    # === Panel 2: JSON Tree ===
    ax_json = fig.add_subplot(gs[0, 1])
    ax_json.set_title('JSON Structure', fontweight='bold', fontsize=11)
    ax_json.axis('off')

    def json_to_tree(obj, prefix="", max_depth=4, depth=0, max_keys=6, max_lines=30):
        """Convert JSON to tree representation with depth control."""
        lines = []
        if depth >= max_depth:
            return []

        if isinstance(obj, dict):
            items = list(obj.items())[:max_keys]
            for i, (k, v) in enumerate(items):
                is_last = i == len(items) - 1 and len(obj) <= max_keys
                branch = "└─ " if is_last else "├─ "
                next_prefix = prefix + ("   " if is_last else "│  ")

                if isinstance(v, dict):
                    if depth < max_depth - 1 and v:
                        lines.append(prefix + branch + f"{k}:")
                        lines.extend(json_to_tree(v, next_prefix, max_depth, depth + 1, max_keys, max_lines))
                    else:
                        lines.append(prefix + branch + f"{k}: {{{len(v)} keys}}")
                elif isinstance(v, list):
                    lines.append(prefix + branch + f"{k}: [{len(v)} items]")
                else:
                    val_str = str(v)
                    if len(val_str) > 25:
                        val_str = val_str[:22] + "..."
                    lines.append(prefix + branch + f"{k}: {val_str}")

            if len(obj) > max_keys:
                lines.append(prefix + f"   ... +{len(obj) - max_keys} more")

        return lines[:max_lines]

    json_lines = json_to_tree(spec, max_depth=4, max_keys=8, max_lines=25)
    ax_json.text(0.02, 0.98, '\n'.join(json_lines), transform=ax_json.transAxes,
                 fontsize=7, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    # === Panel 3: PNG Image ===
    ax_png = fig.add_subplot(gs[0, 2])
    ax_png.set_title('plot.png', fontweight='bold', fontsize=11)
    png_path = dir_path / "plot.png"
    if png_path.exists():
        png_img = Image.open(png_path)
        ax_png.imshow(png_img)
        ax_png.set_xlabel(f'{png_img.size[0]}×{png_img.size[1]}', fontsize=9)
    ax_png.axis('off')

    # === Panel 4: Hitmap PNG ===
    ax_hitmap = fig.add_subplot(gs[0, 3])
    ax_hitmap.set_title('plot_hitmap.png', fontweight='bold', fontsize=11)
    hitmap_path = dir_path / "plot_hitmap.png"
    if hitmap_path.exists():
        hitmap_img = Image.open(hitmap_path)
        ax_hitmap.imshow(hitmap_img)
        ax_hitmap.set_xlabel(f'{hitmap_img.size[0]}×{hitmap_img.size[1]}', fontsize=9)
    ax_hitmap.axis('off')

    # === Panel 5: SVG (rendered) ===
    ax_svg = fig.add_subplot(gs[1, 0])
    ax_svg.set_title('plot.svg', fontweight='bold', fontsize=11)
    svg_path = dir_path / "plot.svg"
    if svg_path.exists():
        # Just show file info since SVG can't be directly displayed
        svg_size = svg_path.stat().st_size
        ax_svg.text(0.5, 0.5, f'SVG File\n{svg_size/1024:.1f} KB',
                   transform=ax_svg.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='#e0e0ff'))
    ax_svg.axis('off')

    # === Panel 6: Hitmap SVG ===
    ax_hitmap_svg = fig.add_subplot(gs[1, 1])
    ax_hitmap_svg.set_title('plot_hitmap.svg', fontweight='bold', fontsize=11)
    hitmap_svg_path = dir_path / "plot_hitmap.svg"
    if hitmap_svg_path.exists():
        svg_size = hitmap_svg_path.stat().st_size
        ax_hitmap_svg.text(0.5, 0.5, f'SVG Hitmap\n{svg_size/1024:.1f} KB',
                          transform=ax_hitmap_svg.transAxes, ha='center', va='center',
                          fontsize=12, bbox=dict(boxstyle='round', facecolor='#ffe0e0'))
    ax_hitmap_svg.axis('off')

    # === Panel 7: PNG vs Hitmap Diff ===
    ax_diff = fig.add_subplot(gs[1, 2])
    ax_diff.set_title('PNG vs Hitmap (Overlay)', fontweight='bold', fontsize=11)
    if png_path.exists() and hitmap_path.exists():
        png_arr = np.array(Image.open(png_path).convert('RGB'))
        hitmap_arr = np.array(Image.open(hitmap_path).convert('RGB'))

        if png_arr.shape == hitmap_arr.shape:
            # Create overlay: PNG in blue channel, hitmap in red channel
            overlay = np.zeros_like(png_arr)
            overlay[:, :, 0] = hitmap_arr[:, :, 0]  # Red = hitmap
            overlay[:, :, 2] = np.mean(png_arr, axis=2).astype(np.uint8)  # Blue = PNG
            overlay[:, :, 1] = 128  # Green = neutral
            ax_diff.imshow(overlay)
            ax_diff.set_xlabel('Red=Hitmap, Blue=PNG', fontsize=9)
        else:
            ax_diff.text(0.5, 0.5, 'Size mismatch!', transform=ax_diff.transAxes,
                        ha='center', va='center', fontsize=14, color='red')
    ax_diff.axis('off')

    # === Panel 8: Alignment Validation ===
    ax_valid = fig.add_subplot(gs[1, 3])
    ax_valid.set_title('Alignment Check', fontweight='bold', fontsize=11)
    ax_valid.axis('off')

    validation_text = []
    if png_path.exists() and hitmap_path.exists():
        png_img = Image.open(png_path)
        hitmap_img = Image.open(hitmap_path)

        # Size check
        size_match = png_img.size == hitmap_img.size
        validation_text.append(f"✓ Size match: {png_img.size}" if size_match
                               else f"✗ Size mismatch: PNG={png_img.size}, Hitmap={hitmap_img.size}")

        # Content bounds check
        png_arr = np.array(png_img.convert('RGB'))
        hitmap_arr = np.array(hitmap_img.convert('RGB'))

        def find_bounds(arr):
            white = np.array([255, 255, 255])
            mask = np.any(np.abs(arr.astype(int) - white) > 20, axis=2)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                return (int(x_min), int(y_min), int(x_max), int(y_max))
            return (0, 0, arr.shape[1], arr.shape[0])

        png_bounds = find_bounds(png_arr)
        hitmap_bounds = find_bounds(hitmap_arr)

        bounds_match = png_bounds == hitmap_bounds
        validation_text.append(f"✓ Content aligned" if bounds_match
                               else f"✗ Content offset: {hitmap_bounds[0]-png_bounds[0]}, {hitmap_bounds[1]-png_bounds[1]}")

        validation_text.append("")
        validation_text.append(f"PNG bounds: {png_bounds}")
        validation_text.append(f"Hitmap bounds: {hitmap_bounds}")

    else:
        validation_text.append("Files not found")

    color = 'green' if all('✓' in t for t in validation_text if t.startswith('✓') or t.startswith('✗')) else 'red'
    ax_valid.text(0.05, 0.95, '\n'.join(validation_text), transform=ax_valid.transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='#f0fff0' if color == 'green' else '#fff0f0', alpha=0.8))

    # Save overview
    overview_path = dir_path / "overview.png"
    fig.savefig(overview_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_bundle(
    data: Dict[str, Any],
    path: Union[str, Path],
    bundle_type: Optional[str] = None,
    as_zip: bool = False,
) -> Path:
    """Save data as a bundle.

    Args:
        data: Bundle data dictionary.
        path: Output path (with or without .d suffix).
        bundle_type: Bundle type ('figz', 'pltz', 'statsz'). Auto-detected if None.
        as_zip: If True, save as ZIP archive.

    Returns:
        Path to saved bundle.
    """
    p = Path(path)

    # Determine bundle type
    if bundle_type is None:
        bundle_type = _get_bundle_type(p)
        if bundle_type is None:
            raise ValueError(f"Cannot determine bundle type from path: {path}")

    # Determine if saving as directory or ZIP
    if as_zip or (p.suffix in BUNDLE_EXTENSIONS and not str(p).endswith(".d")):
        # Save as ZIP
        save_as_zip = True
        if p.suffix == ".d":
            zip_path = _dir_to_zip_path(p)
        else:
            zip_path = p
        dir_path = _zip_to_dir_path(zip_path)
    else:
        # Save as directory
        save_as_zip = False
        if not str(p).endswith(".d"):
            dir_path = Path(str(p) + ".d")
        else:
            dir_path = p

    # Create directory
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save specification
    spec = data.get('spec', {})
    if bundle_type == BundleType.FIGZ:
        spec_file = dir_path / "figure.json"
    elif bundle_type == BundleType.PLTZ:
        spec_file = dir_path / "plot.json"
    elif bundle_type == BundleType.STATSZ:
        spec_file = dir_path / "stats.json"
    else:
        raise ValueError(f"Unknown bundle type: {bundle_type}")

    with open(spec_file, 'w') as f:
        json.dump(spec, f, indent=2)

    # Save CSV data for .pltz bundles
    if bundle_type == BundleType.PLTZ and 'data' in data:
        csv_file = dir_path / "plot.csv"
        df = data['data']
        if hasattr(df, 'to_csv'):
            df.to_csv(csv_file, index=False)
        else:
            with open(csv_file, 'w') as f:
                f.write(str(df))

    # Save exports (PNG, SVG, PDF) if provided
    # Embed metadata into image files for standalone viewing
    for fmt in ['png', 'svg', 'pdf']:
        if fmt in data:
            if bundle_type == BundleType.FIGZ:
                out_file = dir_path / f"figure.{fmt}"
            elif bundle_type == BundleType.PLTZ:
                out_file = dir_path / f"plot.{fmt}"
            else:
                continue

            # Handle different data types
            export_data = data[fmt]
            if isinstance(export_data, bytes):
                with open(out_file, 'wb') as f:
                    f.write(export_data)
            elif isinstance(export_data, (str, Path)) and Path(export_data).exists():
                shutil.copy(export_data, out_file)

            # Embed metadata into PNG and PDF files
            if out_file.exists() and spec:
                try:
                    _embed_metadata_in_export(out_file, spec, fmt)
                except Exception as e:
                    # Non-fatal: metadata embedding is optional
                    import logging
                    logging.getLogger("scitex").debug(f"Could not embed metadata in {out_file}: {e}")

    # Save hit map for .pltz bundles (for interactive element selection)
    if bundle_type == BundleType.PLTZ:
        # Save hitmap PNG
        if 'hitmap_png' in data:
            hitmap_file = dir_path / "plot_hitmap.png"
            hitmap_data = data['hitmap_png']
            if isinstance(hitmap_data, bytes):
                with open(hitmap_file, 'wb') as f:
                    f.write(hitmap_data)
            elif isinstance(hitmap_data, (str, Path)) and Path(hitmap_data).exists():
                shutil.copy(hitmap_data, hitmap_file)

        # Save hitmap SVG
        if 'hitmap_svg' in data:
            hitmap_svg_file = dir_path / "plot_hitmap.svg"
            hitmap_svg_data = data['hitmap_svg']
            if isinstance(hitmap_svg_data, bytes):
                with open(hitmap_svg_file, 'wb') as f:
                    f.write(hitmap_svg_data)
            elif isinstance(hitmap_svg_data, (str, Path)) and Path(hitmap_svg_data).exists():
                shutil.copy(hitmap_svg_data, hitmap_svg_file)

    # Generate overview.png for .pltz bundles
    if bundle_type == BundleType.PLTZ:
        try:
            _generate_bundle_overview(dir_path, spec, data)
        except Exception as e:
            import logging
            logging.getLogger("scitex").debug(f"Could not generate overview: {e}")

    # Save nested .pltz bundles for .figz
    if bundle_type == BundleType.FIGZ and 'plots' in data:
        for plot_name, plot_data in data['plots'].items():
            plot_path = dir_path / f"{plot_name}.pltz.d"
            save_bundle(plot_data, plot_path, bundle_type=BundleType.PLTZ)

    # Pack to ZIP if requested
    if save_as_zip:
        pack_bundle(dir_path, zip_path)
        shutil.rmtree(dir_path)  # Remove temp directory
        return zip_path

    return dir_path


# EOF
