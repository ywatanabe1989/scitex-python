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
    if bundle_type == BundleType.PLTZ and 'hitmap_png' in data:
        hitmap_file = dir_path / "plot_hitmap.png"
        hitmap_data = data['hitmap_png']
        if isinstance(hitmap_data, bytes):
            with open(hitmap_file, 'wb') as f:
                f.write(hitmap_data)
        elif isinstance(hitmap_data, (str, Path)) and Path(hitmap_data).exists():
            shutil.copy(hitmap_data, hitmap_file)

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
