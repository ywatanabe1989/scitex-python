#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save.py

"""
Save utilities for various data types to different file formats.

Supported formats include CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML,
JSON, HDF5, PTH, MAT, CBM, and SciTeX bundles (.zip or directory).
"""

import inspect
import os as _os
from pathlib import Path
from typing import Any, Union

from scitex import logging
from scitex.path._clean import clean
from scitex.path._getsize import getsize
from scitex.sh import sh
from scitex.str._clean_path import clean_path
from scitex.str._color_text import color_text
from scitex.str._readable_bytes import readable_bytes

# Import save functions from the modular structure
from ._save_modules import (
    get_figure_with_data,
    handle_image_with_csv,
    save_bibtex,
    save_catboost,
    save_csv,
    save_excel,
    save_hdf5,
    save_html,
    save_joblib,
    save_json,
    save_matlab,
    save_mp4,
    save_npy,
    save_npz,
    save_pickle,
    save_pickle_compressed,
    save_tex,
    save_text,
    save_torch,
    save_yaml,
    save_zarr,
    symlink,
    symlink_to,
)

logger = logging.getLogger()

_get_figure_with_data = get_figure_with_data
_symlink = symlink
_symlink_to = symlink_to
_handle_image_with_csv = handle_image_with_csv


def save(
    obj: Any,
    specified_path: Union[str, Path],
    makedirs: bool = True,
    verbose: bool = True,
    symlink_from_cwd: bool = False,
    symlink_to: Union[str, Path] = None,
    dry_run: bool = False,
    no_csv: bool = False,
    use_caller_path: bool = False,
    auto_crop: bool = True,
    crop_margin_mm: float = 1.0,
    metadata_extra: dict = None,
    json_schema: str = "editable",
    **kwargs,
) -> None:
    """
    Save an object to a file with the specified format.

    Parameters
    ----------
    obj : Any
        The object to be saved.
    specified_path : Union[str, Path]
        The file path where the object should be saved.
    makedirs : bool, optional
        If True, create the directory path if it does not exist. Default is True.
    verbose : bool, optional
        If True, print a message upon successful saving. Default is True.
    symlink_from_cwd : bool, optional
        If True, create a symlink from the current working directory. Default is False.
    symlink_to : Union[str, Path], optional
        If specified, create a symlink at this path. Default is None.
    dry_run : bool, optional
        If True, simulate the saving process. Default is False.
    auto_crop : bool, optional
        If True, automatically crop saved images. Default is True.
    crop_margin_mm : float, optional
        Margin in millimeters for auto_crop. Default is 1.0mm.
    use_caller_path : bool, optional
        If True, determine script path by skipping internal library frames.
    metadata_extra : dict, optional
        Additional metadata to merge with auto-collected metadata.
    json_schema : str, optional
        Schema type for JSON metadata output. Default is "editable".
    **kwargs
        Additional keyword arguments for the underlying save function.
    """
    try:
        if isinstance(specified_path, Path):
            specified_path = str(specified_path)

        # Handle f-string expressions
        specified_path = _parse_fstring_path(specified_path)

        # Determine save path
        spath = _determine_save_path(specified_path, use_caller_path)
        spath_final = clean(spath)

        # Prepare symlink path from cwd
        spath_cwd = _os.getcwd() + "/" + specified_path
        spath_cwd = clean(spath_cwd)

        # Remove existing files (skip for CSV/HDF5 with key)
        _cleanup_existing_files(spath_final, spath_cwd, kwargs)

        if dry_run:
            _handle_dry_run(spath, verbose)
            return

        if makedirs:
            _os.makedirs(_os.path.dirname(spath_final), exist_ok=True)

        # Main save
        _save(
            obj,
            spath_final,
            verbose=verbose,
            symlink_from_cwd=symlink_from_cwd,
            symlink_to=symlink_to,
            dry_run=dry_run,
            no_csv=no_csv,
            auto_crop=auto_crop,
            crop_margin_mm=crop_margin_mm,
            metadata_extra=metadata_extra,
            json_schema=json_schema,
            **kwargs,
        )

        # Symbolic links
        _symlink(spath, spath_cwd, symlink_from_cwd, verbose)
        _symlink_to(spath_final, symlink_to, verbose)
        return Path(spath)

    except AssertionError:
        raise
    except Exception as e:
        logger.error(f"Error occurred while saving: {str(e)}")
        return False


def _parse_fstring_path(specified_path):
    """Parse f-string expressions in path."""
    if not (specified_path.startswith('f"') or specified_path.startswith("f'")):
        return specified_path

    import re

    path_content = specified_path[2:-1]
    frame = inspect.currentframe().f_back.f_back
    try:
        variables = re.findall(r"\{([^}]+)\}", path_content)
        format_dict = {}
        for var in variables:
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var):
                if var in frame.f_locals:
                    format_dict[var] = frame.f_locals[var]
                elif var in frame.f_globals:
                    format_dict[var] = frame.f_globals[var]
            else:
                raise ValueError(f"Invalid variable name in f-string: {var}")
        return path_content.format(**format_dict)
    finally:
        del frame


def _determine_save_path(specified_path, use_caller_path):
    """Determine the full save path based on environment."""
    if specified_path.startswith("/"):
        return specified_path

    from scitex.gen._detect_environment import detect_environment
    from scitex.gen._get_notebook_path import get_notebook_info_simple

    env_type = detect_environment()

    if env_type == "jupyter":
        notebook_name, notebook_dir = get_notebook_info_simple()
        if notebook_name:
            notebook_base = _os.path.splitext(notebook_name)[0]
            sdir = _os.path.join(notebook_dir or _os.getcwd(), f"{notebook_base}_out")
        else:
            sdir = _os.path.join(_os.getcwd(), "notebook_out")
        return _os.path.join(sdir, specified_path)

    elif env_type == "script":
        if use_caller_path:
            script_path = _find_caller_script_path()
        else:
            script_path = inspect.stack()[2].filename
        sdir = clean_path(_os.path.splitext(script_path)[0] + "_out")
        return _os.path.join(sdir, specified_path)

    else:
        script_path = inspect.stack()[2].filename
        if (
            ("ipython" in script_path)
            or ("<stdin>" in script_path)
            or env_type in ["ipython", "interactive"]
        ):
            sdir = f"/tmp/{_os.getenv('USER')}"
        else:
            sdir = _os.path.join(_os.getcwd(), "output")
        return _os.path.join(sdir, specified_path)


def _find_caller_script_path():
    """Find the first non-scitex frame in the call stack."""
    scitex_src_path = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..")
    )
    for frame_info in inspect.stack()[3:]:
        frame_path = _os.path.abspath(frame_info.filename)
        if not frame_path.startswith(scitex_src_path):
            return frame_path
    return inspect.stack()[2].filename


def _cleanup_existing_files(spath_final, spath_cwd, kwargs):
    """Remove existing files to prevent circular links."""
    should_skip = spath_final.endswith(".csv") or (
        (spath_final.endswith(".hdf5") or spath_final.endswith(".h5"))
        and "key" in kwargs
    )
    if not should_skip:
        for path in [spath_final, spath_cwd]:
            sh(["rm", "-f", f"{path}"], verbose=False)


def _handle_dry_run(spath, verbose):
    """Handle dry run mode."""
    if verbose:
        try:
            rel_path = _os.path.relpath(spath, _os.getcwd())
        except ValueError:
            rel_path = spath
        logger.success(color_text(f"(dry run) Saved to: ./{rel_path}", c="yellow"))


def _save(
    obj,
    spath,
    verbose=True,
    symlink_from_cwd=False,
    dry_run=False,
    no_csv=False,
    symlink_to=None,
    auto_crop=False,
    crop_margin_mm=1.0,
    metadata_extra=None,
    json_schema="editable",
    **kwargs,
):
    """Core dispatcher for saving objects to various formats."""
    ext = _os.path.splitext(spath)[1].lower()

    # Check if this is a matplotlib figure being saved to SciTeX bundle format
    # SciTeX bundles use .zip (archive) or no extension (directory)
    if _is_matplotlib_figure(obj):
        # Save as SciTeX bundle if:
        # 1. Path ends with .zip (create ZIP bundle)
        # 2. Path has no extension and doesn't match other formats (create directory bundle)
        if ext == ".zip" or (ext == "" and not spath.endswith("/")):
            # Check if explicitly requesting SciTeX bundle or just .zip
            # Pop as_zip from kwargs to avoid duplicate parameter error
            as_zip = kwargs.pop("as_zip", ext == ".zip")
            _save_scitex_bundle(
                obj, spath, as_zip, verbose, symlink_from_cwd, symlink_to, **kwargs
            )
            return

    # Dispatch to format handlers
    if ext in _FILE_HANDLERS:
        _dispatch_handler(
            ext,
            obj,
            spath,
            verbose,
            no_csv,
            symlink_from_cwd,
            symlink_to,
            dry_run,
            auto_crop,
            crop_margin_mm,
            metadata_extra,
            json_schema,
            kwargs,
        )
    elif spath.endswith(".csv"):
        save_csv(obj, spath, **kwargs)
    elif spath.endswith(".pkl.gz"):
        save_pickle_compressed(obj, spath, **kwargs)
    else:
        logger.warning(f"Unsupported file format. {spath} was not saved.")
        return

    if verbose and _os.path.exists(spath):
        file_size = readable_bytes(getsize(spath))
        try:
            rel_path = _os.path.relpath(spath, _os.getcwd())
        except ValueError:
            rel_path = spath
        logger.success(f"Saved to: ./{rel_path} ({file_size})")


def _is_matplotlib_figure(obj):
    """Check if object is a matplotlib figure or a wrapped figure.

    Handles both raw matplotlib.figure.Figure and SciTeX FigWrapper objects.
    """
    try:
        import matplotlib.figure

        # Direct matplotlib figure
        if isinstance(obj, matplotlib.figure.Figure):
            return True

        # Wrapped figure (e.g., FigWrapper from scitex.plt)
        if hasattr(obj, "figure") and isinstance(obj.figure, matplotlib.figure.Figure):
            return True

        return False
    except ImportError:
        return False


def _save_scitex_bundle(
    obj, spath, as_zip, verbose, symlink_from_cwd, symlink_to_path, **kwargs
):
    """Save matplotlib figure as SciTeX bundle (.zip or directory).

    Delegates to scitex.io.bundle.from_matplotlib as the single source of truth
    for bundle structure (canonical/artifacts/payload/children).

    When figrecipe is available and enabled on the figure, also saves
    recipe.yaml for reproducibility.
    """
    # Get the actual matplotlib figure
    import matplotlib.figure

    from scitex.io.bundle import from_matplotlib

    from ._save_modules._figure_utils import get_figure_with_data

    if isinstance(obj, matplotlib.figure.Figure):
        fig = obj
        fig_wrapper = None
    elif hasattr(obj, "figure") and isinstance(obj.figure, matplotlib.figure.Figure):
        fig = obj.figure
        fig_wrapper = obj  # Keep wrapper for figrecipe access
    else:
        raise TypeError(f"Expected matplotlib figure, got {type(obj)}")

    # Extract optional parameters
    # Support both "csv_df" and "data" parameter names for user convenience
    csv_df = kwargs.get("csv_df") or kwargs.get("data")
    dpi = kwargs.get("dpi", 300)
    name = kwargs.get("name") or Path(spath).stem

    # Extract CSV data from scitex.plt tracking if available
    scitex_source = get_figure_with_data(obj)
    if csv_df is None and scitex_source is not None:
        if hasattr(scitex_source, "export_as_csv"):
            try:
                csv_df = scitex_source.export_as_csv()
            except Exception:
                pass

    # Delegate to Bundle (single source of truth)
    # Encoding is built from CSV columns directly for consistency
    from_matplotlib(fig, spath, name=name, csv_df=csv_df, dpi=dpi)

    # Save figrecipe recipe.yaml if available
    try:
        from scitex.bridge._figrecipe import _save_recipe_to_path

        bundle_path = Path(spath)
        if bundle_path.suffix != ".zip":  # Skip zip for now
            _save_recipe_to_path(fig_wrapper or obj, bundle_path / "recipe.yaml")
    except (ImportError, Exception):
        pass  # figrecipe is optional

    bundle_path = spath
    if verbose and _os.path.exists(bundle_path):
        file_size = readable_bytes(getsize(bundle_path))
        try:
            rel_path = _os.path.relpath(bundle_path, _os.getcwd())
        except ValueError:
            rel_path = bundle_path
        logger.success(f"Saved to: ./{rel_path} ({file_size})")

    if symlink_from_cwd and _os.path.exists(bundle_path):
        bundle_basename = _os.path.basename(bundle_path)
        bundle_cwd = _os.path.join(_os.getcwd(), bundle_basename)
        _symlink(bundle_path, bundle_cwd, symlink_from_cwd, verbose)

    if symlink_to_path and _os.path.exists(bundle_path):
        _symlink_to(bundle_path, symlink_to_path, verbose)


def _dispatch_handler(
    ext,
    obj,
    spath,
    verbose,
    no_csv,
    symlink_from_cwd,
    symlink_to_path,
    dry_run,
    auto_crop,
    crop_margin_mm,
    metadata_extra,
    json_schema,
    kwargs,
):
    """Dispatch to the appropriate file handler."""
    image_exts = [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".svg", ".pdf"]
    if ext in image_exts:
        _handle_image_with_csv(
            obj,
            spath,
            verbose=verbose,
            no_csv=no_csv,
            symlink_from_cwd=symlink_from_cwd,
            symlink_to_path=symlink_to_path,
            dry_run=dry_run,
            auto_crop=auto_crop,
            crop_margin_mm=crop_margin_mm,
            metadata_extra=metadata_extra,
            json_schema=json_schema,
            **kwargs,
        )
    elif ext in [".hdf5", ".h5", ".zarr"]:
        _FILE_HANDLERS[ext](obj, spath, **kwargs)
    else:
        _FILE_HANDLERS[ext](obj, spath, **kwargs)


# Dispatch dictionary for O(1) file format lookup
_FILE_HANDLERS = {
    ".xlsx": save_excel,
    ".xls": save_excel,
    ".npy": save_npy,
    ".npz": save_npz,
    ".pkl": save_pickle,
    ".pickle": save_pickle,
    ".pkl.gz": save_pickle_compressed,
    ".joblib": save_joblib,
    ".pth": save_torch,
    ".pt": save_torch,
    ".mat": save_matlab,
    ".cbm": save_catboost,
    ".json": save_json,
    ".yaml": save_yaml,
    ".yml": save_yaml,
    ".txt": save_text,
    ".md": save_text,
    ".py": save_text,
    ".css": save_text,
    ".js": save_text,
    ".tex": save_tex,
    ".bib": save_bibtex,
    ".html": save_html,
    ".hdf5": save_hdf5,
    ".h5": save_hdf5,
    ".zarr": save_zarr,
    ".mp4": save_mp4,
    ".png": handle_image_with_csv,
    ".jpg": handle_image_with_csv,
    ".jpeg": handle_image_with_csv,
    ".gif": handle_image_with_csv,
    ".tiff": handle_image_with_csv,
    ".tif": handle_image_with_csv,
    ".svg": handle_image_with_csv,
    ".pdf": handle_image_with_csv,
}

# EOF
