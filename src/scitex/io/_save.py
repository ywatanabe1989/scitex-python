#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 08:56:29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save.py


import os

__FILE__ = __file__

import warnings


"""
1. Functionality:
   - Provides utilities for saving various data types to different file formats.
2. Input:
   - Objects to be saved (e.g., NumPy arrays, PyTorch tensors, Pandas DataFrames, etc.)
   - File path or name where the object should be saved
3. Output:
   - Saved files in various formats (e.g., CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, CBM)
4. Prerequisites:
   - Python 3.x
   - Required libraries: numpy, pandas, torch, matplotlib, plotly, h5py, joblib, PIL, ruamel.yaml
"""

"""Imports"""
import inspect
import os as _os
from pathlib import Path
from typing import Any
from typing import Union

from scitex import logging

from scitex.sh import sh
from scitex.path._clean import clean
from scitex.path._getsize import getsize
from scitex.str._clean_path import clean_path
from scitex.str._color_text import color_text
from scitex.str._readable_bytes import readable_bytes

# Import save functions from the new modular structure
from ._save_modules import save_catboost
from ._save_modules import save_csv
from ._save_modules import save_excel
from ._save_modules import save_hdf5
from ._save_modules import save_html
from ._save_modules import save_image
from ._save_modules import save_joblib
from ._save_modules import save_json
from ._save_modules import save_matlab
from ._save_modules import save_mp4
from ._save_modules import save_npy
from ._save_modules import save_npz
from ._save_modules import save_pickle
from ._save_modules import save_pickle_compressed
from ._save_modules import save_tex
from ._save_modules import save_text
from ._save_modules import save_torch
from ._save_modules import save_yaml
from ._save_modules import save_zarr
from ._save_modules._bibtex import save_bibtex
from ._save_modules._canvas import save_canvas

logger = logging.getLogger()


def _get_figure_with_data(obj):
    """
    Extract figure or axes object that may contain plotting data for CSV export.

    Parameters
    ----------
    obj : various matplotlib objects
        Could be Figure, Axes, FigWrapper, AxisWrapper, or other matplotlib objects

    Returns
    -------
    object or None
        Figure or axes object that has export_as_csv methods, or None if not found
    """
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.pyplot as plt

    # Check if object already has export methods (SciTeX wrapped objects)
    if hasattr(obj, "export_as_csv"):
        return obj

    # Handle matplotlib Figure objects
    if isinstance(obj, matplotlib.figure.Figure):
        # Get the current axes that might be wrapped with SciTeX functionality
        current_ax = plt.gca()
        if hasattr(current_ax, "export_as_csv"):
            return current_ax

        # Check all axes in the figure
        for ax in obj.axes:
            if hasattr(ax, "export_as_csv"):
                return ax

        return None

    # Handle matplotlib Axes objects
    if isinstance(obj, matplotlib.axes.Axes):
        if hasattr(obj, "export_as_csv"):
            return obj
        return None

    # Handle FigWrapper or similar SciTeX objects
    if hasattr(obj, "figure") and hasattr(obj.figure, "axes"):
        # Check if the wrapper itself has export methods
        if hasattr(obj, "export_as_csv"):
            return obj

        # Check the underlying figure's axes
        for ax in obj.figure.axes:
            if hasattr(ax, "export_as_csv"):
                return ax

        return None

    # Handle AxisWrapper or similar SciTeX objects
    if hasattr(obj, "_axis_mpl") or hasattr(obj, "_ax"):
        if hasattr(obj, "export_as_csv"):
            return obj
        return None

    # Try to get the current figure and its axes as fallback
    try:
        current_fig = plt.gcf()
        current_ax = plt.gca()

        if hasattr(current_ax, "export_as_csv"):
            return current_ax
        elif hasattr(current_fig, "export_as_csv"):
            return current_fig

        # Check all axes in current figure
        for ax in current_fig.axes:
            if hasattr(ax, "export_as_csv"):
                return ax

    except:
        pass

    return None


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
        The object to be saved. Can be a NumPy array, PyTorch tensor, Pandas DataFrame, or any serializable object.
    specified_path : Union[str, Path]
        The file name or path where the object should be saved. Can be a string or pathlib.Path object. The file extension determines the format.
    makedirs : bool, optional
        If True, create the directory path if it does not exist. Default is True.
    verbose : bool, optional
        If True, print a message upon successful saving. Default is True.
    symlink_from_cwd : bool, optional
        If True, create a _symlink from the current working directory. Default is False.
    symlink_to : Union[str, Path], optional
        If specified, create a symlink at this path pointing to the saved file. Default is None.
    dry_run : bool, optional
        If True, simulate the saving process without actually writing files. Default is False.
    auto_crop : bool, optional
        If True, automatically crop the saved image to content area with margin (for PNG/JPEG/TIFF).
        Vector formats (PDF/SVG) are not cropped. Default is True.
    crop_margin_mm : float, optional
        Margin in millimeters to add around content when auto_crop=True.
        At 300 DPI: 1mm = ~12 pixels. Default is 1.0mm (Nature Reviews style).
    use_caller_path : bool, optional
        If True, intelligently determine the script path by skipping internal library frames.
        This is useful when stx.io.save is called from within scitex library code.
        Default is False.
    metadata_extra : dict, optional
        Additional metadata to merge with auto-collected metadata. Useful for specifying
        plot_type, style information, etc. Example:
            metadata_extra = {
                "plot_type": "line",
                "style": {
                    "name": "SCITEX_STYLE",
                    "overrides": {"ax_width_mm": 50}
                }
            }
        Default is None.
    json_schema : str, optional
        Schema type for JSON metadata output. Options:
        - "editable": Schema v0.3.0 with element geometry for interactive editing (default)
        - "recipe": Minimal schema with method calls + data refs
        - "verbose": Full schema with all artist details
        Default is "editable".
    **kwargs
        Additional keyword arguments to pass to the underlying save function of the specific format.

    Returns
    -------
    None

    Notes
    -----
    Supported formats include CSV, NPY, PKL, JOBLIB, PNG, HTML, TIFF, MP4, YAML, JSON, HDF5, PTH, MAT, CBM,
    and SciTeX bundles (.figz, .pltz, .statsz).
    The function dynamically selects the appropriate saving mechanism based on the file extension.

    Bundle Formats:
    - .figz: Publication figure bundle (panels dict). Default: ZIP archive.
    - .pltz: Plot bundle (matplotlib figure). Default: directory bundle.
    - .statsz: Statistics bundle (comparisons list). Default: directory bundle.
    - Use .d suffix (e.g., "Figure1.figz.d") to force directory format for .figz.

    Examples
    --------
    >>> import scitex
    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> import matplotlib.pyplot as plt

    >>> # Save NumPy array
    >>> arr = np.array([1, 2, 3])
    >>> scitex.io.save(arr, "data.npy")

    >>> # Save Pandas DataFrame
    >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    >>> scitex.io.save(df, "data.csv")

    >>> # Save PyTorch tensor
    >>> tensor = torch.tensor([1, 2, 3])
    >>> scitex.io.save(tensor, "model.pth")

    >>> # Save dictionary
    >>> data_dict = {"a": 1, "b": 2, "c": [3, 4, 5]}
    >>> scitex.io.save(data_dict, "data.pkl")

    >>> # Save matplotlib figure
    >>> plt.figure()
    >>> plt.plot(np.array([1, 2, 3]))
    >>> scitex.io.save(plt, "plot.png")

    >>> # Save as YAML
    >>> scitex.io.save(data_dict, "config.yaml")

    >>> # Save as JSON
    >>> scitex.io.save(data_dict, "data.json")
    """
    try:
        # Convert Path objects to strings for consistency
        if isinstance(specified_path, Path):
            specified_path = str(specified_path)

        ########################################
        # DO NOT MODIFY THIS SECTION
        ########################################
        #
        # Determine saving directory from the script.
        #
        # When called in /path/to/script.py,
        # data will be saved under `/path/to/script.py_out/`
        #
        # When called in a Jupyter notebook /path/to/notebook.ipynb,
        # data will be saved under `/path/to/notebook_out/`
        #
        # When called in ipython environment,
        # data will be saved under `/tmp/{_os.getenv("USER")/`
        #
        ########################################
        spath, sfname = None, None

        # f-expression handling - safely parse f-strings
        if specified_path.startswith('f"') or specified_path.startswith("f'"):
            # Remove the f prefix and quotes
            path_content = specified_path[2:-1]

            # Get the caller's frame to access their local variables
            frame = inspect.currentframe().f_back
            try:
                # Use string formatting with the caller's locals and globals
                # This is much safer than eval() as it only does string substitution
                import re

                # Find all {variable} patterns
                variables = re.findall(r"\{([^}]+)\}", path_content)
                format_dict = {}
                for var in variables:
                    # Only allow simple variable names, not arbitrary expressions
                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var):
                        if var in frame.f_locals:
                            format_dict[var] = frame.f_locals[var]
                        elif var in frame.f_globals:
                            format_dict[var] = frame.f_globals[var]
                    else:
                        raise ValueError(f"Invalid variable name in f-string: {var}")

                # Use str.format() which is safe
                specified_path = path_content.format(**format_dict)
            finally:
                del frame  # Avoid reference cycles

        # When full path
        if specified_path.startswith("/"):
            spath = specified_path

        # When relative path
        else:
            # Import here to avoid circular imports
            from scitex.gen._detect_environment import detect_environment
            from scitex.gen._get_notebook_path import get_notebook_info_simple

            # Detect the current environment
            env_type = detect_environment()

            if env_type == "jupyter":
                # Special handling for Jupyter notebooks
                notebook_name, notebook_dir = get_notebook_info_simple()

                if notebook_name:
                    # Remove .ipynb extension and add _out
                    notebook_base = _os.path.splitext(notebook_name)[0]
                    sdir = _os.path.join(
                        notebook_dir or _os.getcwd(), f"{notebook_base}_out"
                    )
                else:
                    # Fallback if we can't detect notebook name
                    sdir = _os.path.join(_os.getcwd(), "notebook_out")

                spath = _os.path.join(sdir, specified_path)

            elif env_type == "script":
                # Regular script handling
                if use_caller_path:
                    # Smart path detection: skip internal scitex library frames
                    script_path = None
                    scitex_src_path = _os.path.join(
                        _os.path.dirname(__file__), "..", ".."
                    )
                    scitex_src_path = _os.path.abspath(scitex_src_path)

                    # Walk through the call stack from caller to find the first non-scitex frame
                    for frame_info in inspect.stack()[1:]:
                        frame_path = _os.path.abspath(frame_info.filename)
                        # Skip frames from scitex library
                        if not frame_path.startswith(scitex_src_path):
                            script_path = frame_path
                            break

                    # Fallback to stack[1] if we couldn't find a non-scitex frame
                    if script_path is None:
                        script_path = inspect.stack()[1].filename
                else:
                    script_path = inspect.stack()[1].filename

                sdir = clean_path(_os.path.splitext(script_path)[0] + "_out")
                spath = _os.path.join(sdir, specified_path)

            else:
                # IPython console or interactive mode
                script_path = inspect.stack()[1].filename

                if (
                    ("ipython" in script_path)
                    or ("<stdin>" in script_path)
                    or env_type in ["ipython", "interactive"]
                ):
                    script_path = f"/tmp/{_os.getenv('USER')}"
                    sdir = script_path
                else:
                    # Unknown environment, use current directory
                    sdir = _os.path.join(_os.getcwd(), "output")

                spath = _os.path.join(sdir, specified_path)

        # Sanitization
        spath_final = clean(spath)
        ########################################

        # Potential path to _symlink
        spath_cwd = _os.getcwd() + "/" + specified_path
        spath_cwd = clean(spath_cwd)

        # Removes spath and spath_cwd to prevent potential circular links
        # Skip deletion for CSV files to allow caching to work
        # Also skip deletion for HDF5 files when a key is specified
        should_skip_deletion = spath_final.endswith(".csv") or (
            (spath_final.endswith(".hdf5") or spath_final.endswith(".h5"))
            and "key" in kwargs
        )

        if not should_skip_deletion:
            for path in [spath_final, spath_cwd]:
                sh(["rm", "-f", f"{path}"], verbose=False)

        if dry_run:
            # Get relative path from current working directory
            try:
                rel_path = _os.path.relpath(spath, _os.getcwd())
            except ValueError:
                rel_path = spath

            if verbose:
                logger.success(
                    color_text(f"(dry run) Saved to: ./{rel_path}", c="yellow")
                )
            return

        # Ensure directory exists
        if makedirs:
            _os.makedirs(_os.path.dirname(spath_final), exist_ok=True)

        # Main
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
        # return True

    except AssertionError:
        # Re-raise assertion errors - these are validation failures that should stop execution
        raise
    except Exception as e:
        logger.error(
            f"Error occurred while saving: {str(e)}\n"
            f"Debug: Initial script_path = {inspect.stack()[1].filename}\n"
            f"Debug: Final spath = {spath}\n"
            f"Debug: specified_path type = {type(specified_path)}\n"
            f"Debug: specified_path = {specified_path}"
        )
        return False


def _symlink(spath, spath_cwd, symlink_from_cwd, verbose):
    """Create a symbolic link from the current working directory."""
    if symlink_from_cwd and (spath != spath_cwd):
        _os.makedirs(_os.path.dirname(spath_cwd), exist_ok=True)
        sh(["rm", "-f", f"{spath_cwd}"], verbose=False)
        sh(["ln", "-sfr", f"{spath}", f"{spath_cwd}"], verbose=False)
        if verbose:
            # Get file extension to provide more informative message
            ext = _os.path.splitext(spath_cwd)[1].lower()
            logger.success(color_text(f"(Symlinked to: {spath_cwd})"))


def _symlink_to(spath_final, symlink_to, verbose):
    """Create a symbolic link at the specified path pointing to the saved file."""
    if symlink_to:
        # Convert Path objects to strings for consistency
        if isinstance(symlink_to, Path):
            symlink_to = str(symlink_to)

        # Clean the symlink path
        symlink_to = clean(symlink_to)

        # Ensure the symlink directory exists (only if there is a directory component)
        symlink_dir = _os.path.dirname(symlink_to)
        if symlink_dir:  # Only create directory if there's a directory component
            _os.makedirs(symlink_dir, exist_ok=True)

        # Remove existing symlink or file
        sh(["rm", "-f", f"{symlink_to}"], verbose=False)

        # Create the symlink using relative path for robustness
        sh(["ln", "-sfr", f"{spath_final}", f"{symlink_to}"], verbose=False)

        if verbose:
            symlink_to_full = (
                os.path.realpath(symlink_to) + "/" + os.path.basename(spath_final)
            )
            logger.success(f"Symlinked: {spath_final} -> {symlink_to_full}")


def _save_pltz_bundle(obj, spath, as_zip=False, data=None, layered=True, **kwargs):
    """Save a matplotlib figure as a .pltz bundle.

    Bundle structure v2.0 (layered - default):
        plot.pltz.d/
            spec.json           # Semantic: WHAT to plot (canonical)
            style.json          # Appearance: HOW it looks (canonical)
            data.csv            # Raw data (immutable)
            exports/            # PNG, SVG, hitmap
            cache/              # geometry_px.json, render_manifest.json

    Bundle structure v1.0 (legacy):
        plot.json  - specification (axes, styles, theme, etc.)
        plot.csv   - raw data (immutable)
        plot.png   - raster export (required)
        plot.svg   - vector export (optional)
        plot.pdf   - publication export (optional)

    Parameters
    ----------
    obj : matplotlib.figure.Figure
        The figure to save.
    spath : str or Path
        Output path (e.g., "plot.pltz.d" or "plot.pltz").
    as_zip : bool
        If True, save as ZIP archive.
    data : pandas.DataFrame, optional
        Data to embed in the bundle as plot.csv.
    layered : bool
        If True (default), use new layered format (spec/style/geometry).
        If False, use legacy single JSON format.
    **kwargs
        Additional arguments passed to savefig.
    """
    from pathlib import Path
    import tempfile
    import json
    import numpy as np
    from ._bundle import save_bundle, BundleType

    p = Path(spath)

    # Extract basename from path (e.g., "myplot.pltz" -> "myplot", "myplot.pltz.d" -> "myplot")
    basename = p.stem  # e.g., "myplot.pltz" or "myplot"
    if basename.endswith('.pltz'):
        basename = basename[:-5]  # Remove .pltz suffix
    elif basename.endswith('.d'):
        # Handle myplot.pltz.d -> myplot.pltz -> myplot
        basename = Path(basename).stem
        if basename.endswith('.pltz'):
            basename = basename[:-5]

    # Extract figure from various matplotlib object types
    import matplotlib.figure
    fig = obj
    if hasattr(obj, 'figure'):
        fig = obj.figure
    elif hasattr(obj, 'fig'):
        fig = obj.fig

    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(f"Expected matplotlib Figure, got {type(obj).__name__}")

    dpi = kwargs.pop('dpi', 300)

    # === Always use layered format ===
    from scitex.plt.io import save_layered_pltz_bundle
    import shutil
    import tempfile

    # Determine bundle directory path
    if as_zip:
        # For ZIP: save to temp dir, then compress
        temp_dir = Path(tempfile.mkdtemp())
        bundle_dir = temp_dir / f"{basename}.pltz.d"
        zip_path = p if not str(p).endswith('.d') else Path(str(p)[:-2])
    else:
        # For directory: save directly
        bundle_dir = p if str(p).endswith('.d') else Path(str(p) + '.d')

    # Get CSV data from figure if not provided
    csv_df = data
    if csv_df is None:
        csv_source = _get_figure_with_data(obj)
        if csv_source is not None and hasattr(csv_source, 'export_as_csv'):
            try:
                csv_df = csv_source.export_as_csv()
            except Exception:
                pass

    save_layered_pltz_bundle(
        fig=fig,
        bundle_dir=bundle_dir,
        basename=basename,
        dpi=dpi,
        csv_df=csv_df,
    )

    # Compress to ZIP if requested
    if as_zip:
        from ._bundle import pack_bundle
        pack_bundle(bundle_dir, zip_path)
        shutil.rmtree(temp_dir)  # Clean up temp directory

    return  # Done with layered format

    # === Legacy format below (DEPRECATED - kept for reference) ===

    # Calculate size info
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    fig_dpi = fig.get_dpi()

    # Build spec according to contract (using basename for file references)
    spec = {
        'schema': {'name': 'scitex.plt.plot', 'version': '1.0.0'},
        'backend': 'mpl',
        'data': {
            'source': f'{basename}.csv',
            'path': f'{basename}.csv',
            'hash': None,  # Will be computed after data extraction
            'columns': [],  # Will be populated after data extraction
        },
        'size': {
            'width_inch': round(fig_width_inch, 2),
            'height_inch': round(fig_height_inch, 2),
            'width_mm': round(fig_width_inch * 25.4, 2),
            'height_mm': round(fig_height_inch * 25.4, 2),
            'width_px': int(fig_width_inch * dpi),
            'height_px': int(fig_height_inch * dpi),
            'dpi': dpi,
            'crop_margin_mm': 1.0,
        },
        'axes': [],
        'theme': {
            'mode': 'light',
            'colors': {
                'background': 'transparent',
                'axes_bg': 'white',
                'text': 'black',
                'spine': 'black',
                'tick': 'black',
            }
        },
    }

    # Extract data from plot lines if no data provided
    extracted_data = {}

    # Extract axes metadata
    for i, ax in enumerate(fig.axes):
        # Get axes bounding box in figure coordinates (0-1)
        bbox = ax.get_position()

        ax_info = {
            'xlabel': ax.get_xlabel() or None,
            'ylabel': ax.get_ylabel() or None,
            'title': ax.get_title() or None,
            'xlim': [round(v, 2) for v in ax.get_xlim()],
            'ylim': [round(v, 2) for v in ax.get_ylim()],
            'plot_type': 'line',  # Default, could be detected
            # Bounding box in normalized figure coordinates (0-1)
            'bbox': {
                'x0': round(bbox.x0, 4),
                'y0': round(bbox.y0, 4),
                'x1': round(bbox.x1, 4),
                'y1': round(bbox.y1, 4),
                'width': round(bbox.width, 4),
                'height': round(bbox.height, 4),
            },
            # Bounding box in mm
            'bbox_mm': {
                'x0': round(bbox.x0 * fig_width_inch * 25.4, 2),
                'y0': round(bbox.y0 * fig_height_inch * 25.4, 2),
                'x1': round(bbox.x1 * fig_width_inch * 25.4, 2),
                'y1': round(bbox.y1 * fig_height_inch * 25.4, 2),
                'width': round(bbox.width * fig_width_inch * 25.4, 2),
                'height': round(bbox.height * fig_height_inch * 25.4, 2),
            },
            # Bounding box in pixels
            'bbox_px': {
                'x0': int(bbox.x0 * fig_width_inch * dpi),
                'y0': int(bbox.y0 * fig_height_inch * dpi),
                'x1': int(bbox.x1 * fig_width_inch * dpi),
                'y1': int(bbox.y1 * fig_height_inch * dpi),
                'width': int(bbox.width * fig_width_inch * dpi),
                'height': int(bbox.height * fig_height_inch * dpi),
            },
        }

        # SciTeX-specific axis dimensions
        if hasattr(ax, '_scitex_axes_width_mm'):
            ax_info['axes_width_mm'] = ax._scitex_axes_width_mm
        else:
            ax_info['axes_width_mm'] = round(bbox.width * fig_width_inch * 25.4, 1)

        if hasattr(ax, '_scitex_axes_height_mm'):
            ax_info['axes_height_mm'] = ax._scitex_axes_height_mm
        else:
            ax_info['axes_height_mm'] = round(bbox.height * fig_height_inch * 25.4, 1)

        # Extract line data for CSV and build lines array
        lines_info = []
        for j, line in enumerate(ax.get_lines()):
            label = line.get_label()
            if label is None or label.startswith('_'):
                label = f'series_{j}'
            xdata, ydata = line.get_data()
            if len(xdata) > 0:
                col_x = f'{label}_x' if i == 0 else f'ax{i}_{label}_x'
                col_y = f'{label}_y' if i == 0 else f'ax{i}_{label}_y'
                extracted_data[col_x] = np.array(xdata)
                extracted_data[col_y] = np.array(ydata)

                # Get line color (convert RGBA to hex)
                color = line.get_color()
                if isinstance(color, (list, tuple)):
                    import matplotlib.colors as mcolors
                    color = mcolors.to_hex(color)

                lines_info.append({
                    'label': label,
                    'x_col': col_x,
                    'y_col': col_y,
                    'color': color,
                    'linewidth': line.get_linewidth(),
                })

        if lines_info:
            ax_info['lines'] = lines_info

        spec['axes'].append(ax_info)

    # Handle theme from figure
    if hasattr(fig, '_scitex_theme'):
        theme_mode = fig._scitex_theme
        spec['theme']['mode'] = theme_mode
        # Update colors based on theme mode
        if theme_mode == 'dark':
            spec['theme']['colors'] = {
                'background': 'transparent',
                'axes_bg': 'transparent',
                'text': '#e8e8e8',
                'spine': '#e8e8e8',
                'tick': '#e8e8e8',
            }
            # Re-apply theme colors to ensure legends and other elements get the correct colors
            from scitex.plt.utils._figure_mm import _apply_theme_colors
            for ax in fig.axes:
                _apply_theme_colors(ax, theme='dark')

    # Build bundle data (include basename for file naming)
    bundle_data = {'spec': spec, 'basename': basename}

    # Use provided data or extracted data for CSV
    # Priority: 1) explicit data param, 2) export_as_csv method, 3) line extraction fallback
    csv_df = None
    if data is not None:
        csv_df = data
        bundle_data['data'] = data
    else:
        # Try to use export_as_csv from SciTeX wrapped objects (handles all plot types)
        csv_source = _get_figure_with_data(obj)
        if csv_source is not None and hasattr(csv_source, 'export_as_csv'):
            try:
                csv_df = csv_source.export_as_csv()
                if csv_df is not None and not csv_df.empty:
                    bundle_data['data'] = csv_df
                    logger.debug(f"CSV data extracted via export_as_csv: {len(csv_df)} rows, {len(csv_df.columns)} cols")
            except Exception as e:
                logger.debug(f"export_as_csv failed: {e}")
                csv_df = None

        # Fallback to line extraction if export_as_csv didn't work
        if csv_df is None and extracted_data:
            try:
                import pandas as pd
                # Pad arrays to same length
                max_len = max(len(v) for v in extracted_data.values())
                padded = {}
                for k, v in extracted_data.items():
                    if len(v) < max_len:
                        padded[k] = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
                    else:
                        padded[k] = v
                csv_df = pd.DataFrame(padded)
                bundle_data['data'] = csv_df
                logger.debug(f"CSV data extracted via line fallback: {len(csv_df)} rows")
            except ImportError:
                pass

    # Compute hash and columns for data section
    if csv_df is not None:
        import hashlib
        # Get CSV string for hash computation
        csv_str = csv_df.to_csv(index=False)
        csv_hash = hashlib.sha256(csv_str.encode()).hexdigest()
        spec['data']['hash'] = f'sha256:{csv_hash[:16]}'
        spec['data']['columns'] = list(csv_df.columns)

    # Save figure to multiple formats
    import warnings
    from PIL import Image as PILImage
    from scitex.plt.utils._hitmap import (
        apply_hitmap_colors, restore_original_colors, extract_path_data,
        extract_selectable_regions, HITMAP_BACKGROUND_COLOR, HITMAP_AXES_COLOR
    )

    crop_box = None
    color_map = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Suppress tight_layout warnings for SciTeX figures with custom axes
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*tight_layout.*')

            # Always use transparent background for SciTeX figures (both light and dark themes)
            use_transparent = True

            # Save PNG (raster) - required
            png_path = tmp_path / "plot.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png', transparent=use_transparent)

            # Save SVG (vector) - optional
            svg_path = tmp_path / "plot.svg"
            fig.savefig(svg_path, bbox_inches='tight', format='svg')

            # Save PDF (vector) - optional
            pdf_path = tmp_path / "plot.pdf"
            fig.savefig(pdf_path, bbox_inches='tight', format='pdf')

            # Now generate hitmap by applying ID colors to data elements ONLY
            # Keep axes/spines/labels with original colors to preserve bbox_inches='tight' bounds
            # Also detects logical groups (histogram, bar_series, etc.)
            original_props, color_map, groups = apply_hitmap_colors(fig)

            # Store original background colors and set hitmap colors
            original_fig_facecolor = fig.patch.get_facecolor()
            original_ax_facecolors = []
            original_ax_props = []
            for ax in fig.axes:
                original_ax_facecolors.append(ax.get_facecolor())
                # Store axis element colors for restoration
                ax_props = {
                    'ax': ax,
                    'spine_colors': {k: v.get_edgecolor() for k, v in ax.spines.items()},
                    'tick_colors': ax.tick_params,  # Will restore later
                    'xlabel_color': ax.xaxis.label.get_color(),
                    'ylabel_color': ax.yaxis.label.get_color(),
                    'title_color': ax.title.get_color(),
                }
                original_ax_props.append(ax_props)
                # Set hitmap colors for non-data elements
                ax.set_facecolor(HITMAP_BACKGROUND_COLOR)
                for spine in ax.spines.values():
                    spine.set_color(HITMAP_AXES_COLOR)
                ax.tick_params(colors=HITMAP_AXES_COLOR, labelcolor=HITMAP_AXES_COLOR)
                ax.xaxis.label.set_color(HITMAP_AXES_COLOR)
                ax.yaxis.label.set_color(HITMAP_AXES_COLOR)
                ax.title.set_color(HITMAP_AXES_COLOR)

            fig.patch.set_facecolor(HITMAP_BACKGROUND_COLOR)

            # Save hitmap PNG with same bbox_inches='tight'
            hitmap_path = tmp_path / "plot_hitmap.png"
            fig.savefig(hitmap_path, dpi=dpi, bbox_inches='tight', format='png', facecolor=HITMAP_BACKGROUND_COLOR)

            # Optimize hitmap PNG size using zlib compression
            try:
                hitmap_img = PILImage.open(hitmap_path).convert('RGB')
                hitmap_img.save(hitmap_path, format='PNG', optimize=True, compress_level=9)
            except Exception:
                pass  # Keep original if optimization fails

            # Save hitmap SVG with same bbox_inches='tight'
            hitmap_svg_path = tmp_path / "plot_hitmap.svg"
            fig.savefig(hitmap_svg_path, bbox_inches='tight', format='svg')

            # Restore original colors (data elements)
            restore_original_colors(original_props)

            # Restore original figure and axes colors
            fig.patch.set_facecolor(original_fig_facecolor)
            for i, ax in enumerate(fig.axes):
                ax.set_facecolor(original_ax_facecolors[i])
                if i < len(original_ax_props):
                    props = original_ax_props[i]
                    for spine_name, color in props['spine_colors'].items():
                        ax.spines[spine_name].set_edgecolor(color)
                    ax.xaxis.label.set_color(props['xlabel_color'])
                    ax.yaxis.label.set_color(props['ylabel_color'])
                    ax.title.set_color(props['title_color'])

            # Now apply auto-crop to BOTH PNG and hitmap with same parameters
            try:
                from scitex.plt.utils._crop import crop

                # Crop PNG and get crop coordinates
                _, crop_offset = crop(
                    str(png_path),
                    output_path=str(png_path),
                    overwrite=True,
                    margin=12,  # ~1mm at 300 DPI
                    verbose=False,
                    return_offset=True,
                )
                crop_box = (crop_offset['left'], crop_offset['upper'],
                           crop_offset['right'], crop_offset['lower'])

                # Apply SAME crop to hitmap PNG
                crop(
                    str(hitmap_path),
                    output_path=str(hitmap_path),
                    overwrite=True,
                    crop_box=crop_box,
                    verbose=False,
                )
            except Exception as e:
                crop_box = None
                logger.debug(f"Crop failed: {e}")

            # Validate sizes match
            with PILImage.open(png_path) as png_img, PILImage.open(hitmap_path) as hm_img:
                if png_img.size != hm_img.size:
                    logger.warning(f"Size mismatch: PNG={png_img.size}, Hitmap={hm_img.size}")

            with open(png_path, 'rb') as f:
                bundle_data['png'] = f.read()

            with open(hitmap_path, 'rb') as f:
                bundle_data['hitmap_png'] = f.read()

            with open(svg_path, 'rb') as f:
                bundle_data['svg'] = f.read()

            with open(hitmap_svg_path, 'rb') as f:
                bundle_data['hitmap_svg'] = f.read()

            with open(pdf_path, 'rb') as f:
                bundle_data['pdf'] = f.read()

    # Add hit_regions to spec
    try:
        path_data = extract_path_data(fig)

        spec['hit_regions'] = {
            'strategy': 'hybrid',
            'hit_map': f'{basename}_hitmap.png',
            'hit_map_svg': f'{basename}_hitmap.svg',
            'color_map': {str(k): v for k, v in color_map.items()},
            'groups': groups,  # Logical groups (histogram, bar_series, etc.)
            'path_data': path_data,
        }

        if crop_box is not None:
            spec['hit_regions']['crop_box'] = {
                'left': int(crop_box[0]),
                'upper': int(crop_box[1]),
                'right': int(crop_box[2]),
                'lower': int(crop_box[3]),
            }

        # Extract selectable regions (bounding boxes for axis/annotation elements)
        # This complements hitmap color-based selection with bbox-based selection
        selectable_regions = extract_selectable_regions(fig)
        if selectable_regions and selectable_regions.get('axes'):
            spec['selectable_regions'] = selectable_regions

    except Exception as e:
        logger.debug(f"Hit regions spec failed: {e}")

    # Save the bundle
    save_bundle(bundle_data, p, bundle_type=BundleType.PLTZ, as_zip=as_zip)


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
    # Don't use object's own save method - use consistent handlers
    # This ensures all saves go through the same pipeline and get
    # the yellow confirmation message

    # Get file extension
    ext = _os.path.splitext(spath)[1].lower()

    # Handle .canvas directories (special case - path ends with .canvas)
    if spath.endswith(".canvas"):
        save_canvas(obj, spath, **kwargs)
        return

    # Handle bundle formats (.figz, .pltz, .statsz and their .d variants)
    # These use special naming: file.figz (ZIP) or file.figz.d (directory)
    # Note: .figz defaults to ZIP (as_zip=True), .pltz/.statsz default to directory
    bundle_extensions = (".figz", ".pltz", ".statsz")
    for bext in bundle_extensions:
        if spath.endswith(bext) or spath.endswith(f"{bext}.d"):
            # Remove as_zip from kwargs if present to avoid duplicate
            bundle_kwargs = {k: v for k, v in kwargs.items() if k != 'as_zip'}
            as_zip = kwargs.get('as_zip', not spath.endswith(".d"))
            if bext == ".figz":
                import scitex.fig as sfig
                # figz defaults to ZIP, so always pass as_zip explicitly
                sfig.save_figz(obj, spath, as_zip=as_zip, **bundle_kwargs)
            elif bext == ".pltz":
                _save_pltz_bundle(obj, spath, as_zip=as_zip, **bundle_kwargs)
            elif bext == ".statsz":
                import scitex.stats as sstats
                sstats.save_statsz(obj, spath, as_zip=as_zip, **bundle_kwargs)

            # Log "Saved to:" for bundle formats (consistent with other formats)
            # For bundles, determine the actual saved path (zip or directory)
            bundle_path = spath if as_zip else f"{spath}.d" if not spath.endswith(".d") else spath

            if verbose and _os.path.exists(bundle_path):
                file_size = getsize(bundle_path)
                file_size = readable_bytes(file_size)
                try:
                    rel_path = _os.path.relpath(bundle_path, _os.getcwd())
                except ValueError:
                    rel_path = bundle_path
                logger.success(f"Saved to: ./{rel_path} ({file_size})")

            # Handle symlinks for bundle formats (consistent with other formats)
            if symlink_from_cwd and _os.path.exists(bundle_path):
                # Create symlink from cwd to bundle path
                bundle_basename = _os.path.basename(bundle_path)
                bundle_cwd = _os.path.join(_os.getcwd(), bundle_basename)
                _symlink(bundle_path, bundle_cwd, symlink_from_cwd, verbose)

            if symlink_to and _os.path.exists(bundle_path):
                _symlink_to(bundle_path, symlink_to, verbose)

            return

    # Try dispatch dictionary first for O(1) lookup
    if ext in _FILE_HANDLERS:
        # Check if handler needs special parameters
        if ext in [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".tiff",
            ".tif",
            ".svg",
            ".pdf",
        ]:
            _FILE_HANDLERS[ext](
                obj,
                spath,
                verbose=verbose,
                no_csv=no_csv,
                symlink_from_cwd=symlink_from_cwd,
                symlink_to=symlink_to,
                dry_run=dry_run,
                auto_crop=auto_crop,
                crop_margin_mm=crop_margin_mm,
                metadata_extra=metadata_extra,
                json_schema=json_schema,
                **kwargs,
            )
        elif ext in [".hdf5", ".h5", ".zarr"]:
            # HDF5 and Zarr files may need special 'key' parameter
            _FILE_HANDLERS[ext](obj, spath, **kwargs)
        else:
            _FILE_HANDLERS[ext](obj, spath, **kwargs)
    # csv - special case as it doesn't have a dot prefix in dispatch
    elif spath.endswith(".csv"):
        save_csv(obj, spath, **kwargs)
    # Check for special extension cases not in dispatch
    elif spath.endswith(".pkl.gz"):
        save_pickle_compressed(obj, spath, **kwargs)
    else:
        logger.warning(f"Unsupported file format. {spath} was not saved.")

    if verbose:
        if _os.path.exists(spath):
            file_size = getsize(spath)
            file_size = readable_bytes(file_size)
            # Get relative path from current working directory
            try:
                rel_path = _os.path.relpath(spath, _os.getcwd())
            except ValueError:
                rel_path = spath

            logger.success(f"Saved to: ./{rel_path} ({file_size})")


def _save_separate_legends(obj, spath, symlink_from_cwd=False, dry_run=False, **kwargs):
    """Save separate legend files if ax.legend('separate') was used."""
    if dry_run:
        return

    import matplotlib.figure
    import matplotlib.pyplot as plt

    # Get the matplotlib figure object
    fig = None
    if isinstance(obj, matplotlib.figure.Figure):
        fig = obj
    elif hasattr(obj, "_fig_mpl"):
        fig = obj._fig_mpl
    elif hasattr(obj, "figure"):
        if isinstance(obj.figure, matplotlib.figure.Figure):
            fig = obj.figure
        elif hasattr(obj.figure, "_fig_mpl"):
            fig = obj.figure._fig_mpl

    if fig is None:
        return

    # Check if there are separate legend parameters stored
    if not hasattr(fig, "_separate_legend_params"):
        return

    # Save each legend as a separate file
    base_path = _os.path.splitext(spath)[0]
    ext = _os.path.splitext(spath)[1]

    for legend_params in fig._separate_legend_params:
        # Create a new figure for the legend
        legend_fig = plt.figure(figsize=legend_params["figsize"])
        legend_ax = legend_fig.add_subplot(111)

        # Create the legend
        legend = legend_ax.legend(
            legend_params["handles"],
            legend_params["labels"],
            loc="center",
            frameon=legend_params["frameon"],
            fancybox=legend_params["fancybox"],
            shadow=legend_params["shadow"],
            **legend_params["kwargs"],
        )

        # Remove axes
        legend_ax.axis("off")

        # Adjust layout to fit the legend
        legend_fig.tight_layout()

        # Save the legend figure
        legend_filename = f"{base_path}_{legend_params['axis_id']}_legend{ext}"
        save_image(legend_fig, legend_filename, **kwargs)

        # Close the legend figure to free memory
        plt.close(legend_fig)

        if not dry_run and _os.path.exists(legend_filename):
            file_size = getsize(legend_filename)
            file_size = readable_bytes(file_size)
            print(
                color_text(
                    f"\nSaved legend to: {legend_filename} ({file_size})",
                    c="yellow",
                )
            )


def _handle_image_with_csv(
    obj,
    spath,
    verbose=False,
    no_csv=False,
    symlink_from_cwd=False,
    dry_run=False,
    symlink_to=None,
    auto_crop=True,
    crop_margin_mm=1.0,
    metadata_extra=None,
    json_schema="editable",
    **kwargs,
):
    """Handle image file saving with optional CSV export and auto-cropping."""
    if dry_run:
        return

    # Auto-collect metadata from scitex figures if not explicitly provided
    collected_metadata = None
    if "metadata" not in kwargs or kwargs["metadata"] is None:
        try:
            # Check if this is a matplotlib figure or scitex wrapper
            import matplotlib.figure

            fig_mpl = None
            if isinstance(obj, matplotlib.figure.Figure):
                fig_mpl = obj
            elif hasattr(obj, "_fig_mpl"):  # FigWrapper
                fig_mpl = obj._fig_mpl
            elif hasattr(obj, "figure") and isinstance(
                obj.figure, matplotlib.figure.Figure
            ):
                fig_mpl = obj.figure

            # If we have a figure, try to collect metadata
            if fig_mpl is not None:
                # Get axes from scitex wrapper if available (for multi-axes support)
                # Priority: FigWrapper.axes (AxesWrapper) > mpl axes with _scitex_wrapper > mpl axes
                ax = None

                # First try to get AxesWrapper from FigWrapper (obj)
                if hasattr(obj, "axes"):
                    # obj is FigWrapper, get its axes (could be AxisWrapper or AxesWrapper)
                    ax = obj.axes
                elif hasattr(fig_mpl, "axes") and len(fig_mpl.axes) > 0:
                    mpl_ax = fig_mpl.axes[0]
                    # Try to get scitex wrapper which has history for recipe schema
                    if hasattr(mpl_ax, '_scitex_wrapper'):
                        ax = mpl_ax._scitex_wrapper
                    else:
                        ax = mpl_ax

                # Collect metadata using scitex's metadata collector
                try:
                    if json_schema == "editable":
                        from scitex.plt.utils.metadata import export_editable_figure
                        auto_metadata = export_editable_figure(fig_mpl)
                    elif json_schema == "recipe":
                        from scitex.plt.utils import collect_recipe_metadata
                        auto_metadata = collect_recipe_metadata(
                            fig_mpl, ax,
                            auto_crop=auto_crop,
                            crop_margin_mm=crop_margin_mm,
                        )
                    else:
                        from scitex.plt.utils import collect_figure_metadata
                        auto_metadata = collect_figure_metadata(fig_mpl, ax)

                    if auto_metadata:
                        kwargs["metadata"] = auto_metadata
                        collected_metadata = auto_metadata  # Save for JSON export
                        if verbose:
                            schema_names = {"editable": "editable v0.3", "recipe": "recipe", "verbose": "verbose"}
                            schema_name = schema_names.get(json_schema, json_schema)
                            logger.info(f"  • Auto-collected metadata ({schema_name} schema)")
                except ImportError:
                    pass  # collect_figure_metadata not available
                except Exception as e:
                    if verbose:
                        logger.warning(f"Could not auto-collect metadata: {e}")
        except Exception:
            pass  # Silently continue if auto-collection fails
    else:
        # Use explicitly provided metadata
        collected_metadata = kwargs.get("metadata")

    # Merge metadata_extra with collected_metadata
    if metadata_extra is not None and collected_metadata is not None:
        # Deep merge: metadata_extra takes precedence
        import copy

        collected_metadata = copy.deepcopy(collected_metadata)

        # If metadata_extra has plot_type and it doesn't exist in collected, add it
        if "plot_type" in metadata_extra:
            collected_metadata["plot_type"] = metadata_extra["plot_type"]

        # Merge style information
        if "style" in metadata_extra:
            collected_metadata["style"] = metadata_extra["style"]

        # Merge any other fields from metadata_extra
        for key, value in metadata_extra.items():
            if key not in ["plot_type", "style"]:
                collected_metadata[key] = value

        # Update kwargs metadata for image saving
        kwargs["metadata"] = collected_metadata

    save_image(obj, spath, verbose=verbose, **kwargs)

    # Auto-crop if requested (only for raster formats)
    crop_offset = None
    if auto_crop and not dry_run:
        # Get file extension
        ext = spath.lower()

        # Only crop raster formats (PNG, JPEG, TIFF)
        # Skip vector formats (PDF, SVG) as they don't benefit from cropping
        if ext.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif")):
            try:
                from scitex.plt.utils._crop import crop

                # Convert mm to pixels (assuming 300 DPI)
                # 1mm at 300 DPI = 11.81 pixels ≈ 12 pixels
                dpi = kwargs.get("dpi", 300)
                margin_px = int(crop_margin_mm * dpi / 25.4)  # 25.4mm per inch

                # Crop the saved image in place, get crop offset for metadata adjustment
                _, crop_offset = crop(
                    spath,
                    output_path=spath,
                    margin=margin_px,
                    overwrite=True,
                    verbose=False,
                    return_offset=True,
                )

                # Adjust axes_bbox_px in metadata to account for crop offset
                if crop_offset and collected_metadata:
                    if "axes_bbox_px" in collected_metadata:
                        bbox = collected_metadata["axes_bbox_px"]
                        # Subtract crop offset from all coordinates
                        # left/upper is where the crop started
                        left_offset = crop_offset["left"]
                        upper_offset = crop_offset["upper"]
                        bbox["x0"] = bbox.get("x0", 0) - left_offset
                        bbox["x1"] = bbox.get("x1", 0) - left_offset
                        bbox["y0"] = bbox.get("y0", 0) - upper_offset
                        bbox["y1"] = bbox.get("y1", 0) - upper_offset
                        # Update width/height to match new image size
                        # (bbox width/height shouldn't change, but figure size does)

                    # Also update figure size in metadata
                    if "figure" in collected_metadata:
                        fig_meta = collected_metadata["figure"]
                        if "size_px" in fig_meta:
                            fig_meta["size_px"] = [
                                crop_offset["new_width"],
                                crop_offset["new_height"],
                            ]
                    if "dimensions" in collected_metadata:
                        dim_meta = collected_metadata["dimensions"]
                        if "figure_size_px" in dim_meta:
                            dim_meta["figure_size_px"] = [
                                crop_offset["new_width"],
                                crop_offset["new_height"],
                            ]

                if verbose:
                    logger.info(
                        f"  • Auto-cropped with {crop_margin_mm}mm margin ({margin_px}px at {dpi} DPI)"
                    )

            except Exception as e:
                logger.warning(f"Auto-crop failed: {e}. Image saved without cropping.")

    # Handle separate legend saving
    _save_separate_legends(
        obj,
        spath,
        symlink_from_cwd=symlink_from_cwd,
        dry_run=dry_run,
        **kwargs,
    )

    if not no_csv:
        ext = _os.path.splitext(spath)[1].lower()
        ext_wo_dot = ext.replace(".", "")

        # Check if the path contains an image extension directory (e.g., ./png/, ./jpg/)
        # If so, save CSV in a parallel ./csv/ directory
        image_extensions = ["png", "jpg", "jpeg", "gif", "tiff", "tif", "svg", "pdf"]
        parent_dir = _os.path.dirname(spath)
        parent_name = _os.path.basename(parent_dir)
        filename_without_ext = _os.path.splitext(_os.path.basename(spath))[0]

        csv_path = None  # Initialize to avoid UnboundLocalError when CSV export is skipped
        try:
            # Get the figure object that may contain plot data
            fig_obj = _get_figure_with_data(obj)

            if fig_obj is not None:
                # Save regular CSV if export method exists
                if hasattr(fig_obj, "export_as_csv"):
                    csv_data = fig_obj.export_as_csv()
                    if csv_data is not None and not csv_data.empty:
                        # Determine CSV path based on parent directory name
                        if parent_name.lower() in image_extensions:
                            # Parent directory is named after an image extension (e.g., png/)
                            # Create parallel csv/ directory
                            grandparent_dir = _os.path.dirname(parent_dir)
                            csv_dir = _os.path.join(grandparent_dir, "csv")
                            csv_path = _os.path.join(
                                csv_dir, filename_without_ext + ".csv"
                            )
                        else:
                            # Save CSV in same directory as image
                            csv_path = _os.path.splitext(spath)[0] + ".csv"

                        # Ensure parent directory exists
                        _os.makedirs(_os.path.dirname(csv_path), exist_ok=True)
                        # Save directly using _save to avoid path doubling
                        # Don't pass image-specific kwargs to CSV save
                        _save(
                            csv_data,
                            csv_path,
                            verbose=True,
                            symlink_from_cwd=False,  # Will handle symlink manually
                            dry_run=dry_run,
                            no_csv=True,
                        )

                        # Update metadata with actual CSV info (after export)
                        # This ensures column names match exactly, including any
                        # deduplication suffixes added by pandas
                        if collected_metadata is not None:
                            try:
                                from scitex.plt.utils._collect_figure_metadata import (
                                    _compute_csv_hash,
                                )

                                # Ensure data section exists
                                if "data" not in collected_metadata:
                                    collected_metadata["data"] = {}

                                # Get actual column names from exported DataFrame
                                actual_columns = list(csv_data.columns)

                                # Update data section with csv_path (relative to JSON)
                                # Since JSON and CSV are in the same or parallel directories,
                                # use just the filename for simplicity
                                collected_metadata["data"]["csv_path"] = _os.path.basename(csv_path)

                                # Update columns to use flat list of actual columns
                                collected_metadata["data"]["columns_actual"] = actual_columns

                                # Compute hash of actual CSV data
                                collected_metadata["data"]["csv_hash"] = _compute_csv_hash(
                                    csv_data
                                )
                            except Exception:
                                pass  # Silently continue if update fails

                        # Create symlink_to for CSV if it was specified for the image
                        if symlink_to:
                            # Apply same directory transformation for symlink
                            symlink_parent_dir = _os.path.dirname(symlink_to)
                            symlink_parent_name = _os.path.basename(symlink_parent_dir)
                            symlink_filename_without_ext = _os.path.splitext(
                                _os.path.basename(symlink_to)
                            )[0]

                            if symlink_parent_name.lower() in image_extensions:
                                symlink_grandparent_dir = _os.path.dirname(
                                    symlink_parent_dir
                                )
                                csv_symlink_to = _os.path.join(
                                    symlink_grandparent_dir,
                                    "csv",
                                    symlink_filename_without_ext + ".csv",
                                )
                            else:
                                csv_symlink_to = (
                                    _os.path.splitext(symlink_to)[0] + ".csv"
                                )

                            _symlink_to(csv_path, csv_symlink_to, True)

                        # Create symlink for CSV manually if needed
                        if symlink_from_cwd:
                            # Get the relative path from the original specified path
                            # This preserves the directory structure for the symlink
                            import inspect

                            frame_info = inspect.stack()
                            # Find the original specified_path from the parent save() call
                            for frame in frame_info:
                                if "specified_path" in frame.frame.f_locals:
                                    original_path = frame.frame.f_locals[
                                        "specified_path"
                                    ]
                                    if isinstance(original_path, str):
                                        # Apply same directory transformation for symlink
                                        orig_parent_dir = _os.path.dirname(
                                            original_path
                                        )
                                        orig_parent_name = _os.path.basename(
                                            orig_parent_dir
                                        )
                                        orig_filename_without_ext = _os.path.splitext(
                                            _os.path.basename(original_path)
                                        )[0]

                                        if orig_parent_name.lower() in image_extensions:
                                            orig_grandparent_dir = _os.path.dirname(
                                                orig_parent_dir
                                            )
                                            csv_relative = _os.path.join(
                                                orig_grandparent_dir,
                                                "csv",
                                                orig_filename_without_ext + ".csv",
                                            )
                                        else:
                                            csv_relative = original_path.replace(
                                                _os.path.splitext(original_path)[1],
                                                ".csv",
                                            )

                                        csv_cwd = _os.path.join(
                                            _os.getcwd(), csv_relative
                                        )
                                        _symlink(csv_path, csv_cwd, True, True)
                                        break
                            else:
                                # Fallback to basename if we can't find the original path
                                csv_cwd = (
                                    _os.getcwd() + "/" + _os.path.basename(csv_path)
                                )
                                _symlink(csv_path, csv_cwd, True, True)

                # Save SigmaPlot CSV if method exists
                if hasattr(fig_obj, "export_as_csv_for_sigmaplot"):
                    sigmaplot_data = fig_obj.export_as_csv_for_sigmaplot()
                    if sigmaplot_data is not None and not sigmaplot_data.empty:
                        # Determine SigmaPlot CSV path based on parent directory name
                        if parent_name.lower() in image_extensions:
                            grandparent_dir = _os.path.dirname(parent_dir)
                            csv_dir = _os.path.join(grandparent_dir, "csv")
                            csv_sigmaplot_path = _os.path.join(
                                csv_dir, filename_without_ext + "_for_sigmaplot.csv"
                            )
                        else:
                            csv_sigmaplot_path = spath.replace(
                                ext_wo_dot, "csv"
                            ).replace(".csv", "_for_sigmaplot.csv")

                        # Ensure parent directory exists
                        _os.makedirs(
                            _os.path.dirname(csv_sigmaplot_path), exist_ok=True
                        )
                        # Save directly using _save to avoid path doubling
                        # Don't pass image-specific kwargs to CSV save
                        _save(
                            sigmaplot_data,
                            csv_sigmaplot_path,
                            verbose=True,
                            symlink_from_cwd=False,  # Will handle symlink manually
                            dry_run=dry_run,
                            no_csv=True,
                        )

                        # Create symlink_to for SigmaPlot CSV if it was specified for the image
                        if symlink_to:
                            symlink_parent_dir = _os.path.dirname(symlink_to)
                            symlink_parent_name = _os.path.basename(symlink_parent_dir)
                            symlink_filename_without_ext = _os.path.splitext(
                                _os.path.basename(symlink_to)
                            )[0]

                            if symlink_parent_name.lower() in image_extensions:
                                symlink_grandparent_dir = _os.path.dirname(
                                    symlink_parent_dir
                                )
                                csv_sigmaplot_symlink_to = _os.path.join(
                                    symlink_grandparent_dir,
                                    "csv",
                                    symlink_filename_without_ext + "_for_sigmaplot.csv",
                                )
                            else:
                                csv_sigmaplot_symlink_to = (
                                    _os.path.splitext(symlink_to)[0]
                                    + "_for_sigmaplot.csv"
                                )

                            _symlink_to(
                                csv_sigmaplot_path,
                                csv_sigmaplot_symlink_to,
                                True,
                            )

                        # Create symlink for SigmaPlot CSV manually if needed
                        if symlink_from_cwd:
                            csv_cwd = (
                                _os.getcwd()
                                + "/"
                                + _os.path.basename(csv_sigmaplot_path)
                            )
                            _symlink(csv_sigmaplot_path, csv_cwd, True, True)
        except Exception as e:
            logger.warning(f"CSV export failed: {e}")

    # Save metadata as JSON if collected
    if collected_metadata is not None and not dry_run:
        try:
            # Check if the path contains an image extension directory (e.g., ./png/, ./jpg/)
            # If so, save JSON in a parallel ./json/ directory
            # Example: ./path/to/output/png/fig.png -> ./path/to/output/json/fig.json
            # Example: ./path/to/output/fig.png -> ./path/to/output/fig.json (same dir)
            image_extensions = [
                "png",
                "jpg",
                "jpeg",
                "gif",
                "tiff",
                "tif",
                "svg",
                "pdf",
            ]
            parent_dir = _os.path.dirname(spath)
            parent_name = _os.path.basename(parent_dir)
            filename_without_ext = _os.path.splitext(_os.path.basename(spath))[0]

            if parent_name.lower() in image_extensions:
                # Parent directory is named after an image extension (e.g., png/)
                # Create parallel json/ directory
                grandparent_dir = _os.path.dirname(parent_dir)
                json_dir = _os.path.join(grandparent_dir, "json")
                json_path = _os.path.join(json_dir, filename_without_ext + ".json")
            else:
                # Save JSON in same directory as image
                json_path = _os.path.splitext(spath)[0] + ".json"

            # Ensure parent directory exists
            _os.makedirs(_os.path.dirname(json_path), exist_ok=True)

            # Save metadata as JSON
            _save(
                collected_metadata,
                json_path,
                verbose=True,
                symlink_from_cwd=False,  # Will handle symlink manually
                dry_run=dry_run,
                no_csv=True,
            )

            # Verify CSV/JSON consistency (data_ref must match columns_actual)
            # Only check for verbose schema - recipe/editable schemas use different data_ref structure
            if csv_path and not dry_run and json_schema == "verbose":
                from scitex.plt.utils._collect_figure_metadata import (
                    assert_csv_json_consistency,
                )
                assert_csv_json_consistency(csv_path, json_path)

            # Create symlink_to for JSON if it was specified for the image
            if symlink_to:
                # Apply same directory transformation for symlink
                symlink_parent_dir = _os.path.dirname(symlink_to)
                symlink_parent_name = _os.path.basename(symlink_parent_dir)
                symlink_filename_without_ext = _os.path.splitext(
                    _os.path.basename(symlink_to)
                )[0]

                if symlink_parent_name.lower() in image_extensions:
                    symlink_grandparent_dir = _os.path.dirname(symlink_parent_dir)
                    json_symlink_to = _os.path.join(
                        symlink_grandparent_dir,
                        "json",
                        symlink_filename_without_ext + ".json",
                    )
                else:
                    json_symlink_to = _os.path.splitext(symlink_to)[0] + ".json"

                _symlink_to(json_path, json_symlink_to, True)

            # Create symlink for JSON manually if needed
            if symlink_from_cwd:
                # Get the relative path from the original specified path
                # This preserves the directory structure for the symlink
                import inspect

                frame_info = inspect.stack()
                # Find the original specified_path from the parent save() call
                for frame in frame_info:
                    if "specified_path" in frame.frame.f_locals:
                        original_path = frame.frame.f_locals["specified_path"]
                        if isinstance(original_path, str):
                            # Apply same directory transformation for symlink
                            orig_parent_dir = _os.path.dirname(original_path)
                            orig_parent_name = _os.path.basename(orig_parent_dir)
                            orig_filename_without_ext = _os.path.splitext(
                                _os.path.basename(original_path)
                            )[0]

                            if orig_parent_name.lower() in image_extensions:
                                orig_grandparent_dir = _os.path.dirname(orig_parent_dir)
                                json_relative = _os.path.join(
                                    orig_grandparent_dir,
                                    "json",
                                    orig_filename_without_ext + ".json",
                                )
                            else:
                                json_relative = original_path.replace(
                                    _os.path.splitext(original_path)[1],
                                    ".json",
                                )

                            json_cwd = _os.path.join(_os.getcwd(), json_relative)
                            _symlink(json_path, json_cwd, True, True)
                            break
                else:
                    # Fallback to basename if we can't find the original path
                    json_cwd = _os.getcwd() + "/" + _os.path.basename(json_path)
                    _symlink(json_path, json_cwd, True, True)

        except AssertionError:
            # Re-raise assertion errors - these are validation failures that should stop execution
            raise
        except Exception as e:
            logger.warning(f"JSON metadata export failed: {e}")


# Dispatch dictionary for O(1) file format lookup
_FILE_HANDLERS = {
    # Canvas directory format (scitex.fig)
    ".canvas": save_canvas,
    # Excel formats
    ".xlsx": save_excel,
    ".xls": save_excel,
    # NumPy formats
    ".npy": save_npy,
    ".npz": save_npz,
    # Pickle formats
    ".pkl": save_pickle,
    ".pickle": save_pickle,
    ".pkl.gz": save_pickle_compressed,
    # Other binary formats
    ".joblib": save_joblib,
    ".pth": save_torch,
    ".pt": save_torch,
    ".mat": save_matlab,
    ".cbm": save_catboost,
    # Text formats
    ".json": save_json,
    ".yaml": save_yaml,
    ".yml": save_yaml,
    ".txt": save_text,
    ".md": save_text,
    ".py": save_text,
    ".css": save_text,
    ".js": save_text,
    ".tex": save_tex,
    # Bibliography
    ".bib": save_bibtex,
    # Data formats
    ".html": save_html,
    ".hdf5": save_hdf5,
    ".h5": save_hdf5,
    ".zarr": save_zarr,
    # Media formats
    ".mp4": save_mp4,
    ".png": _handle_image_with_csv,
    ".jpg": _handle_image_with_csv,
    ".jpeg": _handle_image_with_csv,
    ".gif": _handle_image_with_csv,
    ".tiff": _handle_image_with_csv,
    ".tif": _handle_image_with_csv,
    ".svg": _handle_image_with_csv,
    ".pdf": _handle_image_with_csv,
}

# EOF
