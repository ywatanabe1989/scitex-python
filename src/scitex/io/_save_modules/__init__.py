#!/usr/bin/env python3
# Timestamp: "2025-06-12 13:05:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/scitex_repo/src/scitex/io/_save_modules/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/io/_save_modules/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Save modules for scitex.io.save functionality

This package contains format-specific save handlers for various file types.
Each module provides a save_<format> function that handles saving objects
to that specific format.
"""

# Import save functions from individual modules
from ._bibtex import save_bibtex
from ._catboost import _save_catboost as save_catboost
from ._csv import _save_csv as save_csv
from ._excel import save_excel
from ._figure_utils import get_figure_with_data

# Optional: image (requires plotly)
try:
    from ._image import save_image
except ImportError:
    save_image = None

# Optional: HTML (requires plotly)
try:
    from ._html import save_html
except ImportError:
    save_html = None

# Optional: HDF5 (requires h5py)
try:
    from ._hdf5 import _save_hdf5 as save_hdf5
except ImportError:
    save_hdf5 = None

# Import extracted utilities
from ._image_csv import handle_image_with_csv
from ._json import _save_json as save_json
from ._legends import save_separate_legends

# Import additional save utilities
from ._listed_dfs_as_csv import _save_listed_dfs_as_csv as save_listed_dfs_as_csv
from ._listed_scalars_as_csv import (
    _save_listed_scalars_as_csv as save_listed_scalars_as_csv,
)
from ._mp4 import _mk_mp4 as save_mp4
from ._numpy import _save_npy as save_npy
from ._numpy import _save_npz as save_npz
from ._optuna_study_as_csv_and_pngs import save_optuna_study_as_csv_and_pngs
from ._pickle import _save_pickle as save_pickle
from ._pickle import _save_pickle_gz as save_pickle_compressed
from ._plot_bundle import save_plot_bundle
from ._plot_scitex import save_plot_as_scitex
from ._stx_bundle import save_stx_bundle
from ._symlink import symlink, symlink_to
from ._tex import _save_tex as save_tex
from ._text import _save_text as save_text
from ._yaml import _save_yaml as save_yaml

# Optional: joblib (requires joblib)
try:
    from ._joblib import _save_joblib as save_joblib
except ImportError:
    save_joblib = None

# Optional: matlab (requires scipy)
try:
    from ._matlab import _save_matlab as save_matlab
except ImportError:
    save_matlab = None

# Optional: torch (requires torch)
try:
    from ._torch import _save_torch as save_torch
except ImportError:
    save_torch = None

# Optional: zarr (requires zarr)
try:
    from ._zarr import _save_zarr as save_zarr
except ImportError:
    save_zarr = None

# Define what gets imported with "from scitex.io._save_modules import *"
__all__ = [
    # Core save functions
    "save_csv",
    "save_excel",
    "save_npy",
    "save_npz",
    "save_pickle",
    "save_pickle_compressed",
    "save_joblib",
    "save_torch",
    "save_json",
    "save_yaml",
    "save_hdf5",
    "save_matlab",
    "save_catboost",
    "save_text",
    "save_tex",
    "save_html",
    "save_image",
    "save_mp4",
    "save_zarr",
    "save_bibtex",
    "save_listed_dfs_as_csv",
    "save_listed_scalars_as_csv",
    "save_optuna_study_as_csv_and_pngs",
    # Bundle save functions
    "save_stx_bundle",
    "save_plot_bundle",
    "save_plot_as_scitex",
    "save_separate_legends",
    "handle_image_with_csv",
    # Utilities
    "get_figure_with_data",
    "symlink",
    "symlink_to",
]

# EOF
