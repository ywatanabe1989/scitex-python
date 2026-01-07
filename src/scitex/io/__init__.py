#!/usr/bin/env python3
# Timestamp: "2025-12-16 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/__init__.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""Scitex IO module with lazy imports to avoid circular dependencies.

Bundle I/O is handled via scitex.io.bundle:
    from scitex.io.bundle import Bundle
    bundle = Bundle("Figure1.figure")
    bundle.save()
"""

# Import commonly used functions directly
# Bundle I/O - import the bundle submodule
from . import bundle
from ._cache import cache
from ._flush import flush
from ._glob import glob, parse_glob
from ._load import load
from ._load_configs import load_configs
from ._reload import reload
from ._save import save

# Optional: HDF5 explorer (requires h5py)
try:
    from ._load_modules._H5Explorer import H5Explorer, explore_h5, has_h5_key
except ImportError:
    H5Explorer = None
    explore_h5 = None
    has_h5_key = None

# Optional: Zarr explorer (requires zarr)
try:
    from ._load_modules._ZarrExplorer import ZarrExplorer, explore_zarr, has_zarr_key
except ImportError:
    ZarrExplorer = None
    explore_zarr = None
    has_zarr_key = None

# Import load cache control functions
from ._load_cache import clear_cache as clear_load_cache
from ._load_cache import configure_cache, get_cache_info

# Import load cache control functions

# Import save module functions
try:
    from ._save_modules import (
        save_image,
        save_listed_dfs_as_csv,
        save_listed_scalars_as_csv,
        save_mp4,
        save_optuna_study_as_csv_and_pngs,
        save_text,
    )
except ImportError:
    # Fallback for missing functions
    save_image = None
    save_text = None
    save_mp4 = None
    save_listed_dfs_as_csv = None
    save_listed_scalars_as_csv = None
    save_optuna_study_as_csv_and_pngs = None

# Optional imports that might fail
try:
    from ._path import path
except ImportError:
    path = None

try:
    from ._mv_to_tmp import mv_to_tmp
except ImportError:
    mv_to_tmp = None

try:
    from ._json2md import json2md
except ImportError:
    json2md = None

# Import utils
try:
    from .utils import migrate_h5_to_zarr, migrate_h5_to_zarr_batch
except ImportError:
    migrate_h5_to_zarr = None
    migrate_h5_to_zarr_batch = None

# Import metadata functions
try:
    from ._metadata import embed_metadata, has_metadata, read_metadata
except ImportError:
    read_metadata = None
    embed_metadata = None
    has_metadata = None

__all__ = [
    # Primary I/O
    "save",
    "load",
    # Bundle submodule
    "bundle",
    # Config loading
    "load_configs",
    # File utilities
    "glob",
    "reload",
    "flush",
    "cache",
]

# EOF
