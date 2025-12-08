#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/config/paths.py

"""
Centralized path management for SciTeX.

Provides a single source of truth for all directory paths used across
the SciTeX ecosystem. All paths respect the SCITEX_DIR environment variable.

Usage:
    from scitex.config import ScitexPaths

    paths = ScitexPaths()

    # Method 1: Direct property access (uses default)
    print(paths.logs)           # ~/.scitex/logs
    print(paths.cache)          # ~/.scitex/cache

    # Method 2: resolve() with direct value override (recommended for modules)
    cache_dir = paths.resolve("cache", direct_val=user_provided_path)
    # If user_provided_path is None -> uses default from SCITEX_DIR

    # Thread-safe: pass explicit base_dir
    paths = ScitexPaths(base_dir="/custom/path")
"""

import os
from pathlib import Path
from typing import Optional, Union

from ._PriorityConfig import get_scitex_dir, load_dotenv


class ScitexPaths:
    """Centralized path manager for SciTeX directories.

    All paths are derived from SCITEX_DIR (default: ~/.scitex).
    Priority: direct_val → SCITEX_DIR env → .env file → default

    Directory Structure:
        $SCITEX_DIR/
        ├── browser/              # Browser profiles and data
        │   ├── screenshots/      # Browser debugging screenshots
        │   ├── sessions/         # Shared browser sessions
        │   └── persistent/       # Persistent browser profiles
        ├── cache/                # General cache
        │   └── functions/        # Function cache (joblib)
        ├── capture/              # Screen captures
        ├── impact_factor_cache/  # Impact factor data cache
        ├── logs/                 # Log files
        ├── openathens_cache/     # OpenAthens auth cache
        ├── rng/                  # Random number generator state
        ├── scholar/              # Scholar module data
        │   ├── cache/            # Scholar-specific cache
        │   └── library/          # PDF library
        ├── screenshots/          # General screenshots
        ├── test_monitor/         # Test monitoring screenshots
        └── writer/               # Writer module data
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize ScitexPaths.

        Parameters
        ----------
        base_dir : str, optional
            Explicit base directory. If None, uses SCITEX_DIR env var
            or falls back to ~/.scitex.
        """
        self._base_dir = get_scitex_dir(base_dir)

    @property
    def base(self) -> Path:
        """Base SciTeX directory ($SCITEX_DIR or ~/.scitex)."""
        return self._base_dir

    # ========== Core directories ==========

    @property
    def logs(self) -> Path:
        """Log files directory."""
        return self._base_dir / "logs"

    @property
    def cache(self) -> Path:
        """General cache directory."""
        return self._base_dir / "cache"

    @property
    def capture(self) -> Path:
        """Screen capture directory."""
        return self._base_dir / "capture"

    @property
    def screenshots(self) -> Path:
        """General screenshots directory."""
        return self._base_dir / "screenshots"

    @property
    def rng(self) -> Path:
        """Random number generator state directory."""
        return self._base_dir / "rng"

    # ========== Browser directories ==========

    @property
    def browser(self) -> Path:
        """Browser module base directory."""
        return self._base_dir / "browser"

    @property
    def browser_screenshots(self) -> Path:
        """Browser debugging screenshots."""
        return self.browser / "screenshots"

    @property
    def browser_sessions(self) -> Path:
        """Shared browser sessions."""
        return self.browser / "sessions"

    @property
    def browser_persistent(self) -> Path:
        """Persistent browser profiles."""
        return self.browser / "persistent"

    @property
    def test_monitor(self) -> Path:
        """Test monitoring screenshots directory."""
        return self._base_dir / "test_monitor"

    # ========== Cache directories ==========

    @property
    def function_cache(self) -> Path:
        """Function cache (joblib memory)."""
        return self.cache / "functions"

    @property
    def impact_factor_cache(self) -> Path:
        """Impact factor data cache."""
        return self._base_dir / "impact_factor_cache"

    @property
    def openathens_cache(self) -> Path:
        """OpenAthens authentication cache."""
        return self._base_dir / "openathens_cache"

    # ========== Scholar directories ==========

    @property
    def scholar(self) -> Path:
        """Scholar module base directory."""
        return self._base_dir / "scholar"

    @property
    def scholar_cache(self) -> Path:
        """Scholar-specific cache directory."""
        return self.scholar / "cache"

    @property
    def scholar_library(self) -> Path:
        """Scholar PDF library directory."""
        return self.scholar / "library"

    # ========== Writer directories ==========

    @property
    def writer(self) -> Path:
        """Writer module directory."""
        return self._base_dir / "writer"

    # ========== Resolve method (recommended for modules) ==========

    def resolve(
        self,
        path_name: str,
        direct_val: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Resolve a path with priority: direct_val → default from SCITEX_DIR.

        This is the recommended method for modules that accept optional path
        parameters. It follows the same pattern as PriorityConfig.resolve().

        Parameters
        ----------
        path_name : str
            Name of the path property (e.g., "cache", "logs", "scholar_library")
        direct_val : str or Path, optional
            Direct value (highest precedence). If None, uses default.

        Returns
        -------
        Path
            Resolved path

        Examples
        --------
        >>> paths = ScitexPaths()
        >>> # User didn't provide path -> use default
        >>> cache_dir = paths.resolve("cache", None)
        >>> # User provided custom path -> use it
        >>> cache_dir = paths.resolve("cache", "/custom/cache")

        Usage in modules:
        >>> class MyModule:
        ...     def __init__(self, cache_dir=None):
        ...         self.cache_dir = get_paths().resolve("cache", cache_dir)
        """
        if direct_val is not None:
            return Path(direct_val).expanduser()

        # Get the default path from property
        if hasattr(self, path_name):
            return getattr(self, path_name)

        raise ValueError(
            f"Unknown path name: {path_name}. Available: {list(self.list_all().keys())}"
        )

    # ========== Utility methods ==========

    def ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists, creating if necessary.

        Parameters
        ----------
        path : Path
            Directory path to ensure exists.

        Returns
        -------
        Path
            The same path, guaranteed to exist.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_all(self) -> None:
        """Create all standard directories."""
        dirs = [
            self.logs,
            self.cache,
            self.function_cache,
            self.capture,
            self.screenshots,
            self.rng,
            self.browser,
            self.browser_screenshots,
            self.browser_sessions,
            self.browser_persistent,
            self.test_monitor,
            self.impact_factor_cache,
            self.openathens_cache,
            self.scholar,
            self.scholar_cache,
            self.scholar_library,
            self.writer,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def list_all(self) -> dict:
        """List all configured paths.

        Returns
        -------
        dict
            Dictionary of path names to Path objects.
        """
        return {
            "base": self.base,
            "logs": self.logs,
            "cache": self.cache,
            "function_cache": self.function_cache,
            "capture": self.capture,
            "screenshots": self.screenshots,
            "rng": self.rng,
            "browser": self.browser,
            "browser_screenshots": self.browser_screenshots,
            "browser_sessions": self.browser_sessions,
            "browser_persistent": self.browser_persistent,
            "test_monitor": self.test_monitor,
            "impact_factor_cache": self.impact_factor_cache,
            "openathens_cache": self.openathens_cache,
            "scholar": self.scholar,
            "scholar_cache": self.scholar_cache,
            "scholar_library": self.scholar_library,
            "writer": self.writer,
        }

    def __repr__(self) -> str:
        return f"ScitexPaths(base='{self._base_dir}')"


# Singleton instance for convenience (uses default SCITEX_DIR)
_default_paths: Optional[ScitexPaths] = None


def get_paths(base_dir: Optional[str] = None) -> ScitexPaths:
    """Get ScitexPaths instance.

    Parameters
    ----------
    base_dir : str, optional
        Explicit base directory. If None, returns cached default instance.

    Returns
    -------
    ScitexPaths
        Path manager instance.
    """
    global _default_paths

    if base_dir is not None:
        return ScitexPaths(base_dir)

    if _default_paths is None:
        _default_paths = ScitexPaths()

    return _default_paths


# EOF
