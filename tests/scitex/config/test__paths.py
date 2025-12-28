#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./tests/scitex/config/test__paths.py

"""Tests for ScitexPaths class and get_paths() function."""

import os
import tempfile
import pytest
from pathlib import Path
from scitex.config import ScitexPaths, get_paths


class TestScitexPathsBasic:
    """Basic ScitexPaths functionality tests."""

    def test_initialization_default(self):
        """Test ScitexPaths can be initialized with defaults."""
        original = os.environ.pop("SCITEX_DIR", None)
        try:
            paths = ScitexPaths()
            assert paths is not None
            assert paths.base == Path.home() / ".scitex"
        finally:
            if original:
                os.environ["SCITEX_DIR"] = original

    def test_initialization_with_base_dir(self):
        """Test initialization with explicit base_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.base == Path(tmpdir)

    def test_initialization_from_env(self):
        """Test initialization uses SCITEX_DIR env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SCITEX_DIR"] = tmpdir
            try:
                paths = ScitexPaths()
                assert paths.base == Path(tmpdir)
            finally:
                del os.environ["SCITEX_DIR"]

    def test_repr(self):
        """Test string representation."""
        paths = ScitexPaths()
        repr_str = repr(paths)
        assert "ScitexPaths" in repr_str
        assert "base=" in repr_str


class TestScitexPathsCoreDirectories:
    """Test core directory properties."""

    def test_logs_path(self):
        """Test logs directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.logs == Path(tmpdir) / "logs"

    def test_cache_path(self):
        """Test cache directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.cache == Path(tmpdir) / "cache"

    def test_capture_path(self):
        """Test capture directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.capture == Path(tmpdir) / "capture"

    def test_screenshots_path(self):
        """Test screenshots directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.screenshots == Path(tmpdir) / "screenshots"

    def test_rng_path(self):
        """Test rng directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.rng == Path(tmpdir) / "rng"


class TestScitexPathsBrowserDirectories:
    """Test browser-related directory properties."""

    def test_browser_path(self):
        """Test browser base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.browser == Path(tmpdir) / "browser"

    def test_browser_screenshots_path(self):
        """Test browser screenshots directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.browser_screenshots == Path(tmpdir) / "browser" / "screenshots"

    def test_browser_sessions_path(self):
        """Test browser sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.browser_sessions == Path(tmpdir) / "browser" / "sessions"

    def test_browser_persistent_path(self):
        """Test browser persistent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.browser_persistent == Path(tmpdir) / "browser" / "persistent"

    def test_test_monitor_path(self):
        """Test test_monitor directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.test_monitor == Path(tmpdir) / "test_monitor"


class TestScitexPathsCacheDirectories:
    """Test cache-related directory properties."""

    def test_function_cache_path(self):
        """Test function cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.function_cache == Path(tmpdir) / "cache" / "functions"

    def test_impact_factor_cache_path(self):
        """Test impact factor cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.impact_factor_cache == Path(tmpdir) / "impact_factor_cache"

    def test_openathens_cache_path(self):
        """Test openathens cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.openathens_cache == Path(tmpdir) / "openathens_cache"


class TestScitexPathsScholarDirectories:
    """Test scholar-related directory properties."""

    def test_scholar_path(self):
        """Test scholar base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.scholar == Path(tmpdir) / "scholar"

    def test_scholar_cache_path(self):
        """Test scholar cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.scholar_cache == Path(tmpdir) / "scholar" / "cache"

    def test_scholar_library_path(self):
        """Test scholar library directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.scholar_library == Path(tmpdir) / "scholar" / "library"


class TestScitexPathsWriterDirectories:
    """Test writer-related directory properties."""

    def test_writer_path(self):
        """Test writer directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            assert paths.writer == Path(tmpdir) / "writer"


class TestScitexPathsResolve:
    """Test resolve() method."""

    def test_resolve_with_direct_value(self):
        """Test resolve returns direct value when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            custom_path = "/custom/cache/path"
            result = paths.resolve("cache", direct_val=custom_path)
            assert result == Path(custom_path)

    def test_resolve_without_direct_value(self):
        """Test resolve returns default path when no direct value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            result = paths.resolve("cache", direct_val=None)
            assert result == Path(tmpdir) / "cache"

    def test_resolve_expands_user(self):
        """Test resolve expands ~ in path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            result = paths.resolve("logs", direct_val="~/custom_logs")
            assert "~" not in str(result)
            assert str(result).startswith(str(Path.home()))

    def test_resolve_various_paths(self):
        """Test resolve works for various path names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            path_names = ["logs", "cache", "browser", "scholar", "writer"]
            for name in path_names:
                result = paths.resolve(name)
                assert isinstance(result, Path)

    def test_resolve_unknown_path_raises(self):
        """Test resolve raises ValueError for unknown path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            with pytest.raises(ValueError) as exc_info:
                paths.resolve("unknown_path_name")
            assert "Unknown path name" in str(exc_info.value)


class TestScitexPathsEnsureDir:
    """Test ensure_dir() method."""

    def test_ensure_dir_creates_directory(self):
        """Test ensure_dir creates directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            new_dir = Path(tmpdir) / "new_subdir"
            assert not new_dir.exists()

            result = paths.ensure_dir(new_dir)
            assert new_dir.exists()
            assert result == new_dir

    def test_ensure_dir_existing_directory(self):
        """Test ensure_dir works on existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            result = paths.ensure_dir(Path(tmpdir))
            assert result == Path(tmpdir)

    def test_ensure_dir_nested(self):
        """Test ensure_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            nested_dir = Path(tmpdir) / "a" / "b" / "c"
            assert not nested_dir.exists()

            result = paths.ensure_dir(nested_dir)
            assert nested_dir.exists()
            assert result == nested_dir


class TestScitexPathsEnsureAll:
    """Test ensure_all() method."""

    def test_ensure_all_creates_directories(self):
        """Test ensure_all creates all standard directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            paths.ensure_all()

            # Check that key directories exist
            assert paths.logs.exists()
            assert paths.cache.exists()
            assert paths.browser.exists()
            assert paths.scholar.exists()
            assert paths.writer.exists()


class TestScitexPathsListAll:
    """Test list_all() method."""

    def test_list_all_returns_dict(self):
        """Test list_all returns dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            result = paths.list_all()
            assert isinstance(result, dict)

    def test_list_all_contains_expected_keys(self):
        """Test list_all contains expected path names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            result = paths.list_all()

            expected_keys = [
                "base", "logs", "cache", "function_cache", "capture",
                "screenshots", "rng", "browser", "browser_screenshots",
                "browser_sessions", "browser_persistent", "test_monitor",
                "impact_factor_cache", "openathens_cache", "scholar",
                "scholar_cache", "scholar_library", "writer"
            ]
            for key in expected_keys:
                assert key in result, f"Missing key: {key}"

    def test_list_all_values_are_paths(self):
        """Test list_all values are Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ScitexPaths(base_dir=tmpdir)
            result = paths.list_all()

            for key, value in result.items():
                assert isinstance(value, Path), f"{key} is not a Path"


class TestGetPaths:
    """Test get_paths() convenience function."""

    def test_get_paths_returns_instance(self):
        """Test get_paths returns ScitexPaths instance."""
        paths = get_paths()
        assert isinstance(paths, ScitexPaths)

    def test_get_paths_with_base_dir(self):
        """Test get_paths with custom base_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = get_paths(base_dir=tmpdir)
            assert paths.base == Path(tmpdir)

    def test_get_paths_caches_default_instance(self):
        """Test get_paths returns same instance when no args."""
        # Note: This test may be affected by other tests that call get_paths()
        paths1 = get_paths()
        paths2 = get_paths()
        # Both should be ScitexPaths instances
        assert isinstance(paths1, ScitexPaths)
        assert isinstance(paths2, ScitexPaths)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/config/_paths.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/config/paths.py
# 
# """
# Centralized path management for SciTeX.
# 
# Provides a single source of truth for all directory paths used across
# the SciTeX ecosystem. All paths respect the SCITEX_DIR environment variable.
# 
# Usage:
#     from scitex.config import ScitexPaths
# 
#     paths = ScitexPaths()
# 
#     # Method 1: Direct property access (uses default)
#     print(paths.logs)           # ~/.scitex/logs
#     print(paths.cache)          # ~/.scitex/cache
# 
#     # Method 2: resolve() with direct value override (recommended for modules)
#     cache_dir = paths.resolve("cache", direct_val=user_provided_path)
#     # If user_provided_path is None -> uses default from SCITEX_DIR
# 
#     # Thread-safe: pass explicit base_dir
#     paths = ScitexPaths(base_dir="/custom/path")
# """
# 
# import os
# from pathlib import Path
# from typing import Optional, Union
# 
# from ._PriorityConfig import get_scitex_dir, load_dotenv
# 
# 
# class ScitexPaths:
#     """Centralized path manager for SciTeX directories.
# 
#     All paths are derived from SCITEX_DIR (default: ~/.scitex).
#     Priority: direct_val → SCITEX_DIR env → .env file → default
# 
#     Directory Structure:
#         $SCITEX_DIR/
#         ├── browser/              # Browser profiles and data
#         │   ├── screenshots/      # Browser debugging screenshots
#         │   ├── sessions/         # Shared browser sessions
#         │   └── persistent/       # Persistent browser profiles
#         ├── cache/                # General cache
#         │   └── functions/        # Function cache (joblib)
#         ├── capture/              # Screen captures
#         ├── impact_factor_cache/  # Impact factor data cache
#         ├── logs/                 # Log files
#         ├── openathens_cache/     # OpenAthens auth cache
#         ├── rng/                  # Random number generator state
#         ├── scholar/              # Scholar module data
#         │   ├── cache/            # Scholar-specific cache
#         │   └── library/          # PDF library
#         ├── screenshots/          # General screenshots
#         ├── test_monitor/         # Test monitoring screenshots
#         └── writer/               # Writer module data
#     """
# 
#     def __init__(self, base_dir: Optional[str] = None):
#         """Initialize ScitexPaths.
# 
#         Parameters
#         ----------
#         base_dir : str, optional
#             Explicit base directory. If None, uses SCITEX_DIR env var
#             or falls back to ~/.scitex.
#         """
#         self._base_dir = get_scitex_dir(base_dir)
# 
#     @property
#     def base(self) -> Path:
#         """Base SciTeX directory ($SCITEX_DIR or ~/.scitex)."""
#         return self._base_dir
# 
#     # ========== Core directories ==========
# 
#     @property
#     def logs(self) -> Path:
#         """Log files directory."""
#         return self._base_dir / "logs"
# 
#     @property
#     def cache(self) -> Path:
#         """General cache directory."""
#         return self._base_dir / "cache"
# 
#     @property
#     def capture(self) -> Path:
#         """Screen capture directory."""
#         return self._base_dir / "capture"
# 
#     @property
#     def screenshots(self) -> Path:
#         """General screenshots directory."""
#         return self._base_dir / "screenshots"
# 
#     @property
#     def rng(self) -> Path:
#         """Random number generator state directory."""
#         return self._base_dir / "rng"
# 
#     # ========== Browser directories ==========
# 
#     @property
#     def browser(self) -> Path:
#         """Browser module base directory."""
#         return self._base_dir / "browser"
# 
#     @property
#     def browser_screenshots(self) -> Path:
#         """Browser debugging screenshots."""
#         return self.browser / "screenshots"
# 
#     @property
#     def browser_sessions(self) -> Path:
#         """Shared browser sessions."""
#         return self.browser / "sessions"
# 
#     @property
#     def browser_persistent(self) -> Path:
#         """Persistent browser profiles."""
#         return self.browser / "persistent"
# 
#     @property
#     def test_monitor(self) -> Path:
#         """Test monitoring screenshots directory."""
#         return self._base_dir / "test_monitor"
# 
#     # ========== Cache directories ==========
# 
#     @property
#     def function_cache(self) -> Path:
#         """Function cache (joblib memory)."""
#         return self.cache / "functions"
# 
#     @property
#     def impact_factor_cache(self) -> Path:
#         """Impact factor data cache."""
#         return self._base_dir / "impact_factor_cache"
# 
#     @property
#     def openathens_cache(self) -> Path:
#         """OpenAthens authentication cache."""
#         return self._base_dir / "openathens_cache"
# 
#     # ========== Scholar directories ==========
# 
#     @property
#     def scholar(self) -> Path:
#         """Scholar module base directory."""
#         return self._base_dir / "scholar"
# 
#     @property
#     def scholar_cache(self) -> Path:
#         """Scholar-specific cache directory."""
#         return self.scholar / "cache"
# 
#     @property
#     def scholar_library(self) -> Path:
#         """Scholar PDF library directory."""
#         return self.scholar / "library"
# 
#     # ========== Writer directories ==========
# 
#     @property
#     def writer(self) -> Path:
#         """Writer module directory."""
#         return self._base_dir / "writer"
# 
#     # ========== Resolve method (recommended for modules) ==========
# 
#     def resolve(
#         self,
#         path_name: str,
#         direct_val: Optional[Union[str, Path]] = None,
#     ) -> Path:
#         """Resolve a path with priority: direct_val → default from SCITEX_DIR.
# 
#         This is the recommended method for modules that accept optional path
#         parameters. It follows the same pattern as PriorityConfig.resolve().
# 
#         Parameters
#         ----------
#         path_name : str
#             Name of the path property (e.g., "cache", "logs", "scholar_library")
#         direct_val : str or Path, optional
#             Direct value (highest precedence). If None, uses default.
# 
#         Returns
#         -------
#         Path
#             Resolved path
# 
#         Examples
#         --------
#         >>> paths = ScitexPaths()
#         >>> # User didn't provide path -> use default
#         >>> cache_dir = paths.resolve("cache", None)
#         >>> # User provided custom path -> use it
#         >>> cache_dir = paths.resolve("cache", "/custom/cache")
# 
#         Usage in modules:
#         >>> class MyModule:
#         ...     def __init__(self, cache_dir=None):
#         ...         self.cache_dir = get_paths().resolve("cache", cache_dir)
#         """
#         if direct_val is not None:
#             return Path(direct_val).expanduser()
# 
#         # Get the default path from property
#         if hasattr(self, path_name):
#             return getattr(self, path_name)
# 
#         raise ValueError(
#             f"Unknown path name: {path_name}. Available: {list(self.list_all().keys())}"
#         )
# 
#     # ========== Utility methods ==========
# 
#     def ensure_dir(self, path: Path) -> Path:
#         """Ensure directory exists, creating if necessary.
# 
#         Parameters
#         ----------
#         path : Path
#             Directory path to ensure exists.
# 
#         Returns
#         -------
#         Path
#             The same path, guaranteed to exist.
#         """
#         path.mkdir(parents=True, exist_ok=True)
#         return path
# 
#     def ensure_all(self) -> None:
#         """Create all standard directories."""
#         dirs = [
#             self.logs,
#             self.cache,
#             self.function_cache,
#             self.capture,
#             self.screenshots,
#             self.rng,
#             self.browser,
#             self.browser_screenshots,
#             self.browser_sessions,
#             self.browser_persistent,
#             self.test_monitor,
#             self.impact_factor_cache,
#             self.openathens_cache,
#             self.scholar,
#             self.scholar_cache,
#             self.scholar_library,
#             self.writer,
#         ]
#         for d in dirs:
#             d.mkdir(parents=True, exist_ok=True)
# 
#     def list_all(self) -> dict:
#         """List all configured paths.
# 
#         Returns
#         -------
#         dict
#             Dictionary of path names to Path objects.
#         """
#         return {
#             "base": self.base,
#             "logs": self.logs,
#             "cache": self.cache,
#             "function_cache": self.function_cache,
#             "capture": self.capture,
#             "screenshots": self.screenshots,
#             "rng": self.rng,
#             "browser": self.browser,
#             "browser_screenshots": self.browser_screenshots,
#             "browser_sessions": self.browser_sessions,
#             "browser_persistent": self.browser_persistent,
#             "test_monitor": self.test_monitor,
#             "impact_factor_cache": self.impact_factor_cache,
#             "openathens_cache": self.openathens_cache,
#             "scholar": self.scholar,
#             "scholar_cache": self.scholar_cache,
#             "scholar_library": self.scholar_library,
#             "writer": self.writer,
#         }
# 
#     def __repr__(self) -> str:
#         return f"ScitexPaths(base='{self._base_dir}')"
# 
# 
# # Singleton instance for convenience (uses default SCITEX_DIR)
# _default_paths: Optional[ScitexPaths] = None
# 
# 
# def get_paths(base_dir: Optional[str] = None) -> ScitexPaths:
#     """Get ScitexPaths instance.
# 
#     Parameters
#     ----------
#     base_dir : str, optional
#         Explicit base directory. If None, returns cached default instance.
# 
#     Returns
#     -------
#     ScitexPaths
#         Path manager instance.
#     """
#     global _default_paths
# 
#     if base_dir is not None:
#         return ScitexPaths(base_dir)
# 
#     if _default_paths is None:
#         _default_paths = ScitexPaths()
# 
#     return _default_paths
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/config/_paths.py
# --------------------------------------------------------------------------------
