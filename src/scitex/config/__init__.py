#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 (ywatanabe)"
# File: ./src/scitex/config/__init__.py

"""
SciTeX configuration module.

Provides two configuration patterns (both use same priority order):

1. **ScitexConfig** (YAML-based, recommended):
   - Loads configuration from YAML files
   - YAML supports env var substitution: ${VAR:-default}

2. **PriorityConfig** (dict-based, for programmatic use):
   - Uses a Python dictionary for configuration

**Priority Order** (same for both):
   direct → config (YAML/dict) → env → default

Usage:
    from scitex.config import ScitexConfig, ScitexPaths, get_config, get_paths

    # YAML-based configuration (Scholar pattern)
    config = get_config()
    log_level = config.resolve("logging.level", default="INFO")

    # Centralized path manager
    paths = get_paths()
    print(paths.logs)      # ~/.scitex/logs
    print(paths.cache)     # ~/.scitex/cache

    # Use resolve() pattern in modules
    cache_dir = paths.resolve("cache", user_provided_path)
"""

from ._PriorityConfig import PriorityConfig, get_scitex_dir, load_dotenv
from ._paths import ScitexPaths, get_paths
from ._ScitexConfig import ScitexConfig, get_config, load_yaml

__all__ = [
    # YAML-based config (Scholar pattern)
    "ScitexConfig",
    "get_config",
    "load_yaml",
    # Path management
    "ScitexPaths",
    "get_paths",
    # Legacy/utility
    "PriorityConfig",
    "get_scitex_dir",
    "load_dotenv",
]


# EOF
