#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/config/ScitexConfig.py

"""
YAML-based configuration for SciTeX with environment variable substitution.

Similar to ScholarConfig, provides:
- YAML configuration loading
- Environment variable substitution (${VAR:-default} syntax)
- Cascade resolution (direct → config → env → default)

Usage:
    from scitex.config import ScitexConfig

    # Load default configuration
    config = ScitexConfig()

    # Load custom configuration
    config = ScitexConfig(config_path="/path/to/config.yaml")

    # Resolve values with precedence
    log_level = config.resolve("logging.level", default="INFO")
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from ._PriorityConfig import PriorityConfig, load_dotenv


def load_yaml(path: Path) -> dict:
    """Load YAML file with environment variable substitution.

    Supports ${VAR:-default} syntax for environment variable expansion.

    Parameters
    ----------
    path : Path
        Path to YAML file

    Returns
    -------
    dict
        Parsed YAML with environment variables substituted
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML required for YAML config. Install with: pip install pyyaml"
        )

    try:
        with open(path) as f:
            content = f.read()

        def env_replacer(match):
            """Replace ${VAR:-default} with environment variable or default."""
            env_expr = match.group(1)
            if ":-" in env_expr:
                var_name, default_value = env_expr.split(":-", 1)
                value = os.getenv(var_name, default_value.strip('"'))
            else:
                value = os.getenv(env_expr)

            # Handle special values
            if value in ["true", "false"]:
                return value
            elif value == "null":
                return "null"
            elif value and not (value.startswith('"') and value.endswith('"')):
                return f'"{value}"'
            else:
                return value or "null"

        content = re.sub(r"\$\{([^}]+)\}", env_replacer, content)
        return yaml.safe_load(content)
    except Exception as e:
        raise ValueError(f"Failed to load YAML config from {path}: {e}")


class ScitexConfig:
    """YAML-based configuration manager for SciTeX.

    Loads configuration from YAML files with environment variable substitution.
    Values can be resolved with priority: direct → config → env → default.

    Examples
    --------
    >>> from scitex.config import ScitexConfig
    >>> config = ScitexConfig()
    >>> config.resolve("logging.level", default="INFO")
    'INFO'
    >>> config.get("debug.enabled")
    False
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        env_prefix: str = "SCITEX_",
    ):
        """Initialize ScitexConfig.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to custom YAML config file. If None, uses default.yaml.
        env_prefix : str
            Prefix for environment variables (default: "SCITEX_")
        """
        # Load .env file first
        load_dotenv()

        # Load YAML configuration
        if config_path and Path(config_path).exists():
            self._config_data = load_yaml(Path(config_path))
            self._config_path = Path(config_path)
        else:
            default_path = Path(__file__).parent / "default.yaml"
            if default_path.exists():
                self._config_data = load_yaml(default_path)
            else:
                self._config_data = {}
            self._config_path = default_path

        # Flatten nested config for easy access
        self._flat_config = self._flatten_dict(self._config_data)

        # Initialize PriorityConfig for resolution
        self._priority_config = PriorityConfig(
            config_dict=self._flat_config,
            env_prefix=env_prefix,
        )

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dictionary with dot notation keys.

        Parameters
        ----------
        d : dict
            Dictionary to flatten
        parent_key : str
            Parent key for recursion
        sep : str
            Separator for nested keys

        Returns
        -------
        dict
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from config directly (no precedence resolution).

        Supports dot notation for nested keys.

        Parameters
        ----------
        key : str
            Configuration key (e.g., "logging.level" or "debug.enabled")
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value
        """
        return self._flat_config.get(key, default)

    def resolve(
        self,
        key: str,
        direct_val: Any = None,
        default: Any = None,
        type: Type = str,
    ) -> Any:
        """Resolve value with precedence: direct → config → env → default.

        This follows the Scholar module's CascadeConfig pattern where
        YAML config takes higher priority than environment variables.

        Parameters
        ----------
        key : str
            Configuration key (e.g., "logging.level")
        direct_val : Any
            Direct value (highest precedence)
        default : Any
            Default value (lowest precedence)
        type : Type
            Type conversion (str, int, float, bool, list)

        Returns
        -------
        Any
            Resolved value
        """
        # Priority: direct → config → env → default
        # (Same as Scholar's CascadeConfig pattern)
        if direct_val is not None:
            return direct_val

        # Config (YAML) takes priority over env
        config_val = self._flat_config.get(key)
        if config_val is not None:
            return config_val

        # Then check environment variable
        normalized_key = key.replace(".", "_")
        env_key = f"SCITEX_{normalized_key.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            return self._convert_type(env_val, type)

        return default

    def _convert_type(self, value: str, type: Type) -> Any:
        """Convert string value to specified type."""
        if type == int:
            return int(value)
        elif type == float:
            return float(value)
        elif type == bool:
            return value.lower() in ("true", "1", "yes")
        elif type == list:
            return value.split(",")
        return value

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested value from original config structure.

        Parameters
        ----------
        *keys : str
            Keys to traverse (e.g., "browser", "screenshots_dir")
        default : Any
            Default value if not found

        Returns
        -------
        Any
            Nested value
        """
        current = self._config_data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @property
    def config_path(self) -> Path:
        """Get the path to the loaded config file."""
        return self._config_path

    @property
    def raw(self) -> dict:
        """Get raw configuration data (original nested structure)."""
        return self._config_data

    @property
    def flat(self) -> dict:
        """Get flattened configuration data."""
        return self._flat_config

    def print(self) -> None:
        """Print configuration resolution log."""
        self._priority_config.print_resolutions()

    def __repr__(self) -> str:
        return f"ScitexConfig(path='{self._config_path}')"


# Module-level convenience functions

_default_config: Optional[ScitexConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> ScitexConfig:
    """Get ScitexConfig instance.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to custom config. If None, returns cached default instance.

    Returns
    -------
    ScitexConfig
        Configuration instance
    """
    global _default_config

    if config_path is not None:
        return ScitexConfig(config_path)

    if _default_config is None:
        _default_config = ScitexConfig()

    return _default_config


# EOF
