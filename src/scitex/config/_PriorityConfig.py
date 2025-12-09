#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/config/PriorityConfig.py


"""
Priority-based configuration resolver.

Provides clean precedence hierarchy: direct → config_dict → env → default

Based on priority-config by ywatanabe (https://github.com/ywatanabe1989/priority-config)
Incorporated into scitex for self-contained configuration management.

Note: config_dict values (from YAML or passed dict) take priority over
environment variables. This follows the Scholar module's CascadeConfig pattern.
"""

import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional, Type, Any


def load_dotenv(dotenv_path: Optional[str] = None) -> bool:
    """Load environment variables from .env file.

    Searches for .env file in the following order:
    1. Explicit dotenv_path if provided
    2. Current working directory
    3. User home directory

    Parameters
    ----------
    dotenv_path : str, optional
        Path to .env file. If None, searches default locations.

    Returns
    -------
    bool
        True if .env file was found and loaded, False otherwise.
    """
    paths_to_try = []

    if dotenv_path:
        paths_to_try.append(Path(dotenv_path))
    else:
        # Default search paths
        paths_to_try.extend(
            [
                Path.cwd() / ".env",
                Path.home() / ".env",
            ]
        )

    for path in paths_to_try:
        if path.exists() and path.is_file():
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if not line or line.startswith("#"):
                            continue
                        # Handle export prefix
                        if line.startswith("export "):
                            line = line[7:]
                        # Parse key=value
                        if "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or (
                                value.startswith("'") and value.endswith("'")
                            ):
                                value = value[1:-1]
                            # Only set if not already in environment (env takes precedence)
                            if key not in os.environ:
                                os.environ[key] = value
                return True
            except Exception:
                continue
    return False


def get_scitex_dir(direct_val: Optional[str] = None) -> Path:
    """Get SCITEX_DIR with priority: direct → env → default.

    This is a convenience function for the most common use case.

    Parameters
    ----------
    direct_val : str, optional
        Direct value (highest precedence)

    Returns
    -------
    Path
        Resolved SCITEX_DIR path
    """
    # Try to load .env first (won't override existing env vars)
    load_dotenv()

    if direct_val is not None:
        return Path(direct_val).expanduser()

    env_val = os.getenv("SCITEX_DIR")
    if env_val:
        return Path(env_val).expanduser()

    return Path.home() / ".scitex"


class PriorityConfig:
    """Universal config resolver with precedence: direct → config_dict → env → default

    Config dict (from YAML or passed dict) takes priority over env variables.
    This follows the Scholar module's CascadeConfig pattern.

    Examples
    --------
    >>> from scitex.config import PriorityConfig
    >>> config = PriorityConfig(config_dict={"port": 3000}, env_prefix="SCITEX_")
    >>> port = config.resolve("port", None, default=8000, type=int)
    3000  # from config_dict (highest after direct)
    >>> # With env: SCITEX_PORT=5000 python script.py
    >>> port = config.resolve("port", None, default=8000, type=int)
    3000  # config_dict takes precedence over env
    >>> port = config.resolve("port", 9000, default=8000, type=int)
    9000  # direct value takes highest precedence
    """

    SENSITIVE_EXPRESSIONS = [
        "API",
        "PASSWORD",
        "SECRET",
        "TOKEN",
        "KEY",
        "PASS",
        "AUTH",
        "CREDENTIAL",
        "PRIVATE",
        "CERT",
    ]

    def __init__(
        self,
        config_dict: Optional[Dict[str, Any]] = None,
        env_prefix: str = "",
        auto_uppercase: bool = True,
    ):
        """Initialize PriorityConfig.

        Parameters
        ----------
        config_dict : dict, optional
            Dictionary with configuration values
        env_prefix : str
            Prefix for environment variables (e.g., "SCITEX_")
        auto_uppercase : bool
            Whether to uppercase keys for env lookup
        """
        self.config_dict = config_dict or {}
        self.env_prefix = env_prefix
        self.auto_uppercase = auto_uppercase
        self.resolution_log: List[Dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"PriorityConfig(prefix='{self.env_prefix}', configs={len(self.config_dict)})"

    def get(self, key: str) -> Any:
        """Get value from config dict only."""
        return self.config_dict.get(key)

    def resolve(
        self,
        key: str,
        direct_val: Any = None,
        default: Any = None,
        type: Type = str,
        mask: Optional[bool] = None,
    ) -> Any:
        """Get value with precedence hierarchy.

        Precedence: direct → config_dict → env → default

        This follows the Scholar module's CascadeConfig pattern where
        config dict takes higher priority than environment variables.

        Parameters
        ----------
        key : str
            Configuration key to resolve
        direct_val : Any, optional
            Direct value (highest precedence)
        default : Any, optional
            Default value if not found elsewhere
        type : Type
            Type conversion (str, int, float, bool, list)
        mask : bool, optional
            Override automatic masking of sensitive values

        Returns
        -------
        Any
            Resolved configuration value
        """
        source = None
        final_value = None

        # Replace dots with underscores for env key (e.g., axes.width_mm -> AXES_WIDTH_MM)
        normalized_key = key.replace(".", "_")
        env_key = f"{self.env_prefix}{normalized_key.upper() if self.auto_uppercase else normalized_key}"
        env_val = os.getenv(env_key)

        # Priority: direct → config_dict → env → default
        if direct_val is not None:
            source = "direct"
            final_value = direct_val
        elif key in self.config_dict:
            source = "config_dict"
            final_value = self.config_dict[key]
        elif env_val:
            source = f"env:{env_key}"
            final_value = self._convert_type(env_val, type)
        else:
            source = "default"
            final_value = default

        if mask is False:
            should_mask = False
        else:
            should_mask = self._is_sensitive(key)

        display_value = self._mask_value(final_value) if should_mask else final_value

        self.resolution_log.append(
            {
                "key": key,
                "source": source,
                "value": display_value,
                "type": type.__name__,
            }
        )

        return final_value

    def print_resolutions(self) -> None:
        """Print how each config was resolved."""
        if not self.resolution_log:
            print("No configurations resolved yet")
            return

        print("Configuration Resolution Log:")
        print("-" * 50)
        for entry in self.resolution_log:
            print(f"{entry['key']:<20} = {entry['value']:<20} ({entry['source']})")

    def clear_log(self) -> None:
        """Clear resolution log."""
        self.resolution_log = []

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

    def _is_sensitive(self, key: str) -> bool:
        """Check if key contains sensitive expressions."""
        key_upper = key.upper()
        return any(expr in key_upper for expr in self.SENSITIVE_EXPRESSIONS)

    def _mask_value(self, value: Any) -> str:
        """Mask sensitive values for display."""
        if value is None:
            return None
        value_str = str(value)
        if len(value_str) <= 4:
            return "****"
        return value_str[:2] + "*" * (len(value_str) - 4) + value_str[-2:]


# EOF
