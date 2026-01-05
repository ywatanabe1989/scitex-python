#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./tests/scitex/config/test__ScitexConfig.py

"""Tests for ScitexConfig class and related functions."""

import os
import tempfile
import pytest
from pathlib import Path
from scitex.config import ScitexConfig, get_config, load_yaml


class TestLoadYaml:
    """Test load_yaml() function."""

    def test_load_yaml_basic(self):
        """Test basic YAML loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            temp_path = f.name

        try:
            result = load_yaml(Path(temp_path))
            assert result["key"] == "value"
            assert result["number"] == 42
        finally:
            os.unlink(temp_path)

    def test_load_yaml_nested(self):
        """Test loading nested YAML structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("parent:\n  child: value\n  nested:\n    deep: content\n")
            temp_path = f.name

        try:
            result = load_yaml(Path(temp_path))
            assert result["parent"]["child"] == "value"
            assert result["parent"]["nested"]["deep"] == "content"
        finally:
            os.unlink(temp_path)

    def test_load_yaml_env_substitution_with_default(self):
        """Test ${VAR:-default} syntax substitution."""
        # Ensure env var is not set
        if "TEST_YAML_VAR" in os.environ:
            del os.environ["TEST_YAML_VAR"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('value: ${TEST_YAML_VAR:-"default_value"}\n')
            temp_path = f.name

        try:
            result = load_yaml(Path(temp_path))
            assert result["value"] == "default_value"
        finally:
            os.unlink(temp_path)

    def test_load_yaml_env_substitution_with_env(self):
        """Test env var substitution when var is set."""
        os.environ["TEST_YAML_VAR2"] = "from_env"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('value: ${TEST_YAML_VAR2:-"default"}\n')
            temp_path = f.name

        try:
            result = load_yaml(Path(temp_path))
            assert result["value"] == "from_env"
        finally:
            os.unlink(temp_path)
            del os.environ["TEST_YAML_VAR2"]

    def test_load_yaml_boolean_values(self):
        """Test boolean value handling in env substitution."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("enabled: true\ndisabled: false\n")
            temp_path = f.name

        try:
            result = load_yaml(Path(temp_path))
            assert result["enabled"] is True
            assert result["disabled"] is False
        finally:
            os.unlink(temp_path)

    def test_load_yaml_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(ValueError):
            load_yaml(Path("/nonexistent/path/config.yaml"))


class TestScitexConfigBasic:
    """Basic ScitexConfig functionality tests."""

    def test_initialization_default(self):
        """Test ScitexConfig can be initialized with defaults."""
        config = ScitexConfig()
        assert config is not None

    def test_initialization_with_custom_path(self):
        """Test initialization with custom config path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("custom_key: custom_value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            assert config.get("custom_key") == "custom_value"
            assert config.config_path == Path(temp_path)
        finally:
            os.unlink(temp_path)

    def test_initialization_with_env_prefix(self):
        """Test initialization with custom env prefix."""
        config = ScitexConfig(env_prefix="CUSTOM_")
        assert config is not None

    def test_repr(self):
        """Test string representation."""
        config = ScitexConfig()
        repr_str = repr(config)
        assert "ScitexConfig" in repr_str
        assert "path=" in repr_str


class TestScitexConfigFlattenDict:
    """Test dictionary flattening functionality."""

    def test_flatten_simple(self):
        """Test flattening simple nested dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("parent:\n  child: value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            assert config.get("parent.child") == "value"
        finally:
            os.unlink(temp_path)

    def test_flatten_deep_nesting(self):
        """Test flattening deeply nested dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("a:\n  b:\n    c:\n      d: deep_value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            assert config.get("a.b.c.d") == "deep_value"
        finally:
            os.unlink(temp_path)


class TestScitexConfigGet:
    """Test get() method."""

    def test_get_existing_key(self):
        """Test getting existing key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test_key: test_value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            assert config.get("test_key") == "test_value"
        finally:
            os.unlink(temp_path)

    def test_get_nonexistent_key_returns_none(self):
        """Test getting nonexistent key returns None."""
        config = ScitexConfig()
        assert config.get("nonexistent_key") is None

    def test_get_with_default(self):
        """Test getting nonexistent key with default."""
        config = ScitexConfig()
        assert config.get("nonexistent_key", default="fallback") == "fallback"


class TestScitexConfigResolve:
    """Test resolve() method with priority order."""

    def test_resolve_direct_value_highest(self):
        """Test direct value takes highest priority."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test_key: from_config\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            result = config.resolve("test_key", direct_val="from_direct", default="from_default")
            assert result == "from_direct"
        finally:
            os.unlink(temp_path)

    def test_resolve_config_over_env(self):
        """Test config takes priority over env."""
        os.environ["SCITEX_TEST_KEY"] = "from_env"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test_key: from_config\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            result = config.resolve("test_key", default="from_default")
            assert result == "from_config"
        finally:
            os.unlink(temp_path)
            del os.environ["SCITEX_TEST_KEY"]

    def test_resolve_env_over_default(self):
        """Test env takes priority over default."""
        os.environ["SCITEX_MISSING_KEY"] = "from_env"

        try:
            config = ScitexConfig()
            result = config.resolve("missing_key", default="from_default")
            assert result == "from_env"
        finally:
            del os.environ["SCITEX_MISSING_KEY"]

    def test_resolve_default_fallback(self):
        """Test default is used when nothing else available."""
        config = ScitexConfig()
        result = config.resolve("totally_unknown", default="fallback_value")
        assert result == "fallback_value"

    def test_resolve_type_conversion_int(self):
        """Test type conversion to int."""
        os.environ["SCITEX_INT_VAL"] = "42"

        try:
            config = ScitexConfig()
            result = config.resolve("int_val", default=0, type=int)
            assert result == 42
            assert isinstance(result, int)
        finally:
            del os.environ["SCITEX_INT_VAL"]

    def test_resolve_type_conversion_bool(self):
        """Test type conversion to bool."""
        os.environ["SCITEX_BOOL_VAL"] = "true"

        try:
            config = ScitexConfig()
            result = config.resolve("bool_val", default=False, type=bool)
            assert result is True
        finally:
            del os.environ["SCITEX_BOOL_VAL"]


class TestScitexConfigGetNested:
    """Test get_nested() method."""

    def test_get_nested_simple(self):
        """Test getting nested value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("parent:\n  child: nested_value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            result = config.get_nested("parent", "child")
            assert result == "nested_value"
        finally:
            os.unlink(temp_path)

    def test_get_nested_deep(self):
        """Test getting deeply nested value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("a:\n  b:\n    c: deep\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            result = config.get_nested("a", "b", "c")
            assert result == "deep"
        finally:
            os.unlink(temp_path)

    def test_get_nested_missing_returns_default(self):
        """Test get_nested returns default for missing path."""
        config = ScitexConfig()
        result = config.get_nested("missing", "path", default="default_val")
        assert result == "default_val"


class TestScitexConfigProperties:
    """Test ScitexConfig properties."""

    def test_raw_property(self):
        """Test raw property returns original nested dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("parent:\n  child: value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            raw = config.raw
            assert isinstance(raw, dict)
            assert "parent" in raw
            assert raw["parent"]["child"] == "value"
        finally:
            os.unlink(temp_path)

    def test_flat_property(self):
        """Test flat property returns flattened dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("parent:\n  child: value\n")
            temp_path = f.name

        try:
            config = ScitexConfig(config_path=temp_path)
            flat = config.flat
            assert isinstance(flat, dict)
            assert "parent.child" in flat
            assert flat["parent.child"] == "value"
        finally:
            os.unlink(temp_path)


class TestGetConfig:
    """Test get_config() convenience function."""

    def test_get_config_returns_instance(self):
        """Test get_config returns ScitexConfig instance."""
        config = get_config()
        assert isinstance(config, ScitexConfig)

    def test_get_config_with_path(self):
        """Test get_config with custom path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("custom: value\n")
            temp_path = f.name

        try:
            config = get_config(config_path=temp_path)
            assert config.get("custom") == "value"
        finally:
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/config/_ScitexConfig.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/config/ScitexConfig.py
# 
# """
# YAML-based configuration for SciTeX with environment variable substitution.
# 
# Similar to ScholarConfig, provides:
# - YAML configuration loading
# - Environment variable substitution (${VAR:-default} syntax)
# - Cascade resolution (direct → config → env → default)
# 
# Usage:
#     from scitex.config import ScitexConfig
# 
#     # Load default configuration
#     config = ScitexConfig()
# 
#     # Load custom configuration
#     config = ScitexConfig(config_path="/path/to/config.yaml")
# 
#     # Resolve values with precedence
#     log_level = config.resolve("logging.level", default="INFO")
# """
# 
# import os
# import re
# from pathlib import Path
# from typing import Any, Dict, Optional, Type, Union
# 
# from ._PriorityConfig import PriorityConfig, load_dotenv
# 
# 
# def load_yaml(path: Path) -> dict:
#     """Load YAML file with environment variable substitution.
# 
#     Supports ${VAR:-default} syntax for environment variable expansion.
# 
#     Parameters
#     ----------
#     path : Path
#         Path to YAML file
# 
#     Returns
#     -------
#     dict
#         Parsed YAML with environment variables substituted
#     """
#     try:
#         import yaml
#     except ImportError:
#         raise ImportError(
#             "PyYAML required for YAML config. Install with: pip install pyyaml"
#         )
# 
#     try:
#         with open(path) as f:
#             content = f.read()
# 
#         def env_replacer(match):
#             """Replace ${VAR:-default} with environment variable or default."""
#             env_expr = match.group(1)
#             if ":-" in env_expr:
#                 var_name, default_value = env_expr.split(":-", 1)
#                 value = os.getenv(var_name, default_value.strip('"'))
#             else:
#                 value = os.getenv(env_expr)
# 
#             # Handle special values
#             if value in ["true", "false"]:
#                 return value
#             elif value == "null":
#                 return "null"
#             elif value and not (value.startswith('"') and value.endswith('"')):
#                 return f'"{value}"'
#             else:
#                 return value or "null"
# 
#         content = re.sub(r"\$\{([^}]+)\}", env_replacer, content)
#         return yaml.safe_load(content)
#     except Exception as e:
#         raise ValueError(f"Failed to load YAML config from {path}: {e}")
# 
# 
# class ScitexConfig:
#     """YAML-based configuration manager for SciTeX.
# 
#     Loads configuration from YAML files with environment variable substitution.
#     Values can be resolved with priority: direct → config → env → default.
# 
#     Examples
#     --------
#     >>> from scitex.config import ScitexConfig
#     >>> config = ScitexConfig()
#     >>> config.resolve("logging.level", default="INFO")
#     'INFO'
#     >>> config.get("debug.enabled")
#     False
#     """
# 
#     def __init__(
#         self,
#         config_path: Optional[Union[str, Path]] = None,
#         env_prefix: str = "SCITEX_",
#     ):
#         """Initialize ScitexConfig.
# 
#         Parameters
#         ----------
#         config_path : str or Path, optional
#             Path to custom YAML config file. If None, uses default.yaml.
#         env_prefix : str
#             Prefix for environment variables (default: "SCITEX_")
#         """
#         # Load .env file first
#         load_dotenv()
# 
#         # Load YAML configuration
#         if config_path and Path(config_path).exists():
#             self._config_data = load_yaml(Path(config_path))
#             self._config_path = Path(config_path)
#         else:
#             default_path = Path(__file__).parent / "default.yaml"
#             if default_path.exists():
#                 self._config_data = load_yaml(default_path)
#             else:
#                 self._config_data = {}
#             self._config_path = default_path
# 
#         # Flatten nested config for easy access
#         self._flat_config = self._flatten_dict(self._config_data)
# 
#         # Initialize PriorityConfig for resolution
#         self._priority_config = PriorityConfig(
#             config_dict=self._flat_config,
#             env_prefix=env_prefix,
#         )
# 
#     def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
#         """Flatten nested dictionary with dot notation keys.
# 
#         Parameters
#         ----------
#         d : dict
#             Dictionary to flatten
#         parent_key : str
#             Parent key for recursion
#         sep : str
#             Separator for nested keys
# 
#         Returns
#         -------
#         dict
#             Flattened dictionary
#         """
#         items = []
#         for k, v in d.items():
#             new_key = f"{parent_key}{sep}{k}" if parent_key else k
#             if isinstance(v, dict):
#                 items.extend(self._flatten_dict(v, new_key, sep).items())
#             else:
#                 items.append((new_key, v))
#         return dict(items)
# 
#     def get(self, key: str, default: Any = None) -> Any:
#         """Get value from config directly (no precedence resolution).
# 
#         Supports dot notation for nested keys.
# 
#         Parameters
#         ----------
#         key : str
#             Configuration key (e.g., "logging.level" or "debug.enabled")
#         default : Any
#             Default value if key not found
# 
#         Returns
#         -------
#         Any
#             Configuration value
#         """
#         return self._flat_config.get(key, default)
# 
#     def resolve(
#         self,
#         key: str,
#         direct_val: Any = None,
#         default: Any = None,
#         type: Type = str,
#     ) -> Any:
#         """Resolve value with precedence: direct → config → env → default.
# 
#         This follows the Scholar module's CascadeConfig pattern where
#         YAML config takes higher priority than environment variables.
# 
#         Parameters
#         ----------
#         key : str
#             Configuration key (e.g., "logging.level")
#         direct_val : Any
#             Direct value (highest precedence)
#         default : Any
#             Default value (lowest precedence)
#         type : Type
#             Type conversion (str, int, float, bool, list)
# 
#         Returns
#         -------
#         Any
#             Resolved value
#         """
#         # Priority: direct → config → env → default
#         # (Same as Scholar's CascadeConfig pattern)
#         if direct_val is not None:
#             return direct_val
# 
#         # Config (YAML) takes priority over env
#         config_val = self._flat_config.get(key)
#         if config_val is not None:
#             return config_val
# 
#         # Then check environment variable
#         normalized_key = key.replace(".", "_")
#         env_key = f"SCITEX_{normalized_key.upper()}"
#         env_val = os.getenv(env_key)
#         if env_val:
#             return self._convert_type(env_val, type)
# 
#         return default
# 
#     def _convert_type(self, value: str, type: Type) -> Any:
#         """Convert string value to specified type."""
#         if type == int:
#             return int(value)
#         elif type == float:
#             return float(value)
#         elif type == bool:
#             return value.lower() in ("true", "1", "yes")
#         elif type == list:
#             return value.split(",")
#         return value
# 
#     def get_nested(self, *keys: str, default: Any = None) -> Any:
#         """Get nested value from original config structure.
# 
#         Parameters
#         ----------
#         *keys : str
#             Keys to traverse (e.g., "browser", "screenshots_dir")
#         default : Any
#             Default value if not found
# 
#         Returns
#         -------
#         Any
#             Nested value
#         """
#         current = self._config_data
#         for key in keys:
#             if isinstance(current, dict) and key in current:
#                 current = current[key]
#             else:
#                 return default
#         return current
# 
#     @property
#     def config_path(self) -> Path:
#         """Get the path to the loaded config file."""
#         return self._config_path
# 
#     @property
#     def raw(self) -> dict:
#         """Get raw configuration data (original nested structure)."""
#         return self._config_data
# 
#     @property
#     def flat(self) -> dict:
#         """Get flattened configuration data."""
#         return self._flat_config
# 
#     def print(self) -> None:
#         """Print configuration resolution log."""
#         self._priority_config.print_resolutions()
# 
#     def __repr__(self) -> str:
#         return f"ScitexConfig(path='{self._config_path}')"
# 
# 
# # Module-level convenience functions
# 
# _default_config: Optional[ScitexConfig] = None
# 
# 
# def get_config(config_path: Optional[Union[str, Path]] = None) -> ScitexConfig:
#     """Get ScitexConfig instance.
# 
#     Parameters
#     ----------
#     config_path : str or Path, optional
#         Path to custom config. If None, returns cached default instance.
# 
#     Returns
#     -------
#     ScitexConfig
#         Configuration instance
#     """
#     global _default_config
# 
#     if config_path is not None:
#         return ScitexConfig(config_path)
# 
#     if _default_config is None:
#         _default_config = ScitexConfig()
# 
#     return _default_config
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/config/_ScitexConfig.py
# --------------------------------------------------------------------------------
