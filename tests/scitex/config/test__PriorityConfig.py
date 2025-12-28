#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./tests/scitex/config/test__PriorityConfig.py

"""Tests for PriorityConfig class and load_dotenv, get_scitex_dir functions."""

import os
import tempfile
import pytest
from pathlib import Path
from scitex.config import PriorityConfig, load_dotenv, get_scitex_dir


class TestPriorityConfigBasic:
    """Basic PriorityConfig functionality tests."""

    def test_initialization(self):
        """Test PriorityConfig can be initialized."""
        config = PriorityConfig()
        assert config is not None

    def test_initialization_with_dict(self):
        """Test initialization with config dict."""
        config = PriorityConfig(config_dict={"port": 3000})
        assert config.get("port") == 3000

    def test_initialization_with_prefix(self):
        """Test initialization with env prefix."""
        config = PriorityConfig(env_prefix="TEST_")
        assert config.env_prefix == "TEST_"

    def test_repr(self):
        """Test string representation."""
        config = PriorityConfig(config_dict={"a": 1, "b": 2}, env_prefix="APP_")
        repr_str = repr(config)
        assert "APP_" in repr_str
        assert "2" in repr_str


class TestPriorityConfigResolution:
    """Test priority resolution order: direct → config_dict → env → default."""

    def test_direct_value_highest_priority(self):
        """Test direct value takes highest priority."""
        config = PriorityConfig(config_dict={"port": 3000}, env_prefix="TEST_")
        result = config.resolve("port", direct_val=9000, default=8000)
        assert result == 9000

    def test_config_dict_over_env(self):
        """Test config_dict takes priority over env."""
        os.environ["TEST_PORT"] = "5000"
        try:
            config = PriorityConfig(config_dict={"port": 3000}, env_prefix="TEST_")
            result = config.resolve("port", default=8000)
            assert result == 3000
        finally:
            del os.environ["TEST_PORT"]

    def test_env_over_default(self):
        """Test env takes priority over default."""
        os.environ["TEST_HOST"] = "localhost"
        try:
            config = PriorityConfig(env_prefix="TEST_")
            result = config.resolve("host", default="0.0.0.0")
            assert result == "localhost"
        finally:
            del os.environ["TEST_HOST"]

    def test_default_fallback(self):
        """Test default is used when nothing else available."""
        config = PriorityConfig(env_prefix="TEST_")
        result = config.resolve("unknown_key", default="fallback")
        assert result == "fallback"


class TestPriorityConfigTypeConversion:
    """Test type conversion in resolve()."""

    def test_int_conversion(self):
        """Test integer type conversion."""
        os.environ["TEST_COUNT"] = "42"
        try:
            config = PriorityConfig(env_prefix="TEST_")
            result = config.resolve("count", default=0, type=int)
            assert result == 42
            assert isinstance(result, int)
        finally:
            del os.environ["TEST_COUNT"]

    def test_float_conversion(self):
        """Test float type conversion."""
        os.environ["TEST_RATE"] = "3.14"
        try:
            config = PriorityConfig(env_prefix="TEST_")
            result = config.resolve("rate", default=0.0, type=float)
            assert result == 3.14
        finally:
            del os.environ["TEST_RATE"]

    def test_bool_conversion_true(self):
        """Test boolean true conversion."""
        for true_val in ["true", "1", "yes"]:
            os.environ["TEST_DEBUG"] = true_val
            try:
                config = PriorityConfig(env_prefix="TEST_")
                result = config.resolve("debug", default=False, type=bool)
                assert result is True
            finally:
                del os.environ["TEST_DEBUG"]

    def test_list_conversion(self):
        """Test list type conversion."""
        os.environ["TEST_ITEMS"] = "a,b,c"
        try:
            config = PriorityConfig(env_prefix="TEST_")
            result = config.resolve("items", default=[], type=list)
            assert result == ["a", "b", "c"]
        finally:
            del os.environ["TEST_ITEMS"]


class TestPriorityConfigSensitiveValues:
    """Test sensitive value masking."""

    def test_sensitive_key_masked(self):
        """Test sensitive keys are automatically masked."""
        config = PriorityConfig(config_dict={"api_key": "secret123"})
        config.resolve("api_key", default="")
        log_entry = config.resolution_log[0]
        assert log_entry["value"] != "secret123"

    def test_mask_override_false(self):
        """Test mask=False overrides automatic masking."""
        config = PriorityConfig(config_dict={"api_key": "secret123"})
        config.resolve("api_key", default="", mask=False)
        log_entry = config.resolution_log[0]
        assert log_entry["value"] == "secret123"


class TestLoadDotenv:
    """Test load_dotenv() function."""

    def test_load_dotenv_from_explicit_path(self):
        """Test loading .env from explicit path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_DOTENV_VAR=explicit_value\n")
            temp_path = f.name

        if "TEST_DOTENV_VAR" in os.environ:
            del os.environ["TEST_DOTENV_VAR"]

        try:
            result = load_dotenv(temp_path)
            assert result is True
            assert os.environ.get("TEST_DOTENV_VAR") == "explicit_value"
        finally:
            os.unlink(temp_path)
            if "TEST_DOTENV_VAR" in os.environ:
                del os.environ["TEST_DOTENV_VAR"]

    def test_load_dotenv_returns_false_for_nonexistent(self):
        """Test load_dotenv returns False for nonexistent file."""
        result = load_dotenv("/nonexistent/path/.env")
        assert result is False

    def test_load_dotenv_skips_comments(self):
        """Test load_dotenv skips comment lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Comment\n")
            f.write("TEST_COMMENT_VAR=value\n")
            temp_path = f.name

        if "TEST_COMMENT_VAR" in os.environ:
            del os.environ["TEST_COMMENT_VAR"]

        try:
            load_dotenv(temp_path)
            assert os.environ.get("TEST_COMMENT_VAR") == "value"
        finally:
            os.unlink(temp_path)
            if "TEST_COMMENT_VAR" in os.environ:
                del os.environ["TEST_COMMENT_VAR"]

    def test_load_dotenv_handles_export_prefix(self):
        """Test load_dotenv handles 'export' prefix."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("export TEST_EXPORT_VAR=exported_value\n")
            temp_path = f.name

        if "TEST_EXPORT_VAR" in os.environ:
            del os.environ["TEST_EXPORT_VAR"]

        try:
            load_dotenv(temp_path)
            assert os.environ.get("TEST_EXPORT_VAR") == "exported_value"
        finally:
            os.unlink(temp_path)
            if "TEST_EXPORT_VAR" in os.environ:
                del os.environ["TEST_EXPORT_VAR"]

    def test_load_dotenv_removes_quotes(self):
        """Test load_dotenv removes quotes from values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('TEST_QUOTE_VAR="quoted value"\n')
            temp_path = f.name

        if "TEST_QUOTE_VAR" in os.environ:
            del os.environ["TEST_QUOTE_VAR"]

        try:
            load_dotenv(temp_path)
            assert os.environ.get("TEST_QUOTE_VAR") == "quoted value"
        finally:
            os.unlink(temp_path)
            if "TEST_QUOTE_VAR" in os.environ:
                del os.environ["TEST_QUOTE_VAR"]

    def test_load_dotenv_does_not_override_existing_env(self):
        """Test load_dotenv does not override existing env vars."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_EXISTING_VAR=from_dotenv\n")
            temp_path = f.name

        os.environ["TEST_EXISTING_VAR"] = "from_shell"

        try:
            load_dotenv(temp_path)
            assert os.environ.get("TEST_EXISTING_VAR") == "from_shell"
        finally:
            os.unlink(temp_path)
            del os.environ["TEST_EXISTING_VAR"]


class TestGetScitexDir:
    """Test get_scitex_dir() function."""

    def test_get_scitex_dir_default(self):
        """Test get_scitex_dir returns default ~/.scitex."""
        original = os.environ.pop("SCITEX_DIR", None)
        try:
            result = get_scitex_dir()
            assert result == Path.home() / ".scitex"
        finally:
            if original:
                os.environ["SCITEX_DIR"] = original

    def test_get_scitex_dir_from_env(self):
        """Test get_scitex_dir uses SCITEX_DIR env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SCITEX_DIR"] = tmpdir
            try:
                result = get_scitex_dir()
                assert result == Path(tmpdir)
            finally:
                del os.environ["SCITEX_DIR"]

    def test_get_scitex_dir_direct_value_highest(self):
        """Test get_scitex_dir with direct value takes precedence."""
        with tempfile.TemporaryDirectory() as env_dir:
            with tempfile.TemporaryDirectory() as direct_dir:
                os.environ["SCITEX_DIR"] = env_dir
                try:
                    result = get_scitex_dir(direct_val=direct_dir)
                    assert result == Path(direct_dir)
                finally:
                    del os.environ["SCITEX_DIR"]

    def test_get_scitex_dir_expands_user(self):
        """Test get_scitex_dir expands ~ in direct value."""
        result = get_scitex_dir(direct_val="~/custom_scitex")
        assert "~" not in str(result)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/config/_PriorityConfig.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-09 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/config/PriorityConfig.py
# 
# 
# """
# Priority-based configuration resolver.
# 
# Provides clean precedence hierarchy: direct → config_dict → env → default
# 
# Based on priority-config by ywatanabe (https://github.com/ywatanabe1989/priority-config)
# Incorporated into scitex for self-contained configuration management.
# 
# Note: config_dict values (from YAML or passed dict) take priority over
# environment variables. This follows the Scholar module's CascadeConfig pattern.
# """
# 
# import os
# from pathlib import Path
# from typing import Dict
# from typing import List
# from typing import Optional, Type, Any
# 
# 
# def load_dotenv(dotenv_path: Optional[str] = None) -> bool:
#     """Load environment variables from .env file.
# 
#     Searches for .env file in the following order:
#     1. Explicit dotenv_path if provided
#     2. Current working directory
#     3. User home directory
# 
#     Parameters
#     ----------
#     dotenv_path : str, optional
#         Path to .env file. If None, searches default locations.
# 
#     Returns
#     -------
#     bool
#         True if .env file was found and loaded, False otherwise.
#     """
#     paths_to_try = []
# 
#     if dotenv_path:
#         paths_to_try.append(Path(dotenv_path))
#     else:
#         # Default search paths
#         paths_to_try.extend(
#             [
#                 Path.cwd() / ".env",
#                 Path.home() / ".env",
#             ]
#         )
# 
#     for path in paths_to_try:
#         if path.exists() and path.is_file():
#             try:
#                 with open(path, "r") as f:
#                     for line in f:
#                         line = line.strip()
#                         # Skip empty lines and comments
#                         if not line or line.startswith("#"):
#                             continue
#                         # Handle export prefix
#                         if line.startswith("export "):
#                             line = line[7:]
#                         # Parse key=value
#                         if "=" in line:
#                             key, _, value = line.partition("=")
#                             key = key.strip()
#                             value = value.strip()
#                             # Remove quotes if present
#                             if (value.startswith('"') and value.endswith('"')) or (
#                                 value.startswith("'") and value.endswith("'")
#                             ):
#                                 value = value[1:-1]
#                             # Only set if not already in environment (env takes precedence)
#                             if key not in os.environ:
#                                 os.environ[key] = value
#                 return True
#             except Exception:
#                 continue
#     return False
# 
# 
# def get_scitex_dir(direct_val: Optional[str] = None) -> Path:
#     """Get SCITEX_DIR with priority: direct → env → default.
# 
#     This is a convenience function for the most common use case.
# 
#     Parameters
#     ----------
#     direct_val : str, optional
#         Direct value (highest precedence)
# 
#     Returns
#     -------
#     Path
#         Resolved SCITEX_DIR path
#     """
#     # Try to load .env first (won't override existing env vars)
#     load_dotenv()
# 
#     if direct_val is not None:
#         return Path(direct_val).expanduser()
# 
#     env_val = os.getenv("SCITEX_DIR")
#     if env_val:
#         return Path(env_val).expanduser()
# 
#     return Path.home() / ".scitex"
# 
# 
# class PriorityConfig:
#     """Universal config resolver with precedence: direct → config_dict → env → default
# 
#     Config dict (from YAML or passed dict) takes priority over env variables.
#     This follows the Scholar module's CascadeConfig pattern.
# 
#     Examples
#     --------
#     >>> from scitex.config import PriorityConfig
#     >>> config = PriorityConfig(config_dict={"port": 3000}, env_prefix="SCITEX_")
#     >>> port = config.resolve("port", None, default=8000, type=int)
#     3000  # from config_dict (highest after direct)
#     >>> # With env: SCITEX_PORT=5000 python script.py
#     >>> port = config.resolve("port", None, default=8000, type=int)
#     3000  # config_dict takes precedence over env
#     >>> port = config.resolve("port", 9000, default=8000, type=int)
#     9000  # direct value takes highest precedence
#     """
# 
#     SENSITIVE_EXPRESSIONS = [
#         "API",
#         "PASSWORD",
#         "SECRET",
#         "TOKEN",
#         "KEY",
#         "PASS",
#         "AUTH",
#         "CREDENTIAL",
#         "PRIVATE",
#         "CERT",
#     ]
# 
#     def __init__(
#         self,
#         config_dict: Optional[Dict[str, Any]] = None,
#         env_prefix: str = "",
#         auto_uppercase: bool = True,
#     ):
#         """Initialize PriorityConfig.
# 
#         Parameters
#         ----------
#         config_dict : dict, optional
#             Dictionary with configuration values
#         env_prefix : str
#             Prefix for environment variables (e.g., "SCITEX_")
#         auto_uppercase : bool
#             Whether to uppercase keys for env lookup
#         """
#         self.config_dict = config_dict or {}
#         self.env_prefix = env_prefix
#         self.auto_uppercase = auto_uppercase
#         self.resolution_log: List[Dict[str, Any]] = []
# 
#     def __repr__(self) -> str:
#         return f"PriorityConfig(prefix='{self.env_prefix}', configs={len(self.config_dict)})"
# 
#     def get(self, key: str) -> Any:
#         """Get value from config dict only."""
#         return self.config_dict.get(key)
# 
#     def resolve(
#         self,
#         key: str,
#         direct_val: Any = None,
#         default: Any = None,
#         type: Type = str,
#         mask: Optional[bool] = None,
#     ) -> Any:
#         """Get value with precedence hierarchy.
# 
#         Precedence: direct → config_dict → env → default
# 
#         This follows the Scholar module's CascadeConfig pattern where
#         config dict takes higher priority than environment variables.
# 
#         Parameters
#         ----------
#         key : str
#             Configuration key to resolve
#         direct_val : Any, optional
#             Direct value (highest precedence)
#         default : Any, optional
#             Default value if not found elsewhere
#         type : Type
#             Type conversion (str, int, float, bool, list)
#         mask : bool, optional
#             Override automatic masking of sensitive values
# 
#         Returns
#         -------
#         Any
#             Resolved configuration value
#         """
#         source = None
#         final_value = None
# 
#         # Replace dots with underscores for env key (e.g., axes.width_mm -> AXES_WIDTH_MM)
#         normalized_key = key.replace(".", "_")
#         env_key = f"{self.env_prefix}{normalized_key.upper() if self.auto_uppercase else normalized_key}"
#         env_val = os.getenv(env_key)
# 
#         # Priority: direct → config_dict → env → default
#         if direct_val is not None:
#             source = "direct"
#             final_value = direct_val
#         elif key in self.config_dict:
#             source = "config_dict"
#             final_value = self.config_dict[key]
#         elif env_val:
#             source = f"env:{env_key}"
#             final_value = self._convert_type(env_val, type)
#         else:
#             source = "default"
#             final_value = default
# 
#         if mask is False:
#             should_mask = False
#         else:
#             should_mask = self._is_sensitive(key)
# 
#         display_value = self._mask_value(final_value) if should_mask else final_value
# 
#         self.resolution_log.append(
#             {
#                 "key": key,
#                 "source": source,
#                 "value": display_value,
#                 "type": type.__name__,
#             }
#         )
# 
#         return final_value
# 
#     def print_resolutions(self) -> None:
#         """Print how each config was resolved."""
#         if not self.resolution_log:
#             print("No configurations resolved yet")
#             return
# 
#         print("Configuration Resolution Log:")
#         print("-" * 50)
#         for entry in self.resolution_log:
#             print(f"{entry['key']:<20} = {entry['value']:<20} ({entry['source']})")
# 
#     def clear_log(self) -> None:
#         """Clear resolution log."""
#         self.resolution_log = []
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
#     def _is_sensitive(self, key: str) -> bool:
#         """Check if key contains sensitive expressions."""
#         key_upper = key.upper()
#         return any(expr in key_upper for expr in self.SENSITIVE_EXPRESSIONS)
# 
#     def _mask_value(self, value: Any) -> str:
#         """Mask sensitive values for display."""
#         if value is None:
#             return None
#         value_str = str(value)
#         if len(value_str) <= 4:
#             return "****"
#         return value_str[:2] + "*" * (len(value_str) - 4) + value_str[-2:]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/config/_PriorityConfig.py
# --------------------------------------------------------------------------------
