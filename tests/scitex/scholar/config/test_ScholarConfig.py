#!/usr/bin/env python3
"""Tests for ScholarConfig class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.config.ScholarConfig import ScholarConfig


class TestScholarConfigInit:
    """Tests for ScholarConfig initialization."""

    def test_init_creates_instance(self):
        """ScholarConfig should initialize without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                assert config is not None
                assert config.name == "ScholarConfig"

    def test_init_stores_explicit_scholar_dir(self):
        """Should store explicit scholar_dir for thread-safe access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ScholarConfig(scholar_dir=tmpdir)
            assert config._explicit_scholar_dir == tmpdir

    def test_init_uses_custom_config_path(self):
        """Should load custom config when path exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom config file
            config_path = Path(tmpdir) / "custom.yaml"
            config_path.write_text("scholar_dir: ~/.custom_scitex\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig(config_path=config_path)
                assert config is not None

    def test_init_falls_back_to_default_config(self):
        """Should fall back to default config when path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig(config_path="/nonexistent/path.yaml")
                assert config is not None

    def test_init_creates_cascade_config(self):
        """Should create CascadeConfig with correct prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                assert hasattr(config, "cascade")
                assert config.cascade.env_prefix == "SCITEX_SCHOLAR_"

    def test_init_sets_up_path_manager(self):
        """Should set up PathManager during initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                assert hasattr(config, "path_manager")
                assert config.path_manager is not None


class TestScholarConfigLoadYaml:
    """Tests for load_yaml method."""

    def test_load_yaml_returns_dict(self):
        """load_yaml should return a dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("key: value\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                result = config.load_yaml(yaml_path)
                assert isinstance(result, dict)
                assert result["key"] == "value"

    def test_load_yaml_substitutes_env_vars(self):
        """load_yaml should substitute environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("test_value: ${TEST_VAR}\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir, "TEST_VAR": "hello"}):
                config = ScholarConfig()
                result = config.load_yaml(yaml_path)
                assert result["test_value"] == "hello"

    def test_load_yaml_uses_default_value(self):
        """load_yaml should use default when env var not set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text('test_value: ${UNSET_VAR:-"default_value"}\n')

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}, clear=False):
                # Ensure UNSET_VAR is not set
                os.environ.pop("UNSET_VAR", None)
                config = ScholarConfig()
                result = config.load_yaml(yaml_path)
                assert result["test_value"] == "default_value"

    def test_load_yaml_handles_boolean_values(self):
        """load_yaml should handle true/false values correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("enabled: ${TEST_BOOL:-true}\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                result = config.load_yaml(yaml_path)
                assert result["enabled"] is True

    def test_load_yaml_handles_null_values(self):
        """load_yaml should handle null values correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("value: ${UNSET_VAR:-null}\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                os.environ.pop("UNSET_VAR", None)
                config = ScholarConfig()
                result = config.load_yaml(yaml_path)
                assert result["value"] is None

    def test_load_yaml_raises_on_invalid_path(self):
        """load_yaml should raise ScholarError on invalid path."""
        from scitex.logging import ScholarError

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                with pytest.raises(ScholarError):
                    config.load_yaml(Path("/nonexistent/path.yaml"))


class TestScholarConfigResolve:
    """Tests for resolve method delegation."""

    def test_resolve_returns_direct_value(self):
        """resolve should return direct value when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                result = config.resolve("test_key", direct_val="direct")
                assert result == "direct"

    def test_resolve_returns_config_value(self):
        """resolve should return config value when no direct value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("test_key: config_value\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig(config_path=yaml_path)
                result = config.resolve("test_key")
                assert result == "config_value"

    def test_resolve_returns_env_value(self):
        """resolve should return env value when no config value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"SCITEX_DIR": tmpdir, "SCITEX_SCHOLAR_NEW_KEY": "env_value"},
            ):
                config = ScholarConfig()
                result = config.resolve("new_key")
                assert result == "env_value"

    def test_resolve_returns_default_value(self):
        """resolve should return default when nothing else available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                result = config.resolve("unknown_key", default="default")
                assert result == "default"

    def test_resolve_with_type_conversion(self):
        """resolve should convert types correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"SCITEX_DIR": tmpdir, "SCITEX_SCHOLAR_NUM_KEY": "42"},
            ):
                config = ScholarConfig()
                result = config.resolve("num_key", type=int)
                assert result == 42
                assert isinstance(result, int)


class TestScholarConfigGet:
    """Tests for get method."""

    def test_get_returns_config_value(self):
        """get should return value from config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("my_key: my_value\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig(config_path=yaml_path)
                result = config.get("my_key")
                assert result == "my_value"

    def test_get_returns_none_for_missing_key(self):
        """get should return None for missing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                result = config.get("nonexistent_key")
                assert result is None


class TestScholarConfigGetAttr:
    """Tests for __getattr__ delegation to path_manager."""

    def test_getattr_delegates_get_methods(self):
        """__getattr__ should delegate get_ methods to path_manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                # get_cache_auth_dir should be delegated
                result = config.get_cache_auth_dir()
                assert isinstance(result, Path)

    def test_getattr_raises_for_unknown_attribute(self):
        """__getattr__ should raise AttributeError for unknown attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                with pytest.raises(AttributeError) as excinfo:
                    _ = config.unknown_attribute
                assert "ScholarConfig" in str(excinfo.value)

    def test_getattr_raises_for_non_get_methods(self):
        """__getattr__ should raise for non-get_ method names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                with pytest.raises(AttributeError):
                    _ = config.unknown_method


class TestScholarConfigDir:
    """Tests for __dir__ method."""

    def test_dir_includes_own_attributes(self):
        """__dir__ should include own attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                attrs = dir(config)
                assert "name" in attrs
                assert "cascade" in attrs
                assert "path_manager" in attrs

    def test_dir_includes_path_manager_get_methods(self):
        """__dir__ should include path_manager's get_ methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                attrs = dir(config)
                # Should include delegated methods
                assert "get_cache_auth_dir" in attrs
                assert "get_library_master_dir" in attrs
                assert "get_workspace_dir" in attrs


class TestScholarConfigLoad:
    """Tests for load classmethod."""

    def test_load_returns_instance(self):
        """load should return a ScholarConfig instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig.load()
                assert isinstance(config, ScholarConfig)

    def test_load_with_path(self):
        """load should accept a config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "custom.yaml"
            yaml_path.write_text("custom_key: custom_value\n")

            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig.load(yaml_path)
                assert isinstance(config, ScholarConfig)


class TestScholarConfigPaths:
    """Tests for paths property."""

    def test_paths_returns_path_manager(self):
        """paths property should return the path_manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                assert config.paths is config.path_manager


class TestSetupPathManager:
    """Tests for _setup_path_manager method."""

    def test_uses_explicit_scholar_dir(self):
        """Should use explicit scholar_dir when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ScholarConfig(scholar_dir=tmpdir)
            expected = Path(tmpdir).expanduser() / "scholar"
            assert config.path_manager.scholar_dir == expected

    def test_uses_env_var_when_no_explicit_dir(self):
        """Should use SCITEX_DIR env var when no explicit dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                expected = Path(tmpdir).expanduser() / "scholar"
                assert config.path_manager.scholar_dir == expected

    def test_uses_default_when_nothing_set(self):
        """Should use ~/.scitex when nothing is set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SCITEX_DIR if set
            os.environ.pop("SCITEX_DIR", None)
            config = ScholarConfig()
            expected = Path.home() / ".scitex" / "scholar"
            assert config.path_manager.scholar_dir == expected


class TestScholarConfigPrintAndClearLog:
    """Tests for print and clear_log methods."""

    def test_print_calls_cascade_print(self):
        """print should delegate to cascade.print."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                # Should not raise
                config.print()

    def test_clear_log_clears_resolution_log(self):
        """clear_log should clear the resolution log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                # Add something to the log
                config.resolve("test_key", default="value")
                assert len(config.cascade.resolution_log) > 0
                # Clear it
                config.clear_log()
                assert len(config.cascade.resolution_log) == 0


class TestScholarConfigIntegration:
    """Integration tests for ScholarConfig."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom config
            config_path = Path(tmpdir) / "test_config.yaml"
            config_path.write_text(
                """
scholar_dir: ~/.scitex
project: test_project
debug_mode: true
max_workers: 8
"""
            )

            with patch.dict(
                os.environ,
                {"SCITEX_DIR": tmpdir, "SCITEX_SCHOLAR_NEW_KEY": "env_value"},
            ):
                config = ScholarConfig(config_path=config_path)

                # Test cascade resolution - config takes precedence over env
                project = config.resolve("project")
                assert project == "test_project"

                # Test env value when not in config
                new_key = config.resolve("new_key")
                assert new_key == "env_value"

                # Test config value
                max_workers = config.resolve("max_workers", default=4, type=int)
                assert max_workers == 8

                # Test path manager access
                library_dir = config.paths.library_dir
                assert library_dir.exists()
                assert "library" in str(library_dir)

    def test_thread_safe_multi_user_scenario(self):
        """Test thread-safe usage with explicit scholar_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            user1_dir = Path(tmpdir) / "user1"
            user2_dir = Path(tmpdir) / "user2"

            config1 = ScholarConfig(scholar_dir=user1_dir)
            config2 = ScholarConfig(scholar_dir=user2_dir)

            # Each config should have its own path
            assert config1.path_manager.scholar_dir != config2.path_manager.scholar_dir
            assert "user1" in str(config1.path_manager.scholar_dir)
            assert "user2" in str(config2.path_manager.scholar_dir)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
