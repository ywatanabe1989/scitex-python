#!/usr/bin/env python3
"""Tests for CascadeConfig class."""

import os
from unittest.mock import patch

import pytest

from scitex.scholar.config.core._CascadeConfig import CascadeConfig


class TestCascadeConfigInit:
    """Tests for CascadeConfig initialization."""

    def test_init_creates_instance(self):
        """CascadeConfig should initialize without errors."""
        config = CascadeConfig()
        assert config is not None
        assert config.name == "CascadeConfig"

    def test_init_with_config_dict(self):
        """Should store provided config dict."""
        config_dict = {"key": "value", "num": 42}
        config = CascadeConfig(config_dict=config_dict)
        assert config.config_dict == config_dict

    def test_init_with_empty_config_dict(self):
        """Should default to empty dict when None."""
        config = CascadeConfig(config_dict=None)
        assert config.config_dict == {}

    def test_init_with_env_prefix(self):
        """Should store env prefix."""
        config = CascadeConfig(env_prefix="MYAPP_")
        assert config.env_prefix == "MYAPP_"

    def test_init_with_auto_uppercase(self):
        """Should store auto_uppercase setting."""
        config = CascadeConfig(auto_uppercase=False)
        assert config.auto_uppercase is False

    def test_init_creates_empty_resolution_log(self):
        """Should create empty resolution log."""
        config = CascadeConfig()
        assert config.resolution_log == []


class TestCascadeConfigRepr:
    """Tests for __repr__ method."""

    def test_repr_includes_prefix(self):
        """__repr__ should include env prefix."""
        config = CascadeConfig(env_prefix="TEST_")
        repr_str = repr(config)
        assert "TEST_" in repr_str

    def test_repr_includes_config_count(self):
        """__repr__ should include config count."""
        config = CascadeConfig(config_dict={"a": 1, "b": 2})
        repr_str = repr(config)
        assert "2" in repr_str


class TestCascadeConfigGet:
    """Tests for get method."""

    def test_get_returns_value_from_config(self):
        """get should return value from config dict."""
        config = CascadeConfig(config_dict={"my_key": "my_value"})
        assert config.get("my_key") == "my_value"

    def test_get_returns_none_for_missing_key(self):
        """get should return None for missing key."""
        config = CascadeConfig(config_dict={})
        assert config.get("missing") is None

    def test_get_ignores_env_vars(self):
        """get should only look in config dict, not env."""
        with patch.dict(os.environ, {"SOME_KEY": "env_value"}):
            config = CascadeConfig()
            assert config.get("SOME_KEY") is None


class TestCascadeConfigResolve:
    """Tests for resolve method."""

    def test_resolve_returns_direct_value(self):
        """resolve should return direct value when provided."""
        config = CascadeConfig(config_dict={"key": "config_val"})
        with patch.dict(os.environ, {"KEY": "env_val"}):
            result = config.resolve("key", direct_val="direct_val", default="default")
            assert result == "direct_val"

    def test_resolve_returns_config_value(self):
        """resolve should return config value when no direct value."""
        config = CascadeConfig(config_dict={"key": "config_val"})
        result = config.resolve("key", default="default")
        assert result == "config_val"

    def test_resolve_returns_env_value(self):
        """resolve should return env value when not in config."""
        config = CascadeConfig(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_MY_KEY": "env_value"}):
            result = config.resolve("my_key")
            assert result == "env_value"

    def test_resolve_returns_default_value(self):
        """resolve should return default when nothing else available."""
        config = CascadeConfig()
        result = config.resolve("unknown_key", default="default")
        assert result == "default"

    def test_resolve_respects_precedence_order(self):
        """Precedence: direct → config → env → default."""
        config = CascadeConfig(config_dict={"key": "config"}, env_prefix="")
        with patch.dict(os.environ, {"KEY": "env"}):
            # Direct wins over all
            assert config.resolve("key", direct_val="direct") == "direct"

        # Config wins over env
        config2 = CascadeConfig(config_dict={"test_key": "config"}, env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_TEST_KEY": "env"}):
            assert config2.resolve("test_key") == "config"

    def test_resolve_logs_resolution(self):
        """resolve should log each resolution."""
        config = CascadeConfig()
        config.resolve("key1", default="val1")
        config.resolve("key2", direct_val="val2")
        assert len(config.resolution_log) == 2
        assert config.resolution_log[0]["key"] == "key1"
        assert config.resolution_log[1]["key"] == "key2"

    def test_resolve_log_includes_source(self):
        """Resolution log should include source."""
        config = CascadeConfig(config_dict={"config_key": "val"})
        config.resolve("config_key")
        assert config.resolution_log[0]["source"] == "config"

        config.resolve("direct_key", direct_val="val")
        assert config.resolution_log[1]["source"] == "direct"

        config.resolve("default_key", default="val")
        assert config.resolution_log[2]["source"] == "default"


class TestCascadeConfigTypeConversion:
    """Tests for type conversion in resolve."""

    def test_resolve_converts_to_int(self):
        """resolve should convert env value to int."""
        config = CascadeConfig(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_NUM": "42"}):
            result = config.resolve("num", type=int)
            assert result == 42
            assert isinstance(result, int)

    def test_resolve_converts_to_float(self):
        """resolve should convert env value to float."""
        config = CascadeConfig(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_RATE": "3.14"}):
            result = config.resolve("rate", type=float)
            assert result == 3.14
            assert isinstance(result, float)

    def test_resolve_converts_to_bool_true(self):
        """resolve should convert 'true' to True."""
        config = CascadeConfig(env_prefix="TEST_")
        for true_val in ["true", "1", "yes"]:
            with patch.dict(os.environ, {"TEST_FLAG": true_val}):
                result = config.resolve("flag", type=bool)
                assert result is True

    def test_resolve_converts_to_bool_false(self):
        """resolve should convert other values to False."""
        config = CascadeConfig(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_FLAG": "false"}):
            result = config.resolve("flag", type=bool)
            assert result is False

    def test_resolve_converts_to_list(self):
        """resolve should convert comma-separated to list."""
        config = CascadeConfig(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_ITEMS": "a,b,c"}):
            result = config.resolve("items", type=list)
            assert result == ["a", "b", "c"]

    def test_resolve_string_type_passthrough(self):
        """resolve should return string as-is for str type."""
        config = CascadeConfig(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_NAME": "value"}):
            result = config.resolve("name", type=str)
            assert result == "value"
            assert isinstance(result, str)


class TestCascadeConfigSensitiveDetection:
    """Tests for sensitive value detection and masking."""

    def test_is_sensitive_detects_api(self):
        """Should detect API in key name."""
        config = CascadeConfig()
        assert config._is_sensitive("api_key") is True
        assert config._is_sensitive("API_TOKEN") is True

    def test_is_sensitive_detects_password(self):
        """Should detect PASSWORD in key name."""
        config = CascadeConfig()
        assert config._is_sensitive("password") is True
        assert config._is_sensitive("db_password") is True

    def test_is_sensitive_detects_secret(self):
        """Should detect SECRET in key name."""
        config = CascadeConfig()
        assert config._is_sensitive("secret_key") is True
        assert config._is_sensitive("app_secret") is True

    def test_is_sensitive_detects_token(self):
        """Should detect TOKEN in key name."""
        config = CascadeConfig()
        assert config._is_sensitive("access_token") is True
        assert config._is_sensitive("bearer_token") is True

    def test_is_sensitive_returns_false_for_normal_keys(self):
        """Should return False for non-sensitive keys."""
        config = CascadeConfig()
        assert config._is_sensitive("username") is False
        assert config._is_sensitive("host") is False
        assert config._is_sensitive("port") is False


class TestCascadeConfigMaskValue:
    """Tests for value masking."""

    def test_mask_value_masks_long_string(self):
        """Should mask middle of long string."""
        config = CascadeConfig()
        result = config._mask_value("secretvalue123")
        assert result.startswith("se")
        assert result.endswith("23")
        assert "****" in result or "*" in result

    def test_mask_value_masks_short_string(self):
        """Should fully mask short string."""
        config = CascadeConfig()
        result = config._mask_value("abc")
        assert result == "****"

    def test_mask_value_handles_none(self):
        """Should return None for None value."""
        config = CascadeConfig()
        assert config._mask_value(None) is None

    def test_mask_value_converts_non_string(self):
        """Should convert non-string to string for masking."""
        config = CascadeConfig()
        result = config._mask_value(12345678)
        assert "*" in result


class TestCascadeConfigMaskParameter:
    """Tests for mask parameter in resolve."""

    def test_resolve_masks_sensitive_by_default(self):
        """Should mask sensitive values in log by default."""
        config = CascadeConfig(config_dict={"api_key": "supersecret123"})
        config.resolve("api_key")
        log_value = config.resolution_log[0]["value"]
        assert "supersecret123" != log_value
        assert "*" in log_value

    def test_resolve_mask_false_disables_masking(self):
        """mask=False should disable masking even for sensitive keys."""
        config = CascadeConfig(config_dict={"api_key": "visible_value"})
        config.resolve("api_key", mask=False)
        log_value = config.resolution_log[0]["value"]
        assert log_value == "visible_value"

    def test_resolve_does_not_mask_non_sensitive(self):
        """Should not mask non-sensitive keys."""
        config = CascadeConfig(config_dict={"hostname": "myhost"})
        config.resolve("hostname")
        log_value = config.resolution_log[0]["value"]
        assert log_value == "myhost"


class TestCascadeConfigPrint:
    """Tests for print method."""

    def test_print_outputs_no_configs_message(self, capsys):
        """print should output message when log is empty."""
        config = CascadeConfig()
        config.print()
        captured = capsys.readouterr()
        assert "No configurations resolved yet" in captured.out

    def test_print_outputs_resolution_log(self, capsys):
        """print should output resolution log entries."""
        # Use non-sensitive key name (avoid KEY, API, TOKEN, etc.)
        config = CascadeConfig(config_dict={"hostname": "myhost"})
        config.resolve("hostname")
        config.print()
        captured = capsys.readouterr()
        assert "hostname" in captured.out
        assert "myhost" in captured.out
        assert "config" in captured.out

    def test_print_truncates_long_values(self, capsys):
        """print should truncate long values."""
        # Use non-sensitive key name
        config = CascadeConfig(config_dict={"longval": "a" * 100})
        config.resolve("longval")
        config.print()
        captured = capsys.readouterr()
        # Value should be truncated to 20 chars
        assert "a" * 21 not in captured.out


class TestCascadeConfigClearLog:
    """Tests for clear_log method."""

    def test_clear_log_empties_resolution_log(self):
        """clear_log should empty the resolution log."""
        config = CascadeConfig(config_dict={"a": 1, "b": 2})
        config.resolve("a")
        config.resolve("b")
        assert len(config.resolution_log) == 2
        config.clear_log()
        assert len(config.resolution_log) == 0


class TestCascadeConfigAutoUppercase:
    """Tests for auto_uppercase behavior."""

    def test_auto_uppercase_true_uppercases_env_key(self):
        """With auto_uppercase=True, env key should be uppercased."""
        config = CascadeConfig(env_prefix="TEST_", auto_uppercase=True)
        with patch.dict(os.environ, {"TEST_MY_KEY": "value"}):
            result = config.resolve("my_key")
            assert result == "value"

    def test_auto_uppercase_false_preserves_case(self):
        """With auto_uppercase=False, env key case is preserved."""
        config = CascadeConfig(env_prefix="TEST_", auto_uppercase=False)
        with patch.dict(os.environ, {"TEST_my_key": "value"}):
            result = config.resolve("my_key")
            assert result == "value"


class TestCascadeConfigIntegration:
    """Integration tests for CascadeConfig."""

    def test_django_style_usage(self):
        """Test Django-style configuration pattern."""
        config = CascadeConfig(
            config_dict={}, env_prefix="DJANGO_", auto_uppercase=True
        )

        with patch.dict(os.environ, {"DJANGO_DEBUG": "true"}):
            debug = config.resolve("debug", default=False, type=bool)
            assert debug is True

    def test_app_config_usage(self):
        """Test custom app configuration pattern."""
        yaml_data = {
            "host": "localhost",
            "port": 8080,
            "debug": False,
        }
        config = CascadeConfig(yaml_data, "MYAPP_")

        # Config value
        assert config.resolve("host", default="0.0.0.0") == "localhost"

        # Env override
        with patch.dict(os.environ, {"MYAPP_PORT": "9000"}):
            port = config.resolve("port", type=int)
            # Config takes precedence over env
            assert port == 8080

    def test_sensitive_data_handling(self):
        """Test sensitive data is properly masked."""
        config = CascadeConfig(config_dict={"api_key": "sk-1234567890"})
        config.resolve("api_key")

        # Should be masked in log
        log_value = config.resolution_log[0]["value"]
        assert "1234567890" not in log_value
        assert "*" in log_value


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
