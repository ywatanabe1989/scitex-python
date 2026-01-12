#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/ui/test__config.py

"""Tests for scitex.ui._backends._config module."""

import os

import pytest

from scitex.ui._backends._config import (
    BACKEND_PACKAGES,
    DEFAULT_CONFIG,
    UIConfig,
    get_config,
    is_backend_available,
    is_package_available,
)
from scitex.ui._backends._types import NotifyLevel


class TestPackageAvailability:
    """Tests for package availability checking."""

    def test_is_package_available_none(self):
        """None package (no requirement) should always be available."""
        assert is_package_available(None) is True

    def test_is_package_available_stdlib(self):
        """Standard library packages should be available."""
        assert is_package_available("os") is True
        assert is_package_available("json") is True

    def test_is_package_available_nonexistent(self):
        """Non-existent packages should not be available."""
        assert is_package_available("nonexistent_package_xyz123") is False

    def test_is_backend_available_audio(self):
        """Audio backend has no package requirement."""
        assert is_backend_available("audio") is True

    def test_is_backend_available_desktop(self):
        """Desktop backend has no package requirement."""
        assert is_backend_available("desktop") is True

    def test_is_backend_available_email(self):
        """Email backend has no package requirement (uses stdlib)."""
        assert is_backend_available("email") is True

    def test_is_backend_available_unknown(self):
        """Unknown backend should return True (no package in registry)."""
        assert is_backend_available("unknown_backend") is True

    def test_backend_packages_mapping(self):
        """Verify BACKEND_PACKAGES has expected entries."""
        assert "audio" in BACKEND_PACKAGES
        assert "desktop" in BACKEND_PACKAGES
        assert "matplotlib" in BACKEND_PACKAGES
        assert "playwright" in BACKEND_PACKAGES
        assert "email" in BACKEND_PACKAGES
        assert "webhook" in BACKEND_PACKAGES


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG."""

    def test_default_backend(self):
        assert DEFAULT_CONFIG["default_backend"] == "audio"

    def test_backend_priority_includes_all(self):
        priority = DEFAULT_CONFIG["backend_priority"]
        assert "audio" in priority
        assert "desktop" in priority
        assert "matplotlib" in priority
        assert "playwright" in priority
        assert "email" in priority
        assert "webhook" in priority

    def test_level_backends_exist(self):
        level_backends = DEFAULT_CONFIG["level_backends"]
        assert "info" in level_backends
        assert "warning" in level_backends
        assert "error" in level_backends
        assert "critical" in level_backends

    def test_timeouts_exist(self):
        timeouts = DEFAULT_CONFIG["timeouts"]
        assert "matplotlib" in timeouts
        assert "playwright" in timeouts
        assert timeouts["matplotlib"] == 5.0
        assert timeouts["playwright"] == 5.0


class TestUIConfig:
    """Tests for UIConfig class."""

    def setup_method(self):
        """Reset singleton before each test."""
        UIConfig.reset()

    def test_singleton_pattern(self):
        """UIConfig should be a singleton."""
        config1 = UIConfig()
        config2 = UIConfig()
        assert config1 is config2

    def test_custom_path_creates_new_instance(self):
        """Custom config path should create new instance."""
        config1 = UIConfig()
        config2 = UIConfig(config_path="/nonexistent/path.yaml")
        assert config1 is not config2

    def test_default_backend_property(self):
        config = get_config()
        assert config.default_backend == "audio"

    def test_backend_priority_property(self):
        config = get_config()
        priority = config.backend_priority
        assert isinstance(priority, list)
        assert "audio" in priority

    def test_get_backends_for_level_info(self):
        config = get_config()
        backends = config.get_backends_for_level(NotifyLevel.INFO)
        assert isinstance(backends, list)
        assert "audio" in backends

    def test_get_backends_for_level_critical(self):
        config = get_config()
        backends = config.get_backends_for_level(NotifyLevel.CRITICAL)
        assert isinstance(backends, list)
        # Critical should have more backends than info
        assert len(backends) >= 1

    def test_get_available_backend_priority(self):
        config = get_config()
        available = config.get_available_backend_priority()
        assert isinstance(available, list)
        # All returned backends should be available
        for backend in available:
            assert is_backend_available(backend)

    def test_get_available_backends_for_level(self):
        config = get_config()
        available = config.get_available_backends_for_level(NotifyLevel.INFO)
        assert isinstance(available, list)
        # All returned backends should be available
        for backend in available:
            assert is_backend_available(backend)

    def test_get_first_available_backend(self):
        config = get_config()
        first = config.get_first_available_backend()
        assert isinstance(first, str)
        assert is_backend_available(first)

    def test_get_timeout_matplotlib(self):
        config = get_config()
        timeout = config.get_timeout("matplotlib")
        assert isinstance(timeout, float)
        assert timeout > 0

    def test_get_timeout_unknown(self):
        """Unknown backend should return default timeout."""
        config = get_config()
        timeout = config.get_timeout("unknown_backend")
        assert timeout == 5.0

    def test_reload(self):
        """reload() should not raise."""
        config = get_config()
        config.reload()
        # Should still work after reload
        assert config.default_backend == "audio"


class TestUIConfigEnvOverride:
    """Tests for environment variable overrides."""

    def setup_method(self):
        UIConfig.reset()

    def test_env_override_default_backend(self, monkeypatch):
        monkeypatch.setenv("SCITEX_UI_DEFAULT_BACKEND", "desktop")
        UIConfig.reset()
        config = get_config()
        assert config.default_backend == "desktop"

    def test_env_override_backend_priority(self, monkeypatch):
        monkeypatch.setenv("SCITEX_UI_BACKEND_PRIORITY", "email,webhook")
        UIConfig.reset()
        config = get_config()
        assert config.backend_priority == ["email", "webhook"]

    def test_env_override_info_backends(self, monkeypatch):
        monkeypatch.setenv("SCITEX_UI_INFO_BACKENDS", "desktop,audio")
        UIConfig.reset()
        config = get_config()
        backends = config.get_backends_for_level(NotifyLevel.INFO)
        assert backends == ["desktop", "audio"]

    def test_env_override_timeout(self, monkeypatch):
        monkeypatch.setenv("SCITEX_UI_TIMEOUT_MATPLOTLIB", "10.0")
        UIConfig.reset()
        config = get_config()
        assert config.get_timeout("matplotlib") == 10.0


class TestGetConfig:
    """Tests for get_config function."""

    def setup_method(self):
        UIConfig.reset()

    def test_get_config_returns_uiconfig(self):
        config = get_config()
        assert isinstance(config, UIConfig)

    def test_get_config_with_path(self):
        config = get_config(config_path="/nonexistent/path.yaml")
        assert isinstance(config, UIConfig)

    def test_get_config_caches_singleton(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


if __name__ == "__main__":
    pytest.main([__file__])

# EOF
