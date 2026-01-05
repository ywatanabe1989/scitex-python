#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for custom log levels."""

import logging
import pytest

from scitex.logging._levels import (
    SUCCESS,
    FAIL,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
)


class TestLogLevels:
    """Test custom log level definitions."""

    def test_success_level_value(self):
        """Test SUCCESS level has correct value."""
        assert SUCCESS == 31
        assert SUCCESS > logging.WARNING
        assert SUCCESS < logging.ERROR

    def test_fail_level_value(self):
        """Test FAIL level has correct value."""
        assert FAIL == 35
        assert FAIL > logging.WARNING
        assert FAIL < logging.ERROR

    def test_success_level_name(self):
        """Test SUCCESS level has correct name."""
        assert logging.getLevelName(SUCCESS) == 'SUCC'

    def test_fail_level_name(self):
        """Test FAIL level has correct name."""
        assert logging.getLevelName(FAIL) == 'FAIL'

    def test_standard_level_names(self):
        """Test standard levels have 4-character names."""
        assert logging.getLevelName(DEBUG) == 'DEBU'
        assert logging.getLevelName(INFO) == 'INFO'
        assert logging.getLevelName(WARNING) == 'WARN'
        assert logging.getLevelName(ERROR) == 'ERRO'
        assert logging.getLevelName(CRITICAL) == 'CRIT'

    def test_standard_level_values(self):
        """Test standard levels maintain correct values."""
        assert DEBUG == logging.DEBUG
        assert INFO == logging.INFO
        assert WARNING == logging.WARNING
        assert ERROR == logging.ERROR
        assert CRITICAL == logging.CRITICAL

    def test_level_ordering(self):
        """Test all levels are properly ordered."""
        assert DEBUG < INFO < WARNING < SUCCESS < FAIL < ERROR < CRITICAL

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_levels.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Custom log levels for SciTeX."""
# 
# import logging
# 
# # Custom log levels for success/fail
# SUCCESS = 31  # Between WARNING (30) and ERROR (40)
# FAIL = 35  # Between WARNING (30) and ERROR (40)
# 
# # Add custom levels to logging module with 4-character abbreviations
# logging.addLevelName(SUCCESS, "SUCC")
# logging.addLevelName(FAIL, "FAIL")
# logging.addLevelName(logging.DEBUG, "DEBU")
# logging.addLevelName(logging.INFO, "INFO")
# logging.addLevelName(logging.WARNING, "WARN")
# logging.addLevelName(logging.ERROR, "ERRO")
# logging.addLevelName(logging.CRITICAL, "CRIT")
# 
# # Standard levels for convenience
# DEBUG = logging.DEBUG
# INFO = logging.INFO
# WARNING = logging.WARNING
# ERROR = logging.ERROR
# CRITICAL = logging.CRITICAL
# 
# __all__ = ["SUCCESS", "FAIL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_levels.py
# --------------------------------------------------------------------------------
