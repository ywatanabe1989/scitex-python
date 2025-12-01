#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 21:00:00 (ywatanabe)"
# File: ./src/scitex/config/__init__.py

"""
SciTeX configuration module.

Provides priority-based configuration resolution with clean precedence:
direct → config → env → default

Usage:
    from scitex.config import PriorityConfig

    config = PriorityConfig(config_dict={"port": 3000}, env_prefix="SCITEX_")
    port = config.resolve("port", None, default=8000, type=int)
"""

from .PriorityConfig import PriorityConfig

__all__ = ["PriorityConfig"]


# EOF
