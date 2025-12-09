#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX CLI - Command-line interface for SciTeX platform

Provides unified interface for:
- Cloud operations (wraps tea for Gitea)
- Scholar operations (Django API)
- Code operations (Django API)
- Viz operations (Django API)
- Writer operations (Django API)
- Project operations (integrated workflows)
"""

from .main import cli

__all__ = ["cli"]
