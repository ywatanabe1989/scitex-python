#!/usr/bin/env python3
"""Scitex dev module - Development and debugging utilities."""

# Pyproject utilities (lazy import to avoid tomlkit dependency)
# Installation guide utilities (moved from root scitex module)
from .._install_guide import (
    MODULE_REQUIREMENTS,
    check_module_deps,
    require_module,
    requires,
    show_install_guide,
    warn_module_deps,
)
from . import _pyproject as pyproject
from . import cv
from ._analyze_code_flow import CodeFlowAnalyzer, analyze_code_flow, main, parse_args
from ._reload import reload, reload_auto, reload_stop

__all__ = [
    # Code flow analysis
    "CodeFlowAnalyzer",
    "analyze_code_flow",
    "main",
    "parse_args",
    # Hot reloading
    "reload",
    "reload_auto",
    "reload_stop",
    # Submodules
    "pyproject",
    "cv",
    # Installation guide utilities
    "show_install_guide",
    "check_module_deps",
    "require_module",
    "requires",
    "warn_module_deps",
    "MODULE_REQUIREMENTS",
]
