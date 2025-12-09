#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility for handling optional dependencies with helpful error messages.
"""

from typing import Optional, Dict, Any
import importlib


# Mapping of modules to their installation extras
DEPENDENCY_GROUPS: Dict[str, str] = {
    # Deep Learning
    "torch": "dl",
    "torchvision": "dl",
    "torchaudio": "dl",
    "transformers": "dl",
    "accelerate": "dl",
    "bitsandbytes": "dl",
    "fairscale": "dl",
    # AI APIs
    "openai": "ai-apis",
    "anthropic": "ai-apis",
    "google.generativeai": "ai-apis",
    "groq": "ai-apis",
    # Scholar
    "selenium": "scholar",
    "playwright": "scholar",
    "crawl4ai": "scholar",
    "bs4": "scholar",
    # Neuroscience
    "mne": "neuro",
    "obspy": "neuro",
    "pyedflib": "neuro",
    "tensorpac": "neuro",
    # Web
    "fastapi": "web",
    "flask": "web",
    "streamlit": "web",
    # Jupyter
    "jupyterlab": "jupyter",
    "IPython": "jupyter",
    "ipykernel": "jupyter",
}


def optional_import(
    module_name: str, package_name: Optional[str] = None, raise_error: bool = True
) -> Optional[Any]:
    """
    Import an optional dependency with helpful error message.

    Args:
        module_name: Name of the module to import (e.g., 'torch')
        package_name: Name of the package to install if different from module
        raise_error: If True, raise ImportError. If False, return None.

    Returns:
        The imported module, or None if raise_error=False and import fails

    Examples:
        >>> torch = optional_import('torch')
        >>> torch = optional_import('torch', raise_error=False)  # Returns None if not installed
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not raise_error:
            return None

        pkg_name = package_name or module_name
        extra_group = DEPENDENCY_GROUPS.get(module_name, "all")

        error_msg = (
            f"\n{'=' * 70}\n"
            f"Optional dependency '{pkg_name}' is not installed.\n\n"
            f"To use this feature, install it with:\n"
            f"  pip install scitex[{extra_group}]\n\n"
            f"Or install all optional dependencies:\n"
            f"  pip install scitex[all]\n"
            f"{'=' * 70}\n"
        )
        raise ImportError(error_msg)


def check_optional_deps(*module_names: str) -> Dict[str, bool]:
    """
    Check which optional dependencies are available.

    Args:
        *module_names: Names of modules to check

    Returns:
        Dictionary mapping module names to availability (True/False)

    Example:
        >>> available = check_optional_deps('torch', 'transformers')
        >>> if available['torch']:
        ...     import torch
    """
    result = {}
    for name in module_names:
        try:
            importlib.import_module(name)
            result[name] = True
        except ImportError:
            result[name] = False
    return result


# Convenience: Check common dependency groups
def has_deep_learning() -> bool:
    """Check if deep learning dependencies are available."""
    return check_optional_deps("torch")["torch"]


def has_transformers() -> bool:
    """Check if transformers is available."""
    return check_optional_deps("transformers")["transformers"]


def has_scholar() -> bool:
    """Check if scholar dependencies are available."""
    deps = check_optional_deps("selenium", "bs4")
    return all(deps.values())


def has_jupyter() -> bool:
    """Check if Jupyter dependencies are available."""
    return check_optional_deps("IPython")["IPython"]


# EOF
