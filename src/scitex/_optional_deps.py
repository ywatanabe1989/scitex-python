#!/usr/bin/env python3
"""
Utility for handling optional dependencies with helpful error messages.
"""

import importlib
from typing import Any, Dict, Optional

# Mapping of modules to their installation extras (module-level)
DEPENDENCY_GROUPS: Dict[str, str] = {
    # ai module
    "sklearn": "ai",
    "imblearn": "ai",
    "optuna": "ai",
    # audio module
    "pyttsx3": "audio",
    "gtts": "audio",
    "gTTS": "audio",
    "pydub": "audio",
    "elevenlabs": "audio",
    # browser module
    "selenium": "browser",
    "playwright": "browser",
    # capture module
    "mss": "capture",
    # cli module (uses browser deps)
    # dsp module
    "tensorpac": "dsp",
    "ipdb": "dsp",
    # fig module
    "flask": "fig",
    "dearpygui": "fig",
    # gen module (AI APIs)
    "openai": "gen",
    "anthropic": "gen",
    "google.generativeai": "gen",
    "groq": "gen",
    # ml module
    "skimage": "ml",
    "umap": "ml",
    "sktime": "ml",
    "catboost": "ml",
    "cv2": "ml",
    # msword module
    "docx": "msword",
    "pypandoc": "msword",
    # nn module
    "torch": "nn",
    "torchvision": "nn",
    "torchaudio": "nn",
    "torchsummary": "nn",
    "einops": "nn",
    # scholar module
    "crawl4ai": "scholar",
    "bs4": "scholar",
    "PyPDF2": "scholar",
    "pdfplumber": "scholar",
    "fitz": "scholar",  # PyMuPDF
    "pytesseract": "scholar",
    "bibtexparser": "scholar",
    "feedparser": "scholar",
    "httpx": "scholar",
    "tenacity": "scholar",
    "pydantic": "scholar",
    "watchdog": "scholar",
    # torch module
    # web module
    "fastapi": "web",
    "streamlit": "web",
    "celery": "web",
    # writer module
    "yq": "writer",
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


# Convenience: Check module-level dependency groups
def has_nn() -> bool:
    """Check if neural network (nn) dependencies are available."""
    return check_optional_deps("torch")["torch"]


def has_gen() -> bool:
    """Check if generative AI (gen) dependencies are available."""
    deps = check_optional_deps("openai", "anthropic")
    return any(deps.values())


def has_scholar() -> bool:
    """Check if scholar dependencies are available."""
    deps = check_optional_deps("selenium", "bs4")
    return all(deps.values())


def has_audio() -> bool:
    """Check if audio dependencies are available."""
    deps = check_optional_deps("pyttsx3", "gtts")
    return any(deps.values())


def has_browser() -> bool:
    """Check if browser automation dependencies are available."""
    deps = check_optional_deps("selenium", "playwright")
    return any(deps.values())


# EOF
