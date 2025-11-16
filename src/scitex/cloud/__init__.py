#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SciTeX Cloud Integration Module
# Provides cloud-specific functionality for SciTeX running in SciTeX Cloud

import os
import sys
import base64
from pathlib import Path
from typing import Optional


def is_cloud_environment() -> bool:
    """Check if running in SciTeX Cloud environment."""
    return os.environ.get('SCITEX_CLOUD_CODE_WORKSPACE') == 'true'


def get_cloud_backend() -> str:
    """Get the cloud backend type."""
    return os.environ.get('SCITEX_CLOUD_CODE_BACKEND', 'default')


def get_project_root() -> Optional[Path]:
    """Get the project root directory."""
    project_root = os.environ.get('SCITEX_CLOUD_CODE_PROJECT_ROOT')
    return Path(project_root) if project_root else None


def emit_inline_image(image_path: str, alt_text: str = "Figure") -> None:
    """
    Emit inline image marker for terminal display.

    The terminal frontend will detect this marker and render the image inline.

    Parameters
    ----------
    image_path : str
        Path to the image file (absolute or relative to project root)
    alt_text : str, optional
        Alternative text for the image
    """
    if not is_cloud_environment():
        return

    backend = get_cloud_backend()
    if backend != 'inline':
        return

    # Convert to absolute path if relative
    image_path = Path(image_path)
    if not image_path.is_absolute():
        project_root = get_project_root()
        if project_root:
            image_path = project_root / image_path

    # Check if file exists
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}", file=sys.stderr)
        return

    # Read and encode image as base64
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Emit special marker that frontend can detect
        # Format: \x1b]1337;File=inline=1;size=<size>:<base64_data>\x07
        # This is iTerm2-compatible inline image protocol
        print(f"\n\x1b]1337;File=name={base64.b64encode(image_path.name.encode()).decode()};inline=1:{image_base64}\x07")
        print(f"[Image: {image_path.name}]")  # Fallback text

    except Exception as e:
        print(f"Error displaying inline image: {e}", file=sys.stderr)


def emit_file_link(file_path: str, line_number: Optional[int] = None) -> None:
    """
    Emit file link marker for terminal display.

    The terminal frontend will detect this marker and create a clickable link.

    Parameters
    ----------
    file_path : str
        Path to the file (absolute or relative to project root)
    line_number : int, optional
        Line number in the file
    """
    if not is_cloud_environment():
        return

    # Convert to absolute path if relative
    file_path = Path(file_path)
    if not file_path.is_absolute():
        project_root = get_project_root()
        if project_root:
            file_path = project_root / file_path

    # Emit special marker that frontend can detect
    # Format: [FILE:<path>:<line>]
    if line_number:
        print(f"[FILE:{file_path}:{line_number}]")
    else:
        print(f"[FILE:{file_path}]")


__all__ = [
    'is_cloud_environment',
    'get_cloud_backend',
    'get_project_root',
    'emit_inline_image',
    'emit_file_link',
]

# EOF
