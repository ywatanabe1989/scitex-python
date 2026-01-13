#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/dev/cv/__init__.py
"""SciTeX Development CV Module.

Provides tools for creating professional video content:
- Opening/closing title cards (PNG/MP4)
- Video concatenation
- Transition effects

Features:
- Matplotlib-based title card generation (no browser required)
- ffmpeg-based video composition
- Consistent SciTeX branding

Example
-------
>>> from scitex.dev import cv
>>> # Create opening title card
>>> cv.create_opening(
...     title="My Research Demo",
...     subtitle="Part of SciTeX",
...     output_path="/tmp/opening.png"
... )
>>> # Create closing card
>>> cv.create_closing(
...     output_path="/tmp/closing.png"
... )
>>> # Compose full video
>>> cv.compose(
...     opening="/tmp/opening.png",
...     content="/tmp/main_video.mp4",
...     closing="/tmp/closing.png",
...     output="/tmp/final.mp4"
... )
"""

from ._compose import (
    compose,
    concatenate_videos,
    image_to_video,
)
from ._title_card import (
    create_closing,
    create_opening,
    create_title_card,
)

__all__ = [
    # Title cards
    "create_opening",
    "create_closing",
    "create_title_card",
    # Composition
    "compose",
    "concatenate_videos",
    "image_to_video",
]

# EOF
