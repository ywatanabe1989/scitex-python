#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_dataclasses/__init__.py

"""Figure-specific dataclasses for FTS."""

from ._ChannelEncoding import ChannelEncoding
from ._Encoding import ENCODING_VERSION, AxesConfig, Encoding
from ._Theme import (
    THEME_VERSION,
    Colors,
    Grid,
    Lines,
    Markers,
    Theme,
    TraceTheme,
    Typography,
)
from ._TraceEncoding import TraceEncoding

__all__ = [
    # Encoding
    "ENCODING_VERSION",
    "AxesConfig",
    "ChannelEncoding",
    "TraceEncoding",
    "Encoding",
    # Theme
    "THEME_VERSION",
    "Colors",
    "Typography",
    "Lines",
    "Markers",
    "Grid",
    "TraceTheme",
    "Theme",
]

# EOF
