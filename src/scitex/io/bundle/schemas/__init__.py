#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/io/bundle/schemas/__init__.py
"""SciTeX Bundle Schemas.

This module contains JSON schemas for bundle validation.
"""

from pathlib import Path

__all__ = [
    "SCHEMAS_DIR",
    "NODE_SCHEMA",
    "ENCODING_SCHEMA",
    "THEME_SCHEMA",
    "STATS_SCHEMA",
    "DATA_INFO_SCHEMA",
    "RENDER_MANIFEST_SCHEMA",
]

SCHEMAS_DIR = Path(__file__).parent

NODE_SCHEMA = SCHEMAS_DIR / "node.schema.json"
ENCODING_SCHEMA = SCHEMAS_DIR / "encoding.schema.json"
THEME_SCHEMA = SCHEMAS_DIR / "theme.schema.json"
STATS_SCHEMA = SCHEMAS_DIR / "stats.schema.json"
DATA_INFO_SCHEMA = SCHEMAS_DIR / "data_info.schema.json"
RENDER_MANIFEST_SCHEMA = SCHEMAS_DIR / "render_manifest.schema.json"

# EOF
