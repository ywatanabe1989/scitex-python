#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_saver.py

"""FTS Bundle saving utilities."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ._storage import get_storage

if TYPE_CHECKING:
    from ._dataclasses import DataInfo, Node
    from .._fig import Encoding, Theme
    from .._stats import Stats


def save_bundle_components(
    path: Path,
    node: Optional["Node"] = None,
    encoding: Optional["Encoding"] = None,
    theme: Optional["Theme"] = None,
    stats: Optional["Stats"] = None,
    data_info: Optional["DataInfo"] = None,
) -> None:
    """Save all bundle components to storage.

    Args:
        path: Bundle path (directory or ZIP)
        node: Node metadata
        encoding: Encoding specification
        theme: Theme specification
        stats: Statistics
        data_info: Data info metadata
    """
    storage = get_storage(path)

    # Collect all files to write
    files = {}

    if node:
        files["node.json"] = json.dumps(node.to_dict(), indent=2)

    if encoding:
        files["encoding.json"] = encoding.to_json()

    if theme:
        files["theme.json"] = theme.to_json()

    if stats and stats.analyses:
        files["stats/stats.json"] = stats.to_json()

    if data_info:
        files["data/data_info.json"] = data_info.to_json()

    # Write all files at once (more efficient for ZIP)
    storage.write_all(files)


__all__ = ["save_bundle_components"]

# EOF
