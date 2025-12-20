#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_loader.py

"""FTS Bundle loading utilities."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from ._storage import get_storage

if TYPE_CHECKING:
    from ._dataclasses import DataInfo, Node
    from .._fig import Encoding, Theme
    from .._stats import Stats


def load_bundle_components(
    path: Path,
) -> Tuple[
    Optional["Node"],
    Optional["Encoding"],
    Optional["Theme"],
    Optional["Stats"],
    Optional["DataInfo"],
]:
    """Load all bundle components from storage.

    Args:
        path: Bundle path (directory or ZIP)

    Returns:
        Tuple of (node, encoding, theme, stats, data_info)
    """
    from ._dataclasses import DataInfo, Node
    from .._fig import Encoding, Theme
    from .._stats import Stats

    storage = get_storage(path)

    node = None
    encoding = None
    theme = None
    stats = None
    data_info = None

    # Node
    node_data = storage.read_json("node.json")
    if node_data:
        node = Node.from_dict(node_data)

    # Encoding
    encoding_data = storage.read_json("encoding.json")
    if encoding_data:
        encoding = Encoding.from_dict(encoding_data)

    # Theme
    theme_data = storage.read_json("theme.json")
    if theme_data:
        theme = Theme.from_dict(theme_data)

    # Stats
    stats_data = storage.read_json("stats/stats.json")
    if stats_data:
        stats = Stats.from_dict(stats_data)

    # Data info
    data_info_data = storage.read_json("data/data_info.json")
    if data_info_data:
        data_info = DataInfo.from_dict(data_info_data)

    return node, encoding, theme, stats, data_info


__all__ = ["load_bundle_components"]

# EOF
