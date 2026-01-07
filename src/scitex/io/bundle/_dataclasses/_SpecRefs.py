#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_dataclasses/_SpecRefs.py

"""SpecRefs - References to associated files within the bundle."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SpecRefs:
    """References to associated files within the bundle.

    All paths are relative to the bundle root.
    """

    encoding: str = "encoding.json"
    theme: str = "theme.json"
    data: Optional[str] = None  # data/ directory or specific file
    stats: str = "stats/stats.json"

    def to_dict(self) -> Dict[str, str]:
        result = {
            "encoding": self.encoding,
            "theme": self.theme,
            "stats": self.stats,
        }
        if self.data:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecRefs":
        return cls(
            encoding=data.get("encoding", "encoding.json"),
            theme=data.get("theme", "theme.json"),
            data=data.get("data"),
            stats=data.get("stats", "stats/stats.json"),
        )


__all__ = ["SpecRefs"]

# EOF
