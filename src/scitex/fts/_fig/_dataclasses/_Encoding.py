#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/Encoding.py

"""Encoding - Complete encoding specification for a bundle."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ._TraceEncoding import TraceEncoding

ENCODING_VERSION = "1.0.0"


@dataclass
class Encoding:
    """Complete encoding specification for a bundle.

    Stored in encoding.json at bundle root.
    """

    traces: List[TraceEncoding] = field(default_factory=list)

    # Schema metadata
    schema_name: str = "fsb.encoding"
    schema_version: str = ENCODING_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces": [t.to_dict() for t in self.traces],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Encoding":
        return cls(
            traces=[TraceEncoding.from_dict(t) for t in data.get("traces", [])],
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Encoding":
        return cls.from_dict(json.loads(json_str))


__all__ = ["ENCODING_VERSION", "Encoding"]

# EOF
