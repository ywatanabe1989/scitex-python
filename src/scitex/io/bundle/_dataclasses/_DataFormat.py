#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/DataFormat.py

"""DataFormat - Data file format specification."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DataFormat:
    """Data file format specification."""

    type: str = "csv"  # csv, tsv, parquet, json
    encoding: str = "utf-8"
    delimiter: str = ","
    header_row: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "header_row": self.header_row,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataFormat":
        return cls(
            type=data.get("type", "csv"),
            encoding=data.get("encoding", "utf-8"),
            delimiter=data.get("delimiter", ","),
            header_row=data.get("header_row", 0),
        )


__all__ = ["DataFormat"]

# EOF
