#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/DataInfo.py

"""DataInfo - Complete data info specification for a bundle."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ._ColumnDef import ColumnDef
from ._DataFormat import DataFormat
from ._DataSource import DataSource

if TYPE_CHECKING:
    import pandas

DATA_INFO_VERSION = "1.0.0"


@dataclass
class DataInfo:
    """Complete data info specification for a bundle.

    Stored in data/data_info.json.
    """

    columns: List[ColumnDef] = field(default_factory=list)
    source: Optional[DataSource] = None
    format: DataFormat = field(default_factory=DataFormat)
    shape: Optional[Dict[str, int]] = None  # rows, columns

    # Schema metadata
    schema_name: str = "fsb.data_info"
    schema_version: str = DATA_INFO_VERSION

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "columns": [c.to_dict() for c in self.columns],
            "format": self.format.to_dict(),
        }
        if self.source:
            result["source"] = self.source.to_dict()
        if self.shape:
            result["shape"] = self.shape
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataInfo":
        source = None
        if "source" in data:
            source = DataSource.from_dict(data["source"])
        return cls(
            columns=[ColumnDef.from_dict(c) for c in data.get("columns", [])],
            source=source,
            format=DataFormat.from_dict(data.get("format", {})),
            shape=data.get("shape"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DataInfo":
        return cls.from_dict(json.loads(json_str))

    def get_column(self, name: str) -> Optional[ColumnDef]:
        """Get column definition by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    @classmethod
    def from_dataframe(
        cls, df: "pandas.DataFrame", source_path: Optional[str] = None
    ) -> "DataInfo":
        """Create DataInfo from a pandas DataFrame."""
        import pandas as pd

        columns = []
        for col_name in df.columns:
            col = df[col_name]
            dtype = str(col.dtype)

            # Map pandas dtypes to our types
            if "float" in dtype:
                dtype_str = "float64"
            elif "int" in dtype:
                dtype_str = "int64"
            elif "bool" in dtype:
                dtype_str = "bool"
            elif "datetime" in dtype:
                dtype_str = "datetime"
            elif col.dtype.name == "category":
                dtype_str = "category"
            else:
                dtype_str = "string"

            col_def = ColumnDef(
                name=str(col_name),
                dtype=dtype_str,
                missing_count=int(col.isna().sum()),
                unique_count=int(col.nunique()),
            )

            # Add numeric stats
            if dtype_str in ("float64", "int64"):
                col_def.min = float(col.min()) if not pd.isna(col.min()) else None
                col_def.max = float(col.max()) if not pd.isna(col.max()) else None
                col_def.mean = float(col.mean()) if not pd.isna(col.mean()) else None
                col_def.std = float(col.std()) if not pd.isna(col.std()) else None

            # Add categories
            if dtype_str == "category" or (
                dtype_str == "string" and col.nunique() < 20
            ):
                col_def.categories = col.dropna().unique().tolist()

            columns.append(col_def)

        source = None
        if source_path:
            source = DataSource(path=source_path)

        return cls(
            columns=columns,
            source=source,
            shape={"rows": len(df), "columns": len(df.columns)},
        )


__all__ = ["DATA_INFO_VERSION", "DataInfo"]

# EOF
