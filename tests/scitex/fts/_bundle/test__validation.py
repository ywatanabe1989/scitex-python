# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_validation.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_validation.py
# 
# """
# FTS Multi-Level Validation System.
# 
# Three validation levels:
# 1. schema   - Always ON (init/load/save) - Fast, structural checks
# 2. semantic - On demand (save/export)   - Cross-reference consistency
# 3. strict   - Explicit (CI/publication) - Full scientific rigor
# 
# Usage:
#     result = fts.validate(level="schema")  # Fast, default at init
#     result = fts.validate(level="semantic")  # Cross-ref checks
#     result = fts.validate(level="strict")  # Full validation
# 
#     if result.has_errors:
#         raise FTSValidationError(result)
# """
# 
# import json
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Literal, Optional, Union
# 
# # Schema directory (now in _schemas/)
# SCHEMA_DIR = Path(__file__).parent.parent / "_schemas"
# 
# # Schema versions
# SCHEMA_VERSION = "1.0.0"
# 
# # Loaded schemas (cached)
# _SCHEMAS: Dict[str, Dict[str, Any]] = {}
# 
# # Validation levels
# ValidationLevel = Literal["schema", "semantic", "strict"]
# 
# 
# @dataclass
# class ValidationResult:
#     """Result of FTS validation.
# 
#     Attributes
#     ----------
#     level : str
#         Validation level used
#     errors : list
#         Critical errors (must fix)
#     warnings : list
#         Non-critical issues (should fix)
#     """
# 
#     level: str = "schema"
#     errors: List[str] = field(default_factory=list)
#     warnings: List[str] = field(default_factory=list)
# 
#     @property
#     def has_errors(self) -> bool:
#         """True if validation failed with errors."""
#         return len(self.errors) > 0
# 
#     @property
#     def has_warnings(self) -> bool:
#         """True if there are warnings."""
#         return len(self.warnings) > 0
# 
#     @property
#     def is_valid(self) -> bool:
#         """True if no errors (warnings OK)."""
#         return not self.has_errors
# 
#     def __str__(self) -> str:
#         if self.is_valid and not self.has_warnings:
#             return f"ValidationResult(level={self.level}, valid=True)"
#         parts = [f"ValidationResult(level={self.level}"]
#         if self.errors:
#             parts.append(f", errors={len(self.errors)}")
#         if self.warnings:
#             parts.append(f", warnings={len(self.warnings)}")
#         parts.append(")")
#         return "".join(parts)
# 
#     def raise_if_invalid(self):
#         """Raise BundleValidationError if has errors."""
#         if self.has_errors:
#             from ._utils import BundleValidationError
# 
#             raise BundleValidationError(
#                 f"Validation failed ({len(self.errors)} errors): {self.errors[0]}"
#             )
# 
# 
# def load_schema(name: str) -> Dict[str, Any]:
#     """Load a JSON schema by name."""
#     if name not in _SCHEMAS:
#         schema_path = SCHEMA_DIR / f"{name}.schema.json"
#         if not schema_path.exists():
#             raise FileNotFoundError(f"Schema not found: {schema_path}")
#         with open(schema_path) as f:
#             _SCHEMAS[name] = json.load(f)
#     return _SCHEMAS[name]
# 
# 
# # =============================================================================
# # Level 1: Schema Validation (Always ON)
# # =============================================================================
# 
# 
# def validate_schema(data: Dict[str, Any], schema_name: str) -> List[str]:
#     """Validate data against JSON schema.
# 
#     Fast structural validation:
#     - Required fields present
#     - Types correct
#     - Enum values valid
#     - bbox in range [0, 1]
#     """
#     errors = []
# 
#     try:
#         schema = load_schema(schema_name)
#     except FileNotFoundError as e:
#         return [str(e)]
# 
#     # Check required fields
#     required = schema.get("required", [])
#     for fld in required:
#         if fld not in data:
#             errors.append(f"Missing required field: {fld}")
# 
#     # Check property types
#     properties = schema.get("properties", {})
#     for fld, value in data.items():
#         if fld in properties:
#             prop_schema = properties[fld]
#             expected_type = prop_schema.get("type")
# 
#             if expected_type:
#                 type_map = {
#                     "string": str,
#                     "number": (int, float),
#                     "integer": int,
#                     "boolean": bool,
#                     "array": list,
#                     "object": dict,
#                 }
#                 expected = type_map.get(expected_type)
#                 if expected and not isinstance(value, expected):
#                     errors.append(
#                         f"Field '{fld}' should be {expected_type}, "
#                         f"got {type(value).__name__}"
#                     )
# 
#             # Check enum values
#             if "enum" in prop_schema and value not in prop_schema["enum"]:
#                 errors.append(
#                     f"Field '{fld}' must be one of {prop_schema['enum']}, got '{value}'"
#                 )
# 
#     # Special: bbox range validation
#     if "bbox" in data and isinstance(data["bbox"], dict):
#         bbox = data["bbox"]
#         for key in ["x", "y", "width", "height"]:
#             if key in bbox:
#                 val = bbox[key]
#                 if isinstance(val, (int, float)) and not (0 <= val <= 1):
#                     errors.append(f"bbox.{key} must be in range [0, 1], got {val}")
# 
#     return errors
# 
# 
# def validate_node(data: Dict[str, Any]) -> List[str]:
#     """Validate node.json data."""
#     return validate_schema(data, "node")
# 
# 
# def validate_encoding(data: Dict[str, Any]) -> List[str]:
#     """Validate encoding.json data."""
#     return validate_schema(data, "encoding")
# 
# 
# def validate_theme(data: Dict[str, Any]) -> List[str]:
#     """Validate theme.json data."""
#     return validate_schema(data, "theme")
# 
# 
# def validate_stats(data: Dict[str, Any]) -> List[str]:
#     """Validate stats.json data."""
#     return validate_schema(data, "stats")
# 
# 
# def validate_data_info(data: Dict[str, Any]) -> List[str]:
#     """Validate data_info.json data."""
#     return validate_schema(data, "data_info")
# 
# 
# # =============================================================================
# # Level 2: Semantic Validation (On demand)
# # =============================================================================
# 
# 
# def validate_semantic(
#     node: Optional[Dict] = None,
#     encoding: Optional[Dict] = None,
#     theme: Optional[Dict] = None,
#     stats: Optional[Dict] = None,
#     data_info: Optional[Dict] = None,
# ) -> List[str]:
#     """Validate semantic consistency across bundle components.
# 
#     Cross-reference checks:
#     - encoding columns exist in data_info
#     - stats references valid data
#     - scale/normalization consistency
#     """
#     errors = []
# 
#     # Check encoding references data_info columns
#     if encoding and data_info:
#         columns = set()
#         if "columns" in data_info:
#             columns = {c.get("name") for c in data_info["columns"] if c.get("name")}
# 
#         traces = encoding.get("traces", [])
#         for trace in traces:
#             for channel in ["x", "y", "color", "size"]:
#                 if channel in trace:
#                     col = trace[channel].get("column")
#                     if col and columns and col not in columns:
#                         errors.append(
#                             f"Encoding references unknown column '{col}' "
#                             f"(available: {sorted(columns)})"
#                         )
# 
#     # Check stats references valid data
#     if stats and data_info:
#         analyses = stats.get("analyses", [])
#         columns = set()
#         if "columns" in data_info:
#             columns = {c.get("name") for c in data_info["columns"] if c.get("name")}
# 
#         for analysis in analyses:
#             data_ref = analysis.get("data_ref", {})
#             for key in ["group_column", "value_column"]:
#                 col = data_ref.get(key)
#                 if col and columns and col not in columns:
#                     errors.append(f"Stats references unknown column '{col}'")
# 
#     return errors
# 
# 
# # =============================================================================
# # Level 3: Strict Validation (Explicit call)
# # =============================================================================
# 
# 
# def validate_strict(
#     node: Optional[Dict] = None,
#     encoding: Optional[Dict] = None,
#     theme: Optional[Dict] = None,
#     stats: Optional[Dict] = None,
#     data_info: Optional[Dict] = None,
# ) -> List[str]:
#     """Strict validation for publication/CI.
# 
#     All semantic checks plus:
#     - Units must be specified
#     - Provenance must be complete
#     - All metadata fields populated
#     """
#     errors = []
# 
#     # Include all semantic errors
#     errors.extend(
#         validate_semantic(
#             node=node,
#             encoding=encoding,
#             theme=theme,
#             stats=stats,
#             data_info=data_info,
#         )
#     )
# 
#     # Units must be specified in data_info
#     if data_info:
#         columns = data_info.get("columns", [])
#         for col in columns:
#             if col.get("dtype") in ["float64", "int64", "number"]:
#                 if not col.get("unit"):
#                     errors.append(f"Column '{col.get('name')}' missing unit specification")
# 
#     # Provenance must be present
#     if data_info and not data_info.get("source"):
#         errors.append("data_info.source (provenance) is required for publication")
# 
#     # Node must have all metadata
#     if node:
#         if not node.get("name"):
#             errors.append("node.name is required for publication")
#         if not node.get("created_at"):
#             errors.append("node.created_at is required for publication")
# 
#     return errors
# 
# 
# # =============================================================================
# # Unified Validation Entry Point
# # =============================================================================
# 
# 
# def validate(
#     data: Dict[str, Any],
#     schema_name: str,
#     level: ValidationLevel = "schema",
# ) -> ValidationResult:
#     """Validate data at specified level.
# 
#     Parameters
#     ----------
#     data : dict
#         Data to validate
#     schema_name : str
#         Schema name: node, encoding, theme, stats, data_info
#     level : str
#         Validation level: schema, semantic, strict
# 
#     Returns
#     -------
#     ValidationResult
#         Validation result with errors and warnings
#     """
#     result = ValidationResult(level=level)
# 
#     # Level 1: Schema validation (always)
#     schema_errors = validate_schema(data, schema_name)
#     result.errors.extend(schema_errors)
# 
#     return result
# 
# 
# def validate_bundle(
#     bundle_path: Union[str, Path],
#     level: ValidationLevel = "schema",
# ) -> Dict[str, ValidationResult]:
#     """Validate all JSON files in a bundle at specified level.
# 
#     Parameters
#     ----------
#     bundle_path : str or Path
#         Path to bundle directory or ZIP
#     level : str
#         Validation level: schema, semantic, strict
# 
#     Returns
#     -------
#     dict
#         Dictionary mapping filename to ValidationResult
#     """
#     import zipfile
# 
#     bundle_path = Path(bundle_path)
#     results = {}
# 
#     def validate_json_file(name: str, content: str) -> ValidationResult:
#         """Validate a JSON file content."""
#         result = ValidationResult(level=level)
#         try:
#             data = json.loads(content)
#         except json.JSONDecodeError as e:
#             result.errors.append(f"Invalid JSON: {e}")
#             return result
# 
#         # Determine schema from filename
#         schema_map = {
#             "node.json": "node",
#             "encoding.json": "encoding",
#             "theme.json": "theme",
#         }
# 
#         for suffix, schema in schema_map.items():
#             if name.endswith(suffix):
#                 result.errors.extend(validate_schema(data, schema))
#                 break
#         else:
#             if name.endswith("stats.json"):
#                 result.errors.extend(validate_schema(data, "stats"))
#             elif name.endswith("data_info.json"):
#                 result.errors.extend(validate_schema(data, "data_info"))
# 
#         return result
# 
#     if bundle_path.is_file() and bundle_path.suffix == ".zip":
#         with zipfile.ZipFile(bundle_path, "r") as zf:
#             for name in zf.namelist():
#                 if name.endswith(".json"):
#                     content = zf.read(name).decode("utf-8")
#                     results[name] = validate_json_file(name, content)
#     elif bundle_path.is_dir():
#         for json_file in bundle_path.rglob("*.json"):
#             rel_path = json_file.relative_to(bundle_path)
#             with open(json_file) as f:
#                 content = f.read()
#             results[str(rel_path)] = validate_json_file(str(rel_path), content)
# 
#     return results
# 
# 
# __all__ = [
#     # Version/paths
#     "SCHEMA_VERSION",
#     "SCHEMA_DIR",
#     # Result class
#     "ValidationResult",
#     "ValidationLevel",
#     # Schema loading
#     "load_schema",
#     # Level 1: Schema
#     "validate_schema",
#     "validate_node",
#     "validate_encoding",
#     "validate_theme",
#     "validate_stats",
#     "validate_data_info",
#     # Level 2: Semantic
#     "validate_semantic",
#     # Level 3: Strict
#     "validate_strict",
#     # Unified
#     "validate",
#     "validate_bundle",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_validation.py
# --------------------------------------------------------------------------------
