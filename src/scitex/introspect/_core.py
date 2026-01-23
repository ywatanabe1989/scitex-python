#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_core.py

"""Core introspection module - re-exports all utilities."""

# Basic introspection
# Advanced introspection
from ._call_graph import get_call_graph, get_function_calls
from ._class_hierarchy import get_class_hierarchy, get_mro
from ._docstring import get_docstring
from ._examples import find_examples
from ._imports import get_dependencies, get_imports
from ._members import get_exports, list_members
from ._resolve import get_type_info, resolve_object
from ._signature import get_signature
from ._source import get_source
from ._type_hints import get_class_annotations, get_type_hints_detailed

__all__ = [
    # Basic
    "get_signature",
    "get_docstring",
    "get_source",
    "list_members",
    "get_exports",
    "find_examples",
    "resolve_object",
    "get_type_info",
    # Advanced - Class hierarchy
    "get_class_hierarchy",
    "get_mro",
    # Advanced - Type hints
    "get_type_hints_detailed",
    "get_class_annotations",
    # Advanced - Imports
    "get_imports",
    "get_dependencies",
    # Advanced - Call graph
    "get_call_graph",
    "get_function_calls",
]
