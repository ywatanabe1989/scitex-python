#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/__init__.py

"""
Introspection utilities for Python packages.

Provides IPython-like introspection capabilities for any Python package.

Basic Introspection:
- get_signature: Function/class signature with type hints (like `func?`)
- get_docstring: Docstring extraction with parsing
- get_source: Full source code (like `func??`)
- list_members: List attributes/methods (like `dir()`)
- get_exports: Module's __all__ contents
- find_examples: Find usage examples in tests/examples

Advanced Introspection:
- get_class_hierarchy: Inheritance tree (MRO + subclasses)
- get_mro: Method Resolution Order only
- get_type_hints_detailed: Detailed type annotation analysis
- get_class_annotations: Class variable and method annotations
- get_imports: Static import analysis using AST
- get_dependencies: Module dependency analysis
- get_call_graph: Function call graph (with timeout protection)
- get_function_calls: Simple outgoing calls list
"""

from ._core import (
    # Basic
    find_examples,
    # Advanced - Call graph
    get_call_graph,
    # Advanced - Type hints
    get_class_annotations,
    # Advanced - Class hierarchy
    get_class_hierarchy,
    # Advanced - Imports
    get_dependencies,
    get_docstring,
    get_exports,
    get_function_calls,
    get_imports,
    get_mro,
    get_signature,
    get_source,
    get_type_hints_detailed,
    get_type_info,
    list_members,
    resolve_object,
)

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
