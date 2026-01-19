# Introspect Module

IPython-like introspection for any Python package. Helps AI agents and developers understand function signatures, docstrings, and code without guessing.

## Overview

The introspect module provides tools similar to IPython's `?` and `??` operators:

| IPython | scitex.introspect |
|---------|-------------------|
| `func?` | `get_signature()` |
| `func??` | `get_source()` |
| `dir(module)` | `list_members()` |

## Installation

Included with scitex. No additional dependencies required.

```bash
pip install scitex
```

## Basic Functions

### get_signature

Get function/class signature with parameters.

```python
from scitex.introspect import get_signature

result = get_signature("json.dumps")
# Returns: {
#   "success": True,
#   "name": "dumps",
#   "signature": "dumps(obj, ...)",
#   "parameters": [{"name": "obj", "annotation": "...", "default": "..."}]
# }
```

**Options:**
- `include_defaults=True` - Include default values
- `include_annotations=True` - Include type annotations

### get_docstring

Get docstring in raw, parsed, or summary format.

```python
from scitex.introspect import get_docstring

result = get_docstring("json.dumps", format="summary")
# Returns: {"success": True, "docstring": "Serialize obj to JSON..."}
```

**Formats:**
- `raw` - Full docstring as-is
- `summary` - First line only
- `parsed` - Parsed into sections (parameters, returns, etc.)

### get_source

Get source code with optional line limit.

```python
from scitex.introspect import get_source

result = get_source("json.dumps", max_lines=20)
# Returns: {"success": True, "source": "def dumps(...)...", "file": "...", "line_start": 123}
```

**Options:**
- `max_lines=None` - Limit output lines
- `include_decorators=True` - Include decorator lines

### list_members

List module/class members (like dir()).

```python
from scitex.introspect import list_members

result = list_members("json", filter="public", kind="functions")
# Returns: {"success": True, "members": [{"name": "dumps", "kind": "function", "summary": "..."}]}
```

**Filters:** `all`, `public`, `private`, `dunder`
**Kinds:** `all`, `functions`, `classes`, `data`, `modules`

### get_exports

Get module's `__all__` exports.

```python
from scitex.introspect import get_exports

result = get_exports("json")
# Returns: {"success": True, "exports": ["dump", "dumps", "load", "loads"], "has_all": True}
```

### find_examples

Find usage examples in tests/examples directories.

```python
from scitex.introspect import find_examples

result = find_examples("scitex.plt.plot", max_results=5)
# Returns: {"success": True, "examples": [{"file": "...", "line": 42, "context": "..."}]}
```

## Advanced Functions

### get_class_hierarchy

Get class MRO and subclasses.

```python
from scitex.introspect import get_class_hierarchy

result = get_class_hierarchy("collections.abc.Mapping", max_depth=2)
# Returns: {"success": True, "mro": [...], "subclasses": [...]}
```

### get_type_hints_detailed

Get detailed type hint analysis.

```python
from scitex.introspect import get_type_hints_detailed

result = get_type_hints_detailed("json.dumps")
# Returns: {"success": True, "hints": {"obj": {"raw": "Any", "is_optional": False}}}
```

### get_imports

Get module imports via AST analysis.

```python
from scitex.introspect import get_imports

result = get_imports("scitex.introspect._resolve", categorize=True)
# Returns: {"success": True, "categories": {"stdlib": [...], "third_party": [...], "local": [...]}}
```

### get_dependencies

Get module dependencies.

```python
from scitex.introspect import get_dependencies

result = get_dependencies("scitex.introspect._resolve", recursive=False)
# Returns: {"success": True, "dependencies": ["importlib", "types", ...]}
```

### get_call_graph

Get function call graph with timeout protection.

```python
from scitex.introspect import get_call_graph

result = get_call_graph("scitex.introspect._resolve", timeout_seconds=10)
# Returns: {"success": True, "graph": {...}}
```

## CLI Usage

All functions are available via CLI:

```bash
# Basic introspection
scitex introspect signature json.dumps
scitex introspect docstring json.dumps --format parsed
scitex introspect source scitex.plt.plot --max-lines 50

# Module exploration
scitex introspect members json --kind functions
scitex introspect exports json

# Advanced
scitex introspect hierarchy collections.abc.Mapping
scitex introspect hints scitex.introspect._resolve.resolve_object
scitex introspect imports scitex.introspect._resolve
scitex introspect deps scitex.introspect._resolve
scitex introspect calls scitex.introspect._resolve --timeout 30
```

Add `--json` to any command for JSON output.

## MCP Tools

When used via MCP, tools are prefixed with `introspect_`:

- `introspect_signature` - Get function/class signature
- `introspect_docstring` - Get docstring
- `introspect_source` - Get source code
- `introspect_members` - List module/class members
- `introspect_exports` - Get __all__ exports
- `introspect_examples` - Find usage examples
- `introspect_hierarchy` - Get class hierarchy
- `introspect_hints` - Get type hints
- `introspect_imports` - Get module imports
- `introspect_deps` - Get dependencies
- `introspect_calls` - Get call graph

## Error Handling

All functions return a dict with `success` field:

```python
result = get_signature("nonexistent.module")
if not result["success"]:
    print(f"Error: {result['error']}")
```

## Performance Notes

- Most functions are fast (<100ms)
- `get_call_graph` can be slow for large modules (use `timeout_seconds`)
- `get_dependencies(recursive=True)` can timeout (use `max_depth`)
- `get_imports(categorize=True)` loads stdlib module list once
