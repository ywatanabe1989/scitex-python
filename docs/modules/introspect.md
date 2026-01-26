# Introspect Module

IPython-like introspection for any Python package. Helps AI agents and developers understand function signatures, docstrings, and code without guessing.

## Overview

The introspect module provides tools similar to IPython's `?` and `??` operators:

| IPython | scitex.introspect | CLI |
|---------|-------------------|-----|
| `func?` | `q()` | `scitex introspect q` |
| `func??` | `qq()` | `scitex introspect qq` |
| `dir(module)` | `dir()` | `scitex introspect dir` |
| (recursive) | `list_api()` | `scitex introspect api` |

## Installation

Included with scitex. No additional dependencies required.

```bash
pip install scitex
```

## Core Functions (IPython-style)

### q (signature)

Get function/class signature with parameters (like IPython's `func?`).

```python
from scitex.introspect import q

result = q("json.dumps")
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

### qq (source)

Get source code with optional line limit (like IPython's `func??`).

```python
from scitex.introspect import qq

result = qq("json.dumps", max_lines=20)
# Returns: {"success": True, "source": "def dumps(...)...", "file": "...", "line_start": 123}
```

**Options:**
- `max_lines=None` - Limit output lines
- `include_decorators=True` - Include decorator lines

### dir (members)

List module/class members (like Python's `dir()`).

```python
from scitex.introspect import dir

result = dir("json", filter="public", kind="functions")
# Returns: {"success": True, "members": [{"name": "dumps", "kind": "function", "summary": "..."}]}
```

**Filters:** `all`, `public`, `private`, `dunder`
**Kinds:** `all`, `functions`, `classes`, `data`, `modules`

### list_api (full API tree)

List the full API tree of a module recursively.

```python
from scitex.introspect import list_api

df = list_api("json", max_depth=2)
# Returns DataFrame with Type, Name, Docstring, Depth columns
```

**Options:**
- `max_depth=5` - Maximum recursion depth
- `docstring=False` - Include docstrings
- `root_only=False` - Show only root-level items

## Other Basic Functions

### get_docstring

Get docstring in raw, parsed, or summary format.

```python
from scitex.introspect import get_docstring

result = get_docstring("json.dumps", format="summary")
# Returns: {"success": True, "docstring": "Serialize obj to JSON..."}
```

**Formats:** `raw`, `summary`, `parsed`

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
# IPython-style commands
scitex introspect q json.dumps                    # signature
scitex introspect qq scitex.plt.plot --max-lines 50  # source
scitex introspect dir json --kind functions       # members
scitex introspect api scitex --max-depth 2        # full API tree

# Other basic commands
scitex introspect docstring json.dumps --format parsed
scitex introspect exports json
scitex introspect examples scitex.plt.plot

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

| Python | MCP Tool | Description |
|--------|----------|-------------|
| `q()` | `introspect_q` | Get signature (like func?) |
| `qq()` | `introspect_qq` | Get source (like func??) |
| `dir()` | `introspect_dir` | List members (like dir()) |
| `list_api()` | `introspect_api` | Full API tree |
| `get_docstring()` | `introspect_docstring` | Get docstring |
| `get_exports()` | `introspect_exports` | Get __all__ exports |
| `find_examples()` | `introspect_examples` | Find usage examples |
| `get_class_hierarchy()` | `introspect_class_hierarchy` | Get class hierarchy |
| `get_type_hints_detailed()` | `introspect_type_hints` | Get type hints |
| `get_imports()` | `introspect_imports` | Get module imports |
| `get_dependencies()` | `introspect_dependencies` | Get dependencies |
| `get_call_graph()` | `introspect_call_graph` | Get call graph |

## Error Handling

All functions return a dict with `success` field:

```python
result = q("nonexistent.module")
if not result["success"]:
    print(f"Error: {result['error']}")
```

## Performance Notes

- Most functions are fast (<100ms)
- `get_call_graph` can be slow for large modules (use `timeout_seconds`)
- `get_dependencies(recursive=True)` can timeout (use `max_depth`)
- `get_imports(categorize=True)` loads stdlib module list once
