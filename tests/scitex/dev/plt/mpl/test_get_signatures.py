#!/usr/bin/env python3
"""Tests for scitex.dev.plt.mpl.get_signatures module."""

import inspect
import os

import pytest

from scitex.dev.plt.mpl.get_signatures import (
    format_signature,
    get_default_value,
    get_typehint_str,
    inspect_function,
)


class TestGetTypehintStr:
    """Tests for get_typehint_str function."""

    def test_empty_annotation_returns_none(self):
        """Test that empty annotation returns None."""
        result = get_typehint_str(inspect.Parameter.empty)
        assert result is None

    def test_named_type_returns_name(self):
        """Test that types with __name__ return that name."""
        result = get_typehint_str(int)
        assert result == "int"

    def test_string_annotation_returns_str(self):
        """Test that string annotations work."""
        result = get_typehint_str("str")
        assert result == "str"


class TestGetDefaultValue:
    """Tests for get_default_value function."""

    def test_empty_default_returns_none(self):
        """Test that empty parameter returns None."""
        result = get_default_value(inspect.Parameter.empty)
        assert result is None

    def test_callable_returns_type_name(self):
        """Test that callables return their type name."""
        result = get_default_value(lambda: None)
        assert "<function>" in result

    def test_type_returns_class_name(self):
        """Test that types return class name."""
        result = get_default_value(str)
        # The function returns the type name wrapped in angle brackets
        assert "<" in result and ">" in result

    def test_serializable_value_returned_as_is(self):
        """Test that JSON-serializable values are returned as-is."""
        assert get_default_value(42) == 42
        assert get_default_value("hello") == "hello"
        assert get_default_value(None) is None
        assert get_default_value([1, 2, 3]) == [1, 2, 3]

    def test_non_serializable_value_returns_repr(self):
        """Test that non-serializable values return repr."""
        result = get_default_value(object())
        assert "object" in result.lower()


class TestInspectFunction:
    """Tests for inspect_function function."""

    def test_simple_function(self):
        """Test inspecting a simple function."""

        def simple(a, b):
            pass

        result = inspect_function(simple)
        assert len(result) == 2
        assert result[0]["name"] == "a"
        assert result[1]["name"] == "b"

    def test_function_with_defaults(self):
        """Test inspecting a function with default values."""

        def with_defaults(a, b=10, c="hello"):
            pass

        result = inspect_function(with_defaults)
        assert len(result) == 3
        assert result[1]["default"] == 10
        assert result[2]["default"] == "hello"

    def test_function_with_args_kwargs(self):
        """Test inspecting a function with *args and **kwargs."""

        def with_varargs(*args, **kwargs):
            pass

        result = inspect_function(with_varargs)
        assert any(p["type"] == "*args" for p in result)
        assert any(p["type"] == "**kwargs" for p in result)

    def test_function_with_annotations(self):
        """Test inspecting a function with type annotations."""

        def annotated(a: int, b: str) -> bool:
            pass

        result = inspect_function(annotated)
        assert result[0]["type"] == "int"
        assert result[1]["type"] == "str"

    def test_skips_self_parameter(self):
        """Test that self parameter is skipped."""

        class MyClass:
            def method(self, a, b):
                pass

        result = inspect_function(MyClass.method)
        # Should have a, b but not self
        names = [p["name"] for p in result]
        assert "self" not in names
        assert "a" in names
        assert "b" in names


class TestFormatSignature:
    """Tests for format_signature function."""

    def test_simple_signature(self):
        """Test formatting a simple signature."""
        params = [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "str"},
        ]
        result = format_signature("my_func", params)
        assert result == "my_func(x, y)"

    def test_signature_with_defaults(self):
        """Test formatting a signature with default values."""
        params = [
            {"name": "x", "type": "int"},
            {"name": "y", "type": "str", "default": "hello"},
        ]
        result = format_signature("my_func", params)
        assert 'y="hello"' in result

    def test_signature_with_args_kwargs(self):
        """Test formatting a signature with *args and **kwargs."""
        params = [
            {"name": "*args", "type": "*args"},
            {"name": "**kwargs", "type": "**kwargs"},
        ]
        result = format_signature("my_func", params)
        assert "*args" in result
        assert "**kwargs" in result

    def test_error_signature(self):
        """Test formatting when there's an error."""
        params = [{"error": "Cannot inspect signature"}]
        result = format_signature("my_func", params)
        assert "Cannot inspect signature" in result

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/mpl/get_signatures.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-21 13:21:53 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/mpl/get_signatures.py
# 
# 
# """Inspect matplotlib Axes plotting functions to extract args/kwargs signatures."""
# 
# import scitex as stx
# import inspect
# import json
# from pathlib import Path
# from typing import Any
# 
# import matplotlib.pyplot as plt
# import yaml
# 
# 
# def get_typehint_str(annotation) -> str | None:
#     """Convert annotation to string representation."""
#     if annotation is inspect.Parameter.empty:
#         return None
#     if hasattr(annotation, "__name__"):
#         return annotation.__name__
#     return str(annotation)
# 
# 
# def get_default_value(default) -> Any:
#     """Convert default value to serializable form."""
#     if default is inspect.Parameter.empty:
#         return None
#     if callable(default):
#         return f"<{type(default).__name__}>"
#     if isinstance(default, type):
#         return f"<class {default.__name__}>"
#     try:
#         json.dumps(default)
#         return default
#     except (TypeError, ValueError):
#         return repr(default)
# 
# 
# def inspect_function(func) -> list[dict[str, Any]]:
#     """Inspect a function and return flat list of parameters."""
#     try:
#         sig = inspect.signature(func)
#     except (ValueError, TypeError):
#         return [{"error": "Cannot inspect signature"}]
# 
#     params = []
# 
#     for name, param in sig.parameters.items():
#         if name == "self":
#             continue
# 
#         # Determine type based on parameter kind
#         if param.kind == inspect.Parameter.VAR_POSITIONAL:
#             params.append({
#                 "name": f"*{name}",
#                 "type": "*args",
#             })
#         elif param.kind == inspect.Parameter.VAR_KEYWORD:
#             params.append({
#                 "name": f"**{name}",
#                 "type": "**kwargs",
#             })
#         else:
#             # Get typehint (None if not available)
#             typehint = get_typehint_str(param.annotation)
#             info = {
#                 "name": name,
#                 "type": typehint,  # None if no typehint
#             }
# 
#             # Add default if present
#             if param.default is not inspect.Parameter.empty:
#                 info["default"] = get_default_value(param.default)
# 
#             params.append(info)
# 
#     return params
# 
# 
# def inspect_mpl_plotting_functions(
#     yaml_path: str | Path | None = None,
# ) -> dict[str, dict]:
#     """Inspect all matplotlib plotting functions from YAML config."""
#     if yaml_path is None:
#         yaml_path = "./data/dev/plt/mpl/PLOTTING_FUNCTIONS.yaml"
# 
#     with open(yaml_path) as f:
#         categories = yaml.safe_load(f)
# 
#     _, ax = plt.subplots()
#     plt.close()
# 
#     results = {}
# 
#     for category, functions in categories.items():
#         if not isinstance(functions, list):
#             continue
# 
#         results[category] = {}
# 
#         for func_name in functions:
#             if isinstance(func_name, str):
#                 func_name = func_name.split("#")[0].strip()
# 
#             if hasattr(ax, func_name):
#                 func = getattr(ax, func_name)
#                 results[category][func_name] = inspect_function(func)
#             else:
#                 results[category][func_name] = {"error": "Function not found"}
# 
#     return results
# 
# 
# def format_signature(func_name: str, params: list[dict]) -> str:
#     """Format function signature as a readable string."""
#     if params and "error" in params[0]:
#         return f"{func_name}(): {params[0]['error']}"
# 
#     parts = []
#     for p in params:
#         name = p["name"]
#         ptype = p.get("type")
# 
#         if ptype in ("*args", "**kwargs"):
#             parts.append(name)
#         elif "default" in p:
#             default = p["default"]
#             if isinstance(default, str) and not default.startswith("<"):
#                 parts.append(f'{name}="{default}"')
#             else:
#                 parts.append(f"{name}={default}")
#         else:
#             parts.append(name)
# 
#     return f"{func_name}({', '.join(parts)})"
# 
# 
# @stx.session
# def main(
#     # arg1,
#     # kwarg1="value1",
#     CONFIG=stx.INJECTED,
#     plt=stx.INJECTED,
#     COLORS=stx.INJECTED,
#     rng=stx.INJECTED,
#     logger=stx.INJECTED,
# ):
#     """Inspect and display matplotlib plotting function signatures."""
#     results = inspect_mpl_plotting_functions()
# 
#     stx.io.save(
#         results,
#         "./PLOTTING_SIGNATURES.yaml",
#         symlink_to="./data/dev/plt/mpl",
#     )
# 
#     print("MATPLOTLIB PLOTTING FUNCTION SIGNATURES")
#     print("=" * 60)
# 
#     for category, functions in results.items():
#         print(f"\n## {category.upper()}")
#         print("-" * 40)
#         for func_name, info in functions.items():
#             print(f"  {format_signature(func_name, info)}")
# 
#     return 0
# 
# 
# if __name__ == "__main__":
#     main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/mpl/get_signatures.py
# --------------------------------------------------------------------------------
