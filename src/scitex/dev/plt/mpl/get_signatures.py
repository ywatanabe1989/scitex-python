#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-21 13:21:53 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/mpl/get_signatures.py


"""Inspect matplotlib Axes plotting functions to extract args/kwargs signatures."""

import scitex as stx
import inspect
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


def get_typehint_str(annotation) -> str | None:
    """Convert annotation to string representation."""
    if annotation is inspect.Parameter.empty:
        return None
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def get_default_value(default) -> Any:
    """Convert default value to serializable form."""
    if default is inspect.Parameter.empty:
        return None
    if callable(default):
        return f"<{type(default).__name__}>"
    if isinstance(default, type):
        return f"<class {default.__name__}>"
    try:
        json.dumps(default)
        return default
    except (TypeError, ValueError):
        return repr(default)


def inspect_function(func) -> list[dict[str, Any]]:
    """Inspect a function and return flat list of parameters."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return [{"error": "Cannot inspect signature"}]

    params = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Determine type based on parameter kind
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            params.append({
                "name": f"*{name}",
                "type": "*args",
            })
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            params.append({
                "name": f"**{name}",
                "type": "**kwargs",
            })
        else:
            # Get typehint (None if not available)
            typehint = get_typehint_str(param.annotation)
            info = {
                "name": name,
                "type": typehint,  # None if no typehint
            }

            # Add default if present
            if param.default is not inspect.Parameter.empty:
                info["default"] = get_default_value(param.default)

            params.append(info)

    return params


def inspect_mpl_plotting_functions(
    yaml_path: str | Path | None = None,
) -> dict[str, dict]:
    """Inspect all matplotlib plotting functions from YAML config."""
    if yaml_path is None:
        yaml_path = "./data/dev/plt/mpl/PLOTTING_FUNCTIONS.yaml"

    with open(yaml_path) as f:
        categories = yaml.safe_load(f)

    _, ax = plt.subplots()
    plt.close()

    results = {}

    for category, functions in categories.items():
        if not isinstance(functions, list):
            continue

        results[category] = {}

        for func_name in functions:
            if isinstance(func_name, str):
                func_name = func_name.split("#")[0].strip()

            if hasattr(ax, func_name):
                func = getattr(ax, func_name)
                results[category][func_name] = inspect_function(func)
            else:
                results[category][func_name] = {"error": "Function not found"}

    return results


def format_signature(func_name: str, params: list[dict]) -> str:
    """Format function signature as a readable string."""
    if params and "error" in params[0]:
        return f"{func_name}(): {params[0]['error']}"

    parts = []
    for p in params:
        name = p["name"]
        ptype = p.get("type")

        if ptype in ("*args", "**kwargs"):
            parts.append(name)
        elif "default" in p:
            default = p["default"]
            if isinstance(default, str) and not default.startswith("<"):
                parts.append(f'{name}="{default}"')
            else:
                parts.append(f"{name}={default}")
        else:
            parts.append(name)

    return f"{func_name}({', '.join(parts)})"


@stx.session
def main(
    # arg1,
    # kwarg1="value1",
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Inspect and display matplotlib plotting function signatures."""
    results = inspect_mpl_plotting_functions()

    stx.io.save(
        results,
        "./PLOTTING_SIGNATURES.yaml",
        symlink_to="./data/dev/plt/mpl",
    )

    print("MATPLOTLIB PLOTTING FUNCTION SIGNATURES")
    print("=" * 60)

    for category, functions in results.items():
        print(f"\n## {category.upper()}")
        print("-" * 40)
        for func_name, info in functions.items():
            print(f"  {format_signature(func_name, info)}")

    return 0


if __name__ == "__main__":
    main()

# EOF
