#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-21 13:43:59 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/mpl/get_signatures_details.py


"""Extract signatures with *args/**kwargs flattened from matplotlib plotting functions."""

import inspect
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

import scitex as stx


def parse_parameter_types(docstring: str | None) -> dict[str, str]:
    """Extract parameter types from NumPy-style docstring Parameters section."""
    if not docstring:
        return {}

    types = {}

    # Find Parameters section
    params_match = re.search(
        r'Parameters\s*[-]+\s*(.*?)(?:\n\s*Returns|\n\s*See Also|\n\s*Notes|\n\s*Examples|\n\s*Other Parameters|\Z)',
        docstring,
        re.DOTALL
    )
    if not params_match:
        return {}

    params_text = params_match.group(1)

    # Parse lines like "x, y : array-like or float" or "fmt : str, optional"
    for match in re.finditer(r'^(\w+(?:\s*,\s*\w+)*)\s*:\s*(.+?)(?=\n\s*\n|\n\w+\s*:|\Z)', params_text, re.MULTILINE | re.DOTALL):
        names_str = match.group(1)
        type_str = match.group(2).split('\n')[0].strip()  # First line only

        # Clean up type string
        type_str = re.sub(r',?\s*optional\s*$', '', type_str).strip()
        type_str = re.sub(r',?\s*default[^,]*$', '', type_str, flags=re.IGNORECASE).strip()

        # Handle multiple names like "x, y"
        for name in re.split(r'\s*,\s*', names_str):
            name = name.strip()
            if name:
                types[name.lower()] = type_str

    return types


def parse_args_pattern(args_str: str, param_types: dict[str, str]) -> list[dict[str, Any]]:
    """Parse args pattern like '[x], y, [fmt]' into list of arg dicts."""
    if not args_str:
        return []

    args = []
    # Split by comma, handling brackets
    parts = re.split(r',\s*', args_str)

    for part in parts:
        part = part.strip()
        if not part or part == '/':  # Skip empty or positional-only marker
            continue

        # Check if optional (wrapped in [])
        optional = part.startswith('[') and part.endswith(']')
        if optional:
            name = part[1:-1].strip()
        else:
            # Handle cases like "[X, Y,] Z" where Z is required
            name = part.strip('[]').strip()

        if name and name not in ('...', '*'):
            # Look up type from parsed parameters
            type_str = param_types.get(name.lower())
            args.append({
                "name": name,
                "type": type_str,
                "optional": optional,
            })

    return args


# Manual *args patterns for functions without parseable call signatures
MANUAL_ARGS_PATTERNS = {
    "fill": "[x], y, [color]",
    "stackplot": "x, *ys",
    "legend": "[handles], [labels]",
    "stem": "[locs], heads",
    "tricontour": "[triangulation], x, y, z, [levels]",
    "tricontourf": "[triangulation], x, y, z, [levels]",
    "triplot": "[triangulation], x, y, [triangles]",
    "loglog": "[x], y, [fmt]",
    "semilogx": "[x], y, [fmt]",
    "semilogy": "[x], y, [fmt]",
    "barbs": "[X], [Y], U, V, [C]",
    "quiver": "[X], [Y], U, V, [C]",
    "pcolor": "[X], [Y], C",
    "pcolormesh": "[X], [Y], C",
    "pcolorfast": "[X], [Y], C",
    "acorr": "x",
    "xcorr": "x, y",
}


def extract_args_from_docstring(docstring: str | None, func_name: str = "") -> list[dict[str, Any]]:
    """Extract *args as flattened list from docstring call signature."""
    if not docstring:
        return []

    # First, parse parameter types
    param_types = parse_parameter_types(docstring)

    # Check for manual pattern first
    if func_name in MANUAL_ARGS_PATTERNS:
        return parse_args_pattern(MANUAL_ARGS_PATTERNS[func_name], param_types)

    # Look for "Call signature:" patterns
    patterns = [
        r'Call signatures?::\s*\n\s*(.*?)(?:\n\n|\n[A-Z])',
        r'^\s*(\w+\([^)]+\))\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, docstring, re.MULTILINE | re.DOTALL)
        if match:
            sig_text = match.group(1).strip()
            # Extract first signature line
            first_line = sig_text.split('\n')[0].strip()
            # Parse the args from signature like "plot([x], y, [fmt], *, ...)"
            inner_match = re.search(r'\(([^*]+?)(?:,\s*\*|,\s*data|\))', first_line)
            if inner_match:
                args_str = inner_match.group(1).strip().rstrip(',')
                return parse_args_pattern(args_str, param_types)
    return []

def get_setter_type(obj, prop_name: str) -> str | None:
    """Get type from set_* method docstring."""
    setter_name = f"set_{prop_name}"
    if not hasattr(obj, setter_name):
        return None

    method = getattr(obj, setter_name)
    if not method.__doc__:
        return None

    # Parse Parameters section
    match = re.search(
        r'Parameters\s*[-]+\s*\n\s*(\w+)\s*:\s*(.+?)(?:\n\s*\n|\Z)',
        method.__doc__,
        re.DOTALL
    )
    if match:
        type_str = match.group(2).split('\n')[0].strip()
        return type_str
    return None


def build_kwargs_with_types():
    """Build kwargs lists with types from matplotlib classes."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    from matplotlib.artist import Artist

    # Create instances for introspection
    line = Line2D([0], [0])
    patch = Patch()
    text = Text()
    artist = Artist()

    def get_type(obj, name):
        return get_setter_type(obj, name)

    ARTIST_KWARGS = [
        {"name": "agg_filter", "type": get_type(artist, "agg_filter"), "default": None},
        {"name": "alpha", "type": get_type(artist, "alpha"), "default": None},
        {"name": "animated", "type": get_type(artist, "animated"), "default": False},
        {"name": "clip_box", "type": get_type(artist, "clip_box"), "default": None},
        {"name": "clip_on", "type": get_type(artist, "clip_on"), "default": True},
        {"name": "clip_path", "type": get_type(artist, "clip_path"), "default": None},
        {"name": "gid", "type": get_type(artist, "gid"), "default": None},
        {"name": "label", "type": get_type(artist, "label"), "default": ""},
        {"name": "path_effects", "type": get_type(artist, "path_effects"), "default": None},
        {"name": "picker", "type": get_type(artist, "picker"), "default": None},
        {"name": "rasterized", "type": get_type(artist, "rasterized"), "default": None},
        {"name": "sketch_params", "type": get_type(artist, "sketch_params"), "default": None},
        {"name": "snap", "type": get_type(artist, "snap"), "default": None},
        {"name": "transform", "type": get_type(artist, "transform"), "default": None},
        {"name": "url", "type": get_type(artist, "url"), "default": None},
        {"name": "visible", "type": get_type(artist, "visible"), "default": True},
        {"name": "zorder", "type": get_type(artist, "zorder"), "default": None},
    ]

    LINE2D_KWARGS = [
        {"name": "color", "type": get_type(line, "color"), "default": None},
        {"name": "linestyle", "type": get_type(line, "linestyle"), "default": "-"},
        {"name": "linewidth", "type": get_type(line, "linewidth"), "default": None},
        {"name": "marker", "type": get_type(line, "marker"), "default": ""},
        {"name": "markeredgecolor", "type": get_type(line, "markeredgecolor"), "default": None},
        {"name": "markeredgewidth", "type": get_type(line, "markeredgewidth"), "default": None},
        {"name": "markerfacecolor", "type": get_type(line, "markerfacecolor"), "default": None},
        {"name": "markersize", "type": get_type(line, "markersize"), "default": None},
        {"name": "antialiased", "type": get_type(line, "antialiased"), "default": True},
        {"name": "dash_capstyle", "type": get_type(line, "dash_capstyle"), "default": "butt"},
        {"name": "dash_joinstyle", "type": get_type(line, "dash_joinstyle"), "default": "round"},
        {"name": "solid_capstyle", "type": get_type(line, "solid_capstyle"), "default": "projecting"},
        {"name": "solid_joinstyle", "type": get_type(line, "solid_joinstyle"), "default": "round"},
        {"name": "drawstyle", "type": get_type(line, "drawstyle"), "default": "default"},
        {"name": "fillstyle", "type": get_type(line, "fillstyle"), "default": "full"},
    ]

    PATCH_KWARGS = [
        {"name": "color", "type": get_type(patch, "color"), "default": None},
        {"name": "edgecolor", "type": get_type(patch, "edgecolor"), "default": None},
        {"name": "facecolor", "type": get_type(patch, "facecolor"), "default": None},
        {"name": "fill", "type": get_type(patch, "fill"), "default": True},
        {"name": "hatch", "type": get_type(patch, "hatch"), "default": None},
        {"name": "linestyle", "type": get_type(patch, "linestyle"), "default": "-"},
        {"name": "linewidth", "type": get_type(patch, "linewidth"), "default": None},
        {"name": "antialiased", "type": get_type(patch, "antialiased"), "default": None},
        {"name": "capstyle", "type": get_type(patch, "capstyle"), "default": "butt"},
        {"name": "joinstyle", "type": get_type(patch, "joinstyle"), "default": "miter"},
    ]

    TEXT_KWARGS = [
        {"name": "color", "type": get_type(text, "color"), "default": "black"},
        {"name": "fontfamily", "type": get_type(text, "fontfamily"), "default": None},
        {"name": "fontsize", "type": get_type(text, "fontsize"), "default": None},
        {"name": "fontstretch", "type": get_type(text, "fontstretch"), "default": None},
        {"name": "fontstyle", "type": get_type(text, "fontstyle"), "default": "normal"},
        {"name": "fontvariant", "type": get_type(text, "fontvariant"), "default": "normal"},
        {"name": "fontweight", "type": get_type(text, "fontweight"), "default": "normal"},
        {"name": "horizontalalignment", "type": get_type(text, "horizontalalignment"), "default": "center"},
        {"name": "verticalalignment", "type": get_type(text, "verticalalignment"), "default": "center"},
        {"name": "rotation", "type": get_type(text, "rotation"), "default": None},
        {"name": "linespacing", "type": get_type(text, "linespacing"), "default": None},
        {"name": "multialignment", "type": get_type(text, "multialignment"), "default": None},
        {"name": "wrap", "type": get_type(text, "wrap"), "default": False},
    ]

    return ARTIST_KWARGS, LINE2D_KWARGS, PATCH_KWARGS, TEXT_KWARGS


# Build kwargs with types at module load
ARTIST_KWARGS, LINE2D_KWARGS, PATCH_KWARGS, TEXT_KWARGS = build_kwargs_with_types()

# Mapping of functions to their **kwargs type
KWARGS_MAPPING = {
    "plot": LINE2D_KWARGS + ARTIST_KWARGS,
    "scatter": ARTIST_KWARGS,
    "bar": PATCH_KWARGS + ARTIST_KWARGS,
    "barh": PATCH_KWARGS + ARTIST_KWARGS,
    "fill": PATCH_KWARGS + ARTIST_KWARGS,
    "fill_between": PATCH_KWARGS + ARTIST_KWARGS,
    "fill_betweenx": PATCH_KWARGS + ARTIST_KWARGS,
    "step": LINE2D_KWARGS + ARTIST_KWARGS,
    "errorbar": LINE2D_KWARGS + ARTIST_KWARGS,
    "hist": PATCH_KWARGS + ARTIST_KWARGS,
    "hist2d": ARTIST_KWARGS,
    "imshow": ARTIST_KWARGS,
    "pcolor": ARTIST_KWARGS,
    "pcolormesh": ARTIST_KWARGS,
    "pcolorfast": ARTIST_KWARGS,
    "contour": ARTIST_KWARGS,
    "contourf": ARTIST_KWARGS,
    "hexbin": ARTIST_KWARGS,
    "quiver": ARTIST_KWARGS,
    "barbs": ARTIST_KWARGS,
    "specgram": ARTIST_KWARGS,
    "psd": LINE2D_KWARGS + ARTIST_KWARGS,
    "csd": LINE2D_KWARGS + ARTIST_KWARGS,
    "cohere": LINE2D_KWARGS + ARTIST_KWARGS,
    "acorr": LINE2D_KWARGS + ARTIST_KWARGS,
    "xcorr": LINE2D_KWARGS + ARTIST_KWARGS,
    "angle_spectrum": LINE2D_KWARGS + ARTIST_KWARGS,
    "magnitude_spectrum": LINE2D_KWARGS + ARTIST_KWARGS,
    "phase_spectrum": LINE2D_KWARGS + ARTIST_KWARGS,
    "stackplot": PATCH_KWARGS + ARTIST_KWARGS,
    "stairs": PATCH_KWARGS + ARTIST_KWARGS,
    "eventplot": ARTIST_KWARGS,
    "broken_barh": PATCH_KWARGS + ARTIST_KWARGS,
    "loglog": LINE2D_KWARGS + ARTIST_KWARGS,
    "semilogx": LINE2D_KWARGS + ARTIST_KWARGS,
    "semilogy": LINE2D_KWARGS + ARTIST_KWARGS,
    "annotate": TEXT_KWARGS + ARTIST_KWARGS,
    "text": TEXT_KWARGS + ARTIST_KWARGS,
    "arrow": PATCH_KWARGS + ARTIST_KWARGS,
    "axhline": LINE2D_KWARGS + ARTIST_KWARGS,
    "axvline": LINE2D_KWARGS + ARTIST_KWARGS,
    "hlines": ARTIST_KWARGS,
    "vlines": ARTIST_KWARGS,
    "axhspan": PATCH_KWARGS + ARTIST_KWARGS,
    "axvspan": PATCH_KWARGS + ARTIST_KWARGS,
    "axline": LINE2D_KWARGS + ARTIST_KWARGS,
    "legend": ARTIST_KWARGS,
    "grid": LINE2D_KWARGS + ARTIST_KWARGS,
    "table": ARTIST_KWARGS,
    "clabel": TEXT_KWARGS + ARTIST_KWARGS,
    "bar_label": TEXT_KWARGS + ARTIST_KWARGS,
    "quiverkey": ARTIST_KWARGS,
    "ecdf": LINE2D_KWARGS + ARTIST_KWARGS,
    "tricontour": ARTIST_KWARGS,
    "tricontourf": ARTIST_KWARGS,
    "tripcolor": ARTIST_KWARGS,
    "triplot": LINE2D_KWARGS + ARTIST_KWARGS,
    "matshow": ARTIST_KWARGS,
    "spy": ARTIST_KWARGS + LINE2D_KWARGS,
}


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


def inspect_function_flattened(func, func_name: str) -> dict[str, list[dict[str, Any]]]:
    """Inspect function and flatten *args/**kwargs into args/kwargs structure."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return {"error": "Cannot inspect signature"}

    # Parse parameter types from docstring
    param_types = parse_parameter_types(func.__doc__)

    args = []
    kwargs = []
    has_var_positional = False
    has_var_keyword = False

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        else:
            # Try annotation first, then docstring
            typehint = get_typehint_str(param.annotation)
            if not typehint:
                typehint = param_types.get(name.lower())

            info = {
                "name": name,
                "type": typehint,
            }

            if param.default is not inspect.Parameter.empty:
                info["default"] = get_default_value(param.default)
                kwargs.append(info)
            else:
                args.append(info)

    # Try to extract *args info from docstring and flatten
    if has_var_positional:
        docstring_args = extract_args_from_docstring(func.__doc__, func_name)
        if docstring_args:
            # Insert flattened args at the beginning
            for i, arg in enumerate(docstring_args):
                args.insert(i, arg)
        else:
            # No docstring info, keep generic *args
            args.insert(0, {"name": "*args", "type": None})

    # Expand **kwargs based on function type
    if has_var_keyword and func_name in KWARGS_MAPPING:
        expanded_kwargs = KWARGS_MAPPING[func_name]
        existing_names = {p["name"] for p in args + kwargs}
        for kwarg in expanded_kwargs:
            if kwarg["name"] not in existing_names:
                kwargs.append(
                    {
                        "name": kwarg["name"],
                        "type": kwarg["type"],
                        "default": kwarg["default"],
                    }
                )
    elif has_var_keyword:
        kwargs.append(
            {
                "name": "**kwargs",
                "type": None,
            }
        )

    return {"args": args, "kwargs": kwargs}


def inspect_mpl_plotting_functions_flattened(
    yaml_path: str | Path | None = None,
) -> dict[str, dict]:
    """Inspect matplotlib plotting functions with flattened *args/**kwargs."""
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
                results[category][func_name] = inspect_function_flattened(
                    func, func_name
                )
            else:
                results[category][func_name] = {
                    "error": "Function not found"
                }

    return results


def format_signature(func_name: str, info: dict) -> str:
    """Format function signature as a readable string."""
    if "error" in info:
        return f"{func_name}(): {info['error']}"

    parts = []

    # Process args
    for p in info.get("args", []):
        name = p["name"]
        ptype = p.get("type")
        if ptype == "*args":
            parts.append(name)
        else:
            parts.append(name)

    # Process kwargs
    for p in info.get("kwargs", []):
        name = p["name"]
        ptype = p.get("type")
        if ptype == "**kwargs":
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
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Extract signatures with flattened *args/**kwargs."""
    results = inspect_mpl_plotting_functions_flattened()

    stx.io.save(
        results,
        "./PLOTTING_SIGNATURES_DETAILED.yaml",
        symlink_to="./data/dev/plt/mpl",
    )

    print("MATPLOTLIB PLOTTING FUNCTION SIGNATURES (DETAILED)")
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
