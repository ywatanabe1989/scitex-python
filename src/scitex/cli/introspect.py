#!/usr/bin/env python3
"""
SciTeX CLI - Introspection Commands

Provides IPython-like introspection for Python packages.
"""

import json
import sys

import click


def _normalize_path(ctx, param, value):
    """Normalize dotted path: convert hyphens to underscores for Python module names."""
    if value:
        return value.replace("-", "_")
    return value


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("--help-recursive", is_flag=True, help="Show help for all subcommands")
@click.pass_context
def introspect(ctx, help_recursive):
    """
    Python package introspection utilities

    \b
    IPython-like introspection for any Python package:
      q         - Function/class signature (like func?)
      qq        - Full source code (like func??)
      dir       - List module/class members (like dir())
      api       - Full module API tree
      docstring - Extract docstrings
      exports   - Show __all__ exports
      examples  - Find usage examples

    \b
    Examples:
      scitex introspect q scitex.plt.plot
      scitex introspect qq scitex.stats.run_test --max-lines 50
      scitex introspect dir scitex.plt --kind functions
      scitex introspect api scitex --max-depth 2
    """
    if help_recursive:
        from . import print_help_recursive

        print_help_recursive(ctx, introspect)
        ctx.exit(0)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@introspect.command()
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--no-defaults", is_flag=True, help="Exclude default values")
@click.option("--no-annotations", is_flag=True, help="Exclude type annotations")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def q(dotted_path, no_defaults, no_annotations, as_json):
    """
    Get function/class signature (like IPython's func?)

    \b
    Examples:
      scitex introspect q scitex.plt.plot
      scitex introspect q scitex.audio.speak --json
      scitex introspect q json.dumps
    """
    from scitex.introspect import q as get_q

    result = get_q(
        dotted_path,
        include_defaults=not no_defaults,
        include_annotations=not no_annotations,
    )

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(result["signature"], fg="green", bold=True)
        if result.get("parameters"):
            click.echo("\nParameters:")
            for p in result["parameters"]:
                line = f"  {p['name']}"
                if "annotation" in p:
                    line += f": {p['annotation']}"
                if "default" in p:
                    line += f" = {p['default']}"
                click.echo(line)


@introspect.command()
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--max-lines", "-n", type=int, help="Limit output to N lines")
@click.option("--no-decorators", is_flag=True, help="Exclude decorator lines")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def qq(dotted_path, max_lines, no_decorators, as_json):
    """
    Get source code of a Python object (like IPython's func??)

    \b
    Examples:
      scitex introspect qq scitex.plt.plot
      scitex introspect qq scitex.audio.speak --max-lines 50
    """
    from scitex.introspect import qq as get_qq

    result = get_qq(
        dotted_path,
        max_lines=max_lines,
        include_decorators=not no_decorators,
    )

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"# File: {result['file']}:{result['line_start']}", fg="cyan")
        click.secho(f"# Lines: {result['line_count']}", fg="cyan")
        click.echo()
        click.echo(result["source"])


@introspect.command("dir")
@click.argument("dotted_path", callback=_normalize_path)
@click.option(
    "--filter",
    "-f",
    type=click.Choice(["all", "public", "private", "dunder"]),
    default="public",
    help="Filter members",
)
@click.option(
    "--kind",
    "-k",
    type=click.Choice(["all", "functions", "classes", "data", "modules"]),
    help="Filter by type",
)
@click.option("--inherited", is_flag=True, help="Include inherited members (classes)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def dir_cmd(dotted_path, filter, kind, inherited, as_json):
    """
    List members of a module or class (like dir())

    \b
    Examples:
      scitex introspect dir scitex.plt
      scitex introspect dir scitex.audio --kind functions
      scitex introspect dir scitex.plt.AxisWrapper --filter all
    """
    from scitex.introspect import dir as get_dir

    result = get_dir(
        dotted_path,
        filter=filter,
        kind=kind,
        include_inherited=inherited,
    )

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"Members of {dotted_path} ({result['count']}):", fg="cyan")
        for m in result["members"]:
            kind_str = click.style(f"[{m['kind']}]", fg="yellow")
            name_str = click.style(m["name"], fg="green", bold=True)
            summary = f" - {m['summary']}" if m["summary"] else ""
            click.echo(f"  {kind_str} {name_str}{summary}")


@introspect.command()
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--max-depth", "-d", type=int, default=5, help="Max recursion depth")
@click.option("--root-only", is_flag=True, help="Show only root-level items")
@click.option("-v", "--verbose", count=True, help="Verbosity: -v +doc, -vv full doc")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def api(dotted_path, max_depth, root_only, verbose, as_json):
    """List API tree with types and signatures. -v adds docstrings, -vv full docs."""
    import importlib

    from scitex.introspect import list_api

    df = list_api(
        dotted_path, max_depth=max_depth, docstring=(verbose >= 1), root_only=root_only
    )

    # Color mapping for types
    type_colors = {"M": "blue", "C": "magenta", "F": "green", "V": "cyan"}

    if as_json:
        click.echo(json.dumps(df.to_dict(orient="records"), indent=2))
    else:
        from . import format_python_signature

        click.secho(f"API tree of {dotted_path} ({len(df)} items):", fg="cyan")
        legend = " ".join(
            click.style(f"[{t}]={n}", fg=type_colors[t])
            for t, n in [
                ("M", "Module"),
                ("C", "Class"),
                ("F", "Function"),
                ("V", "Variable"),
            ]
        )
        click.echo(f"Legend: {legend}")
        # Get base module for signature lookup
        base_parts = dotted_path.split(".")
        for _, row in df.iterrows():
            indent = "  " * row["Depth"]
            t = row["Type"]
            type_s = click.style(f"[{t}]", fg=type_colors.get(t, "yellow"))
            name = row["Name"].split(".")[-1]

            if t == "F":
                try:
                    rel_parts = row["Name"].split(".")[:-1]
                    full_mod = (
                        ".".join(base_parts[:-1] + rel_parts)
                        if len(base_parts) > 1
                        else ".".join(rel_parts)
                    )
                    fn = getattr(importlib.import_module(full_mod), name, None)
                    if fn and callable(fn):
                        name_s, sig_s = format_python_signature(fn, indent=indent)
                        click.echo(f"{indent}{type_s} {name_s}{sig_s}")
                    else:
                        name_s = click.style(name, fg="green", bold=True)
                        click.echo(f"{indent}{type_s} {name_s}")
                except Exception:
                    name_s = click.style(name, fg="green", bold=True)
                    click.echo(f"{indent}{type_s} {name_s}")
            else:
                name_s = click.style(name, fg=type_colors.get(t, "white"), bold=True)
                click.echo(f"{indent}{type_s} {name_s}")

            if verbose >= 1 and row.get("Docstring"):
                if verbose == 1:
                    doc = row["Docstring"].split("\n")[0][:60]
                    click.echo(f"{indent}    - {doc}")
                else:
                    for ln in row["Docstring"].split("\n"):
                        click.echo(f"{indent}    {ln}")


@introspect.command()
@click.argument("dotted_path", callback=_normalize_path)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["raw", "parsed", "summary"]),
    default="raw",
    help="Output format",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def docstring(dotted_path, format, as_json):
    """
    Get docstring of a Python object

    \b
    Formats:
      raw     - Full docstring as-is
      parsed  - Parse into sections (summary, parameters, returns, etc.)
      summary - First line/paragraph only

    \b
    Examples:
      scitex introspect docstring scitex.plt.plot
      scitex introspect docstring scitex.audio.speak --format parsed
    """
    from scitex.introspect import get_docstring

    result = get_docstring(dotted_path, format=format)

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(result["docstring"])
        if format == "parsed" and result.get("sections"):
            click.echo("\n--- Parsed Sections ---")
            for key, value in result["sections"].items():
                if value:
                    click.secho(f"\n[{key}]", fg="cyan", bold=True)
                    click.echo(value)


@introspect.command()
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def exports(dotted_path, as_json):
    """
    Get __all__ exports of a module

    \b
    Examples:
      scitex introspect exports scitex.audio
      scitex introspect exports scitex.plt
    """
    from scitex.introspect import get_exports

    result = get_exports(dotted_path)

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        has_all = "defined" if result["has_all"] else "not defined (showing public)"
        click.secho(f"__all__ is {has_all}", fg="cyan")
        click.secho(f"Exports ({result['count']}):", fg="cyan")
        for name in result["exports"]:
            click.echo(f"  {name}")


@introspect.command()
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--search-paths", "-p", help="Comma-separated search paths")
@click.option("--max-results", "-n", type=int, default=10, help="Max examples")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def examples(dotted_path, search_paths, max_results, as_json):
    """
    Find usage examples in tests/examples directories

    \b
    Examples:
      scitex introspect examples scitex.plt.plot
      scitex introspect examples scitex.audio.speak --max-results 5
    """
    from scitex.introspect import find_examples

    paths_list = None
    if search_paths:
        paths_list = [p.strip() for p in search_paths.split(",")]

    result = find_examples(
        dotted_path,
        search_paths=paths_list,
        max_results=max_results,
    )

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"Found {result['count']} examples:", fg="cyan")
        for ex in result["examples"]:
            click.echo()
            click.secho(f"--- {ex['file']}:{ex['line']} ---", fg="yellow")
            click.echo(ex["context"])


# Advanced introspection commands


@introspect.command("hierarchy")
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--builtins", is_flag=True, help="Include builtin classes")
@click.option("--max-depth", "-d", type=int, default=10, help="Max subclass depth")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def class_hierarchy(dotted_path, builtins, max_depth, as_json):
    """Get class inheritance hierarchy (MRO + subclasses)"""
    from scitex.introspect import get_class_hierarchy

    result = get_class_hierarchy(
        dotted_path, include_builtins=builtins, max_depth=max_depth
    )

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"Class: {dotted_path}", fg="cyan", bold=True)
        click.secho(f"\nMRO ({result['mro_count']} classes):", fg="yellow")
        for cls in result["mro"]:
            click.echo(f"  {cls['qualname']}")
        click.secho(f"\nSubclasses ({result['subclass_count']}):", fg="yellow")
        _print_subclasses(result.get("subclasses", []), indent=2)


def _print_subclasses(subclasses, indent=0):
    """Helper to print subclass tree."""
    for sub in subclasses:
        click.echo(" " * indent + f"- {sub['qualname']}")
        if "subclasses" in sub:
            _print_subclasses(sub["subclasses"], indent + 2)


@introspect.command("hints")
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def type_hints(dotted_path, as_json):
    """Get detailed type hint analysis"""
    from scitex.introspect import get_type_hints_detailed

    result = get_type_hints_detailed(dotted_path)

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"Type hints ({result['hint_count']}):", fg="cyan")
        for name, info in result.get("hints", {}).items():
            opt = " (optional)" if info.get("is_optional") else ""
            click.echo(f"  {name}: {info['raw']}{opt}")
        if result.get("return_hint"):
            click.secho(f"\nReturn: {result['return_hint']['raw']}", fg="green")


@introspect.command("imports")
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--no-categorize", is_flag=True, help="Don't group by category")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def imports(dotted_path, no_categorize, as_json):
    """Get all imports from a module (AST-based)"""
    from scitex.introspect import get_imports

    result = get_imports(dotted_path, categorize=not no_categorize)

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"Imports ({result['import_count']}):", fg="cyan")
        if result.get("categories"):
            for cat, imps in result["categories"].items():
                if imps:
                    click.secho(f"\n  [{cat}] ({len(imps)}):", fg="yellow")
                    for imp in imps:
                        click.echo(f"    {imp['module']}")
        else:
            for imp in result["imports"]:
                click.echo(f"  {imp['module']}")


@introspect.command("deps")
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--recursive", "-r", is_flag=True, help="Recursive analysis")
@click.option("--max-depth", "-d", type=int, default=3, help="Max recursion depth")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def dependencies(dotted_path, recursive, max_depth, as_json):
    """Get module dependencies"""
    from scitex.introspect import get_dependencies

    result = get_dependencies(dotted_path, recursive=recursive, max_depth=max_depth)

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.secho(f"Dependencies ({result['dependency_count']}):", fg="cyan")
        for dep in result.get("dependencies", []):
            click.echo(f"  {dep}")


@introspect.command("calls")
@click.argument("dotted_path", callback=_normalize_path)
@click.option("--timeout", "-t", type=int, default=10, help="Timeout in seconds")
@click.option("--all", "all_calls", is_flag=True, help="Include external calls")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def call_graph(dotted_path, timeout, all_calls, as_json):
    """Get function call graph (with timeout protection)"""
    from scitex.introspect import get_call_graph

    result = get_call_graph(
        dotted_path, timeout_seconds=timeout, internal_only=not all_calls
    )

    if not result.get("success", False):
        click.secho(f"Error: {result.get('error', 'Unknown error')}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        if "calls" in result:
            click.secho(f"Calls ({result['call_count']}):", fg="cyan")
            for call in result["calls"]:
                click.echo(f"  -> {call['name']} (line {call['line']})")
            click.secho(f"\nCalled by ({result['caller_count']}):", fg="yellow")
            for caller in result.get("called_by", []):
                click.echo(f"  <- {caller['name']} (line {caller['line']})")
        elif "graph" in result:
            click.secho(
                f"Module call graph ({result['function_count']} functions):", fg="cyan"
            )
            for func, info in result["graph"].items():
                calls = ", ".join(c["name"] for c in info["calls"][:5])
                if len(info["calls"]) > 5:
                    calls += "..."
                click.echo(f"  {func}: {calls or '(no calls)'}")
