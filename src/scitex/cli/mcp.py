#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/cli/mcp.py
"""
SciTeX MCP CLI - Model Context Protocol server management commands.

Commands:
- scitex mcp list       List all available MCP tools
- scitex mcp doctor     Check MCP server health and configuration
- scitex mcp serve      Start the unified MCP server
"""

import click


@click.group()
def mcp():
    """
    MCP (Model Context Protocol) server management.

    \b
    Examples:
      scitex mcp list                    # List all tools
      scitex mcp list --module audio     # List audio tools only
      scitex mcp doctor                  # Check server health
      scitex mcp serve                   # Start stdio server
      scitex mcp serve -t sse -p 8085    # Start SSE server
    """
    pass


@mcp.command("list")
@click.option(
    "--module",
    "-m",
    type=str,
    default=None,
    help="Filter by module (audio, canvas, capture, diagram, plt, scholar, stats, template, ui, writer)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_tools(module: str, as_json: bool):
    """
    List all available MCP tools.

    \b
    Examples:
      scitex mcp list                    # List all 106 tools
      scitex mcp list --module audio     # List audio tools only
      scitex mcp list --json             # Output as JSON
    """
    try:
        from scitex.mcp_server import FASTMCP_AVAILABLE
        from scitex.mcp_server import mcp as mcp_server
    except ImportError:
        click.secho("ERROR: Could not import MCP server", fg="red", err=True)
        raise SystemExit(1)

    if not FASTMCP_AVAILABLE:
        click.secho(
            "ERROR: FastMCP not installed. Run: pip install fastmcp", fg="red", err=True
        )
        raise SystemExit(1)

    if mcp_server is None:
        click.secho("ERROR: MCP server not initialized", fg="red", err=True)
        raise SystemExit(1)

    # Get all tools
    tools = list(mcp_server._tool_manager._tools.keys())

    # Group by module
    modules = {}
    for tool in sorted(tools):
        prefix = tool.split("_")[0]
        if prefix not in modules:
            modules[prefix] = []
        modules[prefix].append(tool)

    # Filter by module if specified
    if module:
        module = module.lower()
        if module not in modules:
            click.secho(f"ERROR: Unknown module '{module}'", fg="red", err=True)
            click.echo(f"Available modules: {', '.join(sorted(modules.keys()))}")
            raise SystemExit(1)
        modules = {module: modules[module]}

    if as_json:
        import json

        output = {
            "total": sum(len(t) for t in modules.values()),
            "modules": {m: {"count": len(t), "tools": t} for m, t in modules.items()},
        }
        click.echo(json.dumps(output, indent=2))
    else:
        total = sum(len(t) for t in modules.values())
        click.secho(f"SciTeX MCP Tools ({total} total)", fg="cyan", bold=True)
        click.echo()

        for mod, tool_list in sorted(modules.items()):
            # Calculate max tool name length for this module
            max_name_len = max(len(t) for t in tool_list)

            click.secho(f"{mod}: {len(tool_list)} tools", fg="green", bold=True)
            for tool in tool_list:
                # Get tool description if available
                tool_obj = mcp_server._tool_manager._tools.get(tool)
                desc = ""
                if tool_obj and hasattr(tool_obj, "description"):
                    desc = tool_obj.description
                    if desc:
                        # Truncate long descriptions
                        desc = desc.split("\n")[0][:50]
                        if len(desc) == 50:
                            desc += "..."
                # Use printf-style formatting for aligned columns
                click.echo(f"  {tool:<{max_name_len}}  {desc}")
            click.echo()


@mcp.command("doctor")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed diagnostics")
def doctor(verbose: bool):
    """
    Check MCP server health and configuration.

    \b
    Checks:
      - FastMCP package installation
      - Module imports
      - Handler availability
      - Tool registration
    """
    issues = []
    warnings = []

    click.secho("SciTeX MCP Doctor", fg="cyan", bold=True)
    click.echo()

    # Check 1: FastMCP installation
    click.echo("Checking FastMCP installation... ", nl=False)
    try:
        from fastmcp import FastMCP

        click.secho("OK", fg="green")
        if verbose:
            import fastmcp

            click.echo(f"  Version: {getattr(fastmcp, '__version__', 'unknown')}")
    except ImportError:
        click.secho("FAIL", fg="red")
        issues.append("FastMCP not installed. Run: pip install fastmcp")

    # Check 2: MCP server import
    click.echo("Checking MCP server module... ", nl=False)
    try:
        from scitex.mcp_server import FASTMCP_AVAILABLE
        from scitex.mcp_server import mcp as mcp_server

        if FASTMCP_AVAILABLE and mcp_server:
            click.secho("OK", fg="green")
        else:
            click.secho("WARN", fg="yellow")
            warnings.append("MCP server initialized but FastMCP not available")
    except ImportError as e:
        click.secho("FAIL", fg="red")
        issues.append(f"Could not import MCP server: {e}")

    # Check 3: _mcp_tools subpackage
    click.echo("Checking _mcp_tools subpackage... ", nl=False)
    try:
        from scitex._mcp_tools import register_all_tools

        click.secho("OK", fg="green")
    except ImportError as e:
        click.secho("FAIL", fg="red")
        issues.append(f"Could not import _mcp_tools: {e}")

    # Check 4: Individual module imports
    modules = [
        "audio",
        "canvas",
        "capture",
        "diagram",
        "plt",
        "scholar",
        "stats",
        "template",
        "ui",
        "writer",
    ]

    click.echo("Checking module registrations... ", nl=False)
    failed_modules = []
    for mod in modules:
        try:
            exec(f"from scitex._mcp_tools.{mod} import register_{mod}_tools")
        except ImportError as e:
            failed_modules.append((mod, str(e)))

    if failed_modules:
        click.secho(f"FAIL ({len(failed_modules)} modules)", fg="red")
        for mod, err in failed_modules:
            issues.append(f"Module {mod}: {err}")
    else:
        click.secho("OK", fg="green")

    # Check 5: Tool count
    click.echo("Checking tool registration... ", nl=False)
    try:
        from scitex.mcp_server import mcp as mcp_server

        if mcp_server:
            tools = list(mcp_server._tool_manager._tools.keys())
            if len(tools) >= 80:
                click.secho(f"OK ({len(tools)} tools)", fg="green")
            else:
                click.secho(f"WARN ({len(tools)} tools, expected 80+)", fg="yellow")
                warnings.append(f"Only {len(tools)} tools registered, expected 80+")
        else:
            click.secho("SKIP", fg="yellow")
    except Exception as e:
        click.secho("FAIL", fg="red")
        issues.append(f"Tool registration check failed: {e}")

    # Check 6: Handler imports (verbose only)
    if verbose:
        click.echo()
        click.secho("Handler Import Checks:", bold=True)
        handler_modules = [
            ("scitex.audio._mcp.handlers", "speak_handler"),
            ("scitex.capture._mcp.handlers", "capture_screenshot_handler"),
            ("scitex.scholar._mcp.handlers", "search_papers_handler"),
            ("scitex.stats._mcp.handlers", "run_test_handler"),
            ("scitex.plt._mcp._handlers_figure", "create_figure_handler"),
            ("scitex.canvas._mcp.handlers", "create_canvas_handler"),
            ("scitex.diagram._mcp.handlers", "create_diagram_handler"),
            ("scitex.template._mcp.handlers", "list_templates_handler"),
            ("scitex.ui._mcp.handlers", "notify_handler"),
            ("scitex.writer._mcp.handlers", "compile_manuscript_handler"),
        ]

        for mod_path, handler_name in handler_modules:
            click.echo(f"  {mod_path}... ", nl=False)
            try:
                mod = __import__(mod_path, fromlist=[handler_name])
                if hasattr(mod, handler_name):
                    click.secho("OK", fg="green")
                else:
                    click.secho(f"WARN (missing {handler_name})", fg="yellow")
            except ImportError as e:
                click.secho(f"FAIL ({e})", fg="red")

    # Summary
    click.echo()
    if issues:
        click.secho(f"Issues Found: {len(issues)}", fg="red", bold=True)
        for issue in issues:
            click.echo(f"  ✗ {issue}")
    if warnings:
        click.secho(f"Warnings: {len(warnings)}", fg="yellow", bold=True)
        for warning in warnings:
            click.echo(f"  ⚠ {warning}")
    if not issues and not warnings:
        click.secho("All checks passed!", fg="green", bold=True)

    raise SystemExit(1 if issues else 0)


@mcp.command("help-recursive")
@click.pass_context
def help_recursive(ctx):
    """Show help for all MCP commands recursively."""
    # Create a fake parent context to show "scitex mcp" in Usage
    fake_parent = click.Context(click.Group(), info_name="scitex")
    parent_ctx = click.Context(mcp, info_name="mcp", parent=fake_parent)

    click.secho("━━━ scitex mcp ━━━", fg="cyan", bold=True)
    click.echo(mcp.get_help(parent_ctx))

    for name in sorted(mcp.list_commands(ctx) or []):
        cmd = mcp.get_command(ctx, name)
        if cmd is None:
            continue
        click.echo()
        click.secho(f"━━━ scitex mcp {name} ━━━", fg="cyan", bold=True)
        with click.Context(cmd, info_name=name, parent=parent_ctx) as sub_ctx:
            click.echo(cmd.get_help(sub_ctx))


@mcp.command("serve")
@click.option(
    "--transport",
    "-t",
    type=click.Choice(["stdio", "sse", "http"]),
    default="stdio",
    help="Transport type (default: stdio)",
)
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
@click.option(
    "--port", "-p", default=8085, type=int, help="Port to bind (default: 8085)"
)
def serve(transport: str, host: str, port: int):
    """
    Start the unified MCP server.

    \b
    Transport types:
      stdio  - Standard I/O (for Claude Desktop)
      sse    - Server-Sent Events (for remote via SSH)
      http   - HTTP streamable (for web clients)

    \b
    Examples:
      scitex mcp serve                      # Start stdio server
      scitex mcp serve -t sse -p 8085       # Start SSE server
      scitex mcp serve -t http -p 8085      # Start HTTP server

    \b
    Remote Setup (SSE):
      1. Local:  scitex mcp serve -t sse -p 8085
      2. SSH:    ssh -R 8085:localhost:8085 remote-host
      3. Config: {"type": "sse", "url": "http://localhost:8085/sse"}
    """
    try:
        from scitex.mcp_server import run_server
    except ImportError:
        click.secho("ERROR: Could not import MCP server", fg="red", err=True)
        click.echo("Run: pip install fastmcp")
        raise SystemExit(1)

    run_server(transport=transport, host=host, port=port)


# EOF
