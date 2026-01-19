#!/usr/bin/env python3
"""
SciTeX Writer Commands - LaTeX Manuscript Management

Provides manuscript project initialization and compilation.
"""

import sys
from pathlib import Path

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def writer():
    """
    Manuscript writing and LaTeX compilation

    \b
    Provides manuscript project management:
    - Initialize new manuscript projects
    - Compile manuscript, supplementary, and revision documents
    - Watch mode for auto-recompilation
    - Project structure management
    """
    pass


@writer.command("help-recursive")
@click.pass_context
def help_recursive(ctx):
    """Show help for all commands recursively."""
    fake_parent = click.Context(click.Group(), info_name="scitex")
    parent_ctx = click.Context(writer, info_name="writer", parent=fake_parent)

    click.secho("━━━ scitex writer ━━━", fg="cyan", bold=True)
    click.echo(writer.get_help(parent_ctx))

    for name in sorted(writer.list_commands(ctx) or []):
        cmd = writer.get_command(ctx, name)
        if cmd is None or name == "help-recursive":
            continue
        click.echo()
        click.secho(f"━━━ scitex writer {name} ━━━", fg="cyan", bold=True)
        with click.Context(cmd, info_name=name, parent=parent_ctx) as sub_ctx:
            click.echo(cmd.get_help(sub_ctx))


@writer.command()
@click.argument("project_dir", type=click.Path())
@click.option(
    "--git-strategy",
    "-g",
    type=click.Choice(["child", "parent", "origin", "none"], case_sensitive=False),
    default="child",
    help="Git initialization strategy (default: child)",
)
@click.option("--branch", "-b", help="Specific branch of template to clone")
@click.option("--tag", "-t", help="Specific tag/release of template to clone")
def clone(project_dir, git_strategy, branch, tag):
    """
    Clone a new writer project from template

    \b
    Arguments:
        PROJECT_DIR  Path to project directory (will be created)

    \b
    Git Strategies:
        child   - Create isolated git in project directory (default)
        parent  - Use parent git repository
        origin  - Preserve template's original git history
        none    - Disable git initialization

    \b
    Examples:
        scitex writer clone my_paper
        scitex writer clone ./papers/my_paper
        scitex writer clone my_paper --git-strategy parent
        scitex writer clone my_paper --branch develop
        scitex writer clone my_paper --tag v1.0.0
    """
    try:
        from scitex.writer._clone_writer_project import clone_writer_project

        # Validate mutual exclusivity of branch and tag
        if branch and tag:
            click.echo("Error: Cannot specify both --branch and --tag", err=True)
            sys.exit(1)

        # Convert git_strategy 'none' to None
        if git_strategy and git_strategy.lower() == "none":
            git_strategy = None

        click.echo(f"Cloning writer project: {project_dir}")

        # Clone writer project
        result = clone_writer_project(
            project_dir=project_dir,
            git_strategy=git_strategy,
            branch=branch,
            tag=tag,
        )

        if result:
            project_path = Path(project_dir)
            click.echo()
            click.secho(
                f"✓ Successfully cloned project at {project_path.absolute()}",
                fg="green",
            )
            click.echo()
            click.echo("Project structure:")
            click.echo(f"  {project_dir}/")
            click.echo(
                "    ├── 00_shared/          # Shared resources (figures, bibliography)"
            )
            click.echo("    ├── 01_manuscript/      # Main manuscript")
            click.echo("    ├── 02_supplementary/   # Supplementary materials")
            click.echo("    ├── 03_revision/        # Revision documents")
            click.echo("    └── scripts/            # Compilation scripts")
            click.echo()
            click.echo("Next steps:")
            click.echo(f"  cd {project_dir}")
            click.echo("  # Edit your manuscript in 01_manuscript/contents/")
            click.echo("  scitex writer compile manuscript")
        else:
            click.secho("✗ Failed to clone project", fg="red", err=True)
            sys.exit(1)

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@writer.command()
@click.argument(
    "document",
    type=click.Choice(
        ["manuscript", "supplementary", "revision"], case_sensitive=False
    ),
    default="manuscript",
)
@click.option(
    "--dir",
    "-d",
    type=click.Path(exists=True),
    help="Project directory (defaults to current directory)",
)
@click.option(
    "--track-changes", is_flag=True, help="Enable change tracking (revision only)"
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Compilation timeout in seconds (default: 300)",
)
def compile(document, dir, track_changes, timeout):
    """
    Compile LaTeX document to PDF

    \b
    Arguments:
        DOCUMENT  Document type to compile (manuscript|supplementary|revision)

    \b
    Examples:
        scitex writer compile manuscript
        scitex writer compile manuscript --dir ./my_paper
        scitex writer compile revision --track-changes
        scitex writer compile supplementary --timeout 600
    """
    from scitex.writer import Writer

    project_dir = Path(dir) if dir else Path.cwd()

    # Check if this is a Writer project before trying to attach
    required_dirs = ["01_manuscript", "02_supplementary", "03_revision"]
    missing = [d for d in required_dirs if not (project_dir / d).exists()]

    if missing:
        click.echo(f"Not a Writer project: {project_dir.absolute()}")
        click.echo(f"Missing directories: {', '.join(missing)}")
        click.echo()
        click.echo("To create a new project, run:")
        click.echo("  scitex writer clone <project-name>")
        return  # Exit gracefully with code 0

    try:
        writer = Writer(project_dir)

        click.echo(f"Compiling {document} in {project_dir}...")
        click.echo()

        # Compile based on document type
        if document == "manuscript":
            result = writer.compile_manuscript(timeout=timeout)
        elif document == "supplementary":
            result = writer.compile_supplementary(timeout=timeout)
        elif document == "revision":
            result = writer.compile_revision(
                track_changes=track_changes, timeout=timeout
            )

        if result.success:
            click.secho("✓ Compilation successful!", fg="green")
            click.echo(f"PDF: {result.output_pdf}")
        else:
            click.secho(
                f"✗ Compilation failed (exit code {result.exit_code})",
                fg="red",
                err=True,
            )
            if result.errors:
                click.echo()
                click.echo("Errors:")
                for error in result.errors[:10]:  # Show first 10 errors
                    click.echo(f"  - {error}")
            sys.exit(result.exit_code)

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@writer.command()
@click.option(
    "--dir",
    "-d",
    type=click.Path(exists=True),
    help="Project directory (defaults to current directory)",
)
def info(dir):
    """
    Show project information

    \b
    Examples:
        scitex writer info
        scitex writer info --dir ./my_paper
    """
    from scitex.writer import Writer

    project_dir = Path(dir) if dir else Path.cwd()

    # Check if this is a Writer project before trying to attach
    required_dirs = ["01_manuscript", "02_supplementary", "03_revision"]
    missing = [d for d in required_dirs if not (project_dir / d).exists()]

    if missing:
        click.echo(f"Not a Writer project: {project_dir.absolute()}")
        click.echo(f"Missing directories: {', '.join(missing)}")
        return  # Exit gracefully with code 0

    try:
        writer = Writer(project_dir)

        click.echo(f"Project: {writer.project_name}")
        click.echo(f"Location: {writer.project_dir.absolute()}")
        click.echo(f"Git root: {writer.git_root}")
        click.echo()
        click.echo("Documents:")
        click.echo(f"  - Manuscript: {writer.manuscript.root}")
        click.echo(f"  - Supplementary: {writer.supplementary.root}")
        click.echo(f"  - Revision: {writer.revision.root}")
        click.echo()

        # Check for compiled PDFs
        click.echo("Compiled PDFs:")
        for doc_type in ["manuscript", "supplementary", "revision"]:
            pdf = writer.get_pdf(doc_type)
            if pdf:
                click.secho(f"  ✓ {doc_type}: {pdf}", fg="green")
            else:
                click.echo(f"  - {doc_type}: not compiled")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@writer.group()
def mcp():
    """
    MCP (Model Context Protocol) tools for writer module.

    \b
    Examples:
      scitex writer mcp list-tools      # List available MCP tools
      scitex writer mcp usage           # Show tool usage and parameters
    """
    pass


@mcp.command("list-tools")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_tools(as_json: bool):
    """
    List available MCP tools for writer module.

    \b
    Examples:
      scitex writer mcp list-tools
      scitex writer mcp list-tools --json
    """
    from scitex.writer._mcp.tool_schemas import get_tool_schemas

    tools = get_tool_schemas()

    if as_json:
        import json

        output = {
            "module": "writer",
            "count": len(tools),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": list(t.inputSchema.get("properties", {}).keys()),
                    "required": t.inputSchema.get("required", []),
                }
                for t in tools
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        click.secho(
            f"SciTeX Writer MCP Tools ({len(tools)} total)", fg="cyan", bold=True
        )
        click.echo()

        for tool in tools:
            click.secho(f"  {tool.name}", fg="green", bold=True)
            # First line of description
            desc = tool.description.split("\n")[0] if tool.description else ""
            click.echo(f"    {desc}")

            # Parameters
            props = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            if props:
                params = []
                for pname, pinfo in props.items():
                    req_marker = "*" if pname in required else ""
                    params.append(f"{pname}{req_marker}")
                click.echo(f"    Parameters: {', '.join(params)}")
            click.echo()


@mcp.command("usage")
@click.argument("tool_name", required=False)
def usage(tool_name: str):
    """
    Show detailed usage for writer MCP tools.

    \b
    Examples:
      scitex writer mcp usage                    # Show all tools
      scitex writer mcp usage compile_manuscript # Show specific tool
    """
    from scitex.writer._mcp.tool_schemas import get_tool_schemas

    tools = get_tool_schemas()

    if tool_name:
        # Find specific tool
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            click.secho(f"Tool '{tool_name}' not found", fg="red", err=True)
            click.echo(f"Available tools: {', '.join(t.name for t in tools)}")
            raise SystemExit(1)
        tools = [tool]

    for tool in tools:
        click.secho(f"━━━ {tool.name} ━━━", fg="cyan", bold=True)
        click.echo()
        click.echo(f"Description: {tool.description}")
        click.echo()

        props = tool.inputSchema.get("properties", {})
        required = tool.inputSchema.get("required", [])

        if props:
            click.secho("Parameters:", bold=True)
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                default = pinfo.get("default")
                default_str = f" [default: {default}]" if default is not None else ""

                click.echo(f"  {pname}: {ptype}{req}{default_str}")
                if pdesc:
                    click.echo(f"    {pdesc}")
        click.echo()


@writer.command()
@click.option(
    "--dir",
    "-d",
    type=click.Path(exists=True),
    help="Project directory (defaults to current directory)",
)
def watch(dir):
    """
    Watch for file changes and auto-recompile

    \b
    Examples:
        scitex writer watch
        scitex writer watch --dir ./my_paper
    """
    from scitex.writer import Writer

    project_dir = Path(dir) if dir else Path.cwd()

    # Check if this is a Writer project before trying to attach
    required_dirs = ["01_manuscript", "02_supplementary", "03_revision"]
    missing = [d for d in required_dirs if not (project_dir / d).exists()]

    if missing:
        click.echo(f"Not a Writer project: {project_dir.absolute()}")
        click.echo(f"Missing directories: {', '.join(missing)}")
        return  # Exit gracefully with code 0

    try:
        writer = Writer(project_dir)

        click.echo(f"Watching {project_dir} for changes...")
        click.echo("Press Ctrl+C to stop")
        click.echo()

        writer.watch()

    except KeyboardInterrupt:
        click.echo()
        click.echo("Stopped watching")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


# EOF
