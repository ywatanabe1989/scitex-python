#!/usr/bin/env python3
"""
SciTeX CLI - Template Commands (Project Scaffolding)

Provides project template cloning and management.
"""

import sys
from pathlib import Path

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def template():
    """
    Project template scaffolding

    \b
    Available templates:
      research      - Full scientific workflow structure
      pip-project   - Pip-installable Python package
      singularity   - Container-based project
      paper         - Academic paper writing template

    \b
    Examples:
      scitex template list                        # Show available templates
      scitex template clone research ./my-project # Clone research template
      scitex template clone pip-project ./my-pkg  # Clone package template
    """
    pass


@template.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_templates(as_json):
    """
    List available project templates

    \b
    Example:
      scitex template list
      scitex template list --json
    """
    try:
        from scitex.template import get_available_templates_info

        templates = get_available_templates_info()

        if as_json:
            import json

            click.echo(json.dumps(templates, indent=2))
        else:
            click.secho("Available SciTeX Templates", fg="cyan", bold=True)
            click.echo("=" * 60)

            for tmpl in templates:
                click.echo()
                click.secho(f"{tmpl['name']} ({tmpl['id']})", fg="green", bold=True)
                click.echo(f"  {tmpl['description']}")
                click.echo(f"  Use case: {tmpl['use_case']}")
                click.echo(f"  GitHub: {tmpl['github_url']}")
                if tmpl.get("features"):
                    click.echo("  Features:")
                    for feature in tmpl["features"][:3]:  # Show first 3
                        click.echo(f"    - {feature}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@template.command()
@click.argument(
    "template_type",
    type=click.Choice(["research", "pip-project", "singularity", "paper"]),
)
@click.argument("destination", type=click.Path())
@click.option(
    "--git-strategy",
    "-g",
    type=click.Choice(["child", "parent", "origin", "none"]),
    default="child",
    help="Git initialization strategy (default: child)",
)
@click.option("--branch", "-b", help="Specific branch to clone")
@click.option("--tag", "-t", help="Specific tag/release to clone")
def clone(template_type, destination, git_strategy, branch, tag):
    """
    Clone a project template

    \b
    Template Types:
      research    - Full scientific workflow (scripts, data, docs, results)
      pip-project - Python package (src, tests, docs, CI/CD)
      singularity - Container-based (definition files, build scripts)
      paper       - Academic paper (LaTeX, BibTeX, figures)

    \b
    Git Strategies:
      child  - Create isolated git in project directory (default)
      parent - Use parent git repository
      origin - Preserve template's original git history
      none   - Disable git initialization

    \b
    Examples:
      scitex template clone research ./my-research
      scitex template clone pip-project ./my-package --git-strategy parent
      scitex template clone paper ./manuscript --branch develop
    """
    try:
        # Validate mutual exclusivity
        if branch and tag:
            click.echo("Error: Cannot specify both --branch and --tag", err=True)
            sys.exit(1)

        # Convert git_strategy 'none' to None
        if git_strategy == "none":
            git_strategy = None

        # Map template types to clone functions
        clone_funcs = {
            "research": "clone_research",
            "pip-project": "clone_pip_project",
            "singularity": "clone_singularity",
            "paper": "clone_writer_directory",
        }

        func_name = clone_funcs[template_type]

        # Import the appropriate function
        from scitex import template as tmpl_module

        clone_func = getattr(tmpl_module, func_name)

        click.echo(f"Cloning {template_type} template to {destination}...")

        # Call clone function
        kwargs = {"path": destination}
        if git_strategy is not None:
            kwargs["git_strategy"] = git_strategy
        if branch:
            kwargs["branch"] = branch
        if tag:
            kwargs["tag"] = tag

        result = clone_func(**kwargs)

        if result:
            dest_path = Path(destination).absolute()
            click.secho("Successfully cloned template!", fg="green")
            click.echo()
            click.echo(f"Location: {dest_path}")
            click.echo()
            click.echo("Next steps:")
            click.echo(f"  cd {destination}")

            if template_type == "research":
                click.echo("  # Start your research project")
                click.echo("  # Edit scripts/, data/, docs/")
            elif template_type == "pip-project":
                click.echo("  # Develop your package")
                click.echo("  pip install -e .")
                click.echo("  pytest")
            elif template_type == "singularity":
                click.echo("  # Build your container")
                click.echo("  singularity build container.sif Singularity.def")
            elif template_type == "paper":
                click.echo("  # Write your paper")
                click.echo("  scitex writer compile manuscript")
        else:
            click.secho("Failed to clone template", fg="red", err=True)
            sys.exit(1)

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@template.command()
@click.argument("template_id")
def info(template_id):
    """
    Show detailed information about a template

    \b
    Example:
      scitex template info research
      scitex template info pip-project
    """
    try:
        from scitex.template import get_available_templates_info

        templates = get_available_templates_info()

        # Find matching template
        template_info = None
        for tmpl in templates:
            if tmpl["id"] == template_id or tmpl["name"].lower() == template_id.lower():
                template_info = tmpl
                break

        if not template_info:
            click.secho(f"Template '{template_id}' not found", fg="red", err=True)
            click.echo("Available templates:")
            for tmpl in templates:
                click.echo(f"  - {tmpl['id']}")
            sys.exit(1)

        click.secho(f"Template: {template_info['name']}", fg="cyan", bold=True)
        click.echo("=" * 50)
        click.echo()
        click.echo(f"ID: {template_info['id']}")
        click.echo(f"Description: {template_info['description']}")
        click.echo(f"Use case: {template_info['use_case']}")
        click.echo(f"GitHub: {template_info['github_url']}")
        click.echo()
        click.secho("Features:", fg="yellow")
        for feature in template_info.get("features", []):
            click.echo(f"  - {feature}")
        click.echo()
        click.echo("Clone with:")
        click.secho(
            f"  scitex template clone {template_info['id']} ./my-project", fg="green"
        )

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    template()
