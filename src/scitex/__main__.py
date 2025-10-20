#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX Package Entry Point

Allows running: python -m scitex [command]
"""

import sys


def main():
    """Main entry point for scitex CLI"""
    try:
        from scitex.cli.main import cli
        cli()
    except ImportError:
        # CLI not available (click not installed)
        print("SciTeX CLI requires 'click' package")
        print("Install: pip install click")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Original content preserved below:
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-13 18:56:15 (ywatanabe)"
# /home/yusukew/proj/scitex_repo/src/scitex/__main__.py

"""
1. Functionality:
   - Serves as the entry point for the scitex package
   - Handles command-line arguments and directs to appropriate submodules
2. Input:
   - Command-line arguments
3. Output:
   - Execution of specified submodule or usage instructions
4. Prerequisites:
   - scitex package and its submodules
"""

import warnings

warnings.filterwarnings("ignore")

import sys
from .gen._print_config import print_config_main
from .template.create_research import main as create_research_main
from .template.create_pip_project import main as create_pip_project_main
from .template.create_singularity import main as create_singularity_main


def main():
    """Main entry point for SciTeX command-line interface.

    Provides command-line access to various SciTeX utilities.

    Commands
    --------
    print_config : Print configuration values
        Usage: python -m scitex print_config [key]

    create_research_project : Create a new research project from template
        Usage: python -m scitex create_research_project <project-name> [target-dir]

    create_pip_project : Create a new pip project from template
        Usage: python -m scitex create_pip_project <project-name> [target-dir]

    create_singularity_project : Create a new singularity project from template
        Usage: python -m scitex create_singularity_project <project-name> [target-dir]

    Examples
    --------
    >>> # Print all configuration
    >>> python -m scitex print_config

    >>> # Print specific configuration key
    >>> python -m scitex print_config DATABASE_URL

    >>> # Create a new research project
    >>> python -m scitex create_research_project my_research_project

    >>> # Create a new pip project
    >>> python -m scitex create_pip_project my_pip_project

    >>> # Create a new singularity project
    >>> python -m scitex create_singularity_project my_singularity_project

    >>> # Create in a specific directory
    >>> python -m scitex create_research_project my_project ~/projects

    Raises
    ------
    SystemExit
        If no command provided or unknown command specified.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m scitex <command> [args]")
        print("")
        print("Available commands:")
        print("  print_config          Print configuration values")
        print("  create_research_project       Create a new research project")
        print("  create_pip_project    Create a new pip project")
        print("  create_singularity_project    Create a new singularity project")
        print("")
        print("Use 'python -m scitex <command> --help' for more information on a command.")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "print_config":
        print_config_main(args)
    elif command == "create_research_project":
        create_research_main(args)
    elif command == "create_pip_project":
        create_pip_project_main(args)
    elif command == "create_singularity_project":
        create_singularity_main(args)
    else:
        print(f"Unknown command: {command}")
        print("")
        print("Available commands:")
        print("  print_config          Print configuration values")
        print("  create_research_project       Create a new research project")
        print("  create_pip_project    Create a new pip project")
        print("  create_singularity_project    Create a new singularity project")
        sys.exit(1)


if __name__ == "__main__":
    main()
