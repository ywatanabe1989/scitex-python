# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/cli/_GlobalArgumentParser.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-27 10:45:47 (ywatanabe)"
# # File: /home/ywatanabe/proj/pip-project-template/src/pip_project_template/cli/_GlobalArgumentParser.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = (
#     "./src/pip_project_template/cli/_GlobalArgumentParser.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """Central argument parser configuration."""
# 
# import argparse
# import importlib
# import pkgutil
# from typing import Dict, Tuple
# 
# 
# class GlobalArgumentParser:
#     """Central argument parser for all CLI commands."""
# 
#     @classmethod
#     def get_command_parsers(cls):
#         """Dynamically discover and load parsers from command modules."""
#         parsers = {}
#         descriptions = {}
# 
#         # Get the directory of this CLI package
#         cli_package_path = os.path.dirname(__file__)
# 
#         # Scan all Python files in the CLI directory
#         for importer, modname, ispkg in pkgutil.iter_modules(
#             [cli_package_path]
#         ):
#             # Skip private modules and the central parser itself
#             if modname.startswith("_") or modname == "__pycache__":
#                 continue
# 
#             try:
#                 # Dynamic import
#                 module = importlib.import_module(
#                     f".{modname}", package="pip_project_template.cli"
#                 )
# 
#                 # Check if module has create_parser function
#                 if hasattr(module, "create_parser"):
#                     parser = module.create_parser()
#                     command_name = modname.replace(
#                         "_", "-"
#                     )  # Convert underscores to hyphens for CLI
#                     parsers[command_name] = parser
#                     descriptions[command_name] = getattr(
#                         parser, "description", f"Run {command_name} command"
#                     )
# 
#             except (
#                 ImportError,
#                 AttributeError,
#                 Exception,
#             ) as e:
#                 # Silently skip modules that can't be imported or don't have create_parser
#                 continue
# 
#         return parsers, descriptions
# 
#     @classmethod
#     def get_main_parser(cls) -> Tuple[argparse.ArgumentParser, Dict]:
#         """Create main parser with subcommands."""
#         parser = argparse.ArgumentParser(
#             prog="python -m pip_project_template",
#             description="Pip Project Template CLI",
#         )
# 
#         subparsers = parser.add_subparsers(
#             dest="command", help="Available commands"
#         )
# 
#         parsers, descriptions = cls.get_command_parsers()
#         subparsers_dict = {}
# 
#         for command_name, command_parser in parsers.items():
#             # Create subparser with description from individual parser
#             subparser = subparsers.add_parser(
#                 command_name,
#                 help=descriptions.get(
#                     command_name, f"Run {command_name} command"
#                 ),
#                 parents=[command_parser],
#                 add_help=False,
#             )
#             subparsers_dict[command_name] = subparser
# 
#         return parser, subparsers_dict
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/cli/_GlobalArgumentParser.py
# --------------------------------------------------------------------------------
