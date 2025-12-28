# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/cli/info.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-08-27 01:10:31 (ywatanabe)"
# # File: /home/ywatanabe/proj/pip-project-template/src/cli/info.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# __FILE__ = (
#     "./pip-project-template/src/cli/_info.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# """Info command."""
# 
# import argparse
# 
# 
# def create_parser():
#     """Create parser for info command."""
#     parser = argparse.ArgumentParser(description="Show system information")
#     return parser
# 
# 
# def main(args=None):
#     """Execute info command."""
#     import sys
#     import os
#     
#     # Get version from package __init__.py  
#     package_init = os.path.join(os.path.dirname(os.path.dirname(__file__)), '__init__.py')
#     version = "0.1.0"  # default
#     try:
#         with open(package_init) as f:
#             for line in f:
#                 if line.startswith('__version__'):
#                     version = line.split('"')[1]
#                     break
#     except Exception:
#         pass
# 
#     print("Pip Project Template - FastMCP Edition")
#     print("=" * 40)
#     print(f"Version: {version}")
#     print("Framework: FastMCP 2.0")
#     print("Commands: calculate, serve01, serve02, info")
#     print()
#     print("MCP Servers:")
#     print("  serve01: Basic FastMCP server with calculator tools")
#     print("  serve02: Enhanced FastMCP server with batch processing")
#     print()
#     print("Transport Options: stdio, http, sse")
#     return 0
# 
# if __name__ == "__main__":
#     import sys
#     sys.exit(main())
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/docs/to_claude/examples/pip-project-template/src/pip_project_template/cli/info.py
# --------------------------------------------------------------------------------
