# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_symlink.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-19
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_symlink.py
# 
# """Symlink creation utilities for save operations."""
# 
# import os
# from pathlib import Path
# 
# from scitex import logging
# from scitex.path._clean import clean
# from scitex.sh import sh
# from scitex.str._color_text import color_text
# 
# logger = logging.getLogger()
# 
# 
# def symlink(spath, spath_cwd, symlink_from_cwd, verbose):
#     """Create a symbolic link from the current working directory."""
#     if symlink_from_cwd and (spath != spath_cwd):
#         os.makedirs(os.path.dirname(spath_cwd), exist_ok=True)
#         sh(["rm", "-f", f"{spath_cwd}"], verbose=False)
#         sh(["ln", "-sfr", f"{spath}", f"{spath_cwd}"], verbose=False)
#         if verbose:
#             # Get file extension to provide more informative message
#             ext = os.path.splitext(spath_cwd)[1].lower()
#             logger.success(color_text(f"(Symlinked to: {spath_cwd})"))
# 
# 
# def symlink_to(spath_final, symlink_to_path, verbose):
#     """Create a symbolic link at the specified path pointing to the saved file."""
#     if symlink_to_path:
#         # Convert Path objects to strings for consistency
#         if isinstance(symlink_to_path, Path):
#             symlink_to_path = str(symlink_to_path)
# 
#         # Clean the symlink path
#         symlink_to_path = clean(symlink_to_path)
# 
#         # Ensure the symlink directory exists (only if there is a directory component)
#         symlink_dir = os.path.dirname(symlink_to_path)
#         if symlink_dir:  # Only create directory if there's a directory component
#             os.makedirs(symlink_dir, exist_ok=True)
# 
#         # Remove existing symlink or file
#         sh(["rm", "-f", f"{symlink_to_path}"], verbose=False)
# 
#         # Create the symlink using relative path for robustness
#         sh(["ln", "-sfr", f"{spath_final}", f"{symlink_to_path}"], verbose=False)
# 
#         if verbose:
#             symlink_to_full = (
#                 os.path.realpath(symlink_to_path) + "/" + os.path.basename(spath_final)
#             )
#             logger.success(f"Symlinked: {spath_final} -> {symlink_to_full}")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_symlink.py
# --------------------------------------------------------------------------------
