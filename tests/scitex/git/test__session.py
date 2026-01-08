# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_session.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/session.py
# 
# """
# Session management utilities for git module CLI tools.
# """
# 
# import sys
# import matplotlib.pyplot as plt
# import scitex as stx
# from ._constants import EXIT_SUCCESS
# 
# 
# def run_with_session(parse_args_func, main_func):
#     """
#     Run main function with scitex session management.
# 
#     Parameters
#     ----------
#     parse_args_func : callable
#         Function to parse command line arguments
#     main_func : callable
#         Main function to execute with parsed args
# 
#     Returns
#     -------
#     None
#         Exits with appropriate status code
#     """
#     global CONFIG, CC, rng
# 
#     args = parse_args_func()
# 
#     CONFIG, sys.stdout, sys.stderr, plt_obj, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file="scitex.git",
#         sdir_suffix=None,
#         verbose=False,
#         agg=True,
#     )
# 
#     exit_status = main_func(args)
# 
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         message="",
#         exit_status=exit_status,
#     )
# 
# 
# __all__ = [
#     "run_with_session",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_session.py
# --------------------------------------------------------------------------------
