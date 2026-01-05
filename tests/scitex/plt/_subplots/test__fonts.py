# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_fonts.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """Font configuration for matplotlib figures."""
# 
# import os
# 
# import matplotlib as mpl
# import matplotlib.font_manager as fm
# 
# 
# def configure_arial_font():
#     """Configure Arial font for matplotlib if available.
# 
#     Returns
#     -------
#     bool
#         True if Arial was successfully configured, False otherwise.
#     """
#     arial_enabled = False
# 
#     # Try to find Arial
#     try:
#         fm.findfont("Arial", fallback_to_default=False)
#         arial_enabled = True
#     except Exception:
#         # Search for Arial font files and register them
#         arial_paths = [
#             f
#             for f in fm.findSystemFonts()
#             if os.path.basename(f).lower().startswith("arial")
#         ]
# 
#         if arial_paths:
#             for path in arial_paths:
#                 try:
#                     fm.fontManager.addfont(path)
#                 except Exception:
#                     pass
# 
#             # Verify Arial is now available
#             try:
#                 fm.findfont("Arial", fallback_to_default=False)
#                 arial_enabled = True
#             except Exception:
#                 pass
# 
#     # Configure matplotlib to use Arial if available
#     if arial_enabled:
#         mpl.rcParams["font.family"] = "Arial"
#         mpl.rcParams["font.sans-serif"] = [
#             "Arial",
#             "Helvetica",
#             "DejaVu Sans",
#             "Liberation Sans",
#         ]
#     else:
#         # Warn about missing Arial
#         from scitex import logging as _logging
# 
#         _logger = _logging.getLogger(__name__)
#         _logger.warning(
#             "Arial font not found. Using fallback fonts (Helvetica/DejaVu Sans). "
#             "For publication figures with Arial: sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv"
#         )
# 
#     return arial_enabled
# 
# 
# # Configure fonts at module import
# _arial_enabled = configure_arial_font()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_fonts.py
# --------------------------------------------------------------------------------
