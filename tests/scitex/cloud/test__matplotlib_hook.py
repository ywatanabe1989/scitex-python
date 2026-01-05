# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cloud/_matplotlib_hook.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Matplotlib hook for SciTeX Cloud
# # Automatically displays figures inline when running in cloud environment
# 
# import os
# import sys
# from pathlib import Path
# from typing import Optional
# 
# _original_savefig = None
# _original_show = None
# _hooked = False
# 
# 
# def _cloud_savefig(fig, fname, *args, **kwargs):
#     """
#     Wrapper for matplotlib's savefig that emits inline image marker.
#     """
#     from scitex.cloud import is_cloud_environment, emit_inline_image
# 
#     # Call original savefig
#     result = _original_savefig(fig, fname, *args, **kwargs)
# 
#     # Emit inline image if in cloud environment
#     if is_cloud_environment():
#         emit_inline_image(str(fname))
# 
#     return result
# 
# 
# def _cloud_show(*args, **kwargs):
#     """
#     Wrapper for matplotlib's show that saves and displays inline in cloud.
#     """
#     import matplotlib.pyplot as plt
#     import os
#     import time
#     from scitex.cloud import is_cloud_environment, emit_inline_image, get_project_root
# 
#     if is_cloud_environment():
#         # Get all figures
#         fignums = plt.get_fignums()
# 
#         if fignums:
#             # Save figures to project directory scitex/temp/
#             project_root = get_project_root()
#             if not project_root:
#                 return None
# 
#             # Use timestamp for this session's outputs
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             output_dir = project_root / "scitex" / "temp" / timestamp
#             output_dir.mkdir(parents=True, exist_ok=True)
# 
#             for fignum in fignums:
#                 fig = plt.figure(fignum)
# 
#                 filename = f"figure_{fignum}.png"
#                 output_path = output_dir / filename
# 
#                 # Save figure
#                 fig.savefig(output_path, dpi=150, bbox_inches="tight")
# 
#                 # Emit inline image marker with project-relative path
#                 relative_path = output_path.relative_to(project_root)
#                 emit_inline_image(str(relative_path))
# 
#                 print(f"Figure {fignum} saved to: {relative_path}")
# 
#         # Don't call original show() in cloud (headless environment)
#         return None
#     else:
#         # Call original show in non-cloud environment
#         return _original_show(*args, **kwargs)
# 
# 
# def install_matplotlib_hook():
#     """
#     Install matplotlib hooks for cloud environment.
# 
#     This function should be called automatically when scitex is imported
#     in a cloud environment.
#     """
#     global _original_savefig, _original_show, _hooked
# 
#     if _hooked:
#         return  # Already hooked
# 
#     try:
#         import matplotlib.pyplot as plt
#         from matplotlib.figure import Figure
# 
#         # Hook savefig on Figure class
#         _original_savefig = Figure.savefig
#         Figure.savefig = _cloud_savefig
# 
#         # Hook plt.show()
#         _original_show = plt.show
#         plt.show = _cloud_show
# 
#         _hooked = True
# 
#     except ImportError:
#         # matplotlib not available
#         pass
# 
# 
# def uninstall_matplotlib_hook():
#     """
#     Uninstall matplotlib hooks.
#     """
#     global _original_savefig, _original_show, _hooked
# 
#     if not _hooked:
#         return
# 
#     try:
#         import matplotlib.pyplot as plt
#         from matplotlib.figure import Figure
# 
#         # Restore originals
#         if _original_savefig:
#             Figure.savefig = _original_savefig
#         if _original_show:
#             plt.show = _original_show
# 
#         _hooked = False
# 
#     except ImportError:
#         pass
# 
# 
# # Auto-install hooks when in cloud environment
# from scitex.cloud import is_cloud_environment
# 
# if is_cloud_environment():
#     install_matplotlib_hook()
# 
# 
# __all__ = [
#     "install_matplotlib_hook",
#     "uninstall_matplotlib_hook",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cloud/_matplotlib_hook.py
# --------------------------------------------------------------------------------
