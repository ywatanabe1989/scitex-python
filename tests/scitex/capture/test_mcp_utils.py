# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/capture/mcp_utils.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """MCP utility functions for SciTeX Capture."""
# 
# import os
# import shutil
# from pathlib import Path
# 
# # Directory configuration
# SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
# SCITEX_CAPTURE_DIR = SCITEX_BASE_DIR / "capture"
# LEGACY_CAPTURE_DIR = Path.home() / ".cache" / "cammy"
# 
# 
# def get_capture_dir() -> Path:
#     """Get screenshot capture directory, migrating from legacy if needed."""
#     new_dir = SCITEX_CAPTURE_DIR
#     old_dir = LEGACY_CAPTURE_DIR
# 
#     new_dir.mkdir(parents=True, exist_ok=True)
# 
#     if old_dir.exists():
#         new_screenshots = list(new_dir.glob("*.jpg"))
#         if not new_screenshots:
#             try:
#                 for img in old_dir.glob("*.jpg"):
#                     shutil.move(str(img), str(new_dir / img.name))
#             except Exception:
#                 pass
# 
#     return new_dir

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/capture/mcp_utils.py
# --------------------------------------------------------------------------------
