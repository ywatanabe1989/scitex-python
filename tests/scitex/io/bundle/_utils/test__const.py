# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_utils/_const.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_utils/_const.py
# 
# """FTS Bundle constants."""
# 
# from typing import Tuple
# 
# # Bundle extension (ZIP for portability, or directory)
# ZIP_EXTENSION: str = ".zip"
# 
# # Supported formats: .zip files or directories (no extension)
# EXTENSIONS: Tuple[str, ...] = (".zip",)
# 
# # Schema constants
# SCHEMA_NAME = "scitex.fts"
# SCHEMA_VERSION = "1.0.0"
# 
# __all__ = [
#     "ZIP_EXTENSION",
#     "EXTENSIONS",
#     "SCHEMA_NAME",
#     "SCHEMA_VERSION",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_utils/_const.py
# --------------------------------------------------------------------------------
