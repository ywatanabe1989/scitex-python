# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_pdf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_pdf.py
# 
# """PDF metadata reading from PDF Info Dictionary."""
# 
# import json
# from typing import Any, Dict, Optional
# 
# 
# def read_metadata_pdf(image_path: str) -> Optional[Dict[str, Any]]:
#     """
#     Read metadata from a PDF file.
# 
#     Args:
#         image_path: Path to the PDF file.
# 
#     Returns:
#         Dictionary containing metadata, or None if no metadata found.
#     """
#     metadata = None
# 
#     try:
#         from pypdf import PdfReader
# 
#         reader = PdfReader(image_path)
# 
#         # Try to read metadata from PDF Info Dictionary
#         if reader.metadata:
#             # Check Subject field for JSON metadata
#             if "/Subject" in reader.metadata:
#                 subject = reader.metadata["/Subject"]
#                 try:
#                     metadata = json.loads(subject)
#                 except json.JSONDecodeError:
#                     # If not JSON, create metadata dict from available fields
#                     metadata = {
#                         "title": reader.metadata.get("/Title", ""),
#                         "author": reader.metadata.get("/Author", ""),
#                         "subject": subject,
#                         "creator": reader.metadata.get("/Creator", ""),
#                     }
#     except ImportError:
#         pass  # pypdf not available, return None
#     except Exception:
#         pass  # PDF metadata corrupted or not readable
# 
#     return metadata
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/read_metadata_pdf.py
# --------------------------------------------------------------------------------
