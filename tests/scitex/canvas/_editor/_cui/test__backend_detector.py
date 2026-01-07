# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_backend_detector.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_editor/_cui/_backend_detector.py
# 
# """Backend detection for figure editor (Flask only)."""
# 
# __all__ = ["print_available_backends", "detect_best_backend"]
# 
# 
# def print_available_backends() -> None:
#     """Print Flask backend availability."""
#     print("\n" + "=" * 50)
#     print("SciTeX Visual Editor")
#     print("=" * 50)
# 
#     try:
#         import flask
# 
#         print(f"  Flask: [OK] {flask.__version__}")
#     except ImportError:
#         print("  Flask: [NOT INSTALLED]")
#         print("\nInstall: pip install flask")
# 
#     print("=" * 50 + "\n")
# 
# 
# def detect_best_backend() -> str:
#     """Return Flask as the only supported backend."""
#     try:
#         import flask
# 
#         return "flask"
#     except ImportError:
#         raise ImportError(
#             "Flask is required for the editor. Install with: pip install flask"
#         )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_backend_detector.py
# --------------------------------------------------------------------------------
