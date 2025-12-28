# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dict/_flatten.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-10 22:38:40 (ywatanabe)"
# 
# 
# def flatten(nested_dict, parent_key="", sep="_"):
#     items = []
#     for key, value in nested_dict.items():
#         new_key = f"{parent_key}{sep}{key}" if parent_key else key
#         if isinstance(value, dict):
#             items.extend(flatten(value, new_key, sep=sep).items())
#         elif isinstance(value, (list, tuple)):
#             for idx, item in enumerate(value):
#                 items.append((f"{new_key}_{idx}", item))
#         else:
#             items.append((new_key, value))
#     return dict(items)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dict/_flatten.py
# --------------------------------------------------------------------------------
