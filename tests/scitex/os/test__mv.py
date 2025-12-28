# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/os/_mv.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-06 09:00:45 (ywatanabe)"
# 
# # import os
# # import shutil
# 
# # def mv(src, tgt):
# #     successful = True
# #     os.makedirs(tgt, exist_ok=True)
# 
# #     if os.path.isdir(src):
# #         # Iterate over the items in the directory
# #         for item in os.listdir(src):
# #             item_path = os.path.join(src, item)
# #             # Check if the item is a file
# #             if os.path.isfile(item_path):
# #                 try:
# #                     shutil.move(item_path, tgt)
# #                     print(f"\nMoved file from {item_path} to {tgt}")
# #                 except OSError as e:
# #                     print(f"\nError: {e}")
# #                     successful = False
# #             else:
# #                 print(f"\nSkipped directory {item_path}")
# #     else:
# #         # If src is a file, just move it
# #         try:
# #             shutil.move(src, tgt)
# #             print(f"\nMoved from {src} to {tgt}")
# #         except OSError as e:
# #             print(f"\nError: {e}")
# #             successful = False
# 
# #     return successful
# 
# 
# def mv(src, tgt):
#     import os
#     import shutil
# 
#     successful = True
#     os.makedirs(tgt, exist_ok=True)
# 
#     try:
#         shutil.move(src, tgt)
#         print(f"\nMoved from {src} to {tgt}")
#     except OSError as e:
#         print(f"\nError: {e}")
#         successful = False

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/os/_mv.py
# --------------------------------------------------------------------------------
