# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_ensure_3d.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 01:04:03 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/dsp/utils/_ensure_3d.py
# 
# from scitex.decorators import torch_fn
# 
# 
# @torch_fn
# def ensure_3d(x):
#     if x.ndim == 1:  # assumes (seq_len,)
#         x = x.unsqueeze(0).unsqueeze(0)
#     elif x.ndim == 2:  # assumes (batch_siize, seq_len)
#         x = x.unsqueeze(1)
#     return x
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dsp/utils/_ensure_3d.py
# --------------------------------------------------------------------------------
