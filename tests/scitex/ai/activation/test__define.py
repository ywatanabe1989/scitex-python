# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/activation/_define.py
# --------------------------------------------------------------------------------
# import torch.nn as nn
# 
# 
# def define(act_str):
#     acts_dict = {
#         "relu": nn.ReLU(),
#         "swish": nn.SiLU(),
#         "mish": nn.Mish(),
#         "lrelu": nn.LeakyReLU(0.1),
#     }
#     return acts_dict[act_str]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/activation/_define.py
# --------------------------------------------------------------------------------
