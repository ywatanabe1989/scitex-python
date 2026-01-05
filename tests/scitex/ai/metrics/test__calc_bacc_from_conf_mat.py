# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/metrics/_calc_bacc_from_conf_mat.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-02 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/ml/metrics/_calc_bacc_from_conf_mat.py
# 
# """Calculate balanced accuracy from confusion matrix."""
# 
# __FILE__ = __file__
# 
# import numpy as np
# 
# 
# def calc_bacc_from_conf_mat(cm: np.ndarray) -> float:
#     """
#     Calculate balanced accuracy from confusion matrix.
# 
#     Parameters
#     ----------
#     cm : np.ndarray
#         Confusion matrix
# 
#     Returns
#     -------
#     float
#         Balanced accuracy
#     """
#     try:
#         per_class = np.diag(cm) / np.sum(cm, axis=1)
#         return float(np.nanmean(per_class))
#     except:
#         return np.nan
# 
# 
# # Convenience aliases
# bACC_from_conf_mat = calc_bacc_from_conf_mat
# balanced_accuracy_from_conf_mat = calc_bacc_from_conf_mat
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/metrics/_calc_bacc_from_conf_mat.py
# --------------------------------------------------------------------------------
