#!/usr/bin/env python3
# Time-stamp: "2025-01-08 09:30:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/io/_save_modules/test__torch.py

"""Tests for PyTorch save functionality."""

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")
torch = pytest.importorskip("torch")


class TestSaveTorchAvailableFlags:
    """Test _AVAILABLE flags for optional dependencies."""

    def test_torch_available_flag_exists(self):
        """Test that TORCH_AVAILABLE flag is exported."""
        from scitex.io._save_modules._torch import TORCH_AVAILABLE

        assert isinstance(TORCH_AVAILABLE, bool)

    def test_torch_available_is_true_when_torch_installed(self):
        """Test that TORCH_AVAILABLE is True when torch is installed."""
        from scitex.io._save_modules._torch import TORCH_AVAILABLE

        assert TORCH_AVAILABLE is True


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_torch.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:25:14 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/io/_save_modules/_torch.py
#
#
# def _save_torch(obj, spath, **kwargs):
#     """
#     Save a PyTorch model or tensor.
#
#     Parameters
#     ----------
#     obj : torch.nn.Module or torch.Tensor
#         The PyTorch model or tensor to save.
#     spath : str
#         Path where the PyTorch file will be saved.
#     **kwargs : dict
#         Additional keyword arguments to pass to torch.save.
#
#     Returns
#     -------
#     None
#     """
#     # Lazy import to avoid circular import issues
#     import torch
#
#     torch.save(obj, spath, **kwargs)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_torch.py
# --------------------------------------------------------------------------------
