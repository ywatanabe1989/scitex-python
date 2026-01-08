#!/usr/bin/env python3
# Time-stamp: "ywatanabe (2024-11-02 22:48:44)"
# File: ./scitex_repo/src/scitex/dsp/reference.py

try:
    import torch as _torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    _torch = None

from scitex.decorators import torch_fn as _torch_fn


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Please install with: pip install torch"
        )


@_torch_fn
def common_average(x, dim=-2):
    _check_torch()
    re_referenced = (x - x.mean(dim=dim, keepdims=True)) / x.std(dim=dim, keepdims=True)
    assert x.shape == re_referenced.shape
    return re_referenced


@_torch_fn
def random(x, dim=-2):
    _check_torch()
    idx_all = [slice(None)] * x.ndim
    idx_rand_dim = _torch.randperm(x.shape[dim])
    idx_all[dim] = idx_rand_dim
    y = x[idx_all]
    re_referenced = x - y
    assert x.shape == re_referenced.shape
    return re_referenced


@_torch_fn
def take_reference(x, tgt_indi, dim=-2):
    _check_torch()
    idx_all = [slice(None)] * x.ndim
    idx_all[dim] = tgt_indi
    ref = x[tuple(idx_all)].unsqueeze(dim)
    re_referenced = x - ref
    assert x.shape == re_referenced.shape
    return re_referenced


if __name__ == "__main__":
    import scitex

    x, f, t = scitex.dsp.demo_sig()
    y = common_average(x)

# EOF
