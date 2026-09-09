"""
Shared normalisation layers for URA-Shree.

Split out of model/transformer.py so model/attention.py can reuse RMSNorm for
QK-normalisation without a circular import between the two modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square layer normalisation:  y = x / sqrt(mean(x^2) + eps) * gamma

    No mean subtraction and no bias, so it costs roughly 10-15% less than
    LayerNorm while being just as stable in practice.
    """

    _HAS_FUSED = hasattr(F, "rms_norm")

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._HAS_FUSED:
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        # Accumulate the variance in fp32 even under autocast; the reciprocal
        # square root of a bf16 mean is where low-precision training goes wrong.
        dtype = x.dtype
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f.to(dtype)) * self.weight
