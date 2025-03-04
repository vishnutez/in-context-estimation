"""
Utility functions for implementing tasks.
Currently unused.
"""

import torch
import numpy as np


def real_to_complex(real_tens):
    """
    Given a real-valued tensor, return a complex-valued tensor
    with the same real part. The imaginary part will be all 0s.
    """
    ndim = real_tens.ndim
    stacked = torch.stack([real_tens, torch.zeros_like(real_tens)], dim=ndim)
    assert stacked.shape[-1] == 2
    return torch.view_as_complex(stacked)
