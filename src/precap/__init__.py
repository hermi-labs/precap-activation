"""PreCap / SPreCap: beschraenkte, selbstnormierende Aktivierungsfunktion.

    y(x) = lam * (1 falls x > 0, sonst alpha) * capexp(s * tanh(b * x / s))
    capexp(w) = exp(w) - 1 (w <= c),  e^c - 1 + e^c (w - c) (w > c)

PyTorch:  from precap import PreCap, precap
          PreCap('spreccap_cs')  # nutzt auf CUDA-Tensoren automatisch den fusionierten Kernel
MLX:      from precap.mlx import PreCap
"""
from ._presets import PRESETS, DEFAULT

__version__ = '0.2.0'
__all__ = ['PRESETS', 'DEFAULT', 'PreCap', 'precap', 'precap_ref', 'precap_ref_grad', '__version__']

_TORCH = ('PreCap', 'precap', 'precap_ref', 'precap_ref_grad')

def __getattr__(name):
    # torch wird erst bei Bedarf importiert, damit 'precap.mlx' auch ohne torch funktioniert
    if name in _TORCH:
        from . import torch as _t
        return getattr(_t, name)
    raise AttributeError(name)
