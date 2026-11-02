"""PreCap / SPreCap: beschraenkte, selbstnormierende Aktivierungsfunktion.

    y(x) = lam * (1 falls x > 0, sonst alpha) * capexp(s * tanh(b * x / s))
    capexp(w) = exp(w) - 1 (w <= c),  e^c - 1 + e^c (w - c) (w > c)

PyTorch:  from precap import PreCap, precap
          PreCap('spreccap_cs')  # nutzt auf CUDA-Tensoren automatisch den fusionierten Kernel
MLX:      from precap.mlx import PreCap
v0.3:     from precap import BPreCap, Lagma   # BPreCap(depth=100 | 200): tiefenkalibrierter Kern in der Lagma-Huelle;
          #   Lagma(M, rho=0.8): weiche, feste Huelle. Beide mit fusioniertem Kernel (CUDA ueber jiterator, MLX ueber Metal)
"""
from ._presets import PRESETS, DEFAULT, BOUNDED, HULL_RHO

__version__ = '0.3.0'
__all__ = ['PRESETS', 'DEFAULT', 'BOUNDED', 'HULL_RHO', 'PreCap', 'precap', 'precap_ref', 'precap_ref_grad', 'BPreCap', 'bprecap_ref', 'lb_ref', 'lagma_hull', 'Lagma', 'bprecap_cuda', 'lagma_cuda', '__version__']

_TORCH = ('PreCap', 'precap', 'precap_ref', 'precap_ref_grad', 'BPreCap', 'bprecap_ref', 'lb_ref', 'lagma_hull', 'Lagma', 'bprecap_cuda', 'lagma_cuda')

def __getattr__(name):
    # torch wird erst bei Bedarf importiert, damit 'precap.mlx' auch ohne torch funktioniert
    if name in _TORCH:
        from . import torch as _t
        return getattr(_t, name)
    raise AttributeError(name)
