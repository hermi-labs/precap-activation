"""PreCap/SPreCap als fusionierter CUDA-Kernel (vorwaerts + rueckwaerts).

Umsetzung ueber torch.cuda.jiterator: echter CUDA-C++-Code, der zur Laufzeit mit dem in
PyTorch enthaltenen NVRTC kompiliert wird. Kein CUDA-Toolkit, kein Visual Studio noetig.
Ein Kernel-Aufruf pro Richtung statt ca. 10 einzelner Elementaroperationen.
Unterstuetzt float32 und float64 (andere Typen werden ueber float32 gerechnet).
"""
import torch
from torch.cuda.jiterator import _create_jit_fn
from reference import PRESETS

_FWD = r"""
template <typename T> T precap_fwd(T x, T lam, T alpha, T b, T s, T c) {
    T t  = tanh(b * x / s);
    T w  = s * t;
    T ec = exp(c);
    T ce = (w <= c) ? (exp(w) - T(1)) : (ec - T(1) + ec * (w - c));
    return (x > T(0)) ? lam * ce : lam * alpha * ce;
}
"""
_BWD = r"""
template <typename T> T precap_bwd(T g, T x, T lam, T alpha, T b, T s, T c) {
    T t   = tanh(b * x / s);
    T w   = s * t;
    T dce = (w <= c) ? exp(w) : exp(c);
    T k   = (x > T(0)) ? lam : lam * alpha;
    return g * k * dce * b * (T(1) - t * t);
}
"""
_P = dict(lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0)
_fwd = _create_jit_fn(_FWD, **_P)
_bwd = _create_jit_fn(_BWD, **_P)

class _PreCapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam, alpha, b, s, c):
        ctx.save_for_backward(x); ctx.p = dict(lam=lam, alpha=alpha, b=b, s=s, c=c)
        return _fwd(x, **ctx.p)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return _bwd(g.contiguous(), x, **ctx.p), None, None, None, None, None

def precap_cuda(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    if not x.is_cuda: raise ValueError('precap_cuda erwartet einen CUDA-Tensor')
    dt = x.dtype
    if dt not in (torch.float32, torch.float64): x = x.float()
    y = _PreCapFn.apply(x.contiguous(), float(lam), float(alpha), float(b), float(s), float(c))
    return y.to(dt)

class PreCap(torch.nn.Module):
    """nn.Module: PreCap(preset='precap0' | 'spreccap' | 'spreccap_cs') oder eigene Parameter."""
    def __init__(self, preset='precap0', **kw):
        super().__init__(); self.p = {**PRESETS[preset], **kw}
    def forward(self, x): return precap_cuda(x, **self.p)
    def extra_repr(self): return ', '.join(f'{k}={v}' for k, v in self.p.items())
