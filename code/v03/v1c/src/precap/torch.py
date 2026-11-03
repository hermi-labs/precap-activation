"""PyTorch-Umsetzung. Auf CUDA-Tensoren (fp32/fp64) wird ein fusionierter Kernel ueber
torch.cuda.jiterator genutzt (kein CUDA-Toolkit noetig), sonst die Referenz mit Autograd."""
import math
import torch as _torch
from ._presets import PRESETS, DEFAULT

def precap_ref(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    """Referenz (reines PyTorch, Autograd). Massgeblich fuer alle Tests."""
    w = s * _torch.tanh(b * x / s)
    ec = math.exp(c)
    ce = _torch.where(w <= c, _torch.exp(_torch.clamp(w, max=c)) - 1, ec - 1 + ec * (w - c))
    return lam * _torch.where(x > 0, ce, alpha * ce)

def precap_ref_grad(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    """Analytische Ableitung dy/dx."""
    t = _torch.tanh(b * x / s); w = s * t
    dce = _torch.where(w <= c, _torch.exp(_torch.clamp(w, max=c)), _torch.full_like(w, math.exp(c)))
    return lam * _torch.where(x > 0, _torch.ones_like(x), _torch.full_like(x, alpha)) * dce * b * (1 - t * t)

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
_jit = {}

def _kernels():
    if not _jit:
        from torch.cuda.jiterator import _create_jit_fn
        p = dict(lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0)
        _jit['f'] = _create_jit_fn(_FWD, **p)
        _jit['b'] = _create_jit_fn(_BWD, **p)
    return _jit['f'], _jit['b']

class _PreCapCUDA(_torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam, alpha, b, s, c):
        ctx.save_for_backward(x); ctx.p = dict(lam=lam, alpha=alpha, b=b, s=s, c=c)
        return _kernels()[0](x, **ctx.p)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return _kernels()[1](g.contiguous(), x, **ctx.p), None, None, None, None, None

def precap_cuda(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    """Fusionierter CUDA-Kernel. fp16/bf16 werden ueber fp32 gerechnet und zurueckgewandelt."""
    if not x.is_cuda: raise ValueError('precap_cuda erwartet einen CUDA-Tensor')
    dt = x.dtype
    xc = x if dt in (_torch.float32, _torch.float64) else x.float()
    y = _PreCapCUDA.apply(xc.contiguous(), float(lam), float(alpha), float(b), float(s), float(c))
    return y.to(dt)

def precap(x, preset=DEFAULT, backend='auto', **kw):
    """Funktionale Form. backend: 'auto' (CUDA-Kernel wenn moeglich), 'cuda' oder 'ref'."""
    p = {**PRESETS[preset], **kw}
    if backend == 'cuda' or (backend == 'auto' and x.is_cuda):
        return precap_cuda(x, **p)
    return precap_ref(x, **p)

class PreCap(_torch.nn.Module):
    """PreCap(preset='spreccap_cs' | 'spreccap_e03' | 'spreccap' | 'precap0', backend='auto', **parameter)"""
    def __init__(self, preset=DEFAULT, backend='auto', **kw):
        super().__init__()
        if preset not in PRESETS: raise ValueError(f'unbekanntes preset {preset!r}, erlaubt: {list(PRESETS)}')
        if backend not in ('auto', 'cuda', 'ref'): raise ValueError("backend muss 'auto', 'cuda' oder 'ref' sein")
        self.preset, self.backend, self.p = preset, backend, {**PRESETS[preset], **kw}
    def forward(self, x):
        if self.backend == 'cuda' or (self.backend == 'auto' and x.is_cuda):
            return precap_cuda(x, **self.p)
        return precap_ref(x, **self.p)
    def extra_repr(self):
        return f'preset={self.preset}, backend={self.backend}, ' + ', '.join(f'{k}={v}' for k, v in self.p.items())
