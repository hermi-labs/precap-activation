"""PyTorch-Umsetzung. Auf CUDA-Tensoren (fp32/fp64) wird ein fusionierter Kernel ueber
torch.cuda.jiterator genutzt (kein CUDA-Toolkit noetig), sonst die Referenz mit Autograd."""
import math
import torch as _torch
from ._presets import PRESETS, DEFAULT, BOUNDED, HULL_RHO

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

# --- v0.3: Lagma-Huelle und BPreCap (Referenz mit Autograd, fusionierter CUDA-Kernel ueber jiterator) ---------
def lagma_hull(y, M, rho=HULL_RHO):
    """Lagma-Huelle h_M(y) = M * phi_rho(y / M): Identitaet fuer |y| <= rho*M, darueber stetig differenzierbar (C1) gegen +-M."""
    u = y / M; a = u.abs()
    k = _torch.tanh((a - rho) / (1 - rho))
    return M * _torch.where(a <= rho, u, _torch.sign(u) * (rho + (1 - rho) * k))

def bprecap_ref(x, preset='bprecap_d100'):
    """BPreCap = Lagma-Huelle um den tiefenkalibrierten PreCap-Kern (Referenz, Autograd). preset: 'bprecap_d100' | 'bprecap_d200'."""
    p = BOUNDED[preset]
    return lagma_hull(precap_ref(x, **p['core']), p['M'])

def lb_ref(x, preset='lb_d100'):
    """LB = Lagma-Huelle um einen linearen Kern beta * x (Vergleichsform aus den Studien)."""
    p = BOUNDED[preset]
    return lagma_hull(p['beta'] * x, p['M'])

_HULL_T = r"""
template <typename T> T lagma_h(T y, T M, T rho) {
    T u = y / M; T a = abs(u);
    if (a <= rho) return y;
    T k = tanh((a - rho) / (T(1) - rho));
    return (u > T(0) ? T(1) : T(-1)) * M * (rho + (T(1) - rho) * k);
}
template <typename T> T lagma_dh(T y, T M, T rho) {
    T a = abs(y / M);
    if (a <= rho) return T(1);
    T k = tanh((a - rho) / (T(1) - rho));
    return T(1) - k * k;
}
"""
_LG_FWD = _HULL_T + r"""
template <typename T> T lagma_fwd(T y, T M, T rho) { return lagma_h(y, M, rho); }
"""
_LG_BWD = _HULL_T + r"""
template <typename T> T lagma_bwd(T g, T y, T M, T rho) { return g * lagma_dh(y, M, rho); }
"""
_BP_FWD = _HULL_T + r"""
template <typename T> T bprecap_fwd(T x, T lam, T alpha, T b, T s, T c, T M, T rho) {
    T t = tanh(b * x / s); T w = s * t; T ec = exp(c);
    T ce = (w <= c) ? (exp(w) - T(1)) : (ec - T(1) + ec * (w - c));
    T y = (x > T(0)) ? lam * ce : lam * alpha * ce;
    return lagma_h(y, M, rho);
}
"""
_BP_BWD = _HULL_T + r"""
template <typename T> T bprecap_bwd(T g, T x, T lam, T alpha, T b, T s, T c, T M, T rho) {
    T t = tanh(b * x / s); T w = s * t; T ec = exp(c);
    T ce  = (w <= c) ? (exp(w) - T(1)) : (ec - T(1) + ec * (w - c));
    T dce = (w <= c) ? exp(w) : ec;
    T kk  = (x > T(0)) ? lam : lam * alpha;
    return g * kk * dce * b * (T(1) - t * t) * lagma_dh(kk * ce, M, rho);
}
"""
_jit3 = {}
def _kernels3():
    if not _jit3:
        from torch.cuda.jiterator import _create_jit_fn
        _jit3['lf'] = _create_jit_fn(_LG_FWD, M=1.0, rho=0.8); _jit3['lb'] = _create_jit_fn(_LG_BWD, M=1.0, rho=0.8)
        p = dict(lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0, M=3.0, rho=0.8)
        _jit3['bf'] = _create_jit_fn(_BP_FWD, **p); _jit3['bb'] = _create_jit_fn(_BP_BWD, **p)
    return _jit3

class _LagmaCUDA(_torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, M, rho):
        ctx.save_for_backward(y); ctx.p = dict(M=M, rho=rho); return _kernels3()['lf'](y, **ctx.p)
    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors; return _kernels3()['lb'](g.contiguous(), y, **ctx.p), None, None

class _BPreCapCUDA(_torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.save_for_backward(x); ctx.p = p; return _kernels3()['bf'](x, **p)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors; return _kernels3()['bb'](g.contiguous(), x, **ctx.p), None

def _as32(x): return x if x.dtype in (_torch.float32, _torch.float64) else x.float()
def lagma_cuda(y, M, rho=HULL_RHO):
    """Fusionierte Lagma-Huelle (CUDA). fp16/bf16 werden ueber fp32 gerechnet und zurueckgewandelt."""
    if not y.is_cuda: raise ValueError('lagma_cuda erwartet einen CUDA-Tensor')
    return _LagmaCUDA.apply(_as32(y).contiguous(), float(M), float(rho)).to(y.dtype)
def bprecap_cuda(x, preset='bprecap_d100'):
    """Fusioniertes BPreCap (CUDA): PreCap-Kern und Lagma-Huelle in einem Kernel, vorwaerts und rueckwaerts."""
    if not x.is_cuda: raise ValueError('bprecap_cuda erwartet einen CUDA-Tensor')
    q = BOUNDED[preset]; p = {k: float(q['core'][k]) for k in ('lam', 'alpha', 'b', 's', 'c')}; p.update(M=float(q['M']), rho=float(HULL_RHO))
    return _BPreCapCUDA.apply(_as32(x).contiguous(), p).to(x.dtype)

class Lagma(_torch.nn.Module):
    """Lagma(M, rho=0.8): weiche, feste Huelle. backend='auto' (CUDA-Kernel auf CUDA-Tensoren, sonst Referenz) oder 'ref'."""
    def __init__(self, M, rho=HULL_RHO, backend='auto'):
        super().__init__(); self.M, self.rho, self.backend = float(M), float(rho), backend
    def forward(self, y):
        return lagma_cuda(y, self.M, self.rho) if (self.backend == 'auto' and y.is_cuda) else lagma_hull(y, self.M, self.rho)
    def extra_repr(self): return f'M={self.M}, rho={self.rho}, backend={self.backend}'

class BPreCap(_torch.nn.Module):
    """BPreCap(depth=100 | 200): beschraenkt auf (-3, 3), E[f^2] = 1 und chi1 = 1 + 2/L fuer z ~ N(0, 1). Gedacht fuer tiefe Netze ohne Normierung.
    backend='auto' nutzt auf CUDA-Tensoren den fusionierten Kernel, sonst die Referenz; 'ref' erzwingt die Referenz."""
    def __init__(self, depth=100, backend='auto'):
        super().__init__()
        self.preset = f'bprecap_d{depth}'; self.backend = backend
        if self.preset not in BOUNDED: raise ValueError(f'depth muss 100 oder 200 sein, nicht {depth!r}')
    def forward(self, x): return bprecap_cuda(x, self.preset) if (self.backend == 'auto' and x.is_cuda) else bprecap_ref(x, self.preset)
    def extra_repr(self): return f'preset={self.preset}, M={BOUNDED[self.preset]["M"]}, backend={self.backend}'
