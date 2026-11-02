"""MLX-Umsetzung (Apple Silicon) mit fusioniertem Metal-Kernel, vorwaerts + rueckwaerts.
Gerechnet wird in float32; das Ergebnis wird in den Eingangstyp zurueckgewandelt."""
import math
import mlx.core as mx
import mlx.nn as _nn
from ._presets import PRESETS, DEFAULT, BOUNDED, HULL_RHO

_HEAD = """
    uint i = thread_position_in_grid.x;
    if (i >= n_el[0]) return;
    float lam = prm[0], alpha = prm[1], b = prm[2], s = prm[3], c = prm[4];
    float x  = float(inp[i]);
    float t  = metal::precise::tanh(b * x / s);
    float w  = s * t;
    float ec = metal::precise::exp(c);
"""
_FWD = _HEAD + """
    float ce = (w <= c) ? (metal::precise::exp(w) - 1.0f) : (ec - 1.0f + ec * (w - c));
    out[i] = (x > 0.0f) ? lam * ce : lam * alpha * ce;
"""
_BWD = _HEAD + """
    float dce = (w <= c) ? metal::precise::exp(w) : ec;
    float k   = (x > 0.0f) ? lam : lam * alpha;
    out[i] = float(cot[i]) * k * dce * b * (1.0f - t * t);
"""
_kf = mx.fast.metal_kernel(name='precap_fwd', input_names=['inp', 'prm', 'n_el'], output_names=['out'], source=_FWD)
_kb = mx.fast.metal_kernel(name='precap_bwd', input_names=['inp', 'cot', 'prm', 'n_el'], output_names=['out'], source=_BWD)

def _launch(k, ins, x):
    n = x.size
    return k(inputs=ins + [mx.array([n], dtype=mx.uint32)], grid=(max(n, 1), 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[x.shape], output_dtypes=[mx.float32])[0]

def make_precap(lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    prm = mx.array([lam, alpha, b, s, c], dtype=mx.float32)
    @mx.custom_function
    def f(x):
        return _launch(_kf, [x.astype(mx.float32), prm], x)
    @f.vjp
    def f_vjp(primals, cotangent, output):
        # MLX uebergibt bei nur einem Eingang das Array selbst (kein Tupel)
        x = primals[0] if isinstance(primals, (tuple, list)) else primals
        g = _launch(_kb, [x.astype(mx.float32), cotangent.astype(mx.float32), prm], x)
        g = g.astype(x.dtype)
        return (g,) if isinstance(primals, (tuple, list)) else g
    return lambda x: f(x).astype(x.dtype)

def precap_ref_mlx(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    w = s * mx.tanh(b * x / s); ec = math.exp(c)
    ce = mx.where(w <= c, mx.exp(mx.minimum(w, c)) - 1, ec - 1 + ec * (w - c))
    return lam * mx.where(x > 0, ce, alpha * ce)

class PreCap(_nn.Module):
    """PreCap(preset='spreccap_cs' | 'spreccap_e03' | 'spreccap' | 'precap0', **parameter)"""
    def __init__(self, preset=DEFAULT, **kw):
        super().__init__()
        if preset not in PRESETS: raise ValueError(f'unbekanntes preset {preset!r}, erlaubt: {list(PRESETS)}')
        self._fn = make_precap(**{**PRESETS[preset], **kw})
    def __call__(self, x): return self._fn(x)

# --- v0.3: Lagma-Huelle und BPreCap, Referenz und fusionierter Metal-Kernel ------------------------------------
def lagma_hull(y, M, rho=HULL_RHO):
    """Lagma-Huelle h_M(y) = M * phi_rho(y / M): Identitaet fuer |y| <= rho*M, darueber C1 gegen +-M (in den Studien: brake)."""
    u = y / M; a = mx.abs(u); k = mx.tanh((a - rho) / (1 - rho))
    return M * mx.where(a <= rho, u, mx.sign(u) * (rho + (1 - rho) * k))

def bprecap_ref_mlx(x, preset='bprecap_d100'):
    p = BOUNDED[preset]
    return lagma_hull(precap_ref_mlx(x, **p['core']), p['M'])

# Huelle in float32: k = tanh((|u| - rho)/(1 - rho)); h = y fuer |u| <= rho, sonst sign(u) M (rho + (1 - rho) k); h'(y) = 1 bzw. 1 - k^2
_HULL = """
    float u = y / M; float a = metal::abs(u); bool lin = (a <= rho);
    float k = lin ? 0.0f : metal::precise::tanh((a - rho) / (1.0f - rho));
"""
_LG_FWD = """
    uint i = thread_position_in_grid.x;
    if (i >= n_el[0]) return;
    float M = prm[0], rho = prm[1];
    float y = float(inp[i]);
""" + _HULL + """
    out[i] = lin ? y : metal::sign(u) * M * (rho + (1.0f - rho) * k);
"""
_LG_BWD = """
    uint i = thread_position_in_grid.x;
    if (i >= n_el[0]) return;
    float M = prm[0], rho = prm[1];
    float y = float(inp[i]);
""" + _HULL + """
    out[i] = float(cot[i]) * (lin ? 1.0f : 1.0f - k * k);
"""
_BP_HEAD = _HEAD + """
    float M = prm[5], rho = prm[6];
"""
_BP_FWD = _BP_HEAD + """
    float ce = (w <= c) ? (metal::precise::exp(w) - 1.0f) : (ec - 1.0f + ec * (w - c));
    float y  = (x > 0.0f) ? lam * ce : lam * alpha * ce;
""" + _HULL + """
    out[i] = lin ? y : metal::sign(u) * M * (rho + (1.0f - rho) * k);
"""
_BP_BWD = _BP_HEAD + """
    float ce  = (w <= c) ? (metal::precise::exp(w) - 1.0f) : (ec - 1.0f + ec * (w - c));
    float dce = (w <= c) ? metal::precise::exp(w) : ec;
    float kk  = (x > 0.0f) ? lam : lam * alpha;
    float y   = kk * ce;
""" + _HULL + """
    out[i] = float(cot[i]) * kk * dce * b * (1.0f - t * t) * (lin ? 1.0f : 1.0f - k * k);
"""
_klf = mx.fast.metal_kernel(name='lagma_fwd', input_names=['inp', 'prm', 'n_el'], output_names=['out'], source=_LG_FWD)
_klb = mx.fast.metal_kernel(name='lagma_bwd', input_names=['inp', 'cot', 'prm', 'n_el'], output_names=['out'], source=_LG_BWD)
_kbf = mx.fast.metal_kernel(name='bprecap_fwd', input_names=['inp', 'prm', 'n_el'], output_names=['out'], source=_BP_FWD)
_kbb = mx.fast.metal_kernel(name='bprecap_bwd', input_names=['inp', 'cot', 'prm', 'n_el'], output_names=['out'], source=_BP_BWD)

def _make(kf, kb, prm):
    @mx.custom_function
    def f(x):
        return _launch(kf, [x.astype(mx.float32), prm], x)
    @f.vjp
    def f_vjp(primals, cotangent, output):
        x = primals[0] if isinstance(primals, (tuple, list)) else primals
        g = _launch(kb, [x.astype(mx.float32), cotangent.astype(mx.float32), prm], x).astype(x.dtype)
        return (g,) if isinstance(primals, (tuple, list)) else g
    return lambda x: f(x).astype(x.dtype)

def make_lagma(M, rho=HULL_RHO):
    """Fusionierte Lagma-Huelle (Metal), vorwaerts und rueckwaerts."""
    return _make(_klf, _klb, mx.array([M, rho], dtype=mx.float32))

def make_bprecap(lam, alpha, b, M, s=1.5, c=1.0, rho=HULL_RHO):
    """Fusioniertes BPreCap = Lagma-Huelle um den PreCap-Kern (Metal), vorwaerts und rueckwaerts."""
    return _make(_kbf, _kbb, mx.array([lam, alpha, b, s, c, M, rho], dtype=mx.float32))

class Lagma(_nn.Module):
    """Lagma(M, rho=0.8): weiche, feste Huelle. backend='kernel' (Standard) oder 'ref'."""
    def __init__(self, M, rho=HULL_RHO, backend='kernel'):
        super().__init__()
        self.M, self.rho = float(M), float(rho)
        self._fn = make_lagma(self.M, self.rho) if backend == 'kernel' else (lambda y: lagma_hull(y, self.M, self.rho))
    def __call__(self, y): return self._fn(y)

class BPreCap(_nn.Module):
    """BPreCap(depth=100 | 200): tiefenkalibrierter PreCap-Kern in der Lagma-Huelle. backend='kernel' (Standard) oder 'ref'."""
    def __init__(self, depth=100, backend='kernel'):
        super().__init__()
        self.preset = f'bprecap_d{depth}'
        if self.preset not in BOUNDED: raise ValueError(f'depth muss 100 oder 200 sein, nicht {depth!r}')
        p = BOUNDED[self.preset]
        self._fn = make_bprecap(M=p['M'], **{k: p['core'][k] for k in ('lam', 'alpha', 'b', 's', 'c')}) if backend == 'kernel' else (lambda x: bprecap_ref_mlx(x, self.preset))
    def __call__(self, x): return self._fn(x)
