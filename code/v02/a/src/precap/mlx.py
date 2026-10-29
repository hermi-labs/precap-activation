"""MLX-Umsetzung (Apple Silicon) mit fusioniertem Metal-Kernel, vorwaerts + rueckwaerts.
Gerechnet wird in float32; das Ergebnis wird in den Eingangstyp zurueckgewandelt."""
import math
import mlx.core as mx
import mlx.nn as _nn
from ._presets import PRESETS, DEFAULT

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
