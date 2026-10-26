"""PreCap/SPreCap als fusionierter Metal-Kernel fuer MLX (Apple Silicon), vorwaerts + rueckwaerts.
Gleiche Formel wie reference.py. Rechnet in float32 (float16/bfloat16 werden ueber float32 gerechnet)."""
import math
import mlx.core as mx

PRESETS = {
    'precap0':     dict(lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0),
    'spreccap':    dict(lam=0.80676869644, alpha=2.39695130922, b=1.0, s=1.5, c=1.0),
    'spreccap_cs': dict(lam=1.102702500, alpha=2.053776267, b=0.71188112, s=1.5, c=1.0),
}
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
        (x,) = primals
        return (_launch(_kb, [x.astype(mx.float32), cotangent.astype(mx.float32), prm], x),)
    return f

def precap_ref_mlx(x, lam=1.0, alpha=1.0, b=1.0, s=1.5, c=1.0):
    w = s * mx.tanh(b * x / s); ec = math.exp(c)
    ce = mx.where(w <= c, mx.exp(mx.minimum(w, c)) - 1, ec - 1 + ec * (w - c))
    return lam * mx.where(x > 0, ce, alpha * ce)

try:
    import mlx.nn as nn
    class PreCap(nn.Module):
        def __init__(self, preset='precap0', **kw):
            super().__init__(); self._fn = make_precap(**{**PRESETS[preset], **kw})
        def __call__(self, x): return self._fn(x)
except ImportError:
    pass
