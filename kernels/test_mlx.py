"""Test des MLX/Metal-Kernels gegen die Referenz (MLX-Operationen und float64-NumPy). Aufruf: python test_mlx.py"""
import sys, time, json, math, platform
import numpy as np, mlx.core as mx
from precap_mlx import make_precap, precap_ref_mlx, PRESETS
ok = True; out = dict(mlx=mx.__version__, python=platform.python_version(), geraet=str(mx.default_device()), checks={})
def check(name, cond, info):
    global ok; ok &= bool(cond); out['checks'][name] = dict(ok=bool(cond), **info); print(('OK    ' if cond else 'FEHLER'), name, info, flush=True)
def np_ref(x, lam, alpha, b, s, c):
    t = np.tanh(b * x / s); w = s * t; ec = math.e ** c
    ce = np.where(w <= c, np.exp(np.minimum(w, c)) - 1, ec - 1 + ec * (w - c)); y = lam * np.where(x > 0, ce, alpha * ce)
    dce = np.where(w <= c, np.exp(np.minimum(w, c)), ec); g = lam * np.where(x > 0, 1.0, alpha) * dce * b * (1 - t * t)
    return y, g
rng = np.random.default_rng(0)
edge = np.array([0.0, -0.0, 1e-30, -1e-30, 1e-7, -1e-7, 0.5, 1.0, 1.2, 2.0, -2.0, 5.0, -5.0, 20.0, -20.0, 80.0, -80.0, 1e4, -1e4])
x64 = np.concatenate([rng.standard_normal(1_000_000) * 3, edge]).astype(np.float32).astype(np.float64)
x = mx.array(x64.astype(np.float32))
for name, p in PRESETS.items():
    f = make_precap(**p); yr64, gr64 = np_ref(x64, **p)
    yk = np.array(f(x)); yr = np.array(precap_ref_mlx(x, **p))
    e64 = np.max(np.abs(yk - yr64) / (1 + np.abs(yr64))); eml = np.max(np.abs(yk - yr) / (1 + np.abs(yr)))
    check(f'{name} vorwaerts', e64 < 2e-6 and np.isfinite(yk).all(), dict(vs_float64=float('%.3g' % e64), vs_mlx_ops=float('%.3g' % eml)))
    cot = mx.array(rng.standard_normal(x.size).astype(np.float32))
    gk = np.array(mx.vjp(f, [x], [cot])[1][0]); gm = np.array(mx.vjp(lambda t: precap_ref_mlx(t, **p), [x], [cot])[1][0])
    ga = np.array(cot).astype(np.float64) * gr64
    e1 = np.max(np.abs(gk - ga) / (1 + np.abs(ga))); e2 = np.max(np.abs(gk - gm) / (1 + np.abs(gm)))
    check(f'{name} rueckwaerts', e1 < 2e-5, dict(vs_float64_analytisch=float('%.3g' % e1), vs_mlx_autograd=float('%.3g' % e2)))
    # Formen und Typen
    xs = mx.array(rng.standard_normal((7, 33, 5)).astype(np.float32)); ys = f(xs)
    check(f'{name} Form 3D', ys.shape == xs.shape and np.allclose(np.array(ys), np.array(precap_ref_mlx(xs, **p)), atol=1e-6), {})
    xh = xs.astype(mx.float16); yh = f(xh)
    check(f'{name} float16-Eingabe', np.allclose(np.array(yh), np.array(precap_ref_mlx(xh.astype(mx.float32), **p)), atol=1e-6), dict(ausgabe_typ=str(yh.dtype)))
# Tempo: vorwaerts+rueckwaerts, 16M Elemente
p = PRESETS['spreccap_cs']; f = make_precap(**p); xb = mx.random.normal((16 * 1024 * 1024,))
def bench(fn, n=30):
    g = mx.grad(lambda t: fn(t).sum())
    for _ in range(3): mx.eval(g(xb))
    t = time.perf_counter()
    for _ in range(n): mx.eval(g(xb))
    return (time.perf_counter() - t) / n * 1e3
import mlx.nn as nn
tk = bench(f); tr = bench(lambda t: precap_ref_mlx(t, **p)); ts = bench(nn.selu)
out['tempo_ms_16M_fwd_bwd'] = dict(kernel=round(tk, 3), referenz=round(tr, 3), selu_mlx=round(ts, 3), speedup=round(tr / tk, 2))
print('Tempo (ms, 16M Elemente, vorwaerts+rueckwaerts):', out['tempo_ms_16M_fwd_bwd'])
out['alles_ok'] = bool(ok); json.dump(out, open('test_mlx_ergebnis.json', 'w'), indent=1)
print('\nERGEBNIS:', 'ALLES OK' if ok else 'FEHLER GEFUNDEN', '-> test_mlx_ergebnis.json'); sys.exit(0 if ok else 1)
