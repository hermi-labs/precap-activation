"""Test der fusionierten v0.3-Kernel (BPreCap, Lagma) in MLX/Metal gegen float64-NumPy und die MLX-Referenz, mit Tempo.
Aufruf (aus dem Repo-Ordner): PYTHONPATH=src python kernels/test_bprecap_mlx.py"""
import sys, time, json, math, platform
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from precap.mlx import BPreCap, Lagma, bprecap_ref_mlx, lagma_hull
from precap._presets import BOUNDED, HULL_RHO
ok = True; out = dict(mlx=mx.__version__, python=platform.python_version(), geraet=str(mx.default_device()), checks={})
def check(name, cond, info):
    global ok; ok &= bool(cond); out['checks'][name] = dict(ok=bool(cond), **info); print(('OK    ' if cond else 'FEHLER'), name, info, flush=True)
def hull64(y, M, rho=HULL_RHO):
    u = y / M; a = np.abs(u); k = np.tanh((a - rho) / (1 - rho))
    return M * np.where(a <= rho, u, np.sign(u) * (rho + (1 - rho) * k)), np.where(a <= rho, 1.0, 1 - k * k)
def bp64(x, preset):
    p = BOUNDED[preset]; q = p['core']; s, c = q['s'], q['c']
    t = np.tanh(q['b'] * x / s); w = s * t; ec = math.e ** c
    ce = np.where(w <= c, np.exp(np.minimum(w, c)) - 1, ec - 1 + ec * (w - c)); kk = q['lam'] * np.where(x > 0, 1.0, q['alpha'])
    dce = np.where(w <= c, np.exp(np.minimum(w, c)), ec); h, dh = hull64(kk * ce, p['M'])
    return h, kk * dce * q['b'] * (1 - t * t) * dh
rng = np.random.default_rng(0)
edge = np.array([0.0, -0.0, 1e-30, -1e-30, 1e-7, -1e-7, 0.5, 1.0, 2.0, -2.0, 5.0, -5.0, 20.0, -20.0, 80.0, -80.0, 1e4, -1e4])
x64 = np.concatenate([rng.standard_normal(1_000_000) * 4, edge]).astype(np.float32).astype(np.float64); x = mx.array(x64.astype(np.float32))
cot = mx.array(rng.standard_normal(x.size).astype(np.float32)); c64 = np.array(cot).astype(np.float64)
for d in (100, 200):
    pre = f'bprecap_d{d}'; f = BPreCap(d); y64, g64 = bp64(x64, pre)
    yk = np.array(f(x)); yr = np.array(bprecap_ref_mlx(x, pre))
    e64 = np.max(np.abs(yk - y64)); eml = np.max(np.abs(yk - yr))
    check(f'{pre} vorwaerts', e64 < 2e-6 and np.isfinite(yk).all() and np.abs(yk).max() <= 3.0, dict(vs_float64=float('%.3g' % e64), vs_mlx_ops=float('%.3g' % eml)))
    gk = np.array(mx.vjp(f, [x], [cot])[1][0]); gm = np.array(mx.vjp(lambda t: bprecap_ref_mlx(t, pre), [x], [cot])[1][0]); ga = c64 * g64
    e1 = np.max(np.abs(gk - ga) / (1 + np.abs(ga))); e2 = np.max(np.abs(gk - gm) / (1 + np.abs(gm)))
    check(f'{pre} rueckwaerts', e1 < 2e-5, dict(vs_float64_analytisch=float('%.3g' % e1), vs_mlx_autograd=float('%.3g' % e2)))
    xs = mx.array(rng.standard_normal((7, 33, 5)).astype(np.float32))
    check(f'{pre} Form 3D', f(xs).shape == xs.shape and np.allclose(np.array(f(xs)), np.array(bprecap_ref_mlx(xs, pre)), atol=1e-6), {})
    xh = xs.astype(mx.float16); check(f'{pre} float16-Eingabe', f(xh).dtype == mx.float16 and np.allclose(np.array(f(xh)).astype(np.float32), np.array(bprecap_ref_mlx(xh.astype(mx.float32), pre)), atol=2e-3), {})
for M in (BOUNDED['lb_d100']['M'], 3.0):
    L = Lagma(M); h64, dh64 = hull64(x64, M)
    e = np.max(np.abs(np.array(L(x)) - h64)); gk = np.array(mx.vjp(L, [x], [cot])[1][0]); e1 = np.max(np.abs(gk - c64 * dh64) / (1 + np.abs(c64 * dh64)))
    check(f'Lagma M={M:.4f}', e < 2e-6 and e1 < 2e-5, dict(vorwaerts_vs_float64=float('%.3g' % e), rueckwaerts_vs_float64=float('%.3g' % e1)))
xb = mx.random.normal((16 * 1024 * 1024,))
def bench(fn, n=30):
    g = mx.grad(lambda t: fn(t).sum())
    for _ in range(3): mx.eval(g(xb))
    t = time.perf_counter()
    for _ in range(n): mx.eval(g(xb))
    return (time.perf_counter() - t) / n * 1e3
tk = bench(BPreCap(100)); tr = bench(lambda t: bprecap_ref_mlx(t, 'bprecap_d100')); ts = bench(nn.selu)
tl = bench(Lagma(3.0)); tlr = bench(lambda t: lagma_hull(t, 3.0))
out['tempo_ms_16M_fwd_bwd'] = dict(bprecap_kernel=round(tk, 3), bprecap_referenz=round(tr, 3), lagma_kernel=round(tl, 3), lagma_referenz=round(tlr, 3), selu_mlx=round(ts, 3),
                                   speedup_bprecap=round(tr / tk, 2), speedup_lagma=round(tlr / tl, 2))
print('Tempo (ms, 16M Elemente, vorwaerts+rueckwaerts):', out['tempo_ms_16M_fwd_bwd'])
out['alles_ok'] = bool(ok); json.dump(out, open(Path(__file__).resolve().parent / 'test_bprecap_mlx_ergebnis.json', 'w'), indent=1)
print('\nERGEBNIS:', 'ALLES OK' if ok else 'FEHLER GEFUNDEN', '-> kernels/test_bprecap_mlx_ergebnis.json'); sys.exit(0 if ok else 1)
