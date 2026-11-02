"""Test der fusionierten v0.3-Kernel (BPreCap, Lagma) in CUDA (torch.cuda.jiterator) gegen float64 und die PyTorch-Referenz, mit Tempo.
Aufruf (aus dem Repo-Ordner, mit CUDA): python kernels/test_bprecap_cuda.py"""
import sys, time, json, math, platform
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from precap.torch import BPreCap, Lagma, bprecap_ref, lagma_hull
from precap._presets import BOUNDED
assert torch.cuda.is_available(), 'CUDA nicht verfuegbar'
torch.backends.cuda.matmul.allow_tf32 = False
ok = True; out = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), python=platform.python_version(), checks={})
def check(name, cond, info):
    global ok; ok &= bool(cond); out['checks'][name] = dict(ok=bool(cond), **info); print(('OK    ' if cond else 'FEHLER'), name, info, flush=True)
rng = np.random.default_rng(0)
edge = np.array([0.0, -0.0, 1e-30, -1e-30, 1e-7, -1e-7, 0.5, 1.0, 2.0, -2.0, 5.0, -5.0, 20.0, -20.0, 80.0, -80.0, 1e4, -1e4])
x32 = torch.from_numpy(np.concatenate([rng.standard_normal(1_000_000) * 4, edge]).astype(np.float32))
x64 = x32.double().requires_grad_(); cot = torch.from_numpy(rng.standard_normal(x32.numel()).astype(np.float32))
for d in (100, 200):
    pre = f'bprecap_d{d}'; f = BPreCap(d)
    y64 = bprecap_ref(x64, pre); g64, = torch.autograd.grad(y64, x64, cot.double())
    xc = x32.cuda().requires_grad_(); yk = f(xc); gk, = torch.autograd.grad(yk, xc, cot.cuda())
    xr = x32.cuda().requires_grad_(); yr = bprecap_ref(xr, pre); gr, = torch.autograd.grad(yr, xr, cot.cuda())
    e64 = float((yk.detach().double().cpu() - y64.detach()).abs().max()); eref = float((yk - yr).detach().abs().max())
    check(f'{pre} vorwaerts', e64 < 2e-6 and torch.isfinite(yk).all() and float(yk.detach().abs().max()) <= 3.0, dict(vs_float64=float('%.3g' % e64), vs_torch_ref=float('%.3g' % eref)))
    e1 = float(((gk.double().cpu() - g64).abs() / (1 + g64.abs())).max()); e2 = float(((gk - gr).abs() / (1 + gr.abs())).max())
    check(f'{pre} rueckwaerts', e1 < 2e-5, dict(vs_float64_autograd=float('%.3g' % e1), vs_torch_ref=float('%.3g' % e2)))
    for dt in (torch.float16, torch.bfloat16):
        xh = torch.randn(1000, device='cuda').to(dt); yh = f(xh)
        check(f'{pre} {str(dt)[6:]}', yh.dtype == dt and float((yh.float() - bprecap_ref(xh.float(), pre)).abs().max()) < 2e-2, {})
    xs = torch.randn(7, 33, 5, device='cuda'); check(f'{pre} Form 3D', f(xs).shape == xs.shape and torch.allclose(f(xs), bprecap_ref(xs, pre), atol=1e-6), {})
    check(f'{pre} gradcheck float64', torch.autograd.gradcheck(lambda t: BPreCap(d)(t), ((torch.randn(64, device='cuda', dtype=torch.float64) * 4 + 0.05).requires_grad_(),), eps=1e-6, atol=1e-6), {})
for M in (BOUNDED['lb_d100']['M'], 3.0):
    L = Lagma(M); h64 = lagma_hull(x64, M); dh64, = torch.autograd.grad(h64, x64, cot.double())
    xc = x32.cuda().requires_grad_(); hk = L(xc); gk, = torch.autograd.grad(hk, xc, cot.cuda())
    e = float((hk.detach().double().cpu() - h64.detach()).abs().max()); e1 = float(((gk.double().cpu() - dh64).abs() / (1 + dh64.abs())).max())
    check(f'Lagma M={M:.4f}', e < 2e-6 and e1 < 2e-5, dict(vorwaerts_vs_float64=float('%.3g' % e), rueckwaerts_vs_float64=float('%.3g' % e1)))
xb = torch.randn(16 * 1024 * 1024, device='cuda')
def bench(fn, n=30):
    def once():
        t = xb.detach().requires_grad_(); fn(t).sum().backward()
    for _ in range(3): once()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): once()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
tk = bench(BPreCap(100)); tr = bench(lambda t: bprecap_ref(t, 'bprecap_d100')); ts = bench(torch.nn.functional.selu)
tl = bench(Lagma(3.0)); tlr = bench(lambda t: lagma_hull(t, 3.0))
out['tempo_ms_16M_fwd_bwd'] = dict(bprecap_kernel=round(tk, 3), bprecap_referenz=round(tr, 3), lagma_kernel=round(tl, 3), lagma_referenz=round(tlr, 3), selu_torch=round(ts, 3),
                                   speedup_bprecap=round(tr / tk, 2), speedup_lagma=round(tlr / tl, 2))
print('Tempo (ms, 16M Elemente, vorwaerts+rueckwaerts):', out['tempo_ms_16M_fwd_bwd'])
out['alles_ok'] = bool(ok); json.dump(out, open(Path(__file__).resolve().parent / 'test_bprecap_cuda_ergebnis.json', 'w'), indent=1)
print('\nERGEBNIS:', 'ALLES OK' if ok else 'FEHLER GEFUNDEN', '-> kernels/test_bprecap_cuda_ergebnis.json'); sys.exit(0 if ok else 1)
