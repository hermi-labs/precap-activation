"""Test des CUDA-Kernels gegen die Referenz. Aufruf:  python test_cuda.py
Prueft: Werte (fp32/fp64), Gradienten, gradcheck (fp64), Randwerte, Training eines kleinen MLP, Tempo."""
import sys, time, math, json, platform, torch
from reference import precap_ref, precap_ref_grad, PRESETS
from precap_cuda import precap_cuda, PreCap
assert torch.cuda.is_available(), 'keine CUDA-GPU gefunden'
dev = 'cuda'; out = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                        python=platform.python_version(), checks={}); ok = True
def check(name, cond, info):
    global ok; ok &= bool(cond); out['checks'][name] = dict(ok=bool(cond), **info)
    print(('OK    ' if cond else 'FEHLER'), name, info, flush=True)
torch.manual_seed(0)
edge = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e-7, -1e-7, 0.5, 1.0, 1.2, 2.0, -2.0, 5.0, -5.0, 20.0, -20.0, 80.0, -80.0, 1e4, -1e4], device=dev)
for name, p in PRESETS.items():
    for dt, tol in ((torch.float32, 2e-6), (torch.float64, 1e-13)):
        x = torch.cat([torch.randn(1_000_000, device=dev, dtype=dt) * 3, edge.to(dt)])
        y_k = precap_cuda(x, **p); y_r = precap_ref(x, **p)
        err = ((y_k - y_r).abs() / (1 + y_r.abs())).max().item()
        check(f'{name} vorwaerts {str(dt)[6:]}', err < tol and torch.isfinite(y_k).all().item(), dict(max_rel_fehler=float('%.3g' % err)))
        xk = x.clone().requires_grad_(); xr = x.clone().requires_grad_(); g = torch.randn_like(x)
        precap_cuda(xk, **p).backward(g); precap_ref(xr, **p).backward(g)
        ga = g * precap_ref_grad(x, **p)
        e1 = ((xk.grad - xr.grad).abs() / (1 + xr.grad.abs())).max().item()
        e2 = ((xk.grad - ga).abs() / (1 + ga.abs())).max().item()
        check(f'{name} rueckwaerts {str(dt)[6:]}', max(e1, e2) < 10 * tol, dict(vs_autograd=float('%.3g' % e1), vs_analytisch=float('%.3g' % e2)))
    xg = (torch.randn(64, device=dev, dtype=torch.float64) * 3).requires_grad_()
    xg = xg + 0.05 * torch.sign(xg.detach())  # weg vom Knick bei 0 und w=c fuer finite Differenzen
    xg = xg.detach().requires_grad_()
    check(f'{name} gradcheck fp64', torch.autograd.gradcheck(lambda t: precap_cuda(t, **p), (xg,), eps=1e-6, atol=1e-8), {})
# kleines tiefes MLP: Kernel vs Referenz, identische Init, 50 Schritte
def mlp(act):
    torch.manual_seed(1); L = []
    for i in range(20): L += [torch.nn.Linear(256, 256), act]
    return torch.nn.Sequential(*L, torch.nn.Linear(256, 10)).to(dev)
class RefMod(torch.nn.Module):
    def __init__(s, p): super().__init__(); s.p = p
    def forward(s, x): return precap_ref(x, **s.p)
for name in ('precap0', 'spreccap_cs'):
    losses = []
    for act in (PreCap(name), RefMod(PRESETS[name])):
        m = mlp(act); opt = torch.optim.Adam(m.parameters(), 1e-4); g = torch.Generator(device=dev).manual_seed(2)
        for _ in range(50):
            xb = torch.randn(128, 256, device=dev, generator=g); yb = torch.randint(0, 10, (128,), device=dev, generator=g)
            l = torch.nn.functional.cross_entropy(m(xb), yb); opt.zero_grad(); l.backward(); opt.step()
        losses.append(l.item())
    check(f'{name} MLP-Training gleich', abs(losses[0] - losses[1]) < 1e-3, dict(loss_kernel=round(losses[0], 6), loss_referenz=round(losses[1], 6)))
# Tempo: vorwaerts+rueckwaerts, 16M Elemente fp32
def bench(f, x, n=50):
    for _ in range(5): y = f(x); y.backward(torch.ones_like(y)); x.grad = None
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): y = f(x); y.backward(torch.ones_like(y)); x.grad = None
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
x = torch.randn(16 * 1024 * 1024, device=dev).requires_grad_()
p = PRESETS['spreccap_cs']
tk = bench(lambda t: precap_cuda(t, **p), x); tr = bench(lambda t: precap_ref(t, **p), x)
ts = bench(torch.nn.functional.selu, x)
out['tempo_ms_16M_fwd_bwd'] = dict(kernel=round(tk, 3), referenz=round(tr, 3), selu_pytorch=round(ts, 3), speedup=round(tr / tk, 2))
print('Tempo (ms, 16M Elemente, vorwaerts+rueckwaerts):', out['tempo_ms_16M_fwd_bwd'])
out['alles_ok'] = bool(ok); json.dump(out, open('test_cuda_ergebnis.json', 'w'), indent=1)
print('\nERGEBNIS:', 'ALLES OK' if ok else 'FEHLER GEFUNDEN', '-> test_cuda_ergebnis.json')
sys.exit(0 if ok else 1)
