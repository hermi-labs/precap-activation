"""Erweiterter CUDA-Test fuer das Paket precap. Aufruf aus dem Repo-Wurzelordner:
    python kernels/test_cuda_erweitert.py
Prueft: fp16/bf16 (Werte, Gradienten, Rueckgabetyp), NaN/Inf, nicht zusammenhaengende Tensoren,
Paket-Modul auf CUDA, Tempo gegen ReLU, SiLU, GELU, ELU, SELU (16M Elemente, vorwaerts+rueckwaerts).
Ergebnis: kernels/test_cuda_erweitert_ergebnis.json"""
import sys, time, json, platform, pathlib, torch
R = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / 'src'))
from precap import PreCap, PRESETS, precap_ref
from precap.torch import precap_cuda
assert torch.cuda.is_available(), 'keine CUDA-GPU gefunden'
dev = 'cuda'; ok = True
out = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(), checks={})
def check(name, cond, info=None):
    global ok; ok &= bool(cond); out['checks'][name] = dict(ok=bool(cond), **(info or {}))
    print(('OK    ' if cond else 'FEHLER'), name, info or '', flush=True)
torch.manual_seed(0)
for name, p in PRESETS.items():
    x64 = torch.randn(1_000_000, device=dev, dtype=torch.float64) * 3
    for dt, tol in ((torch.float16, 2e-3), (torch.bfloat16, 1.6e-2)):
        x = x64.to(dt); xr = x.double()
        y = precap_cuda(x, **p); r = precap_ref(xr, **p)
        e = ((y.double() - r).abs() / (1 + r.abs())).max().item()
        xg = x.clone().requires_grad_(); precap_cuda(xg, **p).sum().backward()
        xrg = xr.clone().requires_grad_(); precap_ref(xrg, **p).sum().backward()
        eg = ((xg.grad.double() - xrg.grad).abs() / (1 + xrg.grad.abs())).max().item()
        check(f'{name} {str(dt)[6:]}', y.dtype == dt and xg.grad.dtype == dt and e < tol and eg < 2 * tol,
              dict(max_rel_vorw=float('%.3g' % e), max_rel_rueckw=float('%.3g' % eg)))
    sp = torch.tensor([float('nan'), float('inf'), -float('inf'), 0.0], device=dev)
    y = precap_cuda(sp, **p); r = precap_ref(sp.double(), **p).float()
    check(f'{name} NaN/Inf', torch.isnan(y[0]).item() and torch.isfinite(y[1:]).all().item() and torch.allclose(y[1:], r[1:], atol=1e-6),
          dict(werte=[float(v) for v in y.cpu()]))
# nicht zusammenhaengend und mehrdimensional
x = torch.randn(512, 1024, device=dev).t()
check('nicht zusammenhaengend', torch.allclose(precap_cuda(x, **PRESETS['spreccap_cs']), precap_ref(x, **PRESETS['spreccap_cs']), atol=2e-6))
# Paket-Modul waehlt Kernel automatisch, Ergebnis gleich Referenz-Backend
m, mr = PreCap('spreccap_e03'), PreCap('spreccap_e03', backend='ref'); x = torch.randn(4096, device=dev)
check('PreCap auto = ref', torch.allclose(m(x), mr(x), atol=2e-6))
# Tempo
def bench(f, x, n=50):
    for _ in range(5): f(x).sum().backward()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): x.grad = None; f(x).sum().backward()
    torch.cuda.synchronize(); return round((time.perf_counter() - t) / n * 1e3, 3)
x = torch.randn(16 * 1024 * 1024, device=dev).requires_grad_()
F = torch.nn.functional
tempo = {'precap_kernel': bench(PreCap('spreccap_cs'), x), 'precap_referenz': bench(PreCap('spreccap_cs', backend='ref'), x),
         'relu': bench(F.relu, x), 'silu': bench(F.silu, x), 'gelu': bench(F.gelu, x), 'elu': bench(F.elu, x), 'selu': bench(F.selu, x)}
xh = x.detach().half().requires_grad_()
tempo['precap_kernel_fp16'] = bench(PreCap('spreccap_cs'), xh); tempo['selu_fp16'] = bench(F.selu, xh)
out['tempo_ms_16M_fwd_bwd'] = tempo; print('Tempo (ms):', tempo)
out['alle_ok'] = bool(ok); print('ALLE OK' if ok else 'ES GAB FEHLER')
json.dump(out, open(pathlib.Path(__file__).with_name('test_cuda_erweitert_ergebnis.json'), 'w'), indent=1)
