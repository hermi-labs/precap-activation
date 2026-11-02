# C6 Aequivalenztest (PyTorch-Seite): prueft, dass c6_run.py dieselbe Rechenvorschrift wie studien/b10/b10_run.py (MLX) umsetzt.
# Braucht test/mlx_ref.npz und test/mlx_ref2.npz (erzeugt mit mlx_referenz.py und mlx_referenz2.py in der MLX-Umgebung).
# Begruendete Schwellen:
#  - Daten und Startgewichte: bitgleich (gleiche numpy-Rechnung bzw. feste Normierungskonstanten).
#  - Logits und Kalibrierung bei der Init: float32-Rundung ueber 100 Schichten, relativ < 1e-3.
#  - Gradienten: float32 ist in 100 Schichten schlecht konditioniert. Massstab ist die float64-Rechnung (PyTorch): die Abweichung
#    PyTorch-float32 gegen MLX-float32 darf hoechstens 3-mal so gross sein wie die groessere der beiden Abweichungen zur float64-Rechnung.
#  - Verlust bei der Init: wie Gradienten, Massstab float64.
#  - LR-Plan: k-tes Update nutzt schedule(k-1), wie MLX (mit SGD in MLX geprueft).
#  - AdamW an identischen Gradienten: hoechstens 8 float32-Stellen (max abs < 1e-7 bei Gewichten um 0,06).
#  - Nach 172 Schritten: Testfehler innerhalb 2 pp (Trajektorien laufen durch Rundung auseinander, das ist erwartet).
# Ausgabe "AEQUIVALENZ OK" oder Fehler.
import sys, hashlib
from pathlib import Path
import numpy as np, torch
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent)); import c6_run as C
C.setup(sys.argv[1] if len(sys.argv) > 1 else 'cpu'); r1 = np.load(H / 'mlx_ref.npz'); r2 = np.load(H / 'mlx_ref2.npz'); R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
for act, lr, wd in (('kL2B_tr', 1e-4, 0.3), ('tat', 1e-4, 0.3), ('selu_tr', 2e-4, 0.3), ('LB_tr', 1e-4, 0.3)):
    D = C.data_for(act); xtr, ytr, xv, yv, xte, yte = D
    h = np.frombuffer(hashlib.sha256(np.ascontiguousarray(xtr).tobytes() + np.ascontiguousarray(xte).tobytes()).digest(), np.uint8)
    ok(f'{act}_daten_bitgleich', np.array_equal(h, r1[f'{act}_datahash']))
    m = C.Net(act, 42168).to(C.DEV).eval()
    ok(f'{act}_init_bitgleich', all(np.array_equal(m.hid[i].weight.detach().cpu().numpy(), r1[f'{act}_W{i}']) for i in (0, 50)) and np.array_equal(m.out.weight.detach().cpu().numpy(), r1[f'{act}_Wout']))
    with torch.no_grad(): lg = m(torch.from_numpy(xte[:512]).to(C.DEV)).cpu().numpy()
    e = rel(lg, r1[f'{act}_logits0']); ok(f'{act}_logits_init', e < 1e-3, f'rel {e:.1e}')
    RA = C.calib_asym(C.inputs(m, xtr[:C.NCAL]))
    e = max(rel(np.array(RA[c], dtype=np.float64), r1[f'{act}_RA_{c}']) for c in C.CALS4); ok(f'{act}_kalibrierung', e < 1e-3, f'max rel {e:.1e}')
    err = 1 - C.evaluate(C.qforward(m, 'i4', RA['mse4'], asym=True), xte[:2500], yte[:2500]); ref = float(r1[f'{act}_w4a_mse4_err2500'][0])
    ok(f'{act}_w4a_init', abs(err - ref) <= 0.004, f'{err:.4f} gegen {ref:.4f}')
    b = np.random.default_rng(42168 + 777).permutation(len(xtr))[:C.BATCH]; G = {}
    for dt in (torch.float32, torch.float64):
        mm = C.Net(act, 42168).to(C.DEV, dtype=dt); loss = torch.nn.functional.cross_entropy(mm(torch.from_numpy(xtr[b]).to(C.DEV, dt)), torch.from_numpy(ytr[b]).to(C.DEV)); loss.backward()
        G[dt] = (float(loss), {k: t.detach().double().cpu().numpy() for k, t in (('g0', mm.hid[0].weight.grad), ('g50', mm.hid[50].weight.grad), ('gout', mm.out.weight.grad))})
    l32, l64, lm = G[torch.float32][0], G[torch.float64][0], float(r2[f'{act}_loss'][0])   # Massstab float64 wie bei den Gradienten
    ok(f'{act}_verlust_init', abs(l32 - lm) <= 3 * max(abs(l32 - l64), abs(lm - l64)) + 1e-6, f'torch {l32:.7f}, mlx {lm:.7f}, f64 {l64:.7f}')
    for k in ('g0', 'g50', 'gout'):
        t32, t64, m32 = G[torch.float32][1][k], G[torch.float64][1][k], r2[f'{act}_{k}']
        d, a1, a2 = rel(t32, m32), rel(t32, t64), rel(m32, t64)
        ok(f'{act}_gradient_{k}', d <= 3 * max(a1, a2) and max(a1, a2) < 5e-2, f'torch-mlx {d:.1e}, torch-f64 {a1:.1e}, mlx-f64 {a2:.1e}')
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    r = np.random.default_rng(42168 + 777); perm = r.permutation(len(xtr)); pos = 0; X = torch.from_numpy(xtr).to(C.DEV); Y = torch.from_numpy(ytr).to(C.DEV)
    for s in range(1, 173):
        if pos + C.BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        bb = torch.from_numpy(perm[pos:pos + C.BATCH]).to(C.DEV); pos += C.BATCH
        for g in opt.param_groups: g['lr'] = C.lr_at(lr, s - 1)
        loss = torch.nn.functional.cross_entropy(m(X[bb]), Y[bb]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval(); te = 1 - C.evaluate(m, xte, yte); ref = float(r1[f'{act}_test_err_172'][0]); ok(f'{act}_172_schritte_nahe', abs(te - ref) < 0.02, f'{te:.4f} gegen {ref:.4f}')
w = torch.nn.Parameter(torch.from_numpy(r2['adam_p0'].copy())); opt = torch.optim.AdamW([w], lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.3)
for s, g in enumerate(r2['adam_G'], 1):
    for grp in opt.param_groups: grp['lr'] = C.lr_at(1e-4, s - 1)
    w.grad = torch.from_numpy(g.copy()); opt.step()
e = float(np.abs(w.detach().numpy() - r2['adam_p3']).max()); ok('adamw_gleich_mlx', e < 1e-7, f'max abs {e:.1e}')
# MLX meldet als learning_rate die zuletzt benutzte LR: nach 172 Updates schedule(171); das k-te Update benutzt schedule(k-1) (geprueft mit SGD, 01.11.)
ok('lr_ende_wie_mlx', abs(C.lr_at(1e-4, 171) - float(r1['kL2B_tr_lr_end_172'][0])) <= 2 * float(np.spacing(np.float32(1e-4))),   # MLX rechnet Kosinus und LR in float32: hoechstens 2 float32-Stellen
    f'{C.lr_at(1e-4, 171):.10e} gegen {float(r1["kL2B_tr_lr_end_172"][0]):.10e}')
print('AEQUIVALENZ OK' if all(R.values()) else 'AEQUIVALENZ FEHLER: ' + ', '.join(k for k, v in R.items() if not v))
sys.exit(0 if all(R.values()) else 1)
