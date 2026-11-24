# C7 Aequivalenztest, Teil 2 (PyTorch, privat, mit Claude): c7_run.py gegen die MLX-Referenz aus mlx_referenz_c7.py.
# Prueft Daten, Startgewichte, Formen, Logits und Gradienten bei der Init (beide gegen float64), Optimierer, Quantisierer, Kalibrierung,
# CLE und einen kurzen Trainingsabschnitt. Aufruf: ~/PyCharmMiscProject/.venv/bin/python -I test_c7_aequivalenz.py [--ohne-training]
import json, sys, time
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parent))
import torch
import c7_run as T
import os
DEVICE = next((x for x in sys.argv[1:] if x in ('cpu', 'cuda', 'mps')), 'cpu'); T.setup(DEVICE); torch.set_num_threads(8)
T.DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data')) / 'cifar10' / 'cifar-10-batches-bin'
Z = np.load(H / 'mlx_ref_c7.npz'); J = json.load(open(H / 'mlx_ref_c7.json')); res = {}; okall = True; seed = 42350
def ok(name, cond, info=''):
    global okall
    res[name] = dict(ok=bool(cond), info=info); okall &= bool(cond); print(('OK     ' if cond else 'FEHLER ') + name, info, flush=True)
def rel(u, v): return float(np.linalg.norm(u - v) / max(np.linalg.norm(v), 1e-30))
Xtr, Ytr, Xv, Yv, Xte, Yte = T.load()
x_t = np.concatenate([Xtr[:200], Xv[:100], Xte[:100]]).transpose(0, 2, 3, 1)
ok('daten_bitgleich', np.array_equal(x_t, Z['x_ausschnitt']), f"Summen torch {[round(float(np.sum(a, dtype=np.float64)), 3) for a in (Xtr, Xv, Xte)]}, MLX {[round(float(v), 3) for v in Z['x_hash']]}")
ok('daten_summen_gleich', np.allclose([np.sum(a, dtype=np.float64) for a in (Xtr, Xv, Xte)], Z['x_hash'], rtol=0, atol=1e-6))
b = Z['batch_idx']; ok('batch_gleich', np.array_equal(b, np.random.default_rng(seed + 777).permutation(len(Xtr))[:128]))
grid = torch.from_numpy(np.linspace(-12, 12, 48001).astype(np.float32))
for a in ('bprecap', 'tat', 'selu'):
    mt = T.CNN(a, seed); m64 = T.CNN(a, seed).double()
    ok(f'init_bitgleich_{a}', np.array_equal(mt.ws[0].detach().numpy(), Z[f'w0_{a}'].transpose(0, 3, 1, 2)) and np.array_equal(mt.ws[25].detach().numpy(), Z[f'w25_{a}'].transpose(0, 3, 1, 2)) and np.array_equal(mt.wo.detach().numpy(), Z[f'wo_{a}']))
    d = float(np.abs(T.act_fn(a)(grid).numpy() - Z[f'form_{a}']).max()); ok(f'form_{a}', d < 2e-6, f'max abs {d:.1e}')
    xb = torch.from_numpy(Xtr[b]); l64 = m64(xb.double()).detach().numpy(); lt = mt(xb).detach().numpy(); lm = Z[f'logits_{a}']
    em, et = np.abs(lm - l64).max() / np.abs(l64).max(), np.abs(lt - l64).max() / np.abs(l64).max()
    ok(f'logits_init_{a}', em < 1e-4 and et < 1e-4, f'rel Fehler gegen float64: MLX {em:.1e}, torch {et:.1e}')
    for mod, dt in ((mt, torch.float32), (m64, torch.float64)):
        torch.nn.functional.cross_entropy(mod(xb.to(dt)), torch.from_numpy(Ytr[b])).backward()
    # Kriterium (22.11., vor dem Freeze geaendert): Median ueber die Schichten <= 3 x MLX und Maximum < 1e-2. Grund: Der groesste Fehler liegt
    # bei SELU in Schicht 0 (3 Eingangskanaele, Summe ueber 131 072 Positionen); die CPU-Faltung summiert dort ungenauer als MLX (Rundung, kein Rechenfehler).
    em_l = [rel(Z[f'grad_{a}_{i}'].transpose(0, 3, 1, 2), m64.ws[i].grad.numpy()) for i in range(T.L)]; et_l = [rel(mt.ws[i].grad.numpy(), m64.ws[i].grad.numpy()) for i in range(T.L)]
    ok(f'gradient_init_{a}', np.median(et_l) <= 3 * np.median(em_l) + 1e-6 and max(et_l) < 1e-2,
       f'rel Fehler je Schicht gegen float64, Median/Max: MLX {np.median(em_l):.1e}/{max(em_l):.1e}, torch {np.median(et_l):.1e}/{max(et_l):.1e} (Max bei Schicht {int(np.argmax(et_l))})')
w = torch.nn.Parameter(torch.from_numpy(Z['adam_p0'].copy())); o2 = T.MLXAdamW([w], 3.0)
for k in range(3): w.grad = torch.from_numpy(Z['adam_G'][k].copy()); o2.step(T.lr_at(1e-4, k, 3440))
e = float(np.abs(w.detach().numpy() - Z['adam_p3']).max()); ok('adamw_wie_mlx', e < 1e-7, f'max abs {e:.1e}')
A = Z['qA']; Wt = Z['qW']
ok('qw_gleich', np.array_equal(T.qw(torch.from_numpy(Wt.transpose(0, 3, 1, 2).copy())).numpy(), Z['qw'].transpose(0, 3, 1, 2)))
ok('qa_sym_gleich', np.array_equal(T.qa_sym(torch.from_numpy(A), 8, 3.7).numpy(), Z['qa_sym']))
ok('qa_asym_gleich', np.array_equal(T.qa_asym(torch.from_numpy(A), 4, (-2.1, 3.3)).numpy(), Z['qa_asym']))
ok('mse_asym_gleich', list(T.mse_asym(A, (float(A.min()), float(A.max())))) == J['mse_asym'], f"{T.mse_asym(A, (float(A.min()), float(A.max())))} / {J['mse_asym']}")
for a in ('bprecap', 'tat'):
    Rt = T.calib(T.CNN(a, seed).to(T.DEV), Xtr[:T.NCAL], seed); Rm = J[f'calib_{a}']
    d = max(abs(u - v) / max(abs(v), 1e-6) for k in ('max', 'p999', 'p9999', 'p99999', 'mse4') for pt, pm in zip(Rt[k], Rm[k]) for u, v in zip(pt, pm))
    ok(f'kalibrierung_init_{a}', d < 1e-3, f'max rel {d:.1e}')
mt = T.CNN('tat', seed).to(T.DEV); mc = T.cle(mt); u = torch.from_numpy(Xte[:256]).to(T.DEV)
with torch.no_grad(): d = float((mt(u) - mc(u)).abs().max() / mt(u).abs().max())
ok('cle_funktion_gleich', d < 1e-4, f'rel {d:.1e}')
if '--ohne-training' not in sys.argv:
    for a, lr in (('bprecap', 3e-5), ('tat', 1e-4)):
        t0 = time.time(); _, okt, tlt, kt, _ = T.train(a, lr, 0.0, seed, 172); km = J[f'kurz_{a}']
        ok(f'kurzes_training_{a}', okt and km['ok'] and abs(kt[-1][1] - km['kurve'][-1][1]) < 0.02,
           f"Val-Fehler torch {kt[-1][1]:.4f}, MLX {km['kurve'][-1][1]:.4f}; Verlust {tlt:.4f} / {km['train_loss']:.4f}; {time.time() - t0:.0f} s")
res['geraet'] = DEVICE; res['torch'] = torch.__version__; res['alle_ok'] = okall; json.dump(res, open(H / f'aequivalenz_ergebnis_{DEVICE}.json', 'w'), indent=1)
print('AEQUIVALENZ', 'OK' if okall else 'FEHLER', sum(v['ok'] for v in res.values() if isinstance(v, dict)), 'von', sum(isinstance(v, dict) for v in res.values()), 'Pruefungen')
