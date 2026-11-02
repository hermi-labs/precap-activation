# Mechanik-Test B6 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen der _tr-Varianten gleich der Grundform, Referenzen bitgleich zu B3, TAT-Rezept fuer _tr (orthogonale Init mal Gain,
# Normierung je Beispiel), Signal-Varianz bei der Init, LR-Quellen, Laufzahlen, Seeds, Kontrollpfad bitgleich zu B1.
import importlib.util, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b6_run.py', 'B6'); B3 = load(H.parent / 'b3' / 'b3_run.py', 'B3'); B1 = load(H.parent / 'b1' / 'b1_run.py', 'B1')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Formen
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B', 'LB'):
    ok(f'{n}_tr_gleiche_form', np.array_equal(np.array(B.make_act(n + '_tr', 100)(z)), np.array(B3.make_act(n, 100)(z))) and B.base(n + '_tr') == n and B.lag(n + '_tr') is None)
for n in ('kL2B', 'LB', 'tat'):
    ok(f'{n}_wie_B3', np.array_equal(np.array(B.make_act(n, 100)(z)), np.array(B3.make_act(n, 100)(z))) and not B.recipe(n))

# 2. Init: _tr orthogonal mal Gain (wie TAT), Referenzen gleich B3
m = B.Net(100, 'kL2B_tr', 42120, din=3072); g = B.GAINS[('kL2B', 100)]
W1 = np.array(m.hid[1].weight); W0 = np.array(m.hid[0].weight); Wo = np.array(m.out.weight)
ok('tr_init_orthogonal_mal_gain', np.allclose(W1 @ W1.T, g * g * np.eye(256), atol=1e-4) and np.allclose(W0 @ W0.T, g * g * np.eye(256), atol=1e-4), round(g, 6))
ok('tr_out_wie_tat', np.allclose(Wo @ Wo.T, np.eye(10), atol=1e-4))
for n in ('kL2B', 'LB', 'tat'):
    a, b = B.Net(100, n, 42120, din=3072), B3.Net(100, n, 42120, din=3072)
    ok(f'{n}_init_wie_B3', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))

# 3. Daten: _tr bekommt die Normierung je Beispiel, Referenzen nicht
d0 = B.data_for('cifar10', 'kL2B'); dt = B.data_for('cifar10', 'kL2B_tr'); dT = B.data_for('cifar10', 'tat')
ok('tr_daten_wie_tat', np.array_equal(dt[0], dT[0]) and np.array_equal(dt[4], dT[4]) and abs(float(np.mean(dt[0][:200] ** 2, axis=1).mean()) - 1) < 1e-4)
ok('ref_daten_wie_B3', np.array_equal(d0[0], B3.data_for('cifar10', 'kL2B')[0]))

# 4. Signal bei der Init: Varianz der Vor-Aktivierung bleibt ueber 100 Schichten nahe 1 (wie Gauss-Init)
def prevar(net, x):
    h = mx.array(x); v = []
    for l in net.hid: y = l(h); v.append(float(mx.mean(y * y))); h = net.post(0, y)
    return v
for n in ('kL2B', 'LB'):
    vt = prevar(B.Net(100, n + '_tr', 42120, din=3072), dt[0][:256]); vg = prevar(B.Net(100, n, 42120, din=3072), d0[0][:256])
    ok(f'{n}_tr_varianz_stabil', 0.7 < vt[-1] < 1.4 and 0.7 < vg[-1] < 1.4, (round(vt[0], 3), round(vt[-1], 3), round(vg[-1], 3)))

# 5. Raster, Seeds, Laufzahlen, LR-Quellen
ok('raster_wie_B3', B.LR_GRID == B3.LR_GRID and B.TUNE_SEEDS == B3.TUNE_SEEDS and B.DEPTHS == [100])
ok('n_tune', len(list(B.tune_configs())) == 48, len(list(B.tune_configs())))
ok('n_confirm', len(B.REPRO) + len(B.DEPTHS) * len(B.DATASETS) * len(B.SEEDS) * len(B.ACTS) == 61)
used = set(range(42000, 42004)) | set(range(42010, 42120))
ok('seeds_neu', set(B.SEEDS) == set(range(42120, 42132)) and not (set(B.SEEDS) & used))
B.TUNEDIR = Path(tempfile.mkdtemp())
sel = B.lrsel(); ok('lr_ref_aus_B3', (sel[('cifar10', 100, 'kL2B')], sel[('cifar10', 100, 'LB')], sel[('cifar10', 100, 'tat')]) == (2e-5, 2e-5, 5e-5))

# 6. execute: Referenzen bitgleich zu B3 (CIFAR-10 d100, 60 Schritte), _tr laeuft, Kontrollpfad bitgleich zu B1
for M in (B, B3, B1): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'act_range_max_median']
for act in ('kL2B', 'LB', 'tat'):
    c = dict(dataset='cifar10', depth=100, act=act, lr=2e-5, seed=42120); o1 = B.execute(dict(c), 0); o2 = B3.execute(dict(c), 0)
    ok(f'{act}_bitgleich_B3', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys))
for act in ('kL2B_tr', 'LB_tr'):
    o = B.execute(dict(dataset='cifar10', depth=100, act=act, lr=2e-5, seed=42120), 0)
    ok(f'{act}_laeuft', o['status'] == 'ok' and o['test_err_fp32'] < 0.9 and math.isfinite(o['test_err_w8a3_best']) and o.get('sat_rate') is not None,
       (round(o['test_err_fp32'], 4), round(o['test_err_w8a4_best'], 4), o.get('sat_rate')))
c = dict(dataset='synth', depth=100, act='kL2B', lr=2e-5, seed=42060); o1 = B.execute(dict(c), 0); o2 = B1.execute(dict(c), 0)
ok('kontrolle_bitgleich_B1', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in ('train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best')))

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
