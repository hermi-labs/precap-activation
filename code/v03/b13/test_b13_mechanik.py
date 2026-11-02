# Mechanik-Test (fuer B12 und B13 gleich, Studie aus dem Ordnernamen). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: bestehende Formen, Init, Daten und Konstanten gleich B10; neue Formen (relu, gelu) gegen float64-Formel und Gains (E[f^2] = 1 nach Gain);
# Laufliste (Tuning, Bestaetigung, Kontrolle, Seeds wie B10); Kontrolle kL2B_tr bitgleich zu B10 (kurzer Lauf, auch an der Zwischenmessung);
# jede eigene Form laeuft kurz ok und die Zwischenmessung veraendert ihr Training nicht.
import importlib.util, math, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent; NAME = H.name
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / f'{NAME}_run.py', NAME); B10 = load(H.parent / 'b10' / 'b10_run.py', 'B10')
mx.set_cache_limit(B.CACHE_LIMIT)   # laeuft neben anderen Workern
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
OWN = list(B.ACTS)
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'LB_tr', 'kL2Bh_tr', 'LBh_tr', 'selu_tr'):
    ok(f'{n}_form_wie_B10', np.array_equal(np.array(B.make_act(n, 100)(z)), np.array(B10.make_act(n, 100)(z))))
for n in OWN:
    a = B.Net(100, n, 42168, din=3072)
    if n in ('kL2Bh_tr', 'LBh_tr', 'selu_tr'):
        b = B10.Net(100, n, 42168, din=3072); ok(f'{n}_init_wie_B10', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
    W = np.array(a.hid[5].weight); g = B.GAINS[(B.base(n), 100)]
    ok(f'{n}_init_orthogonal_mal_gain', np.allclose(W @ W.T, g * g * np.eye(256), atol=1e-4), round(g, 5))
    ok(f'{n}_daten_wie_B10', all(np.array_equal(p, q) for p, q in zip(B.data_for('cifar10', n), B10.data_for('cifar10', 'kL2B_tr'))))
if 'relu_tr' in OWN:
    X = np.linspace(-8, 8, 20001)
    ok('relu_formel', np.array_equal(np.array(B.make_act('relu_tr', 100)(mx.array(X.astype(np.float32)))), np.maximum(X, 0).astype(np.float32)))
    erf = np.vectorize(math.erf); gl = 0.5 * X * (1 + erf(X / math.sqrt(2)))
    ok('gelu_formel_exakt', float(np.abs(np.array(B.make_act('gelu_tr', 100)(mx.array(X.astype(np.float32)))) - gl).max()) < 2e-6)
    Xh, Wh = np.polynomial.hermite_e.hermegauss(150); Wh = Wh / Wh.sum()
    for n in ('relu', 'gelu', 'selu'):
        y = B.np_act(n, Xh) * B.GAINS[(n, 100)]; ok(f'{n}_gain_E_f2_eins', abs(float(np.sum(Wh * y * y)) - 1) < 1e-9)
    ok('relu_gain_he', abs(B.GAINS[('relu', 100)] - math.sqrt(2)) < 1e-6)
ok('konstanten_wie_B10', (B.STEPS, B.CHECKS, B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CALS3, B.CACHE_LIMIT, B.TUNE_SEEDS, B.WD_GRID) ==
   (B10.STEPS, B10.CHECKS, B10.WIDTH, B10.BATCH, B10.NVAL, B10.NCAL, B10.CALS4, B10.CALS3, B10.CACHE_LIMIT, B10.TUNE_SEEDS, B10.WD_GRID))
ok('raster', B.LR_GRID == [5e-5, 1e-4, 2e-4, 5e-4] and B.tune_acts(100) == OWN)
T = list(B.tune_configs()); ok('n_tune', len(T) == len(OWN) * 12 * 2, len(T))
C = list(B.configs()); ok('n_bestaetigung', len(C) == 1 + 12 * len(OWN), len(C))
ok('kontrolle', C[0] == dict(dataset='cifar10', depth=100, act='kL2B_tr', lr=1e-4, wd=0.3, sched='cos', seed=42168), C[0])
ok('seeds_wie_B10', B.SEEDS == B10.SEEDS == list(range(42168, 42180)))
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
c = dict(C[0]); B.STEPS = B10.STEPS = 172; B.CHECKS = B10.CHECKS = (86,)
B.OUTDIR = Path(tempfile.mkdtemp()); o = B.execute(dict(c), 0); B10.OUTDIR = Path(tempfile.mkdtemp()); o10 = B10.execute(dict(c), 0)
ok('kontrolle_bitgleich_B10', o['status'] == 'ok' and all(o[k] == o10[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8', 'lr_end']) and all(o[f'c86_{k}'] == o10[f'c86_{k}'] for k in keys),
   [round(o['test_err_fp32'], 4), round(o10['test_err_fp32'], 4)])
for n in OWN:
    c = dict(dataset='cifar10', depth=100, act=n, lr=1e-4, wd=0.3, sched='cos', seed=42168)
    B.CHECKS = (86,); B.OUTDIR = Path(tempfile.mkdtemp()); oc = B.execute(dict(c), 0)
    B.CHECKS = (); B.OUTDIR = Path(tempfile.mkdtemp()); on = B.execute(dict(c), 0)
    ok(f'{n}_lauf_ok', oc['status'] == 'ok' and 0 < oc['test_err_fp32'] < 0.9, round(oc['test_err_fp32'], 4))
    ok(f'{n}_zwischen_aendert_training_nicht', all(oc[k] == on[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8']) and oc['sec_checks'] > 0)
print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
