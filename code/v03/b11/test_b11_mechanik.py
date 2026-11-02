# Mechanik-Test B11 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen gleich B10, Init gleich B10 (verdeckte Schichten; Ausgabe 100 statt 10), CIFAR-100-Daten (Form, Klassen, Val-Teil, Normierung),
# Raster, Laufzahlen, Seeds neu, Kontrolle mit den B10-Hyperparametern, der CIFAR-10-Pfad ist bitgleich zu B10 (auch an der Zwischenmessung),
# ein kurzer CIFAR-100-Lauf ist ok, und die Zwischenmessung veraendert das Training auf CIFAR-100 nicht.
import csv, importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b11_run.py', 'B11'); B10 = load(H.parent / 'b10' / 'b10_run.py', 'B10')
mx.set_cache_limit(B.CACHE_LIMIT)   # wie die Worker (B8): nur Speicher, laeuft neben B10
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Formen und Init gleich B10
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'LB_tr', 'tat'):
    ok(f'{n}_form_wie_B10', np.array_equal(np.array(B.make_act(n, 100)(z)), np.array(B10.make_act(n, 100)(z))))
    a, b = B.Net(100, n, 42180, din=3072, nout=10), B10.Net(100, n, 42180, din=3072)
    ok(f'{n}_init10_wie_B10', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
    c = B.Net(100, n, 42180, din=3072, nout=100)
    ok(f'{n}_init100_verdeckt_wie_B10', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(c.hid, b.hid)) and c.out.weight.shape == (100, 256) and c.out.bias.shape == (100,))
    ok(f'{n}_cifar10_daten_wie_B10', all(np.array_equal(p, q) for p, q in zip(B.data_for('cifar10', n), B10.data_for('cifar10', n))))

# 2. CIFAR-100-Daten
xtr, ytr, xv, yv, xte, yte = B.load('cifar100')
ok('c100_formen', (xtr.shape, xv.shape, xte.shape) == ((45000, 3072), (5000, 3072), (10000, 3072)), (xtr.shape, xv.shape, xte.shape))
ok('c100_klassen', set(np.unique(ytr)) == set(range(100)) and set(np.unique(yte)) == set(range(100)) and int(np.bincount(yte).min()) == 100 == int(np.bincount(yte).max()))
ok('c100_train_val_klassen', int(np.bincount(np.concatenate([ytr, yv])).min()) == 500 == int(np.bincount(np.concatenate([ytr, yv])).max()))
ok('c100_normiert', abs(float(xtr.mean())) < 1e-3 and abs(float(xtr.std()) - 1) < 1e-3, (float(xtr.mean()), float(xtr.std())))
ok('c100_pex', np.allclose(np.mean(B.data_for('cifar100', 'tat')[0][:100] ** 2, axis=1), 1, atol=1e-4))
ok('c100_anders_als_c10', not np.array_equal(xte[:10], B.load('cifar10')[4][:10]))

# 3. Konstanten, Raster, Laufzahlen, Seeds, Kontrolle
ok('budget_wie_B10', B.STEPS == B10.STEPS == 6880 and B.CHECKS == B10.CHECKS and B.DEPTHS == [100] and B.ACTS == B10.ACTS and B.DATASETS == ['cifar100'])
ok('konstanten_wie_B10', (B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CALS3, B.CACHE_LIMIT, B.TUNE_SEEDS) == (B10.WIDTH, B10.BATCH, B10.NVAL, B10.NCAL, B10.CALS4, B10.CALS3, B10.CACHE_LIMIT, B10.TUNE_SEEDS))
T = list(B.tune_configs()); ok('raster', B.LR_GRID == [5e-5, 1e-4, 2e-4, 5e-4] and B.WD_GRID == [0.0, 0.3, 3.0])
ok('n_tune', len(T) == 72 == 3 * len(B.LR_GRID) * len(B.WD_GRID) * 2 and all(c['dataset'] == 'cifar100' and c['sched'] == 'cos' for c in T), len(T))
C = list(B.configs()); ok('n_confirm', len(C) == 38, len(C))
hp = {r['act']: (float(r['lr_sel']), float(r['wd_sel'])) for r in csv.DictReader(open(H.parent / 'b10' / 'hp_tuned.csv'))}
ok('kontrolle_hp_aus_B10', [(c['dataset'], c['act'], c['seed'], (c['lr'], c['wd']), c['sched']) for c in C[:2]] == [('cifar10', 'kL2B_tr', 42168, hp['kL2B_tr'], 'cos'), ('cifar10', 'tat', 42168, hp['tat'], 'cos')], C[:2])
used = set(range(42000, 42004)) | set(range(42010, 42180))
ok('seeds_neu', set(B.SEEDS) == set(range(42180, 42192)) and not (set(B.SEEDS) & used))

# 4. CIFAR-10-Pfad bitgleich zu B10 (Kosinus, WD, Zwischenmessung)
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
for act in ('kL2B_tr', 'tat'):
    c = dict(dataset='cifar10', depth=100, act=act, lr=hp[act][0], wd=hp[act][1], sched='cos', seed=42168)
    B.STEPS = B10.STEPS = 172; B.CHECKS = B10.CHECKS = (86,)
    B.OUTDIR = Path(tempfile.mkdtemp()); o = B.execute(dict(c), 0); B10.OUTDIR = Path(tempfile.mkdtemp()); o10 = B10.execute(dict(c), 0)
    ok(f'{act}_cifar10_bitgleich_B10', o['status'] == 'ok' and all(o[k] == o10[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8', 'lr_end']) and all(o[f'c86_{k}'] == o10[f'c86_{k}'] for k in keys),
       [round(o['test_err_fp32'], 4), round(o10['test_err_fp32'], 4)])

# 5. CIFAR-100 kurz: Lauf ok, Fehler unter Zufall, Zwischenmessung veraendert das Training nicht
for act in ('kL2B_tr', 'LB_tr', 'tat'):
    c = dict(dataset='cifar100', depth=100, act=act, lr=1e-4, wd=0.3, sched='cos', seed=42180)
    B.STEPS = 172; B.CHECKS = (86,); B.OUTDIR = Path(tempfile.mkdtemp()); oc = B.execute(dict(c), 0)
    B.CHECKS = (); B.OUTDIR = Path(tempfile.mkdtemp()); on = B.execute(dict(c), 0)
    ok(f'{act}_c100_lauf_ok', oc['status'] == 'ok' and 0 < oc['test_err_fp32'] < 0.97, round(oc['test_err_fp32'], 4))
    ok(f'{act}_c100_zwischen_aendert_training_nicht', all(oc[k] == on[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8']) and oc['sec_checks'] > 0)

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
