# Mechanik-Test B10b (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Code gleich B10 bis auf Laufliste und Laufnamen (Formen, Init, Daten, Konstanten), Laufliste (25, Varianten, Seeds wie B10),
# Laufnamen eindeutig, Kontrolle mit der B10-Wahl, und ein kurzer Lauf jeder TAT-Variante ist bitgleich zum B10-Code mit derselben Einstellung.
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b10b_run.py', 'B10b'); B10 = load(H.parent / 'b10' / 'b10_run.py', 'B10')
mx.set_cache_limit(B.CACHE_LIMIT)   # laeuft neben B11
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
ok('tat_form_wie_B10', np.array_equal(np.array(B.make_act('tat', 100)(z)), np.array(B10.make_act('tat', 100)(z))))
a, b = B.Net(100, 'tat', 42168, din=3072), B10.Net(100, 'tat', 42168, din=3072)
ok('tat_init_wie_B10', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
ok('tat_daten_wie_B10', all(np.array_equal(p, q) for p, q in zip(B.data_for('cifar10', 'tat'), B10.data_for('cifar10', 'tat'))))
ok('konstanten_wie_B10', (B.STEPS, B.CHECKS, B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CALS3, B.CACHE_LIMIT) == (B10.STEPS, B10.CHECKS, B10.WIDTH, B10.BATCH, B10.NVAL, B10.NCAL, B10.CALS4, B10.CALS3, B10.CACHE_LIMIT))
C = list(B.configs())
ok('n_laeufe', len(C) == 25, len(C))
ok('kontrolle', C[0] == dict(dataset='cifar10', depth=100, act='tat', lr=2e-4, wd=0.0, sched='cos', seed=42168), C[0])
ok('varianten', sorted({(c['lr'], c['wd']) for c in C[1:]}) == [(5e-5, 3.0), (1e-4, 0.3)] and all(c['act'] == 'tat' and c['sched'] == 'cos' for c in C[1:]))
ok('seeds_wie_B10', sorted({c['seed'] for c in C[1:]}) == list(range(42168, 42180)) and B.SEEDS == B10.SEEDS)
ok('namen_eindeutig', len({B.rid(c) for c in C}) == 25)
import csv
hp = {r['act']: (float(r['lr_sel']), float(r['wd_sel'])) for r in csv.DictReader(open(H.parent / 'b10' / 'hp_tuned.csv'))}
ok('kontrolle_ist_B10_wahl', hp['tat'] == (2e-4, 0.0) and hp['kL2B_tr'] == (1e-4, 0.3), hp)

keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
for lr, wd in [(2e-4, 0.0), (1e-4, 0.3), (5e-5, 3.0)]:
    c = dict(dataset='cifar10', depth=100, act='tat', lr=lr, wd=wd, sched='cos', seed=42168)
    B.STEPS = B10.STEPS = 172; B.CHECKS = B10.CHECKS = (86,)
    B.OUTDIR = Path(tempfile.mkdtemp()); o = B.execute(dict(c), 0); B10.OUTDIR = Path(tempfile.mkdtemp()); o10 = B10.execute(dict(c), 0)
    ok(f'tat_lr{lr:g}_wd{wd:g}_bitgleich_B10', o['status'] == 'ok' and all(o[k] == o10[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8', 'lr_end']) and all(o[f'c86_{k}'] == o10[f'c86_{k}'] for k in keys),
       [round(o['test_err_fp32'], 4), round(o10['test_err_fp32'], 4)])

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
