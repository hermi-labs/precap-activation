# Mechanik-Test V1 (vor dem Freeze, mit Claude). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen, Init und Daten gleich B10; Raster, Laufzahlen, Seeds wie B10, eindeutige run_ids; der Anlauf-Plan (wcos) ist linear bis Schritt 344,
# dann Kosinus auf 0, und wirkt unter mx.compile; der Pfad "cos" ist bitgleich zu B10 (kurzer Lauf, alle Kennzahlen und Zwischenmessung).
import importlib.util, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.optimizers as optim
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sys.modules[name] = M; sp.loader.exec_module(M); return M
V = load(H / 'v1_run.py', 'V1'); B = load(H.parent / 'b10' / 'b10_run.py', 'B10')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'LB_tr', 'tat'):
    ok(f'{n}_form_wie_B10', np.array_equal(np.array(V.make_act(n, 100)(z)), np.array(B.make_act(n, 100)(z))))
    a, b = V.Net(100, n, 42168, din=3072), B.Net(100, n, 42168, din=3072)
    ok(f'{n}_init_wie_B10', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
    ok(f'{n}_daten_wie_B10', all(np.array_equal(p, q) for p, q in zip(V.data_for('cifar10', n), B.data_for('cifar10', n))))
ok('konstanten_wie_B10', (V.STEPS, V.CHECKS, V.WIDTH, V.BATCH, V.NVAL, V.NCAL, V.CALS4, V.CACHE_LIMIT, V.ACTS) == (B.STEPS, B.CHECKS, B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CACHE_LIMIT, B.ACTS))
ok('raster', V.LR_GRID == [1e-4, 2e-4, 5e-4] and V.WD_GRID == [0.0, 0.3, 3.0] and V.TUNE_SEEDS == [42000, 42001] and V.WARM == 344)
T = list(V.tune_configs()); ok('n_tune', len(T) == 54 and all(c['sched'] == 'wcos' for c in T), len(T))
C = list(V.configs()); ok('n_confirm', len(C) == 38, len(C))
ok('kontrolle', [(c['act'], c['seed'], c['lr'], c['wd'], c['sched']) for c in C[:2]] == [('kL2B_tr', 42168, 1e-4, 0.3, 'cos'), ('tat', 42168, 2e-4, 0.0, 'cos')])
ok('seeds_wie_B10', V.SEEDS == B.SEEDS == list(range(42168, 42180)))
V.STAGE = 'confirm'; ids = [V.rid(c) for c in C]; V.STAGE = 'tune'; tids = [V.rid(c) for c in T]
ok('run_ids_eindeutig', len(set(ids)) == len(ids) and len(set(tids)) == len(tids) and ids[0].endswith('_kontrolle'))
# Plan: linear bis 344, dann Kosinus
sch = optim.join_schedules([optim.linear_schedule(0.0, 2e-4, 344), optim.cosine_decay(2e-4, 6880 - 344)], [344])
v = [float(sch(mx.array(k))) for k in (0, 172, 344, 3612, 6879)]
ok('plan', abs(v[0]) < 1e-12 and abs(v[1] - 1e-4) < 1e-9 and abs(v[2] - 2e-4) < 1e-9 and abs(v[3] - 1e-4) < 2e-7 and v[4] < 1e-9, [f'{x:.3g}' for x in v])
# Kurzlauf: cos-Pfad bitgleich zu B10, wcos-Pfad anders und unter compile wirksam
def short(M, c, steps=60):
    M.STEPS = steps; M.CHECKS = (30,); M.COLS = M.COLS_BASE + [f'c{k}_{x}' for k in M.CHECKS for x in M.CKCOLS] + ['sec_checks']
    M.OUTDIR = Path(tempfile.mkdtemp()); M.STAGE = 'confirm'; return M.execute(c, 0)
c = dict(dataset='cifar10', depth=100, act='kL2B_tr', lr=1e-4, wd=0.3, sched='cos', seed=42168)
o1, o2 = short(V, dict(c)), short(B, dict(c))
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4a_best', 'c30_val_err_fp32', 'c30_test_err_w8a4a_best', 'lr_end']
ok('cos_bitgleich_B10', all(o1[k] == o2[k] for k in keys), {k: (o1[k], o2[k]) for k in keys[:2]})
o3 = short(V, dict(c, sched='wcos'))
ok('wcos_wirkt', o3['train_loss_last50'] != o1['train_loss_last50'] and abs(o3['lr_end'] - float(optim.join_schedules([optim.linear_schedule(0.0, 1e-4, 30), optim.cosine_decay(1e-4, 30)], [30])(mx.array(59)))) < 1e-9, (o3['lr_end'],))
print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER'); sys.exit(0 if all(R.values()) else 1)
