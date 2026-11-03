# Mechanik-Test V1b (vor dem Freeze, mit Claude). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen, Init, Daten und Konstanten gleich V1; Raster und 32 eindeutige run_ids; 'wcos344' ist bitgleich zum V1-Pfad 'wcos'
# (Kurzlauf 800 Schritte, Zwischenmessung); 'wcos688' wirkt (anderer Verlauf), und der Plan ist linear bis 688, dann Kosinus.
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.optimizers as optim
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sys.modules[name] = M; sp.loader.exec_module(M); return M
V = load(H / 'v1b_run.py', 'V1b'); B = load(H.parent / 'v1' / 'v1_run.py', 'V1')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'tat'):
    ok(f'{n}_form_wie_V1', np.array_equal(np.array(V.make_act(n, 100)(z)), np.array(B.make_act(n, 100)(z))))
    a, b = V.Net(100, n, 42000, din=3072), B.Net(100, n, 42000, din=3072)
    ok(f'{n}_init_wie_V1', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
    ok(f'{n}_daten_wie_V1', all(np.array_equal(p, q) for p, q in zip(V.data_for('cifar10', n), B.data_for('cifar10', n))))
ok('konstanten_wie_V1', (V.STEPS, V.CHECKS, V.WIDTH, V.BATCH, V.NVAL, V.NCAL, V.CALS4, V.CACHE_LIMIT, V.TUNE_SEEDS, V.WARM) == (B.STEPS, B.CHECKS, B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CACHE_LIMIT, B.TUNE_SEEDS, B.WARM))
ok('raster', V.LR_GRID == [5e-5, 2e-4, 3e-4, 5e-4] and V.WARM_GRID == [344, 688] and V.WD_FIX == {'kL2B_tr': 0.3, 'tat': 0.0} and V.tune_acts(100) == ['kL2B_tr', 'tat'])
T = list(V.tune_configs()); V.STAGE = 'tune'; ids = [V.rid(c) for c in T]
ok('n_tune', len(T) == 32 and len(set(ids)) == 32 and {c['sched'] for c in T} == {'wcos344', 'wcos688'}, len(T))
ok('wd_je_form', all(c['wd'] == V.WD_FIX[c['act']] for c in T))
sch = optim.join_schedules([optim.linear_schedule(0.0, 3e-4, 688), optim.cosine_decay(3e-4, 6880 - 688)], [688])
v = [float(sch(mx.array(k))) for k in (0, 344, 688, 3784, 6879)]
ok('plan_688', abs(v[0]) < 1e-12 and abs(v[1] - 1.5e-4) < 1e-9 and abs(v[2] - 3e-4) < 1e-9 and abs(v[3] - 1.5e-4) < 3e-7 and v[4] < 1e-9, [f'{x:.3g}' for x in v])
def short(M, c, steps=800):
    M.STEPS = steps; M.CHECKS = (400,); M.COLS = M.COLS_BASE + [f'c{k}_{x}' for k in M.CHECKS for x in M.CKCOLS] + ['sec_checks']
    M.OUTDIR = Path(tempfile.mkdtemp()); M.STAGE = 'confirm'; return M.execute(c, 0)
c = dict(dataset='cifar10', depth=100, act='kL2B_tr', lr=2e-4, wd=0.3, seed=42000)
o1, o2 = short(V, dict(c, sched='wcos344')), short(B, dict(c, sched='wcos'))
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4a_best', 'c400_val_err_fp32', 'c400_test_err_w8a4a_best', 'lr_end']
ok('wcos344_bitgleich_V1', all(o1[k] == o2[k] for k in keys), {k: (o1[k], o2[k]) for k in keys[:2]})
o3 = short(V, dict(c, sched='wcos688'))
ok('wcos688_wirkt', o3['train_loss_last50'] != o1['train_loss_last50'] and o3['c400_train_loss_last50'] != o1['c400_train_loss_last50'], (o3['train_loss_last50'], o1['train_loss_last50']))
print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER'); sys.exit(0 if all(R.values()) else 1)
