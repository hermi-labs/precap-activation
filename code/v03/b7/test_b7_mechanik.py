# Mechanik-Test B7 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen, Init und Daten gleich B3, LR aus B3, Laufzahlen, Seeds neu, Spalten der Zwischenmessung,
# und vor allem: Die Zwischenmessung veraendert das Training nicht (Zwischenwert bitgleich zu einem kuerzeren B3-Lauf,
# Endwert bitgleich zu einem gleich langen B3-Lauf ohne Zwischenmessung). Schritte als Vielfache von 43 (Verlust-Protokoll wie B3).
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b7_run.py', 'B7'); B3 = load(H.parent / 'b3' / 'b3_run.py', 'B3')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Formen, Init, Daten gleich B3
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B', 'LB', 'tat'):
    ok(f'{n}_form_wie_B3', np.array_equal(np.array(B.make_act(n, 100)(z)), np.array(B3.make_act(n, 100)(z))))
    a, b = B.Net(100, n, 42132, din=3072), B3.Net(100, n, 42132, din=3072)
    ok(f'{n}_init_wie_B3', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
    ok(f'{n}_daten_wie_B3', all(np.array_equal(p, q) for p, q in zip(B.data_for('cifar10', n), B3.data_for('cifar10', n))))

# 2. Konstanten, LR, Laufzahlen, Seeds, Spalten
ok('budget_4fach', B.STEPS == 4 * B3.STEPS and B.CHECKS == (1720, 3440) and B.DEPTHS == [100] and B.ACTS == ['kL2B', 'LB', 'tat'])
ok('konstanten_wie_B3', (B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CALS3) == (B3.WIDTH, B3.BATCH, B3.NVAL, B3.NCAL, B3.CALS4, B3.CALS3))
sel = B.lrsel(); ok('lr_aus_B3', (sel[('cifar10', 100, 'kL2B')], sel[('cifar10', 100, 'LB')], sel[('cifar10', 100, 'tat')]) == (2e-5, 2e-5, 5e-5))
C = list(B.configs()); ok('n_confirm', len(C) == 38, len(C))
ok('lr_in_configs', all(c['lr'] == sel[('cifar10', 100, c['act'])] for c in C))
used = set(range(42000, 42004)) | set(range(42010, 42132))
ok('seeds_neu', set(B.SEEDS) == set(range(42132, 42144)) and not (set(B.SEEDS) & used))
ok('kontrolle_B3_seed', [(c['act'], c['seed']) for c in C[:2]] == [('kL2B', 42084), ('LB', 42084)])
ok('spalten', all(f'c{k}_{x}' in B.COLS for k in (1720, 3440) for x in B.CKCOLS))

# 3. Zwischenmessung veraendert das Training nicht (CIFAR-10 d100)
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
for act in ('kL2B', 'LB', 'tat'):
    c = dict(dataset='cifar10', depth=100, act=act, lr=sel[('cifar10', 100, act)], seed=42132)
    B.STEPS = 172; B.CHECKS = (86,); B.OUTDIR = Path(tempfile.mkdtemp()); o7 = B.execute(dict(c), 0)
    B3.STEPS = 86; B3.OUTDIR = Path(tempfile.mkdtemp()); o3a = B3.execute(dict(c), 0)
    B3.STEPS = 172; B3.OUTDIR = Path(tempfile.mkdtemp()); o3b = B3.execute(dict(c), 0)
    ok(f'{act}_zwischen_bitgleich_B3', o7['status'] == 'ok' and all(o7[f'c86_{k}'] == o3a[k] for k in keys), [round(o7['c86_test_err_fp32'], 4), round(o3a['test_err_fp32'], 4)])
    ok(f'{act}_ende_bitgleich_B3', all(o7[k] == o3b[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8']), [round(o7['test_err_fp32'], 4), round(o3b['test_err_fp32'], 4)])
    ok(f'{act}_zwischen_ungleich_ende', o7['c86_test_err_fp32'] != o7['test_err_fp32'] and o7['sec_checks'] > 0)

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
