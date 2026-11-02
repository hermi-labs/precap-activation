# Mechanik-Test B10 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen, Init und Daten gleich B9, Konstanten, Raster, Laufzahlen, Seeds neu, Spalten,
# der neue Code-Pfad AdamW mit konstanter LR und WD 0 ist bitgleich zu B9 (Adam), auch an der Zwischenmessung,
# der Kosinus-Abfall wirkt unter mx.compile (LR am Ende wie berechnet, Ergebnis anders als konstant), WD wirkt,
# und die Zwischenmessung veraendert das Training mit Kosinus und WD nicht. Schritte als Vielfache von 43 (Verlust-Protokoll wie B9).
import importlib.util, math, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b10_run.py', 'B10'); B9 = load(H.parent / 'b9' / 'b9_run.py', 'B9')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Formen, Init, Daten gleich B9
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'LB_tr', 'tat'):
    ok(f'{n}_form_wie_B9', np.array_equal(np.array(B.make_act(n, 100)(z)), np.array(B9.make_act(n, 100)(z))))
    a, b = B.Net(100, n, 42168, din=3072), B9.Net(100, n, 42168, din=3072)
    ok(f'{n}_init_wie_B9', all(np.array_equal(np.array(x.weight), np.array(y.weight)) for x, y in zip(a.hid + [a.out], b.hid + [b.out])))
    ok(f'{n}_daten_wie_B9', all(np.array_equal(p, q) for p, q in zip(B.data_for('cifar10', n), B9.data_for('cifar10', n))))

# 2. Konstanten, Raster, Laufzahlen, Seeds, Spalten
ok('budget_wie_B9', B.STEPS == B9.STEPS == 6880 and B.CHECKS == B9.CHECKS and B.DEPTHS == [100] and B.ACTS == ['kL2B_tr', 'LB_tr', 'tat'])
ok('konstanten_wie_B9', (B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CALS3, B.CACHE_LIMIT) == (B9.WIDTH, B9.BATCH, B9.NVAL, B9.NCAL, B9.CALS4, B9.CALS3, B9.CACHE_LIMIT))
ok('raster', B.LR_GRID == [5e-5, 1e-4, 2e-4] and B.WD_GRID == [0.0, 0.3, 3.0] and B.TUNE_SEEDS == [42000, 42001] and B.tune_acts(100) == ['kL2B_tr', 'LB_tr', 'tat'])
T = list(B.tune_configs()); ok('n_tune', len(T) == 54 and all(c['sched'] == 'cos' for c in T), len(T))
C = list(B.configs()); ok('n_confirm', len(C) == 38, len(C))
ok('kontrolle_B9_seed', [(c['act'], c['seed'], c['lr'], c['wd'], c['sched']) for c in C[:2]] == [('kL2B_tr', 42156, 5e-5, 0.0, 'const'), ('tat', 42156, 5e-5, 0.0, 'const')])
ok('bestaetigung_kosinus', all(c['sched'] == 'cos' for c in C[2:]))
used = set(range(42000, 42004)) | set(range(42010, 42168))
ok('seeds_neu', set(B.SEEDS) == set(range(42168, 42180)) and not (set(B.SEEDS) & used))
ok('spalten', all(f'c{k}_{x}' in B.COLS for k in B.CHECKS for x in B.CKCOLS) and all(x in B.COLS for x in ('wd', 'sched', 'lr_end')))
# Wahlregel: kuenstliches Tuning (hoechste mediane Val-Genauigkeit, Gleichstand -> kleinere LR, dann kleinerer WD)
d = Path(tempfile.mkdtemp()); (d / 'workers').mkdir()
import csv
with open(d / 'workers' / 'w0.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['dataset', 'depth', 'act', 'lr', 'wd', 'status', 'val_err_fp32']); w.writeheader()
    for act, best in (('kL2B_tr', (1e-4, 0.3)), ('LB_tr', (2e-4, 3.0)), ('tat', None)):
        for lr in B.LR_GRID:
            for wd in B.WD_GRID:
                e = 0.40 if (lr, wd) == best else 0.45
                if act == 'tat': e = 0.42 if wd in (0.3, 3.0) and lr == 1e-4 else 0.45   # Gleichstand -> kleinerer WD
                for s in B.TUNE_SEEDS: w.writerow(dict(dataset='cifar10', depth=100, act=act, lr=lr, wd=wd, status='ok' if not (act == 'tat' and lr == 2e-4) else 'nonfinite_loss', val_err_fp32=e))
B.TUNEDIR = d; S = {r['act']: (r['lr_sel'], r['wd_sel'], r['lr_at_edge'], r['wd_at_edge']) for r in B.select_lr()}; B.TUNEDIR = H / 'tune'
ok('wahlregel', S == {'kL2B_tr': (1e-4, 0.3, False, False), 'LB_tr': (2e-4, 3.0, True, True), 'tat': (1e-4, 0.3, False, False)}, S)

# 3. Code-Pfad AdamW (konstant, WD 0) bitgleich zu B9, auch an der Zwischenmessung
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
for act in ('kL2B_tr', 'tat'):
    B.STEPS = 172; B.CHECKS = (86,); B.OUTDIR = Path(tempfile.mkdtemp()); o = B.execute(dict(dataset='cifar10', depth=100, act=act, lr=5e-5, wd=0.0, sched='const', seed=42156), 0)
    B9.STEPS = 172; B9.CHECKS = (86,); B9.OUTDIR = Path(tempfile.mkdtemp()); o9 = B9.execute(dict(dataset='cifar10', depth=100, act=act, lr=5e-5, seed=42156), 0)
    ok(f'{act}_adamw_wd0_bitgleich_B9', o['status'] == 'ok' and all(o[k] == o9[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8']) and all(o[f'c86_{k}'] == o9[f'c86_{k}'] for k in keys),
       [round(o['test_err_fp32'], 4), round(o9['test_err_fp32'], 4)])
    ok(f'{act}_konstante_lr_ende', o['lr_end'] == float(np.float32(5e-5)), o['lr_end'])   # LR liegt als float32 vor (5e-5 -> 4.99999987e-05)

# 4. Kosinus wirkt (unter mx.compile), WD wirkt, Zwischenmessung veraendert das Training nicht
for act in ('kL2B_tr', 'LB_tr', 'tat'):
    c = dict(dataset='cifar10', depth=100, act=act, lr=1e-4, wd=0.3, sched='cos', seed=42168)
    B.STEPS = 172; B.CHECKS = (86,); B.OUTDIR = Path(tempfile.mkdtemp()); oc = B.execute(dict(c), 0)
    B.CHECKS = (); B.OUTDIR = Path(tempfile.mkdtemp()); on = B.execute(dict(c), 0)
    B.OUTDIR = Path(tempfile.mkdtemp()); ok_ = B.execute(dict(c, sched='const'), 0)
    B.OUTDIR = Path(tempfile.mkdtemp()); o0 = B.execute(dict(c, wd=0.0), 0)
    want = 0.5 * (1 + math.cos(math.pi * 171 / 172)) * 1e-4   # MLX: Update k nutzt den Plan bei Schritt k-1, das letzte also bei 171
    ok(f'{act}_kosinus_lr_ende', oc['status'] == 'ok' and abs(oc['lr_end'] - want) < 1e-9, [oc['lr_end'], want])
    ok(f'{act}_kosinus_aendert', oc['test_err_fp32'] != ok_['test_err_fp32'] or oc['train_loss_last50'] != ok_['train_loss_last50'])
    ok(f'{act}_wd_aendert', on['train_loss_last50'] != o0['train_loss_last50'])
    ok(f'{act}_zwischen_aendert_training_nicht', all(oc[k] == on[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8']) and oc['sec_checks'] > 0, [round(oc['test_err_fp32'], 4), round(on['test_err_fp32'], 4)])

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
