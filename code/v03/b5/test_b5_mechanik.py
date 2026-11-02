# Mechanik-Test B5 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Formen, Gains und TAT unveraendert gegen B3, Raster/Seeds/Laufzahlen, LR-Quelle fuer TAT (B3), execute bitgleich zu B3
# (CIFAR-10 d200 kurz, alle vier Formen) und Kontrollpfad bitgleich zu B1 (synthetisch).
import importlib.util, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b5_run.py', 'B5'); B3 = load(H.parent / 'b3' / 'b3_run.py', 'B3'); B1 = load(H.parent / 'b1' / 'b1_run.py', 'B1')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Formen wie B3
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B', 'LB', 'kL2', 'tat'):
    ok(f'{n}_d200_wie_B3', np.array_equal(np.array(B.make_act(n, 200)(z)), np.array(B3.make_act(n, 200)(z))))
ok('gains_wie_B3', B.GAINS == B3.GAINS and B.TAT == B3.TAT)

# 2. Raster, Seeds, Laufzahlen, LR-Quellen
ok('raster', B.LR_GRID == [3e-6, 5e-6, 7e-6, 1e-5] and B.DEPTHS == [200] and B.TUNE_SEEDS == B3.TUNE_SEEDS)
ok('tune_formen', B.tune_acts(200) == ['kL2B', 'LB', 'kL2'])
ok('n_tune', len(list(B.tune_configs())) == 48, len(list(B.tune_configs())))
ok('n_confirm', len(B.REPRO) + len(B.DEPTHS) * len(B.DATASETS) * len(B.SEEDS) * len(B.ACTS) == 49)
used = set(range(42000, 42004)) | set(range(42010, 42108))   # Tuning, Lagma-3 bis B4
ok('seeds_neu', set(B.SEEDS) == set(range(42108, 42120)) and not (set(B.SEEDS) & used))
B.TUNEDIR = Path(tempfile.mkdtemp())   # ohne eigenes Tuning: LR fuer TAT d200 aus B3, MNIST-Kontrolle aus A
sel = B.lrsel(); ok('lr_tat_aus_B3', sel[('cifar10', 200, 'tat')] == 2e-5 and sel[('mnist', 100, 'kL2')] == 2e-5, (sel[('cifar10', 200, 'tat')], sel[('mnist', 100, 'kL2')]))
ok('tune_rid_1e-5_wie_B3', (lambda c: (setattr(B, 'STAGE', 'tune'), setattr(B3, 'STAGE', 'tune'), B.rid(c) == B3.rid(c))[-1])(dict(dataset='cifar10', depth=200, act='LB', lr=1e-5, seed=42000)))
B.STAGE = B3.STAGE = 'confirm'

# 3. execute bitgleich zu B3 (CIFAR-10 d200, 60 Schritte) und Kontrollpfad zu B1
for M in (B, B3, B1): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'act_range_max_median']
for act in ('kL2B', 'LB', 'kL2', 'tat'):
    c = dict(dataset='cifar10', depth=200, act=act, lr=5e-6, seed=42108)
    o1 = B.execute(dict(c), 0); o2 = B3.execute(dict(c), 0)
    ok(f'{act}_bitgleich_B3', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys) and math.isfinite(o1['test_err_w8a3_best']), (round(o1['test_err_fp32'], 4), round(o2['test_err_fp32'], 4)))
c = dict(dataset='synth', depth=100, act='kL2B', lr=2e-5, seed=42060); o1 = B.execute(dict(c), 0); o2 = B1.execute(dict(c), 0)
ok('kontrolle_bitgleich_B1', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in ('train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best')))

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
