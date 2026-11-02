# Mechanik-Test B3 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: kL2B/LB/kL2 unveraendert gegen B2/B1, TAT-Rezept gleich Q3 (Aktivierung, SUO-Init, Normierung je Beispiel, Konstanten wie A2),
# CIFAR-10-Lader (Formen, Klassen, Standardisierung, erster Datensatz roh), Netz mit 3072 Eingaengen, execute auf synthetischen und echten Daten,
# Kontrolllauf-Pfad bitgleich zu B1, Laufzahlen.
import importlib.util, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b3_run.py', 'B3'); B2 = load(H.parent / 'b2' / 'b2_run.py', 'B2'); B1 = load(H.parent / 'b1' / 'b1_run.py', 'B1'); Q3 = load(H.parent / 'q3' / 'q3_run.py', 'Q3')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Aktivierungen
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for d in (100, 200):
    for n in ('kL2B', 'LB', 'kL2'):
        ok(f'{n}_d{d}_wie_B2', np.array_equal(np.array(B.make_act(n, d)(z)), np.array(B2.make_act(n, d)(z))) and B.GAINS[(n, d)] == B2.GAINS[(n, d)])
ok('tat_konstanten_wie_A2', B.TAT[100] == dict(neg=0.5704395323991776, scale=1.2284042441106242) and B.TAT[200] == dict(neg=0.6795996427536011, scale=1.169668377971502))
Q3.TAT_DEPTH[0] = 100
ok('tat_d100_wie_Q3', np.array_equal(np.array(B.make_act('tat', 100)(z)), np.array(Q3.make_act('tat')(z))))
ok('tat_scale_formel', all(abs(B.TAT[d]['scale'] - math.sqrt(2 / (1 + B.TAT[d]['neg'] ** 2))) < 1e-9 for d in (100, 200)))
r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
ok('suo_wie_Q3', np.array_equal(B.suo(r1, 256, 3072), Q3.suo(r2, 256, 3072)) and np.array_equal(B.suo(r1, 256, 256), Q3.suo(r2, 256, 256)))
a = np.random.default_rng(1).normal(size=(50, 3072)).astype(np.float32); ok('pex_wie_Q3', np.array_equal(B.pex(a), Q3.pex(a)))

# 2. CIFAR-10-Lader
xtr, ytr, xv, yv, xte, yte = B.load('cifar10')
ok('cifar_formen', xtr.shape == (45000, 3072) and xv.shape == (5000, 3072) and xte.shape == (10000, 3072), (xtr.shape, xv.shape, xte.shape))
ok('cifar_klassen', set(np.unique(ytr)) == set(range(10)) and all(np.bincount(yte) == 1000), np.bincount(yte).tolist())
ok('cifar_standardisiert', abs(float(xtr.mean())) < 1e-3 and abs(float(xtr.std()) - 1) < 1e-3, (float(xtr.mean()), float(xtr.std())))
raw = np.fromfile(B.DATA / 'cifar10' / 'cifar-10-batches-bin' / 'data_batch_1.bin', dtype=np.uint8, count=3073)
ok('cifar_erster_datensatz', int(raw[0]) == int(ytr[0]), (int(raw[0]), int(ytr[0])))
xp = B.data_for('cifar10', 'tat')[0]; ok('cifar_tat_normiert', abs(float(np.mean(xp[:100] ** 2, axis=1).mean()) - 1) < 1e-4)

# 3. Netz mit 3072 Eingaengen, TAT orthogonal
m = B.Net(100, 'tat', 42084, din=3072); W1 = np.array(m.hid[1].weight)
ok('tat_init_orthogonal', np.allclose(W1 @ W1.T, np.eye(256), atol=1e-4))
m2 = B.Net(100, 'kL2B', 42084, din=3072); ok('netz_3072_eingaenge', m2.hid[0].weight.shape == (256, 3072))
m3 = B.Net(100, 'kL2B', 42060); m4 = B1.Net(100, 'kL2B', 42060)
ok('netz_784_wie_B1', all(np.array_equal(np.array(a.weight), np.array(b.weight)) for a, b in zip(m3.hid, m4.hid)))

# 4. Laufzahlen
ok('n_tune', len(list(B.tune_configs())) == 192, len(list(B.tune_configs())))
ok('n_confirm', len(B.REPRO) + len(B.DEPTHS) * len(B.DATASETS) * len(B.SEEDS) * len(B.ACTS) == 97)
ok('seeds_neu', set(B.SEEDS) == set(range(42084, 42096)))

# 5. execute: Kontrollpfad bitgleich zu B1 (synthetisch), CIFAR kurz fuer alle Varianten
for M in (B, B1): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
keys = ['train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'act_range_max_median', 'sat_rate']
c = dict(dataset='synth', depth=100, act='kL2B', lr=2e-5, seed=42060); o1 = B.execute(dict(c), 0); o2 = B1.execute(dict(c), 0)
ok('kL2B_bitgleich_B1', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys), {k: (o1[k], o2[k]) for k in keys[:2]})
for act in ('kL2B', 'LB', 'kL2', 'tat'):
    o = B.execute(dict(dataset='cifar10', depth=100, act=act, lr=1e-4, seed=42084), 0)
    ok(f'{act}_cifar_laeuft', o['status'] == 'ok' and math.isfinite(o['test_err_w8a3_best']) and o['test_err_fp32'] < 0.9,
       (round(o['test_err_fp32'], 4), round(o['test_err_w8a4_best'], 4), round(o['test_err_w8a4a_best'], 4), round(o['test_err_w8a3_best'], 4)))

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
