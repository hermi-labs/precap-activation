# Mechanik-Test B14 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: bestehende Formen, Daten und Konstanten gleich B10; PACT-Formen: bei der Init Vorwaertsrechnung bitgleich zur harten Form aus B13,
# Grenze je Schicht ist trainierbarer Parameter (Anzahl, Startwert, Gradient ungleich 0), Grenze aendert sich beim Training, Gewichtsabfall
# zieht sie nach unten, Kalibrierung und Quantisierung sehen die gelernte Grenze (Bereich <= alpha); Laufliste und Seeds; Kontrolle bitgleich zu B10;
# Zwischenmessung veraendert das Training nicht.
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx.utils import tree_flatten
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b14_run.py', 'B14'); B10 = load(H.parent / 'b10' / 'b10_run.py', 'B10'); B13 = load(H.parent / 'b13' / 'b13_run.py', 'B13')
mx.set_cache_limit(B.CACHE_LIMIT)
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for n in ('kL2B_tr', 'LB_tr', 'kL2Bh_tr', 'LBh_tr'):
    ok(f'{n}_form_wie_B10', np.array_equal(np.array(B.make_act(n, 100)(z)), np.array(B10.make_act(n, 100)(z))))
x = mx.array(B.data_for('cifar10', 'kL2Bp_tr')[0][:256])
ok('daten_wie_B10', all(np.array_equal(p, q) for p, q in zip(B.data_for('cifar10', 'kL2Bp_tr'), B10.data_for('cifar10', 'kL2B_tr'))))
for p_, h_, m0 in (('kL2Bp_tr', 'kL2Bh_tr', 3.0), ('LBp_tr', 'LBh_tr', B.CONST[('LBh', 100)]['M'])):
    a = B.Net(100, p_, 42168, din=3072); b = B13.Net(100, h_, 42168, din=3072); a.eval(); b.eval()
    ok(f'{p_}_init_gewichte_wie_{h_}', all(np.array_equal(np.array(u.weight), np.array(w.weight)) for u, w in zip(a.hid + [a.out], b.hid + [b.out])))
    ok(f'{p_}_vorwaerts_bei_init_bitgleich_{h_}', np.array_equal(np.array(a(x)), np.array(b(x))))
    pa = [k for k, _ in tree_flatten(a.trainable_parameters()) if k.startswith('pa.')]
    ok(f'{p_}_grenze_trainierbar', len(pa) == 100 and all(float(v) == np.float32(m0) for v in a.pa), (len(pa), float(a.pa[0])))
    a.train(); _, g = nn.value_and_grad(a, B.loss_fn)(a, x, mx.array(B.data_for('cifar10', p_)[1][:256]))
    gn = [float(mx.abs(v)) for v in g['pa']]; ok(f'{p_}_gradient_grenze', sum(v > 0 for v in gn) >= 50, f'{sum(v > 0 for v in gn)} von 100 Schichten')
    a.eval(); Rr = B.calib(a, np.array(x)); ok(f'{p_}_kalibrierung_sieht_grenze', max(Rr['max'][1:-1]) <= m0 + 1e-5, round(max(Rr['max'][1:-1]), 4))
ok('konstanten_wie_B10', (B.STEPS, B.CHECKS, B.WIDTH, B.BATCH, B.NVAL, B.NCAL, B.CALS4, B.CALS3, B.CACHE_LIMIT, B.TUNE_SEEDS, B.WD_GRID) ==
   (B10.STEPS, B10.CHECKS, B10.WIDTH, B10.BATCH, B10.NVAL, B10.NCAL, B10.CALS4, B10.CALS3, B10.CACHE_LIMIT, B10.TUNE_SEEDS, B10.WD_GRID))
ok('raster_wie_B13', B.LR_GRID == B13.LR_GRID == [5e-5, 1e-4, 2e-4, 5e-4] and B.tune_acts(100) == ['kL2Bp_tr', 'LBp_tr'])
T = list(B.tune_configs()); ok('n_tune', len(T) == 48, len(T))
C = list(B.configs()); ok('n_bestaetigung', len(C) == 25, len(C))
ok('kontrolle', C[0] == dict(dataset='cifar10', depth=100, act='kL2B_tr', lr=1e-4, wd=0.3, sched='cos', seed=42168), C[0])
ok('seeds_wie_B10', B.SEEDS == B10.SEEDS == list(range(42168, 42180)))
keys = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
B.STEPS = B10.STEPS = 172; B.CHECKS = B10.CHECKS = (86,)
B.OUTDIR = Path(tempfile.mkdtemp()); o = B.execute(dict(C[0]), 0); B10.OUTDIR = Path(tempfile.mkdtemp()); o10 = B10.execute(dict(C[0]), 0)
ok('kontrolle_bitgleich_B10', o['status'] == 'ok' and all(o[k] == o10[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8', 'lr_end']) and all(o[f'c86_{k}'] == o10[f'c86_{k}'] for k in keys))
B13.STEPS = 172; B13.CHECKS = (86,)
for p_, h_, m0 in (('kL2Bp_tr', 'kL2Bh_tr', 3.0), ('LBp_tr', 'LBh_tr', B.CONST[('LBh', 100)]['M'])):
    c = dict(dataset='cifar10', depth=100, act=p_, lr=1e-4, wd=0.3, sched='cos', seed=42168)
    B.CHECKS = (86,); B.OUTDIR = Path(tempfile.mkdtemp()); oc = B.execute(dict(c), 0)
    B.CHECKS = (); B.OUTDIR = Path(tempfile.mkdtemp()); on = B.execute(dict(c), 0)
    B.OUTDIR = Path(tempfile.mkdtemp()); ow = B.execute(dict(c, wd=3.0), 0)
    B13.OUTDIR = Path(tempfile.mkdtemp()); oh = B13.execute(dict(c, act=h_), 0)
    ok(f'{p_}_lauf_ok', oc['status'] == 'ok' and 0 < oc['test_err_fp32'] < 0.9, round(oc['test_err_fp32'], 4))
    ok(f'{p_}_grenze_lernt', oc['pact_alpha_min'] != oc['pact_alpha_max'] and abs(oc['pact_alpha_median'] - m0) > 0, (round(oc['pact_alpha_min'], 4), round(oc['pact_alpha_median'], 4), round(oc['pact_alpha_max'], 4)))
    ok(f'{p_}_wd_zieht_grenze', ow['pact_alpha_median'] < oc['pact_alpha_median'], (round(ow['pact_alpha_median'], 4), round(oc['pact_alpha_median'], 4)))
    ok(f'{p_}_anders_als_hart_nach_training', oc['train_loss_last50'] != oh['train_loss_last50'] or oc['test_err_fp32'] != oh['test_err_fp32'])
    ok(f'{p_}_bereich_bis_grenze', oc['act_range_max_median'] <= oc['pact_alpha_max'] + 1e-5, (round(oc['act_range_max_median'], 4), round(oc['pact_alpha_max'], 4)))
    ok(f'{p_}_zwischen_aendert_training_nicht', all(oc[k] == on[k] for k in keys + ['test_err_w8a3a_best', 'test_err_w8a8', 'pact_alpha_median']) and oc['sec_checks'] > 0)
print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
