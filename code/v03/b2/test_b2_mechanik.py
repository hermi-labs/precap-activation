# Mechanik-Test B2 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: harte Formen LBh und kL2Bh am Fixpunkt (Mittel 0, Varianz 1, chi1 = 1 + 2/L) und begrenzt, MLX = numpy, harte Huelle = clip,
# LB und kL2B unveraendert gegen B1 (Konstanten, Aktivierung, execute bitgleich auf synthetischen Daten), 3-Bit-Quantisierung (7 bzw. 8 Stufen),
# Laeufe, Seeds und Lernraten.
import importlib.util, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b2_run.py', 'B2'); B1 = load(H.parent / 'b1' / 'b1_run.py', 'B1')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Harte Formen am Fixpunkt (feines Gitter, unabhaengig vom Loeser in math/p7)
X = np.linspace(-16, 16, 1_600_001); W = np.exp(-X * X / 2); W /= W.sum(); h = 1e-7
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for d in (100, 200):
    for n, Mx in (('LBh', B.CONST[('LBh', d)]['M']), ('kL2Bh', 3.0)):
        y = B.np_act(n, X, d); dy = (B.np_act(n, X + h, d) - B.np_act(n, X - h, d)) / (2 * h)
        m, v, c = float(np.sum(W * y)), float(np.sum(W * y * y)), float(np.sum(W * dy * dy))
        ok(f'{n}_d{d}_fixpunkt', abs(m) < 1e-5 and abs(v - 1) < 1e-4 and abs(c - (1 + 2 / d)) < 2e-4, f'Mittel {m:.1e} Varianz {v:.6f} chi1 {c:.6f} Soll {1 + 2 / d:.4f}')
        a = np.array(B.make_act(n, d)(z)); ok(f'{n}_d{d}_grenze', float(np.abs(a).max()) <= Mx + 1e-5 and abs(float(np.abs(a).max()) - Mx) < 1e-4, float(np.abs(a).max()))
        ok(f'{n}_d{d}_mlx_vs_numpy', float(np.abs(a - B.np_act(n, np.array(z, dtype=np.float64), d)).max()) < 1e-5)
    p = B.CONST[('kL2Bh', d)]
    ok(f'kL2Bh_d{d}_ist_clip_von_precap', np.array_equal(np.array(B.make_act('kL2Bh', d)(z)), np.array(mx.clip(B.pc_mlx(z, **p), -3.0, 3.0))))
    pl = B.CONST[('LBh', d)]
    ok(f'LBh_d{d}_ist_clip_linear', np.array_equal(np.array(B.make_act('LBh', d)(z)), np.array(mx.clip(pl['beta'] * z, -pl['M'], pl['M']))))
    # 2. LB und kL2B unveraendert gegen B1
    for n in ('LB', 'kL2B'):
        ok(f'{n}_d{d}_konstanten_wie_B1', B.CONST[(n, d)] == B1.CONST[(n, d)])
        ok(f'{n}_d{d}_akt_wie_B1', np.array_equal(np.array(B.make_act(n, d)(z)), np.array(B1.make_act(n, d)(z))))
ok('gains_LB_kL2B_wie_B1', all(B.GAINS[(n, d)] == B1.GAINS[(n, d)] for n in ('LB', 'kL2B') for d in (100, 200)))

# 3. Laeufe, Seeds, Lernraten
cf = list(B.configs()); ok('n_laeufe', len(cf) == 194, len(cf))
ok('seeds_neu', set(c['seed'] for c in cf) - {42060} == set(range(42072, 42084)))
lr = {(c['dataset'], c['depth'], c['act']): c['lr'] for c in cf}; l1 = {(c['dataset'], c['depth'], c['act']): c['lr'] for c in B1.configs()}
ok('lr_alle_wie_kL2_in_B1', all(lr[(ds, d, a)] == l1[(ds, d, 'kL2')] for ds in ('mnist', 'fashion') for d in (100, 200) for a in B.ACTS))

# 4. 3-Bit-Quantisierung: symmetrisch 7 Stufen, asymmetrisch 8 Stufen mit exakter 0
t = mx.array(np.random.default_rng(3).normal(size=50000).astype(np.float32))
ok('q3_sym_7_stufen', len(np.unique(np.array(B.qa(t, 'i3', 2.5)))) == 7, len(np.unique(np.array(B.qa(t, 'i3', 2.5)))))
ok('q3_asym_8_stufen', len(np.unique(np.array(B.qa_asym(t, 3, (-2.0, 2.5))))) == 8 and float(B.qa_asym(mx.array([0.0]), 3, (-2.0, 2.5))[0]) == 0.0)
ok('q4_unveraendert', np.array_equal(np.array(B.qa(t, 'i4', 2.5)), np.array(B1.qa(t, 'i4', 2.5))))

# 5. execute: LB und kL2B bitgleich zu B1 (synthetische Daten, Tiefe 100, 60 Schritte), harte Formen laufen
for M in (B, B1): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
keys = ['train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'w8a4_best_cal', 'test_err_w8a4a_best', 'w8a4a_best_cal', 'test_err_w8a8', 'act_range_max_median', 'grad_log10_first_last', 'sat_rate']
for act in ('LB', 'kL2B'):
    c = dict(dataset='synth', depth=100, act=act, lr=2e-5, seed=42060)
    o1 = B.execute(dict(c), 0); o2 = B1.execute(dict(c), 0)
    ok(f'{act}_bitgleich_B1', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys), {k: (o1[k], o2[k]) for k in keys[:3]})
    ok(f'{act}_3bit_da', 0 <= o1['test_err_w8a3_best'] <= 1 and o1['w8a3_best_cal'] in B.CALS3 and 0 <= o1['test_err_w8a3a_best'] <= 1, (o1['test_err_w8a3_best'], o1['w8a3_best_cal'], o1['test_err_w8a3a_best']))
for act in ('LBh', 'kL2Bh'):
    o = B.execute(dict(dataset='synth', depth=100, act=act, lr=2e-5, seed=42072), 0)
    Mx = B.CONST[('LBh', 100)]['M'] if act == 'LBh' else 3.0
    ok(f'{act}_laeuft', o['status'] == 'ok' and math.isfinite(o['test_err_w8a3_best']), (round(o['test_err_fp32'], 4), round(o['test_err_w8a4_best'], 4), round(o['test_err_w8a3_best'], 4)))
    ok(f'{act}_bereich_bis_M', o['act_range_max_median'] <= Mx + 1e-5, o['act_range_max_median'])
    ok(f'{act}_saettigung_gemessen', 0 <= o['sat_rate'] <= 1, o['sat_rate'])

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
