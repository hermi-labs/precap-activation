# Mechanik-Test B4 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: LBhM (hart, M wie LB) und LBM (weich, M wie LBh) mit Varianz 1 und chi1 wie in math/p11 (Gitter), Grenzen exakt gleich den
# Partnern, Formen gleich clip bzw. Lagma-Bremse, LB/LBh unveraendert gegen B2 (Konstanten, Aktivierung, execute bitgleich), Laeufe/Seeds/LR.
import importlib.util, json, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b4_run.py', 'B4'); B2 = load(H.parent / 'b2' / 'b2_run.py', 'B2'); J = json.load(open(H / 'p11_gleiches_M.json'))
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)
X = np.linspace(-16, 16, 1_600_001); W = np.exp(-X * X / 2); W /= W.sum(); h = 1e-7
z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
for d in (100, 200):
    for n, partner in (('LBhM', 'LB'), ('LBM', 'LBh')):
        y = B.np_act(n, X, d); dy = (B.np_act(n, X + h, d) - B.np_act(n, X - h, d)) / (2 * h)
        v, c = float(np.sum(W * y * y)), float(np.sum(W * dy * dy))
        ok(f'{n}_d{d}_varianz_chi1', abs(v - 1) < 1e-4 and abs(c - J[f'{n}_{d}']['chi1']) < 2e-4, f'Varianz {v:.6f} chi1 {c:.6f} (p11 {J[f"{n}_{d}"]["chi1"]:.6f})')
        ok(f'{n}_d{d}_grenze_gleich_{partner}', B.CONST[(n, d)]['M'] == B.CONST[(partner, d)]['M'], B.CONST[(n, d)]['M'])
        a = np.array(B.make_act(n, d)(z)); ok(f'{n}_d{d}_mlx_vs_numpy', float(np.abs(a - B.np_act(n, np.array(z, dtype=np.float64), d)).max()) < 1e-5)
    p = B.CONST[('LBhM', d)]; ok(f'LBhM_d{d}_ist_clip', np.array_equal(np.array(B.make_act('LBhM', d)(z)), np.array(mx.clip(p['beta'] * z, -p['M'], p['M']))))
    p = B.CONST[('LBM', d)]; ok(f'LBM_d{d}_ist_bremse', np.array_equal(np.array(B.make_act('LBM', d)(z)), np.array(B.brake(p['beta'] * z, mx.array(p['M'])))))
    for n in ('LB', 'LBh'):
        ok(f'{n}_d{d}_wie_B2', B.CONST[(n, d)] == B2.CONST[(n, d)] and np.array_equal(np.array(B.make_act(n, d)(z)), np.array(B2.make_act(n, d)(z))) and B.GAINS[(n, d)] == B2.GAINS[(n, d)])
cf = list(B.configs()); ok('n_laeufe', len(cf) == 194, len(cf)); ok('seeds_neu', set(c['seed'] for c in cf) - {42072} == set(range(42096, 42108)))
l2 = {(c['dataset'], c['depth'], c['act']): c['lr'] for c in B2.configs()}
ok('lr_wie_B2', all(c['lr'] == l2[(c['dataset'], c['depth'], 'LB')] for c in cf))
for M in (B, B2): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
keys = ['train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median', 'sat_rate']
for act in ('LB', 'LBh'):
    c = dict(dataset='synth', depth=100, act=act, lr=2e-5, seed=42072); o1 = B.execute(dict(c), 0); o2 = B2.execute(dict(c), 0)
    ok(f'{act}_bitgleich_B2', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys), {k: (o1[k], o2[k]) for k in keys[:2]})
for act in ('LBhM', 'LBM'):
    o = B.execute(dict(dataset='synth', depth=100, act=act, lr=2e-5, seed=42096), 0)
    ok(f'{act}_laeuft', o['status'] == 'ok' and math.isfinite(o['test_err_w8a4_best']) and o['act_range_max_median'] <= B.CONST[(act, 100)]['M'] + 1e-5 and 0 <= o['sat_rate'] <= 1,
       (round(o['test_err_fp32'], 4), round(o['act_range_max_median'], 4), round(o['sat_rate'], 4)))
print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
