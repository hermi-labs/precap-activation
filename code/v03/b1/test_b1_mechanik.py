# Mechanik-Test B1 (vor dem Freeze). Ausgabe "MECHANIK OK" oder Fehler.
# Prueft: Konstanten von kL2B exakt (Mittel 0, Varianz 1, chi1 = 1 + 2/L, Grenze 3), Bremse in der Aktivierung = brake(),
# kL2 und kL2_LD100 laufen bitgleich zu Lagma-5 (execute auf synthetischen Daten), asymmetrische Quantisierung wie Q3,
# LB (linearer Kern + Grenze) mit Varianz 1 und chi1 = 1 + 2/L, LR-Zuordnung, Zahl der Laeufe.
import importlib.util, sys, tempfile, math
from pathlib import Path
import numpy as np, mlx.core as mx
H = Path(__file__).resolve().parent
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); M = importlib.util.module_from_spec(sp); sp.loader.exec_module(M); return M
B = load(H / 'b1_run.py', 'B1'); L5 = load(H.parent / 'lagma5' / 'lagma5_run.py', 'L5')
R = {}
def ok(name, cond, info=''):
    R[name] = bool(cond); print(('ok   ' if cond else 'FEHLER ') + name, info, flush=True)

# 1. Konstanten von kL2B auf feinem Gitter (unabhaengig von der Gauss-Hermite-Pruefung im Selbsttest)
X = np.linspace(-14, 14, 1_400_001); W = np.exp(-X * X / 2); W /= W.sum(); h = 1e-6
for d in (100, 200):
    y = B.np_act('kL2B', X, d); dy = (B.np_act('kL2B', X + h, d) - B.np_act('kL2B', X - h, d)) / (2 * h)
    m, v, c = float(np.sum(W * y)), float(np.sum(W * y * y)), float(np.sum(W * dy * dy))
    ok(f'kL2B_d{d}_fixpunkt', abs(m) < 1e-5 and abs(v - 1) < 1e-5 and abs(c - (1 + 2 / d)) < 1e-4, f'Mittel {m:.2e} Varianz {v:.6f} chi1 {c:.6f} Soll {1 + 2 / d:.4f}')
    yr = B.np_act('kL2B', np.array([-1e3, 1e3]), d); ok(f'kL2B_d{d}_grenze', yr[0] >= -3 - 1e-9 and yr[1] <= 3 + 1e-9, yr.round(4).tolist())
    z = mx.array(np.linspace(-30, 30, 60001).astype(np.float32))
    a = np.array(B.make_act('kL2B', d)(z)); b = np.array(B.brake(B.pc_mlx(z, **B.CONST[('kL2B', d)]), mx.array(3.0)))
    ok(f'kL2B_d{d}_akt_gleich_bremse', np.array_equal(a, b))
    ok(f'kL2B_d{d}_mlx_vs_numpy', float(np.abs(a - B.np_act('kL2B', np.array(z, dtype=np.float64), d)).max()) < 1e-5)
    ok(f'kL2_d{d}_konstanten_wie_lagma5', B.CONST[('kL2', d)] == L5.CONST[('kL2', d)])
    y = B.np_act('LB', X, d); dy = (B.np_act('LB', X + h, d) - B.np_act('LB', X - h, d)) / (2 * h); pl = B.CONST[('LB', d)]
    m, v, c = float(np.sum(W * y)), float(np.sum(W * y * y)), float(np.sum(W * dy * dy))
    ok(f'LB_d{d}_fixpunkt', abs(m) < 1e-9 and abs(v - 1) < 1e-5 and abs(c - (1 + 2 / d)) < 1e-4, f'Mittel {m:.1e} Varianz {v:.6f} chi1 {c:.6f} M {pl["M"]:.4f}')
    a3 = np.array(B.make_act('LB', d)(z)); ok(f'LB_d{d}_grenze_und_numpy', float(np.abs(a3).max()) <= pl['M'] + 1e-5 and float(np.abs(a3 - B.np_act('LB', np.array(z, dtype=np.float64), d)).max()) < 1e-5)
    ok(f'kL2_d{d}_akt_wie_lagma5', np.array_equal(np.array(B.make_act('kL2', d)(z)), np.array(L5.make_act('kL2', d)(z))))
ok('cs_konstanten_wie_paket', all(abs(B.CONST[('spreccap_cs', 0)][k] - B.PRESETS['spreccap_cs'][k]) < 1e-9 for k in ('lam', 'alpha', 'b')))
ok('lagma_param_wie_lagma5', (B.LG_Q, B.LG_KAPPA, B.LG_RHO, B.LG_ETA, B.LG_RAMP, B.LG_UP['LD100']) == (L5.LG_Q, L5.LG_KAPPA, L5.LG_RHO, L5.LG_ETA, L5.LG_RAMP, L5.LG_UP['LD100']))

# 2. Laeufe, Seeds, Lernraten
cf = list(B.configs()); ok('n_laeufe', len(cf) == 242, len(cf))
lr = {(c['dataset'], c['depth'], c['act']): c['lr'] for c in cf}
ok('lr_kL2B_gleich_kL2', all(lr[(ds, d, 'kL2B')] == lr[(ds, d, 'kL2')] == lr[(ds, d, 'LB')] == lr[(ds, d, 'kL2_LD100')] for ds in ('mnist', 'fashion') for d in (100, 200)))
l5 = {(c['dataset'], c['depth'], c['act']): c['lr'] for c in L5.configs()}
ok('lr_kL2_wie_lagma5', all(lr[(ds, d, 'kL2')] == l5[(ds, d, 'kL2')] for ds in ('mnist', 'fashion') for d in (100, 200)))
ok('lr_cs_vorhanden', all((ds, d, 'spreccap_cs') in lr for ds in ('mnist', 'fashion') for d in (100, 200)), {k: v for k, v in lr.items() if k[2] == 'spreccap_cs'})
ok('seeds_neu', set(c['seed'] for c in cf) - {42045} == set(range(42060, 42072)))

# 3. kL2 und kL2_LD100 bitgleich zu Lagma-5 (execute, synthetische Daten, Tiefe 100, 60 Schritte)
for M in (B, L5): M.STEPS = 60; M.OUTDIR = Path(tempfile.mkdtemp())
for act in ('kL2', 'kL2_LD100'):
    c = dict(dataset='synth', depth=100, act=act, lr=2e-5, seed=42045)
    o1 = B.execute(dict(c), 0); o2 = L5.execute(dict(c), 0)
    keys = ['train_loss_last50', 'test_err_fp32', 'test_err_w8a4_best', 'w8a4_best_cal', 'test_err_w8a8', 'act_range_max_median', 'grad_log10_first_last']
    if act != 'kL2': keys += ['lg_c_median', 'lg_intervention']
    ok(f'{act}_bitgleich_lagma5', o1['status'] == 'ok' and all(o1[k] == o2[k] for k in keys), {k: (o1[k], o2[k]) for k in keys[:3]})
    ok(f'{act}_asym_da', 0 <= o1['test_err_w8a4a_best'] <= 1 and o1['w8a4a_best_cal'] in B.CALS4, (o1['test_err_w8a4a_best'], o1['w8a4a_best_cal']))
for act in ('kL2B', 'LB', 'spreccap_cs'):
    o = B.execute(dict(dataset='synth', depth=100, act=act, lr=2e-5, seed=42060), 0)
    ok(f'{act}_laeuft', o['status'] == 'ok' and math.isfinite(o['test_err_w8a4a_best']), (round(o['test_err_fp32'], 4), round(o['test_err_w8a4_best'], 4), round(o['test_err_w8a4a_best'], 4)))
    if act != 'spreccap_cs': ok(f'{act}_bereich_unter_M', o['act_range_max_median'] <= (3.0 if act == 'kL2B' else B.CONST[('LB', 100)]['M']) + 1e-5, o['act_range_max_median'])
    if act != 'spreccap_cs': ok(f'{act}_saettigung_gemessen', 0 <= o['sat_rate'] <= 1, o['sat_rate'])

# 4. Asymmetrische Quantisierung wie Q3 (Funktionen bitgleich)
Q3 = load(H.parent / 'q3' / 'q3_run.py', 'Q3')
t = np.random.default_rng(5).normal(size=40000) * 1.3 + 0.4; tm = mx.array(t.astype(np.float32))
ok('qa_asym_wie_q3', np.array_equal(np.array(B.qa_asym(tm, 4, (-1.0, 2.5))), np.array(Q3.qa_asym(tm, 4, (-1.0, 2.5)))))
ok('mse_clip_asym_wie_q3', B.mse_clip_asym(t, 4) == Q3.mse_clip_asym(t, 4), B.mse_clip_asym(t, 4))
ok('asym_null_exakt', float(B.qa_asym(mx.array([0.0]), 4, (-1.0, 2.5))[0]) == 0.0)

print('MECHANIK OK' if all(R.values()) else 'MECHANIK FEHLER: ' + ', '.join(k for k, v in R.items() if not v), flush=True)
sys.exit(0 if all(R.values()) else 1)
