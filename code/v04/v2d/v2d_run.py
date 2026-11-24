# V2d (privat, mit Claude, praeregistriert in V2d_Praeregistrierung.md): kalibrierte tanh als Form im V2a2-Netz (CNN L = 50 ohne Normierung
# und ohne Residuen, Delta-orthogonal, CIFAR-10, 4 Bit PTQ). Die eine Aenderung gegenueber V2a2/V2c: Aktivierung ctanh(x) = LAM * tanh(B x),
# LAM und B nach demselben Rezept wie BPreCap (E f = 0, E f^2 = 1 bei z ~ N(0,1), chi1 = E[f'(z)^2] = 1 + 2/L = 1,04).
# Alles andere ist der unveraenderte, eingefrorene Code studien/v2a2/v2a2_run.py (Import, keine Kopie). BPreCap, SELU, TAT kommen historisch
# aus V2a2, tanh aus V2c (gleiche Seeds 42350-42361, gleicher Code, gleiches Geraet). PreCap, Kalibrierungsrezept und Huelle: Ideen Aarons.
# Aufruf: --selfcheck | --smoke | --control | --worker i --workers n
import json, sys, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent; V2A2 = ROOT.parent / 'v2a2'; sys.path.insert(0, str(V2A2))
import numpy as np
import v2a2_run as R
import mlx.core as mx
LAM, B = 2.2948782077912586, 0.5328431680484216   # Loesung von E f^2 = 1, chi1 = 1.04 (scipy brentq, Gauss-Hermite 301 Punkte, 19.11.)
_orig_act = R.make_act
def ctanh(x): return LAM * mx.tanh(B * x)
def make_act(name): return ctanh if name == 'ctanh' else _orig_act(name)
R.make_act = make_act
LR_GRID = {'ctanh': (1e-5, 3e-5, 1e-4)}   # Regel der Praeregistrierung: beste Pilot-LR (3e-5, Val 51,9 %) und ihre Nachbarn im Pilotgitter
R.ACTS = ('ctanh',); R.LR_GRID = LR_GRID
def selfcheck():
    out = dict(L=R.L, C=R.CH, steps=R.STEPS, n_tune=len(R.tune_cfgs()), lr_grid=LR_GRID, wd_grid=R.WD_GRID, tune_seeds=R.TUNE_SEEDS, seeds=list(R.SEEDS))
    X, Wq = np.polynomial.hermite_e.hermegauss(301); Wq = Wq / Wq.sum(); x = mx.array(X.astype(np.float32))
    y = np.array(make_act('ctanh')(x), dtype=np.float64); out['Ef_ctanh'] = round(float(np.sum(Wq * y)), 4); out['Ef2_ctanh'] = round(float(np.sum(Wq * y * y)), 4)
    d = LAM * B * (1 - np.tanh(B * X) ** 2); out['chi1_ctanh'] = round(float(np.sum(Wq * d * d)), 4); out['bereich_ctanh'] = round(LAM, 4)
    out['ctanh_wie_numpy'] = bool(np.allclose(y, LAM * np.tanh(B * X), atol=1e-5))
    out['bprecap_unveraendert'] = make_act('bprecap') is not ctanh and abs(float(np.sum(Wq * np.array(make_act('bprecap')(x), dtype=np.float64) ** 2)) - 1) < 0.01
    fake = {'ctanh': dict(lr=LR_GRID['ctanh'][1], wd=0.0)}; cc = R.confirm_cfgs(fake); out['n_confirm'] = len(cc); out['confirm_acts'] = sorted({c['act'] for c in cc})
    out['seeds_wie_v2a2'] = list(R.SEEDS) == list(range(42350, 42362)) and tuple(R.TUNE_SEEDS) == (42000, 42001)
    print(json.dumps(out, indent=1)); return out
def control():
    # Volle Kontrolle: V2a2-Zeile BPreCap Seed 42350 (LR 3e-5, WD 0) mit diesem Skript neu rechnen, alle Messspalten muessen als Text gleich sein.
    ref = [r for r in csv.DictReader(open(V2A2 / 'workers' / 'w0.csv')) if r['run_id'] == 'confirm_bprecap_lr3e-05_wd0_s42350'][0]
    o = R.run_one(dict(stage='confirm', act='bprecap', lr=3e-5, wd=0.0, seed=42350))
    skip = ('sec', 'timestamp'); diff = []
    for c in R.COLS:
        if c in skip: continue
        a, b = str(o.get(c, '')), ref.get(c, '')
        if a != b: diff.append((c, a, b))
    print(json.dumps(dict(n_cols=len(R.COLS) - len(skip), n_diff=len(diff), diff=diff[:10]), indent=1))
    print('KONTROLLE OK' if not diff else 'KONTROLLE FEHLER'); return not diff
def main():
    a = sys.argv[1:]
    print('Geraet:', mx.default_device(), flush=True)
    if '--selfcheck' in a: selfcheck(); return
    if '--control' in a: sys.exit(0 if control() else 1)
    if '--smoke' in a:
        g = {k: v[:1] for k, v in LR_GRID.items()}; R.LR_GRID = g
        R.work(0, 1, ROOT / 'smoke', R.tune_cfgs(R.TUNE_SEEDS[:1], g), lambda rows: R.select(rows, g, R.TUNE_SEEDS[:1]), dict(steps=20, _seeds=R.SEEDS[:1])); return
    wid = int(a[a.index('--worker') + 1]) if '--worker' in a else 0; nw = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    R.work(wid, nw, ROOT, R.tune_cfgs(), R.select)
if __name__ == '__main__': main()
