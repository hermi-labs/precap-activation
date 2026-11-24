# V2h (privat, mit Claude, praeregistriert in V2h_Praeregistrierung.md): zweite kritische tanh (htanh) im V2a2-Netz, Punkt der kritischen
# Linie nach der Regel von Hayou et al. 2019 (beta_q = L = 50). Die eine Aenderung gegenueber V2gb: sigma_b^2 (und das zugehoerige sigma_w^2).
# Alles andere ist der unveraenderte, eingefrorene Code studien/v2a2/v2a2_run.py (Import, keine Kopie). PreCap, BPreCap, Huelle: Ideen Aarons.
# Aufruf: --selfcheck | --smoke | --worker i --workers n
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent; V2A2 = ROOT.parent / 'v2a2'; sys.path.insert(0, str(V2A2))
import numpy as np
import v2a2_run as R
import mlx.core as mx
from mlx.utils import tree_flatten
# scipy brentq, Gauss-Hermite 301 Punkte, 21.11. 18:35: sigma_b so, dass beta_q = 2 E[tanh'^2] / (q* E[tanh''^2]) = 50 (Hayou 2019, Prop. 4 und Abschnitt 4)
SB = 0.041221694095456875; SB2 = SB * SB; SW2 = 1.2273528768534143; QSTAR = 0.13186237664182868; BETA_SOLL = 50.0
def htanh(x): return mx.tanh(x)
_orig_act = R.make_act; _OrigCNN = R.CNN
def make_act(name): return htanh if name == 'htanh' else _orig_act(name)
class HCNN(_OrigCNN):
    def __init__(self, act, seed, Ld=None):
        super().__init__(act, seed, Ld)
        if act is htanh:
            rb = np.random.default_rng([seed, 7]); sw = math.sqrt(SW2)
            for c in self.convs:
                c.weight = c.weight * sw; c.bias = mx.array(rb.normal(0, SB, size=c.bias.shape).astype(np.float32))
R.make_act = make_act; R.CNN = HCNN
LR_GRID = {'htanh': (1e-5, 3e-5, 1e-4)}   # Raster von V2c/V2d/V2gb, ohne eigenen Pilot (Praeregistrierung)
R.ACTS = ('htanh',); R.LR_GRID = LR_GRID
def selfcheck():
    out = dict(L=R.L, C=R.CH, steps=R.STEPS, n_tune=len(R.tune_cfgs()), lr_grid=LR_GRID, wd_grid=R.WD_GRID, tune_seeds=R.TUNE_SEEDS, seeds=list(R.SEEDS), SB=SB, SB2=SB2, SW2=SW2)
    X, Wq = np.polynomial.hermite_e.hermegauss(301); Wq = Wq / Wq.sum(); u = math.sqrt(QSTAR) * X; t = np.tanh(u); d1 = 1 - t ** 2; d2 = -2 * t * d1
    out['chi1_qstar'] = round(float(SW2 * np.sum(Wq * d1 ** 2)), 6); out['fixpunkt_fehler'] = float(abs(SW2 * np.sum(Wq * t ** 2) + SB2 - QSTAR))
    out['beta_q'] = round(float(2 * np.sum(Wq * d1 ** 2) / (QSTAR * np.sum(Wq * d2 ** 2))), 6)
    out['fixpunkt_ok'] = bool(out['fixpunkt_fehler'] < 1e-9 and abs(out['chi1_qstar'] - 1) < 1e-6 and abs(out['beta_q'] - BETA_SOLL) < 1e-4)
    x = mx.array(X.astype(np.float32)); out['htanh_ist_tanh'] = bool(np.allclose(np.array(make_act('htanh')(x)), np.tanh(X), atol=1e-6))
    pb, po = dict(tree_flatten(HCNN(make_act('bprecap'), 42000).parameters())), dict(tree_flatten(_OrigCNN(_orig_act('bprecap'), 42000).parameters()))
    out['bprecap_init_bitgleich'] = bool(all(bool(mx.all(pb[k] == po[k])) for k in po))
    px, pt = dict(tree_flatten(HCNN(htanh, 42000).parameters())), dict(tree_flatten(_OrigCNN(mx.tanh, 42000).parameters()))
    out['htanh_gewichte_sigma_w'] = bool(all(bool(mx.all(px[f'convs.{i}.weight'] == pt[f'convs.{i}.weight'] * math.sqrt(SW2))) for i in range(R.L)))
    bs = np.concatenate([np.array(px[f'convs.{i}.bias']) for i in range(R.L)]); out['bias_std'] = round(float(bs.std()), 6); out['bias_std_soll'] = round(SB, 6)
    out['bias_ok'] = bool(abs(bs.std() / SB - 1) < 0.05 and abs(bs.mean()) < 5 * SB / math.sqrt(bs.size))
    out['ausgabe_unveraendert'] = bool(bool(mx.all(px['out.weight'] == pt['out.weight'])) and bool(mx.all(px['out.bias'] == 0)))
    # beschreibend: gemessene Varianz der Vor-Aktivierungen bei der Init (256 Trainingsbilder, Seed 42000) gegen die Mittelfeld-Rechnung
    m = HCNN(htanh, 42000); h = mx.array(R.load()[0][:256]); q_mess = []
    for c in m.convs:
        z = c(h); q_mess.append(float(mx.mean(z * z))); h = mx.tanh(z)
    q0 = q_mess[0]; q_mf = [q0]
    for _ in range(R.L - 1):
        uu = math.sqrt(q_mf[-1]) * X; q_mf.append(float(SW2 * np.sum(Wq * np.tanh(uu) ** 2) + SB2))
    out['q_init_mess'] = {str(i + 1): round(q_mess[i], 5) for i in (0, 1, 4, 9, 24, 49)}; out['q_init_mittelfeld'] = {str(i + 1): round(q_mf[i], 5) for i in (0, 1, 4, 9, 24, 49)}
    out['q_init_endlich'] = bool(all(math.isfinite(v) and v > 0 for v in q_mess))
    # 20 Trainingsschritte BPreCap: gepatchte gegen Original-Klasse bitgleich
    R.CNN = _OrigCNN; m0 = R.train('bprecap', 3e-5, 0.0, 'cos', 42000, 20)[0]; R.CNN = HCNN; m1 = R.train('bprecap', 3e-5, 0.0, 'cos', 42000, 20)[0]
    a0, a1 = dict(tree_flatten(m0.parameters())), dict(tree_flatten(m1.parameters())); out['bprecap_20schritte_bitgleich'] = bool(all(bool(mx.all(a0[k] == a1[k])) for k in a0))
    fake = {'htanh': dict(lr=LR_GRID['htanh'][1], wd=0.0)}; cc = R.confirm_cfgs(fake); out['n_confirm'] = len(cc); out['confirm_acts'] = sorted({c['act'] for c in cc})
    out['seeds_wie_v2a2'] = bool(list(R.SEEDS) == list(range(42350, 42362)) and tuple(R.TUNE_SEEDS) == (42000, 42001))
    print(json.dumps(out, indent=1)); return out
def main():
    a = sys.argv[1:]
    if '--cpu' in a: mx.set_default_device(mx.cpu)
    print('Geraet:', mx.default_device(), flush=True)
    if '--selfcheck' in a: selfcheck(); return
    if '--smoke' in a:
        g = {k: v[:1] for k, v in LR_GRID.items()}; R.LR_GRID = g
        R.work(0, 1, ROOT / 'smoke', R.tune_cfgs(R.TUNE_SEEDS[:1], g), lambda rows: R.select(rows, g, R.TUNE_SEEDS[:1]), dict(steps=20, _seeds=R.SEEDS[:1])); return
    wid = int(a[a.index('--worker') + 1]) if '--worker' in a else 0; nw = int(a[a.index('--workers') + 1]) if '--workers' in a else 1
    R.work(wid, nw, ROOT, R.tune_cfgs(), R.select)
if __name__ == '__main__': main()
