# V2g (privat, mit Claude, praeregistriert in V2g_Praeregistrierung.md): kritische tanh nach Xiao et al. 2018 (xtanh) im V2a2-Netz
# (CNN L = 50 ohne Normierung und ohne Residuen, Delta-orthogonal, CIFAR-10, 4 Bit PTQ). Die eine Aenderung: Aktivierung tanh, Faltungsgewichte
# x sigma_w, Bias je Kanal aus N(0, sigma_b^2) mit eigenem Generator; sigma_b^2 = 2e-5 (Xiao Abb. 2), sigma_w^2 so, dass chi1 = 1 im Fixpunkt q*.
# Alles andere ist der unveraenderte, eingefrorene Code studien/v2a2/v2a2_run.py (Import, keine Kopie). PreCap, BPreCap, Huelle: Ideen Aarons.
# Aufruf: --selfcheck | --smoke | --worker i --workers n
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent; V2A2 = ROOT.parent / 'v2a2'; sys.path.insert(0, str(V2A2))
import numpy as np
import v2a2_run as R
import mlx.core as mx
from mlx.utils import tree_flatten
SB2, SW2, QSTAR = 2e-5, 1.04991163197033, 0.025873538320557977   # scipy brentq, Gauss-Hermite 301 Punkte, 20.11. 07:00
def xtanh(x): return mx.tanh(x)
_orig_act = R.make_act; _OrigCNN = R.CNN
def make_act(name): return xtanh if name == 'xtanh' else _orig_act(name)
class XCNN(_OrigCNN):
    def __init__(self, act, seed, Ld=None):
        super().__init__(act, seed, Ld)
        if act is xtanh:
            rb = np.random.default_rng([seed, 7]); sw = math.sqrt(SW2)
            for c in self.convs:
                c.weight = c.weight * sw; c.bias = mx.array(rb.normal(0, math.sqrt(SB2), size=c.bias.shape).astype(np.float32))
R.make_act = make_act; R.CNN = XCNN
LR_GRID = {'xtanh': (1e-5, 3e-5, 1e-4)}   # Raster von V2c/V2d, ohne eigenen Pilot (Praeregistrierung)
R.ACTS = ('xtanh',); R.LR_GRID = LR_GRID
def selfcheck():
    out = dict(L=R.L, C=R.CH, steps=R.STEPS, n_tune=len(R.tune_cfgs()), lr_grid=LR_GRID, wd_grid=R.WD_GRID, tune_seeds=R.TUNE_SEEDS, seeds=list(R.SEEDS), SB2=SB2, SW2=SW2)
    X, Wq = np.polynomial.hermite_e.hermegauss(301); Wq = Wq / Wq.sum(); u = math.sqrt(QSTAR) * X
    out['chi1_qstar'] = round(float(SW2 * np.sum(Wq * (1 - np.tanh(u) ** 2) ** 2)), 6); out['fixpunkt_fehler'] = float(abs(SW2 * np.sum(Wq * np.tanh(u) ** 2) + SB2 - QSTAR))
    out['fixpunkt_ok'] = out['fixpunkt_fehler'] < 1e-9 and abs(out['chi1_qstar'] - 1) < 1e-6
    x = mx.array(X.astype(np.float32)); out['xtanh_ist_tanh'] = bool(np.allclose(np.array(make_act('xtanh')(x)), np.tanh(X), atol=1e-6))
    pb, po = dict(tree_flatten(XCNN(make_act('bprecap'), 42000).parameters())), dict(tree_flatten(_OrigCNN(_orig_act('bprecap'), 42000).parameters()))
    out['bprecap_init_bitgleich'] = all(bool(mx.all(pb[k] == po[k])) for k in po)
    px, pt = dict(tree_flatten(XCNN(xtanh, 42000).parameters())), dict(tree_flatten(_OrigCNN(mx.tanh, 42000).parameters()))
    out['xtanh_gewichte_sigma_w'] = all(bool(mx.all(px[f'convs.{i}.weight'] == pt[f'convs.{i}.weight'] * math.sqrt(SW2))) for i in range(R.L))
    bs = np.concatenate([np.array(px[f'convs.{i}.bias']) for i in range(R.L)]); out['bias_std'] = round(float(bs.std()), 6); out['bias_std_soll'] = round(math.sqrt(SB2), 6)
    out['bias_ok'] = abs(bs.std() / math.sqrt(SB2) - 1) < 0.05 and abs(bs.mean()) < 5 * math.sqrt(SB2 / bs.size)
    out['ausgabe_unveraendert'] = bool(mx.all(px['out.weight'] == pt['out.weight'])) and bool(mx.all(px['out.bias'] == 0))
    # 20 Trainingsschritte BPreCap: gepatchte gegen Original-Klasse bitgleich
    R.CNN = _OrigCNN; m0 = R.train('bprecap', 3e-5, 0.0, 'cos', 42000, 20)[0]; R.CNN = XCNN; m1 = R.train('bprecap', 3e-5, 0.0, 'cos', 42000, 20)[0]
    a0, a1 = dict(tree_flatten(m0.parameters())), dict(tree_flatten(m1.parameters())); out['bprecap_20schritte_bitgleich'] = all(bool(mx.all(a0[k] == a1[k])) for k in a0)
    fake = {'xtanh': dict(lr=LR_GRID['xtanh'][1], wd=0.0)}; cc = R.confirm_cfgs(fake); out['n_confirm'] = len(cc); out['confirm_acts'] = sorted({c['act'] for c in cc})
    out['seeds_wie_v2a2'] = list(R.SEEDS) == list(range(42350, 42362)) and tuple(R.TUNE_SEEDS) == (42000, 42001)
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
