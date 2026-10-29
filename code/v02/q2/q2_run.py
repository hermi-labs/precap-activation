# Q2 (privat): TAT (Leaky ReLU, Zhang et al. 2022) im Q1b-Ablauf, gegen SPreCap-cs und SELU aus Q1b (gleiche Seeds, gleicher Code-Pfad).
# Frage: Ist das beschraenkte cs bei 4-Bit-Aktivierungen (faire Kalibrierung, w8a4_best) besser als das unbeschraenkte TAT?
# Abgeleitet aus q1b_run.py. Neu: Aktivierung 'tat' mit SUO-Init und Normierung je Beispiel wie in A2, LR-Tuning fuer TAT (wie A2),
# Reproduktion von zwei cs-Laeufen aus Q1b. Quantisierung, Kalibrierung und Auswertung je Lauf unveraendert.
import argparse, csv, gzip, json, math, os, struct, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data'))
sys.path.insert(0, str(Path(os.environ.get('PRECAP_SRC', ROOT / 'src'))))

DATASETS = ['mnist', 'fashion']; DEPTHS = [50, 100]; WIDTH = 256; BATCH = 128; STEPS = 1720; NVAL = 5000
ACTS = ['tat']                               # neu; cs nur zur Reproduktion
REF_ACTS = ['spreccap_cs', 'spreccap_e03', 'selu', 'elu']
LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]; TUNE_SEEDS = [42000, 42001, 42002, 42003]; STAGE = 'confirm'
REPRO = [('mnist', 100, 'spreccap_cs', 42010), ('mnist', 100, 'spreccap_cs', 42011)]
# TAT-Parameter (eta = 0.9 wie dks): d100 aus dks (wie A2), d50 per Bisektion der L-fachen C-Map (reproduziert dks d100/d200 auf 1e-9)
TAT = {50: dict(neg=0.43052294850240236), 100: dict(neg=0.5704395323991776)}
for _d in TAT: TAT[_d]['scale'] = math.sqrt(2 / (1 + TAT[_d]['neg'] ** 2))
SEEDS = list(range(42010, 42025))           # dieselben 15 Bestaetigungs-Seeds wie 3a-3
NCAL = 1000                                  # Kalibrierung: erste 1000 Trainingsbilder (nicht Test, nicht Val)
SAVE_SEED = -1                               # Gewichte nur fuer diesen Seed speichern (Platz)
# Quantisierungsstufen: (Name, Gewichte, Aktivierungen); 'f8' = FP8 E4M3 simuliert
QUANT = [('w8a8', 'i8', 'i8', 'max'), ('w8a4', 'i8', 'i4', 'max'), ('w4a8', 'i4', 'i8', 'max'),
         ('w8a4_p999', 'i8', 'i4', 'p999'), ('w8a4_p9999', 'i8', 'i4', 'p9999'), ('w8a4_mse', 'i8', 'i4', 'mse4'), ('w8a8_mse', 'i8', 'i8', 'mse8')]
CALS4 = ['max', 'p999', 'p9999', 'mse4']   # Kandidaten fuer w8a4_best (Wahl je Lauf ueber den Fehler auf den Kalibrierbildern)
BACKEND = 'kernel'

# --- Aktivierungen ---------------------------------------------------------------
from precap._presets import PRESETS
TAT_DEPTH = [100]
def make_act(n):
    if n == 'tat': p = TAT[TAT_DEPTH[0]]; return lambda x: p['scale'] * mx.where(x > 0, x, p['neg'] * x)
    if n == 'selu': return nn.selu
    if n == 'elu': return nn.elu
    if BACKEND == 'kernel':
        from precap.mlx import make_precap; return make_precap(**PRESETS[n])
    from precap.mlx import precap_ref_mlx
    p = PRESETS[n]; return lambda x: precap_ref_mlx(x, **p)

def np_act(n, x):  # float64-Referenz fuer die Init-Gains
    if n == 'selu': l, a = 1.0507009873554805, 1.6732632423543772; return l * np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
    if n == 'elu': return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
    p = PRESETS[n]; w = p['s'] * np.tanh(p['b'] * x / p['s']); ec = math.exp(p['c'])
    ce = np.where(w <= p['c'], np.exp(np.minimum(w, p['c'])) - 1, ec - 1 + ec * (w - p['c']))
    return p['lam'] * np.where(x > 0, ce, p['alpha'] * ce)
def gain(n):  # wie 3a-3: g = 1/sqrt(E[f(Z)^2]), Gauss-Hermite 150 Knoten
    X, W = np.polynomial.hermite_e.hermegauss(150); W = W / W.sum(); y = np_act(n, X); return float(1 / np.sqrt(np.sum(W * y * y)))
GAINS = {n: gain(n) for n in REF_ACTS}

# --- Daten (identisch zu 3a-3) ------------------------------------------------------
def idx(p):
    with gzip.open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xff; sh = struct.unpack('>' + 'I' * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(sh)
_C = {}
def load(ds):
    if ds in _C: return _C[ds]
    if ds.startswith('synth'):  # nur fuer Tests ohne echte Daten
        r = np.random.default_rng(0); X = r.normal(size=(12000, 784)).astype(np.float32); Y = (X[:, :10].argmax(1)).astype(np.int64)
        xtr, ytr, xv, yv, xte, yte = X[:6000], Y[:6000], X[6000:9000], Y[6000:9000], X[9000:], Y[9000:]
    else:
        d = DATA / ds
        xtr = idx(d / 'train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255; ytr = idx(d / 'train-labels-idx1-ubyte.gz').astype(np.int64)
        xte = idx(d / 't10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255; yte = idx(d / 't10k-labels-idx1-ubyte.gz').astype(np.int64)
        xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
    m, s = xtr.mean(), xtr.std(); T = lambda a: ((a - m) / s).astype(np.float32)
    _C[ds] = (T(xtr), ytr, T(xv), yv, T(xte), yte); return _C[ds]
def pex(a): return (a / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1e-12)).astype(np.float32)   # wie A2
def data_for(ds, act):
    xtr, ytr, xv, yv, xte, yte = load(ds)
    if act != 'tat': return xtr, ytr, xv, yv, xte, yte
    k = ds + '_pex'
    if k not in _C: _C[k] = (pex(xtr), ytr, pex(xv), yv, pex(xte), yte)
    return _C[k]

# --- Netz ----------------------------------------------------------------------------
def suo(r, b, a):  # skaliert-gleichverteilt-orthogonal wie dks (identisch zu A2)
    M = r.normal(size=(max(a, b), min(a, b))); Q, R_ = np.linalg.qr(M); Q = Q * np.sign(np.diag(R_))
    W = Q if b >= a else Q.T
    return (W * max(math.sqrt(b / a), 1.0)).astype(np.float32)
class Net(nn.Module):
    def __init__(self, depth, n, seed):
        super().__init__()
        r = np.random.default_rng(seed); dims = [784] + [WIDTH] * depth
        self.hid = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]; self.out = nn.Linear(WIDTH, 10)
        for l, (a, b) in zip(self.hid, zip(dims[:-1], dims[1:])):
            if n == 'tat': l.weight = mx.array(suo(r, b, a))
            else: l.weight = mx.array(r.normal(0, GAINS[n] / math.sqrt(a), size=(b, a)).astype(np.float32))
            l.bias = mx.zeros((b,))
        if n == 'tat': self.out.weight = mx.array(suo(r, 10, WIDTH))
        else: self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(WIDTH), size=(10, WIDTH)).astype(np.float32))
        self.out.bias = mx.zeros((10,))
        TAT_DEPTH[0] = depth; self.act = make_act(n)
    def __call__(self, x):
        for l in self.hid: x = self.act(l(x))
        return self.out(x)

def loss_fn(m, x, y): return nn.losses.cross_entropy(m(x), y, reduction='mean')

def evaluate(fwd, x, y):
    lo = 0.0; ac = 0
    for i in range(0, len(x), 2500):
        o = fwd(mx.array(x[i:i + 2500])); yy = mx.array(y[i:i + 2500])
        lo += float(nn.losses.cross_entropy(o, yy, reduction='sum')); ac += int((mx.argmax(o, 1) == yy).sum())
    return lo / len(x), ac / len(x)

# --- Quantisierung (simuliert, fp32-Rechnung mit gerundeten Werten) ------------------
def q_int(a, bits, scale):
    qm = 2 ** (bits - 1) - 1; return mx.clip(mx.round(a / scale), -qm, qm) * scale
def q_fp8(a, scale):  # E4M3: 3 Mantissenbits, kleinster Normal-Exponent -6, Max 448
    # Exponent und Stufe exakt ueber Bits (Metal-pow/log2 sind fuer Zweierpotenzen nicht exakt)
    v = a / scale; av = mx.maximum(mx.abs(v), 2.0 ** -9).astype(mx.float32)
    e = mx.maximum(((av.view(mx.int32) >> 23) & 255) - 127, -6)
    step = ((e - 3 + 127) << 23).view(mx.float32); q = mx.round(v / step) * step; return mx.clip(q, -448.0, 448.0) * scale
def qw(W, kind):  # Gewichte: pro Ausgangskanal symmetrisch
    am = mx.maximum(mx.max(mx.abs(W), axis=1, keepdims=True), 1e-12)
    if kind == 'i8': return q_int(W, 8, am / 127)
    if kind == 'i4': return q_int(W, 4, am / 7)
    if kind == 'f8': return q_fp8(W, am / 448)
    raise ValueError(kind)
def qa(a, kind, rng):  # Aktivierungen: pro Tensor symmetrisch, rng = kalibrierter Betrag
    if kind == 'i8': return q_int(a, 8, rng / 127)
    if kind == 'i4': return q_int(a, 4, rng / 7)
    if kind == 'f8': return q_fp8(a, rng / 448)
    raise ValueError(kind)
def mse_clip(a, bits):  # Clip-Wert mit minimalem Quantisierungsfehler (MSE), Gitter 1..100 % des Maximums (vektorisiert, 4 Teilbloecke)
    qm = 2 ** (bits - 1) - 1; x = mx.array(a.astype(np.float32)); amax = float(mx.max(mx.abs(x)))
    if amax == 0: return 1e-12
    E = []
    for ks in (range(1, 26), range(26, 51), range(51, 76), range(76, 101)):
        sc = mx.array([amax * k / 100 / qm for k in ks], dtype=mx.float32)[:, None]
        E.append(mx.mean((mx.clip(mx.round(x[None, :] / sc), -qm, qm) * sc - x[None, :]) ** 2, axis=1))
    E = np.array(mx.concatenate(E)); k = int(np.argmin(E)) + 1; return amax * k / 100
def calib(m, xc):
    # Eingaenge jeder Linearschicht (fp32) auf den Kalibrierbildern: max, Perzentile, MSE-optimaler Clip fuer 4 und 8 Bit
    R = {k: [] for k in ['max', 'p999', 'p9999', 'mse4', 'mse8']}; h = mx.array(xc)
    for l in m.hid + [m.out]:
        a = np.array(h, dtype=np.float64).ravel(); aa = np.abs(a)
        R['max'].append(float(aa.max())); R['p999'].append(float(np.quantile(aa, 0.999))); R['p9999'].append(float(np.quantile(aa, 0.9999)))
        R['mse4'].append(mse_clip(a, 4)); R['mse8'].append(mse_clip(a, 8))
        if l is not m.out: h = m.act(l(h))
    return R
def qforward(m, wk, ak, rngs):
    Ws = [(qw(l.weight, wk), l.bias) for l in m.hid + [m.out]]
    def f(x):
        h = x
        for i, (W, b) in enumerate(Ws):
            h = qa(h, ak, rngs[i]) @ W.T + b
            if i < len(Ws) - 1: h = m.act(h)
        return h
    return f

# --- Ablauf --------------------------------------------------------------------------
COLS = ['run_id', 'dataset', 'depth', 'act', 'lr', 'seed', 'status', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32'] + \
       [f'test_err_{q[0]}' for q in QUANT] + ['test_err_w8a4_best', 'w8a4_best_cal', 'cal_err_w8a4_best'] + ['act_range_max_median', 'act_range_max_last', 'act_range_p9999_median', 'act_clip_mse4_median', 'sec_train', 'ms_per_step', 'backend', 'timestamp']
def rid(c): return f"{c['dataset']}_d{c['depth']}_{c['act']}_s{c['seed']}" + (f"_lr{c['lr']:g}" if STAGE == 'tune' else '')
TUNEDIR = ROOT / 'tune'
def tune_configs():
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in TUNE_SEEDS:
                for lr in LR_GRID: yield dict(dataset=ds, depth=depth, act='tat', lr=lr, seed=seed)
def select_lr():  # wie A2: hoechste mediane val_acc am letzten Schritt; divergiert -> 0.1; Gleichstand -> kleinere LR
    rows = []
    for f in sorted((TUNEDIR / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    acc = {}
    for r in rows:
        a_ = (1 - float(r['val_err_fp32'])) if r['status'] == 'ok' else 0.1
        acc.setdefault((r['dataset'], int(r['depth']), r['act'], float(r['lr'])), []).append(a_)
    out = []
    for depth in DEPTHS:
        for ds in DATASETS:
            cand = [(float(np.median(acc[(ds, depth, 'tat', lr)])), -lr, lr) for lr in LR_GRID if (ds, depth, 'tat', lr) in acc]
            if not cand: continue
            best = max(cand); out.append(dict(dataset=ds, depth=depth, act='tat', lr_sel=best[2], med_val_acc=best[0],
                                              at_edge=best[2] in (LR_GRID[0], LR_GRID[-1]), n_seeds=len(acc[(ds, depth, 'tat', best[2])])))
    return out
def lrsel():
    sel = {}
    for r in csv.DictReader(open(ROOT / 'lr_from_p3a3.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
    for r in select_lr(): sel[(r['dataset'], r['depth'], 'tat')] = r['lr_sel']
    return sel
def configs():
    sel = lrsel(); L = lambda ds, d, a: sel[(ds if not ds.startswith('synth') else 'mnist', d, a)]
    for ds, d, a, s in REPRO:
        if ds in DATASETS and d in DEPTHS and s in SEEDS: yield dict(dataset=ds, depth=d, act=a, lr=L(ds, d, a), seed=s)
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in SEEDS:
                for act in ACTS: yield dict(dataset=ds, depth=depth, act=act, lr=L(ds, depth, act), seed=seed)

def execute(c, wid):
    t0 = time.time(); xtr, ytr, xv, yv, xte, yte = data_for(c['dataset'], c['act'])
    mx.random.seed(c['seed']); m = Net(c['depth'], c['act'], c['seed']); mx.eval(m.parameters())
    opt = optim.Adam(learning_rate=c['lr'], betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    vg = nn.value_and_grad(m, loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(c['seed'] + 777); perm = r.permutation(len(xtr)); pos = 0; run = []; status = 'ok'; ts = None
    for s in range(1, STEPS + 1):
        if pos + BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + BATCH]; pos += BATCH
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s == 20: ts = time.time()
        if s % 43 == 0 or s == STEPS:
            lv = float(loss); run.append(lv)
            if not math.isfinite(lv): status = 'nonfinite_loss'; break
    sec = time.time() - t0; msps = (time.time() - ts) / max(STEPS - 20, 1) * 1000 if ts and status == 'ok' else float('nan')
    out = dict(run_id=rid(c), **c, status=status, train_loss_last50=float(np.mean(run[-2:])) if run else float('nan'), sec_train=round(sec, 1), ms_per_step=round(msps, 3), backend=BACKEND)
    if status == 'ok' and STAGE == 'tune':
        _, va = evaluate(lambda x: m(x), xv, yv); out['val_err_fp32'] = 1 - va
    elif status == 'ok':
        fwd = lambda x: m(x)
        _, va = evaluate(fwd, xv, yv); _, ta = evaluate(fwd, xte, yte); out['val_err_fp32'] = 1 - va; out['test_err_fp32'] = 1 - ta
        R = calib(m, xtr[:NCAL]); out['act_range_max_median'] = float(np.median(R['max'][1:])); out['act_range_max_last'] = R['max'][-1]
        out['act_range_p9999_median'] = float(np.median(R['p9999'][1:])); out['act_clip_mse4_median'] = float(np.median(R['mse4'][1:]))
        for name, wk, ak, cal in QUANT:
            f = qforward(m, wk, ak, R[cal]); _, a = evaluate(f, xte, yte); out[f'test_err_{name}'] = 1 - a
        # w8a4_best: je Lauf die Kalibrierung mit dem kleinsten Fehler auf den 1000 Kalibrierbildern (Trainingsdaten), Gleichstand -> Reihenfolge CALS4
        ce = {}
        for cal in CALS4:
            _, a = evaluate(qforward(m, 'i8', 'i4', R[cal]), xtr[:NCAL], ytr[:NCAL]); ce[cal] = 1 - a
        bc = min(CALS4, key=lambda k: (ce[k], CALS4.index(k))); out['w8a4_best_cal'] = bc; out['cal_err_w8a4_best'] = ce[bc]
        _, a = evaluate(qforward(m, 'i8', 'i4', R[bc]), xte, yte); out['test_err_w8a4_best'] = 1 - a
        if c['seed'] == SAVE_SEED:
            (OUTDIR / 'weights').mkdir(parents=True, exist_ok=True); m.save_weights(str(OUTDIR / 'weights' / f"{rid(c)}.safetensors"))
    out['timestamp'] = datetime.now().isoformat(timespec='seconds')
    p = OUTDIR / 'workers' / f'w{wid}.csv'; p.parent.mkdir(parents=True, exist_ok=True); new = not p.exists()
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(out)
    return out

def finished():
    d = set()
    for f in (OUTDIR / 'workers').glob('w*.csv'):
        for r in csv.DictReader(open(f)): d.add(r['run_id'])
    return d

def selfcheck():
    out = {'gains': GAINS, 'backend': BACKEND, 'device': str(mx.default_device()), 'n_tune': len(list(tune_configs())),
           'n_confirm': len(REPRO) + len(DEPTHS) * len(DATASETS) * len(SEEDS) * len(ACTS)}
    def cmap(c, a): return ((1 - a) ** 2 * (np.sqrt(1 - c ** 2) + (np.pi - np.arccos(c)) * c) / np.pi + 2 * a * c) / (1 + a ** 2)
    X, W = np.polynomial.hermite_e.hermegauss(150); W = W / W.sum()
    for d, p in TAT.items():
        c = 0.0
        for _ in range(d): c = cmap(c, p['neg'])
        y = p['scale'] * np.where(X > 0, X, p['neg'] * X)
        TAT_DEPTH[0] = d; z = mx.array(np.linspace(-8, 8, 2001).astype(np.float32)); zn = np.array(z, dtype=np.float64)
        out[f'tat_d{d}'] = dict(Cf0=c, q=float(np.sum(W * y * y)), mlx_vs_numpy=float(np.abs(np.array(make_act('tat')(z)) - p['scale'] * np.where(zn > 0, zn, p['neg'] * zn)).max()))
    z = mx.array(np.linspace(-8, 8, 20001).astype(np.float32))
    for n in ['spreccap_cs', 'spreccap_e03']:
        out[n + '_max_abs_vs_numpy'] = float(np.abs(np.array(make_act(n)(z)) - np_act(n, np.array(z, dtype=np.float64))).max())
    out['selu_max_abs_vs_numpy'] = float(np.abs(np.array(nn.selu(z)) - np_act('selu', np.array(z, dtype=np.float64))).max())
    a = mx.array(np.linspace(-3, 3, 13).astype(np.float32))
    out['q_int8_example'] = np.array(q_int(a, 8, 3 / 127)).round(4).tolist(); out['q_fp8_448'] = float(q_fp8(mx.array([448.0, 500.0, 0.0]), 1.0)[1])
    out['q_fp8_grid_ok'] = bool(np.allclose(np.array(q_fp8(mx.array([1.0, 1.0625, 1.125, 0.015625 * 1.1]), 1.0)), [1.0, 1.0, 1.125, 0.017578125]))
    return out

def main():
    global STEPS, OUTDIR, SEEDS, DATASETS, DEPTHS, BACKEND, STAGE, TUNEDIR, TUNE_SEEDS, LR_GRID, REPRO
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--probe', action='store_true')
    p.add_argument('--backend', default='kernel', choices=['kernel', 'ref']); p.add_argument('--synth', action='store_true'); p.add_argument('--limit', type=int, default=0); a = p.parse_args()
    BACKEND = a.backend
    if a.synth: DATASETS = ['synth']
    if a.smoke:
        STEPS = 40; OUTDIR = ROOT / 'smoke'; SEEDS = [42010]; TUNE_SEEDS = [42000]; LR_GRID = [2e-5, 5e-5]; DEPTHS = [100]; DATASETS = DATASETS[:1]
        REPRO = [(DATASETS[0], 100, 'spreccap_cs', 42010)]
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    if a.probe:
        # Zeit pro Trainingsschritt: Kernel gegen Referenz gegen SELU, je Tiefe
        global ACTS
        res = {}
        for d in DEPTHS:
            for bk, n in [('kernel', 'spreccap_cs'), ('ref', 'spreccap_cs'), ('kernel', 'selu')]:
                BACKEND = bk; STEPS = 120; OUTDIR = ROOT / 'probe'
                o = execute(dict(dataset=DATASETS[0], depth=d, act=n, lr=5e-5, seed=1), 0); res[f'd{d}_{n}_{bk}_ms_per_step'] = o['ms_per_step']
        print(json.dumps(res)); return
    base = OUTDIR
    STAGE = 'tune'; OUTDIR = base / 'tune'; TUNEDIR = OUTDIR; skip = finished(); T = list(tune_configs())
    for i, c in enumerate(T):
        if i % a.workers != a.worker or rid(c) in skip: continue
        o = execute(c, a.worker); print('tune', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)
    while len(finished()) < len(T): time.sleep(20)
    sel = select_lr()
    if a.worker == 0:
        with open(base / 'lr_tuned.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sel[0].keys())); w.writeheader(); w.writerows(sel)
    print('LR-Wahl', json.dumps(sel), flush=True)
    STAGE = 'confirm'; OUTDIR = base; skip = finished(); nrun = 0
    for i, c in enumerate(configs()):
        if a.limit and nrun >= a.limit: break
        if i % a.workers != a.worker: continue
        if rid(c) in skip: continue
        nrun += 1; o = execute(c, a.worker); print('done', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
