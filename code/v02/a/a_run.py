# Idee A (privat): Tiefen-PreCap. chi1 = 1 + k/L (L = verdeckte Schichten), k in {1, 2}; Form wie SPreCap-cs (s=1.5, c=1, tanh).
# Konstanten (lam, alpha, b) je (k, L) exakt geloest: E[f]=0, E[f^2]=1, E[f'^2]=1+k/L (a_constants.py). Init Gauss wie 3a-3 (eine Aenderung: chi1).
# Training wie 3a-3/C1 (MLX). Stufe 1: LR-Tuning (kL1/kL2 alle Tiefen, cs/e03/SELU nur Tiefe 200). Stufe 2: Bestaetigung mit 15 Seeds.
import argparse, csv, gzip, json, math, os, struct, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data'))
sys.path.insert(0, str(Path(os.environ.get('PRECAP_SRC', ROOT / 'src'))))

DATASETS = ['mnist', 'fashion']; DEPTHS = [50, 100, 200]; WIDTH = 256; BATCH = 128; STEPS = 1720; NVAL = 5000
ACTS = ['spreccap_cs', 'spreccap_e03', 'kL1', 'kL2', 'selu']  # Bestaetigung
def tune_acts(depth): return ['kL1', 'kL2'] + (['spreccap_cs', 'spreccap_e03', 'selu'] if depth == 200 else [])  # 50/100: LR aus 3a-3
LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]               # wie 3a-3
TUNE_SEEDS = [42000, 42001, 42002, 42003]                     # wie 3a-3
S_, C_ = 1.5, 1.0
CONST = {('spreccap_cs', 0): dict(lam=1.1027025, alpha=2.053776267, b=0.71188112),       # wie Repo-Preset
         ('spreccap_e03', 0): dict(lam=1.985552225, alpha=1.608136397, b=0.4118896262),    # wie Repo-Preset
         ('kL1', 50): dict(lam=2.649586999, alpha=1.461343209, b=0.318766948),  ('kL1', 100): dict(lam=4.170485099, alpha=1.296823062, b=0.212387949),
         ('kL1', 200): dict(lam=6.296620086, alpha=1.197830367, b=0.145678357),
         ('kL2', 50): dict(lam=1.614100769, alpha=1.739743617, b=0.496159697),  ('kL2', 100): dict(lam=2.649586999, alpha=1.461343209, b=0.318766948),
         ('kL2', 200): dict(lam=4.170485099, alpha=1.296823062, b=0.212387949)}
def _chi(n, d):
    X = np.linspace(-12, 12, 240001); W = np.exp(-X * X / 2); W = W / W.sum(); h = 1e-6   # feines Gitter (Knick bei 0)
    y1 = np_act(n, X + h, d); y0 = np_act(n, X - h, d); return float(np.sum(W * ((y1 - y0) / (2 * h)) ** 2))
def par(n, depth): return CONST[(n, 0)] if (n, 0) in CONST else CONST[(n, depth)]
STAGE = 'confirm'; TUNEDIR = ROOT / 'tune'
SEEDS = list(range(42010, 42025))           # dieselben 15 Bestaetigungs-Seeds wie 3a-3
NCAL = 1000                                  # Kalibrierung: erste 1000 Trainingsbilder (nicht Test, nicht Val)
SAVE_SEED = -1                               # Gewichte nur fuer diesen Seed speichern (Platz)
# Quantisierungsstufen: (Name, Gewichte, Aktivierungen); 'f8' = FP8 E4M3 simuliert
QUANT = [('w8a8', 'i8', 'i8', 'max'), ('w8a4', 'i8', 'i4', 'max'), ('w4a8', 'i4', 'i8', 'max'),
         ('w8a4_p999', 'i8', 'i4', 'p999'), ('w8a4_p9999', 'i8', 'i4', 'p9999'), ('w8a4_mse', 'i8', 'i4', 'mse4'), ('w8a8_mse', 'i8', 'i8', 'mse8')]
CALS4 = ['max', 'p999', 'p9999', 'mse4']   # Kandidaten fuer w8a4_best (Wahl je Lauf ueber den Fehler auf den Kalibrierbildern)
BACKEND = 'ref'   # alle SPreCap-Varianten (auch cs) ueber dieselbe MLX-Funktion, damit nur der Begrenzer verschieden ist

# --- Aktivierungen ---------------------------------------------------------------
from precap._presets import PRESETS
def pc_mlx(x, lam, alpha, b):
    w = S_ * mx.tanh(b * x / S_)
    ec = math.exp(C_); ce = mx.where(w <= C_, mx.exp(mx.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    return lam * mx.where(x > 0, ce, alpha * ce)
def make_act(n, depth):
    if n == 'selu': return nn.selu
    if n == 'elu': return nn.elu
    p = par(n, depth); return lambda x: pc_mlx(x, **p)

def np_act(n, x, depth=0):  # float64-Referenz fuer die Init-Gains
    if n == 'selu': l, a = 1.0507009873554805, 1.6732632423543772; return l * np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
    if n == 'elu': return np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
    p = par(n, depth); w = S_ * np.tanh(p['b'] * x / S_)
    ec = math.exp(C_); ce = np.where(w <= C_, np.exp(np.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    return p['lam'] * np.where(x > 0, ce, p['alpha'] * ce)
def gain(n, depth):  # wie 3a-3: g = 1/sqrt(E[f(Z)^2]), Gauss-Hermite 150 Knoten
    X, W = np.polynomial.hermite_e.hermegauss(150); W = W / W.sum(); y = np_act(n, X, depth); return float(1 / np.sqrt(np.sum(W * y * y)))
GAINS = {(n, d): gain(n, d) for n in ACTS for d in [50, 100, 200]}
CHI1 = {(n, d): _chi(n, d) for n in ACTS for d in [50, 100, 200]}

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

# --- Netz ----------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, depth, n, seed):
        super().__init__()
        r = np.random.default_rng(seed); g = GAINS[(n, depth)]; dims = [784] + [WIDTH] * depth
        self.hid = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]; self.out = nn.Linear(WIDTH, 10)
        for l, (a, b) in zip(self.hid, zip(dims[:-1], dims[1:])):
            l.weight = mx.array(r.normal(0, g / math.sqrt(a), size=(b, a)).astype(np.float32)); l.bias = mx.zeros((b,))
        self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(WIDTH), size=(10, WIDTH)).astype(np.float32)); self.out.bias = mx.zeros((10,))
        self.act = make_act(n, depth)
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
       [f'test_err_{q[0]}' for q in QUANT] + ['test_err_w8a4_best', 'w8a4_best_cal', 'cal_err_w8a4_best'] + ['act_range_max_median', 'act_range_max_last', 'act_range_p9999_median', 'act_clip_mse4_median', 'chi1', 'grad_log10_first_last', 'sec_train', 'ms_per_step', 'backend', 'timestamp']
def rid(c): return f"{c['dataset']}_d{c['depth']}_{c['act']}_s{c['seed']}" + (f"_lr{c['lr']:g}" if STAGE == 'tune' else '')
def tune_configs():
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in TUNE_SEEDS:
                for act in tune_acts(depth):
                    for lr in LR_GRID: yield dict(dataset=ds, depth=depth, act=act, lr=lr, seed=seed)
def select_lr():
    # wie 3a-3: hoechste mediane val_acc (letzter Schritt) ueber die Tuning-Seeds; divergiert -> 0.1; Gleichstand -> kleinere LR
    rows = []
    for f in sorted((TUNEDIR / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    acc = {}
    for r in rows:
        a = (1 - float(r['val_err_fp32'])) if r['status'] == 'ok' else 0.1
        acc.setdefault((r['dataset'], int(r['depth']), r['act'], float(r['lr'])), []).append(a)
    out = []
    for depth in DEPTHS:
        for ds in DATASETS:
            for act in tune_acts(depth):
                cand = [(float(np.median(acc[(ds, depth, act, lr)])), -lr, lr) for lr in LR_GRID if (ds, depth, act, lr) in acc]
                if not cand: continue
                best = max(cand); out.append(dict(dataset=ds, depth=depth, act=act, lr_sel=best[2], med_val_acc=best[0],
                                                  at_edge=best[2] in (LR_GRID[0], LR_GRID[-1]), n_seeds=len(acc[(ds, depth, act, best[2])])))
    return out
def lrsel():
    sel = {}
    for r in csv.DictReader(open(ROOT / 'lr_from_p3a3.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
    for r in select_lr(): sel[('mnist' if r['dataset'].startswith('synth') else r['dataset'], r['depth'], r['act'])] = r['lr_sel']
    return sel
def configs():
    sel = lrsel()
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in SEEDS:
                for act in ACTS: yield dict(dataset=ds, depth=depth, act=act, lr=sel[(ds if not ds.startswith('synth') else 'mnist', depth, act)], seed=seed)

def execute(c, wid):
    t0 = time.time(); xtr, ytr, xv, yv, xte, yte = load(c['dataset'])
    mx.random.seed(c['seed']); m = Net(c['depth'], c['act'], c['seed']); mx.eval(m.parameters())
    # Gradientengefaelle bei der Init (vor dem ersten Schritt): log10(|dL/dW_1| / |dL/dW_L|), erster Trainings-Batch dieses Laufs
    r0 = np.random.default_rng(c['seed'] + 777); b0 = r0.permutation(len(xtr))[:BATCH]
    _, g0 = nn.value_and_grad(m, loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    n1 = float(mx.sqrt(mx.sum(g0['hid'][0]['weight'] ** 2))); nL = float(mx.sqrt(mx.sum(g0['hid'][-1]['weight'] ** 2)))
    glog = math.log10(n1 / nL) if n1 > 0 and nL > 0 and math.isfinite(n1 / nL) else float('nan')
    chi1 = CHI1.get((c['act'], c['depth']), float('nan'))
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
    out = dict(run_id=rid(c), **c, status=status, train_loss_last50=float(np.mean(run[-2:])) if run else float('nan'), sec_train=round(sec, 1), ms_per_step=round(msps, 3), backend=BACKEND, chi1=chi1, grad_log10_first_last=glog)
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
    out = {'backend': BACKEND, 'device': str(mx.default_device()), 'n_tune': len(list(tune_configs())), 'n_confirm': len(DEPTHS) * len(DATASETS) * len(SEEDS) * len(ACTS)}
    z = mx.array(np.linspace(-8, 8, 20001).astype(np.float32)); X, W = np.polynomial.hermite_e.hermegauss(200); W = W / W.sum()
    for n in ACTS:
        for d in [50, 100, 200]:
            y = np_act(n, X, d); k = f'{n}_d{d}'
            out[k] = dict(gain=round(GAINS[(n, d)], 6), chi1=round(CHI1[(n, d)], 5), mean=round(float(np.sum(W * y)), 6), var=round(float(np.sum(W * y * y)), 6),
                          mlx_vs_numpy=float(np.abs(np.array(make_act(n, d)(z)) - np_act(n, np.array(z, dtype=np.float64), d)).max()),
                          range=[float(make_act(n, d)(mx.array([-50.0]))[0]), float(make_act(n, d)(mx.array([50.0]))[0])])
    from precap.mlx import precap_ref_mlx
    for n in ['spreccap_cs', 'spreccap_e03']: out[n + '_vs_precap_ref'] = float(mx.max(mx.abs(make_act(n, 100)(z) - precap_ref_mlx(z, **PRESETS[n]))))
    a = mx.array(np.linspace(-3, 3, 13).astype(np.float32))
    out['q_int8_example'] = np.array(q_int(a, 8, 3 / 127)).round(4).tolist(); out['q_fp8_448'] = float(q_fp8(mx.array([448.0, 500.0, 0.0]), 1.0)[1])
    out['q_fp8_grid_ok'] = bool(np.allclose(np.array(q_fp8(mx.array([1.0, 1.0625, 1.125, 0.015625 * 1.1]), 1.0)), [1.0, 1.0, 1.125, 0.017578125]))
    return out

def main():
    global STEPS, OUTDIR, SEEDS, DATASETS, DEPTHS, BACKEND, STAGE, TUNE_SEEDS, LR_GRID, TUNEDIR
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--probe', action='store_true')
    p.add_argument('--backend', default='ref', choices=['ref']); p.add_argument('--synth', action='store_true'); p.add_argument('--limit', type=int, default=0); a = p.parse_args()
    BACKEND = a.backend
    if a.synth: DATASETS = ['synth']
    if a.smoke: STEPS = 40; OUTDIR = ROOT / 'smoke'
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    if a.smoke: TUNE_SEEDS = [42000]; LR_GRID = [5e-5, 1e-4]; SEEDS = [42010]; DEPTHS = [200]; DATASETS = DATASETS[:1]
    base = OUTDIR
    # Stufe 1: Tuning
    STAGE = 'tune'; OUTDIR = base / 'tune'; TUNEDIR = OUTDIR; skip = finished(); nrun = 0; T = list(tune_configs())
    for i, c in enumerate(T):
        if a.limit and nrun >= a.limit: break
        if i % a.workers != a.worker: continue
        if rid(c) in skip: continue
        nrun += 1; o = execute(c, a.worker); print('tune', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)
    if a.limit: return
    while len(finished()) < len(T): time.sleep(20)          # auf den anderen Worker warten
    sel = select_lr()
    if a.worker == 0:
        with open(base / 'lr_tuned.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sel[0].keys())); w.writeheader(); w.writerows(sel)
    print('LR-Wahl', json.dumps(sel), flush=True)
    # Stufe 2: Bestaetigung
    STAGE = 'confirm'; OUTDIR = base; skip = finished()
    for i, c in enumerate(configs()):
        if i % a.workers != a.worker: continue
        if rid(c) in skip: continue
        o = execute(c, a.worker); print('done', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
