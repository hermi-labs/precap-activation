# Studie A3 (privat): kL2 im TAT-Rezept (kL2r) gegen TAT aus A2. Einzige Aenderung gegenueber kL2 in A/A2: SUO-Init (skaliert-orthogonal,
# Bias 0) und Normierung je Beispiel wie bei TAT; LR neu getunt (gleiches Raster, gleiche Tuning-Seeds wie TAT in A2).
# Aktivierung kL2 unveraendert (Konstanten wie A). TAT-Werte stammen aus A2; zwei TAT-Laeufe werden zur Kontrolle neu gerechnet.
# Code abgeleitet aus a2_run.py; Datenpfad, Netz, Training, Auswertung je Lauf unveraendert.
import argparse, csv, gzip, json, math, os, struct, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data'))

DATASETS = ['mnist', 'fashion']; DEPTHS = [100, 200]; WIDTH = 256; BATCH = 128; NVAL = 5000
SHORT, LONG = 1720, 6880
ACTS = ['kL2r']                                               # neu; 'tat' nur fuer die Reproduktion
LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]               # wie 3a-3 / A
TUNE_SEEDS = [42000, 42001, 42002, 42003]
SEEDS_SHORT = list(range(42010, 42025))                       # 15 Seeds, wie A
SEEDS_LONG = list(range(42010, 42020))                        # 10 Seeds
REPRO = [('mnist', 100, 'tat', 42010), ('mnist', 100, 'tat', 42011)]   # Kontrolle gegen A2
S_, C_ = 1.5, 1.0
KL2 = {100: dict(lam=2.649586999, alpha=1.461343209, b=0.318766948), 200: dict(lam=4.170485099, alpha=1.296823062, b=0.212387949)}  # wie A
# TAT Leaky-ReLU-Parameter aus dks.base.activation_transform._get_activations_params(['leaky_relu'], method='TAT', subnet_max_func=L-fache Komposition)
TAT = {100: dict(neg=0.5704395323991776, scale=1.2284042441106242), 200: dict(neg=0.6795996427536011, scale=1.169668377971502)}
ETA = 0.9

# --- Aktivierungen ---------------------------------------------------------------
def pc_mlx(x, lam, alpha, b):
    w = S_ * mx.tanh(b * x / S_)
    ec = math.exp(C_); ce = mx.where(w <= C_, mx.exp(mx.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    return lam * mx.where(x > 0, ce, alpha * ce)
def make_act(n, depth):
    if n == 'selu': return nn.selu
    if n in ('kL2', 'kL2r'): p = KL2[depth]; return lambda x: pc_mlx(x, **p)
    if n == 'tat': p = TAT[depth]; return lambda x: p['scale'] * mx.where(x > 0, x, p['neg'] * x)
    raise ValueError(n)
def np_act(n, x, depth):
    if n == 'selu': l, a = 1.0507009873554805, 1.6732632423543772; return l * np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
    if n == 'tat': p = TAT[depth]; return p['scale'] * np.where(x > 0, x, p['neg'] * x)
    p = KL2[depth]; w = S_ * np.tanh(p['b'] * x / S_)   # kL2 und kL2r
    ec = math.exp(C_); ce = np.where(w <= C_, np.exp(np.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    return p['lam'] * np.where(x > 0, ce, p['alpha'] * ce)
def gain(n, depth):  # wie A: g = 1/sqrt(E[f(Z)^2]), Gauss-Hermite 150 Knoten (nur kL2/SELU)
    X, W = np.polynomial.hermite_e.hermegauss(150); W = W / W.sum(); y = np_act(n, X, depth); return float(1 / np.sqrt(np.sum(W * y * y)))
GAINS = {(n, d): gain(n, d) for n in ('kL2', 'selu') for d in DEPTHS}
def lrelu_cmap(c, a):  # wie dks.base.activation_transform._leaky_relu_cmap
    return ((1 - a) ** 2 * (np.sqrt(1 - c ** 2) + (np.pi - np.arccos(c)) * c) / np.pi + 2 * a * c) / (1 + a ** 2)

# --- Daten (wie A; fuer TAT zusaetzlich Normierung je Beispiel auf mittleres Quadrat 1, wie dks per-example-Normierung) ---
def idx(p):
    with gzip.open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xff; sh = struct.unpack('>' + 'I' * nd, f.read(4 * nd))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(sh)
_C = {}
def load(ds):
    if ds in _C: return _C[ds]
    if ds.startswith('synth'):
        r = np.random.default_rng(0); X = r.normal(size=(12000, 784)).astype(np.float32); Y = (X[:, :10].argmax(1)).astype(np.int64)
        xtr, ytr, xv, yv, xte, yte = X[:6000], Y[:6000], X[6000:9000], Y[6000:9000], X[9000:], Y[9000:]
    else:
        d = DATA / ds
        xtr = idx(d / 'train-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255; ytr = idx(d / 'train-labels-idx1-ubyte.gz').astype(np.int64)
        xte = idx(d / 't10k-images-idx3-ubyte.gz').reshape(-1, 784).astype(np.float32) / 255; yte = idx(d / 't10k-labels-idx1-ubyte.gz').astype(np.int64)
        xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
    m, s = xtr.mean(), xtr.std(); T = lambda a: ((a - m) / s).astype(np.float32)
    _C[ds] = (T(xtr), ytr, T(xv), yv, T(xte), yte); return _C[ds]
def pex(a): return (a / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
def data_for(ds, act):
    xtr, ytr, xv, yv, xte, yte = load(ds)
    if act not in ('tat', 'kL2r'): return xtr, ytr, xv, yv, xte, yte
    k = ds + '_pex'
    if k not in _C: _C[k] = (pex(xtr), ytr, pex(xv), yv, pex(xte), yte)
    return _C[k]

# --- Netz ----------------------------------------------------------------------------
def suo(r, b, a):  # skaliert-gleichverteilt-orthogonal wie dks: orthogonal, mal max(sqrt(out/in), 1)
    M = r.normal(size=(max(a, b), min(a, b))); Q, R = np.linalg.qr(M); Q = Q * np.sign(np.diag(R))
    W = Q if b >= a else Q.T                                    # Form (b, a)
    return (W * max(math.sqrt(b / a), 1.0)).astype(np.float32)
class Net(nn.Module):
    def __init__(self, depth, n, seed):
        super().__init__()
        r = np.random.default_rng(seed); dims = [784] + [WIDTH] * depth
        self.hid = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]; self.out = nn.Linear(WIDTH, 10)
        for l, (a, b) in zip(self.hid, zip(dims[:-1], dims[1:])):
            if n in ('tat', 'kL2r'): l.weight = mx.array(suo(r, b, a))
            else: g = GAINS[(n, depth)]; l.weight = mx.array(r.normal(0, g / math.sqrt(a), size=(b, a)).astype(np.float32))
            l.bias = mx.zeros((b,))
        if n in ('tat', 'kL2r'): self.out.weight = mx.array(suo(r, 10, WIDTH))
        else: self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(WIDTH), size=(10, WIDTH)).astype(np.float32))
        self.out.bias = mx.zeros((10,))
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

# --- Ablauf --------------------------------------------------------------------------
COLS = ['run_id', 'dataset', 'depth', 'act', 'steps', 'lr', 'seed', 'status', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32',
        'val_err_at_1720', 'val_err_at_3440', 'act_range_max_median', 'grad_log10_first_last', 'sec_train', 'ms_per_step', 'timestamp']
STAGE = 'confirm'; TUNEDIR = ROOT / 'tune'
def rid(c): return f"{c['dataset']}_d{c['depth']}_{c['act']}_T{c['steps']}_s{c['seed']}" + (f"_lr{c['lr']:g}" if STAGE == 'tune' else '')
def tune_configs():
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in TUNE_SEEDS:
                for lr in LR_GRID: yield dict(dataset=ds, depth=depth, act='kL2r', steps=SHORT, lr=lr, seed=seed)
def select_lr():  # wie A: hoechste mediane val_acc am letzten Schritt; divergiert -> 0.1; Gleichstand -> kleinere LR
    rows = []
    for f in sorted((TUNEDIR / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    acc = {}
    for r in rows:
        a = (1 - float(r['val_err_fp32'])) if r['status'] == 'ok' else 0.1
        acc.setdefault((r['dataset'], int(r['depth']), r['act'], float(r['lr'])), []).append(a)
    out = []
    for depth in DEPTHS:
        for ds in DATASETS:
            cand = [(float(np.median(acc[(ds, depth, 'kL2r', lr)])), -lr, lr) for lr in LR_GRID if (ds, depth, 'kL2r', lr) in acc]
            if not cand: continue
            best = max(cand); out.append(dict(dataset=ds, depth=depth, act='kL2r', lr_sel=best[2], med_val_acc=best[0],
                                              at_edge=best[2] in (LR_GRID[0], LR_GRID[-1]), n_seeds=len(acc[(ds, depth, 'kL2r', best[2])])))
    return out
def lrsel():
    sel = {}
    for r in csv.DictReader(open(ROOT / 'lr_tat_from_a2.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
    for r in select_lr(): sel[(r['dataset'], r['depth'], 'kL2r')] = r['lr_sel']
    return sel
def configs():
    sel = lrsel(); L = lambda ds, d, a: sel[(ds, d, a)] if (ds, d, a) in sel else sel[('mnist', d, a)]
    for ds, d, a, s in REPRO:                                   # Reproduktion von A (kurzes Budget)
        if ds in DATASETS and d in DEPTHS and s in SEEDS_SHORT: yield dict(dataset=ds, depth=d, act=a, steps=SHORT, lr=L(ds, d, a), seed=s)
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in SEEDS_SHORT: yield dict(dataset=ds, depth=depth, act='kL2r', steps=SHORT, lr=L(ds, depth, 'kL2r'), seed=seed)
    for depth in DEPTHS:
        for ds in DATASETS:
            for seed in SEEDS_LONG:
                for act in ACTS: yield dict(dataset=ds, depth=depth, act=act, steps=LONG, lr=L(ds, depth, act), seed=seed)

def execute(c, wid):
    t0 = time.time(); xtr, ytr, xv, yv, xte, yte = data_for(c['dataset'], c['act'])
    mx.random.seed(c['seed']); m = Net(c['depth'], c['act'], c['seed']); mx.eval(m.parameters())
    r0 = np.random.default_rng(c['seed'] + 777); b0 = r0.permutation(len(xtr))[:BATCH]
    _, g0 = nn.value_and_grad(m, loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    n1 = float(mx.sqrt(mx.sum(g0['hid'][0]['weight'] ** 2))); nL = float(mx.sqrt(mx.sum(g0['hid'][-1]['weight'] ** 2)))
    glog = math.log10(n1 / nL) if n1 > 0 and nL > 0 and math.isfinite(n1 / nL) else float('nan')
    opt = optim.Adam(learning_rate=c['lr'], betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    vg = nn.value_and_grad(m, loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    STEPS = c['steps']; r = np.random.default_rng(c['seed'] + 777); perm = r.permutation(len(xtr)); pos = 0; run = []; status = 'ok'; ts = None; mid = {}
    for s in range(1, STEPS + 1):
        if pos + BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + BATCH]; pos += BATCH
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s == 20: ts = time.time()
        if s % 43 == 0 or s == STEPS:
            lv = float(loss); run.append(lv)
            if not math.isfinite(lv): status = 'nonfinite_loss'; break
        if STAGE == 'confirm' and s in (1720, 3440) and s < STEPS:
            t1 = time.time(); _, va = evaluate(lambda x: m(x), xv, yv); mid[s] = 1 - va
            if ts: ts += time.time() - t1                          # Zwischenauswertung nicht in ms/Schritt zaehlen
    sec = time.time() - t0; msps = (time.time() - ts) / max(STEPS - 20, 1) * 1000 if ts and status == 'ok' else float('nan')
    out = dict(run_id=rid(c), **c, status=status, train_loss_last50=float(np.mean(run[-2:])) if run else float('nan'), sec_train=round(sec, 1),
               ms_per_step=round(msps, 3), grad_log10_first_last=glog, val_err_at_1720=mid.get(1720, ''), val_err_at_3440=mid.get(3440, ''))
    if status == 'ok':
        fwd = lambda x: m(x); _, va = evaluate(fwd, xv, yv); out['val_err_fp32'] = 1 - va
        if STAGE == 'confirm':
            _, ta = evaluate(fwd, xte, yte); out['test_err_fp32'] = 1 - ta
            h = mx.array(xtr[:1000]); mxs = []
            for l in m.hid: h = m.act(l(h)); mxs.append(float(mx.max(mx.abs(h))))
            out['act_range_max_median'] = float(np.median(mxs))
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
    out = {'device': str(mx.default_device()), 'n_tune': len(list(tune_configs()))}
    z = mx.array(np.linspace(-8, 8, 20001).astype(np.float32)); X = np.linspace(-12, 12, 240001); W = np.exp(-X * X / 2); W = W / W.sum()
    for n in ACTS + ['tat']:
        for d in DEPTHS:
            y = np_act(n, X, d); h = 1e-6; dy = (np_act(n, X + h, d) - np_act(n, X - h, d)) / (2 * h)
            out[f'{n}_d{d}'] = dict(mean=round(float(np.sum(W * y)), 6), q=round(float(np.sum(W * y * y)), 6), chi1=round(float(np.sum(W * dy * dy)), 6),
                                    mlx_vs_numpy=float(np.abs(np.array(make_act(n, d)(z)) - np_act(n, np.array(z, dtype=np.float64), d)).max()))
    for d in DEPTHS:  # TAT-Ziel: L-fache Komposition der C-Map bei 0 = eta
        c = 0.0
        for _ in range(d): c = lrelu_cmap(c, TAT[d]['neg'])
        out[f'tat_d{d}_Cf0'] = c; out[f'tat_d{d}_scale_ok'] = abs(TAT[d]['scale'] - math.sqrt(2 / (1 + TAT[d]['neg'] ** 2))) < 1e-12
        # unabhaengige Kontrolle der lokalen C-Map bei c=0: E[phi]^2 / E[phi^2] geschlossen, E[LReLU]=(1-a)/sqrt(2pi), E[LReLU^2]=(1+a^2)/2
        a_ = TAT[d]['neg']; out[f'tat_d{d}_cmap0_closed_vs_formula'] = [float(((1 - a_) / math.sqrt(2 * math.pi)) ** 2 / ((1 + a_ ** 2) / 2)), float(lrelu_cmap(0.0, a_))]
    r = np.random.default_rng(0); Wt = suo(r, 256, 256); W1 = suo(r, 256, 784); Wo = suo(r, 10, 256)
    out['suo_orth_err'] = [float(np.abs(Wt @ Wt.T - np.eye(256)).max()), float(np.abs(W1 @ W1.T - np.eye(256)).max()), float(np.abs(Wo @ Wo.T - np.eye(10)).max())]
    out['n_confirm'] = None
    return out

def main():
    global OUTDIR, SEEDS_SHORT, SEEDS_LONG, DATASETS, DEPTHS, STAGE, TUNE_SEEDS, LR_GRID, TUNEDIR, SHORT, LONG, REPRO
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--synth', action='store_true'); a = p.parse_args()
    if a.synth: DATASETS = ['synth']
    if a.selfcheck:
        o = selfcheck(); o['n_confirm'] = sum(1 for _ in configs_count()); print(json.dumps(o, indent=1)); return
    if a.smoke:
        OUTDIR = ROOT / 'smoke'; SHORT, LONG = 40, 80; TUNE_SEEDS = [42000]; LR_GRID = [2e-5, 5e-5]; SEEDS_SHORT = [42010]; SEEDS_LONG = [42010]
        DEPTHS = [200]; DATASETS = DATASETS[:1]; REPRO = [('mnist', 200, 'tat', 42010)] if not a.synth else [('synth', 200, 'tat', 42010)]
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
    STAGE = 'confirm'; OUTDIR = base; skip = finished()
    C = list(configs()); C = [c for _, c in sorted(enumerate(C), key=lambda t: (t[1]['steps'], t[0]))]  # erst kurzes Budget
    for i, c in enumerate(C):
        if i % a.workers != a.worker or rid(c) in skip: continue
        o = execute(c, a.worker); print('done', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)

def configs_count():  # Anzahl Bestaetigungslaeufe ohne LR-Auswahl
    n = len([r for r in REPRO]) + len(DEPTHS) * len(DATASETS) * len(SEEDS_SHORT) + len(DEPTHS) * len(DATASETS) * len(SEEDS_LONG) * len(ACTS)
    return range(n)

if __name__ == '__main__': main()
