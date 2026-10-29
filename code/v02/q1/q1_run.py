# Q1 (privat): Post-Training-Quantisierung tiefer MLPs ohne Normierung, MLX auf Apple Silicon
# Frage: Verliert SPreCap (beidseitig beschraenkt) bei int8/int4-Aktivierungen weniger Genauigkeit als SELU/ELU?
# Training wie 3a-3 (gleiche Daten, Breite, Batch, Schritte, Adam, Init, gewaehlte LRs aus 3a-3), aber in MLX.
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
ACTS = ['spreccap_cs', 'spreccap_e03', 'selu', 'elu']
SEEDS = list(range(42010, 42025))           # dieselben 15 Bestaetigungs-Seeds wie 3a-3
NCAL = 1000                                  # Kalibrierung: erste 1000 Trainingsbilder (nicht Test, nicht Val)
SAVE_SEED = 42010                            # Gewichte nur fuer diesen Seed speichern (Platz)
# Quantisierungsstufen: (Name, Gewichte, Aktivierungen); 'f8' = FP8 E4M3 simuliert
QUANT = [('w8a8', 'i8', 'i8'), ('w8a4', 'i8', 'i4'), ('w4a8', 'i4', 'i8'), ('fp8', 'f8', 'f8'), ('w8a8_p9999', 'i8', 'i8p')]
BACKEND = 'kernel'

# --- Aktivierungen ---------------------------------------------------------------
from precap._presets import PRESETS
def make_act(n):
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
GAINS = {n: gain(n) for n in ACTS}

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
        r = np.random.default_rng(seed); g = GAINS[n]; dims = [784] + [WIDTH] * depth
        self.hid = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]; self.out = nn.Linear(WIDTH, 10)
        for l, (a, b) in zip(self.hid, zip(dims[:-1], dims[1:])):
            l.weight = mx.array(r.normal(0, g / math.sqrt(a), size=(b, a)).astype(np.float32)); l.bias = mx.zeros((b,))
        self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(WIDTH), size=(10, WIDTH)).astype(np.float32)); self.out.bias = mx.zeros((10,))
        self.act = make_act(n)
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
    if kind in ('i8', 'i8p'): return q_int(a, 8, rng / 127)
    if kind == 'i4': return q_int(a, 4, rng / 7)
    if kind == 'f8': return q_fp8(a, rng / 448)
    raise ValueError(kind)
def calib(m, xc):
    # Betrag der Eingaenge jeder Linearschicht auf den Kalibrierbildern: max und 99.99-Perzentil
    rmax, rp = [], []; h = mx.array(xc)
    for l in m.hid + [m.out]:
        a = np.abs(np.array(h)).ravel(); rmax.append(float(a.max())); rp.append(float(np.quantile(a, 0.9999)))
        if l is not m.out: h = m.act(l(h))
    return rmax, rp
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
       [f'test_err_{q[0]}' for q in QUANT] + ['act_range_max_median', 'act_range_max_last', 'act_range_p9999_median', 'sec_train', 'ms_per_step', 'backend', 'timestamp']
def rid(c): return f"{c['dataset']}_d{c['depth']}_{c['act']}_s{c['seed']}"
def lrsel():
    sel = {}
    for r in csv.DictReader(open(ROOT / 'lr_from_p3a3.csv')): sel[(r['dataset'], int(r['depth']), r['act'])] = float(r['lr_sel'])
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
    if status == 'ok':
        fwd = lambda x: m(x)
        _, va = evaluate(fwd, xv, yv); _, ta = evaluate(fwd, xte, yte); out['val_err_fp32'] = 1 - va; out['test_err_fp32'] = 1 - ta
        rmax, rp = calib(m, xtr[:NCAL]); out['act_range_max_median'] = float(np.median(rmax[1:])); out['act_range_max_last'] = rmax[-1]; out['act_range_p9999_median'] = float(np.median(rp[1:]))
        for name, wk, ak in QUANT:
            f = qforward(m, wk, ak, rp if ak == 'i8p' else rmax); _, a = evaluate(f, xte, yte); out[f'test_err_{name}'] = 1 - a
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
    out = {'gains': GAINS, 'backend': BACKEND, 'device': str(mx.default_device()), 'n_configs': len(list(configs()))}
    z = mx.array(np.linspace(-8, 8, 20001).astype(np.float32))
    for n in ['spreccap_cs', 'spreccap_e03']:
        out[n + '_max_abs_vs_numpy'] = float(np.abs(np.array(make_act(n)(z)) - np_act(n, np.array(z, dtype=np.float64))).max())
    out['selu_max_abs_vs_numpy'] = float(np.abs(np.array(nn.selu(z)) - np_act('selu', np.array(z, dtype=np.float64))).max())
    a = mx.array(np.linspace(-3, 3, 13).astype(np.float32))
    out['q_int8_example'] = np.array(q_int(a, 8, 3 / 127)).round(4).tolist(); out['q_fp8_448'] = float(q_fp8(mx.array([448.0, 500.0, 0.0]), 1.0)[1])
    out['q_fp8_grid_ok'] = bool(np.allclose(np.array(q_fp8(mx.array([1.0, 1.0625, 1.125, 0.015625 * 1.1]), 1.0)), [1.0, 1.0, 1.125, 0.017578125]))
    return out

def main():
    global STEPS, OUTDIR, SEEDS, DATASETS, DEPTHS, BACKEND
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--probe', action='store_true')
    p.add_argument('--backend', default='kernel', choices=['kernel', 'ref']); p.add_argument('--synth', action='store_true'); p.add_argument('--limit', type=int, default=0); a = p.parse_args()
    BACKEND = a.backend
    if a.synth: DATASETS = ['synth']
    if a.smoke: STEPS = 40; OUTDIR = ROOT / 'smoke'; SEEDS = [42010, 42011]
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
    skip = finished(); nrun = 0
    for i, c in enumerate(configs()):
        if a.limit and nrun >= a.limit: break
        if i % a.workers != a.worker: continue
        if rid(c) in skip: continue
        nrun += 1; o = execute(c, a.worker); print('done', rid(c), o['status'], datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
