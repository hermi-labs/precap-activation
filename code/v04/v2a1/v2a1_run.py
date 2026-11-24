# V2a-1 (privat, mit Claude, praeregistriert in V2a1_Praeregistrierung.md): BPreCap gegen SELU im CNN ohne Normierung und ohne Residuen,
# 4 Bit PTQ, auf dem M1 Pro. Netz, Init, Daten, Training und Quantisierung exakt wie der Pilot studien/v2a_entwurf/v2a_pilot4.py
# (L = 32, 32 Kanaele, Delta-orthogonal, Adam mit konstanter LR, 1720 Schritte, Batch 128); LR je Form aus den Piloten
# (BPreCap 1e-4, SELU 3e-5, gewaehlt am Val-Satz mit Seed 42000). Neu: 12 neue Seeds (42330-42341), Messung auf dem Test-Satz,
# dazu die Kalibrierung p999 und w8a4a_best (Wahl am Fehler auf den Kalibrierbildern).
# Aufruf: --selfcheck | --worker i --workers n | --smoke
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
ROOT = Path(__file__).resolve().parent; OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data')) / 'cifar10' / 'cifar-10-batches-bin'
KC = json.load(open(ROOT / 'kL2B_d32.json')); S_, C_, RHO, M = 1.5, 1.0, 0.8, 3.0
L, CH, BATCH, STEPS, NVAL, NCAL, NSUB = 32, 32, 128, 1720, 5000, 1000, 2_000_000
LR = {'bprecap': 1e-4, 'selu': 3e-5}; ACTS = ('bprecap', 'selu'); SEEDS = tuple(range(42330, 42342))
MODES = ('fp32', 'w8a8', 'w8a4a_max', 'w8a4a_p999', 'w8a4a_p9999'); CALS4 = ('max', 'p999', 'p9999')
_C = {}
def load():
    if 'd' in _C: return _C['d']
    def rd(fs):
        a = np.concatenate([np.fromfile(DATA / f, dtype=np.uint8).reshape(-1, 3073) for f in fs]); x = a[:, 1:].astype(np.float32) / 255
        return x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), a[:, 0].astype(np.int64)
    xtr, ytr = rd([f'data_batch_{i}.bin' for i in range(1, 6)]); xte, yte = rd(['test_batch.bin'])
    xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
    m_, s_ = xtr.mean(), xtr.std(); xtr = (xtr - m_) / s_; xv = (xv - m_) / s_; xte = (xte - m_) / s_
    _C['d'] = (xtr, ytr, xv, yv, xte, yte); return _C['d']
def make_act(name):
    if name == 'selu': return nn.selu
    lam, al, b = KC['lam'], KC['alpha'], KC['b']; ec = math.exp(C_)
    def f(x):
        w = S_ * mx.tanh(b * x / S_); ce = mx.where(w <= C_, mx.exp(mx.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_)); y = lam * mx.where(x > 0, ce, al * ce)
        u = y / M; a = mx.abs(u); return M * mx.where(a <= RHO, u, mx.sign(u) * (RHO + (1 - RHO) * mx.tanh((a - RHO) / (1 - RHO))))
    return f
def delta_orth(r, cin, cout):
    A = r.normal(size=(max(cin, cout), max(cin, cout))); Q, R_ = np.linalg.qr(A); Q = Q * np.sign(np.diag(R_))
    w = np.zeros((cout, 3, 3, cin), dtype=np.float32); w[:, 1, 1, :] = Q[:cout, :cin] * (math.sqrt(cout / cin) if cin < cout else 1.0); return w
class CNN(nn.Module):
    def __init__(self, act, seed):
        super().__init__(); r = np.random.default_rng(seed); dims = [3] + [CH] * L
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in zip(dims[:-1], dims[1:])]
        for c, (a, b) in zip(self.convs, zip(dims[:-1], dims[1:])): c.weight = mx.array(delta_orth(r, a, b)); c.bias = mx.zeros((b,))
        self.out = nn.Linear(CH, 10); self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(CH), size=(10, CH)).astype(np.float32)); self.out.bias = mx.zeros((10,))
        self.act = act
    def __call__(self, x):
        for c in self.convs: x = self.act(c(x))
        return self.out(mx.mean(x, axis=(1, 2)))
def train(name, lr, seed, steps=STEPS):
    xtr, ytr = load()[:2]
    mx.random.seed(seed); m = CNN(make_act(name), seed); mx.eval(m.parameters())
    opt = optim.Adam(learning_rate=lr); lf = lambda mm, a, b: nn.losses.cross_entropy(mm(a), b, reduction='mean')
    vg = nn.value_and_grad(m, lf); state = [m.state, opt.state]
    def step(a, b):
        l, g = vg(m, a, b); opt.update(m, g); return l
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(seed + 777); perm = r.permutation(len(xtr)); pos = 0; ok = True; tail = []
    for s in range(1, steps + 1):
        if pos + 128 > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + 128]; pos += 128; l = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state); lf_ = float(l)
        if not math.isfinite(lf_): ok = False; break
        if s > steps - 50: tail.append(lf_)
    return m, ok, (float(np.mean(tail)) if tail else float('nan'))
def qw(W):
    ax = tuple(range(1, W.ndim)); am = mx.maximum(mx.max(mx.abs(W), axis=ax, keepdims=True), 1e-12); sc = am / 127
    return mx.clip(mx.round(W / sc), -127, 127) * sc
def qa_sym(a, bits, rng):
    qm = 2 ** (bits - 1) - 1; sc = max(rng, 1e-12) / qm; return mx.clip(mx.round(a / sc), -qm, qm) * sc
def qa_asym(a, bits, lohi):
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 2 ** bits - 1; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc)
    return (mx.clip(mx.round(a / sc) + zp, 0, n) - zp) * sc
def layer_inputs(m, x):
    outs = None
    for i in range(0, len(x), 250):
        h = mx.array(x[i:i + 250]); cur = []
        for c in m.convs: cur.append(h); h = m.act(c(h))
        cur.append(mx.mean(h, axis=(1, 2))); mx.eval(cur)
        cur = [np.array(t, dtype=np.float32).ravel() for t in cur]
        outs = cur if outs is None else [np.concatenate([o, c_]) for o, c_ in zip(outs, cur)]
    return outs
def calib(m, xc, seed):   # wie v2a_pilot4.calib, dazu p999 (aus derselben Stichprobe; aendert die anderen Werte nicht)
    r = np.random.default_rng(seed); R = {'max_abs': [], 'max': [], 'p999': [], 'p9999': []}
    for a in layer_inputs(m, xc):
        R['max_abs'].append(float(np.abs(a).max())); R['max'].append((float(a.min()), float(a.max())))
        s = a if len(a) <= NSUB else a[r.choice(len(a), NSUB, replace=False)]
        R['p9999'].append((float(np.quantile(s, 0.00005)), float(np.quantile(s, 0.99995)))); R['p999'].append((float(np.quantile(s, 0.0005)), float(np.quantile(s, 0.9995))))
    return R
def qforward(m, mode, R):
    g = (lambda W: W) if mode == 'fp32' else qw
    Wc = [(g(c.weight), c.bias) for c in m.convs]; Wo = (g(m.out.weight), m.out.bias)
    def qx(h, i):
        if mode == 'fp32': return h
        if mode == 'w8a8': return qa_sym(h, 8, R['max_abs'][i])
        return qa_asym(h, 4, R[mode.split('_')[1]][i])
    def f(x):
        h = x
        for i, (W, b) in enumerate(Wc): h = m.act(mx.conv2d(qx(h, i), W, padding=1) + b)
        h = qx(mx.mean(h, axis=(1, 2)), len(Wc)); return h @ Wo[0].T + Wo[1]
    return f
def err(f, x, y):
    ac = 0
    for i in range(0, len(x), 500): ac += int((mx.argmax(f(mx.array(x[i:i + 500])), 1) == mx.array(y[i:i + 500])).sum())
    return 1 - ac / len(x)
COLS = ['run_id', 'act', 'lr', 'seed', 'steps', 'status', 'train_loss_last50', 'val_err_fp32'] + [f'test_err_{k}' for k in MODES] + \
       ['test_err_w8a4a_best', 'w8a4a_best_cal', 'act_lo_max_median', 'act_hi_max_median', 'act_hi_p9999_median', 'sec', 'timestamp']
def rid(c): return f"{c['act']}_lr{c['lr']:g}_s{c['seed']}"
def run_one(c):
    t0 = time.time(); xtr, ytr, xv, yv, xte, yte = load(); m, ok, tl = train(c['act'], c['lr'], c['seed'], c.get('steps', STEPS))
    o = dict(run_id=rid(c), act=c['act'], lr=c['lr'], seed=c['seed'], steps=c.get('steps', STEPS), status='ok' if ok else 'fail', train_loss_last50=tl)
    if ok:
        o['val_err_fp32'] = err(m, xv, yv); R = calib(m, xtr[:NCAL], c['seed'])
        for k in MODES: o[f'test_err_{k}'] = err(qforward(m, k, R), xte, yte)
        ce = {cal: err(qforward(m, f'w8a4a_{cal}', R), xtr[:NCAL], ytr[:NCAL]) for cal in CALS4}
        bc = min(CALS4, key=lambda k: (ce[k], CALS4.index(k))); o['w8a4a_best_cal'] = bc; o['test_err_w8a4a_best'] = o[f'test_err_w8a4a_{bc}']
        o['act_lo_max_median'] = float(np.median([l_ for l_, _ in R['max'][1:-1]])); o['act_hi_max_median'] = float(np.median([h for _, h in R['max'][1:-1]]))
        o['act_hi_p9999_median'] = float(np.median([h for _, h in R['p9999'][1:-1]]))
    o['sec'] = round(time.time() - t0, 1); o['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S'); return o
def configs():   # je Seed beide Formen; Reihenfolge je Seed abwechselnd, damit beide Worker beide Formen rechnen
    out = []
    for j, s in enumerate(SEEDS):
        for a in (ACTS if j % 2 == 0 else ACTS[::-1]): out.append(dict(act=a, lr=LR[a], seed=s))
    return out
def read_rows(d):
    rows = []
    for f in sorted((d / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    return rows
def work(cfgs, wid, nw, outdir):
    p = outdir / 'workers' / f'w{wid}.csv'; p.parent.mkdir(parents=True, exist_ok=True)
    done = {r['run_id'] for r in read_rows(outdir)} if (outdir / 'workers').exists() else set()
    for i, c in enumerate(cfgs):
        if i % nw != wid or rid(c) in done: continue
        o = run_one(c); new = not p.exists()
        with open(p, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
            if new: w.writeheader()
            w.writerow(o)
        print(time.strftime('%H:%M'), o['run_id'], o['status'], o.get('test_err_fp32'), o.get('test_err_w8a4a_max'), flush=True)
def selfcheck():
    out = dict(L=L, C=CH, steps=STEPS, n_confirm=len(configs()), lr=LR, chi1=KC['chi1'])
    X, Wq = np.polynomial.hermite_e.hermegauss(150); Wq = Wq / Wq.sum(); x = mx.array(X.astype(np.float32))
    for a in ACTS: y = np.array(make_act(a)(x), dtype=np.float64); out[f'Ef2_{a}'] = round(float(np.sum(Wq * y * y)), 4)
    w = delta_orth(np.random.default_rng(0), CH, CH)[:, 1, 1, :]; out['delta_orth_ok'] = bool(np.abs(w @ w.T - np.eye(CH)).max() < 1e-5)
    out['bprecap_Ef2_ok'] = abs(out['Ef2_bprecap'] - 1) < 0.01
    out['seeds_new'] = not (set(SEEDS) & ({42000, 42001} | set(range(42300, 42322))))
    w2 = {0: [], 1: []}
    for i, c in enumerate(configs()): w2[i % 2].append(c['act'])
    out['workers_mixed'] = all(len(set(v)) == 2 for v in w2.values()); print(json.dumps(out, indent=1)); return out
def main():
    global OUTDIR, STEPS, SEEDS
    p = argparse.ArgumentParser(); p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1); a = p.parse_args()
    print('Geraet:', mx.default_device(), flush=True)
    if a.selfcheck: selfcheck(); return
    if a.smoke: OUTDIR = ROOT / 'smoke'; STEPS = 20; SEEDS = SEEDS[:2]; work([dict(c, steps=STEPS) for c in configs()], 0, 1, OUTDIR); return
    work(configs(), a.worker, a.workers, ROOT)
if __name__ == '__main__': main()
