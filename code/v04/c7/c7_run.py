# C7 (Entwurf, privat, mit Claude): unabhaengige Wiederholung der V2a2-Kernvergleiche mit PyTorch/CUDA auf anderer Hardware (Aarons PC, RTX 4070).
# Eigenstaendige Uebertragung von studien/v2a2/v2a2_run.py (MLX) nach PyTorch: Daten, Init (numpy-RNG, daher identische Startgewichte), Formen,
# AdamW wie MLX 0.32 (ohne Bias-Korrektur, entkoppelter WD w <- w(1 - lr wd)), Kosinus auf 0, Batch-Reihenfolge, 8 Messpunkte, fruehes Stoppen
# am kleinsten Val-Fehler, PTQ (w8 je Ausgangskanal; a8 symmetrisch; a4 asymmetrisch mit max/p999/p9999/p99999/mse4, Wahl auf 1000 Trainings-
# bildern), fuer TAT zusaetzlich CLE. Hyperparameter aus dem V2a2-Tuning uebernommen (kein Tuning). Nicht bitgleich zu MLX zu erwarten.
# PreCap, BPreCap, Huelle, Rezept: Ideen Aarons.
# Aufruf: python c7_run.py --selfcheck | --smoke | --worker W --workers N [--device cuda|mps|cpu]
import argparse, csv, json, math, os, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent; OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', str(Path.home() / 'precap_data'))) / 'cifar10' / 'cifar-10-batches-bin'
L, CH, BATCH, STEPS, NVAL, NCAL, NSUB, NPTS = 50, 32, 128, 3440, 5000, 1000, 2_000_000, 8
SEEDS = tuple(range(42350, 42362))                                       # wie V2a2 (gleiche Init und Batch-Reihenfolge)
RUNS = (('bprecap', 3e-5, 0.0), ('tat', 1e-4, 0.0), ('selu', 3e-5, 0.0))  # V2a2 tune/hp_sel.json
KC = dict(lam=1.7524206867535228, alpha=1.6813138724568801, b=0.4647301472997325)   # studien/v2a2/kL2B_d50.json
TAT = dict(neg=0.4305229485024027, scale=1.2989477863189371)                         # studien/v2a2/v2a_tat_params.json, L = 50
S_, C_, RHO, M = 1.5, 1.0, 0.8, 3.0
CALS4 = ('max', 'p999', 'p9999', 'p99999', 'mse4'); MODES = ('fp32', 'w8a8') + tuple(f'w8a4a_{k}' for k in CALS4)
QS = {'p999': 0.0005, 'p9999': 0.00005, 'p99999': 0.000005}
DEV = 'cpu'; _C = {}

def load():
    if 'd' in _C: return _C['d']
    def rd(fs):
        a = np.concatenate([np.fromfile(DATA / f, dtype=np.uint8).reshape(-1, 3073) for f in fs]); x = a[:, 1:].astype(np.float32) / 255
        return x.reshape(-1, 3, 32, 32), a[:, 0].astype(np.int64)            # NCHW (MLX: NHWC, gleiche Werte)
    xtr, ytr = rd([f'data_batch_{i}.bin' for i in range(1, 6)]); xte, yte = rd(['test_batch.bin'])
    xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
    # Mittel und Std exakt wie in den MLX-Studien (numpy 2.5.4, float32). Andere numpy-Versionen summieren float32 anders (2.0.2: 0x3ef25d60),
    # deshalb fest eingetragen statt neu berechnet (wie C6).
    m_ = np.frombuffer(bytes.fromhex('3ef25d29'), dtype='>f4')[0].astype(np.float32); s_ = np.frombuffer(bytes.fromhex('3e80e268'), dtype='>f4')[0].astype(np.float32)
    xtr = (xtr - m_) / s_; xv = (xv - m_) / s_; xte = (xte - m_) / s_
    _C['d'] = (xtr, ytr, xv, yv, xte, yte); _C['ms'] = (float(m_), float(s_)); return _C['d']

def act_fn(name):
    if name == 'selu': return lambda x: 1.0507 * torch.where(x > 0, x, 1.67326 * (torch.exp(x) - 1))   # wie mlx.nn.selu (MLX 0.32: gerundete Konstanten)
    if name == 'tat':
        a, s = TAT['neg'], TAT['scale']; return lambda x: s * torch.where(x > 0, x, a * x)
    lam, al, b = KC['lam'], KC['alpha'], KC['b']; ec = math.exp(C_)
    def f(x):
        w = S_ * torch.tanh(b * x / S_); ce = torch.where(w <= C_, torch.exp(torch.clamp(w, max=C_)) - 1, ec - 1 + ec * (w - C_))
        y = lam * torch.where(x > 0, ce, al * ce); u = y / M; a = torch.abs(u)
        return M * torch.where(a <= RHO, u, torch.sign(u) * (RHO + (1 - RHO) * torch.tanh((a - RHO) / (1 - RHO))))
    return f

def delta_orth(r, cin, cout):   # identisch zu v2a2_run.delta_orth (numpy), Layout (cout, kh, kw, cin)
    A = r.normal(size=(max(cin, cout), max(cin, cout))); Q, R_ = np.linalg.qr(A); Q = Q * np.sign(np.diag(R_))
    w = np.zeros((cout, 3, 3, cin), dtype=np.float32); w[:, 1, 1, :] = Q[:cout, :cin] * (math.sqrt(cout / cin) if cin < cout else 1.0); return w

class CNN(torch.nn.Module):
    def __init__(self, name, seed):
        super().__init__(); r = np.random.default_rng(seed); dims = [3] + [CH] * L
        self.ws = torch.nn.ParameterList(); self.bs = torch.nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            self.ws.append(torch.nn.Parameter(torch.from_numpy(delta_orth(r, a, b).transpose(0, 3, 1, 2).copy())))   # -> (cout, cin, kh, kw)
            self.bs.append(torch.nn.Parameter(torch.zeros(b)))
        self.wo = torch.nn.Parameter(torch.from_numpy(r.normal(0, 1 / math.sqrt(CH), size=(10, CH)).astype(np.float32)))
        self.bo = torch.nn.Parameter(torch.zeros(10)); self.name = name; self.act = act_fn(name)
    def forward(self, x):
        for W, b in zip(self.ws, self.bs): x = self.act(F.conv2d(x, W, b, padding=1))
        return x.mean(dim=(2, 3)) @ self.wo.T + self.bo

@torch.no_grad()
def err(f, x, y):
    ac = 0
    for i in range(0, len(x), 500):
        ac += int((f(torch.from_numpy(x[i:i + 500]).to(DEV)).argmax(1).cpu() == torch.from_numpy(y[i:i + 500])).sum())
    return 1 - ac / len(x)

class MLXAdamW:   # MLX 0.32 AdamW, Voreinstellungen: betas (0.9, 0.999), eps 1e-8, bias_correction False; WD vor dem Adam-Schritt
    def __init__(self, params, wd): self.p = list(params); self.wd = wd; self.m = [torch.zeros_like(q) for q in self.p]; self.v = [torch.zeros_like(q) for q in self.p]
    @torch.no_grad()
    def step(self, lr):
        for q, m, v in zip(self.p, self.m, self.v):
            g = q.grad; q.mul_(1 - lr * self.wd)
            m.mul_(0.9).add_(0.1 * g); v.mul_(0.999).add_(0.001 * g * g)
            q.sub_(lr * m / (torch.sqrt(v) + 1e-8))

def lr_at(peak, k, steps): return peak * 0.5 * (1 + math.cos(math.pi * min(k, steps) / steps))   # MLX cosine_decay: k-tes Update nutzt schedule(k-1)

def train(name, lr, wd, seed, steps):
    xtr, ytr, xv, yv = load()[:4]
    torch.manual_seed(seed); m = CNN(name, seed).to(DEV); opt = MLXAdamW(m.parameters(), wd)
    r = np.random.default_rng(seed + 777); perm = r.permutation(len(xtr)); pos = 0; ok = True; tail = []
    xtr_d = torch.from_numpy(xtr).to(DEV); ytr_d = torch.from_numpy(ytr).to(DEV)
    pts = sorted({max(1, round(steps * k / NPTS)) for k in range(1, NPTS + 1)}); kurve = []; best = None
    for s in range(1, steps + 1):
        if pos + BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = torch.from_numpy(perm[pos:pos + BATCH]).to(DEV); pos += BATCH
        loss = F.cross_entropy(m(xtr_d[b]), ytr_d[b])
        for q in m.parameters(): q.grad = None
        loss.backward(); opt.step(lr_at(lr, s - 1, steps)); lf_ = loss.item()
        if not math.isfinite(lf_): ok = False; break
        if s > steps - 50: tail.append(lf_)
        if s in pts:
            e = err(m, xv, yv); kurve.append((s, e))
            if best is None or e < best[1]: best = (s, e, {k: v.detach().clone() for k, v in m.state_dict().items()})
    return m, ok, (float(np.mean(tail)) if tail else float('nan')), kurve, best

def qw(W):
    am = torch.clamp(W.abs().amax(dim=tuple(range(1, W.ndim)), keepdim=True), min=1e-12); sc = am / 127
    return torch.clamp(torch.round(W / sc), -127, 127) * sc
def qa_sym(a, bits, rng):
    qm = 2 ** (bits - 1) - 1; sc = max(rng, 1e-12) / qm; return torch.clamp(torch.round(a / sc), -qm, qm) * sc
def qa_asym(a, bits, lohi):
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 2 ** bits - 1; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc)
    return (torch.clamp(torch.round(a / sc) + zp, 0, n) - zp) * sc
def mse_asym(s, lohi, bits=4):   # wie v2a2_run.mse_asym: (lo, hi) = k/100 * (min, max), k = 1..100, kleinster MSE (float32)
    lo, hi = min(lohi[0], 0.0), max(lohi[1], 0.0); x = torch.from_numpy(s.astype(np.float32)).to(DEV); n = 2 ** bits - 1; E = []
    for ks in (range(1, 26), range(26, 51), range(51, 76), range(76, 101)):
        k = np.array(list(ks), dtype=np.float64) / 100; sc = np.maximum((hi - lo) * k, 1e-12) / n; zp = np.round(-lo * k / sc)
        scm = torch.from_numpy(sc.astype(np.float32))[:, None].to(DEV); zpm = torch.from_numpy(zp.astype(np.float32))[:, None].to(DEV)
        q = (torch.clamp(torch.round(x[None, :] / scm) + zpm, 0, n) - zpm) * scm; E.append(((q - x[None, :]) ** 2).mean(dim=1).cpu())
    E = torch.cat(E).numpy(); k = (int(np.argmin(E)) + 1) / 100; return (lo * k, hi * k)

@torch.no_grad()
def calib(m, xc, seed):   # wie v2a2_run.calib: Eingaenge aller Faltungen und der Linearschicht, Teile zu 250 Bildern, Stichprobe je Tensor
    r = np.random.default_rng(seed); R = {'max_abs': [], 'max': [], 'p999': [], 'p9999': [], 'p99999': [], 'mse4': []}
    hs = [torch.from_numpy(xc[i:i + 250]).to(DEV) for i in range(0, len(xc), 250)]
    def stats(parts):
        a = np.concatenate([t.permute(*((0, 2, 3, 1) if t.ndim == 4 else (0, 1))).contiguous().cpu().numpy().ravel() for t in parts])   # NHWC-Reihenfolge wie MLX
        R['max_abs'].append(float(np.abs(a).max())); R['max'].append((float(a.min()), float(a.max())))
        s = a if len(a) <= NSUB else a[r.choice(len(a), NSUB, replace=False)]
        for k, q in QS.items(): R[k].append((float(np.quantile(s, q)), float(np.quantile(s, 1 - q))))
        R['mse4'].append(mse_asym(s, R['max'][-1]))
    for W, b in zip(m.ws, m.bs):
        stats(hs); hs = [m.act(F.conv2d(h, W, b, padding=1)) for h in hs]
    stats([h.mean(dim=(2, 3)) for h in hs]); return R

def qforward(m, mode, R):
    g = (lambda W: W) if mode == 'fp32' else qw
    Wc = [(g(W.detach()), b.detach()) for W, b in zip(m.ws, m.bs)]; Wo = (g(m.wo.detach()), m.bo.detach())
    def qx(h, i):
        if mode == 'fp32': return h
        if mode == 'w8a8': return qa_sym(h, 8, R['max_abs'][i])
        return qa_asym(h, 4, R[mode.split('_')[1]][i])
    @torch.no_grad()
    def f(x):
        h = x
        for i, (W, b) in enumerate(Wc): h = m.act(F.conv2d(qx(h, i), W, b, padding=1))
        h = qx(h.mean(dim=(2, 3)), len(Wc)); return h @ Wo[0].T + Wo[1]
    return f

def cle(m, sweeps=30):   # wie v2a2_run.cle (numpy float64); Gewichte hier (cout, cin, kh, kw)
    Ws = [W.detach().cpu().double().numpy() for W in m.ws] + [m.wo.detach().cpu().double().numpy()]
    Bs = [b.detach().cpu().double().numpy() for b in m.bs]
    for _ in range(sweeps):
        for k in range(len(m.ws)):
            r1 = np.abs(Ws[k]).max(axis=(1, 2, 3)); nxt = Ws[k + 1]
            r2 = np.abs(nxt).max(axis=(0, 2, 3)) if nxt.ndim == 4 else np.abs(nxt).max(axis=0)
            s = np.where((r1 > 0) & (r2 > 0), np.sqrt(r1 / np.maximum(r2, 1e-30)), 1.0)
            Ws[k] = Ws[k] / s[:, None, None, None]; Bs[k] = Bs[k] / s
            Ws[k + 1] = nxt * (s[None, :, None, None] if nxt.ndim == 4 else s[None, :])
    m2 = CNN(m.name, 0).to(DEV)
    with torch.no_grad():
        for p, W in zip(m2.ws, Ws[:-1]): p.copy_(torch.from_numpy(W.astype(np.float32)))
        for p, b in zip(m2.bs, Bs): p.copy_(torch.from_numpy(b.astype(np.float32)))
        m2.wo.copy_(torch.from_numpy(Ws[-1].astype(np.float32))); m2.bo.copy_(m.bo.detach())
    return m2

def ptq(m, seed, pre):
    xtr, ytr, _, _, xte, yte = load(); R = calib(m, xtr[:NCAL], seed); o = {}
    for k in MODES: o[f'{pre}test_err_{k}'] = err(qforward(m, k, R), xte, yte)
    ce = {cal: err(qforward(m, f'w8a4a_{cal}', R), xtr[:NCAL], ytr[:NCAL]) for cal in CALS4}
    bc = min(CALS4, key=lambda k: (ce[k], CALS4.index(k))); o[f'{pre}w8a4a_best_cal'] = bc; o[f'{pre}test_err_w8a4a_best'] = o[f'{pre}test_err_w8a4a_{bc}']
    mid = slice(1, -1)
    o[f'{pre}act_lo_max_median'] = float(np.median([l_ for l_, _ in R['max'][mid]])); o[f'{pre}act_hi_max_median'] = float(np.median([h for _, h in R['max'][mid]]))
    o[f'{pre}act_absmax_max'] = float(max(R['max_abs'][mid])); return o

PT = ['test_err_' + k for k in MODES + ('w8a4a_best',)] + ['w8a4a_best_cal', 'act_lo_max_median', 'act_hi_max_median', 'act_absmax_max']
COLS = ['run_id', 'act', 'lr', 'wd', 'seed', 'steps', 'status', 'device', 'torch', 'train_loss_last50', 'val_err_end', 'val_err_es', 'es_step', 'val_kurve'] + \
       ['es_' + c for c in PT] + ['es_cle_' + c for c in PT] + ['sec', 'timestamp']
def rid(c): return f"c7_{c['act']}_lr{c['lr']:g}_wd{c['wd']:g}_s{c['seed']}"
def run_one(c, steps=STEPS):
    t0 = time.time(); m, ok, tl, kurve, best = train(c['act'], c['lr'], c['wd'], c['seed'], steps)
    o = dict(run_id=rid(c), act=c['act'], lr=c['lr'], wd=c['wd'], seed=c['seed'], steps=steps, status='ok' if ok else 'fail', device=DEV,
             torch=torch.__version__, train_loss_last50=tl, val_kurve=';'.join(f'{s}:{e:.4f}' for s, e in kurve))
    if ok:
        o['val_err_end'] = kurve[-1][1]; o['val_err_es'] = best[1]; o['es_step'] = best[0]
        m.load_state_dict(best[2]); o.update(ptq(m, c['seed'], 'es_'))
        if c['act'] == 'tat': o.update(ptq(cle(m), c['seed'], 'es_cle_'))
    o['sec'] = round(time.time() - t0, 1); o['timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S'); return o
def configs(): return [dict(act=a, lr=lr, wd=wd, seed=s) for j, s in enumerate(SEEDS) for a, lr, wd in RUNS[j % 3:] + RUNS[:j % 3]]
def write(p, o):
    new = not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)
def finished(d):
    out = set()
    for f in d.glob('w*.csv'):
        for r in csv.DictReader(open(f)): out.add(r['run_id'])
    return out
def setup(device):
    global DEV
    DEV = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # echtes float32, kein TF32
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); torch.use_deterministic_algorithms(True)
def selfcheck():
    X, Wq = np.polynomial.hermite_e.hermegauss(150); Wq = Wq / Wq.sum(); x = torch.from_numpy(X.astype(np.float32))
    o = dict(device=DEV, torch=torch.__version__, cuda=torch.cuda.is_available(), n_runs=len(configs()))
    for a in ('bprecap', 'tat', 'selu'):
        y = act_fn(a)(x).double().numpy(); o[f'Ef2_{a}'] = round(float(np.sum(Wq * y * y)), 4)
    w = delta_orth(np.random.default_rng(0), CH, CH)[:, 1, 1, :]; o['delta_orth_ok'] = bool(np.abs(w @ w.T - np.eye(CH)).max() < 1e-5)
    o['lr_schedule'] = [lr_at(1e-4, 0, STEPS), lr_at(1e-4, STEPS // 2, STEPS), lr_at(1e-4, STEPS - 1, STEPS)]
    o['data_found'] = (DATA / 'test_batch.bin').exists()
    o['ok'] = o['n_runs'] == 36 and o['delta_orth_ok'] and o['data_found'] and all(abs(o[f'Ef2_{a}'] - 1) < 0.01 for a in ('bprecap', 'tat', 'selu'))
    return o
def main():
    global OUTDIR
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--device', default=None); a = p.parse_args()
    setup(a.device)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    if a.smoke:   # 3 kurze Laeufe (je Form einer), eigener Ordner
        OUTDIR = ROOT / 'smoke'
        for c in configs()[:3]: o = run_one(c, steps=40); write(OUTDIR / 'workers' / 'w0.csv', o); print('smoke', o['run_id'], o['status'], o.get('es_test_err_fp32'), o['sec'], flush=True)
        return
    skip = finished(OUTDIR / 'workers')
    for i, c in enumerate(configs()):
        if i % a.workers != a.worker or rid(c) in skip: continue
        o = run_one(c); write(OUTDIR / 'workers' / f'w{a.worker}.csv', o)
        print('done', o['run_id'], o['status'], o.get('es_test_err_fp32'), o.get('es_test_err_w8a4a_best'), o['sec'], datetime.now().strftime('%H:%M:%S'), flush=True)
if __name__ == '__main__': main()
