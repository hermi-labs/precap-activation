# C6: unabhaengige Wiederholung der B-Kernbefunde mit PyTorch/CUDA auf anderer Hardware (Aarons PC, RTX 4070). Mit Claude geschrieben.
# Eigenstaendige Uebertragung von studien/b10/b10_run.py (MLX) nach PyTorch: Daten, Init (numpy-RNG, daher identische Startgewichte),
# Formen, AdamW mit Kosinus auf 0, Batch-Reihenfolge, Messpunkte alle 860 Schritte, fruehes Stoppen nach fp32-Val, Quantisierung nach dem Training
# (w8 je Ausgangskanal; a4 symmetrisch und asymmetrisch, je Lauf beste von vier Kalibrierungen auf 1000 Trainingsbildern; a3 symmetrisch).
# Nicht bitgleich zu MLX zu erwarten (andere Kernel), aber gleiche Rechenvorschrift. Hyperparameter aus den MLX-Studien uebernommen (kein Tuning).
# Aufruf: python c6_run.py --selfcheck | --smoke | --worker W --workers N [--device cuda|mps|cpu]
import argparse, csv, json, math, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent; OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', str(Path.home() / 'precap_data')))
WIDTH = 256; DEPTH = 100; BATCH = 128; STEPS = 6880; NVAL = 5000; NCAL = 1000
CHECKS = (860, 1720, 2580, 3440, 4300, 5160, 6020)   # Ende 6880 = Endmessung
SEEDS = list(range(42168, 42180))                    # wie B10/B10b/B12 (gleiche Init und Batch-Reihenfolge wie dort)
# Formen und Hyperparameter aus den MLX-Studien (dortige Wahl nach Tuning), nicht neu getunt
RUNS = [('kL2B_tr', 1e-4, 0.3, 'B10'), ('LB_tr', 1e-4, 0.3, 'B10'), ('tat', 1e-4, 0.3, 'B10b TAT a'), ('tat', 2e-4, 0.0, 'B10'), ('selu_tr', 2e-4, 0.3, 'B12')]
TAT = dict(neg=0.5704395323991776, scale=1.2284042441106242)   # dks, eta 0.9, d100
S_, C_, B_M, RHO = 1.5, 1.0, 3.0, 0.8
KL2B = dict(lam=2.9843721185770757, alpha=1.406734313682772, b=0.28967082304815317)   # math/p4_bounded_depth.py, d100, M = 3
LB = dict(beta=1.0353165703655918, M=2.2165089809922915)                              # math/p5_linear_bound.py, d100
CALS4 = ['max', 'p999', 'p9999', 'mse4']; CALS3 = ['max', 'p999', 'p9999', 'mse3']
DEV = 'cpu'

# --- Formen (float32 in torch) und float64-Referenz fuer die Gains -----------------------
def base(n): return n.removesuffix('_tr')
def recipe(n): return n.endswith('_tr')
def hull_t(y, M):
    u = y / M; a = u.abs(); k = torch.tanh((a - RHO) / (1 - RHO)); return M * torch.where(a <= RHO, u, torch.sign(u) * (RHO + (1 - RHO) * k))
def pc_t(x, lam, alpha, b):
    w = S_ * torch.tanh(b * x / S_); ec = math.exp(C_)
    ce = torch.where(w <= C_, torch.exp(torch.clamp(w, max=C_)) - 1, ec - 1 + ec * (w - C_)); return lam * torch.where(x > 0, ce, alpha * ce)
def act_t(n):
    n = base(n)
    if n == 'tat': return lambda x: TAT['scale'] * torch.where(x > 0, x, TAT['neg'] * x)
    if n == 'selu': return torch.nn.functional.selu
    if n == 'kL2B': return lambda x: hull_t(pc_t(x, **KL2B), B_M)
    if n == 'LB': return lambda x: hull_t(LB['beta'] * x, LB['M'])
    raise ValueError(n)
def np_act(n, x):
    n = base(n)
    if n == 'tat': return TAT['scale'] * np.where(x > 0, x, TAT['neg'] * x)
    if n == 'selu': l, a = 1.0507009873554805, 1.6732632423543772; return l * np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
    def hull(y, M): u = y / M; a = np.abs(u); return M * np.where(a <= RHO, u, np.sign(u) * (RHO + (1 - RHO) * np.tanh((a - RHO) / (1 - RHO))))
    if n == 'LB': return hull(LB['beta'] * x, LB['M'])
    w = S_ * np.tanh(KL2B['b'] * x / S_); ec = math.exp(C_); ce = np.where(w <= C_, np.exp(np.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_))
    return hull(KL2B['lam'] * np.where(x > 0, ce, KL2B['alpha'] * ce), B_M)
def gain(n):   # g = 1/sqrt(E[f(Z)^2]), Gauss-Hermite 150 Knoten (wie b10_run.gain)
    X, W = np.polynomial.hermite_e.hermegauss(150); W = W / W.sum(); y = np_act(n, X); return float(1 / np.sqrt(np.sum(W * y * y)))

# --- Daten (wie b10_run.load/data_for) ---------------------------------------------------
_C = {}
def load():
    if 'c10' in _C: return _C['c10']
    def rd(fs):
        a = np.concatenate([np.fromfile(DATA / 'cifar10' / 'cifar-10-batches-bin' / f, dtype=np.uint8).reshape(-1, 3073) for f in fs])
        return a[:, 1:].astype(np.float32) / 255, a[:, 0].astype(np.int64)
    xtr, ytr = rd([f'data_batch_{i}.bin' for i in range(1, 6)]); xte, yte = rd(['test_batch.bin'])
    xv, yv = xtr[-NVAL:], ytr[-NVAL:]; xtr, ytr = xtr[:-NVAL], ytr[:-NVAL]
    # Normierung mit den exakten float32-Werten der MLX-Studien (numpy 2.5.4). Aeltere numpy-Versionen summieren float32 ungenauer
    # (2.0.2: 0.47336864 statt 0.47336701), deshalb fest statt neu gerechnet; geprueft: entspricht xtr.mean()/xtr.std() unter numpy 2.5.4.
    m, s = np.float32(0.47336700558662415), np.float32(0.2517273426055908); T = lambda a: ((a - m) / s).astype(np.float32)
    _C['c10'] = (T(xtr), ytr, T(xv), yv, T(xte), yte); return _C['c10']
def pex(a): return (a / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1e-12)).astype(np.float32)   # wie b10_run.pex
def data_for(n):   # alle Formen hier im TAT-Rezept: Normierung je Beispiel
    if 'pex' not in _C: xtr, ytr, xv, yv, xte, yte = load(); _C['pex'] = (pex(xtr), ytr, pex(xv), yv, pex(xte), yte)
    return _C['pex']

# --- Netz (Init wie b10_run.Net mit Rezept: SUO-orthogonal x Gain, TAT ohne Gain, Biases 0) -----
def suo(r, b, a):
    M = r.normal(size=(max(a, b), min(a, b))); Q, R_ = np.linalg.qr(M); Q = Q * np.sign(np.diag(R_))
    W = Q if b >= a else Q.T
    return (W * max(math.sqrt(b / a), 1.0)).astype(np.float32)
class Net(torch.nn.Module):
    def __init__(self, n, seed, din=3072):
        super().__init__()
        r = np.random.default_rng(seed); tat = base(n) == 'tat'; g = 1.0 if tat else gain(n); dims = [din] + [WIDTH] * DEPTH
        self.hid = torch.nn.ModuleList([torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]); self.out = torch.nn.Linear(WIDTH, 10)
        with torch.no_grad():
            for l, (a, b) in zip(self.hid, zip(dims[:-1], dims[1:])):
                W = suo(r, b, a) if tat else suo(r, b, a) * np.float32(g); l.weight.copy_(torch.from_numpy(W)); l.bias.zero_()
            self.out.weight.copy_(torch.from_numpy(suo(r, 10, WIDTH))); self.out.bias.zero_()
        self.act = act_t(n)
    def layers(self): return list(self.hid) + [self.out]
    def forward(self, x):
        for l in self.hid: x = self.act(l(x))
        return self.out(x)

@torch.no_grad()
def evaluate(fwd, x, y):
    ac = 0
    for i in range(0, len(x), 2500):
        o = fwd(torch.from_numpy(x[i:i + 2500]).to(DEV)); ac += int((o.argmax(1).cpu() == torch.from_numpy(y[i:i + 2500])).sum())
    return ac / len(x)

# --- Quantisierung (wie b10_run) -----------------------------------------------------------
def q_int(a, bits, scale): qm = 2 ** (bits - 1) - 1; return torch.clamp(torch.round(a / scale), -qm, qm) * scale
def qw(W, kind):
    am = torch.clamp(W.abs().amax(dim=1, keepdim=True), min=1e-12)
    return q_int(W, 8, am / 127) if kind == 'i8' else q_int(W, 4, am / 7)
def qa(a, kind, rng): return {'i8': lambda: q_int(a, 8, rng / 127), 'i4': lambda: q_int(a, 4, rng / 7), 'i3': lambda: q_int(a, 3, rng / 3)}[kind]()
def mse_clip(a, bits):
    qm = 2 ** (bits - 1) - 1; x = torch.from_numpy(a.astype(np.float32)).to(DEV); amax = float(x.abs().max())
    if amax == 0: return 1e-12
    sc = torch.tensor([amax * k / 100 / qm for k in range(1, 101)], dtype=torch.float32, device=DEV)[:, None]
    E = torch.cat([((torch.clamp(torch.round(x[None, :] / s), -qm, qm) * s - x[None, :]) ** 2).mean(1) for s in sc.split(25)])
    return amax * (int(E.argmin()) + 1) / 100
def qa_asym(a, bits, lohi):
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 2 ** bits - 1; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc)
    return (torch.clamp(torch.round(a / sc) + zp, 0, n) - zp) * sc
def mse_clip_asym(a, bits):
    lo0, hi0 = float(min(a.min(), 0.0)), float(max(a.max(), 0.0)); x = torch.from_numpy(a.astype(np.float32)).to(DEV); n = 2 ** bits - 1
    if hi0 - lo0 == 0: return (0.0, 1e-12)
    ks = np.arange(1, 101); sc = np.array([(hi0 - lo0) * k / 100 / n for k in ks], dtype=np.float32); zp = np.round(-lo0 * ks / 100 / sc).astype(np.float32)
    S = torch.from_numpy(sc).to(DEV)[:, None]; Z = torch.from_numpy(zp).to(DEV)[:, None]
    E = torch.cat([(((torch.clamp(torch.round(x[None, :] / s) + z, 0, n) - z) * s - x[None, :]) ** 2).mean(1) for s, z in zip(S.split(25), Z.split(25))])
    k = int(E.argmin()) + 1; return (lo0 * k / 100, hi0 * k / 100)
@torch.no_grad()
def inputs(m, xc):
    h = torch.from_numpy(xc).to(DEV); out = []
    for i, l in enumerate(m.layers()):
        out.append(h.detach().cpu().numpy().astype(np.float64).ravel())
        if l is not m.out: h = m.act(l(h))
    return out
def calib(A):
    R = {k: [] for k in ['max', 'p999', 'p9999', 'mse4', 'mse3']}
    for a in A:
        aa = np.abs(a); R['max'].append(float(aa.max())); R['p999'].append(float(np.quantile(aa, 0.999))); R['p9999'].append(float(np.quantile(aa, 0.9999)))
        R['mse4'].append(mse_clip(a, 4)); R['mse3'].append(mse_clip(a, 3))
    return R
def calib_asym(A):
    R = {k: [] for k in ['max', 'p999', 'p9999', 'mse4']}
    for a in A:
        R['max'].append((float(a.min()), float(a.max()))); R['p999'].append((float(np.quantile(a, 0.0005)), float(np.quantile(a, 0.9995))))
        R['p9999'].append((float(np.quantile(a, 0.00005)), float(np.quantile(a, 0.99995)))); R['mse4'].append(mse_clip_asym(a, 4))
    return R
def qforward(m, ak, rngs, asym=False, bits=4):
    Ws = [(qw(l.weight, 'i8'), l.bias) for l in m.layers()]
    def f(x):
        h = x
        for i, (W, b) in enumerate(Ws):
            h = (qa_asym(h, bits, rngs[i]) if asym else qa(h, ak, rngs[i])) @ W.T + b
            if i < len(Ws) - 1: h = m.act(h)
        return h
    return f
@torch.no_grad()
def measure(m, D):
    xtr, ytr, xv, yv, xte, yte = D; o = {}
    o['val_err_fp32'] = 1 - evaluate(m, xv, yv); o['test_err_fp32'] = 1 - evaluate(m, xte, yte)
    A = inputs(m, xtr[:NCAL]); R = calib(A); RA = calib_asym(A)
    o['act_range_max_median'] = float(np.median(R['max'][1:])); o['act_range_p9999_median'] = float(np.median(R['p9999'][1:]))
    for name, ak, cals, asym, bits, RR in (('w8a4_best', 'i4', CALS4, False, 4, R), ('w8a4a_best', 'i4', CALS4, True, 4, RA), ('w8a3_best', 'i3', CALS3, False, 3, R)):
        ce = {c: 1 - evaluate(qforward(m, ak, RR[c], asym, bits), xtr[:NCAL], ytr[:NCAL]) for c in cals}
        bc = min(cals, key=lambda k: (ce[k], cals.index(k))); o[f'test_err_{name}'] = 1 - evaluate(qforward(m, ak, RR[bc], asym, bits), xte, yte); o[f'{name}_cal'] = bc
    return o

# --- Lauf -------------------------------------------------------------------------------
MET = ['val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median', 'act_range_p9999_median']
COLS = ['run_id', 'act', 'lr', 'wd', 'seed', 'source', 'status', 'train_loss_last', 'device', 'torch', 'sec'] + MET + [f'c{k}_{x}' for k in CHECKS for x in MET]
def lr_at(peak, k): return peak * 0.5 * (1 + math.cos(math.pi * min(k, STEPS) / STEPS))   # MLX cosine_decay: k-tes Update nutzt schedule(k-1)
def rid(c): return f"c6_{c['act']}_lr{c['lr']:g}_wd{c['wd']:g}_s{c['seed']}"
def execute(c, wid, steps=None, checks=None):
    steps = steps or STEPS; checks = CHECKS if checks is None else checks; t0 = time.time(); D = data_for(c['act']); xtr, ytr = D[0], D[1]
    torch.manual_seed(c['seed']); m = Net(c['act'], c['seed']).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=c['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=c['wd'])
    r = np.random.default_rng(c['seed'] + 777); perm = r.permutation(len(xtr)); pos = 0; status = 'ok'; ck = {}; last = float('nan')
    xtr_d = torch.from_numpy(xtr).to(DEV); ytr_d = torch.from_numpy(ytr).to(DEV)
    for s in range(1, steps + 1):
        if pos + BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = torch.from_numpy(perm[pos:pos + BATCH]).to(DEV); pos += BATCH
        for g in opt.param_groups: g['lr'] = c['lr'] * 0.5 * (1 + math.cos(math.pi * (s - 1) / steps))   # = lr_at(peak, s-1) bei steps = STEPS
        loss = torch.nn.functional.cross_entropy(m(xtr_d[b]), ytr_d[b]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 43 == 0 or s == steps:
            last = loss.item()
            if not math.isfinite(last): status = 'nonfinite_loss'; break
        if s in checks: m.eval(); ck.update({f'c{s}_{k}': v for k, v in measure(m, D).items() if k in MET}); m.train()
    m.eval(); out = dict(run_id=rid(c), **c, status=status, train_loss_last=last, device=DEV, torch=torch.__version__, **ck)
    if status == 'ok': out.update({k: v for k, v in measure(m, D).items() if k in MET})
    out['sec'] = round(time.time() - t0, 1)
    p = OUTDIR / 'workers' / f'w{wid}.csv'; p.parent.mkdir(parents=True, exist_ok=True); new = not p.exists()
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS[:COLS.index(MET[-1]) + 1] + [f'c{k}_{x}' for k in checks for x in MET], extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(out)
    return out
def configs(): return [dict(act=a, lr=lr, wd=wd, seed=s, source=src) for s in SEEDS for a, lr, wd, src in RUNS]
def finished():
    d = set()
    for f in (OUTDIR / 'workers').glob('w*.csv'):
        for r in csv.DictReader(open(f)): d.add(r['run_id'])
    return d

def setup(device):
    global DEV
    DEV = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # echtes float32, kein TF32 (Ampere/Ada)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); torch.use_deterministic_algorithms(True)
def selfcheck():
    X = np.linspace(-30, 30, 60001)
    o = dict(device=DEV, torch=torch.__version__, cuda=torch.cuda.is_available(), n_runs=len(configs()), gains={n: round(gain(n), 6) for n in ('kL2B_tr', 'LB_tr', 'selu_tr')})
    o['form_vs_float64'] = {n: float(np.abs(act_t(n)(torch.from_numpy(X.astype(np.float32))).numpy() - np_act(n, X.astype(np.float32).astype(np.float64))).max()) for n in ('kL2B_tr', 'LB_tr', 'tat', 'selu_tr')}
    o['lr_schedule'] = [lr_at(1e-4, 0), lr_at(1e-4, STEPS // 2), lr_at(1e-4, STEPS - 1)]
    o['data_found'] = (DATA / 'cifar10' / 'cifar-10-batches-bin' / 'test_batch.bin').exists()
    o['ok'] = o['n_runs'] == 60 and max(o['form_vs_float64'].values()) < 1e-5 and o['data_found']
    return o

def main():
    global OUTDIR
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--device', default=None); a = p.parse_args()
    setup(a.device)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    if a.smoke:   # 3 kurze Laeufe mit Zwischenmessung, eigener Ordner
        OUTDIR = ROOT / 'smoke'
        for c in configs()[:3]: o = execute(c, 0, steps=40, checks=(20,)); print('smoke', o['run_id'], o['status'], round(o.get('test_err_fp32', -1), 4), o['sec'], flush=True)
        return
    skip = finished()
    for i, c in enumerate(configs()):
        if i % a.workers != a.worker or rid(c) in skip: continue
        o = execute(c, a.worker); print('done', rid(c), o['status'], o['sec'], datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
