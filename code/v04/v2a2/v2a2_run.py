# V2a groß / V2a2 (privat, mit Claude, praeregistriert in V2a2_Praeregistrierung.md): BPreCap gegen SELU und gegen TAT im CNN ohne
# Normierung und ohne Residuen (L = 50, 32 Kanaele, Delta-orthogonal), CIFAR-10, 4 Bit PTQ. PreCap, Tiefenfamilie und Huelle: Ideen Aarons.
# Netz, Init, Daten, Quantisierung wie studien/v2a_entwurf/v2a_pilot4.py bzw. studien/v2a1/v2a1_run.py; neu gegenueber V2a-1:
# L = 50, TAT als dritte Form, Regime wie B10 (AdamW mit entkoppeltem WD, Kosinus auf 0 ohne Anlauf, 3440 Schritte), LR x WD je Form
# getunt (2 Tuning-Seeds, Val am Ende), fruehes Stoppen am kleinsten Val-Fehler (8 Messpunkte), Kalibrierer p99999 und mse4 (asymmetrisch),
# fuer TAT zusaetzlich PTQ nach Cross-Layer Equalization (CLE, Nagel 2019; nachgetragen vor dem Freeze nach der Recherche vom 16.11.).
# Aufruf: --selfcheck | --smoke | --worker i --workers n
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
ROOT = Path(__file__).resolve().parent; OUTDIR = ROOT
DATA = Path(os.environ.get('PRECAP_DATA', '/Users/aaronschmitt/PyCharmMiscProject/data')) / 'cifar10' / 'cifar-10-batches-bin'
KC = json.load(open(ROOT / 'kL2B_d50.json')); TAT = json.load(open(ROOT / 'v2a_tat_params.json'))['50']; S_, C_, RHO, M = 1.5, 1.0, 0.8, 3.0
L, CH, BATCH, STEPS, NVAL, NCAL, NSUB, NPTS = 50, 32, 128, 3440, 5000, 1000, 2_000_000, 8
ACTS = ('bprecap', 'tat', 'selu')
LR_GRID = {'bprecap': (1e-5, 3e-5, 1e-4), 'selu': (1e-5, 3e-5, 1e-4), 'tat': (3e-5, 1e-4, 3e-4)}   # je Form um die beste Pilot-LR (Pilot 4, L = 50)
WD_GRID = (0.0, 3.0)                     # MLX AdamW: w <- w (1 - lr wd); 3.0 wie B10 (schrumpft ohne Gradient merklich), 0 ohne WD
TUNE_SEEDS = (42000, 42001); SEEDS = tuple(range(42350, 42362))
CALS4 = ('max', 'p999', 'p9999', 'p99999', 'mse4'); MODES = ('fp32', 'w8a8') + tuple(f'w8a4a_{k}' for k in CALS4)
QS = {'p999': 0.0005, 'p9999': 0.00005, 'p99999': 0.000005}
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
    if name == 'tat':
        a, s = TAT['neg'], TAT['scale']; return lambda x: s * mx.where(x > 0, x, a * x)
    lam, al, b = KC['lam'], KC['alpha'], KC['b']; ec = math.exp(C_)
    def f(x):
        w = S_ * mx.tanh(b * x / S_); ce = mx.where(w <= C_, mx.exp(mx.minimum(w, C_)) - 1, ec - 1 + ec * (w - C_)); y = lam * mx.where(x > 0, ce, al * ce)
        u = y / M; a = mx.abs(u); return M * mx.where(a <= RHO, u, mx.sign(u) * (RHO + (1 - RHO) * mx.tanh((a - RHO) / (1 - RHO))))
    return f
def delta_orth(r, cin, cout):
    A = r.normal(size=(max(cin, cout), max(cin, cout))); Q, R_ = np.linalg.qr(A); Q = Q * np.sign(np.diag(R_))
    w = np.zeros((cout, 3, 3, cin), dtype=np.float32); w[:, 1, 1, :] = Q[:cout, :cin] * (math.sqrt(cout / cin) if cin < cout else 1.0); return w
class CNN(nn.Module):
    def __init__(self, act, seed, Ld=None):
        super().__init__(); Ld = Ld or L; r = np.random.default_rng(seed); dims = [3] + [CH] * Ld
        self.convs = [nn.Conv2d(a, b, 3, padding=1) for a, b in zip(dims[:-1], dims[1:])]
        for c, (a, b) in zip(self.convs, zip(dims[:-1], dims[1:])): c.weight = mx.array(delta_orth(r, a, b)); c.bias = mx.zeros((b,))
        self.out = nn.Linear(CH, 10); self.out.weight = mx.array(r.normal(0, 1 / math.sqrt(CH), size=(10, CH)).astype(np.float32)); self.out.bias = mx.zeros((10,))
        self.act = act
    def __call__(self, x):
        for c in self.convs: x = self.act(c(x))
        return self.out(mx.mean(x, axis=(1, 2)))
def err(f, x, y):
    ac = 0
    for i in range(0, len(x), 500): ac += int((mx.argmax(f(mx.array(x[i:i + 500])), 1) == mx.array(y[i:i + 500])).sum())
    return 1 - ac / len(x)
def train(name, lr, wd, sched, seed, steps):
    # Kontrollpfad (sched 'const', wd 0) = Pilot 4 (Adam, konstante LR). Messpunkte alle steps/8 auf dem Val-Satz (fp32); die Messung
    # veraendert den Trainingszustand nicht. Rueckgabe: Netz am Ende, Parameter am Punkt des kleinsten Val-Fehlers (Gleichstand: frueher).
    xtr, ytr, xv, yv = load()[:4]
    mx.random.seed(seed); m = CNN(make_act(name), seed); mx.eval(m.parameters())
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps) if sched == 'cos' else lr, weight_decay=wd)
    lf = lambda mm, a, b: nn.losses.cross_entropy(mm(a), b, reduction='mean')
    vg = nn.value_and_grad(m, lf); state = [m.state, opt.state]
    def step(a, b):
        l, g = vg(m, a, b); opt.update(m, g); return l
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(seed + 777); perm = r.permutation(len(xtr)); pos = 0; ok = True; tail = []
    pts = sorted({max(1, round(steps * k / NPTS)) for k in range(1, NPTS + 1)}); kurve = []; best = None
    for s in range(1, steps + 1):
        if pos + 128 > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + 128]; pos += 128; l = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state); lf_ = float(l)
        if not math.isfinite(lf_): ok = False; break
        if s > steps - 50: tail.append(lf_)
        if s in pts:
            e = err(m, xv, yv); kurve.append((s, e))
            if best is None or e < best[1]: best = (s, e, dict(tree_flatten(m.parameters())))
    return m, ok, (float(np.mean(tail)) if tail else float('nan')), kurve, best
def qw(W):
    ax = tuple(range(1, W.ndim)); am = mx.maximum(mx.max(mx.abs(W), axis=ax, keepdims=True), 1e-12); sc = am / 127
    return mx.clip(mx.round(W / sc), -127, 127) * sc
def qa_sym(a, bits, rng):
    qm = 2 ** (bits - 1) - 1; sc = max(rng, 1e-12) / qm; return mx.clip(mx.round(a / sc), -qm, qm) * sc
def qa_asym(a, bits, lohi):
    lo, hi = lohi; lo = min(lo, 0.0); hi = max(hi, 0.0); n = 2 ** bits - 1; sc = max(hi - lo, 1e-12) / n; zp = round(-lo / sc)
    return (mx.clip(mx.round(a / sc) + zp, 0, n) - zp) * sc
def mse_asym(s, lohi, bits=4):   # (lo, hi) = k/100 * (min, max), k = 1..100, mit kleinstem MSE der asymmetrischen Quantisierung auf der Stichprobe
    lo, hi = min(lohi[0], 0.0), max(lohi[1], 0.0); x = mx.array(s.astype(np.float32)); n = 2 ** bits - 1; E = []
    for ks in (range(1, 26), range(26, 51), range(51, 76), range(76, 101)):
        k = np.array(list(ks), dtype=np.float64) / 100; sc = np.maximum((hi - lo) * k, 1e-12) / n; zp = np.round(-lo * k / sc)
        scm = mx.array(sc.astype(np.float32))[:, None]; zpm = mx.array(zp.astype(np.float32))[:, None]
        q = (mx.clip(mx.round(x[None, :] / scm) + zpm, 0, n) - zpm) * scm; E.append(mx.mean((q - x[None, :]) ** 2, axis=1)); mx.eval(E[-1])
    E = np.array(mx.concatenate(E)); k = (int(np.argmin(E)) + 1) / 100; return (lo * k, hi * k)
def calib(m, xc, seed):
    # Eingaenge aller Faltungen und der Linearschicht auf den Kalibrierbildern, Schicht fuer Schicht (Speicher), in Teilen zu 250 Bildern
    # wie v2a_pilot4.layer_inputs (gleiche Rechnung je Teil). Zufallsstichprobe je Tensor wie im Pilot (ein Generator, gleiche Reihenfolge);
    # p999, p99999 und mse4 kommen aus derselben Stichprobe und aendern max_abs, max und p9999 nicht.
    r = np.random.default_rng(seed); R = {'max_abs': [], 'max': [], 'p999': [], 'p9999': [], 'p99999': [], 'mse4': []}
    hs = [mx.array(xc[i:i + 250]) for i in range(0, len(xc), 250)]
    def stats(parts):
        a = np.concatenate([np.array(t, dtype=np.float32).ravel() for t in parts])
        R['max_abs'].append(float(np.abs(a).max())); R['max'].append((float(a.min()), float(a.max())))
        s = a if len(a) <= NSUB else a[r.choice(len(a), NSUB, replace=False)]
        for k, q in QS.items(): R[k].append((float(np.quantile(s, q)), float(np.quantile(s, 1 - q))))
        R['mse4'].append(mse_asym(s, R['max'][-1]))
    for c in m.convs:
        stats(hs); hs = [m.act(c(h)) for h in hs]; mx.eval(hs)
    stats([mx.mean(h, axis=(1, 2)) for h in hs]); return R
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
def cle(m, sweeps=30):
    # Cross-Layer Equalization (Nagel et al. 2019, DFQ) fuer TAT: Leaky ReLU mit Skala ist positiv homogen, f(x/s) = f(x)/s fuer s > 0.
    # Je Paar (Faltung k, naechste Faltung bzw. Linear nach dem globalen Mittel): Ausgangskanal i von k durch s_i teilen (Gewicht und Bias),
    # Eingangskanal i der naechsten Schicht mit s_i malnehmen, s_i = sqrt(r1_i / r2_i). Funktion in fp32 unveraendert (bis auf Rundung).
    Ws = [np.array(c.weight, dtype=np.float64) for c in m.convs] + [np.array(m.out.weight, dtype=np.float64)]
    Bs = [np.array(c.bias, dtype=np.float64) for c in m.convs]
    for _ in range(sweeps):
        for k in range(len(m.convs)):
            r1 = np.abs(Ws[k]).max(axis=(1, 2, 3)); nxt = Ws[k + 1]
            r2 = np.abs(nxt).max(axis=(0, 1, 2)) if nxt.ndim == 4 else np.abs(nxt).max(axis=0)
            s = np.where((r1 > 0) & (r2 > 0), np.sqrt(r1 / np.maximum(r2, 1e-30)), 1.0)
            Ws[k] = Ws[k] / s[:, None, None, None]; Bs[k] = Bs[k] / s
            Ws[k + 1] = nxt * (s[None, None, None, :] if nxt.ndim == 4 else s[None, :])
    m2 = CNN(m.act, 0)
    for c, W, b in zip(m2.convs, Ws, Bs): c.weight = mx.array(W.astype(np.float32)); c.bias = mx.array(b.astype(np.float32))
    m2.out.weight = mx.array(Ws[-1].astype(np.float32)); m2.out.bias = m.out.bias; mx.eval(m2.parameters()); return m2
def ptq(m, seed, pre):   # alle Messgroessen eines Netzzustands, Praefix pre ('es_' oder 'end_')
    xtr, ytr, _, _, xte, yte = load(); R = calib(m, xtr[:NCAL], seed); o = {}
    for k in MODES: o[f'{pre}test_err_{k}'] = err(qforward(m, k, R), xte, yte)
    ce = {cal: err(qforward(m, f'w8a4a_{cal}', R), xtr[:NCAL], ytr[:NCAL]) for cal in CALS4}
    bc = min(CALS4, key=lambda k: (ce[k], CALS4.index(k))); o[f'{pre}w8a4a_best_cal'] = bc; o[f'{pre}test_err_w8a4a_best'] = o[f'{pre}test_err_w8a4a_{bc}']
    mid = slice(1, -1)   # nur die inneren Faltungseingaenge (ohne Bild und ohne Linear-Eingang), wie V2a-1
    o[f'{pre}act_lo_max_median'] = float(np.median([l_ for l_, _ in R['max'][mid]])); o[f'{pre}act_hi_max_median'] = float(np.median([h for _, h in R['max'][mid]]))
    o[f'{pre}act_hi_p9999_median'] = float(np.median([h for _, h in R['p9999'][mid]])); o[f'{pre}act_absmax_max'] = float(max(R['max_abs'][mid]))
    return o
PT = ['test_err_' + k for k in MODES + ('w8a4a_best',)] + ['w8a4a_best_cal', 'act_lo_max_median', 'act_hi_max_median', 'act_hi_p9999_median', 'act_absmax_max']
COLS = ['run_id', 'stage', 'act', 'lr', 'wd', 'sched', 'seed', 'steps', 'status', 'train_loss_last50', 'val_err_end', 'val_err_es', 'es_step', 'val_kurve'] + \
       ['es_' + c for c in PT] + ['end_' + c for c in PT] + ['es_cle_' + c for c in PT] + ['es_cle_fp32_delta', 'sec', 'timestamp']
def rid(c): return f"{c['stage']}_{c['act']}_lr{c['lr']:g}_wd{c['wd']:g}_s{c['seed']}"
def run_one(c):
    t0 = time.time(); steps = c.get('steps', STEPS)
    m, ok, tl, kurve, best = train(c['act'], c['lr'], c['wd'], c.get('sched', 'cos'), c['seed'], steps)
    o = dict(run_id=rid(c), stage=c['stage'], act=c['act'], lr=c['lr'], wd=c['wd'], sched=c.get('sched', 'cos'), seed=c['seed'], steps=steps,
             status='ok' if ok else 'fail', train_loss_last50=tl, val_kurve=';'.join(f'{s}:{e:.4f}' for s, e in kurve))
    if ok:
        o['val_err_end'] = kurve[-1][1]; o['val_err_es'] = best[1]; o['es_step'] = best[0]
        if c['stage'] == 'confirm':
            o.update(ptq(m, c['seed'], 'end_')); m.update(tree_unflatten(list(best[2].items()))); mx.eval(m.parameters()); o.update(ptq(m, c['seed'], 'es_'))
            if c['act'] == 'tat': mc = cle(m); o.update(ptq(mc, c['seed'], 'es_cle_')); o['es_cle_fp32_delta'] = o['es_cle_test_err_fp32'] - o['es_test_err_fp32']
    o['sec'] = round(time.time() - t0, 1); o['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S'); return o
def tune_cfgs(seeds=TUNE_SEEDS, grid=None):
    grid = grid or LR_GRID; out = []
    for s in seeds:
        for a in ACTS:
            for lr in grid[a]:
                for wd in WD_GRID: out.append(dict(stage='tune', act=a, lr=lr, wd=wd, seed=s))
    return out
def read_rows(d):
    rows = []
    for f in sorted(d.glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    return rows
def select(rows, grid=None, seeds=TUNE_SEEDS):
    # wie B10: hoechste mediane Val-Genauigkeit am Ende ueber die Tuning-Seeds; Ausfall/fehlend = 0,1; Gleichstand: kleinere LR, dann kleinerer WD
    grid = grid or LR_GRID; D = {(r['act'], float(r['lr']), float(r['wd']), int(r['seed'])): r for r in rows}; out = {}
    for a in ACTS:
        cand = []
        for lr in grid[a]:
            for wd in WD_GRID:
                acc = []
                for s in seeds:
                    r = D.get((a, lr, wd, s)); acc.append(1 - float(r['val_err_end']) if r is not None and r['status'] == 'ok' and r.get('val_err_end', '') != '' else 0.1)
                cand.append((float(np.median(acc)), -lr, -wd, lr, wd))
        b = max(cand); out[a] = dict(lr=b[3], wd=b[4], med_val_acc=b[0], lr_at_edge=b[3] in (grid[a][0], grid[a][-1]), wd_at_edge=b[4] == WD_GRID[-1],
                                     all_failed=all(x[0] <= 0.1 for x in cand))
    return out
def confirm_cfgs(sel, seeds=SEEDS):   # je Seed alle drei Formen, Reihenfolge je Seed rotiert
    out = []
    for j, s in enumerate(seeds):
        for a in ACTS[j % 3:] + ACTS[:j % 3]: out.append(dict(stage='confirm', act=a, lr=sel[a]['lr'], wd=sel[a]['wd'], seed=s))
    return out
def write(p, o):
    new = not p.exists(); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)
def work(wid, nw, outdir, tcfg, scfg_fn, extra=None):
    td = outdir / 'tune' / 'workers'; cd = outdir / 'workers'; extra = extra or {}
    done = {r['run_id'] for r in read_rows(td)} if td.exists() else set()
    for i, c in enumerate(tcfg):
        if i % nw != wid or rid(c) in done: continue
        o = run_one(dict(c, **extra)); write(td / f'w{wid}.csv', o)
        print(time.strftime('%H:%M'), o['run_id'], o['status'], o.get('val_err_end'), o.get('sec'), flush=True)
    want = {rid(c) for c in tcfg}
    while not want <= {r['run_id'] for r in read_rows(td)}:
        time.sleep(60)
    sel = scfg_fn(read_rows(td)); sp = outdir / 'tune' / 'hp_sel.json'
    if wid == 0: json.dump(sel, open(sp, 'w'), indent=1)
    print('Auswahl', json.dumps(sel), flush=True)
    if any(v['all_failed'] for v in sel.values()):
        print('ABBRUCH: eine Form bricht bei jeder LR ein (Regel Anlauf, neue Version der Praeregistrierung)', flush=True); return
    ccfg = confirm_cfgs(sel, extra.get('_seeds', SEEDS)); done = {r['run_id'] for r in read_rows(cd)} if cd.exists() else set()
    for i, c in enumerate(ccfg):
        if i % nw != wid or rid(c) in done: continue
        o = run_one(dict(c, **{k: v for k, v in extra.items() if not k.startswith('_')})); write(cd / f'w{wid}.csv', o)
        print(time.strftime('%H:%M'), o['run_id'], o['status'], o.get('es_test_err_fp32'), o.get('es_test_err_w8a4a_best'), o.get('sec'), flush=True)
    print('Worker fertig', time.strftime('%H:%M'), flush=True)
def selfcheck():
    out = dict(L=L, C=CH, steps=STEPS, n_tune=len(tune_cfgs()), lr_grid=LR_GRID, wd_grid=WD_GRID, chi1=KC['chi1'], tat=TAT)
    X, Wq = np.polynomial.hermite_e.hermegauss(150); Wq = Wq / Wq.sum(); x = mx.array(X.astype(np.float32))
    for a in ACTS:
        y = np.array(make_act(a)(x), dtype=np.float64); out[f'Ef2_{a}'] = round(float(np.sum(Wq * y * y)), 4); out[f'Ef_{a}'] = round(float(np.sum(Wq * y)), 4)
    w = delta_orth(np.random.default_rng(0), CH, CH)[:, 1, 1, :]; out['delta_orth_ok'] = bool(np.abs(w @ w.T - np.eye(CH)).max() < 1e-5)
    out['Ef2_ok'] = all(abs(out[f'Ef2_{a}'] - 1) < 0.01 for a in ACTS)
    used = set(range(42000, 42350)) - set(TUNE_SEEDS) | set(range(42640, 42732))   # bisher genutzte Bereiche (Suche in studien/, 16.11.)
    out['seeds_new'] = not (set(SEEDS) & used) and not (set(SEEDS) & set(TUNE_SEEDS))
    fake = {a: dict(lr=LR_GRID[a][1], wd=0.0) for a in ACTS}; cc = confirm_cfgs(fake); out['n_confirm'] = len(cc)
    out['order_rotates'] = len({tuple(c['act'] for c in cc[i:i + 3]) for i in range(0, len(cc), 3)}) == 3
    print(json.dumps(out, indent=1)); return out
def main():
    p = argparse.ArgumentParser(); p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true');
    p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1); a = p.parse_args()
    print('Geraet:', mx.default_device(), flush=True)
    if a.selfcheck: selfcheck(); return
    if a.smoke:   # 20 Schritte, ein Tuning-Seed, je Form die erste LR und beide WD, dann Auswahl und Bestaetigung mit einem Seed
        g = {k: v[:1] for k, v in LR_GRID.items()}
        work(0, 1, ROOT / 'smoke', tune_cfgs(TUNE_SEEDS[:1], g), lambda rows: select(rows, g, TUNE_SEEDS[:1]), dict(steps=20, _seeds=SEEDS[:1])); return
    work(a.worker, a.workers, ROOT, tune_cfgs(), select)
if __name__ == '__main__': main()
