# V3 (V3_Praeregistrierung.md, privat, mit Claude): QAT als Obergrenze. Feste weiche Huelle (BPreCap) mit PTQ gegen TAT mit LSQ- bzw.
# PACT-QAT, beide Formen in allen drei Varianten. Laedt studien/v1/v1_run.py unveraendert als Modul (Netz, Daten, Init, PTQ-Protokoll).
# Ablauf je (Form, Seed): Basisnetz exakt wie die V1-Bestaetigung (Anlauf 5 %, Kosinus, V1-Wahl), Pruefung bitgleich gegen die V1-Zeile;
# danach drei Feintunings ueber FT_STEPS = 1376 Schritte (20 %) aus derselben Basis, gleiche Batch-Reihenfolge:
#   FT   ohne Quantisierung im Training, danach PTQ wie V1 (w8a4a_best)
#   LSQ  QAT: Eingaenge aller Linearschichten 4 Bit asymmetrisch mit lernbarer Schrittweite s und lernbarem Versatz b (LSQ+, Esser et al. 2020,
#        Bhalgat et al. 2020), Gradientenskalierung g = 1/sqrt(N*Qp); Gewichte int8 je Ausgangskanal (STE)
#   PACT QAT: lernbare Grenzen [lo, hi] je Schicht, Gradient nur aus geclippten Werten (Choi et al. 2018), L2-Strafe auf lo und hi, Gewichte wie LSQ
# Start der Quantisierer: [lo, hi] der PTQ-Kalibrierung nach der V1-Wahlregel am Basisnetz. Gewichte: AdamW mit 0,1 x LR-Spitze, Kosinus auf 0,
# gleicher WD; Quantisierer: eigener Adam ohne WD mit LR_Q (getunt auf den Tuning-Seeds), Kosinus auf 0.
# Aufruf: python v3_qat.py --selfcheck | --stage tune|confirm --worker W --workers N [--smoke]
import argparse, csv, importlib.util, json, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent; ST = ROOT.parent
spec = importlib.util.spec_from_file_location('v1_run', ST / 'v1' / 'v1_run.py'); V = importlib.util.module_from_spec(spec); sys.modules['v1_run'] = V; spec.loader.exec_module(V)

FORMS = {'kL2B_tr': (1e-4, 0.3), 'tat': (2e-4, 0.0)}   # V1-Wahl (LR-Spitze, WD)
METHODS = ('FT', 'LSQ', 'PACT')
FT_STEPS = 1376; FT_LR_FAC = 0.1; FT_SEED_OFF = 999; BITS = 4; NQ = 2 ** BITS - 1
LRQ_GRID = [1e-4, 1e-3, 1e-2]; PACT_L2 = 2e-4
TUNE_SEEDS = [42000, 42001]; SEEDS = list(range(42168, 42180))
COLS = ['run_id', 'stage', 'act', 'method', 'lrq', 'seed', 'status', 'base_valid', 'train_loss_last50', 'val_err_fp32', 'test_err_fp32',
        'val_err_qat', 'test_err_qat', 'test_err_w8a4a_best', 'w8a4a_best_cal', 'range_med_start', 'range_med_end', 'range_move_med', 'sec', 'timestamp']
OUTDIR = ROOT

def ref_row(act, seed, stage):
    d = ST / 'v1' / ('tune' if stage == 'tune' else '') / 'workers'; lr, wd = FORMS[act]
    for f in sorted(d.glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if (r['act'], int(r['seed']), float(r['lr']), float(r['wd']), r['sched']) == (act, seed, lr, wd, 'wcos') and int(r['depth']) == 100: return r
    raise KeyError((act, seed, stage))

def sched_base(lr):   # wie V1: linearer Anlauf ueber WARM Schritte, dann Kosinus auf 0
    w_ = min(V.WARM, V.STEPS // 2); return optim.join_schedules([optim.linear_schedule(0.0, lr, w_), optim.cosine_decay(lr, V.STEPS - w_)], [w_])

def train_base(act, seed, D):   # wie V.execute (STAGE confirm/tune, sched wcos), ohne Zwischenmessungen (laut V1/D1-Mechanik ohne Einfluss)
    xtr, ytr = D[0], D[1]; lr, wd = FORMS[act]
    mx.random.seed(seed); m = V.Net(100, act, seed, din=xtr.shape[1]); mx.eval(m.parameters()); m.eval()
    r0 = np.random.default_rng(seed + 777); b0 = r0.permutation(len(xtr))[:V.BATCH]
    _ = nn.value_and_grad(m, V.loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    m.train(); opt = optim.AdamW(learning_rate=sched_base(lr), betas=[0.9, 0.999], eps=1e-8, weight_decay=wd, bias_correction=True)
    vg = nn.value_and_grad(m, V.loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(seed + 777); perm = r.permutation(len(xtr)); pos = 0; status = 'ok'
    for s in range(1, V.STEPS + 1):
        if pos + V.BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + V.BATCH]; pos += V.BATCH
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if (s % 43 == 0 or s == V.STEPS) and not math.isfinite(float(loss)): status = 'nonfinite_loss'; break
    m.eval(); return m, status

def ptq(m, D):   # PTQ wie V1: w8a4a mit der besten der vier Kalibrierungen (Wahl auf den Kalibrierbildern)
    xtr, ytr, xv, yv, xte, yte = D; RA = V.calib_asym(m, xtr[:V.NCAL]); ce = {}
    for cal in V.CALS4:
        _, a = V.evaluate(V.qforward_asym(m, 4, RA[cal]), xtr[:V.NCAL], ytr[:V.NCAL]); ce[cal] = 1 - a
    bc = min(V.CALS4, key=lambda k: (ce[k], V.CALS4.index(k))); _, a = V.evaluate(V.qforward_asym(m, 4, RA[bc]), xte, yte)
    return 1 - a, bc, RA[bc]

def ste(x, q): return x + mx.stop_gradient(q - x)   # Durchreich-Schaetzer: vorwaerts q, rueckwaerts Identitaet
def gscale(p, g): return p * g + mx.stop_gradient(p - p * g)   # LSQ: Gradient von p mit g skalieren, Wert unveraendert

class Quant(nn.Module):   # Quantisierer fuer die Eingaenge aller Linearschichten (101), lernbare Parameter je Schicht
    def __init__(self, method, lohi, numel):
        super().__init__(); self.method = method; self.numel = numel
        sc, zp, lo, hi = [], [], [], []   # Start exakt wie die PTQ-Rundung qa_asym: Schritt (hi-lo)/n, ganzzahliger Nullpunkt zp
        for l0, h0 in lohi:
            l0 = min(l0, 0.0); h0 = max(h0, 0.0); st_ = max(h0 - l0, 1e-12) / NQ; sc.append(st_); zp.append(float(round(-l0 / st_))); lo.append(l0); hi.append(h0)
        if method == 'LSQ': self.s = [mx.array(v, dtype=mx.float32) for v in sc]; self.z = [mx.array(v, dtype=mx.float32) for v in zp]   # LSQ+: Schritt und Nullpunkt lernbar
        else:   # PACT: Grenzen = Raender des PTQ-Gitters; Schritt = Startschritt mal relative Breite (am Start exakt der PTQ-Schritt)
            self.lo = [mx.array(-z_ * v, dtype=mx.float32) for v, z_ in zip(sc, zp)]; self.hi = [mx.array((NQ - z_) * v, dtype=mx.float32) for v, z_ in zip(sc, zp)]
            self.s0 = [float(v) for v in sc]; self.w0 = [float(h_) - float(l_) for l_, h_ in zip(self.lo, self.hi)]
    def __call__(self, i, x):
        if self.method == 'LSQ':   # gleiche Rechenfolge wie qa_asym: (clip(round(x/s) + z, 0, n) - z) * s, Rundung mit STE
            g = 1.0 / math.sqrt(self.numel[i] * NQ); s = gscale(self.s[i], g); z = gscale(self.z[i], g); u = x / s
            return (mx.clip(ste(u, mx.round(u)) + z, 0, NQ) - z) * s
        lo, hi = self.lo[i], self.hi[i]; c = mx.minimum(mx.maximum(x, lo), hi)   # PACT: Gradient zu lo/hi nur aus geclippten Werten
        s = mx.stop_gradient(self.s0[i] * ((hi - lo) / self.w0[i])); zp = mx.stop_gradient(mx.round(-lo / s))
        return ste(c, (mx.clip(mx.round(c / s) + zp, 0, NQ) - zp) * s)
    def bounds(self):
        if self.method == 'LSQ': return [(-float(z) * float(s), (NQ - float(z)) * float(s)) for s, z in zip(self.s, self.z)]
        return [(float(l), float(h)) for l, h in zip(self.lo, self.hi)]
    def penalty(self):
        if self.method != 'PACT': return mx.array(0.0)
        return PACT_L2 * sum(mx.square(l) + mx.square(h) for l, h in zip(self.lo, self.hi))

class QNet(nn.Module):   # Basisnetz mit Fake-Quantisierung: Gewichte int8 je Kanal (STE), Eingaenge 4 Bit ueber Quant
    def __init__(self, m, q):
        super().__init__(); self.m = m; self.q = q
    def __call__(self, x):
        h = x; L = self.m.hid + [self.m.out]
        for i, l in enumerate(L):
            h = self.q(i, h); W = ste(l.weight, V.qw(l.weight, 'i8')); h = h @ W.T + l.bias
            if i < len(L) - 1: h = self.m.post(i, h)
        return h

def qloss(qm, x, y): return nn.losses.cross_entropy(qm(x), y, reduction='mean') + qm.q.penalty()

def finetune(m, act, seed, method, lrq, D, lohi):
    xtr, ytr = D[0], D[1]; lr, wd = FORMS[act]; lr_ft = FT_LR_FAC * lr
    ow = optim.AdamW(learning_rate=optim.cosine_decay(lr_ft, FT_STEPS), betas=[0.9, 0.999], eps=1e-8, weight_decay=wd, bias_correction=True)
    if method == 'FT':
        net = m; vg = nn.value_and_grad(m, V.loss_fn); state = [m.state, ow.state]
        def step(x, y):
            loss, g = vg(m, x, y); ow.update(m, g); return loss
    else:
        numel = [l.weight.shape[1] for l in m.hid + [m.out]]   # LSQ: N = Zahl der Merkmale am Eingang der Schicht
        q = Quant(method, lohi, numel); net = QNet(m, q); oq = optim.Adam(learning_rate=optim.cosine_decay(lrq, FT_STEPS), betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
        vg = nn.value_and_grad(net, qloss); state = [net.state, ow.state, oq.state]
        def step(x, y):
            loss, g = vg(net, x, y); ow.update(net.m, g['m']); oq.update(net.q, g['q']); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    net.train(); r = np.random.default_rng(seed + FT_SEED_OFF); perm = r.permutation(len(xtr)); pos = 0; run = []; status = 'ok'
    for s in range(1, FT_STEPS + 1):
        if pos + V.BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + V.BATCH]; pos += V.BATCH
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s % 43 == 0 or s == FT_STEPS:
            lv = float(loss); run.append(lv)
            if not math.isfinite(lv): status = 'nonfinite_loss'; break
    net.eval(); return net, status, run

def copy_net(m, act, seed):
    f = V.Net(100, act, seed, din=m.hid[0].weight.shape[1])
    for a, b in zip(f.hid + [f.out], m.hid + [m.out]): a.weight = mx.array(b.weight); a.bias = mx.array(b.bias)
    mx.eval(f.parameters()); f.eval(); return f

def write(o):
    o['timestamp'] = datetime.now().isoformat(timespec='seconds')
    p = OUTDIR / 'workers' / f"w{o.pop('_wid')}.csv"; p.parent.mkdir(parents=True, exist_ok=True); new = not p.exists()
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(o)

def rid(stage, act, method, lrq, seed): return f"{stage}_{act}_{method}" + (f"_q{lrq:g}" if method != 'FT' and stage == 'tune' else '') + f"_s{seed}"

def group(stage, act, seed, wid, lrqs):   # Basis + Feintunings fuer eine (Form, Seed)
    t0 = time.time(); D = V.data_for('cifar10', act); xtr, ytr, xv, yv, xte, yte = D
    m, status = train_base(act, seed, D); row = ref_row(act, seed, stage); fwd = lambda x: m(x)
    _, va = V.evaluate(fwd, xv, yv); same = (1 - va) == float(row['val_err_fp32'])
    b = dict(run_id=rid(stage, act, 'BASE', 0, seed), stage=stage, act=act, method='BASE', lrq='', seed=seed, status=status, val_err_fp32=1 - va, _wid=wid)
    if status == 'ok':
        _, ta = V.evaluate(fwd, xte, yte); e4, bc, lohi = ptq(m, D); b.update(test_err_fp32=1 - ta, test_err_w8a4a_best=e4, w8a4a_best_cal=bc)
        if stage == 'confirm': same = same and (1 - ta) == float(row['test_err_fp32']) and e4 == float(row['test_err_w8a4a_best']) and bc == row['w8a4a_best_cal']
    b.update(base_valid=same, sec=round(time.time() - t0, 1)); write(b); print('base', act, seed, status, same, datetime.now().strftime('%H:%M:%S'), flush=True)
    if status != 'ok': return
    for method, lrq in [('FT', 0.0)] + [(mm, q) for mm in ('LSQ', 'PACT') for q in lrqs[mm]]:
        t1 = time.time(); f = copy_net(m, act, seed); net, st, run = finetune(f, act, seed, method, lrq, D, lohi)
        o = dict(run_id=rid(stage, act, method, lrq, seed), stage=stage, act=act, method=method, lrq=lrq if method != 'FT' else '', seed=seed, status=st, base_valid=same,
                 train_loss_last50=float(np.mean(run[-2:])) if run else float('nan'), _wid=wid)
        if st == 'ok':
            fm = net.m if method != 'FT' else net; ff = lambda x: fm(x)
            _, va = V.evaluate(ff, xv, yv); _, ta = V.evaluate(ff, xte, yte); o.update(val_err_fp32=1 - va, test_err_fp32=1 - ta)
            e4, bc, _ = ptq(fm, D); o.update(test_err_w8a4a_best=e4, w8a4a_best_cal=bc)   # PTQ am feingetunten Netz (fuer FT die Messgroesse)
            if method != 'FT':
                qf = lambda x: net(x); _, va = V.evaluate(qf, xv, yv); _, ta = V.evaluate(qf, xte, yte); o.update(val_err_qat=1 - va, test_err_qat=1 - ta)
                bd = net.q.bounds(); b0 = Quant(method, lohi, [1] * len(lohi)).bounds(); end = float(np.median([h - l for l, h in bd]))   # Start = Grenzen des Quantisierers vor dem Feintuning
                o.update(range_med_start=float(np.median([h - l for l, h in b0])), range_med_end=end,
                         range_move_med=float(np.median([abs((h - l) / (h0 - l0) - 1) for (l, h), (l0, h0) in zip(bd, b0)])))
        o['sec'] = round(time.time() - t1, 1); write(o); print('ft', act, method, lrq, seed, st, datetime.now().strftime('%H:%M:%S'), flush=True)

def selected_lrq():   # Wahl je (Form, Methode): kleinster medianer Val-Fehler mit QAT-Inferenz ueber die Tuning-Seeds; Ausfall = 0,9; Gleichstand -> kleinere LR_Q
    rows = []
    for f in sorted((ROOT / 'tune' / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    out = {}
    for act in FORMS:
        for mm in ('LSQ', 'PACT'):
            cand = []
            for q in LRQ_GRID:
                v = [float(r['val_err_qat']) if r['status'] == 'ok' and r['val_err_qat'] not in ('', None) else 0.9 for r in rows if r['act'] == act and r['method'] == mm and float(r['lrq']) == q]
                if v: cand.append((float(np.median(v)), q))
            if cand: out[(act, mm)] = min(cand)[1]
    return out

def selfcheck():
    out = {'device': str(mx.default_device()), 'ft_steps': FT_STEPS, 'lrq_grid': LRQ_GRID}; rng = np.random.default_rng(0)
    u = mx.array(rng.uniform(-1, 3, (64, 256)).astype(np.float32)); lohi = [(-1.0, 3.0)]
    for mm in ('LSQ', 'PACT'):   # vorwaerts gleich zur PTQ-Rundung qa_asym (bis auf den ganzzahligen Nullpunkt), Gradienten endlich
        q = Quant(mm, lohi, [u.size]); y = q(0, u); ref = V.qa_asym(u, 4, (-1.0, 3.0)); out[f'{mm}_vs_ptq_maxdiff'] = float(mx.max(mx.abs(y - ref)))
        out[f'{mm}_levels'] = int(len(np.unique(np.array(y).round(5))))
        g = nn.value_and_grad(q, lambda qq, x: mx.sum(qq(0, x) ** 2))(q, u * 1.5)[1]
        out[f'{mm}_grad_finite'] = bool(all(np.isfinite(np.array(v)).all() for v in (g['s'] + g['z'] if mm == 'LSQ' else g['lo'] + g['hi'])))
        out[f'{mm}_grad_nonzero'] = bool(any(float(mx.abs(v)) > 0 for v in (g['s'] if mm == 'LSQ' else g['hi'])))
    out['ref_rows'] = all(ref_row(a, s, 'confirm') is not None for a in FORMS for s in SEEDS) and all(ref_row(a, s, 'tune') is not None for a in FORMS for s in TUNE_SEEDS)
    out['n_tune_runs'] = len(FORMS) * len(TUNE_SEEDS) * (2 + 2 * len(LRQ_GRID)); out['n_confirm_runs'] = len(FORMS) * len(SEEDS) * 4
    out['ok'] = all(out[f'{mm}_grad_finite'] and out[f'{mm}_grad_nonzero'] and out[f'{mm}_levels'] <= NQ + 1 for mm in ('LSQ', 'PACT')) and out['ref_rows'] and out['LSQ_vs_ptq_maxdiff'] < 1e-5 and out['PACT_vs_ptq_maxdiff'] < 1e-5
    return out

def main():
    global FT_STEPS, OUTDIR, SEEDS, TUNE_SEEDS, LRQ_GRID
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); p.add_argument('--stage', choices=['tune', 'confirm', 'both'], default='both')
    a = p.parse_args(); mx.set_cache_limit(V.CACHE_LIMIT)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    base_out = ROOT / 'smoke' if a.smoke else ROOT
    if a.smoke: V.STEPS = 60; FT_STEPS = 20; TUNE_SEEDS = [42000]; SEEDS = [42168]; LRQ_GRID = [1e-3]
    if a.stage in ('tune', 'both'):
        OUTDIR = base_out / 'tune'; jobs = [(act, s) for s in TUNE_SEEDS for act in FORMS]
        for i, (act, s) in enumerate(jobs):
            if i % a.workers == a.worker: group('tune', act, s, a.worker, {'LSQ': LRQ_GRID, 'PACT': LRQ_GRID})
        while sum(1 for f in (OUTDIR / 'workers').glob('w*.csv') for _ in csv.DictReader(open(f))) < len(jobs) * (2 + 2 * len(LRQ_GRID)): time.sleep(20)
    if a.stage in ('confirm', 'both'):
        sel = selected_lrq() if not a.smoke else {(act, mm): LRQ_GRID[0] for act in FORMS for mm in ('LSQ', 'PACT')}
        if a.worker == 0: json.dump({f'{k[0]}_{k[1]}': v for k, v in sel.items()}, open(base_out / 'lrq_selected.json', 'w'), indent=1)
        OUTDIR = base_out; jobs = [(act, s) for s in SEEDS for act in FORMS]
        for i, (act, s) in enumerate(jobs):
            if i % a.workers == a.worker: group('confirm', act, s, a.worker, {mm: [sel[(act, mm)]] for mm in ('LSQ', 'PACT')})
    print('fertig', flush=True)

if __name__ == '__main__': main()
