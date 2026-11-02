# D1 Diagnose (D1_Praeregistrierung.md): warum zahlt TAT bei 4 Bit mehr als BPreCap? Beschreibend, mit Claude geschrieben.
# Laedt studien/b10/b10_run.py unveraendert als Modul (Netz, Daten, Quantisierung), trainiert exakt wie dort (ohne Zwischenmessungen),
# prueft die Nachbildung bitgleich gegen die Roh-CSV von B10/B10b und misst dann am Ende: Form je Schicht, lokaler Rundungsfehler,
# Fortpflanzung einzelner Rundungen, Rausch-Verstaerkung, alle Schichten gerundet.
# Aufruf: python d1_diag.py --selfcheck | --worker W --workers N
import argparse, csv, importlib.util, json, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent; ST = ROOT.parent
def _load(name):   # eingefrorenes Studienskript unveraendert als Modul laden
    spec = importlib.util.spec_from_file_location(name, ST / name.split('_')[0] / f'{name}.py'); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
MODS = {'b10_run': _load('b10_run'), 'b12_run': _load('b12_run'), 'b13_run': _load('b13_run')}   # 01.11.: b12_run fuer SELU (Erweiterung nach B12, siehe Praeregistrierung)
B = MODS['b10_run']

RUNS = [dict(name=n, act=a, lr=lr, wd=wd, src=src, seed=s, mod=mod)
        for s in (42168, 42169)
        for n, a, lr, wd, src, mod in (('bp', 'kL2B_tr', 1e-4, 0.3, 'b10', 'b10_run'), ('lb', 'LB_tr', 1e-4, 0.3, 'b10', 'b10_run'),
                                       ('ta', 'tat', 1e-4, 0.3, 'b10b', 'b10_run'), ('t0', 'tat', 2e-4, 0.0, 'b10', 'b10_run'),
                                       ('se', 'selu_tr', 2e-4, 0.3, 'b12', 'b12_run'), ('lh', 'LBh_tr', 5e-5, 0.3, 'b13', 'b13_run'))]
KS = (2, 10, 25, 50, 75, 99)        # Nummer der Linearschicht (1-basiert), deren Eingang gerundet bzw. verrauscht wird
NTEST = 2000; NOISE = 0.01; BITS = 4

def ref_row(c):
    for f in sorted((ST / c['src'] / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if (r['dataset'], int(r['depth']), r['act'], int(r['seed'])) == ('cifar10', 100, c['act'], c['seed']) and \
               float(r['lr']) == c['lr'] and float(r['wd']) == c['wd'] and r['sched'] == 'cos': return r
    raise KeyError(c)

def train(c):   # wie B.execute fuer STAGE confirm, ohne Zwischenmessungen (laut B9/B10-Mechaniktest ohne Einfluss auf das Training)
    xtr, ytr, xv, yv, xte, yte = B.data_for('cifar10', B.base(c['act']) + ('_tr' if B.recipe(c['act']) else ''))
    mx.random.seed(c['seed']); m = B.Net(100, c['act'], c['seed'], din=xtr.shape[1]); mx.eval(m.parameters()); m.eval()
    r0 = np.random.default_rng(c['seed'] + 777); b0 = r0.permutation(len(xtr))[:B.BATCH]
    _ = nn.value_and_grad(m, B.loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    m.train()
    opt = optim.AdamW(learning_rate=optim.cosine_decay(c['lr'], B.STEPS), betas=[0.9, 0.999], eps=1e-8, weight_decay=c['wd'], bias_correction=True)
    vg = nn.value_and_grad(m, B.loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(c['seed'] + 777); perm = r.permutation(len(xtr)); pos = 0; status = 'ok'
    for s in range(1, B.STEPS + 1):
        if pos + B.BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + B.BATCH]; pos += B.BATCH
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if (s % 43 == 0 or s == B.STEPS) and not math.isfinite(float(loss)): status = 'nonfinite_loss'; break
    m.eval(); return m, status, (xtr, ytr, xv, yv, xte, yte)

def verify(m, D, row):
    xtr, ytr, xv, yv, xte, yte = D; fwd = lambda x: m(x)
    _, va = B.evaluate(fwd, xv, yv); _, ta = B.evaluate(fwd, xte, yte)
    RA = B.calib_asym(m, xtr[:B.NCAL]); ce = {}
    for cal in B.CALS4:
        _, a = B.evaluate(B.qforward_asym(m, 4, RA[cal]), xtr[:B.NCAL], ytr[:B.NCAL]); ce[cal] = 1 - a
    bc = min(B.CALS4, key=lambda k: (ce[k], B.CALS4.index(k))); _, a = B.evaluate(B.qforward_asym(m, 4, RA[bc]), xte, yte)
    got = dict(val_err_fp32=1 - va, test_err_fp32=1 - ta, test_err_w8a4a_best=1 - a, w8a4a_best_cal=bc)
    same = {k: (got[k] == (float(row[k]) if k != 'w8a4a_best_cal' else row[k])) for k in got}
    return got, same, RA[bc]

def fwd_mod(m, x, k, mod):   # Vorwaerts in fp32, nur der Eingang der Linearschicht k (1-basiert) wird durch mod ersetzt
    h = x; L = m.hid + [m.out]
    for i, l in enumerate(L):
        if i + 1 == k: h = mod(h)
        h = l(h)
        if i < len(L) - 1: h = m.post(i, h)
    return h

def inputs(m, x):   # Eingaenge aller 101 Linearschichten (fp32)
    h = mx.array(x); out = []
    for i, l in enumerate(m.hid + [m.out]):
        out.append(h)
        if l is not m.out: h = m.post(i, l(h))
    return out

def rel(a, b): return float(mx.sqrt(mx.sum((a - b) ** 2)) / mx.sqrt(mx.sum(b ** 2)))

def measure(m, D, rngs):
    xtr, ytr, xv, yv, xte, yte = D; xc = xtr[:B.NCAL]; xt = mx.array(xte[:NTEST]); yt = mx.array(yte[:NTEST])
    A = inputs(m, xc); per = []
    for i in range(1, len(A)):   # Eingaenge der Schichten 2..101 (101 = Ausgabeschicht)
        a64 = np.array(A[i], dtype=np.float64).ravel(); mu = a64.mean(); v = a64.var(); aa = np.abs(a64); amax = aa.max()
        q = B.qa_asym(A[i], BITS, rngs[i]); eps = float(mx.mean((q - A[i]) ** 2) / mx.mean(A[i] ** 2))
        per.append(dict(layer=i + 1, kurt=float(((a64 - mu) ** 4).mean() / v ** 2) if v > 0 else float('nan'),
                        max_over_p9999=float(amax / max(np.quantile(aa, 0.9999), 1e-30)), frac_near_max=float((aa > 0.95 * amax).mean()),
                        rms=float(np.sqrt((a64 ** 2).mean())), eps_local=eps))
    base = m(xt); pred0 = mx.argmax(base, 1); err0 = float(mx.mean(pred0 != yt))
    prop = []
    for k in KS:
        e = next(p['eps_local'] for p in per if p['layer'] == k)
        lq = fwd_mod(m, xt, k, lambda h, k=k: B.qa_asym(h, BITS, rngs[k - 1]))
        hk = inputs(m, xte[:NTEST])[k - 1]; rms = float(mx.sqrt(mx.mean(hk ** 2)))
        nz = mx.random.normal(hk.shape, key=mx.random.key(1000 + k)) * (NOISE * rms)
        ln = fwd_mod(m, xt, k, lambda h, nz=nz: h + nz)
        rq = rel(lq, base); rn = rel(ln, base)
        prop.append(dict(layer=k, eps_local=e, r_logits=rq, gain=rq / math.sqrt(e) if e > 0 else float('nan'),
                         flips=float(mx.mean(mx.argmax(lq, 1) != pred0)), noise_gain=rn / NOISE))
    full = B.qforward_asym(m, BITS, rngs)(xt)
    return dict(per_layer=per, propagation=prop, err_fp32_2000=err0, r_all=rel(full, base), err_all_2000=float(mx.mean(mx.argmax(full, 1) != yt)),
                eps_local_median=float(np.median([p['eps_local'] for p in per[:-1]])),
                kurt_median=float(np.median([p['kurt'] for p in per[:-1]])), gain_median=float(np.median([p['gain'] for p in prop])),
                noise_gain_median=float(np.median([p['noise_gain'] for p in prop])))

def selfcheck():
    out = {}; rng = np.random.default_rng(0)
    u = mx.array(rng.uniform(-1, 1, 10 ** 6).astype(np.float32)); q = B.qa_asym(u, 4, (-1.0, 1.0))
    out['eps_uniform'] = float(mx.mean((q - u) ** 2) / mx.mean(u ** 2)); out['eps_uniform_theory'] = (2 / 15) ** 2 / 4
    out['uniform_ok'] = abs(out['eps_uniform'] / out['eps_uniform_theory'] - 1) < 0.02
    m = B.Net(100, 'kL2B_tr', 1, din=3072); mx.eval(m.parameters()); m.eval()
    x = rng.normal(size=(64, 3072)).astype(np.float32); x = B.pex(x); xm = mx.array(x)
    out['fwd_mod_identity_exact'] = bool(mx.all(fwd_mod(m, xm, 50, lambda h: h) == m(xm)).item())
    out['n_inputs'] = len(inputs(m, x)); out['n_runs'] = len(RUNS); out['runs'] = [f"{c['name']}_s{c['seed']}" for c in RUNS]
    for c in RUNS: ref_row(c)
    out['ref_rows_found'] = True
    out['ok'] = out['uniform_ok'] and out['fwd_mod_identity_exact'] and out['n_inputs'] == 101 and out['n_runs'] == 12
    return out

def main():
    global B
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); a = p.parse_args(); mx.set_cache_limit(B.CACHE_LIMIT)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    (ROOT / 'results').mkdir(exist_ok=True)
    for i, c in enumerate(RUNS):
        f = ROOT / 'results' / f"d1_{c['name']}_s{c['seed']}.json"
        if i % a.workers != a.worker or f.exists(): continue
        B = MODS[c['mod']]   # Netz, Daten und Quantisierung aus dem Skript der Studie, aus der die Vergleichszeile stammt
        t0 = time.time(); m, status, D = train(c); row = ref_row(c); o = dict(run=c, status=status)
        if status == 'ok':
            got, same, rngs = verify(m, D, row); o.update(verify_got=got, verify_same=same, valid=all(same.values()))
            o.update(measure(m, D, rngs))
        o['sec'] = round(time.time() - t0, 1); json.dump(o, open(f, 'w'), indent=1)
        print('done', f.name, status, o.get('valid'), datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
