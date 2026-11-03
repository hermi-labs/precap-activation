# D3 Diagnose (D3_Praeregistrierung.md): Ausreisser und Fehlerfortpflanzung je Schicht bei Tiefe 200, int4 und int8. Beschreibend, mit Claude.
# Code aus D1 (d1_diag.py), Aenderungen: Tiefe als Parameter, Netze aus B15 (d200) und B10 (d100), zusaetzlich 8 Bit, Anteil im weichen
# Saettigungsbereich je Schicht und die Anhaeufung (Eingaenge der Schichten 1..k gerundet). Laedt b15_run.py bzw. b10_run.py unveraendert.
# Aufruf: python d3_diag.py --selfcheck | --worker W --workers N
import argparse, csv, importlib.util, json, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent; ST = ROOT.parent
def _load(name):   # eingefrorenes Studienskript unveraendert als Modul laden
    spec = importlib.util.spec_from_file_location(name, ST / name.split('_')[0] / f'{name}.py'); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
MODS = {'b10_run': _load('b10_run'), 'b15_run': _load('b15_run')}
B = MODS['b15_run']

RUNS = [dict(name=n, act=a, depth=d, lr=lr, wd=wd, src=src, seed=s, mod=mod)
        for n, a, d, lr, wd, src, mod, seeds in (('bp200', 'kL2B_tr', 200, 5e-5, 0.3, 'b15', 'b15_run', (42192, 42193)),
                                                ('se200', 'selu_tr', 200, 2e-4, 0.3, 'b15', 'b15_run', (42192, 42193)),
                                                ('bp100', 'kL2B_tr', 100, 1e-4, 0.3, 'b10', 'b10_run', (42168, 42169)))
        for s in seeds]
KS = {100: (2, 10, 25, 50, 75, 99), 200: (2, 10, 25, 50, 100, 150, 199)}   # Linearschicht (1-basiert), deren Eingang gerundet bzw. verrauscht wird
NTEST = 2000; NOISE = 0.01; BITS = (4, 8); SOFT = 0.8 * 3.0   # weicher Bereich der Huelle: |a| > rho*M (rho = 0,8, M = 3)

def ref_row(c):
    for f in sorted((ST / c['src'] / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if (r['dataset'], int(r['depth']), r['act'], int(r['seed'])) == ('cifar10', c['depth'], c['act'], c['seed']) and \
               float(r['lr']) == c['lr'] and float(r['wd']) == c['wd'] and r['sched'] == 'cos': return r
    raise KeyError(c)

def train(c):   # wie B.execute fuer STAGE confirm, ohne Zwischenmessungen (wie D1)
    xtr, ytr, xv, yv, xte, yte = B.data_for('cifar10', B.base(c['act']) + ('_tr' if B.recipe(c['act']) else ''))
    mx.random.seed(c['seed']); m = B.Net(c['depth'], c['act'], c['seed'], din=xtr.shape[1]); mx.eval(m.parameters()); m.eval()
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
    return got, same, {4: RA[bc], 8: RA['max']}   # 4 Bit: Kalibrierung nach der Wahlregel der Studie; 8 Bit: Min/Max (ueblich fuer int8)

def fwd_mod(m, x, mods):   # Vorwaerts in fp32; mods = {k: f}: der Eingang der Linearschicht k (1-basiert) wird durch f ersetzt
    h = x; L = m.hid + [m.out]
    for i, l in enumerate(L):
        if i + 1 in mods: h = mods[i + 1](h)
        h = l(h)
        if i < len(L) - 1: h = m.post(i, h)
    return h

def inputs(m, x):   # Eingaenge aller Linearschichten (fp32)
    h = mx.array(x); out = []
    for i, l in enumerate(m.hid + [m.out]):
        out.append(h)
        if l is not m.out: h = m.post(i, l(h))
    return out

def rel(a, b): return float(mx.sqrt(mx.sum((a - b) ** 2)) / mx.sqrt(mx.sum(b ** 2)))

def measure(m, D, rng, depth):
    xtr, ytr, xv, yv, xte, yte = D; xc = xtr[:B.NCAL]; xt = mx.array(xte[:NTEST]); yt = mx.array(yte[:NTEST])
    A = inputs(m, xc); per = []
    for i in range(1, len(A)):   # Eingaenge der Schichten 2..depth+1 (letzte = Ausgabeschicht)
        a64 = np.array(A[i], dtype=np.float64).ravel(); mu = a64.mean(); v = a64.var(); aa = np.abs(a64); amax = aa.max()
        p = dict(layer=i + 1, kurt=float(((a64 - mu) ** 4).mean() / v ** 2) if v > 0 else float('nan'),
                 max_over_p9999=float(amax / max(np.quantile(aa, 0.9999), 1e-30)), frac_near_max=float((aa > 0.95 * amax).mean()),
                 frac_soft=float((aa > SOFT).mean()), rms=float(np.sqrt((a64 ** 2).mean())), absmax=float(amax))
        for bits in BITS:
            q = B.qa_asym(A[i], bits, rng[bits][i]); p[f'eps{bits}'] = float(mx.mean((q - A[i]) ** 2) / mx.mean(A[i] ** 2))
        per.append(p)
    base = m(xt); pred0 = mx.argmax(base, 1); err0 = float(mx.mean(pred0 != yt)); HT = inputs(m, xte[:NTEST])
    prop = []
    for k in KS[depth]:
        pk = next(p for p in per if p['layer'] == k); hk = HT[k - 1]; rms = float(mx.sqrt(mx.mean(hk ** 2)))
        nz = mx.random.normal(hk.shape, key=mx.random.key(1000 + k)) * (NOISE * rms)
        o = dict(layer=k, noise_gain=rel(fwd_mod(m, xt, {k: lambda h, nz=nz: h + nz}), base) / NOISE)
        for bits in BITS:
            lq = fwd_mod(m, xt, {k: lambda h, k=k, bits=bits: B.qa_asym(h, bits, rng[bits][k - 1])}); rq = rel(lq, base); e = pk[f'eps{bits}']
            o[f'r{bits}'] = rq; o[f'gain{bits}'] = rq / math.sqrt(e) if e > 0 else float('nan'); o[f'flips{bits}'] = float(mx.mean(mx.argmax(lq, 1) != pred0))
        prop.append(o)
    accum = []   # Anhaeufung: Eingaenge der Schichten 1..k gerundet (nur Aktivierungen, Gewichte fp32)
    for k in KS[depth] + (depth + 1,):
        o = dict(upto=k)
        for bits in BITS:
            lq = fwd_mod(m, xt, {j: (lambda h, j=j, bits=bits: B.qa_asym(h, bits, rng[bits][j - 1])) for j in range(1, k + 1)})
            o[f'r{bits}'] = rel(lq, base); o[f'err{bits}'] = float(mx.mean(mx.argmax(lq, 1) != yt))
        accum.append(o)
    out = dict(per_layer=per, propagation=prop, accum=accum, err_fp32_2000=err0)
    for bits in BITS:   # alle Schichten gerundet, Gewichte int8 (wie die Studien)
        full = B.qforward_asym(m, bits, rng[bits])(xt); out[f'r_all{bits}'] = rel(full, base); out[f'err_all{bits}_2000'] = float(mx.mean(mx.argmax(full, 1) != yt))
        out[f'eps{bits}_median'] = float(np.median([p[f'eps{bits}'] for p in per[:-1]]))
        out[f'gain{bits}_median'] = float(np.median([p[f'gain{bits}'] for p in prop]))
    q = len(per[:-1]) // 4; first, last = per[:q], per[:-1][-q:]
    for key in ('kurt', 'max_over_p9999', 'frac_soft', 'eps4', 'eps8'):
        out[f'{key}_median'] = float(np.median([p[key] for p in per[:-1]]))
        out[f'{key}_q1'] = float(np.median([p[key] for p in first])); out[f'{key}_q4'] = float(np.median([p[key] for p in last]))
    out['noise_gain_median'] = float(np.median([p['noise_gain'] for p in prop]))
    return out

def selfcheck():
    out = {}; rng = np.random.default_rng(0)
    u = mx.array(rng.uniform(-1, 1, 10 ** 6).astype(np.float32))
    for bits in BITS:
        q = B.qa_asym(u, bits, (-1.0, 1.0)); n = 2 ** bits - 1
        out[f'eps_uniform{bits}'] = float(mx.mean((q - u) ** 2) / mx.mean(u ** 2)); out[f'eps_uniform{bits}_theory'] = (2 / n) ** 2 / 12 / (1 / 3)
    out['uniform_ok'] = all(abs(out[f'eps_uniform{b}'] / out[f'eps_uniform{b}_theory'] - 1) < 0.03 for b in BITS)
    m = B.Net(200, 'kL2B_tr', 1, din=3072); mx.eval(m.parameters()); m.eval()
    x = rng.normal(size=(64, 3072)).astype(np.float32); x = B.pex(x); xm = mx.array(x)
    out['fwd_mod_identity_exact'] = bool(mx.all(fwd_mod(m, xm, {50: lambda h: h, 150: lambda h: h}) == m(xm)).item())
    out['n_inputs_d200'] = len(inputs(m, x)); out['n_runs'] = len(RUNS); out['runs'] = [f"{c['name']}_s{c['seed']}" for c in RUNS]
    out['b15_steps_like_b10'] = (MODS['b15_run'].STEPS, MODS['b15_run'].BATCH, MODS['b15_run'].NCAL) == (MODS['b10_run'].STEPS, MODS['b10_run'].BATCH, MODS['b10_run'].NCAL)
    for c in RUNS: ref_row(c)
    out['ref_rows_found'] = True
    out['ok'] = out['uniform_ok'] and out['fwd_mod_identity_exact'] and out['n_inputs_d200'] == 201 and out['n_runs'] == 6 and out['b15_steps_like_b10']
    return out

def main():
    global B
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); a = p.parse_args(); mx.set_cache_limit(B.CACHE_LIMIT)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    (ROOT / 'results').mkdir(exist_ok=True)
    for i, c in enumerate(RUNS):
        f = ROOT / 'results' / f"d3_{c['name']}_s{c['seed']}.json"
        if i % a.workers != a.worker or f.exists(): continue
        B = MODS[c['mod']]   # Netz, Daten und Quantisierung aus dem Skript der Studie, aus der die Vergleichszeile stammt
        t0 = time.time(); m, status, D = train(c); row = ref_row(c); o = dict(run=c, status=status)
        if status == 'ok':
            got, same, rng = verify(m, D, row); o.update(verify_got=got, verify_same=same, valid=all(same.values()))
            o.update(measure(m, D, rng, c['depth']))
        o['sec'] = round(time.time() - t0, 1); json.dump(o, open(f, 'w'), indent=1)
        print('done', f.name, status, o.get('valid'), datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
