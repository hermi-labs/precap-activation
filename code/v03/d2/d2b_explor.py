# D2b EXPLORATIV (nicht praeregistriert, mit Claude): Nachfrage zu D2. (1) Spektrum der Gewichte (Isometrie-Verlust?), (2) linearer Anlauf der LR
# (5 % der Schritte, dann Kosinus) als moeglicher Gegenmittel-Kandidat fuer V1. Basis: d2_diag.py, sonst unveraendert.
# Laedt b10_run.py bzw. b13_run.py unveraendert als Modul, trainiert exakt wie deren Tuning-Laeufe (AdamW, Kosinus, 6880 Schritte),
# prueft die Nachbildung bitgleich gegen die Tuning-CSV (val_err_fp32) und misst alle 86 Schritte auf einem festen Probe-Batch
# (512 Trainingsbilder, ohne Aktualisierung): Saettigung der Huelle je Schicht, Gradientennormen je Schicht, Gewichtsnormen, Verlust.
# Aufruf: python d2_diag.py --selfcheck | --worker W --workers N
import argparse, csv, importlib.util, json, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent; ST = ROOT.parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ST / name.split('_')[0] / f'{name}.py'); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
MODS = {'b10_run': _load('b10_run'), 'b13_run': _load('b13_run')}
B = MODS['b10_run']
CFG = (('bp1', 'kL2B_tr', 1e-4, 0.3, 'b10', 'b10_run'), ('bp2', 'kL2B_tr', 2e-4, 0.3, 'b10', 'b10_run'), ('bp2w3', 'kL2B_tr', 2e-4, 3.0, 'b10', 'b10_run'),
       ('lb2', 'LB_tr', 2e-4, 0.3, 'b10', 'b10_run'), ('bh2', 'kL2Bh_tr', 2e-4, 0.3, 'b13', 'b13_run'), ('ta2', 'tat', 2e-4, 0.0, 'b10', 'b10_run'))
RUNS = [dict(name=n, act=a, lr=lr, wd=wd, src=src, mod=mod, seed=42000, warm=w) for n, a, lr, wd, src, mod in CFG if n in ('bp1', 'bp2', 'lb2') for w in ((0, 344) if n != 'bp1' else (0,))]
SVL = (1, 25, 50, 75, 100)
PROBE_EVERY = 86; NPROBE = 512; MBOUND = 3.0; RHO = 0.8; LAYERS = (1, 10, 25, 50, 75, 100)

def ref_row(c):
    for f in sorted((ST / c['src'] / 'tune' / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if (r['dataset'], int(r['depth']), r['act'], int(r['seed'])) == ('cifar10', 100, c['act'], c['seed']) and \
               float(r['lr']) == c['lr'] and float(r['wd']) == c['wd'] and r['sched'] == 'cos': return r
    raise KeyError(c)

def probe(m, xp, yp):
    h = mx.array(xp); knee, deep, rms = [], [], []
    for i, l in enumerate(m.hid):
        z = l(h); h = m.post(i, z); a = mx.abs(h)
        knee.append(float(mx.mean(a > RHO * MBOUND))); deep.append(float(mx.mean(a > 0.99 * MBOUND))); rms.append(float(mx.sqrt(mx.mean(z * z))))
    lo = float(nn.losses.cross_entropy(m.out(h), mx.array(yp), reduction='mean'))
    _, g = nn.value_and_grad(m, B.loss_fn)(m, mx.array(xp), mx.array(yp))
    gn = [float(mx.sqrt(mx.sum(gl['weight'] ** 2))) for gl in g['hid']]
    wn = [float(mx.sqrt(mx.sum(l.weight ** 2))) for l in m.hid]
    lg = lambda v: math.log10(v) if v > 0 and math.isfinite(v) else float('nan')
    sv = [np.linalg.svd(np.array(m.hid[k - 1].weight, dtype=np.float64), compute_uv=False) for k in SVL]
    return dict(sv_max=[float(v[0]) for v in sv], sv_min=[float(v[-1]) for v in sv], sv_cond_med=float(np.median([v[0] / v[-1] for v in sv])), probe_loss=lo, knee_med=float(np.median(knee)), knee_max=float(np.max(knee)), deep_med=float(np.median(deep)), deep_max=float(np.max(deep)),
                knee_layers=[knee[k - 1] for k in LAYERS], deep_layers=[deep[k - 1] for k in LAYERS], prerms_med=float(np.median(rms)), prerms_max=float(np.max(rms)),
                glog_first_last=lg(gn[0] / gn[-1]) if gn[-1] > 0 else float('nan'), glog_med=lg(float(np.median(gn))), glog_layers=[lg(gn[k - 1]) for k in LAYERS],
                wnorm_med=float(np.median(wn)), wnorm_max=float(np.max(wn)))

def run(c):
    xtr, ytr, xv, yv, xte, yte = B.data_for('cifar10', B.base(c['act']) + ('_tr' if B.recipe(c['act']) else ''))
    mx.random.seed(c['seed']); m = B.Net(100, c['act'], c['seed'], din=xtr.shape[1]); mx.eval(m.parameters()); m.eval()
    r0 = np.random.default_rng(c['seed'] + 777); b0 = r0.permutation(len(xtr))[:B.BATCH]
    _ = nn.value_and_grad(m, B.loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    m.train(); xp, yp = xtr[:NPROBE], ytr[:NPROBE]
    sch = optim.cosine_decay(c['lr'], B.STEPS) if not c['warm'] else optim.join_schedules([optim.linear_schedule(0.0, c['lr'], c['warm']), optim.cosine_decay(c['lr'], B.STEPS - c['warm'])], [c['warm']])
    opt = optim.AdamW(learning_rate=sch, betas=[0.9, 0.999], eps=1e-8, weight_decay=c['wd'], bias_correction=True)
    vg = nn.value_and_grad(m, B.loss_fn); state = [m.state, opt.state]
    def step(x, y):
        loss, g = vg(m, x, y); opt.update(m, g); return loss
    step = mx.compile(step, inputs=state, outputs=state)
    r = np.random.default_rng(c['seed'] + 777); perm = r.permutation(len(xtr)); pos = 0; status = 'ok'; trace = [dict(step=0, **probe(m, xp, yp))]
    for s in range(1, B.STEPS + 1):
        if pos + B.BATCH > len(xtr): perm = r.permutation(len(xtr)); pos = 0
        b = perm[pos:pos + B.BATCH]; pos += B.BATCH
        loss = step(mx.array(xtr[b]), mx.array(ytr[b])); mx.eval(state)
        if s % PROBE_EVERY == 0 or s == B.STEPS: trace.append(dict(step=s, train_loss=float(loss), lr=float(opt.learning_rate), **probe(m, xp, yp)))
        if (s % 43 == 0 or s == B.STEPS) and not math.isfinite(float(loss)): status = 'nonfinite_loss'; break
    m.eval(); _, va = B.evaluate(lambda x: m(x), xv, yv)
    return status, 1 - va, trace

def selfcheck():
    out = {'n_runs': len(RUNS), 'runs': [f"{c['name']}_s{c['seed']}" for c in RUNS]}
    for c in RUNS: ref_row(c)
    out['ref_rows_found'] = True
    m = B.Net(100, 'kL2B_tr', 1, din=3072); mx.eval(m.parameters()); x = B.pex(np.random.default_rng(0).normal(size=(64, 3072)).astype(np.float32))
    p = probe(m, x, np.zeros(64, dtype=np.int64)); out['probe_keys'] = sorted(p); out['probe_finite'] = all(math.isfinite(v) for k, v in p.items() if not isinstance(v, list))
    out['steps'] = B.STEPS; out['ok'] = out['n_runs'] == 5 and out['probe_finite'] and B.STEPS == 6880
    return out

def main():
    global B
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); p.add_argument('--smoke', action='store_true'); a = p.parse_args(); mx.set_cache_limit(B.CACHE_LIMIT)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    runs = RUNS
    if a.smoke:
        for M in MODS.values(): M.STEPS = 172
        runs = [RUNS[1], RUNS[4]]
    od = ROOT / ('smoke_b' if a.smoke else 'results_b'); od.mkdir(exist_ok=True)
    for i, c in enumerate(runs):
        f = od / f"d2b_{c['name']}_w{c['warm']}_s{c['seed']}.json"
        if i % a.workers != a.worker or f.exists(): continue
        B = MODS[c['mod']]; t0 = time.time(); status, verr, trace = run(c); row = ref_row(c)
        o = dict(run=c, status=status, val_err=verr, ref_val_err=float(row['val_err_fp32']), bitgleich=(verr == float(row['val_err_fp32'])) if (not a.smoke and not c['warm']) else None,
                 trace=trace, sec=round(time.time() - t0, 1))
        json.dump(o, open(f, 'w'), indent=1); print('done', f.name, status, round(verr, 4), o['bitgleich'], datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
