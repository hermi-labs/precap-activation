# D3b Diagnose (D3b_Praeregistrierung.md): D3-Messung je Schicht an V1c-Netzen (d200, mit LR-Anlauf). Beschreibend, mit Claude.
# Laedt d3_diag.py (Messung) und v1c_run.py (Netz, Daten, Quantisierung) unveraendert. Neu sind nur die Laufliste, die Vergleichszeile
# (sched "wcos") und das Training mit Anlauf wie in v1c_run.execute. Verglichen wird mit den D3-Ergebnissen der B15-Netze (gleiche Seeds).
# Aufruf: python d3b_diag.py --selfcheck | --worker W --workers N
import argparse, csv, importlib.util, json, math, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim

ROOT = Path(__file__).resolve().parent; ST = ROOT.parent
sp = importlib.util.spec_from_file_location('d3_diag', ST / 'd3' / 'd3_diag.py'); D3 = importlib.util.module_from_spec(sp)
sys.modules['d3_diag'] = D3; sp.loader.exec_module(D3)
V = D3._load('v1c_run'); D3.B = V   # measure/verify/qa_asym nutzen das globale B in d3_diag

RUNS = [dict(name=n, act=a, depth=200, lr=lr, wd=0.3, seed=s)
        for n, a, lr in (('bp200w', 'kL2B_tr', 5e-5), ('se200w', 'selu_tr', 3e-4)) for s in (42192, 42193)]   # V1c-Wahl, Seeds wie D3

def ref_row(c):   # Vergleichszeile aus der V1c-Bestaetigung (mit Anlauf)
    for f in sorted((ST / 'v1c' / 'workers').glob('w*.csv')):
        for r in csv.DictReader(open(f)):
            if (r['dataset'], int(r['depth']), r['act'], int(r['seed']), r['sched']) == ('cifar10', 200, c['act'], c['seed'], 'wcos') and \
               float(r['lr']) == c['lr'] and float(r['wd']) == c['wd']: return r
    raise KeyError(c)

def sched(c):   # wie v1c_run.execute fuer sched "wcos"
    w_ = min(V.WARM, V.STEPS // 2)
    return optim.join_schedules([optim.linear_schedule(0.0, c['lr'], w_), optim.cosine_decay(c['lr'], V.STEPS - w_)], [w_])

def train(c):   # wie D3.train, nur mit dem Plan aus sched()
    B = V; xtr, ytr, xv, yv, xte, yte = B.data_for('cifar10', B.base(c['act']) + ('_tr' if B.recipe(c['act']) else ''))
    mx.random.seed(c['seed']); m = B.Net(c['depth'], c['act'], c['seed'], din=xtr.shape[1]); mx.eval(m.parameters()); m.eval()
    r0 = np.random.default_rng(c['seed'] + 777); b0 = r0.permutation(len(xtr))[:B.BATCH]
    _ = nn.value_and_grad(m, B.loss_fn)(m, mx.array(xtr[b0]), mx.array(ytr[b0]))
    m.train()
    opt = optim.AdamW(learning_rate=sched(c), betas=[0.9, 0.999], eps=1e-8, weight_decay=c['wd'], bias_correction=True)
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

def selfcheck():
    out = dict(n_runs=len(RUNS), runs=[f"{c['name']}_s{c['seed']}" for c in RUNS], warm=V.WARM, steps=V.STEPS)
    sc = sched(dict(lr=1e-4)); out['lr_0'] = float(sc(0)); out['lr_344'] = float(sc(V.WARM)); out['lr_end'] = float(sc(V.STEPS))
    for c in RUNS: ref_row(c)
    out['ref_rows_found'] = True
    out['d3_results_found'] = all((ST / 'd3' / 'results' / f"d3_{n}_s{s}.json").exists() for n in ('bp200', 'se200') for s in (42192, 42193))
    out['same_consts_as_b15'] = (V.STEPS, V.BATCH, V.NCAL, V.CALS4) == (D3.MODS['b15_run'].STEPS, D3.MODS['b15_run'].BATCH, D3.MODS['b15_run'].NCAL, D3.MODS['b15_run'].CALS4)
    out['ok'] = out['n_runs'] == 4 and out['lr_0'] == 0.0 and abs(out['lr_344'] - 1e-4) < 1e-9 and out['lr_end'] < 1e-9 and out['d3_results_found'] and out['same_consts_as_b15']
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--selfcheck', action='store_true'); a = p.parse_args(); mx.set_cache_limit(V.CACHE_LIMIT)
    if a.selfcheck: print(json.dumps(selfcheck(), indent=1)); return
    (ROOT / 'results').mkdir(exist_ok=True)
    for i, c in enumerate(RUNS):
        f = ROOT / 'results' / f"d3b_{c['name']}_s{c['seed']}.json"
        if i % a.workers != a.worker or f.exists(): continue
        t0 = time.time(); m, status, D = train(c); row = ref_row(c); o = dict(run=c, status=status)
        if status == 'ok':
            got, same, rng = D3.verify(m, D, row); o.update(verify_got=got, verify_same=same, valid=all(same.values()))
            o.update(D3.measure(m, D, rng, c['depth']))
        o['sec'] = round(time.time() - t0, 1); json.dump(o, open(f, 'w'), indent=1)
        print('done', f.name, status, o.get('valid'), datetime.now().strftime('%H:%M:%S'), flush=True)

if __name__ == '__main__': main()
