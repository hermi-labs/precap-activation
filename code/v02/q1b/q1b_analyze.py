# Q1b Auswertung nach Praeregistrierung (Q1b_Praeregistrierung.md)
# Aufruf: python q1b_analyze.py [Q1b-Ordner] [Q1-workers-Ordner fuer die Wiederholungskontrolle]
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
Q1W = Path(sys.argv[2]) if len(sys.argv) > 2 else None
def readall(d):
    rows = []
    for f in sorted(Path(d).glob('w*.csv')): rows += list(csv.DictReader(open(f)))
    return rows
rows = readall(ROOT / 'workers')
CELLS = [(ds, d) for d in (50, 100) for ds in ('mnist', 'fashion')]
SECQ = ['w8a4_mse', 'w8a4_p999', 'w8a4_p9999', 'w8a8_mse']
D = {(r['dataset'], int(r['depth']), r['act'], int(r['seed'])): r for r in rows}
seeds = sorted({int(r['seed']) for r in rows})
def delta(r, q):
    if r['status'] != 'ok': return None
    return (float(r[f'test_err_{q}']) - float(r['test_err_fp32'])) * 100
def test(ds, d, cand, ref, q):
    x = []
    for s in seeds:
        a, b = D.get((ds, d, cand, s)), D.get((ds, d, ref, s))
        if a is None or b is None: continue
        da, db = delta(a, q), delta(b, q)
        if da is None or db is None: continue
        x.append(da - db)
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, cand=cand, ref=ref, quant=q, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    return res
def verdict(r): return 'unklar' if r['holm'] >= 0.05 else ('SPreCap robuster' if r['med_diff_pp'] < 0 else ('SPreCap anfaelliger' if r['med_diff_pp'] > 0 else 'unklar'))
prim = holm([test(ds, d, 'spreccap_cs', 'selu', 'w8a4_best') for ds, d in CELLS])
for r in prim: r['verdict'] = verdict(r)
nr = sum(r['verdict'] == 'SPreCap robuster' for r in prim); na = sum(r['verdict'] == 'SPreCap anfaelliger' for r in prim)
overall = 'Vorteil bleibt' if nr >= 2 and na == 0 else ('Vorteil kehrt sich um' if na >= 1 and nr == 0 else ('Vorteil verschwindet' if nr == 0 else 'gemischt'))
sec = []
fam = holm([test(ds, d, c, rf, 'w8a4_best') for ds, d in CELLS for c, rf in [('spreccap_e03', 'selu'), ('spreccap_cs', 'elu'), ('spreccap_e03', 'elu')]])
for r in fam: r['verdict'] = verdict(r); sec.append(r)
for q in SECQ:
    fam = holm([test(ds, d, c, rf, q) for ds, d in CELLS for c, rf in [('spreccap_cs', 'selu'), ('spreccap_e03', 'selu'), ('spreccap_cs', 'elu'), ('spreccap_e03', 'elu')]])
    for r in fam: r['verdict'] = verdict(r); sec.append(r)
desc = []
for ds, d in CELLS:
    for a in ('spreccap_cs', 'spreccap_e03', 'selu', 'elu'):
        ok = [D[k] for k in D if k[:3] == (ds, d, a) and D[k]['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=a, n_ok=len(ok), med_test_err_fp32=float(np.median([float(r['test_err_fp32']) * 100 for r in ok])),
                 chosen=dict(Counter(r['w8a4_best_cal'] for r in ok)), med_clip_mse4=float(np.median([float(r['act_clip_mse4_median']) for r in ok])),
                 med_range_max=float(np.median([float(r['act_range_max_median']) for r in ok])))
        for q in ['w8a8', 'w8a4', 'w4a8', 'w8a4_best'] + SECQ: e[f'med_delta_{q}'] = float(np.median([delta(r, q) for r in ok]))
        desc.append(e)
rep = None
if Q1W:
    Q = {r['run_id']: r for r in readall(Q1W)}; cols = ['test_err_fp32', 'test_err_w8a8', 'test_err_w8a4', 'test_err_w4a8']; same = tot = 0; mx_ = 0.0
    for r in rows:
        q = Q.get(r['run_id'])
        if q is None or r['status'] != 'ok' or q['status'] != 'ok': continue
        for c in cols:
            tot += 1; dv = abs(float(r[c]) - float(q[c])) * 100; same += dv < 1e-9; mx_ = max(mx_, dv)
    rep = dict(n_values=tot, identical_share=same / tot if tot else None, max_abs_diff_pp=mx_)
out = dict(n_rows=len(rows), failed=sum(r['status'] != 'ok' for r in rows), primary=prim, overall=overall, secondary=sec, descriptive=desc, repro_vs_q1=rep)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'q1b_result.json', 'w'), indent=1)
print('Gesamturteil:', overall, '| Wiederholung:', rep)
for r in prim: print(f"{r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
