# Q1 Auswertung nach Praeregistrierung (Q1_Praeregistrierung.md)
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, spearmanr
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
rows = []
for f in sorted((ROOT / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
CELLS = [(ds, d) for d in (50, 100) for ds in ('mnist', 'fashion')]
PRIM = ['w8a8', 'w8a4']; ALLQ = ['w8a8', 'w8a4', 'w4a8', 'fp8', 'w8a8_p9999']
def delta(r, q):
    if r['status'] != 'ok': return None
    a, b = float(r[f'test_err_{q}']), float(r['test_err_fp32']); return (a - b) * 100
D = {}
for r in rows: D[(r['dataset'], int(r['depth']), r['act'], int(r['seed']))] = r
seeds = sorted({int(r['seed']) for r in rows})
def test(ds, d, cand, ref, q):
    x = []
    for s in seeds:
        a, b = D.get((ds, d, cand, s)), D.get((ds, d, ref, s))
        if a is None or b is None: continue
        da, db = delta(a, q), delta(b, q)
        if da is None or db is None: continue
        x.append(da - db)
    x = np.array(x)
    if len(x) == 0 or np.all(x == 0): p = 1.0
    else: p = float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, cand=cand, ref=ref, quant=q, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    return res
def verdict(r): return 'unklar' if r['holm'] >= 0.05 else ('SPreCap robuster' if r['med_diff_pp'] < 0 else ('SPreCap anfaelliger' if r['med_diff_pp'] > 0 else 'unklar'))
prim = holm([test(ds, d, 'spreccap_cs', 'selu', q) for q in PRIM for ds, d in CELLS])
for r in prim: r['verdict'] = verdict(r)
nr = sum(r['verdict'] == 'SPreCap robuster' for r in prim); na = sum(r['verdict'] == 'SPreCap anfaelliger' for r in prim)
overall = 'bestaetigt' if nr >= 2 and na == 0 else ('widerlegt' if na >= 1 and nr == 0 else 'offen')
sec = []
for q in ALLQ:
    fam = [test(ds, d, c, rf, q) for ds, d in CELLS for c, rf in [('spreccap_e03', 'selu'), ('spreccap_cs', 'elu'), ('spreccap_e03', 'elu')] + ([] if q in PRIM else [('spreccap_cs', 'selu')])]
    for r in holm(fam): r['verdict'] = verdict(r); sec.append(r)
desc = []
for ds, d in CELLS:
    for a in ('spreccap_cs', 'spreccap_e03', 'selu', 'elu'):
        rr = [D[k] for k in D if k[:3] == (ds, d, a)]; ok = [r for r in rr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=a, n=len(rr), failed=len(rr) - len(ok), med_test_err_fp32=float(np.median([float(r['test_err_fp32']) * 100 for r in ok])) if ok else None,
                 med_range=float(np.median([float(r['act_range_max_median']) for r in ok])) if ok else None)
        for q in ALLQ: e[f'med_delta_{q}'] = float(np.median([delta(r, q) for r in ok])) if ok else None
        desc.append(e)
ok = [r for r in rows if r['status'] == 'ok']
rho = {q: float(spearmanr([float(r['act_range_max_median']) for r in ok], [delta(r, q) for r in ok]).statistic) for q in ALLQ} if len(ok) > 3 else {}
out = dict(n_rows=len(rows), primary=prim, overall=overall, secondary=sec, descriptive=desc, spearman_range_vs_delta=rho)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'q1_result.json', 'w'), indent=1)
for name, L in [('primary', prim), ('secondary', sec), ('descriptive', desc)]:
    with open(ROOT / 'analysis' / f'q1_{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(L[0].keys())); w.writeheader(); w.writerows(L)
print('Gesamturteil:', overall)
for r in prim: print(f"{r['dataset']:7s} d{r['depth']:3d} {r['quant']:5s} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
