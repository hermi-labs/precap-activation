import csv, glob, statistics as st, math, sys
from collections import defaultdict
rows = []
for f in glob.glob('experiments_v7_[ABC]_shard*.csv'):
    rows += list(csv.DictReader(open(f, newline='')))
print('rows', len(rows), 'failures', sum(r['nan_or_inf'] == 'True' for r in rows))
K = ['target','depth','lr','init','arch','act','c','in_scale','gamma','p','soft_s']
g = defaultdict(list)
for r in rows:
    g[tuple(r[k] for k in K)].append(float(r['test_loss']))
agg = [(dict(zip(K, k)), st.median(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)) for k, v in g.items() if len(v) >= 3]
print('configs with >=3 seeds', len(agg))
for tgt in sorted(set(a[0]['target'] for a in agg)):
    for d in sorted(set(a[0]['depth'] for a in agg), key=int):
        sub = [a for a in agg if a[0]['target'] == tgt and a[0]['depth'] == d]
        if not sub: continue
        print('\n==', tgt, 'depth', d)
        for act in ['capexp','precap','silu','gelu','relu']:
            s = sorted([a for a in sub if a[0]['act'] == act], key=lambda a: a[1])
            if s:
                b = s[0]; c = b[0]
                print('  %-6s med=%.4f sd=%.4f n=%d | lr=%s init=%s arch=%s c=%s in=%s g=%s ss=%s' % (act, b[1], b[2], b[3], c['lr'], c['init'], c['arch'], c['c'], c['in_scale'], c['gamma'], c['soft_s']))
for act in ['capexp','precap']:
    for key in ['c','in_scale']:
        print('\nbest median per', key, 'for', act, '(per target/depth)')
        for tgt in sorted(set(a[0]['target'] for a in agg)):
            for d in sorted(set(a[0]['depth'] for a in agg), key=int):
                m = defaultdict(lambda: math.inf)
                for a in agg:
                    if a[0]['act'] == act and a[0]['target'] == tgt and a[0]['depth'] == d:
                        m[a[0][key]] = min(m[a[0][key]], a[1])
                if m: print(' ', tgt, d, {k: round(v, 4) for k, v in sorted(m.items(), key=lambda x: float(x[0]))})
