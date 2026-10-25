import csv, glob, math, statistics as st
from collections import defaultdict

def load(pat):
    rows = []
    for f in glob.glob(pat):
        rows += list(csv.DictReader(open(f, newline='')))
    return rows

def val(r):
    try:
        x = float(r['test_loss'])
    except Exception:
        return math.inf
    return x if math.isfinite(x) and r['nan_or_inf'] != 'True' else math.inf

K = ['target','depth','lr','init','arch','act','c','in_scale','gamma','p','soft_s']

def agg(rows, minn=3):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in K)].append(val(r))
    out = []
    for k, v in g.items():
        if len(v) < minn: continue
        fin = [x for x in v if math.isfinite(x)]
        out.append((dict(zip(K, k)), st.median(v), (st.pstdev(fin) if len(fin) > 1 else float('nan')), len(v), len(v) - len(fin)))
    return out

def fmt(a):
    c = a[0]
    return 'med=%.4f sd=%.4f n=%d fail=%d | lr=%s init=%s arch=%s c=%s in=%s g=%s ss=%s' % (a[1], a[2], a[3], a[4], c['lr'], c['init'], c['arch'], c['c'], c['in_scale'], c['gamma'], c['soft_s'])

rows = load('experiments_v7_[ABC]_shard*.csv')
A = agg(rows)
print('rows', len(rows), 'configs>=3 seeds', len(A))
print('\nFAIL RATE per act/depth/arch/init (runs):')
fr = defaultdict(lambda: [0, 0])
for r in rows:
    k = (r['act'], r['depth'], r['arch'], r['init'])
    fr[k][1] += 1; fr[k][0] += (val(r) == math.inf)
for k, (b, n) in sorted(fr.items()):
    if b: print(' ', k, '%d/%d' % (b, n))

for lrsel in ['0.001']:
  for tgt in ['sin3_quad','sin_mix','abs_wave']:
    for d in ['16','32','64']:
        sub = [a for a in A if a[0]['target'] == tgt and a[0]['depth'] == d and a[0]['lr'] == lrsel]
        if not sub: continue
        print('\n== %s depth %s lr=%s (best per act)' % (tgt, d, lrsel))
        for act in ['capexp','precap','silu','gelu','relu']:
            s = sorted([a for a in sub if a[0]['act'] == act], key=lambda a: a[1])
            if s: print('  %-6s %s' % (act, fmt(s[0])))

print('\n== LR sweep stage C (best per act, per target/depth)')
C = agg(load('experiments_v7_C_shard*.csv'))
for tgt in ['sin3_quad','sin_mix','abs_wave']:
    for d in ['16','32']:
        for lr in ['0.0003','0.001','0.003']:
            sub = [a for a in C if a[0]['target'] == tgt and a[0]['depth'] == d and a[0]['lr'] == lr]
            line = []
            for act in ['capexp','precap','silu','gelu','relu']:
                s = sorted([a for a in sub if a[0]['act'] == act], key=lambda a: a[1])
                if s: line.append('%s=%.4f' % (act, s[0][1]))
            print(' ', tgt, d, lr, ' '.join(line))

print('\n== best median per c / in_scale (capexp, precap) stage A, lr=1e-3')
for act in ['capexp','precap']:
    for key in ['c','in_scale','init','arch']:
        print(' ', act, 'by', key)
        for tgt in ['sin3_quad','sin_mix']:
            for d in ['16','32']:
                m = {}
                for a in A:
                    c = a[0]
                    if c['act'] == act and c['target'] == tgt and c['depth'] == d and c['lr'] == '0.001':
                        m[c[key]] = min(m.get(c[key], math.inf), a[1])
                print('    ', tgt, d, {k: round(v, 4) for k, v in sorted(m.items())})

print('\n== precap: gamma / soft_s effect (best median)')
for key in ['gamma','soft_s']:
    for tgt in ['sin3_quad','sin_mix']:
        for d in ['16','32']:
            m = {}
            for a in A:
                c = a[0]
                if c['act'] == 'precap' and c['target'] == tgt and c['depth'] == d and c['lr'] == '0.001':
                    m[c[key]] = min(m.get(c[key], math.inf), a[1])
            print('  ', key, tgt, d, {k: round(v, 4) for k, v in sorted(m.items())})

print('\n== Stage D long training (3000 steps), best per act')
D = agg(load('experiments_v7_long_D_shard*.csv'))
for tgt in ['sin3_quad','sin_mix']:
    for d in ['16','32']:
        sub = [a for a in D if a[0]['target'] == tgt and a[0]['depth'] == d]
        for act in ['capexp','precap','silu','gelu','relu']:
            s = sorted([a for a in sub if a[0]['act'] == act], key=lambda a: a[1])
            if s: print(' ', tgt, d, act, fmt(s[0]))
