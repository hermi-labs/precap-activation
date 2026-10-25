import csv, glob, math, statistics as st, collections
K = ['target','act','c','in_scale','gamma','soft_s','depth','arch','init_k','lr','clip','warmup','seed']
seen = {}
for f in glob.glob('experiments_v8_E_shard*.csv'):
    for r in csv.DictReader(open(f)):
        seen[tuple(r[k] for k in K)] = r
rows = list(seen.values())
print('unique runs', len(rows), 'of 1008')
def val(r):
    try: x = float(r['test_loss'])
    except: return math.inf
    return x if (math.isfinite(x) and r['nan_or_inf'] != 'True') else math.inf
g = collections.defaultdict(list)
for r in rows:
    g[(r['target'], r['depth'], r['arch'], float(r['c']), float(r['in_scale']))].append(val(r))
def cell(v):
    bad = sum(1 for x in v if x == math.inf)
    m = st.median(v)
    return ('%.4f' % m if m < 1e3 else '>1e3') if m != math.inf else 'FAIL', bad, len(v)
for t in ['sin3_quad','sin_mix','abs_wave']:
    for d in ['16','32']:
        for a in ['plain','res_inv']:
            print('\n%s depth %s %s  (median test loss [fails/n]) a=1.0 | a=1.1' % (t, d, a))
            for c in [0.6,0.8,1.0,1.25,1.5,2.0,2.5]:
                out = []
                for s in [1.0, 1.1]:
                    v = g.get((t,d,a,c,s))
                    if v:
                        m,b,n = cell(v); out.append('%8s [%d/%d]' % (m,b,n))
                print('  c=%-4s %s' % (c, ' | '.join(out)))
print('\nBEST per target/depth/arch (min median, finite):')
for t in ['sin3_quad','sin_mix','abs_wave']:
    for d in ['16','32']:
        for a in ['plain','res_inv']:
            cands = [(st.median(v), k) for k, v in g.items() if k[0]==t and k[1]==d and k[2]==a and len(v)>=3]
            if cands:
                m, k = min(cands)
                v = g[k]; fin = [x for x in v if math.isfinite(x)]
                print('  %-9s d%-2s %-7s med=%.4f sd=%.4f best c=%s a=%s fails=%d/%d' % (t,d,a,m,(st.pstdev(fin) if len(fin)>1 else float('nan')),k[3],k[4],len(v)-len(fin),len(v)))
print('\nFAIL threshold (plain): share of failed runs by depth, c, a (a*e^c)')
for d in ['16','32']:
    for c in [1.5,2.0,2.5]:
        for s in [1.0,1.1]:
            v = [x for (t,dd,aa,cc,ss),vv in g.items() if dd==d and aa=='plain' and cc==c and ss==s for x in vv]
            if v: print('  d%s c=%s a=%s gain=%.2f fail %d/%d' % (d,c,s,s*math.exp(c),sum(1 for x in v if x==math.inf),len(v)))
