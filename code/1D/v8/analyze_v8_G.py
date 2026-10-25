import csv,glob,math,statistics as st,collections,os
K=['target','act','c','in_scale','gamma','soft_s','depth','arch','init_k','lr','clip','warmup','seed']
D={}
for f in glob.glob('experiments_v8_G_shard*.csv'):
    for r in csv.DictReader(open(f)):
        D[tuple(r[k] for k in K)] = r
R=list(D.values())
def loss(r):
    try: z=float(r['test_loss'])
    except: return math.inf
    return z if r['nan_or_inf']=='False' and math.isfinite(z) else math.inf
def stat(rs):
    v=[loss(r) for r in rs]; ok=[x for x in v if math.isfinite(x)]
    return {'n':len(v),'fail':len(v)-len(ok),'median':st.median(v),'mean':(st.mean(ok) if ok else math.inf),'sd':(st.pstdev(ok) if len(ok)>1 else 0.0),'best':(min(ok) if ok else math.inf)}
def fmt(x):
    if not math.isfinite(x): return 'FAIL'
    if x>=1e4:return '%.2e'%x
    return '%.4f'%x
out=[]
out.append('V8 STAGE G — FAIR ACTIVATION COMPARISON')
out.append('Unique runs: %d / expected 3780'%len(R))
out.append('Rows use test_loss; failures are NaN/Inf only. Medians include failures as infinity.')
# Full config stats
G=collections.defaultdict(list)
for r in R:
    G[tuple(r[k] for k in ['target','depth','arch','act','c','in_scale','gamma','soft_s','init_k','lr'])].append(r)
with open('v8_G_config_summary.csv','w',newline='') as f:
    fields=['target','depth','arch','act','c','in_scale','gamma','soft_s','init_k','lr','n','fail','median_test_loss','mean_finite_test_loss','sd_finite_test_loss','best_seed_test_loss']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for k,rs in sorted(G.items()):
        s=stat(rs); w.writerow(dict(zip(fields[:10],k))|{'n':s['n'],'fail':s['fail'],'median_test_loss':s['median'],'mean_finite_test_loss':s['mean'],'sd_finite_test_loss':s['sd'],'best_seed_test_loss':s['best']})
# One best stable config each activation (0 failures, then min median); fallback min failure then median
out.append('\n== Best configuration per activation (per task/depth/architecture) ==')
headers=['target','depth','arch','act','c','a','gamma','s','k','lr','median','sd','fail/n']
out.append(' | '.join(headers))
best_rows=[]
for target in ['sin3_quad','sin_mix','abs_wave']:
 for depth in ['16','32','64']:
  for arch in ['plain','res_inv']:
   candidates=[]
   for k,rs in G.items():
    if k[:3]!=(target,depth,arch):continue
    s=stat(rs);candidates.append((s['fail'],s['median'],k,s))
   for fail,med,k,s in sorted(candidates,key=lambda z:(z[0]>0,z[1])):
    if s['fail']==0:break
   fields=dict(zip(['target','depth','arch','act','c','a','gamma','soft_s','k','lr'],k))
   best_rows.append(fields|s)
   out.append('%s d%s %s | %s c=%s a=%s g=%s s=%s k=%s lr=%s | %s | %s | %d/%d'%(target,depth,arch,fields['act'],fields['c'],fields['a'],fields['gamma'],fields['soft_s'],fields['k'],fields['lr'],fmt(s['median']),fmt(s['sd']),s['fail'],s['n']))
# Head-to-head winner per cell, using best stable config per activation
out.append('\n== Head-to-head winner per task/depth/architecture ==')
wins=collections.Counter(); rank_medians=collections.defaultdict(list)
for target in ['sin3_quad','sin_mix','abs_wave']:
 for depth in ['16','32','64']:
  for arch in ['plain','res_inv']:
   arows=[x for x in best_rows if x['target']==target and x['depth']==depth and x['arch']==arch]
   arows.sort(key=lambda x:(x['fail']>0,x['median']))
   win=arows[0];wins[win['act']]+=1
   leaderboard=', '.join('%s %s%s'%(x['act'],fmt(x['median']),'*' if x['fail'] else '') for x in arows)
   out.append('%-9s d%-2s %-7s: WIN %-7s %s | %s'%(target,depth,arch,win['act'],fmt(win['median']),leaderboard))
   for x in arows:
    if x['fail']==0: rank_medians[x['act']].append(x['median'])
out.append('\nWins: '+', '.join('%s=%d'%(a,wins[a]) for a in ['capexp','precap','silu','gelu','relu']))
out.append('Geometric median loss across stable best-cells: '+', '.join('%s=%s'%(a,fmt(math.exp(st.median([math.log(max(v,1e-30)) for v in vs]))) if vs else 'FAIL') for a,vs in rank_medians.items()))
# Stability aggregation for activations across all runs and by arch
out.append('\n== NaN/Inf stability (all G runs) ==')
for keys,label in [(['act'],'activation'),(['act','arch'],'activation / architecture'),(['act','depth'],'activation / depth')]:
 gg=collections.defaultdict(list)
 for r in R:gg[tuple(r[x] for x in keys)].append(r)
 out.append(label+':')
 for k,rs in sorted(gg.items()):
  s=stat(rs);out.append('  %s: %d/%d NaN/Inf (%.1f%%), finite median %s'%(' / '.join(k),s['fail'],s['n'],100*s['fail']/s['n'],fmt(st.median([loss(r) for r in rs if math.isfinite(loss(r))]) if s['fail']<s['n'] else math.inf)))
# highest LR with all seeds finite, per activation aggregate
out.append('\n== Learning-rate tolerance (configurations with all five seeds finite) ==')
for act in ['capexp','precap','silu','gelu','relu']:
 bylr=collections.defaultdict(lambda:[0,0])
 for k,rs in G.items():
  if k[3]!=act:continue
  s=stat(rs); bylr[k[-1]][0]+=1;bylr[k[-1]][1]+=(s['fail']==0)
 out.append('  %s: '%act+', '.join('lr=%s: %d/%d stable'%(lr,v[1],v[0]) for lr,v in sorted(bylr.items(),key=lambda x:float(x[0]))))
# write
open('v8_G_summary.txt','w').write('\n'.join(out)+'\n')
print('\n'.join(out))
