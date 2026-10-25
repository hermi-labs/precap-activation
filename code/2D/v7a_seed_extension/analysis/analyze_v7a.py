import pandas as pd, numpy as np, json
from math import comb
from scipy.stats import wilcoxon, trim_mean
d=pd.read_csv('results/checkpoint_metrics.csv'); f=d[d.step==1200]
sign=lambda w,n: sum(comb(n,k) for k in range(w,n+1))/2**n
out=[]; verdict={}
for (t,dep),g in f.groupby(['task','depth']):
 p=g.pivot(index='seed',columns='act',values='test_mse'); n=len(p); anchor=(dep==16)
 print(f'\n== {t} d{dep} n={n} ==')
 for a in p.columns:
  r=(p[a]>5*p[a].median()).mean(); print(f'  outlier-rate {a}: {r:.2f}  CV={p[a].std()/p[a].mean():.2f}  median={p[a].median():.5f}')
 for v in ['precap_c125','precap_s225']:
  dd=np.log10(p[v]/p['precap_base']); w=int((dd<0).sum()); pw=wilcoxon(dd,alternative='less').pvalue
  ok=(dd.median()<0 and pw<0.025 and w>=13) if not anchor else (dd.median()<=0.05)
  print(f'  {v} vs base: median={dd.median():+.3f} trim10={trim_mean(dd,0.1):+.3f} wins={w}/{n} wilcoxon_p={pw:.4f} sign_p={sign(w,n):.4f} -> {"PASS" if ok else "no"}{" (Anker: nicht schlechter)" if anchor else ""}')
  verdict[(t,dep,v)]=ok; out.append((t,dep,v,dd.median(),trim_mean(dd,0.1),w,n,pw,ok))
 for r in ['silu','gelu']:
  dd=np.log10(p['precap_base']/p[r]); w=int((dd<0).sum()); pw=wilcoxon(dd,alternative='less').pvalue
  ok=dd.median()<0 and w>=13
  print(f'  base vs {r}: median={dd.median():+.3f} trim10={trim_mean(dd,0.1):+.3f} wins={w}/{n} wilcoxon_p={pw:.4f} -> {"PASS" if ok else "no"}')
  out.append((t,dep,'base_vs_'+r,dd.median(),trim_mean(dd,0.1),w,n,pw,ok))
pd.DataFrame(out,columns=['task','depth','comparison','median_d','trim10_d','wins','n','wilcoxon_p','pass']).to_csv('analysis/v7a_results.csv',index=False)
print('\n== Gesamtbewertung ==')
for v in ['precap_c125','precap_s225']:
 m=[verdict[(t,64,v)] for t in ['radial_ripple','ridge_mix']]; an=verdict[('radial_ripple',16,v)]
 lab='ROBUST (beide Tiefe-64-Zellen, Anker ok)' if all(m) and an else ('AUFGABENABHAENGIG (eine Zelle)' if any(m) and an else 'KEIN BELASTBARER EFFEKT')
 print(v,lab)
