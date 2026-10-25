import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));a=ap.parse_args();D=Path(a.dir);rng=np.random.RandomState(1701)
d=pd.concat([pd.read_csv(p) for p in glob.glob(str(D/'workers'/'w*_checkpoints.csv'))],ignore_index=True)
def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);a=np.empty(len(p));z=0.
 for i,j in enumerate(o):z=max(z,(len(p)-i)*p[j]);a[j]=min(1,z)
 return a
def wil(x):
 try:return wilcoxon(x).pvalue
 except:return np.nan
def ci(x):
 x=np.asarray(x);b=[np.median(x[rng.randint(0,len(x),len(x))]) for _ in range(2000)];return 10**np.quantile(b,[.05,.95])
rows=[]
for step in [1200,3000]:
 f=d[(d.step==step)&(d.run_status=='ok')];T=f.pivot_table(index=['task','depth','scale','seed'],columns='act',values='test_mse')
 for (task,depth,scale),g in T.groupby(['task','depth','scale']):
  for ref in ['silu','gelu']:
   x=np.log10(g.precap_base/g[ref]).values;lo,hi=ci(x)
   rows.append(dict(step=step,task=task,depth=depth,scale=scale,ref=ref,n=len(x),ratio=10**np.median(x),ci90_lo=lo,ci90_hi=hi,wins=int((x<0).sum()),p=wil(x)))
r=pd.DataFrame(rows);final=r[r.step==3000].copy();final['holm']=holm(final.p.fillna(1));final['verdict']=np.where((final.holm<.05)&(final.ratio<1),'PreCap besser',np.where((final.holm<.05)&(final.ratio>1),'PreCap schlechter','unklar'))
r.to_csv(D/'analysis'/'v17_all.csv',index=False);final.to_csv(D/'analysis'/'v17_confirmatory.csv',index=False)
m=d[d.step.isin([1200,3000])].groupby(['depth','scale','act','step'])[['test_mse','global_grad_norm','pre_rms','post_rms_scaled','frac_abs_pre_gt_half','cap_fraction','layer_pre_p99']].median().reset_index();m.to_csv(D/'analysis'/'v17_mechanism.csv',index=False)
# scale causality: correlation in PreCap only across task/depth/scale at final
f=d[(d.step==3000)&(d.act=='precap_base')];z=f.groupby(['task','depth','scale']).median(numeric_only=True);q=spearmanr(z.post_rms_scaled,z.test_mse) if len(z)>3 else None
v={'n_rows':len(d),'complete_ok_runs':int(d[d.step==3000].run_id.nunique()),'final_counts':final.verdict.value_counts().to_dict(),'by_scale':{s:g.verdict.value_counts().to_dict() for s,g in final.groupby('scale')},'post_rms_vs_test_mse':None if q is None else {'rho':float(q.statistic),'p':float(q.pvalue),'n':len(z)}}
(D/'analysis'/'v17_verdict.json').write_text(json.dumps(v,indent=2));(D/'analysis'/'V17_Auswertung.md').write_text('# V17 Auswertung\n\n'+final.round(4).to_csv(index=False)+'\n\n'+json.dumps(v,indent=2));print(json.dumps(v,indent=2))
