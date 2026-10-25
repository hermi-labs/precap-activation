import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));a=ap.parse_args();D=Path(a.dir);rng=np.random.RandomState(1801)
TUNE=list(range(9900,9910));CONF=list(range(9910,9930));FINAL=3000;BAD=1e6
(D/'analysis').mkdir(exist_ok=True)
d=pd.concat([pd.read_csv(p) for p in glob.glob(str(D/'workers'/'w*_checkpoints.csv'))],ignore_index=True)
FINAL=int(d.step.max()) if d.step.max()<3000 else 3000
bad_runs=sorted(d[d.run_status!='ok'].run_id.unique())
def final_table(step):
 runs=d[['run_id','task','depth','lr','act','seed']].drop_duplicates()
 f=d[(d.step==step)&(d.run_status=='ok')][['run_id','test_mse','frac_abs_pre_gt_half','layer_pre_p99','cap_fraction']]
 t=runs.merge(f,on='run_id',how='left');t['nonfinite']=t.test_mse.isna();t['test_mse']=t.test_mse.fillna(BAD);t['l']=np.log10(t.test_mse);return t
def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);out=np.empty(len(p));z=0.
 for i,j in enumerate(o):z=max(z,(len(p)-i)*p[j]);out[j]=min(1,z)
 return out
def wil(x):
 try:return float(wilcoxon(x).pvalue)
 except Exception:return np.nan
def ci(x):
 x=np.asarray(x);b=[np.median(x[rng.randint(0,len(x),len(x))]) for _ in range(2000)];return 10**np.quantile(b,[.05,.95])
T=final_table(FINAL)
# 1) LR-Auswahl nur auf Tuning-Seeds
sel=T[T.seed.isin(TUNE)].groupby(['task','depth','act','lr']).l.median().reset_index()
best=sel.loc[sel.groupby(['task','depth','act']).l.idxmin()][['task','depth','act','lr']].rename(columns={'lr':'lr_sel'})
sel.to_csv(D/'analysis'/'v18_lr_tuning_medians.csv',index=False);best.to_csv(D/'analysis'/'v18_lr_selection.csv',index=False)
# 2) Konfirmatorisch auf Bestaetigungs-Seeds mit gewaehlter LR
C=T[T.seed.isin(CONF)].merge(best,on=['task','depth','act']);C=C[C.lr==C.lr_sel]
P=C.pivot_table(index=['task','depth','seed'],columns='act',values='l')
rows=[]
for (task,depth),g in P.groupby(['task','depth']):
 for ref in ['silu','gelu']:
  x=(g.precap_base-g[ref]).values;lo,hi=ci(x)
  rows.append(dict(task=task,depth=depth,ref=ref,n=len(x),lr_precap=float(best[(best.task==task)&(best.depth==depth)&(best.act=='precap_base')].lr_sel.iloc[0]),lr_ref=float(best[(best.task==task)&(best.depth==depth)&(best.act==ref)].lr_sel.iloc[0]),ratio=10**np.median(x),ci90_lo=lo,ci90_hi=hi,wins=int((x<0).sum()),p=wil(x)))
cf=pd.DataFrame(rows);cf['holm']=holm(cf.p.fillna(1));cf['verdict']=np.where((cf.holm<.05)&(cf.ratio<1),'PreCap besser',np.where((cf.holm<.05)&(cf.ratio>1),'PreCap schlechter','unklar'))
cf.to_csv(D/'analysis'/'v18_confirmatory.csv',index=False)
crit={}
for ref,g in cf.groupby('ref'):
 nb=int((g.verdict=='PreCap besser').sum());nw=int((g.verdict=='PreCap schlechter').sum())
 crit[ref]={'besser':nb,'schlechter':nw,'unklar':int(len(g)-nb-nw),'H_A_bestaetigt':bool(nb>=8 and nw<=1),'je_tiefe':{int(k):v.verdict.value_counts().to_dict() for k,v in g.groupby('depth')}}
# 3) Mechanismus (vorab festgelegtes Muster): Vorsprung gegen SiLU vs. Anteil |pre|>0.5 bei PreCap
cs=cf[cf.ref=='silu'].set_index(['task','depth'])
pm=C[C.act=='precap_base'].groupby(['task','depth'])[['frac_abs_pre_gt_half','layer_pre_p99','cap_fraction']].median()
mm=cs.join(pm);mm['log_ratio']=np.log10(mm.ratio)
q=spearmanr(mm.frac_abs_pre_gt_half,mm.log_ratio);rho=float(q.statistic);p1=float(q.pvalue/2 if rho<0 else 1-q.pvalue/2)
mm.reset_index().to_csv(D/'analysis'/'v18_mechanism.csv',index=False)
# 4) Explorativ: feste LR 1e-3 (wie V14-V17) und Schritt 1200, LR-Empfindlichkeit
ex=[]
for step in [1200,FINAL]:
 Ts=final_table(step);F=Ts[(Ts.lr==1e-3)&Ts.seed.isin(CONF)].pivot_table(index=['task','depth','seed'],columns='act',values='l')
 for (task,depth),g in F.groupby(['task','depth']):
  for ref in ['silu','gelu']:
   x=(g.precap_base-g[ref]).values;ex.append(dict(step=step,lr=1e-3,task=task,depth=depth,ref=ref,ratio=10**np.median(x),wins=int((x<0).sum()),p=wil(x)))
pd.DataFrame(ex).to_csv(D/'analysis'/'v18_fixed_lr_explorative.csv',index=False)
sens=T.groupby(['depth','act','lr']).l.median().unstack('lr');sens.to_csv(D/'analysis'/'v18_lr_sensitivity.csv')
v={'n_rows':len(d),'runs':int(d.run_id.nunique()),'nonfinite_runs':len(bad_runs),'final_step':FINAL,'lr_selection_counts':best.groupby(['act','lr_sel']).size().rename('n').reset_index().to_dict('records'),'confirmatory_counts':cf.verdict.value_counts().to_dict(),'criterion':crit,'mechanism_spearman_frac_gt_half_vs_log_ratio_silu':{'rho':rho,'p_one_sided_rho_lt_0':p1,'n':len(mm),'gestuetzt':bool(rho<0 and p1<.05)}}
(D/'analysis'/'v18_verdict.json').write_text(json.dumps(v,indent=2))
(D/'analysis'/'V18_Auswertung.md').write_text('# V18 Auswertung (automatisch)\n\n## Konfirmatorisch\n\n'+cf.round(4).to_csv(index=False)+'\n\n## Verdikt\n\n'+json.dumps(v,indent=2)+'\n\n## LR-Empfindlichkeit (Median log10 Test-MSE, alle Seeds)\n\n'+sens.round(3).to_csv()+'\n')
print(json.dumps(v,indent=2))
