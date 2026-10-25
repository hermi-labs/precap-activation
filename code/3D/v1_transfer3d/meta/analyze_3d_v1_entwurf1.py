import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));a=ap.parse_args();D=Path(a.dir);rng=np.random.RandomState(3101)
TUNE=list(range(30000,30010));CONF=list(range(30010,30030));BAD=1e6;K=['task','depth','norm']
(D/'analysis').mkdir(exist_ok=True)
d=pd.concat([pd.read_csv(p) for p in glob.glob(str(D/'workers'/'w*_checkpoints.csv'))],ignore_index=True)
FINAL=min(3000,int(d.step.max()))
def table(step):
 runs=d[['run_id']+K+['lr','act','seed']].drop_duplicates()
 f=d[(d.step==step)&(d.run_status=='ok')][['run_id','test_mse','frac_abs_pre_gt_half','layer_pre_p99','cap_fraction']]
 t=runs.merge(f,on='run_id',how='left');t['test_mse']=t.test_mse.fillna(BAD);t['l']=np.log10(t.test_mse);return t
def holm(p):
 p=np.asarray(p,float);o=np.argsort(p);out=np.empty(len(p));z=0.
 for i,j in enumerate(o):z=max(z,(len(p)-i)*p[j]);out[j]=min(1,z)
 return out
def wil(x):
 try:return float(wilcoxon(x).pvalue)
 except Exception:return np.nan
def ci(x):
 x=np.asarray(x);b=[np.median(x[rng.randint(0,len(x),len(x))]) for _ in range(2000)];return 10**np.quantile(b,[.05,.95])
T=table(FINAL)
sel=T[T.seed.isin(TUNE)].groupby(K+['act','lr']).l.median().reset_index()
best=sel.loc[sel.groupby(K+['act']).l.idxmin()][K+['act','lr']].rename(columns={'lr':'lr_sel'})
best.to_csv(D/'analysis'/'v1_lr_selection.csv',index=False)
C=T[T.seed.isin(CONF)].merge(best,on=K+['act']);C=C[C.lr==C.lr_sel]
P=C.pivot_table(index=K+['seed'],columns='act',values='l')
rows=[]
for key,g in P.groupby(K):
 for ref in ['silu','gelu']:
  x=(g.precap_base-g[ref]).values;lo,hi=ci(x)
  rows.append(dict(zip(K,key),ref=ref,n=len(x),ratio=10**np.median(x),ci90_lo=lo,ci90_hi=hi,wins=int((x<0).sum()),p=wil(x)))
cf=pd.DataFrame(rows);cf['holm']=holm(cf.p.fillna(1));cf['verdict']=np.where((cf.holm<.05)&(cf.ratio<1),'PreCap besser',np.where((cf.holm<.05)&(cf.ratio>1),'PreCap schlechter','unklar'))
cf.to_csv(D/'analysis'/'v1_confirmatory.csv',index=False)
crit={}
for (norm,ref),g in cf.groupby(['norm','ref']):
 nb=int((g.verdict=='PreCap besser').sum());nw=int((g.verdict=='PreCap schlechter').sum())
 crit[f'{norm}/{ref}']={'besser':nb,'schlechter':nw,'unklar':int(len(g)-nb-nw),'Vorteil_bestaetigt':bool(nb>=6 and nw<=1)}
# Explorativ: Aendert LayerNorm den Vorsprung? (gepaarte Differenz je Seed)
inter=[]
for (task,depth),g in P.reset_index().groupby(['task','depth']):
 w=g.pivot_table(index='seed',columns='norm',values=['precap_base','silu','gelu'])
 for ref in ['silu','gelu']:
  x=((w['precap_base']['layernorm']-w[ref]['layernorm'])-(w['precap_base']['none']-w[ref]['none'])).dropna().values
  inter.append(dict(task=task,depth=depth,ref=ref,median_change_log10=float(np.median(x)),p=wil(x)))
pd.DataFrame(inter).to_csv(D/'analysis'/'v1_layernorm_interaction_explorative.csv',index=False)
mech=C[C.act=='precap_base'].groupby(K)[['frac_abs_pre_gt_half','layer_pre_p99','cap_fraction']].median().reset_index()
mech.to_csv(D/'analysis'/'v1_mechanism.csv',index=False)
mm=cf[cf.ref=='silu'].merge(mech,on=K);q=spearmanr(mm.frac_abs_pre_gt_half,np.log10(mm.ratio))
v={'n_rows':len(d),'runs':int(d.run_id.nunique()),'nonfinite_runs':int(d[d.run_status!='ok'].run_id.nunique()),'final_step':FINAL,'confirmatory_counts':cf.verdict.value_counts().to_dict(),'criterion':crit,'frac_gt_half_by_norm':mech.groupby('norm').frac_abs_pre_gt_half.median().to_dict(),'spearman_frac_vs_log_ratio_silu':{'rho':float(q.statistic),'p_two_sided':float(q.pvalue),'n':len(mm)}}
(D/'analysis'/'v1_verdict.json').write_text(json.dumps(v,indent=2))
(D/'analysis'/'V1_3D_Auswertung.md').write_text('# 3D-V1 Auswertung (automatisch)\n\n'+cf.round(4).to_csv(index=False)+'\n\n'+json.dumps(v,indent=2)+'\n')
print(json.dumps(v,indent=2))
