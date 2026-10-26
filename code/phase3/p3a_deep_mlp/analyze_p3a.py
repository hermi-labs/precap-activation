# Phase 3a - praeregistrierte Auswertung (vor dem Start eingefroren, siehe docs/P3A_Praeregistrierung.md)
import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--final',type=int,default=860)
ap.add_argument('--tune',default='40000-40004');ap.add_argument('--conf',default='40010-40024');a=ap.parse_args()
rng=lambda s:list(range(int(s.split('-')[0]),int(s.split('-')[1])+1))
D=Path(a.dir);OUT=D/'analysis';OUT.mkdir(exist_ok=True);TUNE=rng(a.tune);CONF=rng(a.conf);K=['dataset','depth']
CANDS=['precap0','spreccap','spreccap_e03'];REFS=['selu','relu','gelu','silu','elu','tanh']
PRIMARY=[('spreccap','selu'),('precap0','silu'),('precap0','gelu')]
d=pd.concat([pd.read_csv(p) for p in glob.glob(str(D/'workers'/'w*.csv'))],ignore_index=True)
runs=d[['run_id']+K+['act','lr','seed']].drop_duplicates()
assert runs.run_id.is_unique
fin=d[(d.step==a.final)&(d.run_status=='ok')][['run_id','val_acc','test_acc','test_loss']]
T=runs.merge(fin,on='run_id',how='left');T['diverged']=T.test_acc.isna()
T['val_acc']=T.val_acc.fillna(0.1);T['test_acc']=T.test_acc.fillna(0.1);T['err']=1-T.test_acc   # divergiert = Zufallsniveau
Tu=T[T.seed.isin(TUNE)];Co=T[T.seed.isin(CONF)]
# 1) LR-Auswahl (Tuning-Seeds, mediane val_acc, Gleichstand -> kleinere LR) und Abgleich mit der Datei, die den Bestaetigungslauf gesteuert hat
med=Tu.groupby(K+['act','lr']).val_acc.median().reset_index()
best=med.sort_values(K+['act','val_acc','lr'],ascending=[True,True,True,False,True]).groupby(K+['act']).head(1)[K+['act','lr']].rename(columns={'lr':'lr_sel'})
best.to_csv(OUT/'p3a_lr_selection.csv',index=False)
sf=D/'meta'/'lr_selection.json';sel_match=None
if sf.exists():
    S=json.load(open(sf));sel_match=all(abs(S[f'{r.dataset}|{r.depth}|{r.act}']-r.lr_sel)<1e-12 for r in best.itertuples())
C=Co.merge(best,on=K+['act']);assert (abs(C.lr-C.lr_sel)<1e-12).all(),'Bestaetigungslauf nicht mit gewaehlter LR'
P=C.pivot_table(index=K+['seed'],columns='act',values='err')
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);out=np.empty(len(p));z=0.
    for i,j in enumerate(o):z=max(z,(len(p)-i)*p[j]);out[j]=min(1,z)
    return out
rows=[]
for key,g in P.groupby(K):
    for c in CANDS:
        for r in REFS:
            x=(g[c]-g[r]).values
            try:p=float(wilcoxon(x,zero_method='zsplit').pvalue) if np.any(x!=0) else 1.0
            except Exception:p=1.0
            rows.append(dict(dataset=key[0],depth=key[1],cand=c,ref=r,primary=(c,r) in PRIMARY,n=len(x),med_err_cand=float(np.median(g[c])),med_err_ref=float(np.median(g[r])),med_diff_pp=float(np.median(x))*100,wins=int((x<0).sum()),losses=int((x>0).sum()),p=p))
R=pd.DataFrame(rows);R['holm_all']=holm(R.p);R['holm_primary']=np.nan
R.loc[R.primary,'holm_primary']=holm(R.loc[R.primary,'p'])
def verd(h,dif):return np.where(h<0.05,np.where(dif<0,'Kandidat besser',np.where(dif>0,'Kandidat schlechter','unklar')),'unklar')
R['verdict_primary']=np.where(R.primary,verd(R.holm_primary.fillna(1),R.med_diff_pp),'')
R['verdict_all']=verd(R.holm_all,R.med_diff_pp)
R.to_csv(OUT/'p3a_confirmatory.csv',index=False)
ncell=R[K].drop_duplicates().shape[0];need=int(np.ceil(2*ncell/3))
def crit(g,col):
    b=int((g[col]=='Kandidat besser').sum());w=int((g[col]=='Kandidat schlechter').sum())
    return dict(besser=b,schlechter=w,unklar=int(len(g)-b-w),Vorteil_bestaetigt=bool(b>=need and w==0),Nachteil_bestaetigt=bool(w>=need and b==0))
prim={f'{c}/{r}':crit(R[(R.cand==c)&(R.ref==r)],'verdict_primary') for c,r in PRIMARY}
allc={f'{c}/{r}':crit(g,'verdict_all') for (c,r),g in R.groupby(['cand','ref'])}
# 2) Sekundaer (deskriptiv): LR-Fenster auf Tuning-Seeds, Divergenzraten, Rasterrand, Rang
win=med.groupby(K+['act']).apply(lambda g:int((g.val_acc>=g.val_acc.max()-0.005).sum()),include_groups=False).rename('lr_window_0.5pp')
div_t=Tu.groupby(K+['act']).diverged.mean().rename('divergence_rate_tune');div_c=Co.groupby(K+['act']).diverged.mean().rename('divergence_rate_conf')
macc=P.groupby(K).median().stack().rename('median_test_err_conf')
sec=pd.concat([macc,win,div_t,div_c],axis=1).reset_index().rename(columns={'level_2':'act'});sec.to_csv(OUT/'p3a_secondary.csv',index=False)
med.pivot_table(index=K+['act'],columns='lr',values='val_acc').reset_index().to_csv(OUT/'p3a_tune_median_val_acc_by_lr.csv',index=False)
cnt=lambda m:{k:int(v) for k,v in best.assign(e=m).groupby('act').e.sum().items()}
v=dict(n_rows=len(d),runs=len(runs),tune_runs=len(Tu),conf_runs=len(Co),diverged_runs=int(T.diverged.sum()),cells=ncell,need_cells=need,lr_selection_matches_file=sel_match,
       primary=prim,all_pairs_counts={k:int(x) for k,x in R.verdict_all.value_counts().items()},all_pairs=allc,
       lr_at_upper_edge=cnt(best.lr_sel==3e-2),lr_at_lower_edge=cnt(best.lr_sel==1e-4),
       mean_lr_window=sec.groupby('act')['lr_window_0.5pp'].mean().round(2).to_dict(),
       mean_divergence_tune=sec.groupby('act').divergence_rate_tune.mean().round(3).to_dict(),
       mean_rank_conf=sec.pivot_table(index=K,columns='act',values='median_test_err_conf').rank(axis=1).mean().round(2).to_dict())
json.dump(v,open(OUT/'p3a_verdict.json','w'),indent=2,ensure_ascii=False);print(json.dumps(v,indent=2,ensure_ascii=False))
open(OUT/'P3a_Auswertung.md','w').write('\n'.join(['# Phase 3a - Auswertung (automatisch)','','## Bestaetigende Tests','',R.round(5).to_csv(index=False),'','## Sekundaer','',sec.round(5).to_csv(index=False)]))
