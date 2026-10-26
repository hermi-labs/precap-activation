# Phase 3a-3 - praeregistrierte Auswertung (vor dem Start eingefroren, siehe docs/P3A3_Praeregistrierung.md)
import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from itertools import product
from scipy.stats import wilcoxon,rankdata
def wsr_exact(x):
    # exakter Vorzeichen-Permutationstest der Wilcoxon-Rangsumme (Nullen entfernt, Mittelraenge bei Bindungen), zweiseitig.
    # Versionsunabhaengig (3a-2 zeigte: scipy 1.13 und 1.18 liefern bei Bindungen verschiedene p-Werte).
    x=np.asarray(x,float);x=x[x!=0]
    if len(x)==0:return 1.0
    r=rankdata(np.abs(x));t=r[x>0].sum();E=r.sum()/2
    S=np.array(list(product([0,1],repeat=len(x))),dtype=float)@r
    return float(np.mean(np.abs(S-E)>=abs(t-E)-1e-9))
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--final',type=int,default=1720)
ap.add_argument('--tune',default='42000-42003');ap.add_argument('--conf',default='42010-42024');a=ap.parse_args()
rng=lambda s:list(range(int(s.split('-')[0]),int(s.split('-')[1])+1))
D=Path(a.dir);OUT=D/'analysis';OUT.mkdir(exist_ok=True);TUNE=rng(a.tune);CONF=rng(a.conf);K=['dataset','depth']
PRIMARY=[('spreccap_cs','selu')]
PAIRS=PRIMARY+[('spreccap_cs','spreccap'),('spreccap_cs_ls','spreccap_cs'),('spreccap_cs','elu'),('spreccap_e03','selu'),('spreccap_e03','spreccap_cs'),('spreccap','selu')]
CHI1={'spreccap':1.1305,'spreccap_cs':1.0716,'spreccap_e03':1.0300}   # exakt, math/p3a3_chi1_family.py
LRS=[1e-5,2e-5,5e-5,1e-4,2e-4,5e-4]
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
best.to_csv(OUT/'p3a3_lr_selection.csv',index=False)
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
    for c,r in PAIRS:
            x=(g[c]-g[r]).values
            p=wsr_exact(x)
            try:ps=float(wilcoxon(x,zero_method='zsplit').pvalue) if np.any(x!=0) else 1.0
            except Exception:ps=float('nan')
            rows.append(dict(dataset=key[0],depth=key[1],cand=c,ref=r,primary=(c,r) in PRIMARY,n=len(x),med_err_cand=float(np.median(g[c])),med_err_ref=float(np.median(g[r])),med_diff_pp=float(np.median(x))*100,wins=int((x<0).sum()),losses=int((x>0).sum()),p=p,p_scipy_info=ps))
R=pd.DataFrame(rows);R['holm_all']=np.nan;R.loc[~R.primary,'holm_all']=holm(R.loc[~R.primary,'p']);R['holm_primary']=np.nan   # getrennte Holm-Familien
R.loc[R.primary,'holm_primary']=holm(R.loc[R.primary,'p'])
def verd(h,dif):return np.where(h<0.05,np.where(dif<0,'Kandidat besser',np.where(dif>0,'Kandidat schlechter','unklar')),'unklar')
R['verdict_primary']=np.where(R.primary,verd(R.holm_primary.fillna(1),R.med_diff_pp),'')
R['verdict_all']=np.where(R.primary,R.verdict_primary,verd(R.holm_all.fillna(1),R.med_diff_pp))
R.to_csv(OUT/'p3a3_confirmatory.csv',index=False)
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
sec=pd.concat([macc,win,div_t,div_c],axis=1).reset_index().rename(columns={'level_2':'act'});sec.to_csv(OUT/'p3a3_secondary.csv',index=False)
med.pivot_table(index=K+['act'],columns='lr',values='val_acc').reset_index().to_csv(OUT/'p3a3_tune_median_val_acc_by_lr.csv',index=False)
G0=d[(d.step==0)&d.seed.isin(CONF)].copy();G0=G0[(G0.gn_first>0)&(G0.gn_last>0)];G0['r']=np.log10(G0.gn_first/G0.gn_last)
mech={f'{k[0]}|{k[1]}':round(float(v),3) for k,v in G0.groupby(['depth','act']).r.median().items()}   # log10(|grad erste Schicht| / |grad letzte Schicht|) bei Init
G0.groupby(['dataset','depth','act'])[['gn_first','gn_mid','gn_last','r']].median().reset_index().to_csv(OUT/'p3a3_init_grad_profile.csv',index=False)
# 3) Entscheidungsregel fuer das Zwischenziel v0.1 (praeregistriert): spreccap_cs 'bewaehrt', wenn in KEINER Zelle signifikant
#    schlechter als SELU (holm_primary) UND in mindestens 3 von 4 Zellen median_diff <= +0.5 pp
Pp=R[R.primary];bew=dict(zellen_signifikant_schlechter=int((Pp.verdict_primary=='Kandidat schlechter').sum()),zellen_diff_le_0_5pp=int((Pp.med_diff_pp<=0.5).sum()))
bew['bewaehrt']=bool(bew['zellen_signifikant_schlechter']==0 and bew['zellen_diff_le_0_5pp']>=3)
# 4) chi1-Dosis-Wirkung (deskriptiv) und Vorhersage des Gradientenprofils log10(g1/gL) ~ (L-1)/2*log10(chi1)
dose={f'{k[0]}|{k[1]}':{a:round(float(P.loc[k][a].median()),4) for a in CHI1} for k in P.groupby(K).groups}
pred={f'{L}|{a}':round((L-1)/2*np.log10(c),3) for L in sorted(d.depth.unique()) for a,c in {**CHI1,'selu':1.0716}.items()}
# 5) gelerntes s (nur spreccap_cs_ls, Bestaetigungs-Seeds, Endschritt)
Ls=d[(d.act=='spreccap_cs_ls')&(d.step==a.final)&d.seed.isin(CONF)&(d.run_status=='ok')]
s_learned={f'{k[0]}|{k[1]}':[round(float(x),3) for x in g[['s_min','s_med','s_max']].median()] for k,g in Ls.groupby(K)} if 's_min' in d else {}
cnt=lambda m:{k:int(v) for k,v in best.assign(e=m).groupby('act').e.sum().items()}
v=dict(n_rows=len(d),runs=len(runs),tune_runs=len(Tu),conf_runs=len(Co),diverged_runs=int(T.diverged.sum()),cells=ncell,need_cells=need,lr_selection_matches_file=sel_match,
       primary=prim,all_pairs_counts={k:int(x) for k,x in R.verdict_all.value_counts().items()},all_pairs=allc,
       lr_at_upper_edge=cnt(best.lr_sel==max(LRS)),lr_at_lower_edge=cnt(best.lr_sel==min(LRS)),
       trainable_cells_best_tune_val_acc_gt_0_5={k:int(v) for k,v in (med.groupby(K+['act']).val_acc.max()>0.5).groupby('act').sum().items()},
       init_grad_ratio_first_over_last_median=mech,init_grad_ratio_predicted_chi1=pred,entscheidung_v01=bew,chi1_dose_median_err=dose,s_learned_min_med_max=s_learned,
       mean_lr_window=sec.groupby('act')['lr_window_0.5pp'].mean().round(2).to_dict(),
       mean_divergence_tune=sec.groupby('act').divergence_rate_tune.mean().round(3).to_dict(),
       mean_rank_conf=sec.pivot_table(index=K,columns='act',values='median_test_err_conf').rank(axis=1).mean().round(2).to_dict())
json.dump(v,open(OUT/'p3a3_verdict.json','w'),indent=2,ensure_ascii=False);print(json.dumps(v,indent=2,ensure_ascii=False))
open(OUT/'P3a3_Auswertung.md','w').write('\n'.join(['# Phase 3a-3 - Auswertung (automatisch)','','## Bestaetigende Tests','',R.round(5).to_csv(index=False),'','## Sekundaer','',sec.round(5).to_csv(index=False)]))
