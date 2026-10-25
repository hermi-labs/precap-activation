import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
D=Path(a.dir).resolve();ANCHOR='rot_ang53_k10_a025'
d=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(D/'workers'/'w*_checkpoints.csv')))],ignore_index=True)
f=d[d.step==a.steps]
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);adj=np.empty(m);run=0.
    for i,j in enumerate(o):run=max(run,(m-i)*p[j]);adj[j]=min(1.,run)
    return adj
def wx(x,alt):
    x=np.asarray(x,float)
    if len(x)<1 or np.allclose(x,0):return np.nan
    try:return float(wilcoxon(x,alternative=alt).pvalue)
    except Exception:return np.nan
T=f.pivot_table(index=['task','depth','seed'],columns='act',values='test_mse')
P=f.pivot_table(index=['task','depth','seed'],columns='act',values='layer_pre_p99')
dd=pd.DataFrame({'d225':np.log10(T['precap_s225']/T['precap_base']),'d300':np.log10(T['precap_s300']/T['precap_base']),'d300_225':np.log10(T['precap_s300']/T['precap_s225']),'dp225':np.log10(P['precap_s225']/P['precap_base'])}).reset_index()
rows=[]
for (t,k),g in dd.groupby(['task','depth']):
    rows.append(dict(task=t,depth=k,n=len(g),median_d225=g.d225.median(),ratio225=10**g.d225.median(),seeds_better225=int((g.d225<0).sum()),p225_less=wx(g.d225,'less'),median_d300=g.d300.median(),ratio300=10**g.d300.median(),seeds_better300=int((g.d300<0).sum()),p300_less=wx(g.d300,'less'),median_d300_vs_225=g.d300_225.median(),p300_vs_225_two=wx(g.d300_225,'two-sided'),median_dpre_p99_225=g.dp225.median()))
cells=pd.DataFrame(rows)
con=[]
for k in sorted(dd.depth.unique()):
    an=dd[(dd.task==ANCHOR)&(dd.depth==k)].set_index('seed').d225
    for t in sorted(dd.task.unique()):
        if t==ANCHOR:continue
        x=dd[(dd.task==t)&(dd.depth==k)].set_index('seed').d225;j=x.index.intersection(an.index);df=(x[j]-an[j])
        con.append(dict(depth=k,task=t,n=len(j),median_diff=float(df.median()),p_two=wx(df,'two-sided')))
con=pd.DataFrame(con)
if len(con):con['holm']=holm(con.p_two.fillna(1.))
ar=cells[(cells.task==ANCHOR)&(cells.depth==16)]
repl=bool(len(ar) and ar.median_d225.iloc[0]<0 and ar.p225_less.iloc[0]<0.05)
c=dd.copy();c['d225c']=c.d225-c.groupby(['task','depth']).d225.transform('median');c['dpc']=c.dp225-c.groupby(['task','depth']).dp225.transform('median')
rho,p=spearmanr(c.d225c,c.dpc,nan_policy='omit')
sig=con[con.holm<0.05] if len(con) else con
v=dict(R_anchor_depth16_replicated=repl,n_confirmatory_contrasts=int(len(con)),n_significant_holm=int(len(sig)),significant_contrasts=sig[['depth','task','median_diff','holm']].to_dict('records') if len(sig) else [],dose_s300_vs_s225_median_by_cell=cells.set_index(['task','depth']).median_d300_vs_225.round(4).rename(lambda x:str(x)).to_dict() if False else {f'{r.task}_d{r.depth}':round(float(r.median_d300_vs_225),4) for r in cells.itertuples()},exploratory_spearman_d225_vs_dprep99=dict(rho=float(rho),p=float(p)))
(D/'analysis').mkdir(exist_ok=True)
(D/'analysis'/'v12_verdict.json').write_text(json.dumps(v,indent=2,default=str))
(D/'analysis'/'V12_Auswertung.md').write_text('# V12 Auswertung (preregistered)\n\n## Zellen\n\n'+cells.round(4).to_csv(index=False)+'\n\n## Konfirmatorische Kontraste (Holm)\n\n'+con.round(4).to_csv(index=False)+'\n\n## Verdict\n\n'+json.dumps(v,indent=2,default=str)+'\n')
print(json.dumps(v,indent=2,default=str))
