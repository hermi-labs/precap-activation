import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
D=Path(a.dir).resolve()
d=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(D/'workers'/'w*_checkpoints.csv')))],ignore_index=True)
f=d[d.step==a.steps]
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);adj=np.empty(m);run=0.
    for i,j in enumerate(o):run=max(run,(m-i)*p[j]);adj[j]=min(1.,run)
    return adj
def wx(x,alt='two-sided'):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)<1 or np.allclose(x,0):return np.nan
    try:return float(wilcoxon(x,alternative=alt).pvalue)
    except Exception:return np.nan
T=f.pivot_table(index=['task','depth','seed'],columns='act',values='test_mse')
cols={
 'd_s225_base':np.log10(T['precap_s225']/T['precap_base']),
 'd_s225_silu':np.log10(T['precap_s225']/T['silu']),
 'd_s225_gelu':np.log10(T['precap_s225']/T['gelu']),
 'd_base_silu':np.log10(T['precap_base']/T['silu']),
 'd_base_gelu':np.log10(T['precap_base']/T['gelu'])}
dd=pd.DataFrame(cols).reset_index()
rows=[]
for (t,k),g in dd.groupby(['task','depth']):
    r=dict(task=t,depth=k,n=len(g))
    for c in cols:
        r[c+'_median']=g[c].median();r[c+'_ratio']=10**g[c].median();r[c+'_better']=int((g[c]<0).sum());r[c+'_p']=wx(g[c])
    rows.append(r)
cells=pd.DataFrame(rows)
con=[]
for t,g in dd.groupby('task'):
    m=g.groupby('seed').d_s225_base.mean();con.append(dict(task=t,n=len(m),median_d=float(m.median()),ratio=float(10**m.median()),seeds_better=int((m<0).sum()),p_two=wx(m)))
con=pd.DataFrame(con);con['holm']=holm(con.p_two.fillna(1.))
con['verdict']=np.where((con.holm<.05)&(con.median_d<0),'s225 besser',np.where((con.holm<.05)&(con.median_d>0),'s225 schlechter','kein gesicherter Unterschied'))
v=dict(n_tasks=int(len(con)),n_s225_better_holm=int((con.verdict=='s225 besser').sum()),n_s225_worse_holm=int((con.verdict=='s225 schlechter').sum()),n_no_difference=int((con.verdict=='kein gesicherter Unterschied').sum()),per_task=con.round(4).to_dict('records'))
(D/'analysis').mkdir(exist_ok=True)
(D/'analysis'/'v14_verdict.json').write_text(json.dumps(v,indent=2,default=str))
(D/'analysis'/'V14_Auswertung.md').write_text('# V14 Auswertung (preregistered)\n\n## Konfirmatorisch: s225 gegen base je Aufgabe (Mittel ueber Tiefen je Seed, Holm)\n\n'+con.round(4).to_csv(index=False)+'\n\n## Explorativ: Zellen\n\n'+cells.round(4).to_csv(index=False)+'\n\n## Verdict\n\n'+json.dumps(v,indent=2,default=str)+'\n')
print(json.dumps(v,indent=2,default=str))
