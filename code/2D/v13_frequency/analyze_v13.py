import argparse,glob,json,re
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
D=Path(a.dir).resolve();REF='rot_ang53_f100_a025'
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
dd=pd.DataFrame({'d':np.log10(T['precap_s225']/T['precap_base'])}).reset_index()
dd['k']=dd.task.str.extract(r'_f(\d+)_')[0].astype(int)/100.
rows=[]
for (k,dep),g in dd.groupby(['k','depth']):
    rows.append(dict(k=k,depth=dep,n=len(g),median_d=g.d.median(),ratio=10**g.d.median(),seeds_better=int((g.d<0).sum()),p_less=wx(g.d,'less'),p_greater=wx(g.d,'greater')))
cells=pd.DataFrame(rows)
sl=[];con=[]
for dep,g in dd.groupby('depth'):
    W=g.pivot(index='seed',columns='k',values='d').dropna();x=np.log10(W.columns.values.astype(float))
    s=np.array([np.polyfit(x,r.values,1)[0] for _,r in W.iterrows()])
    con.append(dict(depth=dep,n=len(s),median_slope=float(np.median(s)),p_one=wx(s,'greater')))
    for k in (1.5,2.0):
        dif=(W[k]-W[1.0]);sl.append(dict(depth=dep,k=k,n=len(dif),median_diff_vs_k100=float(dif.median()),p_two=wx(dif,'two-sided')))
con=pd.DataFrame(con);con['holm']=holm(con.p_one.fillna(1.));sec=pd.DataFrame(sl)
cross={}
for dep,g in cells.groupby('depth'):
    g=g.sort_values('k');m=g.median_d.values;kk=g.k.values;c=None
    for i in range(len(m)-1):
        if m[i]<0<=m[i+1]:c=float(kk[i]+(0-m[i])*(kk[i+1]-kk[i])/(m[i+1]-m[i]));break
    cross[int(dep)]=c
rho={}
for dep,g in cells.groupby('depth'):
    r,p=spearmanr(g.k,g.median_d);rho[int(dep)]=dict(rho=float(r),p=float(p))
sig=con[con.holm<0.05]
v=dict(n_confirmatory=int(len(con)),n_significant_holm=int(len(sig)),slope_tests=con.round(4).to_dict('records'),exploratory_crossover_k=cross,exploratory_cell_median_spearman=rho,exploratory_k150_k200_vs_k100=sec.round(4).to_dict('records'))
(D/'analysis').mkdir(exist_ok=True)
(D/'analysis'/'v13_verdict.json').write_text(json.dumps(v,indent=2,default=str))
(D/'analysis'/'V13_Auswertung.md').write_text('# V13 Auswertung (preregistered)\n\n## Zellen\n\n'+cells.round(4).to_csv(index=False)+'\n\n## Konfirmatorische Steigungstests (Holm)\n\n'+con.round(4).to_csv(index=False)+'\n\n## Explorativ: k=1.5/2.0 gegen k=1.0\n\n'+sec.round(4).to_csv(index=False)+'\n\n## Verdict\n\n'+json.dumps(v,indent=2,default=str)+'\n')
print(json.dumps(v,indent=2,default=str))
