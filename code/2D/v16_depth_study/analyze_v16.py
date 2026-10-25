import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--final',type=int,default=3000);ap.add_argument('--early',type=int,default=1200);ap.add_argument('--margin',type=float,default=.10);a=ap.parse_args()
D=Path(a.dir).resolve();rng=np.random.RandomState(12345)
d=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(D/'workers'/'w*_checkpoints.csv')))],ignore_index=True)
def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);m=len(p);r=0.;adj=np.empty(m)
    for i,j in enumerate(o):r=max(r,(m-i)*p[j]);adj[j]=min(1.,r)
    return adj
def wx(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)<1 or np.allclose(x,0):return np.nan
    try:return float(wilcoxon(x).pvalue)
    except Exception:return np.nan
def ci(x,n=2000):
    x=np.asarray(x,float);b=[np.median(x[rng.randint(0,len(x),len(x))]) for _ in range(n)];return tuple(10**np.quantile(b,[.05,.95]))
PAIRS=[('precap_base','silu'),('precap_base','gelu'),('precap_s225','silu'),('precap_s225','gelu'),('precap_s225','precap_base')]
def table(step):
    f=d[d.step==step];T=f.pivot_table(index=['task','depth','seed'],columns='act',values='test_mse');rows=[]
    for (t,k),g in T.groupby(['task','depth']):
        for x,y in PAIRS:
            v=np.log10(g[x]/g[y]).values;lo,hi=ci(v)
            rows.append(dict(step=step,task=t,depth=k,cmp=f'{x}_vs_{y}',n=len(v),ratio=10**np.median(v),ci90_lo=lo,ci90_hi=hi,seeds_first_better=int((v<0).sum()),p=wx(v),equivalent=bool(lo>=1-a.margin and hi<=1+a.margin)))
    return pd.DataFrame(rows)
A=pd.concat([table(a.early),table(a.final)],ignore_index=True)
conf=A[(A.step==a.final)&(A.depth.isin([64,128]))&(A.cmp.isin(['precap_base_vs_silu','precap_base_vs_gelu']))].copy()
conf['holm']=holm(conf.p.fillna(1.))
conf['verdict']=np.where((conf.holm<.05)&(conf.ratio<1),'PreCap besser',np.where((conf.holm<.05)&(conf.ratio>1),'PreCap schlechter',np.where(conf.equivalent,'aequivalent (90%-KI in +-10%)','unklar')))
A.round(4).to_csv(D/'analysis'/'v16_all_comparisons.csv',index=False);conf.round(4).to_csv(D/'analysis'/'v16_confirmatory.csv',index=False)
f=d[d.step.isin([a.early,a.final])]
mech=f.groupby(['act','depth','step'])[['global_grad_norm','cap_fraction','layer_pre_p99','pre_rms','post_rms_scaled','frac_abs_pre_gt_half','train_mse','test_mse']].median().reset_index()
mech.round(5).to_csv(D/'analysis'/'v16_mechanism_medians.csv',index=False)
c=d[d.act.isin(['precap_base','silu'])].pivot_table(index=['task','depth','seed','step'],columns='act',values='test_mse').dropna();c['d']=np.log10(c.precap_base/c.silu)
curve=c.groupby(['depth','step']).d.median().apply(lambda v:10**v).rename('geo_ratio_base_over_silu').reset_index();curve.round(4).to_csv(D/'analysis'/'v16_learning_curve_ratio.csv',index=False)
fn=f[(f.step==a.final)&(f.act=='precap_base')].groupby(['task','depth']).frac_abs_pre_gt_half.median()
dd=conf[conf.cmp=='precap_base_vs_silu'].set_index(['task','depth']).ratio.apply(lambda r:abs(np.log10(r)))
j=pd.concat([fn,dd],axis=1,keys=['frac','absd']).dropna()
rho=spearmanr(j.frac,j.absd) if len(j)>=4 else None
v=dict(final_step=a.final,n_confirmatory=len(conf),counts=conf.verdict.value_counts().to_dict(),by_depth={int(k):g.verdict.value_counts().to_dict() for k,g in conf.groupby('depth')},spearman_fracpre_vs_absd=None if rho is None else dict(rho=float(rho.statistic),p=float(rho.pvalue),n=len(j)),learning_curve=curve.round(4).to_dict('records'))
(D/'analysis'/'v16_verdict.json').write_text(json.dumps(v,indent=2,default=str))
(D/'analysis'/'V16_Auswertung.md').write_text('# V16 Auswertung (preregistered)\n\n## Konfirmatorisch (Schritt %d, Tiefe 64/128, base vs SiLU/GELU, Holm ueber %d)\n\n'%(a.final,len(conf))+conf.round(4).to_csv(index=False)+'\n\n## Verdict\n\n'+json.dumps(v,indent=2,default=str)+'\n\nWeitere Tabellen: v16_all_comparisons.csv, v16_mechanism_medians.csv, v16_learning_curve_ratio.csv\n')
print(json.dumps(v,indent=2,default=str)[:2500])
