import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
ap=argparse.ArgumentParser()
ap.add_argument('--workers-dir',default='workers');ap.add_argument('--out',default='analysis')
ap.add_argument('--success',nargs='+',default=['geo_a025','rot_a025']);ap.add_argument('--failure',nargs='+',default=['rot_a075'])
ap.add_argument('--expected-rows',type=int,default=1680);ap.add_argument('--alpha',type=float,default=.05);ap.add_argument('--margin',type=float,default=-.05)
a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
df=pd.concat([pd.read_csv(p) for p in glob.glob(str(Path(a.workers_dir)/'w*_checkpoints.csv'))],ignore_index=True)
assert len(df)==a.expected_rows,('rows',len(df));assert not df.duplicated(['run_id','step']).any()
num=['train_mse','test_mse','global_grad_norm','hmax_peak','cap_fraction','layer_pre_p99','layer_post_p99','pre_p99_first','pre_p99_mid','pre_p99_last'] if 'pre_p99_first' in df else ['train_mse','test_mse','global_grad_norm','hmax_peak','cap_fraction','layer_pre_p99','layer_post_p99']
assert np.isfinite(df[num].values).all()
f=df[df.step==df.step.max()]
vals=['test_mse']+[c for c in num if c!='test_mse']
pv=f.pivot_table(index=['task','depth','seed'],columns='act',values=vals)
res=pd.DataFrame({'d':np.log10(pv[('test_mse','precap_s225')]/pv[('test_mse','precap_base')])})
for c in num:
 if c!='test_mse': res['d_'+c]=pv[(c,'precap_s225')]-pv[(c,'precap_base')]
res=res.reset_index();res.to_csv(out/'v11_per_seed_differences.csv',index=False)
def holm(ps):
 o=np.argsort(ps);m=len(ps);adj=np.empty(m);run=0.
 for r,i in enumerate(o): run=max(run,(m-r)*ps[i]);adj[i]=min(1.,run)
 return adj
def wp(x,alt):
 x=np.asarray(x);return float(wilcoxon(x,alternative=alt).pvalue)
rows=[]
for t in a.success+a.failure:
 for dp in sorted(res.depth.unique()):
  g=res[(res.task==t)&(res.depth==dp)]
  if len(g)==0: continue
  rows.append(dict(task=t,depth=int(dp),role='success' if t in a.success else 'failure',n=len(g),median_d=g.d.median(),ratio=10**g.d.median(),seeds_better=int((g.d<0).sum()),p_perf_less=wp(g.d,'less'),median_d_peak=g.d_hmax_peak.median(),p_peak_less=wp(g.d_hmax_peak,'less'),p_peak_greater=wp(g.d_hmax_peak,'greater'),median_d_prep99=g.d_layer_pre_p99.median(),p_prep99_less=wp(g.d_layer_pre_p99,'less'),median_d_cap=g.d_cap_fraction.median(),median_d_grad=g.d_global_grad_norm.median()))
T=pd.DataFrame(rows)
S=T.role=='success';Fl=T.role=='failure'
T['holm_perf']=np.nan;T.loc[S,'holm_perf']=holm(T.loc[S,'p_perf_less'].values)
T['holm_peak']=np.nan;T.loc[S,'holm_peak']=holm(T.loc[S,'p_peak_less'].values)
T['holm_prep99']=np.nan;T.loc[S,'holm_prep99']=holm(T.loc[S,'p_prep99_less'].values)
T.to_csv(out/'v11_cell_tests.csv',index=False)
c=res.copy();cen=[]
for col in ['d','d_hmax_peak','d_layer_pre_p99']: c[col+'_c']=c[col]-c.groupby(['task','depth'])[col].transform('median')
sp={}
for col in ['d_hmax_peak','d_layer_pre_p99']:
 r,p=spearmanr(c['d_c'],c[col+'_c']);sp[col]={'rho':float(r),'p':float(p)}
H1=bool((T.loc[S,'holm_perf']<a.alpha).all() and (T.loc[S,'median_d']<0).all())
H2=bool((T.loc[Fl,'median_d']>=a.margin).all())
H3=bool((T.loc[S,'holm_prep99']<a.alpha).sum()>=max(1,int(np.ceil(.75*S.sum()))) and (T.loc[Fl,'median_d_prep99']>0).all())
verdict={'H1_performance_replicated_in_all_success_cells':H1,'H2_no_meaningful_benefit_in_failure_cells':H2,'H3_mechanism_prep99_confirmed_primary':H3,'centered_spearman':sp,'secondary_output_peak_success_cells_significant_holm':int((T.loc[S,'holm_peak']<a.alpha).sum()),'secondary_output_peak_failure_cells_positive':bool((T.loc[Fl,'median_d_peak']>0).all()),'n_success_cells':int(S.sum()),'n_failure_cells':int(Fl.sum())}
json.dump(verdict,open(out/'v11_verdict.json','w'),indent=2)
with open(out/'V11_Auswertung.md','w') as fh:
 fh.write('# V11 Auswertung (preregistered)\n\n');fh.write(T.round(4).to_csv(index=False));fh.write('\n\n'+json.dumps(verdict,indent=2)+'\n')
pd.set_option('display.width',250);print(T.round(4).to_string(index=False));print(json.dumps(verdict,indent=2))
