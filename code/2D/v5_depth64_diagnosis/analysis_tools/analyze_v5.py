import numpy as np, pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parents[1]; OUT=R/'analysis'; OUT.mkdir(exist_ok=True)
ck=pd.read_csv(R/'results'/'checkpoint_metrics.csv'); la=pd.read_csv(R/'results'/'layer_metrics.csv')
la=la[la.tensor=='post'].copy()
la['bin']=pd.cut(la.layer/(la.depth-1),[-.01,1/3,2/3,1.01],labels=['early','mid','late'])
lg=la.groupby(['task','depth','act','seed','step','bin'],observed=True).agg(gn=('weight_grad_norm','median'),rr=('residual_to_post_std','median'),sd=('std','median')).reset_index()
keys=['task','depth','seed','step']
base=ck.pivot_table(index=keys,columns='act',values=['test_mse','global_grad_norm','hmax_peak','train_mse'])
rows=[]
for ref in ['silu','gelu']:
 d=pd.DataFrame({'log10_test':np.log10(base['test_mse']['precap']/base['test_mse'][ref]),
  'grad_ratio':base['global_grad_norm']['precap']/base['global_grad_norm'][ref].replace(0,np.nan),
  'peak_ratio':base['hmax_peak']['precap']/base['hmax_peak'][ref].replace(0,np.nan)}).reset_index();d['ref']=ref;rows.append(d)
P=pd.concat(rows)
S=P.groupby(['task','depth','ref','step']).agg(n=('seed','count'),med_log10_test=('log10_test','median'),n_precap_better=('log10_test',lambda x:int((x<0).sum())),med_grad_ratio=('grad_ratio','median'),med_peak_ratio=('peak_ratio','median')).reset_index()
S.to_csv(OUT/'v5_paired_global.csv',index=False)
lp=lg.pivot_table(index=['task','depth','seed','step','bin'],columns='act',values=['gn','rr'],observed=True)
L=[]
for ref in ['silu','gelu']:
 d=pd.DataFrame({'layer_grad_ratio':lp['gn']['precap']/lp['gn'][ref].replace(0,np.nan),'resid_ratio':lp['rr']['precap']/lp['rr'][ref].replace(0,np.nan)}).reset_index();d['ref']=ref;L.append(d)
L=pd.concat(L);LS=L.groupby(['task','depth','ref','bin','step'],observed=True).agg(med_layer_grad_ratio=('layer_grad_ratio','median'),med_resid_ratio=('resid_ratio','median')).reset_index()
LS.to_csv(OUT/'v5_paired_layerbins.csv',index=False)
pd.set_option('display.width',250);pd.set_option('display.max_columns',30);pd.set_option('display.float_format',lambda v:f'{v:.3f}')
for task,depth in [('radial_ripple',64),('ridge_mix',64),('radial_ripple',16),('highfreq_grid',32)]:
 print(f'\n=== {task} d{depth} ===')
 for ref in ['silu','gelu']:
  s=S[(S.task==task)&(S.depth==depth)&(S.ref==ref)].drop(columns=['task','depth','ref','n'])
  print(f'-- vs {ref}');print(s.to_string(index=False))
print('\n=== layer bins, radial_ripple d64, vs gelu & silu (grad ratio / resid ratio) ===')
x=LS[(LS.task=='radial_ripple')&(LS.depth==64)&(LS.step.isin([300,800,1500,2200,3000]))]
print(x.pivot_table(index=['ref','bin','step'],values=['med_layer_grad_ratio','med_resid_ratio'],observed=True).to_string())
print('\n=== layer bins, ridge_mix d64 ===')
x=LS[(LS.task=='ridge_mix')&(LS.depth==64)&(LS.step.isin([300,800,1500,2200,3000]))]
print(x.pivot_table(index=['ref','bin','step'],values=['med_layer_grad_ratio','med_resid_ratio'],observed=True).to_string())
print('\n=== layer bins controls (step 3000) ===')
x=LS[(LS.task.isin(['radial_ripple','highfreq_grid']))&(LS.depth.isin([16,32]))&(LS.step==3000)]
print(x.to_string(index=False))
