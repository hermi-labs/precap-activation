import pandas as pd, numpy as np
d=pd.read_csv('results/checkpoint_metrics.csv')
print('rows',len(d),'runs',d.run_id.nunique(),'nan',int(d[['test_mse','train_mse']].isna().sum().sum()),'steps',sorted(d.step.unique()))
f=d[d.step==d.step.max()]
acts=['precap_c075','precap_base','precap_c125','precap_s075','precap_s225','silu','gelu']
for (t,dep),g in f.groupby(['task','depth']):
    p=g.pivot(index='seed',columns='act',values='test_mse')[acts]
    print(f'\n== {t} d{dep} n={len(p)} ==')
    print('median:',p.median().round(5).to_dict())
    print('CV:',(p.std()/p.mean()).round(2).to_dict())
    for a in acts:
        if a=='precap_base': continue
        dd=np.log10(p[a]/p['precap_base'])
        print(f'  {a:12s} vs base med={dd.median():+.3f} mean={dd.mean():+.3f} wins={int((dd<0).sum())}/{len(p)} seeds={dd.round(2).tolist()}')
    for r in ['silu','gelu']:
        dd=np.log10(p['precap_base']/p[r]);print(f'  base vs {r}: med={dd.median():+.3f} wins={int((dd<0).sum())}/{len(p)}')
print('\n== final medians hmax_peak / cap_fraction / grad ==')
print(f.groupby(['task','depth','act'])[['hmax_peak','cap_fraction','global_grad_norm']].median().round(3).to_string())
print('\n== trajectory: median log10(variant/base) per step ==')
for (t,dep),g in d.groupby(['task','depth']):
    piv=g.pivot_table(index=['seed','step'],columns='act',values='test_mse')
    for a in ['precap_c075','precap_c125','precap_s075','precap_s225']:
        print(t,dep,a,np.log10(piv[a]/piv['precap_base']).groupby('step').median().round(3).to_dict())
