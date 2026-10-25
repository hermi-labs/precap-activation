import pandas as pd, numpy as np
d=pd.read_csv('results/checkpoint_metrics.csv')
for (t,dep),g in d.groupby(['task','depth']):
    piv=g.pivot_table(index=['seed','step'],columns='act',values='test_mse')
    print(f'== {t} d{dep}: median log10(variant/base) by step ==')
    for a in ['precap_c075','precap_c125','precap_s075','precap_s225','silu','gelu']:
        r=np.log10(piv[a]/piv['precap_base']).groupby('step').median().round(2)
        print(f'  {a:12s}',' '.join(f'{s}:{v:+.2f}' for s,v in r.items()))
    b=piv['precap_base'].unstack('step')
    print('  base test_mse per seed @100/500/1200:'); print(b[[100,500,1200]].round(4).to_string())
    print('  base loss spikes (max step-to-step ratio):', (b.iloc[:,1:].values/b.iloc[:,:-1].values).max(axis=1).round(2).tolist())
