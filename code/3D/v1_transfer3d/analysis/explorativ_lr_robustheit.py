
import glob,numpy as np,pandas as pd
from pathlib import Path
pd.set_option('display.width',250);pd.set_option('display.max_rows',300)
D=Path('/Users/aaronschmitt/PyCharmMiscProject/3D/v1_transfer3d')
d=pd.concat([pd.read_csv(p) for p in glob.glob(str(D/'workers'/'w*_checkpoints.csv'))])
f=d[d.step==3000].copy();f['l']=np.log10(f.test_mse.where(f.run_status=='ok',1e6))
K=['task','depth','norm']
m=f.groupby(K+['act','lr']).l.median().unstack('lr')
out=[]
spread=(m.max(1)-m.min(1)).rename('spread')
best=m.min(1).rename('best')
# regret if all use same LR (geteilte Lernrate)
r=m.sub(m.min(1),axis=0)
res=pd.concat([best,spread,r.add_prefix('regret_')],axis=1).reset_index()
res.to_csv(D/'analysis'/'v1_explorativ_lr_robustheit.csv',index=False)
g=res.groupby(['norm','act'])[['best','spread','regret_0.001','regret_0.003','regret_0.01']].median().round(3)
print('MEDIAN ueber 10 Zellen (log10 MSE; regret=Abstand zum eigenen Optimum)');print(g)
# Fenster: LR innerhalb 0.0434 log10 (=10% MSE) vom Optimum
res['win10']=(r<=np.log10(1.1)).sum(1).values
print(res.groupby(['norm','act']).win10.mean().round(2).unstack())
# Rangfolge bestes Ergebnis je Zelle
rk=res.pivot_table(index=K,columns='act',values='best')
print('Bestwert je Zelle');print(rk.round(2))
print('Rangmittel');print(rk.rank(axis=1).groupby('norm').mean().round(2))
# Seed-Streuung (IQR) an eigener bester LR
bl=m.idxmin(1).rename('lrb').reset_index()
ff=f.merge(bl,on=K+['act']);ff=ff[ff.lr==ff.lrb]
iq=ff.groupby(K+['act']).l.apply(lambda x:x.quantile(.75)-x.quantile(.25)).unstack('act')
print('IQR Seeds median');print(iq.groupby('norm').median().round(3))
