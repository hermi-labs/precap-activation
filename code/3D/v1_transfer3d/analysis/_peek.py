
import pandas as pd,glob,numpy as np
d=pd.concat([pd.read_csv(f) for f in glob.glob('workers/*.csv')])
m=d.step.max(); f=d[d.step==m]
f=f[(f.task=='smooth_bowl3d')&(f.depth==8)]
f['lmse']=np.log10(f.test_mse.clip(lower=1e-12))
print('maxstep',m, len(f))
print(f.groupby(['norm','act','lr']).lmse.median().unstack().round(2))
print(f.groupby(['norm','act','lr']).run_status.apply(lambda s:(s!='ok').sum()).unstack())
