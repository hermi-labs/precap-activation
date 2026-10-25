import pandas as pd, numpy as np
from scipy.stats import wilcoxon
D=pd.read_csv('results/checkpoint_metrics.csv'); F=D[D.step==1200]; rows=[]
for (task,depth),g in F.groupby(['task','depth']):
 p=g.pivot(index='seed',columns='act',values='test_mse'); x=np.log10(p['precap_s225']/p['precap_base']); w=int((x<0).sum()); pv=wilcoxon(x,alternative='less').pvalue
 rows.append([task,depth,x.median(),w,len(x),pv])
 print(task,depth,'s225_vs_base median_d',round(x.median(),3),'wins',f'{w}/{len(x)}','p',round(pv,5))
out=pd.DataFrame(rows,columns=['task','depth','median_d','wins','n','wilcoxon_p']);out.to_csv('analysis/v7b_s225_vs_base.csv',index=False)
print('\nPrimary radial depth profile:')
print(out[out.task.eq('radial_ripple')].sort_values('depth').to_string(index=False))
