import pandas as pd,numpy as np
from scipy.stats import wilcoxon
D=pd.read_csv('results/checkpoint_metrics.csv');F=D[D.step==1200];rows=[]
for (group,task,depth),g in F.groupby(['group','task','depth']):
 p=g.pivot(index='seed',columns='act',values='test_mse');x=np.log10(p['precap_s225']/p['precap_base']);rows.append([group,task,depth,x.median(),int((x<0).sum()),len(x),wilcoxon(x,alternative='less').pvalue])
o=pd.DataFrame(rows,columns=['group','task','depth','median_d','wins','n','p']);o.to_csv('analysis/v8_s225_vs_base.csv',index=False);print(o.to_string(index=False))
for group,g in o.groupby('group'): print(group,'median-of-cells',round(g.median_d.median(),3),'negative cells',int((g.median_d<0).sum()),'/',len(g))
