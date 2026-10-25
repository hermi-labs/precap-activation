import sys;sys.path.insert(0,'.')
import pandas as pd,numpy as np,run_v9_geometry as R
from scipy.stats import wilcoxon,spearmanr
D=pd.read_csv('results/checkpoint_metrics.csv');F=D[D.step==1200];A={n:R.anisotropy(n) for _,n in R.TASKS};rows=[]
P=F.pivot_table(index=['task','depth','seed'],columns='act',values='test_mse');P['d']=np.log10(P.precap_s225/P.precap_base);d=P.d.reset_index()
for (t,dep),g in d.groupby(['task','depth']):
 rows.append([t,dep,A[t],g.d.median(),int((g.d<0).sum()),len(g),wilcoxon(g.d,alternative='less').pvalue])
o=pd.DataFrame(rows,columns=['task','depth','anisotropy','median_d','wins','n','p']);o.to_csv('analysis/v9_cells.csv',index=False);print(o.to_string(index=False))
rho,pr=spearmanr(o.anisotropy,o.median_d);print('Spearman anisotropy vs median_d (10 Zellen): rho',round(rho,3),'p',round(pr,4))
m=d.groupby(['task','seed']).d.mean().unstack(0);e0=m['geo_a000'];e1=m['geo_a100'];delta=e0-e1
pe=wilcoxon(delta,alternative='less').pvalue
print('Endpunkte seed-gemittelt: median d radial',round(e0.median(),3),'ridge',round(e1.median(),3),'Kontrast',round(delta.median(),3),'wins',int((delta<0).sum()),'/20 p',round(pe,5))
ok=(rho>=0.6) and (pe<0.05) and (e0.median()<0)
print('URTEIL:','Geometrie-Skalierung UNTERSTUETZT' if ok else 'Geometrie-Skalierung NICHT BESTAETIGT')
