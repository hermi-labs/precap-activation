import argparse,glob,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
ap=argparse.ArgumentParser();ap.add_argument('--dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--steps',type=int,default=1200);a=ap.parse_args()
D=Path(a.dir).resolve()
d=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(D/'workers'/'w*_checkpoints.csv')))],ignore_index=True)
f=d[d.step==a.steps]
T=f.pivot_table(index=['task','depth','seed'],columns='act',values='test_mse')
q=(np.log10(T['precap_s225']/T['precap_base']).rename('d_s225_base')).reset_index()
arr=q['d_s225_base'].to_numpy(float)
res={
 'n':int(np.isfinite(arr).sum()),
 'median_d':float(np.nanmedian(arr)),
 'ratio':float(10**np.nanmedian(arr)),
 'seeds_s225_better':int(np.nansum(arr<0)),
 'seeds_base_better':int(np.nansum(arr>0)),
 'p_two_sided':float(wilcoxon(arr).pvalue) if len(arr) and not np.allclose(arr,0) else float('nan'),
 'p_less_s225_better':float(wilcoxon(arr,alternative='less').pvalue) if len(arr) and not np.allclose(arr,0) else float('nan'),
 'q10':float(np.nanquantile(arr,.10)), 'q25':float(np.nanquantile(arr,.25)), 'q50':float(np.nanquantile(arr,.50)), 'q75':float(np.nanquantile(arr,.75)), 'q90':float(np.nanquantile(arr,.90))
}
out=q.sort_values('d_s225_base').copy(); out['ratio']=10**out['d_s225_base']
(D/'analysis').mkdir(exist_ok=True)
out.to_csv(D/'analysis'/'v15_per_seed_differences.csv',index=False)
Path(D/'analysis'/'v15_verdict.json').write_text(json.dumps(res,indent=2,default=str))
Path(D/'analysis'/'V15_Auswertung.md').write_text('# V15 Auswertung (Ausnahmezelle rot_k15, Tiefe 32)\n\n'+json.dumps(res,indent=2,default=str)+'\n\n## Pro Seed\n\n'+out.to_csv(index=False))
print(json.dumps(res,indent=2,default=str))
