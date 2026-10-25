import pandas as pd, numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'analysis'; O.mkdir(exist_ok=True)
ck=pd.read_csv(R/'results'/'checkpoint_metrics.csv')
# Per-seed wide trajectories, exact paired values.
idx=['class','task','depth','seed','step']
wide=ck.pivot_table(index=idx,columns='act',values=['test_mse','train_mse','global_grad_norm','hmax_peak','cap_fraction'],aggfunc='first').sort_index()
flat=wide.copy(); flat.columns=[f'{a}_{b}' for a,b in flat.columns]; flat=flat.reset_index()
for ref in ['silu','gelu']:
 flat[f'precap_minus_{ref}_test']=flat['test_mse_precap']-flat[f'test_mse_{ref}']
 flat[f'precap_over_{ref}_test']=flat['test_mse_precap']/flat[f'test_mse_{ref}']
 flat[f'precap_over_{ref}_grad']=flat['global_grad_norm_precap']/flat[f'global_grad_norm_{ref}'].replace(0,np.nan)
 flat[f'precap_over_{ref}_peak']=flat['hmax_peak_precap']/flat[f'hmax_peak_{ref}'].replace(0,np.nan)
flat.to_csv(O/'v5_per_seed_wide.csv',index=False)
# Focus trajectories.
focus=flat[((flat.task=='radial_ripple')&(flat.depth.isin([16,64])))|((flat.task=='ridge_mix')&(flat.depth==64))|((flat.task=='highfreq_grid')&(flat.depth==32))].copy()
focus.to_csv(O/'v5_per_seed_focus.csv',index=False)
# Describe per seed ending results and endpoint rank.
end=focus[focus.step==3000].copy()
rows=[]
for _,r in end.sort_values(['task','depth','seed']).iterrows():
 for ref in ['silu','gelu']:
  rows.append({'class':r['class'],'task':r.task,'depth':r.depth,'seed':r.seed,'ref':ref,
               'precap_test_mse':r['test_mse_precap'],'ref_test_mse':r[f'test_mse_{ref}'],
               'absolute_difference':r[f'precap_minus_{ref}_test'],'ratio':r[f'precap_over_{ref}_test'],
               'precap_better':r[f'precap_minus_{ref}_test']<0,
               'precap_grad_norm':r['global_grad_norm_precap'],'ref_grad_norm':r[f'global_grad_norm_{ref}'],
               'precap_peak':r['hmax_peak_precap'],'ref_peak':r[f'hmax_peak_{ref}'],
               'precap_cap_fraction':r['cap_fraction_precap']})
ends=pd.DataFrame(rows); ends.to_csv(O/'v5_endpoint_per_seed_pairs.csv',index=False)
# Within-seed switch diagnostics for radial ripple d64: compare step 1200 vs 1500 and endpoint.
sw=[]
for ref in ['silu','gelu']:
 for seed in sorted(focus[(focus.task=='radial_ripple')&(focus.depth==64)].seed.unique()):
  d=focus[(focus.task=='radial_ripple')&(focus.depth==64)&(focus.seed==seed)].set_index('step')
  for step in [800,1200,1500,2200,3000]:
   rr=d.loc[step]
   sw.append({'seed':seed,'ref':ref,'step':step,'precap_test_mse':rr['test_mse_precap'],'ref_test_mse':rr[f'test_mse_{ref}'],
              'difference':rr[f'precap_minus_{ref}_test'],'ratio':rr[f'precap_over_{ref}_test'],
              'precap_grad':rr['global_grad_norm_precap'],'ref_grad':rr[f'global_grad_norm_{ref}'],
              'grad_ratio':rr[f'precap_over_{ref}_grad'],'precap_peak':rr['hmax_peak_precap'],'ref_peak':rr[f'hmax_peak_{ref}'],
              'peak_ratio':rr[f'precap_over_{ref}_peak']})
switch=pd.DataFrame(sw); switch.to_csv(O/'v5_radial64_switch_per_seed.csv',index=False)
# Human-readable report.
def fm(x): return f'{x:.6g}'
lines=['# v5 — Per-Seed-Analyse und "Warum"','',
'Interpretation: Ein negativer Unterschied bedeutet kleineren Test-MSE von PreCap. Ein Quotient unter 1 bedeutet ebenfalls besseren Test-MSE von PreCap. Alle Vergleiche sind innerhalb desselben Seeds gepaart.','',
'## Radial ripple, Tiefe 64 — Umschwung je Seed','']
for ref in ['silu','gelu']:
 lines.append(f'### Gegen {ref}')
 for seed in sorted(switch.seed.unique()):
  q=switch[(switch.ref==ref)&(switch.seed==seed)].sort_values('step')
  bits=[]
  for _,r in q.iterrows(): bits.append(f"{int(r.step)}: Δ={r.difference:+.3g}, Grad×={r.grad_ratio:.2f}, Peak×={r.peak_ratio:.2f}")
  lines.append(f'- Seed {seed}: ' + '; '.join(bits)+'.')
 lines.append('')
lines += ['## Endpunkte nach Seed','']
for task,depth in [('radial_ripple',64),('ridge_mix',64),('radial_ripple',16),('highfreq_grid',32)]:
 lines.append(f'### {task}, Tiefe {depth}')
 q=ends[(ends.task==task)&(ends.depth==depth)]
 for ref in ['silu','gelu']:
  z=q[q.ref==ref]
  lines.append(f'- Gegen {ref}: PreCap gewinnt {int(z.precap_better.sum())}/5 Seeds; medianer absoluter Δ-Test-MSE={z.absolute_difference.median():+.6g}; medianer MSE-Quotient={z.ratio.median():.3f}.')
 lines.append('')
lines += ['## Was die Daten bisher tragen','',
'- PreCap ist nicht "schlecht". Es besitzt einen wiederkehrenden deutlichen Früh- bis Mittellernvorteil auf radial_ripple und funktioniert stabil beim Tiefe-16-Anker.', '- Das Problem ist vielmehr die Konsistenz des späten Endpunkts in bestimmten Tiefe-64-/Referenz-Paaren; dieser Umschwung ist seedabhängig und nicht durch einen einzigen universellen Gradientenmarker erklärt.', '- Ein niedrigerer laufender Aktivierungspeak ist häufig vorhanden, aber weder notwendig noch hinreichend für besseren End-Test-MSE.', '', '## Grenzen','', '- Die gespeicherte globale Gradientennorm stammt nur vom letzten Minibatch eines Zeitpunkts. Sie ist eine Diagnose, kein stabiler Schätzer eines ganzen Trainingsfensters.', '- Die Daten zeigen Korrelationen und zeitliche Muster, aber keine kausale Mechanismusbestätigung.']
(O/'v5_per_seed_report.md').write_text('\n'.join(lines)+'\n')
print('WROTE',O/'v5_per_seed_wide.csv')
print('WROTE',O/'v5_endpoint_per_seed_pairs.csv')
print('WROTE',O/'v5_radial64_switch_per_seed.csv')
print('WROTE',O/'v5_per_seed_report.md')
print('\n--- radial64 switch table ---')
print(switch.to_string(index=False))
print('\n--- endpoint summary ---')
print(ends.groupby(['task','depth','ref']).agg(wins=('precap_better','sum'),median_abs_delta=('absolute_difference','median'),median_ratio=('ratio','median'),median_precap_mse=('precap_test_mse','median'),median_ref_mse=('ref_test_mse','median')).to_string())
