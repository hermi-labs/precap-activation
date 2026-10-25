import pandas as pd, numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'analysis';O.mkdir(exist_ok=True)
ck=pd.read_csv(R/'results'/'checkpoint_metrics.csv')
# Exact trajectories for each activation and seed.
base=ck[['class','task','depth','seed','act','step','train_mse','test_mse','global_grad_norm','hmax_peak','cap_fraction']].copy()
base['generalization_gap']=base.test_mse-base.train_mse
base.to_csv(O/'v5_seed_trajectories_long.csv',index=False)
# paired differences by ref, full trajectory.
idx=['class','task','depth','seed','step']
p=base.pivot_table(index=idx,columns='act',values=['train_mse','test_mse','generalization_gap','global_grad_norm','hmax_peak'],aggfunc='first')
p.columns=[f'{m}_{a}' for m,a in p.columns];p=p.reset_index()
out=[]
for ref in ['silu','gelu']:
 d=p.copy();d['ref']=ref
 for metric in ['train_mse','test_mse','generalization_gap','global_grad_norm','hmax_peak']:
  d[f'precap_minus_{ref}_{metric}']=d[f'{metric}_precap']-d[f'{metric}_{ref}']
 d['precap_better_test']=d[f'precap_minus_{ref}_test_mse']<0
 out.append(d)
pair=pd.concat(out,ignore_index=True);pair.to_csv(O/'v5_seed_paired_decomposition.csv',index=False)
# interval moves from 1200/1500/2200/3000 for PreCap and ref.
focus_steps=[800,1200,1500,2200,3000]
q=pair[pair.step.isin(focus_steps)].copy(); iv=[]
for (klass,task,depth,seed,ref),g in q.groupby(['class','task','depth','seed','ref']):
 g=g.set_index('step')
 for a,b in zip(focus_steps[:-1],focus_steps[1:]):
  iv.append({'class':klass,'task':task,'depth':depth,'seed':seed,'ref':ref,'from_step':a,'to_step':b,
             'precap_test_change':g.loc[b,'test_mse_precap']-g.loc[a,'test_mse_precap'],
             'ref_test_change':g.loc[b,f'test_mse_{ref}']-g.loc[a,f'test_mse_{ref}'],
             'relative_change_pre_minus_ref':(g.loc[b,'test_mse_precap']-g.loc[a,'test_mse_precap'])-(g.loc[b,f'test_mse_{ref}']-g.loc[a,f'test_mse_{ref}']),
             'precap_train_change':g.loc[b,'train_mse_precap']-g.loc[a,'train_mse_precap'],
             'ref_train_change':g.loc[b,f'train_mse_{ref}']-g.loc[a,f'train_mse_{ref}'],
             'precap_gap_change':g.loc[b,'generalization_gap_precap']-g.loc[a,'generalization_gap_precap'],
             'ref_gap_change':g.loc[b,f'generalization_gap_{ref}']-g.loc[a,f'generalization_gap_{ref}']})
interval=pd.DataFrame(iv);interval.to_csv(O/'v5_seed_interval_decomposition.csv',index=False)
# classify using PreCap-vs-reference status at middle and final points.
classes=[]
for (klass,task,depth,seed,ref),g in q.groupby(['class','task','depth','seed','ref']):
 g=g.set_index('step');status={s:bool(g.loc[s,'precap_better_test']) for s in focus_steps}
 early=any(status[s] for s in [800,1200]);mid=status[1500];late=status[2200];end=status[3000]
 if early and end: label='Vorsprung gehalten/erholt'
 elif early and not end: label='Später Vorsprung verloren'
 elif not early and end: label='Spät aufgeholt'
 else: label='Kein stabiler Vorsprung'
 classes.append({'class':klass,'task':task,'depth':depth,'seed':seed,'ref':ref,'at_800':status[800],'at_1200':status[1200],'at_1500':mid,'at_2200':late,'at_3000':end,'trajectory_class':label,
  'final_test_delta':g.loc[3000,f'precap_minus_{ref}_test_mse'],'final_train_delta':g.loc[3000,f'precap_minus_{ref}_train_mse'],'final_gap_delta':g.loc[3000,f'precap_minus_{ref}_generalization_gap'],
  'final_precap_mse':g.loc[3000,'test_mse_precap'],'final_ref_mse':g.loc[3000,f'test_mse_{ref}']})
cl=pd.DataFrame(classes);cl.to_csv(O/'v5_seed_trajectory_classes.csv',index=False)
# ranked activations at final each seed
rank=[]
for keys,g in base[base.step==3000].groupby(['class','task','depth','seed']):
 z=g.sort_values('test_mse').copy();z['test_rank']=range(1,len(z)+1);rank.append(z)
ranks=pd.concat(rank);ranks.to_csv(O/'v5_endpoint_ranks.csv',index=False)
# German report.
lines=['# v5 — Vollständige Seed-Zerlegung','',
'Alle Aussagen beruhen auf gepaarten Runs mit identischem Seed. Negative Differenzen begünstigen PreCap. Der Generalisierungsabstand ist Test-MSE minus Train-MSE: kleiner ist hier nicht automatisch besser, sondern muss zusammen mit absolutem Testfehler gelesen werden.','',
'## Klassifikation der Trainingspfade','']
for (task,depth,ref),g in cl.groupby(['task','depth','ref']):
 counts=g.trajectory_class.value_counts().to_dict()
 lines.append(f'### {task}, Tiefe {depth}, gegen {ref}')
 lines.append('- ' + '; '.join(f'{k}: {v}/5' for k,v in counts.items()) + '.')
 for _,r in g.sort_values('seed').iterrows():
  lines.append(f"- Seed {int(r.seed)}: {r.trajectory_class}; End-Δ Test={r.final_test_delta:+.6g}, End-Δ Train={r.final_train_delta:+.6g}, End-Δ Gap={r.final_gap_delta:+.6g}.")
 lines.append('')
lines += ['## Richtige Lesart der Formel','',
'- PreCap ist keine starre Regelung, die immer "zu hart" oder immer "zu weich" ist. Ihre CapExp-Komponente wird oberhalb des Caps linear statt exponentiell; dadurch verhindert sie exponentielle Verstärkung, schneidet aber nicht hart ab.', '- Der `tanh`-Teil begrenzt den Eingang vor CapExp weich. Zusammen ergibt das eine glatte, aktivitätsbegrenzende Form mit weiterhin vorhandener Steigung.', '- Ob das passend ist, hängt mit Aufgabe, Tiefe, Initialisierung, Optimierungspfad und Referenzaktivierung zusammen. Die Daten stützen deshalb ein Zusammenkommen mehrerer Faktoren statt eines einzelnen Fehlers.', '', '## Was v5 leisten kann und was nicht','', '- v5 zeigt, ob sich Test- und Trainingsfehler innerhalb eines Seeds unterschiedlich bewegen und ob der späte Wechsel von PreCap selbst oder von der Referenz kommt.', '- v5 beweist keinen einzelnen kausalen Mechanismus, weil die Gradientenspeicherung nur punktuell erfolgt und keine gezielte Ablation der Formelparameter durchgeführt wurde.']
(O/'v5_complete_seed_decomposition.md').write_text('\n'.join(lines)+'\n')
print('WROTE',O/'v5_seed_trajectories_long.csv')
print('WROTE',O/'v5_seed_paired_decomposition.csv')
print('WROTE',O/'v5_seed_interval_decomposition.csv')
print('WROTE',O/'v5_seed_trajectory_classes.csv')
print('WROTE',O/'v5_endpoint_ranks.csv')
print('WROTE',O/'v5_complete_seed_decomposition.md')
print('\n--- radial ripple d64 trajectory classes ---')
print(cl[(cl.task=='radial_ripple')&(cl.depth==64)].sort_values(['ref','seed']).to_string(index=False))
print('\n--- radial ripple d64 intervals vs silu ---')
print(interval[(interval.task=='radial_ripple')&(interval.depth==64)&(interval.ref=='silu')].to_string(index=False))
