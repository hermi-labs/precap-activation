import csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
outdir=ROOT/'analysis'; outdir.mkdir(exist_ok=True)
metrics=list(csv.DictReader((ROOT/'results'/'checkpoint_metrics.csv').open()))
layers=list(csv.DictReader((ROOT/'results'/'layer_metrics.csv').open()))

def med(xs): return statistics.median(xs) if xs else float('nan')
def ratio(a,b): return a/b if b else float('nan')
# Aggregate each activation and checkpoint; medians across five paired seeds.
by=defaultdict(list)
for r in metrics:
    k=(r['class'],r['task'],int(r['depth']),r['act'],int(r['step']))
    by[k].append(r)
summary=[]
for k,rs in sorted(by.items()):
    summary.append({'class':k[0],'task':k[1],'depth':k[2],'act':k[3],'step':k[4],'n':len(rs),
                    'test_mse_median':med([float(r['test_mse']) for r in rs]),
                    'grad_norm_median':med([float(r['grad_norm']) for r in rs]),
                    'hmax_peak_median':med([float(r['hmax_peak']) for r in rs]),
                    'cap_fraction_median':med([float(r['cap_fraction']) for r in rs])})
with (outdir/'checkpoint_medians.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
# Seed-paired PreCap/reference ratios and log effect by cell/step.
index={(r['task'],r['depth'],r['seed'],r['act'],r['step']):r for r in metrics}
effects=[]
for r in metrics:
    if r['act']!='precap': continue
    for ref in ['silu','gelu']:
        q=index[(r['task'],r['depth'],r['seed'],ref,r['step'])]
        effects.append({'class':r['class'],'task':r['task'],'depth':int(r['depth']),'seed':int(r['seed']),'step':int(r['step']),'ref':ref,
                        'log10_test_ratio':math.log10(float(r['test_mse'])/float(q['test_mse'])),
                        'grad_ratio':ratio(float(r['grad_norm']),float(q['grad_norm'])),
                        'peak_ratio':ratio(float(r['hmax_peak']),float(q['hmax_peak'])),
                        'cap_fraction':float(r['cap_fraction'])})
byeff=defaultdict(list)
for r in effects: byeff[(r['class'],r['task'],r['depth'],r['step'],r['ref'])].append(r)
effsum=[]
for k,rs in sorted(byeff.items()):
    effsum.append({'class':k[0],'task':k[1],'depth':k[2],'step':k[3],'ref':k[4],'n':len(rs),
                   'median_log10_test_ratio':med([r['log10_test_ratio'] for r in rs]),
                   'median_grad_ratio':med([r['grad_ratio'] for r in rs]),
                   'median_peak_ratio':med([r['peak_ratio'] for r in rs]),
                   'median_cap_fraction':med([r['cap_fraction'] for r in rs])})
with (outdir/'paired_temporal_effects.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=effsum[0].keys());w.writeheader();w.writerows(effsum)
# Layer dynamics at final checkpoint: PreCap/reference paired ratios by layer / tensor.
li={(r['task'],r['depth'],r['seed'],r['act'],r['step'],r['layer'],r['tensor']):r for r in layers}
layerfx=[]
for r in layers:
    if r['act']!='precap' or r['step']!='3000': continue
    for ref in ['silu','gelu']:
        q=li[(r['task'],r['depth'],r['seed'],ref,r['step'],r['layer'],r['tensor'])]
        layerfx.append({'class':r['class'],'task':r['task'],'depth':int(r['depth']),'seed':int(r['seed']),'ref':ref,'layer':int(r['layer']),'tensor':r['tensor'],
                        'std_ratio':ratio(float(r['std']),float(q['std'])),'maxabs_ratio':ratio(float(r['max_abs']),float(q['max_abs']))})
lfxg=defaultdict(list)
for r in layerfx: lfxg[(r['class'],r['task'],r['depth'],r['ref'],r['layer'],r['tensor'])].append(r)
lfxs=[]
for k,rs in sorted(lfxg.items()):
    lfxs.append({'class':k[0],'task':k[1],'depth':k[2],'ref':k[3],'layer':k[4],'tensor':k[5],'n':len(rs),
                 'median_std_ratio':med([r['std_ratio'] for r in rs]),'median_maxabs_ratio':med([r['maxabs_ratio'] for r in rs])})
with (outdir/'final_layer_ratios.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=lfxs[0].keys());w.writeheader();w.writerows(lfxs)
# Compact interpretation strictly based on medians.
def row(task,depth,step,ref):
    return next(x for x in effsum if x['task']==task and x['depth']==depth and x['step']==step and x['ref']==ref)
lines=['# CapExp v4 — Mechanismusauswertung (deskriptiv)','',
       'Negative log10-Testquotienten begünstigen PreCap. Gradienten- und Peak-Verhältnisse unter 1 bedeuten geringere Werte bei PreCap.','',
       '## Endpunkt bei Schritt 3000','']
for task,depth in [('radial_ripple',16),('angled_kink',32),('radial_ripple',64),('ridge_mix',64),('highfreq_grid',32)]:
    lines.append(f'### {task}, Tiefe {depth}')
    for ref in ['silu','gelu']:
        x=row(task,depth,3000,ref)
        lines.append(f"- Gegen {ref}: log10(Test-MSE-Quotient) {x['median_log10_test_ratio']:+.3f}; Gradienten-Verhältnis {x['median_grad_ratio']:.3f}; Peak-Verhältnis {x['median_peak_ratio']:.3f}; Cap-Anteil {x['median_cap_fraction']:.4f}.")
    lines.append('')
lines += ['## Zeitliche Diagnose','', 'Die Datei `paired_temporal_effects.csv` enthält dieselben gepaarten Kennzahlen bei Schritt 0, 100, 500, 1500 und 3000. Sie erlaubt die Prüfung, ob Unterschiede in Gradient oder Peak vor dem Testfehler-Effekt sichtbar werden.', '', '## Vorsichtige Hypothesenbewertung','', 'Diese Studie ist diagnostisch und deskriptiv. Sie kann zeitliche und layerweise Muster zeigen, aber keine kausale Wirkung von Aktivierungs- oder Gradientenunterschieden beweisen.']
(outdir/'v4_mechanismusbericht.md').write_text('\n'.join(lines)+'\n')
print('wrote',outdir/'checkpoint_medians.csv')
print('wrote',outdir/'paired_temporal_effects.csv')
print('wrote',outdir/'final_layer_ratios.csv')
print('wrote',outdir/'v4_mechanismusbericht.md')
