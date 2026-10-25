import csv, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('/Users/aaronschmitt/PyCharmMiscProject/2D/v4_mechanism_2d')
OUT=Path('/Users/aaronschmitt/PyCharmMiscProject/2D/v5_depth64_diagnosis/analysis_tools/output')
OUT.mkdir(parents=True, exist_ok=True)
fx=list(csv.DictReader((ROOT/'analysis'/'paired_temporal_effects.csv').open()))
lr=list(csv.DictReader((ROOT/'analysis'/'final_layer_ratios.csv').open()))

def wanted_task(task,depth): return task in {'radial_ripple','ridge_mix'} and int(depth)==64
# Temporal slice
rows=[r for r in fx if wanted_task(r['task'],r['depth'])]
with (OUT/'depth64_temporal_focus.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
# Layer summary with coarse early/mid/late bins based on final ratios only (from v4 end state)
lay=[r for r in lr if wanted_task(r['task'],r['depth'])]
with (OUT/'depth64_final_layer_focus.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=lay[0].keys()); w.writeheader(); w.writerows(lay)
# Compact interpretation
lines=['# Depth-64-Diagnose aus v4','', '## Zeitliches Muster','']
for task in ['radial_ripple','ridge_mix']:
    lines.append(f'### {task}')
    for ref in ['silu','gelu']:
        sub=[r for r in rows if r['task']==task and r['ref']==ref]
        sub=sorted(sub,key=lambda r:int(r['step']))
        for r in sub:
            lines.append(f"- Schritt {r['step']} vs {ref}: log10-Testquotient {float(r['median_log10_test_ratio']):+.3f}, Gradienten-Verhältnis {float(r['median_grad_ratio']) if r['median_grad_ratio']!='nan' else float('nan'):.3f}, Peak-Verhältnis {float(r['median_peak_ratio']) if r['median_peak_ratio']!='nan' else float('nan'):.3f}.")
    lines.append('')
lines += ['## Vorläufige Lesart','', '- `radial_ripple` bei 64 zeigt ein auffälliges Muster: Zwischen Schritt 1500 und 3000 kippt der Testvorteil zurück, obwohl die Peaks klein bleiben.', '- `ridge_mix` bei 64 ist gemischt und referenzabhängig: gegen SiLU bleibt PreCap positiv, gegen GELU nicht.', '- Damit ist die nächste Studie am besten als kleine, worker-parallele Tiefe-64-Diagnose mit dichteren Checkpoints angelegt.']
(OUT/'depth64_lesart.md').write_text('\n'.join(lines)+'\n')
print('wrote', OUT/'depth64_temporal_focus.csv')
print('wrote', OUT/'depth64_final_layer_focus.csv')
print('wrote', OUT/'depth64_lesart.md')
