import csv, math, statistics as st, json, glob, os
from collections import defaultdict
TOL=math.log10(1.5)
rows=list(csv.DictReader(open('v11_mini_results.csv')))
N=len(rows)
keyf=lambda r:(r['target'],r['act'],r['knee'],r['mix_beta'],r['depth'],r['arch'],r['init_k'],r['lr'],r['seed'])
uniq=len({keyf(r) for r in rows}); bad=sum(r['bad']=='True' for r in rows)
finite=all(math.isfinite(float(r['test_loss'])) and float(r['test_loss'])>0 for r in rows)
expected=918
def vname(r):
    a=r['act']
    if a=='precap_wg': return f"precap_wg_k{r['knee']}"
    if a=='hybrid_gate': return f"hybrid_k{r['knee']}_b{r['mix_beta']}"
    return a
M=defaultdict(dict)
for r in rows:
    M[(r['target'],int(r['depth']),r['arch'],r['init_k'],r['lr'],int(r['seed']))][vname(r)]=float(r['test_loss'])
variants=['precap_wg_k0.5','precap_wg_k0.75','hybrid_k0.5_b0.2','hybrid_k0.75_b0.3']
comps=['precap','silu','gelu']
cells=sorted({(k[0],k[1],k[2]) for k in M})
def pooled(v,c,cell):
    d=[]
    for k,m in M.items():
        if (k[0],k[1],k[2])==cell and v in m and c in m: d.append(math.log10(m[v]/m[c]))
    return d
out=[]; res={}
for c in comps:
    for v in (['precap']+variants if c!='precap' else variants):
        for cell in cells:
            d=pooled(v,c,cell)
            if d: res[(v,c,cell)]=(st.median(d),sum(x<0 for x in d),len(d))
flat=[x for x in cells if x[1]<=16]; deep=[x for x in cells if x[1]>=32]
summ={}
for v in variants:
    fl=[res[(v,'precap',c)] for c in flat]; dp=[res[(v,'precap',c)] for c in deep]
    fl_all=[x for c in flat for x in pooled(v,'precap',c)]
    dp_all=[x for c in deep for x in pooled(v,'precap',c)]
    flat_better=sum(m<0 for m,_,_ in fl)
    deep_better=sum(m<0 for m,_,_ in dp)
    deep_pres=all(m<=TOL for m,_,_ in dp)
    flat_improve=flat_better>=7 and st.median(fl_all)<0 and sum(x<0 for x in fl_all)/len(fl_all)>0.5
    deep_improve=deep_better>=4 and sum(x<0 for x in dp_all)/len(dp_all)>=2/3
    gate=(flat_improve and deep_pres) or deep_improve
    summ[v]=dict(flat_cells_better=f'{flat_better}/{len(fl)}',flat_median_d=st.median(fl_all),flat_win=sum(x<0 for x in fl_all)/len(fl_all),deep_cells_better=f'{deep_better}/{len(dp)}',deep_median_d=st.median(dp_all),deep_win=sum(x<0 for x in dp_all)/len(dp_all),deep_worst_cell_d=max(m for m,_,_ in dp),deep_preserved=deep_pres,flat_improve=flat_improve,deep_improve=deep_improve,gate_B=gate)
# PreCap vs baselines flat/deep reference
ref={}
for c in ['silu','gelu']:
    fl=[x for cell in flat for x in pooled('precap',c,cell)]
    ref[c]=dict(flat_median_d=st.median(fl),flat_win=sum(x<0 for x in fl)/len(fl))
vs_base={}
for v in variants:
    for c in ['silu','gelu']:
        fl=[x for cell in flat for x in pooled(v,c,cell)]
        vs_base[(v,c)]=(st.median(fl),sum(x<0 for x in fl)/len(fl))
# best-of tuned (secondary, unequal budgets)
def best(cell,act_names,seed):
    vals=[min(m[a] for a in act_names if a in m) for k,m in M.items() if (k[0],k[1],k[2])==cell and k[5]==seed and any(a in m for a in act_names)]
    return min(vals) if vals else None
sec={}
for v in variants:
    d=[]
    for cell in cells:
        for s in range(3):
            a=best(cell,[v],s); b=best(cell,['precap'],s)
            if a and b: d.append(math.log10(a/b))
    sec[v]=(st.median(d),sum(x<0 for x in d)/len(d),len(d))
folder=glob.glob('../Notizen Zwischen*/capexp_roadmap')[0]+'/v11_mini_auswertung'
os.makedirs(folder,exist_ok=True)
with open(folder+'/v11_mini_zellvergleiche.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['variante','vergleich','aufgabe','tiefe','arch','median_d','seed_paare_besser','n'])
    for (v,c,cell),(m,wn,n) in sorted(res.items()): w.writerow([v,c,cell[0],cell[1],cell[2],f'{m:.4f}',wn,n])
L=[]
L.append('# v11 mini — Auswertung\n')
L.append(f'Integrität: {N}/{expected} Zeilen, {uniq} eindeutig, {N-uniq} Duplikate, bad=True: {bad}, alle test_loss endlich und positiv: {finite}.\n')
L.append(f'Metrik: d = log10(MSE_Variante / MSE_Vergleich), gepaart je Aufgabe, Tiefe, Architektur, init_k, lr, Seed. d<0 heißt Variante besser. Toleranz tief: Faktor 1,5, also d <= {TOL:.3f}.\n')
L.append('## Gate-B-Prüfung gegen PreCap\n')
L.append('| Variante | flach: Zellen besser | flach: Median d | flach: Win-Rate | tief: Zellen besser | tief: Median d | tief: schlechtester Zellmedian d | tief erhalten | flach verbessert | tief verbessert | Gate B |')
L.append('|---|---|---:|---:|---|---:|---:|---|---|---|---|')
for v,s in summ.items():
    L.append(f"| {v} | {s['flat_cells_better']} | {s['flat_median_d']:+.3f} | {s['flat_win']:.0%} | {s['deep_cells_better']} | {s['deep_median_d']:+.3f} | {s['deep_worst_cell_d']:+.3f} | {s['deep_preserved']} | {s['flat_improve']} | {s['deep_improve']} | {'PASS' if s['gate_B'] else 'FAIL'} |")
L.append('\n## Flache Netze gegen SiLU und GELU (Median d, Win-Rate)\n')
L.append('| Aktivierung | vs SiLU | vs GELU |'); L.append('|---|---|---|')
L.append(f"| precap | {ref['silu']['flat_median_d']:+.3f} ({ref['silu']['flat_win']:.0%}) | {ref['gelu']['flat_median_d']:+.3f} ({ref['gelu']['flat_win']:.0%}) |")
for v in variants:
    a=vs_base[(v,'silu')]; b=vs_base[(v,'gelu')]
    L.append(f"| {v} | {a[0]:+.3f} ({a[1]:.0%}) | {b[0]:+.3f} ({b[1]:.0%}) |")
L.append('\n## Zellen: Median d gegen PreCap (Seed-Paare besser)\n')
L.append('| Aufgabe | Tiefe | Arch | '+' | '.join(variants)+' |'); L.append('|---|---:|---|'+'---|'*len(variants))
for cell in cells:
    L.append(f'| {cell[0]} | {cell[1]} | {cell[2]} | '+' | '.join(f"{res[(v,'precap',cell)][0]:+.2f} ({res[(v,'precap',cell)][1]}/{res[(v,'precap',cell)][2]})" for v in variants)+' |')
L.append('\n## Sekundär: Best-of je Seed (ungleiche Budgets: Varianten 6 Einstellungen je Zelle, PreCap 3)\n')
L.append('| Variante | Median d vs PreCap | Win-Rate | n |'); L.append('|---|---:|---:|---:|')
for v,(m,w,n) in sec.items(): L.append(f'| {v} | {m:+.3f} | {w:.0%} | {n} |')
open(folder+'/CapExp_v11_mini_Auswertung.md','w').write('\n'.join(L))
json.dump(dict(integrity=dict(rows=N,unique=uniq,bad=bad,finite=finite),summary=summ),open(folder+'/v11_mini_gate_summary.json','w'),indent=1,default=str)
print('\n'.join(L)); print('\nSAVED',folder)
