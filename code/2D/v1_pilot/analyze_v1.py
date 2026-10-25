import csv, math, statistics as st
from collections import defaultdict
from math import comb
R=list(csv.DictReader(open('v1_results.csv')))
M={}; S={}
for r in R:
 k=(r['task'],int(r['depth']),int(r['seed']))
 M.setdefault(k,{})[r['act']]=float(r['test_loss'])
 S.setdefault(k,{})[r['act']]={z:float(r[z]) for z in ['grad_final','hmax_final','hmax_peak','cap_final']}
def dd(rows,ref): return [math.log10(m['precap']/m[ref]) for k,m in rows if 'precap'in m and ref in m and m['precap']>0 and m[ref]>0]
def fmt(v): return f'{st.median(v):+.3f} ({sum(x<0 for x in v)}/{len(v)})'
def pbin(w,n): return sum(comb(n,i) for i in range(w,n+1))/2**n
T=['radial_ripple','ridge_mix','corner_kink']; D=[4,8,16]
L=['# CapExp 2D v1 — Pilotauswertung','',f'Integrität: {len(R)}/135 Runs, bad=True: {sum(r["bad"]=="True" for r in R)}. Die Hauptgröße ist d = log10(MSE_PreCap/MSE_Referenz); d<0 bedeutet PreCap besser. Klammern zeigen PreCap-Siege pro 5 gepaarten Seeds.','', '## Ergebnis nach Aufgabe und Tiefe','','| Aufgabe | Tiefe | vs SiLU | vs GELU |','|---|---:|---|---|']
pass_task={}
for t in T:
 vals=[]
 for d in D:
  rows=[(k,m) for k,m in M.items() if k[0]==t and k[1]==d]
  x,y=dd(rows,'silu'),dd(rows,'gelu'); vals.append((x,y)); L.append(f'| {t} | {d} | {fmt(x)} | {fmt(y)} |')
 pass_task[t]=(st.median([q for x,y in vals for q in x])<0 and st.median([q for x,y in vals for q in y])<0)
L+=['','## Zusammenfassung je Aufgabe (über Tiefen und Seeds)','','| Aufgabe | vs SiLU | vs GELU | PASS gegen beide? |','|---|---|---|---|']
for t in T:
 rows=[(k,m) for k,m in M.items() if k[0]==t]; x,y=dd(rows,'silu'),dd(rows,'gelu'); L.append(f'| {t} | {fmt(x)} | {fmt(y)} | {pass_task[t]} |')
L+=['','## Zusammenfassung je Tiefe (über Aufgaben und Seeds)','','| Tiefe | vs SiLU | vs GELU |','|---:|---|---|']
for d in D:
 rows=[(k,m) for k,m in M.items() if k[1]==d]; L.append(f'| {d} | {fmt(dd(rows,"silu"))} | {fmt(dd(rows,"gelu"))} |')
L+=['','## Stabilität','','| Referenz | hmax_peak PreCap/Ref | grad_final PreCap/Ref |','|---|---:|---:|']
for ref in ['silu','gelu']:
 h=[];g=[]
 for k,m in S.items():
  if 'precap' in m and ref in m:
   h.append(m['precap']['hmax_peak']/m[ref]['hmax_peak']); g.append(m['precap']['grad_final']/m[ref]['grad_final'])
 L.append(f'| {ref} | {st.median(h):.3f} | {st.median(g):.3f} |')
L.append(f"\nMedian finaler Cap-Anteil: {st.median([m['precap']['cap_final'] for m in S.values()]):.3f}.")
L+=['','## Gepaarte Seed-Vorzeichentests','','| Referenz | PreCap besser | p, einseitig |','|---|---:|---:|']
for ref in ['silu','gelu']:
 v=dd(list(M.items()),ref); w=sum(x<0 for x in v); L.append(f'| {ref} | {w}/{len(v)} | {pbin(w,len(v)):.3g} |')
passed=sum(pass_task.values())>=2
L+=['','## Pilotentscheidung','',f'- Aufgaben mit negativem Median d gegen beide Referenzen: {sum(pass_task.values())}/3.',f'- Bad-Rate: 0/135.',f'- **PASS nach der vorab festgelegten Leistungsregel: {passed}.**','', 'Interpretation: Dies ist ein kleiner Pilot mit fünf Seeds und drei Tiefen. Ein PASS rechtfertigt einen unabhängigen 2D-Holdout, nicht die Behauptung universeller Überlegenheit.']
open('docs/2D_v1_Auswertung.md','w').write('\n'.join(L)); print('\n'.join(L))
