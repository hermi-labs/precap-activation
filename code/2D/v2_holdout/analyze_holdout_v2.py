import csv,math,statistics as st,json,hashlib
from collections import defaultdict
from math import comb
H='.'; R=list(csv.DictReader(open('v2_holdout_results.csv')))
M=defaultdict(dict);S=defaultdict(dict)
for r in R:
 k=(r['task'],int(r['depth']),int(r['seed'])); M[k][r['act']]=float(r['test_loss']); S[k][r['act']]={z:float(r[z]) for z in ['hmax_peak','grad_final','cap_final']}
expected=3*3*10*3
assert len(R)==expected and len(M)==90 and all(set(m)=={'precap','silu','gelu'} for m in M.values()) and not any(r['bad']=='True' for r in R)
def effects(rows, ref): return [math.log10(m['precap']/m[ref]) for k,m in rows if 'precap' in m and ref in m]
def f(v): return f'{st.median(v):+.3f} ({sum(x<0 for x in v)}/{len(v)})'
def pval(w,n): return sum(comb(n,i) for i in range(w,n+1))/2**n
T=['radial_ripple','ridge_mix','corner_kink'];D=[4,8,16]
L=['# CapExp 2D v2 — unabhängiger Holdout: Endauswertung','', '**Präregistrierter Status:** Die Formel, Architektur, Aufgaben, Seeds, Trainingsprotokoll, Metriken und PASS-Regel waren vor Abschluss eingefroren (`meta/manifest.json`).','', '## Integrität','',f'- Ergebniszeilen: **{len(R)}/{expected}**.',f'- Vollständige gepaarte Zellen (Aufgabe × Tiefe × Seed): **{len(M)}/90**.',f'- `bad=True`: **{sum(r["bad"]=="True" for r in R)}**.', '- Jede Zelle enthält genau PreCap, SiLU und GELU: **ja**.','',r'Primärmetrik: \\(d=\log_{10}(\mathrm{MSE}_{\mathrm{PreCap}}/\mathrm{MSE}_{\mathrm{Ref}})\\). Negative Werte begünstigen PreCap. Klammern zeigen PreCap-Siege / gepaarte Seeds.','', '## Primäre Resultate nach Aufgabe','','| Aufgabe | gegen SiLU | gegen GELU | Gate gegen beide |','|---|---|---|---|']
passes={}
for t in T:
 rows=[(k,m) for k,m in M.items() if k[0]==t]; a=effects(rows,'silu');b=effects(rows,'gelu');passes[t]=st.median(a)<0 and st.median(b)<0
 L.append(f'| {t} | {f(a)} | {f(b)} | {"PASS" if passes[t] else "FAIL"} |')
L+=['','## Resultate nach Tiefe','','| Tiefe | gegen SiLU | gegen GELU |','|---:|---|---|']
for d in D:
 rows=[(k,m) for k,m in M.items() if k[1]==d];L.append(f'| {d} | {f(effects(rows,"silu"))} | {f(effects(rows,"gelu"))} |')
L+=['','## Aufgabe × Tiefe','','| Aufgabe | Tiefe | gegen SiLU | gegen GELU |','|---|---:|---|---|']
for t in T:
 for d in D:
  rows=[(k,m) for k,m in M.items() if k[0]==t and k[1]==d];L.append(f'| {t} | {d} | {f(effects(rows,"silu"))} | {f(effects(rows,"gelu"))} |')
L+=['','## Replikationsstärke','','| Referenz | PreCap besser | einseitiger gepaarter Sign-Test |','|---|---:|---:|']
for ref in ['silu','gelu']:
 x=effects(list(M.items()),ref); w=sum(z<0 for z in x);L.append(f'| {ref} | {w}/{len(x)} | {pval(w,len(x)):.3g} |')
L+=['','## Stabilität und Intervention','','| Referenz | Median hmax_peak PreCap/Ref | Median grad_final PreCap/Ref |','|---|---:|---:|']
for ref in ['silu','gelu']:
 h=[m['precap']['hmax_peak']/m[ref]['hmax_peak'] for m in S.values()];g=[m['precap']['grad_final']/m[ref]['grad_final'] for m in S.values()];L.append(f'| {ref} | {st.median(h):.3f} | {st.median(g):.3f} |')
cap=[m['precap']['cap_final'] for m in S.values()];L.append(f'\nMedian finaler Cap-Anteil: **{st.median(cap):.3f}**.')
verdict=sum(passes.values())>=2
L+=['','## Präregistrierte Entscheidung','',f'- Aufgaben mit PASS gegen beide Referenzen: **{sum(passes.values())}/3**.', '- Höhere PreCap-bad-rate: **nein**.',f'- **Entscheidung: {"PASS" if verdict else "FAIL"}.**','', 'Interpretation: Der Holdout bestätigt die Pilotentscheidung nur im Rahmen dieser festen 2D-Syntheseaufgaben und dieser eingefrorenen Trainingskonfiguration. Er ist kein allgemeiner Beweis für jede Architektur, Datenverteilung oder Tiefe.']
open('docs/2D_v2_Holdout_Auswertung.md','w').write('\n'.join(L))
print('\n'.join(L))
