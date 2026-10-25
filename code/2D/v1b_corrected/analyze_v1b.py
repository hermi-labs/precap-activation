import csv,math,statistics as st
from collections import defaultdict
from math import comb
R=list(csv.DictReader(open('v1b_results.csv'))); M=defaultdict(dict);S=defaultdict(dict)
for r in R:
 k=(r['task'],int(r['depth']),int(r['seed']));M[k][r['act']]=float(r['test_loss']);S[k][r['act']]={z:float(r[z]) for z in ['hmax_peak','grad_final','cap_final']}
def ds(rows,ref):return [math.log10(m['precap']/m[ref]) for k,m in rows if 'precap'in m and ref in m]
def fmt(x):return f'{st.median(x):+.3f} ({sum(v<0 for v in x)}/{len(x)})'
def binom(w,n):return sum(comb(n,i) for i in range(w,n+1))/2**n
T=['radial_ripple','ridge_mix','corner_kink'];D=[4,8,16]
L=['# CapExp 2D v1b — Auswertung der korrigierten PreCap-Formel','',f'Integrität: {len(R)}/135 Runs, `bad=True`: {sum(r["bad"]=="True" for r in R)}. Diese Auswertung ist gültig, weil v1b die 1D-v9-PreCap-Formel auf 10.001 Werten exakt reproduziert hat.','',r'Primärgröße: \(d=\log_{10}(\mathrm{MSE}_{PreCap}/\mathrm{MSE}_{Ref})\). \(d<0\) heißt: PreCap besser. Klammern: PreCap-Siege / gepaarte Seeds.','', '## Nach Aufgabe und Tiefe','','| Aufgabe | Tiefe | gegen SiLU | gegen GELU |','|---|---:|---|---|']
pt={}
for t in T:
 allx=[];ally=[]
 for d in D:
  rows=[(k,m) for k,m in M.items() if k[0]==t and k[1]==d];x=ds(rows,'silu');y=ds(rows,'gelu');allx+=x;ally+=y;L.append(f'| {t} | {d} | {fmt(x)} | {fmt(y)} |')
 pt[t]=(st.median(allx)<0 and st.median(ally)<0)
L+=['','## Pilotentscheidung pro Aufgabe','','| Aufgabe | gegen SiLU | gegen GELU | PASS gegen beide |','|---|---|---|---|']
for t in T:
 rows=[(k,m) for k,m in M.items() if k[0]==t];L.append(f'| {t} | {fmt(ds(rows,"silu"))} | {fmt(ds(rows,"gelu"))} | {pt[t]} |')
L+=['','## Nach Tiefe (alle Aufgaben und Seeds)','','| Tiefe | gegen SiLU | gegen GELU |','|---:|---|---|']
for d in D:
 rows=[(k,m) for k,m in M.items() if k[1]==d];L.append(f'| {d} | {fmt(ds(rows,"silu"))} | {fmt(ds(rows,"gelu"))} |')
L+=['','## Stabilität','','| Referenz | Median hmax_peak PreCap/Ref | Median grad_final PreCap/Ref |','|---|---:|---:|']
for ref in ['silu','gelu']:
 h=[];g=[]
 for k,m in S.items():
  h.append(m['precap']['hmax_peak']/m[ref]['hmax_peak']);g.append(m['precap']['grad_final']/m[ref]['grad_final'])
 L.append(f'| {ref} | {st.median(h):.3f} | {st.median(g):.3f} |')
L.append(f"\nMedian finaler Cap-Anteil (PreCap): {st.median([m['precap']['cap_final'] for m in S.values()]):.3f}.")
L+=['','## Gepaarte Seed-Vorzeichentests','','| Referenz | PreCap besser | p, einseitig |','|---|---:|---:|']
for ref in ['silu','gelu']:
 x=ds(list(M.items()),ref);w=sum(v<0 for v in x);L.append(f'| {ref} | {w}/{len(x)} | {binom(w,len(x)):.3g} |')
ok=sum(pt.values())>=2
L+=['','## Fazit','',f'- Aufgaben mit PreCap-Median besser gegen beide Referenzen: {sum(pt.values())}/3.',f'- Keine bad-Runs: {sum(r["bad"]=="True" for r in R)==0}.',f'- **PASS nach vorab festgelegter Regel: {ok}.**','', 'Die Entscheidung wird nur aus diesem korrigierten v1b-Pilot abgeleitet. Der frühere v1-Lauf bleibt als Implementierungsfehler markiert und fließt nicht ein.']
open('docs/2D_v1b_Auswertung.md','w').write('\n'.join(L));print('\n'.join(L))
