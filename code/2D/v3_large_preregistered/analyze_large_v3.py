import csv,math,statistics as st
from collections import defaultdict
R=list(csv.DictReader(open('v3_large_results.csv')));M=defaultdict(dict)
for r in R:M[(r['task'],int(r['depth']),int(r['seed']))][r['act']]=float(r['test_loss'])
assert len(R)==900 and len(M)==300 and all(set(v)=={'precap','silu','gelu'} for v in M.values()) and not any(r['bad']=='True' for r in R)
def e(rows,ref):return [math.log10(m['precap']/m[ref]) for _,m in rows]
def f(x):return f'{st.median(x):+.3f} ({sum(z<0 for z in x)}/{len(x)})'
T=['radial_ripple','ridge_mix','corner_kink','smooth_bowl','highfreq_grid','angled_kink'];D=[4,8,16,32,64];core=T[:3]
L=['# CapExp v3 — große 2D-Studie: Endauswertung','',f'Integrität: {len(R)}/900 Runs, {len(M)}/300 vollständige Paare, bad={sum(r["bad"]=="True" for r in R)}.','',r'Metrik: \(d=\log_{10}(MSE_{PreCap}/MSE_{Ref})\); negativ begünstigt PreCap.','', '## Nach Aufgabe und Tiefe','','| Aufgabe | Tiefe | vs SiLU | vs GELU |','|---|---:|---|---|']
for t in T:
  for d in D:
    rows=[(k,m) for k,m in M.items() if k[0]==t and k[1]==d];L.append(f'| {t} | {d} | {f(e(rows,"silu"))} | {f(e(rows,"gelu"))} |')
L+=['','## Kernaufgaben: primäres Gate','','| Aufgabe | vs SiLU | vs GELU | PASS |','|---|---|---|---|']
P={}
for t in core:
  rows=[(k,m) for k,m in M.items() if k[0]==t];a=e(rows,'silu');b=e(rows,'gelu');P[t]=st.median(a)<0 and st.median(b)<0;L.append(f'| {t} | {f(a)} | {f(b)} | {P[t]} |')
L+=['','## Nach Tiefe: alle Aufgaben','','| Tiefe | vs SiLU | vs GELU |','|---:|---|---|']
for d in D:
  rows=[(k,m) for k,m in M.items() if k[1]==d];L.append(f'| {d} | {f(e(rows,"silu"))} | {f(e(rows,"gelu"))} |')
L+=['','## Entscheidung','',f"- Core PASS-Aufgaben: {sum(P.values())}/3.",f"- Präregistriertes Gate: {'PASS' if sum(P.values())>=2 else 'FAIL'}.",'- Neue Aufgaben werden separat als Generalisierungsbefund berichtet.']
open('docs/2D_v3_Grosse_Studie_Auswertung.md','w').write('\n'.join(L));print('\n'.join(L))
