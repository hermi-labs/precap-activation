import csv, math, glob, statistics as st
from collections import defaultdict
R=list(csv.DictReader(open('v12_results.csv')))
M=defaultdict(dict); S=defaultdict(dict)
regname={('0.5','0.001'):'base',('0.5','0.003'):'high_lr',('1.0','0.001'):'large_init'}
for r in R:
    k=(r['target'],int(r['depth']),r['arch'],regname[(r['init_k'],r['lr'])],int(r['seed']))
    M[k][r['act']]=float(r['test_loss'])
    try: S[k][r['act']]=(float(r['hmax_peak']),float(r['grad_final']),float(r['cap_final']))
    except: pass
def sel(**f):
    names=['target','depth','arch','regime','seed']; out=[]
    for k,m in M.items():
        if all(k[names.index(n)]==v for n,v in f.items()): out.append((k,m))
    return out
def d(items,ref): return [math.log10(m['precap']/m[ref]) for k,m in items if ref in m and 'precap' in m and m[ref]>0 and m['precap']>0]
def fmt(v): return f'{st.median(v):+.2f} ({sum(x<0 for x in v)}/{len(v)})'
def sp(w,n): return sum(math.comb(n,i) for i in range(w,n+1))/2**n
T=['sin3_quad','sin_mix','abs_wave']; G=['base','high_lr','large_init']; A=['plain','res_inv']
L=['# v12 — Auswertung Plain-Mechanismus\n']
L.append(f'Integrität: {len(R)}/1620 Runs, bad=True: {sum(r["bad"]=="True" for r in R)}. d = log10(MSE_PreCap / MSE_Referenz); d<0 heißt PreCap besser. Klammer: Seed-Paare mit PreCap besser (Tiefen 4/8/16 × 10 Seeds = 30).\n')
rule={}
for a in A:
    L.append(f'## Architektur {a}: Median d je Aufgabe und Regime (gepoolt über Tiefen und Seeds)\n')
    L.append('| Aufgabe | Regime | vs SiLU | vs GELU |'); L.append('|---|---|---|---|')
    for t in T:
        for g in G:
            it=sel(target=t,arch=a,regime=g); ds,dg=d(it,'silu'),d(it,'gelu')
            rule[(a,t,g)]=(st.median(ds)<0 and st.median(dg)<0)
            L.append(f'| {t} | {g} | {fmt(ds)} | {fmt(dg)} |')
    L.append('')
L.append('## Regel aus dem Manifest (Plain)\n')
L.append('| Regime | Aufgaben mit d<0 gegen SiLU und GELU |'); L.append('|---|---|')
for g in G: L.append(f'| {g} | {sum(rule[("plain",t,g)] for t in T)}/3 |')
rec=any(sum(rule[('plain',t,g)] for t in T)>=2 for g in G)
resrob=sum(rule[('res_inv',t,g)] for t in T for g in G)
L.append(f'\nAnzahl Plain-Regime mit Erholung (>=2/3 Aufgaben): {sum(sum(rule[("plain",t,g)] for t in T)>=2 for g in G)}/3. res_inv: {resrob}/9 Aufgabe-Regime-Zellen mit PreCap besser gegen beide Referenzen.\n')
L.append('## Plain nach Tiefe (Median d, alle Aufgaben und Regime)\n'); L.append('| Tiefe | vs SiLU | vs GELU |'); L.append('|---:|---|---|')
for dep in (4,8,16):
    it=sel(depth=dep,arch='plain'); L.append(f'| {dep} | {fmt(d(it,"silu"))} | {fmt(d(it,"gelu"))} |')
L.append('\n## Stabilität: Median-Verhältnis PreCap / Referenz (Plain und res_inv)\n'); L.append('| Arch | Referenz | hmax_peak | grad_final |'); L.append('|---|---|---:|---:|')
stab={}
for a in A:
    for ref in ('silu','gelu'):
        hr=[];gr=[]
        for k,m in S.items():
            if k[2]==a and 'precap' in m and ref in m and m[ref][0]>0 and m[ref][1]>0: hr.append(m['precap'][0]/m[ref][0]); gr.append(m['precap'][1]/m[ref][1])
        stab[(a,ref)]=(st.median(hr),st.median(gr)); L.append(f'| {a} | {ref} | {st.median(hr):.2f} | {st.median(gr):.2f} |')
cap=[m['precap'][2] for k,m in S.items() if 'precap' in m]
L.append(f'\nMedian PreCap-Cap-Anteil am Ende: plain {st.median([m["precap"][2] for k,m in S.items() if k[2]=="plain" and "precap" in m]):.3f}, res_inv {st.median([m["precap"][2] for k,m in S.items() if k[2]=="res_inv" and "precap" in m]):.3f}.')
L.append('\n## Seed-Ebene Vorzeichentest (Median über Tiefen je Aufgabe, Regime, Seed)\n'); L.append('| Arch | Referenz | Einheiten PreCap besser | p (einseitig PreCap besser) |'); L.append('|---|---|---|---:|')
for a in A:
    for ref in ('silu','gelu'):
        u=[]
        for t in T:
            for g in G:
                for s in range(200,210):
                    v=d(sel(target=t,arch=a,regime=g,seed=s),ref)
                    if v: u.append(st.median(v))
        w=sum(x<0 for x in u); L.append(f'| {a} | {ref} | {w}/{len(u)} | {sp(w,len(u)):.3g} |')
plain_all_worse=all(not rule[('plain',t,g)] for t in T for g in G)
L.append('\n## Befund\n')
L.append(f'- Optimierungsempfindlichkeit gestützt: {rec}.')
L.append(f'- PreCap in plain unter allen 9 Aufgabe-Regime-Zellen nicht besser gegen beide Referenzen: {plain_all_worse}.')
out=glob.glob('../Notizen Zwischen*/capexp_roadmap/v12_plain_mechanism')[0]
open(out+'/CapExp_v12_Auswertung.md','w').write('\n'.join(L)); print('\n'.join(L)); print('SAVED',out)
