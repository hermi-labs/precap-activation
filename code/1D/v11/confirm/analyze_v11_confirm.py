import csv, math, glob, os, statistics as st
from collections import defaultdict
TOL=math.log10(1.5)
R=list(csv.DictReader(open('v11_confirm_results.csv')))
M=defaultdict(dict); H=defaultdict(dict)
badrows=[r for r in R if r['bad']=='True']
for r in R:
    k=(r['target'],int(r['depth']),r['arch'],r['init_k'],r['lr'],int(r['seed']))
    M[k][r['act']]=float(r['test_loss'])
    try: H[k][r['act']]=float(r['hmax_peak'])
    except: pass
cells=sorted({(k[0],k[1],k[2]) for k in M}); flat=[c for c in cells if c[1]<=16]; deep=[c for c in cells if c[1]>=32]
def pairs(c,ref,cell=None,cs=None):
    out=[]
    for k,m in M.items():
        if (cell is None or (k[0],k[1],k[2])==cell) and (cs is None or (k[0],k[1],k[2]) in cs) and 'hybrid_gate' in m and ref in m:
            out.append((k,math.log10(m['hybrid_gate']/m[ref])))
    return out
def summ(ds): v=[d for _,d in ds]; return st.median(v), sum(x<0 for x in v)/len(v), len(v)
def sign_p(wins,n): return sum(math.comb(n,i) for i in range(wins,n+1))/2**n
L=['# v11 confirm — Auswertung nach Manifest\n']
L.append(f'Integrität: {len(R)}/2400 Runs, bad=True: {len(badrows)}.')
for r in badrows: L.append(f"- bad: {r['target']} {r['act']} Tiefe {r['depth']} {r['arch']} init_k={r['init_k']} lr={r['lr']} seed={r['seed']} test_loss={r['test_loss']}")
L.append('')
res={}
for ref in ['precap','silu','gelu']:
    for cell in cells: res[(ref,cell)]=summ(pairs(None,ref,cell)) if pairs(None,ref,cell) else None
L.append('## Regeln gegen PreCap\n')
fl=pairs(None,'precap',cs=set(flat)); dp=pairs(None,'precap',cs=set(deep))
fm,fw,fn=summ(fl); dm,dw,dn=summ(dp)
flat_better=sum(res[('precap',c)][0]<0 for c in flat); deep_better=sum(res[('precap',c)][0]<0 for c in deep)
deep_worst=max(res[('precap',c)][0] for c in deep)
r1=all(res[('precap',c)][0]<=TOL for c in deep)
r2=flat_better>=11 and fm<0 and fw>0.5
ratios=[]
for k,m in H.items():
    if 'hybrid_gate' in m and 'precap' in m and m['precap']>0: ratios.append(m['hybrid_gate']/m['precap'])
hr=st.median(ratios)
bad_cand=sum(r['act']=='hybrid_gate' for r in badrows)
r3=bad_cand==0 and hr<=2
alt=deep_better>=4 and dw>=2/3
L.append('| Kriterium | Wert | Schwelle | erfüllt |'); L.append('|---|---|---|---|')
L.append(f'| 1 tief: schlechtester Zellmedian d | {deep_worst:+.3f} (Faktor {10**deep_worst:.2f}) | <= {TOL:.3f} | {r1} |')
L.append(f'| 2 flach: Zellen mit Median d<0 | {flat_better}/18 | >= 11 | {flat_better>=11} |')
L.append(f'| 2 flach: gepoolter Median d / Win-Rate | {fm:+.4f} / {fw:.1%} (n={fn}) | <0 / >50% | {fm<0 and fw>0.5} |')
L.append(f'| 3 Stabilität: bad-Runs des Kandidaten | {bad_cand} | 0 | {bad_cand==0} |')
L.append(f'| 3 Stabilität: Median hmax_peak-Verhältnis | {hr:.2f} | <= 2 | {hr<=2} |')
L.append(f'| Alternative: tief Zellen d<0 / Win-Rate | {deep_better}/6 / {dw:.1%} (n={dn}) | >=4 / >=66,7% | {alt} |')
passed=(r1 and r2 and r3) or alt
L.append(f'\n**Gesamt: {"BESTANDEN" if passed else "NICHT BESTANDEN"}**\n')
L.append('## Median d pro Tiefe (alle Zellen gepoolt)\n'); L.append('| Tiefe | vs PreCap | vs SiLU | vs GELU |'); L.append('|---:|---|---|---|')
for d in (4,8,16,32,64):
    cs={c for c in cells if c[1]==d}; row=[]
    for ref in ['precap','silu','gelu']:
        m,w,n=summ(pairs(None,ref,cs=cs)); row.append(f'{m:+.3f} ({w:.0%}, n={n})')
    L.append(f'| {d} | '+' | '.join(row)+' |')
L.append('\n## Zellen: Median d (Seed-Paare besser)\n'); L.append('| Aufgabe | Tiefe | Arch | vs PreCap | vs SiLU | vs GELU |'); L.append('|---|---:|---|---|---|---|')
for c in cells:
    row=[]
    for ref in ['precap','silu','gelu']:
        ds=[d for _,d in pairs(None,ref,c)]; row.append(f'{st.median(ds):+.2f} ({sum(x<0 for x in ds)}/{len(ds)})')
    L.append(f'| {c[0]} | {c[1]} | {c[2]} | '+' | '.join(row)+' |')
L.append('\n## Seed-Ebene (Median über Paare je Zelle und Seed), Vorzeichentest einseitig\n'); L.append('| Bereich | Vergleich | Seed-Einheiten besser | p |'); L.append('|---|---|---|---:|')
for name,cs in [('flach',set(flat)),('tief',set(deep))]:
    for ref in ['precap','silu','gelu']:
        g=defaultdict(list)
        for k,d in pairs(None,ref,cs=cs): g[(k[0],k[1],k[2],k[5])].append(d)
        u=[st.median(v) for v in g.values()]; w=sum(x<0 for x in u)
        L.append(f'| {name} | {ref} | {w}/{len(u)} | {sign_p(w,len(u)):.3g} |')
L.append('\n## Sensitivität ohne bad-Runs\n')
bk={tuple(r[k] for k in ('target','depth','arch','init_k','lr','seed')) for r in badrows}
fl2=[(k,d) for k,d in fl if (k[0],str(k[1]),k[2],k[3],k[4],str(k[5])) not in bk]
dp2=[(k,d) for k,d in dp if (k[0],str(k[1]),k[2],k[3],k[4],str(k[5])) not in bk]
L.append(f'Flach ohne betroffene Paare: Median d {summ(fl2)[0]:+.4f}, Win-Rate {summ(fl2)[1]:.1%}; tief: Median d {summ(dp2)[0]:+.3f}, Win-Rate {summ(dp2)[1]:.1%}.')
out=glob.glob('../Notizen Zwischen*/capexp_roadmap/v11_confirm')[0]
open(out+'/CapExp_v11_confirm_Auswertung.md','w').write('\n'.join(L))
print('\n'.join(L)); print('SAVED',out)
