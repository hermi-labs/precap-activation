# Studie A3 Auswertung nach Praeregistrierung (A3_Praeregistrierung.md)
# Aufruf: python a3_analyze.py [A3-Ordner] [A2-Ordner] [A-Ordner]
# TAT (beide Budgets) stammt aus A2; kL2 (Gauss-Rezept) kurz aus A, lang aus A2. Zwei TAT-Laeufe in A3 pruefen die Uebernahme.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
A2 = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'a2'
A = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / 'a'
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
SHORT, LONG = 1720, 6880; DEPTHS = (100, 200); DS = ('mnist', 'fashion'); CELLS = [(ds, d) for d in DEPTHS for ds in DS]; BUD = (SHORT, LONG)
new = read(ROOT); a2 = read(A2); a = read(A)
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['steps']) if r.get('steps') else SHORT, int(r['seed']))
D = {}
for r in a:
    if r['act'] == 'kL2' and int(r['depth']) in DEPTHS: D[K(r)] = r                     # kL2 kurz aus A (A hat keine steps-Spalte -> 1720)
for r in a2:
    if r['act'] == 'tat' or (r['act'] == 'kL2' and int(r['steps']) == LONG): D[K(r)] = r  # TAT beide Budgets, kL2 lang aus A2
REP = {}
for r in new:
    if r['act'] == 'tat': REP[K(r)] = r
    else: D[K(r)] = r
def val(r): return float(r['test_err_fp32']) * 100 if r['status'] == 'ok' else 90.0
def test(ds, d, T, cand, ref):
    x = []
    for s in sorted({k[4] for k in D}):
        u, v = D.get((ds, d, cand, T, s)), D.get((ds, d, ref, T, s))
        if u is not None and v is not None: x.append(val(u) - val(v))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, steps=T, cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    for r in res: r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
def overall(res, ref):
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
prim = holm([test(ds, d, T, 'kL2r', 'tat') for T in BUD for ds, d in CELLS])
verdict = overall(prim, 'TAT'); by_budget = {T: overall([r for r in prim if r['steps'] == T], 'TAT') for T in BUD}
s1 = holm([test(ds, d, T, 'kL2r', 'kL2') for T in BUD for ds, d in CELLS])
repro = []
for k, r in sorted(REP.items()):
    ref = D.get(k); repro.append(dict(key=list(k), a3=val(r), a2=val(ref) if ref else None, diff_pp=(val(r) - val(ref)) if ref else None))
repro_ok = bool(repro) and all(x['diff_pp'] is not None and abs(x['diff_pp']) <= 0.10 for x in repro)
desc = []
for ds, d in CELLS:
    for T in BUD:
        for act in ('kL2r', 'kL2', 'tat'):
            allr = [D[k] for k in D if k[:4] == (ds, d, act, T)]; ok = [r for r in allr if r['status'] == 'ok']
            e = dict(dataset=ds, depth=d, steps=T, act=act, n_ok=len(ok), n_all=len(allr))
            if ok:
                e.update(lr=float(ok[0]['lr']), med_test_err=float(np.median([val(r) for r in allr])),
                         med_train_loss=float(np.median([float(r['train_loss_last50']) for r in ok])),
                         med_range_max=float(np.median([float(r['act_range_max_median']) for r in ok])) if ok[0].get('act_range_max_median') else None,
                         med_grad=float(np.nanmedian([float(r['grad_log10_first_last']) for r in ok])) if ok[0].get('grad_log10_first_last') else None)
            desc.append(e)
lrt = list(csv.DictReader(open(ROOT / 'lr_tuned.csv'))) if (ROOT / 'lr_tuned.csv').exists() else []
out = dict(n_rows_a3=len(new), failed_a3=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=verdict, verdict_by_budget=by_budget,
           secondary_S1=s1, S1_verdict=overall(s1, 'kL2-Gauss'), repro=repro, repro_ok=repro_ok, descriptive=desc, lr_tuned=lrt,
           lr_at_edge=[r for r in lrt if r.get('at_edge') == 'True'])
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'a3_result.json', 'w'), indent=1)
print('Urteil kL2r vs TAT:', verdict, by_budget, '| S1 kL2r vs kL2:', out['S1_verdict'], '| Repro ok:', repro_ok, '| Ausfaelle:', out['failed_a3'], '| LR am Rand:', len(out['lr_at_edge']))
for r in prim: print(f"T{r['steps']} {r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
