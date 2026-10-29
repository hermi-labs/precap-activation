# Idee A Auswertung nach Praeregistrierung (A_Praeregistrierung.md)
# Aufruf: python a_analyze.py [A-Ordner]
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
rows = []
for f in sorted((ROOT / 'workers').glob('w*.csv')): rows += list(csv.DictReader(open(f)))
DEPTHS = (50, 100, 200); DS = ('mnist', 'fashion'); ACTS = ('spreccap_cs', 'spreccap_e03', 'kL1', 'kL2', 'selu'); NEW = ('kL1', 'kL2')
CELLS = [(ds, d) for d in DEPTHS for ds in DS]; DEEP = [(ds, d) for d in (100, 200) for ds in DS]
D = {(r['dataset'], int(r['depth']), r['act'], int(r['seed'])): r for r in rows}
seeds = sorted({int(r['seed']) for r in rows})
def val(r, m):
    if m == 'fp32': return float(r['test_err_fp32']) * 100 if r['status'] == 'ok' else 90.0   # Ausfall = Zufallsniveau (vorab festgelegt)
    if r['status'] != 'ok': return None
    return (float(r[f'test_err_{m}']) - float(r['test_err_fp32'])) * 100                      # Quantisierungsverlust (pp)
def test(ds, d, cand, ref, m='fp32'):
    x = []
    for s in seeds:
        a, b = D.get((ds, d, cand, s)), D.get((ds, d, ref, s))
        if a is None or b is None: continue
        va, vb = val(a, m), val(b, m)
        if va is None or vb is None: continue
        x.append(va - vb)
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, cand=cand, ref=ref, metric=m, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    for r in res: r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
def overall(res):
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return 'besser als SELU' if nb >= 2 and ns == 0 else ('schlechter als SELU' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
prim = holm([test(ds, d, k, 'selu') for k in NEW for ds, d in DEEP])
verd = {k: overall([r for r in prim if r['cand'] == k]) for k in NEW}
sec = {}
sec['S1_vs_cs_fp32'] = holm([test(ds, d, k, 'spreccap_cs') for k in NEW for ds, d in CELLS])
sec['S2_vs_e03_fp32'] = holm([test(ds, d, k, 'spreccap_e03') for k in NEW for ds, d in CELLS])
sec['S3_d50_vs_selu_fp32'] = holm([test(ds, 50, k, 'selu') for k in NEW for ds in DS])
sec['S4_fixed_vs_selu_fp32'] = holm([test(ds, d, k, 'selu') for k in ('spreccap_cs', 'spreccap_e03') for ds, d in CELLS])
for m in ('w8a4_best', 'w8a8'):
    sec[f'S5_{m}_vs_selu'] = holm([test(ds, d, k, 'selu', m) for k in NEW + ('spreccap_cs', 'spreccap_e03') for ds, d in CELLS])
    sec[f'S5_{m}_vs_cs'] = holm([test(ds, d, k, 'spreccap_cs', m) for k in NEW for ds, d in CELLS])
C = json.load(open(ROOT / 'a_constants.json')) if (ROOT / 'a_constants.json').exists() else {}
def pred(a, d):
    if a in ('spreccap_cs', 'spreccap_e03'): return C.get(a, {}).get(f'pred_log10_grad_d{d}')
    if a == 'selu': return (d - 1) / 2 * math.log10(1.0716)
    return C.get(f'{a}_d{d}', {}).get('pred_log10_grad')
desc = []
for ds, d in CELLS:
    for a in ACTS:
        allr = [D[k] for k in D if k[:3] == (ds, d, a)]; ok = [r for r in allr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=a, n_ok=len(ok), n_all=len(allr), pred_log10_grad=pred(a, d))
        if allr: e['med_grad_log10_first_last'] = float(np.nanmedian([float(r['grad_log10_first_last']) for r in allr]))
        if ok:
            e.update(lr=float(ok[0]['lr']), chi1=float(ok[0]['chi1']) if ok[0]['chi1'] not in ('', 'nan') else None,
                     med_test_err=float(np.median([val(r, 'fp32') for r in allr])),
                     med_ms_per_step=float(np.median([float(r['ms_per_step']) for r in ok])),
                     med_range_max=float(np.median([float(r['act_range_max_median']) for r in ok])))
            for m in ('w8a8', 'w8a4', 'w8a4_mse', 'w8a4_best'): e[f'med_delta_{m}'] = float(np.median([val(r, m) for r in ok]))
        desc.append(e)
lrt = list(csv.DictReader(open(ROOT / 'lr_tuned.csv'))) if (ROOT / 'lr_tuned.csv').exists() else []
out = dict(n_rows=len(rows), failed=sum(r['status'] != 'ok' for r in rows), primary=prim, verdict=verd, secondary=sec, descriptive=desc, lr_tuned=lrt,
           lr_at_edge=[r for r in lrt if r.get('at_edge') == 'True'])
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'a_result.json', 'w'), indent=1)
print('Urteil:', verd, '| Ausfaelle:', out['failed'], '| LR am Rand:', len(out['lr_at_edge']))
for r in prim: print(f"{r['cand']:5s} {r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
