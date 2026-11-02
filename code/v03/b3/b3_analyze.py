# B3 Auswertung nach Praeregistrierung (B3_Praeregistrierung.md)
# Aufruf: python b3_analyze.py [B3-Ordner] [B1-Ordner]   (Pfade werden absolut aufgeloest)
# CIFAR-10, Tiefe 100 und 200, Varianten kL2B (BPreCap), LB, kL2, tat auf den Seeds 42084-42095. Ein Lauf (kL2B MNIST d100 Seed 42060) prueft B1.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B1D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b1').resolve()
NEWSEEDS = set(range(42084, 42096))
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
CELLS = [('cifar10', 100), ('cifar10', 200)]
ACTS = ('kL2B', 'LB', 'kL2', 'tat')
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(B1D)} if (B1D / 'workers').exists() else {}
new = read(ROOT); REP = {}; D = {}
for r in new:
    if r['dataset'] == 'cifar10' and int(r['seed']) in NEWSEEDS: D[K(r)] = r
    else: REP[K(r)] = r
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0      # Ausfall = 90 % (wie Lagma-1 bis B2)
F32 = lambda r: v(r, 'test_err_fp32'); W4 = lambda r: v(r, 'test_err_w8a4_best'); W4A = lambda r: v(r, 'test_err_w8a4a_best'); W3 = lambda r: v(r, 'test_err_w8a3_best')
def test(ds, d, cand, ref, f):
    x = []
    for s in sorted(NEWSEEDS):
        a, b = D.get((ds, d, cand, s)), D.get((ds, d, ref, s))
        if a is not None and b is not None: x.append(f(a) - f(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(dataset=ds, depth=d, cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    for r in res: r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
def overall(res, ref):   # bei 2 Zellen: beide in eine Richtung (Regel wie Lagma-5/B1/B2: mindestens 2 und keine Gegenrichtung)
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
fam = lambda cand, ref, f: holm([test(ds, d, cand, ref, f) for ds, d in CELLS])
prim = fam('kL2B', 'LB', F32)
SEC = {'S1_kL2B_vs_LB_w4': ('kL2B', 'LB', W4), 'S2_kL2B_vs_kL2_fp32': ('kL2B', 'kL2', F32), 'S3_kL2B_vs_kL2_w4': ('kL2B', 'kL2', W4),
       'S4_kL2B_vs_tat_fp32': ('kL2B', 'tat', F32), 'S5_kL2B_vs_tat_w4a': ('kL2B', 'tat', W4A), 'S6_LB_vs_tat_w4a': ('LB', 'tat', W4A),
       'S7_kL2B_vs_LB_w3': ('kL2B', 'LB', W3), 'S8_kL2_vs_LB_fp32': ('kL2', 'LB', F32)}
sec = {k: fam(*t) for k, t in SEC.items()}; sv = {k: overall(sec[k], SEC[k][1]) for k in sec}
verdict = overall(prim, 'LB'); success = verdict == 'besser als LB'
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r[c] == a[c] for c in ('test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'train_loss_last50'))))
repro_ok = len(repro) == 1 and all(x['equal'] for x in repro)
lr = {}
lp = ROOT / 'lr_tuned.csv'
if lp.exists():
    for r in csv.DictReader(open(lp)): lr[f"{r['act']}_d{r['depth']}"] = dict(lr=float(r['lr_sel']), med_val_acc=float(r['med_val_acc']), at_edge=r['at_edge'] == 'True')
desc = []
for ds, d in CELLS:
    for act in ACTS:
        allr = [D[k] for k in D if k[:3] == (ds, d, act)]; ok = [r for r in allr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=act, n_ok=len(ok), n_all=len(allr))
        if ok:
            med = lambda col: float(np.median([float(r[col]) for r in ok if r.get(col) not in (None, '')])) if any(r.get(col) not in (None, '') for r in ok) else None
            mf = lambda f: float(np.median([f(r) for r in allr]))
            e.update(fp32=mf(F32), w8a4_best=mf(W4), w8a4a_best=mf(W4A), w8a3_best=mf(W3), w8a3a_best=mf(lambda r: v(r, 'test_err_w8a3a_best')),
                     w8a8=float(np.median([v(r, 'test_err_w8a8') for r in ok])), range_max=med('act_range_max_median'), grad_log10=med('grad_log10_first_last'),
                     train_loss=med('train_loss_last50'), ms_per_step=med('ms_per_step'), sat_rate=med('sat_rate'), fp32_max=float(max(F32(r) for r in allr)))
        desc.append(e)
# Vorhersage aus math/p8 (beschreibend): wenn Nichtlinearitaet bei der Init zaehlt, ist fp32 geordnet kL2 < kL2B < LB
p8 = {}
for ds, d in CELLS:
    m = {e['act']: e.get('fp32') for e in desc if e['dataset'] == ds and e['depth'] == d}
    p8[f'd{d}'] = dict(order_fp32=sorted(ACTS, key=lambda a: m.get(a, 1e9)), p8_order_holds=(m.get('kL2', 1e9) < m.get('kL2B', 1e9) < m.get('LB', 1e9)))
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=verdict, success=success,
           secondary=sec, secondary_verdicts=sv, repro=repro, repro_ok=repro_ok, lr_tuned=lr, descriptive=desc, p8_prediction=p8)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b3_result.json', 'w'), indent=1)
print('Urteil kL2B (BPreCap) vs LB, fp32 auf CIFAR-10:', verdict, '| Erfolg:', success, '| Repro ok:', repro_ok, '| Ausfaelle:', out['failed'])
for r in prim: print(f"  d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
print('Sekundaer:', sv); print('LR:', lr); print('p8-Ordnung:', p8)
