# B8 Auswertung nach Praeregistrierung (B8_Praeregistrierung.md)
# Aufruf: python b8_analyze.py [B8-Ordner] [B1-Ordner]   (Pfade werden absolut aufgeloest)
# CIFAR-10, Tiefe 200, kL2B_tr und LB_tr (TAT-Rezept, LR getunt, Raster 5e-6 ... 2e-4), tat (LR aus B3 d200), Seeds 42144-42155.
# Kontrolle: kL2B MNIST d100 Seed 42060 gegen B1.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B1D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b1').resolve()
NEWSEEDS = set(range(42144, 42156)); DEPTH = 200; ACTS = ('kL2B_tr', 'LB_tr', 'tat'); GRID_EDGES = (5e-6, 2e-4)
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(B1D)} if (B1D / 'workers').exists() else {}
new = read(ROOT); REP = {}; D = {}
for r in new:
    if r['dataset'] == 'cifar10' and int(r['seed']) in NEWSEEDS: D[K(r)] = r
    else: REP[K(r)] = r
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0      # Ausfall = 90 % (wie B1 bis B7)
F32 = lambda r: v(r, 'test_err_fp32'); W4 = lambda r: v(r, 'test_err_w8a4_best'); W4A = lambda r: v(r, 'test_err_w8a4a_best'); W3 = lambda r: v(r, 'test_err_w8a3_best')
def test(cand, ref, f):
    x = []
    for s in sorted(NEWSEEDS):
        a, b = D.get(('cifar10', DEPTH, cand, s)), D.get(('cifar10', DEPTH, ref, s))
        if a is not None and b is not None: x.append(f(a) - f(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def verdict(r, pcol):
    return 'unklar' if r[pcol] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):   # res: dict name -> Testergebnis; Holm ueber alle Eintraege
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
# Primaer: kL2B_tr - tat bei w8a4a_best (faire 4 Bit gegen TAT, gleiches Rezept), eine Zelle, alpha 0,05
prim = test('kL2B_tr', 'tat', W4A); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
SEC = {'S1_kL2Btr_vs_tat_fp32': ('kL2B_tr', 'tat', F32), 'S2_kL2Btr_vs_tat_w4': ('kL2B_tr', 'tat', W4), 'S3_LBtr_vs_tat_w4a': ('LB_tr', 'tat', W4A),
       'S4_kL2Btr_vs_LBtr_fp32': ('kL2B_tr', 'LB_tr', F32), 'S5_kL2Btr_vs_LBtr_w4a': ('kL2B_tr', 'LB_tr', W4A), 'S6_LBtr_vs_tat_fp32': ('LB_tr', 'tat', F32),
       'S7_LBtr_vs_tat_w4': ('LB_tr', 'tat', W4), 'S8_kL2Btr_vs_LBtr_w4': ('kL2B_tr', 'LB_tr', W4)}
sec = holm({k: test(*t) for k, t in SEC.items()})
# Kontrollen
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r[c] == a[c] for c in ('test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'train_loss_last50'))))
repro_ok = len(repro) == 1 and all(x['equal'] for x in repro)
T5 = read(ROOT / 'tune')
# Tuning-Kurven und LR-Wahl (beschreibend), Raster 5e-6 ... 2e-4, Randlage an beiden Enden
curve = {}
for r in T5:
    curve.setdefault(f"{r['act']}_lr{float(r['lr']):g}", []).append((1 - float(r['val_err_fp32'])) if r['status'] == 'ok' else 0.1)
curve = {k: float(np.median(x)) for k, x in sorted(curve.items())}
lr = {}
lp = ROOT / 'lr_tuned.csv'
if lp.exists():
    for r in csv.DictReader(open(lp)): lr[f"{r['act']}_d{r['depth']}"] = dict(lr=float(r['lr_sel']), med_val_acc=float(r['med_val_acc']), at_edge=float(r['lr_sel']) in GRID_EDGES)
desc = []
for act in ACTS:
    allr = [D[k] for k in D if k[2] == act]; ok = [r for r in allr if r['status'] == 'ok']
    e = dict(act=act, n_ok=len(ok), n_all=len(allr), lr=float(allr[0]['lr']) if allr else None)
    if ok:
        med = lambda col: float(np.median([float(r[col]) for r in ok if r.get(col) not in (None, '')])) if any(r.get(col) not in (None, '') for r in ok) else None
        mf = lambda f: float(np.median([f(r) for r in allr]))
        e.update(fp32=mf(F32), w8a4_best=mf(W4), w8a4a_best=mf(W4A), w8a3_best=mf(W3), w8a8=float(np.median([v(r, 'test_err_w8a8') for r in ok])),
                 range_max=med('act_range_max_median'), sat_rate=med('sat_rate'), train_loss=med('train_loss_last50'))
    desc.append(e)
m = {e['act']: e.get('fp32') for e in desc}
order = sorted(ACTS, key=lambda a: m.get(a) if m.get(a) is not None else 1e9)
out = dict(n_rows=len(new), n_tune=len(T5), failed=sum(r['status'] != 'ok' for r in new), failed_tune=sum(r['status'] != 'ok' for r in T5),
           primary=prim, success=success, secondary=sec, secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro=repro, repro_ok=repro_ok,
           tune_curve=curve, lr_tuned=lr, descriptive=desc, order_fp32=order)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b8_result.json', 'w'), indent=1)
print(f"Primaer kL2B_tr - tat w8a4a_best d200: med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Kontrolle B1:', repro_ok, '| Ausfaelle:', out['failed'], out['failed_tune'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
print('LR:', lr); print('Kurve:', {k: round(x, 4) for k, x in curve.items()}); print('fp32-Ordnung:', order)
