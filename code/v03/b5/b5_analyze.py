# B5 Auswertung nach Praeregistrierung (B5_Praeregistrierung.md)
# Aufruf: python b5_analyze.py [B5-Ordner] [B1-Ordner] [B3-Ordner]   (Pfade werden absolut aufgeloest)
# CIFAR-10, Tiefe 200, kL2B (BPreCap), LB, kL2 mit nachgetunter LR, tat mit LR aus B3, Seeds 42108-42119.
# Kontrollen: kL2B MNIST d100 Seed 42060 gegen B1; Tuning-Zellen mit LR 1e-5 (Seeds 42000-42003) gegen das Tuning von B3.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B1D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b1').resolve()
B3D = (Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / 'b3').resolve()
NEWSEEDS = set(range(42108, 42120)); DEPTH = 200; ACTS = ('kL2B', 'LB', 'kL2', 'tat')
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
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0      # Ausfall = 90 % (wie B1 bis B4)
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
# Primaer: kL2B - LB fp32, eine Zelle, alpha 0,05
prim = test('kL2B', 'LB', F32); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
SEC = {'S1_kL2B_vs_LB_w4': ('kL2B', 'LB', W4), 'S2_kL2B_vs_LB_w4a': ('kL2B', 'LB', W4A), 'S3_kL2_vs_LB_fp32': ('kL2', 'LB', F32),
       'S4_kL2B_vs_kL2_fp32': ('kL2B', 'kL2', F32), 'S5_kL2B_vs_tat_fp32': ('kL2B', 'tat', F32), 'S6_kL2B_vs_tat_w4a': ('kL2B', 'tat', W4A),
       'S7_LB_vs_tat_w4a': ('LB', 'tat', W4A), 'S8_kL2B_vs_tat_w4': ('kL2B', 'tat', W4)}
sec = holm({k: test(*t) for k, t in SEC.items()})
# Kontrollen
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r[c] == a[c] for c in ('test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'train_loss_last50'))))
repro_ok = len(repro) == 1 and all(x['equal'] for x in repro)
T5 = read(ROOT / 'tune'); T3 = {r['run_id']: r for r in read(B3D / 'tune')} if (B3D / 'tune' / 'workers').exists() else {}
tune_repro = []
for r in T5:
    if abs(float(r['lr']) - 1e-5) < 1e-12:
        a = T3.get(r['run_id'])
        tune_repro.append(dict(run_id=r['run_id'], equal=a is not None and all(r[c] == a[c] for c in ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'train_loss_last50'))))
tune_repro_ok = len(tune_repro) == 12 and all(x['equal'] for x in tune_repro)
# Tuning-Kurven und LR-Wahl (beschreibend). Randlage zaehlt nur unten (3e-6); oben schliesst B3 an (1e-5 ... 5e-4, dort war 1e-5 am besten)
curve = {}
for r in T5:
    curve.setdefault(f"{r['act']}_lr{float(r['lr']):g}", []).append((1 - float(r['val_err_fp32'])) if r['status'] == 'ok' else 0.1)
curve = {k: float(np.median(x)) for k, x in sorted(curve.items())}
lr = {}
lp = ROOT / 'lr_tuned.csv'
if lp.exists():
    for r in csv.DictReader(open(lp)): lr[f"{r['act']}_d{r['depth']}"] = dict(lr=float(r['lr_sel']), med_val_acc=float(r['med_val_acc']), at_lower_edge=abs(float(r['lr_sel']) - 3e-6) < 1e-12)
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
p8 = dict(order_fp32=sorted(ACTS, key=lambda a: m.get(a) if m.get(a) is not None else 1e9), p8_order_holds=bool(m.get('kL2', 1e9) < m.get('kL2B', 1e9) < m.get('LB', 1e9)))
out = dict(n_rows=len(new), n_tune=len(T5), failed=sum(r['status'] != 'ok' for r in new), failed_tune=sum(r['status'] != 'ok' for r in T5),
           primary=prim, success=success, secondary=sec, secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro=repro, repro_ok=repro_ok,
           tune_repro=tune_repro, tune_repro_ok=tune_repro_ok, tune_curve=curve, lr_tuned=lr, descriptive=desc, p8_prediction=p8)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b5_result.json', 'w'), indent=1)
print(f"Primaer kL2B - LB fp32 d200: med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Kontrolle B1:', repro_ok, '| Tuning 1e-5 gleich B3:', tune_repro_ok, f"({sum(x['equal'] for x in tune_repro)}/{len(tune_repro)})", '| Ausfaelle:', out['failed'], out['failed_tune'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
print('LR:', lr); print('Kurve:', {k: round(x, 4) for k, x in curve.items()}); print('p8:', p8)
