# Q2 Auswertung nach Praeregistrierung (Q2_Praeregistrierung.md)
# Aufruf: python q2_analyze.py [Q2-Ordner] [Q1b-Ordner]
# TAT neu in Q2; SPreCap-cs und SELU aus Q1b (gleiche Seeds, gleicher Code-Pfad). Zwei cs-Laeufe in Q2 pruefen die Uebernahme.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
Q1B = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'q1b'
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
DEPTHS = (50, 100); DS = ('mnist', 'fashion'); CELLS = [(ds, d) for d in DEPTHS for ds in DS]
new = read(ROOT); old = read(Q1B)
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
D = {K(r): r for r in old if r['act'] in ('spreccap_cs', 'selu')}
REP = {}
for r in new:
    if r['act'] == 'tat': D[K(r)] = r
    else: REP[K(r)] = r
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0          # Ausfall = 90 % (vorab festgelegt)
def deg(r): return v(r, 'test_err_w8a4_best') - v(r, 'test_err_fp32') if r['status'] == 'ok' else 90.0
def test(ds, d, cand, ref, f):
    x = []
    for s in sorted({k[3] for k in D}):
        a, b = D.get((ds, d, cand, s)), D.get((ds, d, ref, s))
        if a is not None and b is not None: x.append(f(a) - f(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    for r in res: r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
def overall(res, ref):
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
W4 = lambda r: v(r, 'test_err_w8a4_best'); F32 = lambda r: v(r, 'test_err_fp32')
prim = holm([test(ds, d, 'spreccap_cs', 'tat', W4) for ds, d in CELLS])
sec = {'S1_cs_vs_tat_fp32': holm([test(ds, d, 'spreccap_cs', 'tat', F32) for ds, d in CELLS]),
       'S2_cs_vs_tat_degradation': holm([test(ds, d, 'spreccap_cs', 'tat', deg) for ds, d in CELLS]),
       'S3_tat_vs_selu_w8a4best': holm([test(ds, d, 'tat', 'selu', W4) for ds, d in CELLS])}
sv = {'S1': overall(sec['S1_cs_vs_tat_fp32'], 'TAT'), 'S2': overall(sec['S2_cs_vs_tat_degradation'], 'TAT'), 'S3': overall(sec['S3_tat_vs_selu_w8a4best'], 'SELU')}
repro = []
for k, r in sorted(REP.items()):
    a = D.get(k)
    repro.append(dict(key=list(k), q2_fp32=F32(r), q1b_fp32=F32(a) if a else None, q2_w4=W4(r), q1b_w4=W4(a) if a else None,
                      diff_fp32=(F32(r) - F32(a)) if a else None, diff_w4=(W4(r) - W4(a)) if a else None))
repro_ok = bool(repro) and all(x['diff_fp32'] is not None and abs(x['diff_fp32']) <= 0.10 and abs(x['diff_w4']) <= 0.10 for x in repro)
desc = []
for ds, d in CELLS:
    for act in ('spreccap_cs', 'tat', 'selu'):
        allr = [D[k] for k in D if k[:3] == (ds, d, act)]; ok = [r for r in allr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=act, n_ok=len(ok), n_all=len(allr))
        if ok:
            e.update(lr=float(ok[0]['lr']), fp32=float(np.median([F32(r) for r in allr])), w8a8=float(np.median([v(r, 'test_err_w8a8') for r in ok])),
                     w8a4_max=float(np.median([v(r, 'test_err_w8a4') for r in ok])), w8a4_best=float(np.median([W4(r) for r in allr])),
                     range_max=float(np.median([float(r['act_range_max_median']) for r in ok])), clip_mse4=float(np.median([float(r['act_clip_mse4_median']) for r in ok])),
                     best_cal={c: sum(r['w8a4_best_cal'] == c for r in ok) for c in ('max', 'p999', 'p9999', 'mse4')})
        desc.append(e)
lrt = list(csv.DictReader(open(ROOT / 'lr_tuned.csv'))) if (ROOT / 'lr_tuned.csv').exists() else []
out = dict(n_rows_q2=len(new), failed_q2=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=overall(prim, 'TAT'), secondary=sec,
           secondary_verdicts=sv, repro=repro, repro_ok=repro_ok, descriptive=desc, lr_tuned=lrt, lr_at_edge=[r for r in lrt if r.get('at_edge') == 'True'])
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'q2_result.json', 'w'), indent=1)
print('Urteil cs vs TAT (w8a4_best):', out['verdict'], '| Sekundaer:', sv, '| Repro ok:', repro_ok, '| Ausfaelle:', out['failed_q2'], '| LR am Rand:', len(out['lr_at_edge']))
for r in prim: print(f"{r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
