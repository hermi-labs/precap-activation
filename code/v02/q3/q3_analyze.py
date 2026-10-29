# Q3 Auswertung nach Praeregistrierung (Q3_Praeregistrierung.md)
# Aufruf: python q3_analyze.py [Q3-Ordner] [Q2-Ordner] [Q1b-Ordner]
# Alle drei Aktivierungen neu trainiert. Reproduktion: fp32 und symmetrisches w8a4_best je Lauf gegen Q1b (cs, SELU) bzw. Q2 (TAT).
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
Q2 = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'q2'
Q1B = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / 'q1b'
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
DEPTHS = (50, 100); DS = ('mnist', 'fashion'); CELLS = [(ds, d) for d in DEPTHS for ds in DS]
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
new = read(ROOT); D = {K(r): r for r in new}
OLD = {K(r): r for r in read(Q1B) if r['act'] in ('spreccap_cs', 'selu')}
OLD.update({K(r): r for r in read(Q2) if r['act'] == 'tat'})
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0          # Ausfall = 90 % (vorab festgelegt)
A4 = lambda r: v(r, 'test_err_w8a4a_best'); S4 = lambda r: v(r, 'test_err_w8a4_best'); F32 = lambda r: v(r, 'test_err_fp32')
def test(ds, d, ca, fa, cb, fb):
    x = []
    for s in sorted({k[3] for k in D}):
        a, b = D.get((ds, d, ca, s)), D.get((ds, d, cb, s))
        if a is not None and b is not None: x.append(fa(a) - fb(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    for r in res: r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
def overall(res, ref):
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
fam = lambda ca, fa, cb, fb: holm([test(ds, d, ca, fa, cb, fb) for ds, d in CELLS])
prim = fam('spreccap_cs', A4, 'tat', A4)
sec = {'S1_cs_vs_selu_asym': fam('spreccap_cs', A4, 'selu', A4), 'S2_tat_asym_vs_sym': fam('tat', A4, 'tat', S4),
       'S3_cs_asym_vs_sym': fam('spreccap_cs', A4, 'spreccap_cs', S4), 'S4_cs_vs_tat_sym': fam('spreccap_cs', S4, 'tat', S4)}
refs = {'S1_cs_vs_selu_asym': 'SELU', 'S2_tat_asym_vs_sym': 'TAT-sym', 'S3_cs_asym_vs_sym': 'cs-sym', 'S4_cs_vs_tat_sym': 'TAT'}
sv = {k: overall(sec[k], refs[k]) for k in sec}
rd = []
for k, r in D.items():
    o = OLD.get(k)
    if o is not None and r['status'] == 'ok' and o['status'] == 'ok': rd += [abs(F32(r) - F32(o)), abs(S4(r) - S4(o))]
n_matched = sum(1 for k in D if k in OLD)
repro = dict(n_matched=n_matched, n_runs=len(D), max_abs_diff_pp=float(max(rd)) if rd else None)
repro_ok = bool(rd) and n_matched == len(D) and max(rd) <= 0.10
desc = []
for ds, d in CELLS:
    for act in ('spreccap_cs', 'tat', 'selu'):
        allr = [D[k] for k in D if k[:3] == (ds, d, act)]; ok = [r for r in allr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=act, n_ok=len(ok), n_all=len(allr))
        if ok:
            e.update(fp32=float(np.median([F32(r) for r in allr])), w8a4_best_sym=float(np.median([S4(r) for r in allr])), w8a4a_best=float(np.median([A4(r) for r in allr])),
                     w8a8a_max=float(np.median([v(r, 'test_err_w8a8a_max') for r in ok])), w8a4a_max=float(np.median([v(r, 'test_err_w8a4a_max') for r in ok])),
                     lo_max=float(np.median([float(r['act_lo_max_median']) for r in ok])), range_max=float(np.median([float(r['act_range_max_median']) for r in ok])),
                     best_cal_asym={c: sum(r['w8a4a_best_cal'] == c for r in ok) for c in ('max', 'p999', 'p9999', 'mse4')})
        desc.append(e)
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=overall(prim, 'TAT'), secondary=sec, secondary_verdicts=sv,
           repro=repro, repro_ok=repro_ok, descriptive=desc)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'q3_result.json', 'w'), indent=1)
print('Urteil cs vs TAT (w8a4a_best, asymmetrisch):', out['verdict'], '| Sekundaer:', sv, '| Repro ok:', repro_ok, repro, '| Ausfaelle:', out['failed'])
for r in prim: print(f"{r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
