# B2 Auswertung nach Praeregistrierung (B2_Praeregistrierung.md)
# Aufruf: python b2_analyze.py [B2-Ordner] [B1-Ordner]   (Pfade werden absolut aufgeloest, Lehre aus B1)
# Varianten LB, LBh, kL2B (BPreCap), kL2Bh (BPreCap hart) auf den Seeds 42072-42083. Zwei Laeufe auf Seed 42060 pruefen die Gleichheit mit B1.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B1D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b1').resolve()
NEWSEEDS = set(range(42072, 42084)); H1_MAX = 0.5          # H1: mediane fp32-Differenz LB - LBh hoechstens +0,5 pp in jeder Zelle
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
DEPTHS = (100, 200); CELLS = [(ds, d) for d in DEPTHS for ds in ('mnist', 'fashion')]
ACTS = ('LB', 'LBh', 'kL2B', 'kL2Bh')
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(B1D)} if (B1D / 'workers').exists() else {}
new = read(ROOT); REP = {}; D = {}
for r in new:
    if int(r['seed']) in NEWSEEDS: D[K(r)] = r
    else: REP[K(r)] = r
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0      # Ausfall = 90 % (wie Lagma-1 bis B1)
W4 = lambda r: v(r, 'test_err_w8a4_best'); W4A = lambda r: v(r, 'test_err_w8a4a_best'); F32 = lambda r: v(r, 'test_err_fp32')
W3 = lambda r: v(r, 'test_err_w8a3_best'); W3A = lambda r: v(r, 'test_err_w8a3a_best')
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
def overall(res, ref):   # wie Lagma-5/B1: mindestens 2 Zellen in eine Richtung und keine in die andere
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
fam = lambda cand, ref, f: holm([test(ds, d, cand, ref, f) for ds, d in CELLS])
prim = fam('LB', 'LBh', W4)
SEC = {'S1_LB_vs_LBh_fp32': ('LB', 'LBh', F32), 'S2_kL2B_vs_kL2Bh_w4': ('kL2B', 'kL2Bh', W4), 'S3_kL2B_vs_kL2Bh_fp32': ('kL2B', 'kL2Bh', F32),
       'S4_LB_vs_LBh_w4a': ('LB', 'LBh', W4A), 'S5_LB_vs_LBh_w3': ('LB', 'LBh', W3), 'S6_kL2Bh_vs_LBh_fp32': ('kL2Bh', 'LBh', F32),
       'S7_kL2B_vs_LB_w4': ('kL2B', 'LB', W4), 'S8_kL2B_vs_LB_fp32': ('kL2B', 'LB', F32)}
sec = {k: fam(*t) for k, t in SEC.items()}; sv = {k: overall(sec[k], SEC[k][1]) for k in sec}
H1 = all(r['med_diff_pp'] <= H1_MAX for r in sec['S1_LB_vs_LBh_fp32'])
verdict = overall(prim, 'LBh'); success = verdict == 'besser als LBh' and H1
# Kontrolle: LB und kL2B MNIST d100 Seed 42060 bitgleich zu B1 (fp32, w8a4_best, w8a4a_best, Trainingsverlust)
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r[c] == a[c] for c in ('test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'train_loss_last50'))))
repro_ok = len(repro) == 2 and all(x['equal'] for x in repro)
desc = []
for ds, d in CELLS:
    for act in ACTS:
        allr = [D[k] for k in D if k[:3] == (ds, d, act)]; ok = [r for r in allr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=act, n_ok=len(ok), n_all=len(allr))
        if ok:
            med = lambda col: float(np.median([float(r[col]) for r in ok]))
            mf = lambda f: float(np.median([f(r) for r in allr]))
            e.update(fp32=mf(F32), w8a4_best=mf(W4), w8a4a_best=mf(W4A), w8a3_best=mf(W3), w8a3a_best=mf(W3A),
                     w8a8=float(np.median([v(r, 'test_err_w8a8') for r in ok])), range_max=med('act_range_max_median'),
                     grad_log10=med('grad_log10_first_last'), train_loss=med('train_loss_last50'), ms_per_step=med('ms_per_step'),
                     sat_rate=med('sat_rate'), fp32_max=float(max(F32(r) for r in allr)))
        desc.append(e)
# Bezug zu Blumenfeld et al. 2019 (beschreibend): Anstieg der 3-Bit-Kosten (w8a3_best - fp32) von d100 auf d200 je Form und Datensatz
blum = []
for ds in ('mnist', 'fashion'):
    for act in ACTS:
        g = {e['depth']: e for e in desc if e['dataset'] == ds and e['act'] == act and 'fp32' in e}
        if 100 in g and 200 in g:
            blum.append(dict(dataset=ds, act=act, cost3_d100=g[100]['w8a3_best'] - g[100]['fp32'], cost3_d200=g[200]['w8a3_best'] - g[200]['fp32'],
                             cost4_d100=g[100]['w8a4_best'] - g[100]['fp32'], cost4_d200=g[200]['w8a4_best'] - g[200]['fp32']))
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=verdict, H1=H1, success=success,
           secondary=sec, secondary_verdicts=sv, repro=repro, repro_ok=repro_ok, descriptive=desc, blumenfeld=blum)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b2_result.json', 'w'), indent=1)
print('Urteil LB (weich) vs LBh (hart), w8a4_best:', verdict, '| H1 fp32 LB - LBh <= +0,5 pp je Zelle:', H1, '| Erfolg:', success, '| Repro ok:', repro_ok, '| Ausfaelle:', out['failed'])
for r in prim: print(f"  {r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
print('Sekundaer:', sv)
for r in sec['S1_LB_vs_LBh_fp32']: print(f"  H1 {r['dataset']:7s} d{r['depth']:3d} fp32 med={r['med_diff_pp']:+.2f}pp")
