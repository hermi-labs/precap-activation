# B1 Auswertung nach Praeregistrierung (B1_Praeregistrierung.md)
# Aufruf: python b1_analyze.py [B1-Ordner] [Lagma5-Ordner]
# Varianten kL2B, kL2, LB, kL2_LD100, spreccap_cs auf den Seeds 42060-42071. Zwei Laeufe auf Seed 42045 pruefen die Uebereinstimmung mit Lagma-5.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
L5D = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'lagma5'
NEWSEEDS = set(range(42060, 42072)); H1_MAX = 0.5          # H1: mediane fp32-Differenz kL2B - kL2 hoechstens +0,5 pp in jeder Zelle
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
DEPTHS = (100, 200); CELLS = [(ds, d) for d in DEPTHS for ds in ('mnist', 'fashion')]
ACTS = ('kL2B', 'kL2', 'LB', 'kL2_LD100', 'spreccap_cs')
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(L5D)} if (L5D / 'workers').exists() else {}
new = read(ROOT); REP = {}; D = {}
for r in new:
    if int(r['seed']) in NEWSEEDS: D[K(r)] = r
    else: REP[K(r)] = r
def v(r, col): return float(r[col]) * 100 if r['status'] == 'ok' else 90.0      # Ausfall = 90 % (wie Lagma-1 bis 5)
W4 = lambda r: v(r, 'test_err_w8a4_best'); W4A = lambda r: v(r, 'test_err_w8a4a_best'); F32 = lambda r: v(r, 'test_err_fp32')
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
def overall(res, ref):
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
fam = lambda cand, ref, f: holm([test(ds, d, cand, ref, f) for ds, d in CELLS])
prim = fam('kL2B', 'kL2_LD100', W4)
SEC = {'S1_kL2B_vs_kL2_fp32': ('kL2B', 'kL2', F32), 'S2_kL2B_vs_kL2_w4': ('kL2B', 'kL2', W4),
       'S3_kL2B_vs_LB_fp32': ('kL2B', 'LB', F32), 'S4_kL2B_vs_LB_w4': ('kL2B', 'LB', W4),
       'S5_kL2B_vs_LD100_w4a': ('kL2B', 'kL2_LD100', W4A), 'S6_kL2B_vs_cs_fp32': ('kL2B', 'spreccap_cs', F32),
       'S7_kL2B_vs_cs_w4': ('kL2B', 'spreccap_cs', W4)}
sec = {k: fam(*t) for k, t in SEC.items()}; sv = {k: overall(sec[k], SEC[k][1]) for k in sec}
H1 = all(r['med_diff_pp'] <= H1_MAX for r in sec['S1_kL2B_vs_kL2_fp32'])
verdict = overall(prim, 'kL2_LD100'); success = verdict == 'besser als kL2_LD100' and H1
# Kontrolle: kL2 und kL2_LD100 MNIST d100 Seed 42045 bitgleich zu Lagma-5 (fp32, w8a4_best, Trainingsverlust)
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r[c] == a[c] for c in ('test_err_fp32', 'test_err_w8a4_best', 'train_loss_last50'))))
repro_ok = len(repro) == 2 and all(x['equal'] for x in repro)
desc = []
for ds, d in CELLS:
    for act in ACTS:
        allr = [D[k] for k in D if k[:3] == (ds, d, act)]; ok = [r for r in allr if r['status'] == 'ok']
        e = dict(dataset=ds, depth=d, act=act, n_ok=len(ok), n_all=len(allr))
        if ok:
            med = lambda col: float(np.median([float(r[col]) for r in ok]))
            e.update(fp32=float(np.median([F32(r) for r in allr])), w8a4_best=float(np.median([W4(r) for r in allr])), w8a4a_best=float(np.median([W4A(r) for r in allr])),
                     w8a4_max=float(np.median([v(r, 'test_err_w8a4') for r in ok])), w8a8=float(np.median([v(r, 'test_err_w8a8') for r in ok])),
                     range_max=med('act_range_max_median'), range_p9999=med('act_range_p9999_median'), grad_log10=med('grad_log10_first_last'),
                     train_loss=med('train_loss_last50'), ms_per_step=med('ms_per_step'), fp32_max=float(max(F32(r) for r in allr)))
            if ok[0].get('sat_rate'): e['sat_rate'] = med('sat_rate')
            if ok[0].get('lg_c_median'): e.update(lg_c=med('lg_c_median'), lg_c_ratio=med('lg_c_ratio_median'), lg_intervention=med('lg_intervention'))
        desc.append(e)
base_med = {(e['dataset'], e['depth']): e['fp32'] for e in desc if e['act'] == 'kL2' and 'fp32' in e}
for e in desc:   # Einbrueche (beschreibend): fp32-Fehler ueber dem Doppelten des kL2-Medians der Zelle
    b = base_med.get((e['dataset'], e['depth']))
    e['n_fp32_over_2x_kL2'] = int(sum(F32(D[k]) > 2 * b for k in D if k[:3] == (e['dataset'], e['depth'], e['act']))) if b else None
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=verdict, H1=H1, success=success,
           secondary=sec, secondary_verdicts=sv, repro=repro, repro_ok=repro_ok, descriptive=desc)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b1_result.json', 'w'), indent=1)
print('Urteil kL2B vs kL2_LD100 (w8a4_best):', verdict, '| H1 fp32 vs kL2 (<= +0,5 pp je Zelle):', H1, '| Erfolg:', success, '| Repro ok:', repro_ok, '| Ausfaelle:', out['failed'])
for r in prim: print(f"  {r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
print('Sekundaer:', sv)
for r in sec['S1_kL2B_vs_kL2_fp32']: print(f"  H1 {r['dataset']:7s} d{r['depth']:3d} fp32 med={r['med_diff_pp']:+.2f}pp")
