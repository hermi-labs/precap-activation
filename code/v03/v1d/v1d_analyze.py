# V1d Auswertung nach Praeregistrierung (V1d_Praeregistrierung.md), Code aus v1c_analyze.py, mit Claude
# Aufruf: python v1d_analyze.py [V1d-Ordner] [V1c-Ordner] [B15-Ordner]   (Pfade werden absolut aufgeloest)
# V1d = V1c mit kuerzerem LR-Anlauf (69 = 1 %, 172 = 2,5 %), nur kL2B_tr 5e-5/0,3, CIFAR-10 d200. Gepaart ueber die Seeds 42192-42203 mit
# V1c (Anlauf 344 = 5 %) und B15 (kein Anlauf). Fruehes Stoppen wie V1c. Kontrolle: Seed 42192 cos und w344 in allen Mess- und Endwerten bitgleich zu V1c.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
V1C = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'v1c').resolve()
B15 = (Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / 'b15').resolve()
SEEDS = list(range(42192, 42204)); DEPTH = 200; ACT = 'kL2B_tr'; WARMS = (0, 69, 172, 344)
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
CKCOLS = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
ENDCOLS = ['test_err_w8a4a_max', 'w8a4a_best_cal', 'test_err_w8a4_best', 'w8a4_best_cal', 'sat_rate', 'act_lo_max_median']
MET = ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best')
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
def sel(rows, sched): return {int(r['seed']): r for r in rows if r['act'] == ACT and int(r['depth']) == DEPTH and r['sched'] == sched and int(r['seed']) in SEEDS
                              and float(r['lr']) == 5e-5 and float(r['wd']) == 0.3}
new = read(ROOT)
D = {0: sel(read(B15), 'cos') if (B15 / 'workers').exists() else {}, 344: sel(read(V1C), 'wcos') if (V1C / 'workers').exists() else {}}
for w in (69, 172): D[w] = {int(r['seed']): r for r in new if r['sched'] == 'wcos' and r['warm'] == str(w)}
REP = {('cos', int(r['seed'])): r for r in new if r['sched'] == 'cos'}
REP.update({('w344', int(r['seed'])): r for r in new if r['sched'] == 'wcos' and r['warm'] == '344'})
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def has(r, step):
    if step == 6880: return r['status'] == 'ok' and r.get('val_err_fp32') not in (None, '')
    return r.get(col(step, 'val_err_fp32')) not in (None, '')
def v(r, step, name): return float(r[col(step, name)]) * 100 if has(r, step) else 90.0   # Fehler in %, fehlend = 90 %
def es_step(r):
    av = [s for s in CK if has(r, s)]
    return min(av, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s)) if av else None
def es(name): return lambda r: (v(r, es_step(r), name) if es_step(r) else 90.0)
def val(w, s, f):   # fehlender Lauf (Ausfall ohne Zeile) zaehlt als 90 %
    r = D[w].get(s); return f(r) if r is not None else 90.0
def test(a, b, f):   # Anlauf a - Anlauf b, gepaart ueber den Seed
    x = np.array([val(a, s, f) - val(b, s, f) for s in SEEDS])
    p = 1.0 if np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand=f'w{a}', ref=f'w{b}', n=len(x), n_rows=(len(D[a]), len(D[b])), med_diff_pp=float(np.median(x)),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def verdict(r, pcol):
    return 'unklar' if r[pcol] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
W4A, FP = es('test_err_w8a4a_best'), es('test_err_fp32')
prim = holm({'P1_w69_vs_w344_w4a_es': test(69, 344, W4A), 'P2_w69_vs_w0_w4a_es': test(69, 0, W4A)})
p1, p2 = prim['P1_w69_vs_w344_w4a_es']['verdict'], prim['P2_w69_vs_w0_w4a_es']['verdict']
if p2 == 'schlechter': overall = 'auch 1 % Anlauf kostet bei 4 Bit'
elif p1 == 'besser': overall = 'kurzer Anlauf vermeidet den 4-Bit-Preis'
elif p1 == 'schlechter': overall = '1 % Anlauf bei 4 Bit schlechter als 5 %'
else: overall = 'nicht aufgeloest'
sec = holm({'S1_w172_vs_w344_w4a_es': test(172, 344, W4A), 'S2_w172_vs_w0_w4a_es': test(172, 0, W4A),
            'S3_w69_vs_w0_fp32_es': test(69, 0, FP), 'S4_w69_vs_w344_fp32_es': test(69, 344, FP), 'S5_w172_vs_w0_fp32_es': test(172, 0, FP)})
# Kontrolle: V1d cos und w344 Seed 42192 gegen V1c (Kontrollzeile bzw. Bestaetigung), alle Mess- und Endwerte
v1c_rows = read(V1C) if (V1C / 'workers').exists() else []
ref = {('cos', 42192): next((r for r in v1c_rows if r['run_id'] == 'cifar10_d200_kL2B_tr_s42192_kontrolle'), None),
       ('w344', 42192): next((r for r in v1c_rows if r['run_id'] == 'cifar10_d200_kL2B_tr_s42192'), None)}
repro = []
for k in sorted(ref):
    a, b = REP.get(k), ref[k]
    eq = a is not None and b is not None and all(a.get(col(s, c)) == b.get(col(s, c)) for s in CK for c in CKCOLS) and all(a.get(c) == b.get(c) for c in ENDCOLS)
    repro.append(dict(key=list(k), present=a is not None and b is not None, equal=bool(eq)))
repro_ok = len(repro) == 2 and all(x['equal'] for x in repro)
# Beschreibend: je Anlauflaenge Mediane am Stopp-Punkt, 4-Bit-Preis, Saettigung, Stopp-Punkte, Rangkorrelation je Seed (Anlauf -> w4a)
desc = {}
for w in WARMS:
    rs = [D[w][s] for s in SEEDS if s in D[w]]
    if not rs: continue
    e = dict(n=len(rs), n_ok=sum(r['status'] == 'ok' for r in rs), es_steps=sorted(es_step(r) or 0 for r in rs))
    e['es'] = {n: float(np.median([es(n)(r) for r in rs])) for n in MET}
    e['cost4a_pp'] = float(np.median([W4A(r) - FP(r) for r in rs]))
    sr = [float(r['sat_rate']) for r in rs if r.get('sat_rate') not in (None, '')]; e['sat_rate_median'] = float(np.median(sr)) if sr else None
    desc[f'w{w}'] = e
def spearman(xs, ys):
    rx = np.argsort(np.argsort(xs)).astype(float); ry = np.argsort(np.argsort(ys)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(ry) > 0 else 0.0
rho = [spearman(np.array(WARMS, float), np.array([val(w, s, W4A) for w in WARMS])) for s in SEEDS]
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, primary_verdicts={'P1': p1, 'P2': p2}, overall=overall,
           secondary=sec, secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro=repro, repro_ok=repro_ok, descriptive=desc,
           spearman_warm_w4a=dict(per_seed=[round(x, 3) for x in rho], median=float(np.median(rho))))
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v1d_result.json', 'w'), indent=1)
for k, r in prim.items(): print(f"{k}: med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} p={r['p']:.4f} holm={r['holm']:.4f} {r['verdict']} (Zeilen {r['n_rows']})")
print('Gesamt:', overall, '| Kontrolle V1c:', repro_ok, repro, '| Ausfaelle:', out['failed'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
for w, e in desc.items(): print(w, 'Stopp', e['es_steps'], '| fp32', round(e['es']['test_err_fp32'], 2), 'w4a', round(e['es']['test_err_w8a4a_best'], 2), 'Preis', round(e['cost4a_pp'], 2), 'sat', e['sat_rate_median'])
print('Spearman Anlauf -> w4a je Seed, Median:', round(out['spearman_warm_w4a']['median'], 3))
