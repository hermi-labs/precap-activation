# V2f Auswertung nach Praeregistrierung (V2f_Praeregistrierung.md), mit Claude. Aufruf: python v2f_analyze.py <V2f-Ordner, absolut> [<v2e_kal-Ordner>]
# Primaer (Kalibrierung mse4, 9 Paare, q_s = BPreCap / ctanh): H1 superpos_Q kleiner, H2 gbarQ kleiner, H3 rbar groesser; exakter Wilcoxon auf log q,
#   einseitig in Vorhersagerichtung, Holm ueber 3. "bestaetigt" bei p_Holm < 0,05 und >= 7/9 in Richtung; < 9 gueltige Paare: "unklar (Kontrolle)".
# Sekundaer: S1 dasselbe bei max (Holm ueber 3); S2 D_allQ und err_a4 bei mse4 und max kleiner (Holm ueber 4); S3 gepoolt mit v2e_kal (12 Paare,
#   beschreibend); S4 Spearman log q(sup) gegen log q(err_a4) bei mse4; Zerlegung log q(D) = log q(sqrt Sum d^2) + log q(sup).
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, spearmanr
ROOT = Path(sys.argv[1]).resolve(); S = ROOT.parent
KAL = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else S / 'v2e_kal'
SEEDS = tuple(range(42353, 42362)); CALS = ('mse4', 'max'); ALPHA = 0.05; MINDIR = 7
PRIM = (('H1_Kohaerenz', 'superpos_Q', 'less'), ('H2_Weitergabe', 'gbarQ', 'less'), ('H3_Rundung', 'rbar', 'greater'))
rows = list(csv.DictReader(open(ROOT / 'workers' / 'w0.csv'))) if (ROOT / 'workers' / 'w0.csv').exists() else []
D = {}
for r in rows:
    k = (r['act'], int(r['seed']), r['cal'])
    if k not in D or r['status'] == 'ok': D[k] = r
def finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False
def valid1(a, s, cal):
    r = D.get((a, s, cal)); return r is not None and r['status'] == 'ok' and r.get('kontrolle_n_diff', '') == '0' and all(finite(r.get(c)) for c in ('rbar', 'gbarQ', 'superpos_Q', 'D_allQ', 'err_a4_1000', 'sqrt_sum_dQ2'))
def pair_ok(s): return all(valid1(a, s, c) for a in ('bprecap', 'ctanh') for c in CALS)
def q(s, cal, col): return float(D[('bprecap', s, cal)][col]) / float(D[('ctanh', s, cal)][col])
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
def block(tests, cal, seeds):
    res = {}
    if len(seeds) < len(SEEDS):
        for name, col, alt in tests: res[name] = dict(kennzahl=col, kal=cal, n=len(seeds), urteil='unklar (Kontrolle)')
        return res
    ps = []
    for name, col, alt in tests:
        lq = np.log([q(s, cal, col) for s in seeds]); p1 = float(wilcoxon(lq, alternative=alt, method='exact').pvalue)
        p2 = float(wilcoxon(lq, alternative='two-sided', method='exact').pvalue); ndir = int((lq < 0).sum() if alt == 'less' else (lq > 0).sum())
        res[name] = dict(kennzahl=col, kal=cal, richtung='BPreCap kleiner' if alt == 'less' else 'BPreCap groesser', n=len(seeds), q=[round(float(math.exp(v)), 4) for v in lq],
                         q_median=round(float(math.exp(np.median(lq))), 4), in_richtung=ndir, p_einseitig=round(p1, 5), p_zweiseitig=round(p2, 5)); ps.append(p1)
    for (name, _, _), pa in zip(tests, holm(ps)):
        res[name]['p_holm'] = round(pa, 5); res[name]['urteil'] = 'bestaetigt' if pa < ALPHA and res[name]['in_richtung'] >= MINDIR else 'nicht bestaetigt'
    return res
good = [s for s in SEEDS if pair_ok(s)]
out = dict(n_rows=len(rows), soll=36, n_ok=sum(r['status'] == 'ok' for r in rows), n_bitgleich_laeufe=len({r['run_id'] for r in rows if r.get('kontrolle_n_diff', '') == '0'}),
           run_ids_eindeutig=len({(r['run_id'], r['cal']) for r in rows}) == len(rows), gueltige_paare=good)
out['Primaer_mse4'] = block(PRIM, 'mse4', good)
P = out['Primaer_mse4']; u = {k: P[k]['urteil'] for k in P}
if any(v.startswith('unklar') for v in u.values()): out['Deutung'] = 'unklar (Kontrolle)'
elif u['H1_Kohaerenz'] == 'bestaetigt' and u['H3_Rundung'] == 'bestaetigt': out['Deutung'] = 'Vorteil durch weniger gleichgerichtete Summierung, nicht durch feinere Rundung'
elif u['H1_Kohaerenz'] != 'bestaetigt': out['Deutung'] = 'Kohaerenz-Befund nicht bestaetigt (Hinweis im v0.4-Entwurf entfernen)'
else: out['Deutung'] = 'Kohaerenz bestaetigt, Rundung nicht: beschreibend'
out['S1_max'] = block(PRIM, 'max', good)
out['S2'] = block((('D_mse4', 'D_allQ', 'less'), ('a4_mse4', 'err_a4_1000', 'less')), 'mse4', good)
s2b = block((('D_max', 'D_allQ', 'less'), ('a4_max', 'err_a4_1000', 'less')), 'max', good)
if all('p_einseitig' in v for v in list(out['S2'].values()) + list(s2b.values())):   # Holm ueber alle 4
    allv = list(out['S2'].values()) + list(s2b.values()); adj = holm([v['p_einseitig'] for v in allv])
    for v, pa in zip(allv, adj): v['p_holm'] = round(pa, 5); v['urteil'] = 'bestaetigt' if pa < ALPHA and v['in_richtung'] >= MINDIR else 'nicht bestaetigt'
out['S2'].update(s2b)
# S3 gepoolt mit v2e_kal (beschreibend)
pool = {}
for cal in CALS:
    for col in ('superpos_Q', 'gbarQ', 'rbar', 'D_allQ', 'err_a4'):
        qs = []
        for s in (42350, 42351, 42352):
            fb, fc = KAL / 'mess' / f'bprecap_{s}_{cal}.json', KAL / 'mess' / f'ctanh_{s}_{cal}.json'
            if fb.exists() and fc.exists():
                b, c = json.load(open(fb)), json.load(open(fc)); qs.append(b[col] / c[col])
        qs += [q(s, cal, 'err_a4_1000' if col == 'err_a4' else col) for s in good]
        if qs: pool[f'{col}_{cal}'] = dict(n=len(qs), q_median=round(float(np.median(qs)), 4), bprecap_kleiner=int(sum(x < 1 for x in qs)))
out['S3_gepoolt'] = pool
if len(good) >= 3:
    rho = spearmanr(np.log([q(s, 'mse4', 'superpos_Q') for s in good]), np.log([q(s, 'mse4', 'err_a4_1000') for s in good]))
    out['S4_spearman_sup_a4_mse4'] = dict(rho=round(float(rho.statistic), 3), p=round(float(rho.pvalue), 4), n=len(good))
    out['Zerlegung'] = {cal: dict(logq_D=round(float(np.median([math.log(q(s, cal, 'D_allQ')) for s in good])), 4),
                                  logq_sqrtSd2=round(float(np.median([math.log(q(s, cal, 'sqrt_sum_dQ2')) for s in good])), 4),
                                  logq_sup=round(float(np.median([math.log(q(s, cal, 'superpos_Q')) for s in good])), 4)) for cal in CALS}
    med = {}
    for a in ('bprecap', 'ctanh'):
        for cal in CALS:
            med[f'{a}_{cal}'] = {c: round(float(np.median([float(D[(a, s, cal)][c]) for s in good])), 4) for c in ('rbar', 'gbarQ', 'superpos_Q', 'D_allQ', 'err_a4_1000', 'err_fp32_1000')}
    out['Mediane'] = med
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2f_result.json', 'w'), indent=1); print(json.dumps(out, indent=1))
