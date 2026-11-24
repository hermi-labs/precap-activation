# V2k Auswertung nach Praeregistrierung (V2k_Praeregistrierung.md), mit Claude. Aufruf: python v2k_analyze.py <V2k-Ordner, absolut>
# Primaer (Kalibrierung mse4, 9 Paare, q_s = BPreCap / ctanh): K1 D_allC kleiner (log q < 0); K2 log q(D_allC) - log q(D_allR) < 0.
#   Exakter Wilcoxon, einseitig, Holm ueber 2. "bestaetigt" bei p_Holm < 0,05 und >= 7/9 in Richtung; < 9 gueltige Paare: "unklar (Kontrolle)".
# Sekundaer (mse4, zweiseitig, Holm ueber 4): S1 log q(gbarR), S2 log q(gbarC), S3 kappa(BPreCap) - kappa(ctanh), S4 nu(BPreCap) - nu(ctanh).
#   Urteil: p_Holm < 0,05 -> "Unterschied (BPreCap kleiner/groesser)", sonst "kein klarer Unterschied".
# Beschreibend: dasselbe bei max ohne Korrektur, Mediane je Form, Profil von log q(g_i^Q) in 5 Bloecken, Kosinus-Bloecke, Zerlegung.
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]).resolve()
SEEDS = tuple(range(42353, 42362)); CALS = ('mse4', 'max'); FORMS = ('bprecap', 'ctanh'); ALPHA = 0.05; MINDIR = 7; TOL = 1e-3
BLOCKS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 49))
NUM = ('D_allQ', 'D_allC', 'D_allR', 'rbarQ', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'kappa', 'nu', 'superpos_Q', 'cos_nachbar_median', 'cos_mittel_innen', 'frac_clip_median')
rows = list(csv.DictReader(open(ROOT / 'workers' / 'w0.csv'))) if (ROOT / 'workers' / 'w0.csv').exists() else []
D = {}
for r in rows:
    k = (r['act'], int(r['seed']), r['cal'])
    if k not in D or r['status'] == 'ok': D[k] = r
def finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False
def valid1(a, s, cal):   # mse4: alle Kennzahlen endlich und D_allC, D_allR > 0; max: nur Status und Abgleich mit V2f (bei max kann Abschneiden fehlen)
    r = D.get((a, s, cal))
    base = (r is not None and r['status'] == 'ok' and finite(r.get('v2f_abw_dQ')) and float(r['v2f_abw_dQ']) < TOL and finite(r.get('v2f_abw_D')) and float(r['v2f_abw_D']) < TOL)
    if not base or cal != 'mse4': return base
    return all(finite(r.get(c)) for c in NUM) and float(r['D_allC']) > 0 and float(r['D_allR']) > 0
def pair_ok(s): return all(valid1(a, s, c) for a in FORMS for c in CALS)
def val(a, s, cal, col): return float(D[(a, s, cal)][col])
def lq(s, cal, col): return math.log(val('bprecap', s, cal, col) / val('ctanh', s, cal, col))
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
def x_of(name, s, cal):
    if name == 'K1': return lq(s, cal, 'D_allC')
    if name == 'K2': return lq(s, cal, 'D_allC') - lq(s, cal, 'D_allR')
    if name == 'S1': return lq(s, cal, 'gbarR')
    if name == 'S2': return lq(s, cal, 'gbarC')
    if name == 'S3': return val('bprecap', s, cal, 'kappa') - val('ctanh', s, cal, 'kappa')
    if name == 'S4': return val('bprecap', s, cal, 'nu') - val('ctanh', s, cal, 'nu')
TXT = dict(K1='D_allC (nur Abschneiden, alle Schichten), BPreCap kleiner', K2='log q(D_allC) - log q(D_allR) < 0 (Vorteil beim Abschneiden groesser als beim Runden)',
           S1='log q(gbarR), Weitergabe des reinen Rundungsfehlers', S2='log q(gbarC), Weitergabe des reinen Abschneidefehlers',
           S3='kappa BPreCap - ctanh (linearer Kreuzterm)', S4='nu BPreCap - ctanh (nichtlinearer Rest)')
def stats(name, cal, seeds, alt):
    x = np.array([x_of(name, s, cal) for s in seeds]); p = float(wilcoxon(x, alternative=alt, method='exact').pvalue)
    p2 = float(wilcoxon(x, alternative='two-sided', method='exact').pvalue)
    d = dict(test=TXT[name], kal=cal, n=len(seeds), werte=[round(float(v), 5) for v in x], median=round(float(np.median(x)), 5), negativ=int((x < 0).sum()), positiv=int((x > 0).sum()),
             p=round(p, 5), p_zweiseitig=round(p2, 5))
    if name in ('K1', 'S1', 'S2'): d['q_median'] = round(float(math.exp(np.median(x))), 4)
    return d
good = [s for s in SEEDS if pair_ok(s)]
out = dict(n_rows=len(rows), soll=36, n_ok=sum(r['status'] == 'ok' for r in rows), eindeutig=len({(r['run_id'], r['cal']) for r in rows}) == len(rows), gueltige_paare=good,
           v2f_abw_max=max([float(r['v2f_abw_dQ']) for r in rows if finite(r.get('v2f_abw_dQ'))] + [0.0]),
           v2f_abwD_max=max([float(r['v2f_abw_D']) for r in rows if finite(r.get('v2f_abw_D'))] + [0.0]))
if len(good) < len(SEEDS):
    out['Primaer'] = {k: dict(test=TXT[k], urteil='unklar (Kontrolle)', n=len(good)) for k in ('K1', 'K2')}
    out['Sekundaer'] = {k: dict(test=TXT[k], urteil='unklar (Kontrolle)', n=len(good)) for k in ('S1', 'S2', 'S3', 'S4')}
else:
    P = {k: stats(k, 'mse4', good, 'less') for k in ('K1', 'K2')}
    for k, pa in zip(P, holm([P[k]['p'] for k in P])):
        P[k]['p_holm'] = round(pa, 5); P[k]['urteil'] = 'bestaetigt' if pa < ALPHA and P[k]['negativ'] >= MINDIR else 'nicht bestaetigt'
    out['Primaer'] = P
    Sx = {k: stats(k, 'mse4', good, 'two-sided') for k in ('S1', 'S2', 'S3', 'S4')}
    for k, pa in zip(Sx, holm([Sx[k]['p'] for k in Sx])):
        Sx[k]['p_holm'] = round(pa, 5)
        Sx[k]['urteil'] = ('Unterschied (BPreCap kleiner)' if Sx[k]['median'] < 0 else 'Unterschied (BPreCap groesser)') if pa < ALPHA else 'kein klarer Unterschied'
    out['Sekundaer'] = Sx
    u = {k: P[k]['urteil'] for k in P}
    if u['K1'] == 'bestaetigt' and u['K2'] == 'bestaetigt': out['Deutung'] = 'Vorteil sitzt im Abschneiden: BPreCap verliert durch das Abschneiden weniger, und dort mehr als beim Runden'
    elif u['K1'] == 'bestaetigt': out['Deutung'] = 'BPreCap verliert durch das Abschneiden weniger, aber nicht klar mehr als beim Runden'
    elif u['K2'] == 'bestaetigt': out['Deutung'] = 'Vorteil relativ groesser beim Abschneiden, absolut beim Abschneiden aber nicht klar kleiner'
    else: out['Deutung'] = 'Abschneide-Deutung aus V2f nicht bestaetigt'
    def safe(f):
        try:
            v = f(); return v if np.all(np.isfinite(v.get('werte', [0]) if isinstance(v, dict) else v)) else 'nicht berechenbar (nicht endlich)'
        except (ValueError, ZeroDivisionError): return 'nicht berechenbar (kein Abschneiden oder Wert 0)'
    out['beschreibend_max'] = {k: safe(lambda k=k: stats(k, 'max', good, 'less' if k in ('K1', 'K2') else 'two-sided')) for k in ('K1', 'K2', 'S1', 'S2', 'S3', 'S4')}
    out['mediane'] = {f'{a}_{cal}': {c: round(float(np.nanmedian([val(a, s, cal, c) for s in good])), 5) for c in NUM} for a in FORMS for cal in CALS}
    def lqs(cal, c):
        v = []
        for s in good:
            try: v.append(lq(s, cal, c))
            except (ValueError, ZeroDivisionError): v.append(float('nan'))
        return np.array(v)
    out['logq_median'] = {cal: {c: (round(float(np.nanmedian(lqs(cal, c))), 4) if np.isfinite(lqs(cal, c)).any() else None) for c in ('D_allQ', 'D_allC', 'D_allR', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR')} for cal in CALS}
    out['in_richtung_kleiner'] = {cal: {c: int(np.sum(lqs(cal, c) < 0)) for c in ('D_allQ', 'D_allC', 'D_allR', 'gbarQ', 'gbarC', 'gbarR')} for cal in CALS}
    def anteil(a, s, cal):
        c2, r2 = val(a, s, cal, 'D_allC') ** 2, val(a, s, cal, 'D_allR') ** 2; return c2 / (c2 + r2) if c2 + r2 > 0 else float('nan')
    out['anteil_abschneiden_energie'] = {f'{a}_{cal}': round(float(np.nanmedian([anteil(a, s, cal) for s in good])), 4) for a in FORMS for cal in CALS}
    prof, cosb = {}, {}
    for cal in CALS:
        M = {(a, s): json.load(open(ROOT / 'mess' / f'confirm_{a}_lr3e-05_wd0_s{s}_{cal}.json')) for a in FORMS for s in good}
        def gq(a, s, i): return M[(a, s)]['dQ'][i] / M[(a, s)]['rQ'][i]
        prof[cal] = {f'{b0}-{b1}': dict(logq_gQ_median=round(float(np.median([np.mean([math.log(gq('bprecap', s, i) / gq('ctanh', s, i)) for i in range(b0, b1 + 1)]) for s in good])), 4),
                                         bprecap_kleiner=int(sum(np.mean([math.log(gq('bprecap', s, i) / gq('ctanh', s, i)) for i in range(b0, b1 + 1)]) < 0 for s in good)),
                                         logq_rC_median=round(float(np.nanmedian([np.mean([math.log(max(M[('bprecap', s)]['rC'][i], 1e-30) / max(M[('ctanh', s)]['rC'][i], 1e-30)) for i in range(b0, b1 + 1)]) for s in good])), 4))
                     for b0, b1 in BLOCKS}
        cosb[cal] = {a: [[round(float(np.median([np.mean([M[(a, s)]['cos'][i][j] for i in range(b0, b1 + 1) for j in range(c0, c1 + 1) if i != j]) for s in good])), 4)
                          for c0, c1 in BLOCKS] for b0, b1 in BLOCKS] for a in FORMS}
    out['profil_bloecke'] = prof; out['kosinus_bloecke'] = cosb
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2k_result.json', 'w'), indent=1)
for blk in ('Primaer', 'Sekundaer'):
    for k, r in out[blk].items(): print(blk, k, r.get('test'), '| Median', r.get('median'), '| q', r.get('q_median', '-'), '| neg', r.get('negativ'), '| p_holm', r.get('p_holm'), '->', r['urteil'])
print('Zeilen', out['n_rows'], 'ok', out['n_ok'], 'gueltige Paare', len(good), 'max Abw. V2f', out['v2f_abw_max'], out['v2f_abwD_max'], '|', out.get('Deutung', ''))
