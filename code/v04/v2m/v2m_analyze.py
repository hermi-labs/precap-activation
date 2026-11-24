# V2m Auswertung nach Praeregistrierung (V2m_Praeregistrierung.md), mit Claude. Aufruf: python v2m_analyze.py <V2m-Ordner, absolut>
# Paare je Seed (42353-42361): Variante rho 0,4 (V2m) gegen BPreCap rho 0,8 (V2k-Messung der V2f-Netze bzw. V2a2-Zeile) und gegen ctanh (V2k).
# Primaer (mse4, exakter Wilcoxon zweiseitig, Holm ueber 2): E1 log(gbarC_v / gbarC_b), E2 log(D_allC_v / D_allC_b).
#   "bestaetigt": p_Holm < 0,05, Median > 0, >= 7 Paare positiv; "widerlegt": p_Holm < 0,05, Median < 0, >= 7 negativ; sonst "unklar".
#   Weniger als 7 gueltige Paare: "unklar (zu wenige Paare)".
# Sekundaer (zweiseitig, Holm ueber 4): S1 4-Bit-Preis (w8a4a_best - fp32) v - b; S2 w8a4a_best v - b; S3 fp32 v - b; S4 log(gbarC_v / gbarC_ctanh).
#   Ausfall der Variante: S2, S3 mit Fehler 0,9; S1, S4 ohne dieses Paar.
# Beschreibend: Mediane je Form, Lage der Variante zwischen BPreCap und ctanh (Anteil t des log-Abstands), weitere log-Quotienten.
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]).resolve(); S = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT.parent
SEEDS = tuple(range(42353, 42362)); ALPHA = 0.05; MINDIR = 7; MINN = 7; FAIL = 0.9; CAL = 'mse4'
NUM = ('D_allQ', 'D_allC', 'D_allR', 'rbarQ', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'kappa', 'nu', 'superpos_Q', 'frac_clip_median', 'cos_nachbar_median')
def finite(x):
    try: return math.isfinite(float(x))
    except (TypeError, ValueError): return False
rows = list(csv.DictReader(open(ROOT / 'workers' / 'w0.csv'))) if (ROOT / 'workers' / 'w0.csv').exists() else []
V = {}
for r in rows:
    s = int(r['seed'])
    if s not in V or r['status'] == 'ok': V[s] = r
Kr = {(r['act'], int(r['seed'])): r for r in csv.DictReader(open(S / 'v2k' / 'workers' / 'w0.csv')) if r['cal'] == CAL and r['status'] == 'ok'}
A2 = {int(r['seed']): r for r in csv.DictReader(open(S / 'v2a2' / 'workers' / 'w0.csv')) if r['act'] == 'bprecap' and r['stage'] == 'confirm' and r['status'] == 'ok'}
def v_ok(s):
    r = V.get(s); return r is not None and r['status'] == 'ok' and all(finite(r.get(c)) for c in NUM) and float(r['D_allC']) > 0 and float(r['gbarC']) > 0
def ref_ok(s): return ('bprecap', s) in Kr and ('ctanh', s) in Kr and s in A2
good = [s for s in SEEDS if v_ok(s) and ref_ok(s)]
def f(r, c): return float(r[c])
def err_v(s, c):
    r = V.get(s); return f(r, c) if (r is not None and r['status'] == 'ok' and finite(r.get(c))) else FAIL
def x_of(name, s):
    if name == 'E1': return math.log(f(V[s], 'gbarC') / f(Kr[('bprecap', s)], 'gbarC'))
    if name == 'E2': return math.log(f(V[s], 'D_allC') / f(Kr[('bprecap', s)], 'D_allC'))
    if name == 'S1': return (f(V[s], 'es_test_err_w8a4a_best') - f(V[s], 'es_test_err_fp32')) - (f(A2[s], 'es_test_err_w8a4a_best') - f(A2[s], 'es_test_err_fp32'))
    if name == 'S2': return err_v(s, 'es_test_err_w8a4a_best') - f(A2[s], 'es_test_err_w8a4a_best')
    if name == 'S3': return err_v(s, 'es_test_err_fp32') - f(A2[s], 'es_test_err_fp32')
    if name == 'S4': return math.log(f(V[s], 'gbarC') / f(Kr[('ctanh', s)], 'gbarC'))
TXT = dict(E1='log(gbarC rho0,4 / gbarC rho0,8): Weitergabe des Abschneidefehlers', E2='log(D_allC rho0,4 / D_allC rho0,8): Abschneiden in allen Schichten an den Logits',
           S1='4-Bit-Preis (w8a4a_best - fp32), rho0,4 - rho0,8', S2='Testfehler w8a4a_best, rho0,4 - rho0,8', S3='Testfehler fp32, rho0,4 - rho0,8',
           S4='log(gbarC rho0,4 / gbarC ctanh)')
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
def stats(name, seeds):
    x = np.array([x_of(name, s) for s in seeds]); p = float(wilcoxon(x, alternative='two-sided', method='exact').pvalue)
    d = dict(test=TXT[name], n=len(seeds), seeds=list(seeds), werte=[round(float(v), 5) for v in x], median=round(float(np.median(x)), 5),
             positiv=int((x > 0).sum()), negativ=int((x < 0).sum()), p=round(p, 5))
    if name in ('E1', 'E2', 'S4'): d['q_median'] = round(float(math.exp(np.median(x))), 4)
    if name in ('S1', 'S2', 'S3'): d['median_pp'] = round(100 * float(np.median(x)), 2)
    return d
sec_seeds = {'S1': good, 'S4': good, 'S2': [s for s in SEEDS if ref_ok(s)], 'S3': [s for s in SEEDS if ref_ok(s)]}
out = dict(n_rows=len(rows), soll=9, n_ok=sum(r['status'] == 'ok' for r in rows), eindeutig=len({r['run_id'] for r in rows}) == len(rows),
           gueltige_paare=good, ausfaelle=[s for s in SEEDS if not v_ok(s)], referenz_vollstaendig=all(ref_ok(s) for s in SEEDS))
if len(good) < MINN:
    out['Primaer'] = {k: dict(test=TXT[k], urteil='unklar (zu wenige Paare)', n=len(good)) for k in ('E1', 'E2')}
else:
    P = {k: stats(k, good) for k in ('E1', 'E2')}
    for k, pa in zip(P, holm([P[k]['p'] for k in P])):
        P[k]['p_holm'] = round(pa, 5)
        if pa < ALPHA and P[k]['median'] > 0 and P[k]['positiv'] >= MINDIR: P[k]['urteil'] = 'bestaetigt'
        elif pa < ALPHA and P[k]['median'] < 0 and P[k]['negativ'] >= MINDIR: P[k]['urteil'] = 'widerlegt'
        else: P[k]['urteil'] = 'unklar'
    out['Primaer'] = P
Sx = {}
for k in ('S1', 'S2', 'S3', 'S4'):
    Sx[k] = stats(k, sec_seeds[k]) if len(sec_seeds[k]) >= MINN else dict(test=TXT[k], n=len(sec_seeds[k]), p=1.0, median=float('nan'))
for k, pa in zip(Sx, holm([Sx[k]['p'] for k in Sx])):
    Sx[k]['p_holm'] = round(pa, 5); Sx[k]['urteil'] = (('Unterschied (rho0,4 groesser)' if Sx[k]['median'] > 0 else 'Unterschied (rho0,4 kleiner)') if pa < ALPHA else 'kein klarer Unterschied')
out['Sekundaer'] = Sx
u = tuple(out['Primaer'][k]['urteil'] for k in ('E1', 'E2'))
if u == ('bestaetigt', 'bestaetigt'): out['Deutung'] = 'Fruehe Saettigung verstaerkt die Wirkung des Abschneidens: die spaete Saettigung der Huelle traegt den Vorteil beim Abschneiden mit'
elif 'widerlegt' in u: out['Deutung'] = 'Fruehe Saettigung schwaecht die Wirkung des Abschneidens: die Lesart "spaete Saettigung hilft" ist fuer die Huelle falsch'
elif 'bestaetigt' in u: out['Deutung'] = 'Teilweise: nur ' + ' und '.join(k for k, v in zip(('E1', 'E2'), u) if v == 'bestaetigt') + ' bestaetigt'
else: out['Deutung'] = 'Das Knie der Huelle ist nicht klar der Hebel; der Vorteil sitzt eher im Kern oder in der Verteilung nach dem Training'
if good:
    out['mediane'] = {'rho04': {c: round(float(np.median([f(V[s], c) for s in good])), 5) for c in NUM},
                      'rho08': {c: round(float(np.median([f(Kr[('bprecap', s)], c) for s in good])), 5) for c in NUM},
                      'ctanh': {c: round(float(np.median([f(Kr[('ctanh', s)], c) for s in good])), 5) for c in NUM}}
    out['fehler_mediane_pp'] = {'rho04': {c: round(100 * float(np.median([f(V[s], c) for s in good])), 2) for c in ('es_test_err_fp32', 'es_test_err_w8a4a_best', 'es_test_err_w8a4a_mse4')},
                                'rho08': {c: round(100 * float(np.median([f(A2[s], c) for s in good])), 2) for c in ('es_test_err_fp32', 'es_test_err_w8a4a_best', 'es_test_err_w8a4a_mse4')}}
    def lage(c):
        t = []
        for s in good:
            a, b, v = (math.log(f(Kr[('bprecap', s)], c)), math.log(f(Kr[('ctanh', s)], c)), math.log(f(V[s], c)))
            t.append((v - a) / (b - a) if b != a else float('nan'))
        return round(float(np.nanmedian(t)), 3)
    out['lage_t_median'] = {c: lage(c) for c in ('gbarC', 'D_allC', 'gbarQ', 'D_allQ')}
    out['logq_v_gegen_rho08_median'] = {c: round(float(np.median([math.log(f(V[s], c) / f(Kr[('bprecap', s)], c)) for s in good])), 4)
                                        for c in ('D_allQ', 'D_allC', 'D_allR', 'rbarC', 'rbarR', 'gbarQ', 'gbarC', 'gbarR', 'frac_clip_median')}
    out['delta_v_gegen_rho08_median'] = {c: round(float(np.median([f(V[s], c) - f(Kr[('bprecap', s)], c) for s in good])), 4) for c in ('kappa', 'nu', 'superpos_Q')}
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2m_result.json', 'w'), indent=1)
for blk in ('Primaer', 'Sekundaer'):
    for k, r in out[blk].items(): print(blk, k, r.get('test'), '| Median', r.get('median'), '| q', r.get('q_median', '-'), '| +/-', r.get('positiv'), r.get('negativ'), '| p_holm', r.get('p_holm'), '->', r['urteil'])
print('Zeilen', out['n_rows'], 'ok', out['n_ok'], 'gueltige Paare', len(good), '|', out['Deutung'])
