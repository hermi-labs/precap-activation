# V2e Auswertung nach Praeregistrierung (V2e_Praeregistrierung.md), mit Claude. Aufruf: python v2e_analyze.py <V2e-Ordner, absolut> [<V2a2> <V2c> <V2d>]
# Primaer: Pa r_bar, Pb g_bar^Q, BPreCap gegen ctanh, q_s = BPreCap / ctanh je Seed; alle 3 < 1/1,1 "BPreCap kleiner", alle 3 > 1,1 "BPreCap groesser",
#   sonst "kein klarer Unterschied"; fehlt ein gueltiges Paar (Ausfall oder kontrolle_n_diff != 0): "unklar (Kontrolle)".
# Pruefung der Messung: D^Q(BPreCap) < D^Q(ctanh) bei mindestens 2 von 3 Seeds.
# Sekundaer (gleiche Regel, Hinweise): S1 g_bar^G BPreCap/ctanh; S2 BPreCap/SELU fuer r_bar und g_bar^Q; S3 g^Q/g^G je Form; S4 Spearman ueber 5 Formen
#   (Mediane D^Q, r_bar, g_bar^Q gegen den 4-Bit-Preis aus V2a2/V2c/V2d ueber 12 Seeds); Zerlegung der log-Verhaeltnisse.
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(sys.argv[1]).resolve(); S = ROOT.parent
HD = {'v2a2': Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else S / 'v2a2', 'v2c': Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else S / 'v2c',
      'v2d': Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else S / 'v2d'}
HIST = {'bprecap': 'v2a2', 'selu': 'v2a2', 'tat': 'v2a2', 'tanh': 'v2c', 'ctanh': 'v2d'}
SEEDS = (42350, 42351, 42352); FORMS = ('bprecap', 'ctanh', 'selu', 'tanh', 'tat'); THR = 1.1
rows = list(csv.DictReader(open(ROOT / 'workers' / 'w0.csv'))) if (ROOT / 'workers' / 'w0.csv').exists() else []
D = {}
for r in rows:
    k = (r['act'], int(r['seed']))
    if k not in D or r['status'] == 'ok': D[k] = r
def valid(a, s):
    r = D.get((a, s)); return r is not None and r['status'] == 'ok' and r.get('kontrolle_n_diff', '') == '0' and r.get('rbar', '') != ''
def f(a, s, col): return float(D[(a, s)][col])
def rule(a, b, col):
    if not all(valid(a, s) and valid(b, s) for s in SEEDS): return dict(q=None, urteil='unklar (Kontrolle)')
    q = [f(a, s, col) / f(b, s, col) for s in SEEDS]
    u = 'BPreCap kleiner' if all(x < 1 / THR for x in q) else 'BPreCap groesser' if all(x > THR for x in q) else 'kein klarer Unterschied'
    if a != 'bprecap': u = u.replace('BPreCap', a)
    return dict(q=[round(x, 4) for x in q], urteil=u)
out = dict(n_rows=len(rows), n_ok=sum(r['status'] == 'ok' for r in rows), n_bitgleich=sum(r.get('kontrolle_n_diff', '') == '0' for r in rows),
           gueltig={f'{a}_{s}': valid(a, s) for a in FORMS for s in SEEDS})
out['Pa'] = rule('bprecap', 'ctanh', 'rbar'); out['Pb'] = rule('bprecap', 'ctanh', 'gbarQ')
pm = [f('bprecap', s, 'D_allQ') < f('ctanh', s, 'D_allQ') for s in SEEDS if valid('bprecap', s) and valid('ctanh', s)]
out['Pruefung_Messung'] = dict(bprecap_kleiner=sum(pm), von=len(pm), erfuellt=sum(pm) >= 2)
tab = {('kleiner', 'kein klarer'): 'Runden', ('kein klarer', 'kleiner'): 'Weitergabe', ('kleiner', 'kleiner'): 'beides'}
key = lambda u: 'kleiner' if u.endswith('kleiner') else 'kein klarer' if u.startswith('kein') else u
out['Deutung'] = tab.get((key(out['Pa']['urteil']), key(out['Pb']['urteil'])), 'keine der beiden Groessen allein (beschreibend)')
if not out['Pruefung_Messung']['erfuellt']: out['Deutung'] += ' - nur beschreibend, Pruefung der Messung nicht erfuellt'
out['S1_gbarG'] = rule('bprecap', 'ctanh', 'gbarG'); out['S2_rbar_selu'] = rule('bprecap', 'selu', 'rbar'); out['S2_gbarQ_selu'] = rule('bprecap', 'selu', 'gbarQ')
COLS = ('rbar', 'gbarQ', 'gbarG', 'D_allQ', 'D_allG', 'superpos_Q', 'superpos_G', 'flipQ', 'err_fp32_1000', 'err_a4_1000', 'err_w8a4_1000', 'chi_emp_gm', 'eps_end_Q',
        'margin_median', 'bias_anteil_median')
med = {a: {c: float(np.median([f(a, s, c) for s in SEEDS if valid(a, s)])) if any(valid(a, s) for s in SEEDS) else None for c in COLS} for a in FORMS}
for a in FORMS:
    if med[a]['gbarQ'] is not None: med[a]['S3_gQ_durch_gG'] = med[a]['gbarQ'] / med[a]['gbarG']
out['Mediane'] = med
def price(a):
    p = []
    for r in csv.DictReader(open(HD[HIST[a]] / 'workers' / 'w0.csv')):
        if r['act'] == a and r['stage'] == 'confirm' and r['status'] == 'ok': p.append(100 * (float(r['es_test_err_w8a4a_best']) - float(r['es_test_err_fp32'])))
    return float(np.median(p)) if p else None
out['Preis_12_Seeds'] = {a: price(a) for a in FORMS}
fs = [a for a in FORMS if med[a]['rbar'] is not None and out['Preis_12_Seeds'][a] is not None]
if len(fs) >= 3:
    out['S4_spearman'] = {c: round(float(spearmanr([med[a][c] for a in fs], [out['Preis_12_Seeds'][a] for a in fs])[0]), 3) for c in ('D_allQ', 'rbar', 'gbarQ', 'gbarG')}
    out['S4_formen'] = fs
if all(valid('bprecap', s) and valid('ctanh', s) for s in SEEDS):
    z = []
    for s in SEEDS:
        lD, lr, lg = (math.log(f('bprecap', s, c) / f('ctanh', s, c)) for c in ('D_allQ', 'rbar', 'gbarQ')); z.append(dict(seed=s, logD=round(lD, 4), logr=round(lr, 4), logg=round(lg, 4), rest=round(lD - lr - lg, 4)))
    out['Zerlegung'] = z
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2e_result.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('n_rows', 'n_ok', 'n_bitgleich', 'Pa', 'Pb', 'Pruefung_Messung', 'Deutung', 'S1_gbarG', 'S2_rbar_selu', 'S2_gbarQ_selu')}, indent=1))
