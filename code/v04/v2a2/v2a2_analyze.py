# V2a2 Auswertung nach Praeregistrierung (V2a2_Praeregistrierung.md), mit Claude. Aufruf: python v2a2_analyze.py <V2a2-Ordner, absolut>
# Primaer (zwei Tests, Holm ueber 2): P1 BPreCap - SELU und P2 BPreCap - TAT, Test-Fehler w8a4a_best am Punkt des fruehen Stoppens (pp),
#   12 Seeds gepaart, Wilcoxon zweiseitig. Urteil je Test: Holm-p < 0,05 und Median < 0 "besser", > 0 "schlechter", sonst "unklar".
# Sekundaer (Holm ueber 5, fruehes Stoppen): S1 fp32 gegen SELU, S2 fp32 gegen TAT, S3 w8a4a_max gegen SELU, S4 w8a4a_max gegen TAT,
#   S5 w8a4a_best gegen TAT nach Cross-Layer Equalization (TAT-Spalte es_cle_test_err_w8a4a_best).
# Ausfall oder fehlende Zeile = 90 %.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]).resolve()
SEEDS = list(range(42350, 42362)); CAND = 'bprecap'; REFS = ('selu', 'tat'); ACTS = ('bprecap', 'tat', 'selu')
def read(d):
    r = []
    for f in sorted(d.glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
rows = read(ROOT / 'workers'); tune = read(ROOT / 'tune' / 'workers') if (ROOT / 'tune' / 'workers').exists() else []
D = {}
for r in rows:
    k = (r['act'], int(r['seed']))
    if k not in D or r['status'] == 'ok': D[k] = r
def v(r, col):
    if r is None or r['status'] != 'ok' or r.get(col, '') == '': return 90.0
    x = float(r[col]); return x * 100 if np.isfinite(x) else 90.0
def test(ref, col, col_ref=None):
    x = np.array([v(D.get((CAND, s)), col) - v(D.get((ref, s)), col_ref or col) for s in SEEDS])
    p = 1.0 if np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(ref=ref, col=col, col_ref=col_ref or col, n=len(x), med_diff_pp=float(np.median(x)), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=[round(float(a), 2) for a in x])
def holm(res):
    ks = list(res); o = sorted(ks, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(o): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in ks:
        r = res[k]; r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
n_ok = sum(r['status'] == 'ok' for r in rows)
out = dict(n_rows=len(rows), n_ok=n_ok, n_fail=len(rows) - n_ok, soll=36, run_id_eindeutig=len({r['run_id'] for r in rows}) == len(rows),
           missing=[f'{a}_{s}' for a in ACTS for s in SEEDS if (a, s) not in D], n_tune_rows=len(tune), n_tune_ok=sum(r['status'] == 'ok' for r in tune),
           nicht_endlich=sum(1 for r in rows if r['status'] == 'ok' for c in r if c.startswith(('es_test', 'end_test')) and r[c] != '' and not np.isfinite(float(r[c]))))
out['primary'] = holm(dict(P1=test('selu', 'es_test_err_w8a4a_best'), P2=test('tat', 'es_test_err_w8a4a_best')))
v1, v2 = out['primary']['P1']['verdict'], out['primary']['P2']['verdict']
if v1 == 'besser' and v2 == 'besser': summ = 'Erfolg: besser als SELU und als TAT'
elif 'schlechter' in (v1, v2): summ = 'mindestens ein Vergleich schlechter: ' + ', '.join(f'{k} {out["primary"][k]["verdict"]}' for k in ('P1', 'P2'))
elif 'besser' in (v1, v2): summ = 'teilweise: besser als ' + ('SELU' if v1 == 'besser' else 'TAT') + ', gegen ' + ('TAT' if v1 == 'besser' else 'SELU') + ' unklar'
else: summ = 'unklar gegen beide'
out['summary'] = summ
out['secondary'] = holm(dict(S1=test('selu', 'es_test_err_fp32'), S2=test('tat', 'es_test_err_fp32'), S3=test('selu', 'es_test_err_w8a4a_max'), S4=test('tat', 'es_test_err_w8a4a_max'),
                             S5=test('tat', 'es_test_err_w8a4a_best', 'es_cle_test_err_w8a4a_best')))
# beschreibend (keine Tests): Endpunkt-Vergleiche, Mediane je Form, 4-Bit-Preis, Kalibrierwahl, Auswahl im Tuning
out['descr_end'] = {f'{ref}_{c}': {k: test(ref, 'end_test_err_' + c)[k] for k in ('med_diff_pp', 'wins', 'losses')} for ref in REFS for c in ('fp32', 'w8a4a_best', 'w8a4a_max')}
desc = {}
for a in ACTS:
    ok_ = [D[(a, s)] for s in SEEDS if (a, s) in D and D[(a, s)]['status'] == 'ok']
    cols = [c for c in (ok_[0].keys() if ok_ else []) if c.startswith(('es_', 'end_', 'val_err', 'train_loss')) and not c.endswith('best_cal')] + ['es_step']
    med = lambda c: float(np.median([float(r[c]) for r in ok_])) if ok_ and all(r.get(c, '') != '' for r in ok_) else None
    d = {c: med(c) for c in dict.fromkeys(cols)}
    for pre in ('es_', 'end_'):
        d[pre + 'best_cal'] = {}
        for r in ok_: d[pre + 'best_cal'][r[pre + 'w8a4a_best_cal']] = d[pre + 'best_cal'].get(r[pre + 'w8a4a_best_cal'], 0) + 1
        pr = [v(r, pre + 'test_err_w8a4a_best') - v(r, pre + 'test_err_fp32') for r in ok_]; d[pre + 'preis_4bit_pp_median'] = float(np.median(pr)) if pr else None
    d['lr_wd'] = sorted({(r['lr'], r['wd']) for r in ok_}); d['n_ok'] = len(ok_); desc[a] = d
out['descriptive'] = desc
sp = ROOT / 'tune' / 'hp_sel.json'; out['hp_sel'] = json.load(open(sp)) if sp.exists() else None
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2a2_result.json', 'w'), indent=1)
for k, r in out['primary'].items(): print(f"{k} BPreCap - {r['ref']} ({r['col']}): {r['med_diff_pp']:+.2f} pp, {r['wins']}/{r['losses']}, p = {r['p']:.4f}, Holm-p = {r['holm']:.4f} -> {r['verdict']}")
print('Gesamt:', summ)
for k, r in out['secondary'].items(): print(f"{k} BPreCap - {r['ref']} ({r['col']} / {r['col_ref']}): {r['med_diff_pp']:+.2f} pp, {r['wins']}/{r['losses']}, Holm-p = {r['holm']:.4f} -> {r['verdict']}")
print('Zeilen', out['n_rows'], 'ok', out['n_ok'], 'Ausfall', out['n_fail'], 'fehlend', len(out['missing']), 'run_id eindeutig', out['run_id_eindeutig'], 'Tuning', out['n_tune_ok'], '/', out['n_tune_rows'])
