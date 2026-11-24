# V2g Auswertung nach Praeregistrierung (V2gb_Praeregistrierung.md), mit Claude. Aufruf: python v2gb_analyze.py <V2g-Ordner, absolut> [<V2a2> <V2c> <V2d>]
# xtanh-Zeilen aus <V2g>/workers, BPreCap/SELU/TAT historisch aus <V2a2>/workers, tanh aus <V2c>/workers, ctanh aus <V2d>/workers (Standard: Nachbarordner).
# Primaer (Holm ueber 2): P1 BPreCap - xtanh, Test-Fehler w8a4a_best am fruehen Stopp (pp); P2 BPreCap - xtanh, 4-Bit-Preis
#   (w8a4a_best - fp32, pp). 12 Seeds gepaart, Wilcoxon zweiseitig. Holm-p < 0,05 und Median < 0 "BPreCap besser", > 0 "xtanh besser", sonst "unklar".
# Sekundaer (Holm ueber 4): S1 BPreCap - xtanh fp32; S2 xtanh - tanh w8a4a_best; S3 xtanh - ctanh w8a4a_best; S4 xtanh - SELU w8a4a_best.
#   Urteil "besser"/"schlechter"/"unklar" aus Sicht des erstgenannten Arms. Ausfall oder fehlende Zeile = 90 %.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]).resolve()
V2A2 = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT.parent / 'v2a2'; V2C = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else ROOT.parent / 'v2c'
V2D = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else ROOT.parent / 'v2d'
SEEDS = list(range(42350, 42362)); ACTS = ('bprecap', 'tat', 'selu', 'tanh', 'ctanh', 'xtanh'); HIST = ('bprecap', 'tat', 'selu', 'tanh', 'ctanh')
def read(d):
    r = []
    for f in sorted(d.glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
rows = read(ROOT / 'workers'); tune = read(ROOT / 'tune' / 'workers') if (ROOT / 'tune' / 'workers').exists() else []
D = {}
for r in [x for x in read(V2A2 / 'workers') if x['act'] in ('bprecap', 'tat', 'selu')] + [x for x in read(V2C / 'workers') if x['act'] == 'tanh'] + [x for x in read(V2D / 'workers') if x['act'] == 'ctanh'] + [x for x in rows if x['act'] == 'xtanh']:
    k = (r['act'], int(r['seed']))
    if k not in D or r['status'] == 'ok': D[k] = r
def v(r, col):
    if r is None or r['status'] != 'ok' or r.get(col, '') == '': return 90.0
    x = float(r[col]); return x * 100 if np.isfinite(x) else 90.0
def val(r, col):
    if col == 'preis': return v(r, 'es_test_err_w8a4a_best') - v(r, 'es_test_err_fp32') if r is not None and r['status'] == 'ok' else 90.0
    return v(r, col)
def test(a, b, col):
    x = np.array([val(D.get((a, s)), col) - val(D.get((b, s)), col) for s in SEEDS])
    p = 1.0 if np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(a=a, b=b, col=col, n=len(x), med_diff_pp=float(np.median(x)), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=[round(float(y), 2) for y in x])
def verdict(p, med): return 'unklar' if p >= 0.05 else ('besser' if med < 0 else ('schlechter' if med > 0 else 'unklar'))
def holm(res):
    ks = list(res); o = sorted(ks, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(o): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in ks: res[k]['verdict'] = verdict(res[k]['holm'], res[k]['med_diff_pp'])
    return res
ct = [r for r in rows if r['act'] == 'xtanh']; n_ok = sum(r['status'] == 'ok' for r in ct)
out = dict(n_rows=len(ct), n_ok=n_ok, n_fail=len(ct) - n_ok, soll=12, run_id_eindeutig=len({r['run_id'] for r in rows}) == len(rows),
           missing=[f'{a}_{s}' for a in ACTS for s in SEEDS if (a, s) not in D], n_tune_rows=len(tune), n_tune_ok=sum(r['status'] == 'ok' for r in tune),
           n_hist_ok=sum(1 for a in HIST for s in SEEDS if (a, s) in D and D[(a, s)]['status'] == 'ok'),
           nicht_endlich=sum(1 for r in ct if r['status'] == 'ok' for c in r if c.startswith(('es_test', 'end_test')) and r[c] != '' and not np.isfinite(float(r[c]))))
out['primary'] = holm(dict(P1=test('bprecap', 'xtanh', 'es_test_err_w8a4a_best'), P2=test('bprecap', 'xtanh', 'preis')))
for r_ in out['primary'].values(): r_['verdict'] = {'besser': 'BPreCap besser', 'schlechter': 'xtanh besser', 'unklar': 'unklar'}[r_['verdict']]
out['secondary'] = holm(dict(S1=test('bprecap', 'xtanh', 'es_test_err_fp32'), S2=test('xtanh', 'tanh', 'es_test_err_w8a4a_best'),
                             S3=test('xtanh', 'ctanh', 'es_test_err_w8a4a_best'), S4=test('xtanh', 'selu', 'es_test_err_w8a4a_best')))
out['descr_end'] = {k: {q: test(a, b, 'end_test_err_' + c)[q] for q in ('med_diff_pp', 'wins', 'losses')} for k, (a, b, c) in
                    dict(bprecap_xtanh_w8a4a_best=('bprecap', 'xtanh', 'w8a4a_best'), bprecap_xtanh_fp32=('bprecap', 'xtanh', 'fp32')).items()}
out['descr_es_w8a4a_max'] = {q: test('bprecap', 'xtanh', 'es_test_err_w8a4a_max')[q] for q in ('med_diff_pp', 'wins', 'losses', 'p')}
desc = {}
for a in ACTS:
    ok_ = [D[(a, s)] for s in SEEDS if (a, s) in D and D[(a, s)]['status'] == 'ok']
    cols = ['es_test_err_fp32', 'es_test_err_w8a8', 'es_test_err_w8a4a_max', 'es_test_err_w8a4a_best', 'end_test_err_fp32', 'end_test_err_w8a4a_best', 'val_err_end', 'val_err_es', 'es_step',
            'es_act_lo_max_median', 'es_act_hi_max_median', 'es_act_absmax_max', 'train_loss_last50']
    d = {c: (float(np.median([float(r[c]) for r in ok_])) if ok_ and all(r.get(c, '') != '' for r in ok_) else None) for c in cols}
    d['best_cal'] = {}
    for r in ok_: d['best_cal'][r['es_w8a4a_best_cal']] = d['best_cal'].get(r['es_w8a4a_best_cal'], 0) + 1
    pr = [v(r, 'es_test_err_w8a4a_best') - v(r, 'es_test_err_fp32') for r in ok_]; d['preis_4bit_pp_median'] = float(np.median(pr)) if pr else None
    d['lr_wd'] = sorted({(r['lr'], r['wd']) for r in ok_}); d['n_ok'] = len(ok_); desc[a] = d
out['descriptive'] = desc
sp = ROOT / 'tune' / 'hp_sel.json'; out['hp_sel'] = json.load(open(sp)) if sp.exists() else None
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2gb_result.json', 'w'), indent=1)
for k, r in out['primary'].items(): print(f"{k} BPreCap - xtanh ({r['col']}): {r['med_diff_pp']:+.2f} pp, {r['wins']}/{r['losses']}, p = {r['p']:.4f}, Holm-p = {r['holm']:.4f} -> {r['verdict']}")
for k, r in out['secondary'].items(): print(f"{k} {r['a']} - {r['b']} ({r['col']}): {r['med_diff_pp']:+.2f} pp, {r['wins']}/{r['losses']}, Holm-p = {r['holm']:.4f} -> {r['verdict']}")
print('xtanh Zeilen', out['n_rows'], 'ok', out['n_ok'], 'Ausfall', out['n_fail'], 'fehlend', len(out['missing']), 'historisch ok', out['n_hist_ok'], '/ 60', 'Tuning', out['n_tune_ok'], '/', out['n_tune_rows'])
