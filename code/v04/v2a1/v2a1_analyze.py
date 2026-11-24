# V2a-1 Auswertung nach Praeregistrierung (V2a1_Praeregistrierung.md), mit Claude. Struktur wie lagma9_analyze.py.
# Aufruf: python v2a1_analyze.py [V2a1-Ordner]
# Primaer: BPreCap - SELU, Test-Fehler w8a4a_max (pp), 12 Seeds gepaart, Wilcoxon zweiseitig, alpha 0,05.
# Sekundaer (Holm ueber 3): S1 w8a4a_best, S2 fp32, S3 w8a8.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
SEEDS = list(range(42330, 42342)); CAND, REF = 'bprecap', 'selu'
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
rows = read(ROOT); D = {(r['act'], int(r['seed'])): r for r in rows}
def v(r, col): return float(r[col]) * 100 if r is not None and r['status'] == 'ok' and r.get(col, '') != '' else 90.0   # Ausfall/fehlend = 90 %
def test(col):
    x = np.array([v(D.get((CAND, s)), col) - v(D.get((REF, s)), col) for s in SEEDS])
    p = 1.0 if np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(col=col, n=len(x), med_diff_pp=float(np.median(x)), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=[round(float(a), 2) for a in x])
def verdict(r, key): r['verdict'] = 'unklar' if r[key] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar')); return r
def holm(res):
    ks = list(res); o = sorted(ks, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(o): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in ks: verdict(res[k], 'holm')
    return res
out = dict(n_rows=len(rows), n_ok=sum(r['status'] == 'ok' for r in rows), n_fail=sum(r['status'] != 'ok' for r in rows),
           missing=[f'{a}_{s}' for a in (CAND, REF) for s in SEEDS if (a, s) not in D])
out['primary'] = verdict(test('test_err_w8a4a_max'), 'p'); out['success'] = out['primary']['verdict'] == 'besser'
SEC = dict(S1='test_err_w8a4a_best', S2='test_err_fp32', S3='test_err_w8a8')
out['secondary'] = holm({k: test(c) for k, c in SEC.items()})
desc = {}
for a in (CAND, REF):
    rr = [D.get((a, s)) for s in SEEDS]; ok_ = [r for r in rr if r is not None and r['status'] == 'ok']
    med = lambda c: float(np.median([float(r[c]) for r in ok_])) if ok_ and all(r.get(c, '') != '' for r in ok_) else None
    desc[a] = {c: med(c) for c in ['val_err_fp32', 'test_err_fp32', 'test_err_w8a8', 'test_err_w8a4a_max', 'test_err_w8a4a_p999', 'test_err_w8a4a_p9999', 'test_err_w8a4a_best', 'act_lo_max_median', 'act_hi_max_median', 'act_hi_p9999_median', 'train_loss_last50']}
    desc[a]['best_cal'] = {k: sum(r['w8a4a_best_cal'] == k for r in ok_) for k in ('max', 'p999', 'p9999')}
    desc[a]['lr'] = sorted({r['lr'] for r in ok_})
out['descriptive'] = desc
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v2a1_result.json', 'w'), indent=1)
pr = out['primary']; print(f"Primaer BPreCap - SELU (w8a4a_max): {pr['med_diff_pp']:+.2f} pp, {pr['wins']}/{pr['losses']}, p = {pr['p']:.4f} -> {pr['verdict']}")
for k, r in out['secondary'].items(): print(f"{k} {r['col']}: {r['med_diff_pp']:+.2f} pp, {r['wins']}/{r['losses']}, Holm-p = {r['holm']:.4f} -> {r['verdict']}")
print('Zeilen', out['n_rows'], 'ok', out['n_ok'], 'Ausfall', out['n_fail'], 'fehlend', len(out['missing']))
