# C7 Auswertung nach Praeregistrierung (C7_Praeregistrierung.md), mit Claude. Aufruf: python c7_analyze.py <C7-Ergebnisordner, absolut> [<V2a2-Ordner>]
# Primaer: P1 BPreCap - SELU, P2 BPreCap - TAT, Testfehler es_test_err_w8a4a_best, Wilcoxon gepaart zweiseitig, Holm ueber 2, Ausfall = 90 %.
# Sekundaer (beschreibend): Median CUDA gegen Median MLX (V2a2) je Form und Messgroesse.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
D = Path(sys.argv[1]); V = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'v2a2'
SEEDS = list(range(42350, 42362)); FORMS = ('bprecap', 'selu', 'tat'); FAIL = 0.9
rows = [r for f in sorted((D / 'workers').glob('w*.csv')) for r in csv.DictReader(open(f))]
mlx = [r for r in csv.DictReader(open(V / 'workers' / 'w0.csv')) if r['stage'] == 'confirm']
def table(rs, key):
    t = {}
    for r in rs:
        v = float(r[key]) if r['status'] == 'ok' and r.get(key, '') != '' else FAIL
        t[(r['act'], int(r['seed']))] = v
    return t
out = dict(n_rows=len(rows), n_ok=sum(r['status'] == 'ok' for r in rows), soll=36, run_id_eindeutig=len({r['run_id'] for r in rows}) == len(rows))
out['fehlend'] = [f'{a}_{s}' for a in FORMS for s in SEEDS if not any(r['act'] == a and int(r['seed']) == s for r in rows)]
T = table(rows, 'es_test_err_w8a4a_best')
def test(ref):
    d = [(T.get(('bprecap', s), FAIL) - T.get((ref, s), FAIL)) * 100 for s in SEEDS]
    p = float(wilcoxon(d, alternative='two-sided', zero_method='wilcox').pvalue) if any(x != 0 for x in d) else 1.0
    return dict(med_diff_pp=float(np.median(d)), siege=sum(x < 0 for x in d), niederlagen=sum(x > 0 for x in d), p=p, diffs=[round(x, 3) for x in d])
P = {'P1': test('selu'), 'P2': test('tat')}
ps = sorted(P, key=lambda k: P[k]['p']); prev = 0
for i, k in enumerate(ps):
    prev = max(prev, min(1.0, P[k]['p'] * (len(ps) - i))); P[k]['p_holm'] = prev
for k, v in P.items(): v['urteil'] = ('besser' if v['med_diff_pp'] < 0 else 'schlechter') if v['p_holm'] < 0.05 else 'unklar'
out['primaer'] = P
keys = ['es_test_err_fp32', 'es_test_err_w8a8', 'es_test_err_w8a4a_max', 'es_test_err_w8a4a_best']
cmp = {}
for a in FORMS:
    c = {}
    for k in keys:
        tc, tm = table([r for r in rows if r['act'] == a], k), table([r for r in mlx if r['act'] == a], k)
        mc, mm = float(np.median([tc.get((a, s), FAIL) for s in SEEDS])) * 100, float(np.median([tm.get((a, s), FAIL) for s in SEEDS])) * 100
        c[k] = dict(cuda=round(mc, 3), mlx=round(mm, 3), diff_pp=round(mc - mm, 3))
    pc = [(table(rows, 'es_test_err_w8a4a_best').get((a, s), FAIL) - table(rows, 'es_test_err_fp32').get((a, s), FAIL)) * 100 for s in SEEDS]
    pm = [(table(mlx, 'es_test_err_w8a4a_best').get((a, s), FAIL) - table(mlx, 'es_test_err_fp32').get((a, s), FAIL)) * 100 for s in SEEDS]
    c['preis_4bit_pp'] = dict(cuda=round(float(np.median(pc)), 3), mlx=round(float(np.median(pm)), 3))
    cmp[a] = c
out['cuda_gegen_mlx'] = cmp
tc = table([r for r in rows if r['act'] == 'tat'], 'es_cle_test_err_w8a4a_best'); tm = table([r for r in mlx if r['act'] == 'tat'], 'es_cle_test_err_w8a4a_best')
out['tat_cle_w8a4a_best'] = dict(cuda=round(float(np.median([tc.get(('tat', s), FAIL) for s in SEEDS])) * 100, 3), mlx=round(float(np.median([tm.get(('tat', s), FAIL) for s in SEEDS])) * 100, 3))
(D / 'analysis').mkdir(exist_ok=True); json.dump(out, open(D / 'analysis' / 'c7_result.json', 'w'), indent=1)
for k, v in P.items(): print(k, 'Median %.2f pp' % v['med_diff_pp'], '%d/%d' % (v['siege'], v['niederlagen']), 'p_holm %.4f' % v['p_holm'], '->', v['urteil'])
print('Zeilen', out['n_rows'], 'ok', out['n_ok'], 'fehlend', out['fehlend'])
for a in FORMS: print(a, {k.replace('es_test_err_', ''): (v['cuda'], v['mlx']) for k, v in cmp[a].items()})
