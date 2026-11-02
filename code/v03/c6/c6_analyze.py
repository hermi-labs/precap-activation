# C6 Auswertung nach Praeregistrierung (C6_Praeregistrierung.md). Mit Claude geschrieben, Statistik wie B15 (eigener exakter Wilcoxon-Test).
# Aufruf: python c6_analyze.py [C6-Ordner] [Ordner mit den Studien b10, b10b, b12 fuer den beschreibenden Plattformvergleich]
# Fruehes Stoppen: je Lauf der Messpunkt mit dem kleinsten fp32-Val-Fehler (Gleichstand -> frueherer). Ausfall ohne Messpunkt = 90 %.
import csv, itertools, json, sys
from pathlib import Path
import numpy as np
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
ST = (Path(sys.argv[2]) if len(sys.argv) > 2 else (ROOT / 'mlx_vergleich' if (ROOT / 'mlx_vergleich').exists() else ROOT.parent)).resolve()   # Kopie der MLX-Roh-CSVs (b10, b10b, b12)
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880); SEEDS = list(range(42168, 42180)); FAIL = 90.0
FORMS = {'bp': ('kL2B_tr', 1e-4, 0.3), 'lb': ('LB_tr', 1e-4, 0.3), 'ta': ('tat', 1e-4, 0.3), 't0': ('tat', 2e-4, 0.0), 'se': ('selu_tr', 2e-4, 0.3)}
MLXSRC = {'bp': ('b10', 'kL2B_tr', None), 'lb': ('b10', 'LB_tr', None), 'ta': ('b10b', 'tat', (1e-4, 0.3)), 't0': ('b10', 'tat', None), 'se': ('b12', 'selu_tr', None)}
def rows(d):
    out = []
    for f in sorted((d / 'workers').glob('w*.csv')): out += list(csv.DictReader(open(f)))
    return out
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def has(r, step):
    if step == 6880: return r.get('status') == 'ok' and r.get('val_err_fp32') not in (None, '')
    return r.get(col(step, 'val_err_fp32')) not in (None, '')
def es_step(r):
    av = [s for s in CK if has(r, s)]
    return min(av, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s)) if av else None
def es(r, name): s = es_step(r) if r is not None else None; return float(r[col(s, name)]) * 100 if s else FAIL
def exact_p(x):   # zweiseitiger exakter Wilcoxon-Vorzeichen-Rang-Test (Nullen verworfen, Mittelraenge)
    d = [float(v) for v in x if v != 0]; n = len(d)
    if n == 0: return 1.0
    a = [abs(v) for v in d]; idx = sorted(range(n), key=lambda i: a[i]); rk = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and a[idx[j + 1]] == a[idx[i]]: j += 1
        for k in range(i, j + 1): rk[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    wp = sum(r for r, v in zip(rk, d) if v > 0); lo = hi = 0
    for sg in itertools.product((0, 1), repeat=n):
        w = sum(r for r, s_ in zip(rk, sg) if s_); lo += w <= wp + 1e-9; hi += w >= wp - 1e-9
    return min(1.0, 2 * min(lo, hi) / 2 ** n)
R = rows(ROOT); D = {}
for r in R:
    for k, (a, lr, wd) in FORMS.items():
        if r['act'] == a and abs(float(r['lr']) - lr) < 1e-12 and abs(float(r['wd']) - wd) < 1e-12 and int(r['seed']) in SEEDS: D[(k, int(r['seed']))] = r
def test(a, b, name):
    x = np.array([es(D.get((a, s)), name) - es(D.get((b, s)), name) for s in SEEDS])
    return dict(n=len(x), med_diff_pp=float(np.median(x)), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=exact_p(x), diffs=x.round(3).tolist())
def verdict(t, key): return 'unklar' if t[key] >= 0.05 else ('besser' if t['med_diff_pp'] < 0 else ('schlechter' if t['med_diff_pp'] > 0 else 'unklar'))
prim = test('bp', 'ta', 'test_err_w8a4a_best'); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
sec = {'S1_bp_vs_ta_fp32': test('bp', 'ta', 'test_err_fp32'), 'S2_bp_vs_t0_w4a': test('bp', 't0', 'test_err_w8a4a_best'),
       'S3_bp_vs_se_w4a': test('bp', 'se', 'test_err_w8a4a_best'), 'S4_bp_vs_lb_fp32': test('bp', 'lb', 'test_err_fp32'), 'S5_bp_vs_se_fp32': test('bp', 'se', 'test_err_fp32')}
ks = sorted(sec, key=lambda k: sec[k]['p']); run = 0
for i, k in enumerate(ks): run = max(run, min(1.0, (len(ks) - i) * sec[k]['p'])); sec[k]['holm'] = run
for k in sec: sec[k]['verdict'] = verdict(sec[k], 'holm')
# Beschreibend: Mediane je Form am Stopp-Punkt, 4-Bit-Preis, Bereich; Plattformvergleich mit den MLX-Laeufen (gleiche Seeds, gleiche Hyperparameter)
desc = {}
for k in FORMS:
    rr = [D.get((k, s)) for s in SEEDS]; ok = [r for r in rr if r is not None and r.get('status') == 'ok']
    e = dict(n=len(ok), es_steps=sorted(es_step(r) or 0 for r in ok))
    for m in ('test_err_fp32', 'test_err_w8a4a_best', 'test_err_w8a4_best', 'test_err_w8a3_best'): e[m] = float(np.median([es(r, m) for r in rr]))
    e['cost4a_pp'] = float(np.median([es(r, 'test_err_w8a4a_best') - es(r, 'test_err_fp32') for r in rr]))
    e['range_es'] = float(np.median([float(r[col(es_step(r), 'act_range_max_median')]) for r in ok])) if ok else None
    src, act, hp = MLXSRC[k]; mr = {}
    if (ST / src / 'workers').exists():
        for r in rows(ST / src):
            if r['dataset'] == 'cifar10' and int(r['depth']) == 100 and r['act'] == act and int(r['seed']) in SEEDS and (hp is None or (float(r['lr']), float(r['wd'])) == hp): mr[int(r['seed'])] = r
    if len(mr) == 12:
        e['mlx'] = {m: float(np.median([es(mr[s], m) for s in SEEDS])) for m in ('test_err_fp32', 'test_err_w8a4a_best')}
        e['mlx']['cost4a_pp'] = float(np.median([es(mr[s], 'test_err_w8a4a_best') - es(mr[s], 'test_err_fp32') for s in SEEDS]))
        e['cuda_minus_mlx_fp32_pp'] = [round(es(D.get((k, s)), 'test_err_fp32') - es(mr[s], 'test_err_fp32'), 2) for s in SEEDS]
    desc[k] = e
out = dict(n_rows=len(R), failed=sum(r.get('status') != 'ok' for r in R), primary=prim, success=success, secondary=sec,
           secondary_verdicts={k: sec[k]['verdict'] for k in sec}, descriptive=desc, devices=sorted({r.get('device', '') for r in R}), torch=sorted({r.get('torch', '') for r in R}))
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'c6_result.json', 'w'), indent=1)
print(f"Primaer BPreCap - TAT a w8a4a (ES): med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
print('Ausfaelle:', out['failed'], '| Geraete:', out['devices'], '| torch:', out['torch'])
for k, e in desc.items(): print(k, {m: (round(v, 2) if isinstance(v, float) else v) for m, v in e.items() if m not in ('es_steps', 'cuda_minus_mlx_fp32_pp')})
