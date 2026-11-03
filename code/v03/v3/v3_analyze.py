# V3 Auswertung nach V3_Praeregistrierung.md (mit Claude). Aufruf: python v3_analyze.py [V3-Ordner]
# Messgroessen: FT/BASE -> PTQ w8a4a_best (Protokoll wie V1); LSQ/PACT -> Testfehler mit den gelernten Quantisierern (QAT-Inferenz).
# Primaer: kL2B_tr FT (PTQ) - tat LSQ (QAT), 12 Paare, Wilcoxon zweiseitig exakt, alpha 0,05. Sekundaer Holm ueber 6. Ausfall = 90 %.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
SEEDS = list(range(42168, 42180)); FORMS = ('kL2B_tr', 'tat'); METHODS = ('BASE', 'FT', 'LSQ', 'PACT')
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
rows = read(ROOT); D = {(r['act'], r['method'], int(r['seed'])): r for r in rows}
def ok(r): return r is not None and r['status'] == 'ok'
def q4(r):   # 4-Bit-Fehler in %: QAT-Inferenz fuer LSQ/PACT, PTQ w8a4a_best sonst
    if not ok(r): return 90.0
    return float(r['test_err_qat' if r['method'] in ('LSQ', 'PACT') else 'test_err_w8a4a_best']) * 100
def f32(r): return float(r['test_err_fp32']) * 100 if ok(r) else 90.0
def test(a, b, f=q4):
    x = np.array([f(D.get((a[0], a[1], s))) - f(D.get((b[0], b[1], s))) for s in SEEDS if (a[0], a[1], s) in D and (b[0], b[1], s) in D])
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand='_'.join(a), ref='_'.join(b), n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def verdict(r, k): return 'unklar' if r[k] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
B, T = 'kL2B_tr', 'tat'
prim = test((B, 'FT'), (T, 'LSQ')); prim['verdict'] = verdict(prim, 'p')
sec = holm({'S1_BPreCapFT_PTQ_vs_TAT_PACT': test((B, 'FT'), (T, 'PACT')), 'S2_BPreCap_LSQ_vs_TAT_LSQ': test((B, 'LSQ'), (T, 'LSQ')),
            'S3_BPreCap_PACT_vs_TAT_PACT': test((B, 'PACT'), (T, 'PACT')), 'S4_TAT_LSQ_vs_TAT_FT': test((T, 'LSQ'), (T, 'FT')),
            'S5_BPreCap_LSQ_vs_BPreCap_FT': test((B, 'LSQ'), (B, 'FT')), 'S6_BPreCapFT_vs_TATFT_PTQ': test((B, 'FT'), (T, 'FT'))})
base_ok = [D[(a, 'BASE', s)]['base_valid'] == 'True' for a in FORMS for s in SEEDS if (a, 'BASE', s) in D]
desc = {}
for a in FORMS:
    for mm in METHODS:
        rr = [D[(a, mm, s)] for s in SEEDS if (a, mm, s) in D]
        if not rr: continue
        e = dict(n=len(rr), n_ok=sum(ok(r) for r in rr), q4=float(np.median([q4(r) for r in rr])), fp32=float(np.median([f32(r) for r in rr])))
        e['cost4_pp'] = e['q4'] - e['fp32']
        if mm in ('LSQ', 'PACT'):
            mv = [float(r['range_move_med']) for r in rr if ok(r) and r['range_move_med'] not in ('', None)]; e['range_move_med'] = float(np.median(mv)) if mv else None
            e['ptq_after_qat'] = float(np.median([float(r['test_err_w8a4a_best']) * 100 for r in rr if ok(r)]))
            e['lrq'] = rr[0]['lrq']
        desc[f'{a}_{mm}'] = e
tune = {}
for r in read(ROOT / 'tune') if (ROOT / 'tune' / 'workers').exists() else []:
    if r['method'] in ('LSQ', 'PACT'): tune.setdefault(f"{r['act']}_{r['method']}_q{float(r['lrq']):g}", []).append(round(float(r['val_err_qat']) * 100, 2) if ok(r) else 90.0)
lrq_sel = json.load(open(ROOT / 'lrq_selected.json')) if (ROOT / 'lrq_selected.json').exists() else None
out = dict(n_rows=len(rows), failed=sum(not ok(r) for r in rows), base_valid=f'{sum(base_ok)}/{len(base_ok)}', base_all_valid=bool(base_ok) and all(base_ok),
           primary=prim, secondary=sec, secondary_verdicts={k: sec[k]['verdict'] for k in sec}, descriptive=desc, tune_val_qat=tune, lrq_selected=lrq_sel)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v3_result.json', 'w'), indent=1)
print(f"Primaer BPreCap-FT (PTQ) - TAT-LSQ (QAT): med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Basis bitgleich {out['base_valid']} | Ausfaelle {out['failed']}")
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
for k, e in desc.items(): print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in e.items()})
print('Tuning (Val-Fehler QAT %):', tune, '| Wahl LR_Q:', lrq_sel)
