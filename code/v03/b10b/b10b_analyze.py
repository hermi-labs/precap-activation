# B10b Auswertung nach Praeregistrierung (B10b_Praeregistrierung.md), Funktionen wie b10_analyze.py
# Aufruf: python b10b_analyze.py [B10b-Ordner] [B10-Ordner]   (Pfade werden absolut aufgeloest)
# Vergleich je Seed (42168-42179): kL2B_tr aus B10 (1e-4, WD 0,3) gegen TAT a = (1e-4, 0,3) und TAT b = (5e-5, 3,0) aus B10b; TAT0 = B10-Wahl (2e-4, 0) aus B10.
# Fruehes Stoppen wie B10: je Lauf der Messpunkt mit dem kleinsten Val-Fehler fp32 (Gleichstand -> frueherer); ohne Messpunkt 90 %.
# Kontrolle: TAT Seed 42168 (2e-4, 0) aus B10b in allen Mess- und Endwerten bitgleich zu B10.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B10D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b10').resolve()
SEEDS = list(range(42168, 42180)); CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
CKCOLS = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
def key(r): return (r['act'], float(r['lr']), float(r['wd']), int(r['seed']))
NEW = {key(r): r for r in read(ROOT)}; OLD = {key(r): r for r in read(B10D)} if (B10D / 'workers').exists() else {}
ARMS = {'kL2B_tr': OLD, 'tat_a': NEW, 'tat_b': NEW, 'tat_0': OLD}
HP = {'kL2B_tr': ('kL2B_tr', 1e-4, 0.3), 'tat_a': ('tat', 1e-4, 0.3), 'tat_b': ('tat', 5e-5, 3.0), 'tat_0': ('tat', 2e-4, 0.0)}
def get(arm, s): a, lr, wd = HP[arm]; return ARMS[arm].get((a, lr, wd, s))
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def has(r, step):
    if step == 6880: return r['status'] == 'ok' and r.get('val_err_fp32') not in (None, '')
    return r.get(col(step, 'val_err_fp32')) not in (None, '')
def v(r, step, name): return float(r[col(step, name)]) * 100 if has(r, step) else 90.0
def es_step(r):
    av = [s for s in CK if has(r, s)]
    return min(av, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s)) if av else None
def es(name): return lambda r: (v(r, es_step(r), name) if r is not None and es_step(r) else 90.0)
def test(cand, ref, f):
    x = np.array([f(get(cand, s)) - f(get(ref, s)) for s in SEEDS if get(cand, s) is not None and get(ref, s) is not None])
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def verdict(r, pcol):
    return 'unklar' if r[pcol] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
prim = test('kL2B_tr', 'tat_a', es('test_err_w8a4a_best')); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
sec = holm({'S1_kL2Btr_vs_tatb_w4a': test('kL2B_tr', 'tat_b', es('test_err_w8a4a_best')),
            'S2_kL2Btr_vs_tata_fp32': test('kL2B_tr', 'tat_a', es('test_err_fp32')),
            'S3_kL2Btr_vs_tatb_fp32': test('kL2B_tr', 'tat_b', es('test_err_fp32')),
            'S4_kL2Btr_vs_tata_w4': test('kL2B_tr', 'tat_a', es('test_err_w8a4_best')),
            'S5_kL2Btr_vs_tatb_w4': test('kL2B_tr', 'tat_b', es('test_err_w8a4_best')),
            'S6_tata_vs_tat0_fp32': test('tat_a', 'tat_0', es('test_err_fp32')),
            'S7_tata_vs_tat0_w4a': test('tat_a', 'tat_0', es('test_err_w8a4a_best'))})
# Kontrolle
ck = ('tat', 2e-4, 0.0, 42168); a, b = NEW.get(ck), OLD.get(ck)
repro_ok = a is not None and b is not None and all(a.get(col(s, c)) == b.get(col(s, c)) for s in CK for c in CKCOLS)
desc = {}
for arm in HP:
    rr = [get(arm, s) for s in SEEDS if get(arm, s) is not None]
    if not rr: continue
    e = dict(n=len(rr), n_ok=sum(r['status'] == 'ok' for r in rr), es_steps=sorted(es_step(r) or 0 for r in rr))
    e['es'] = {n: float(np.median([es(n)(r) for r in rr])) for n in ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best')}
    e['es']['cost4a_pp'] = float(np.median([es('test_err_w8a4a_best')(r) - es('test_err_fp32')(r) for r in rr]))
    rv = [float(r[col(es_step(r), 'act_range_max_median')]) for r in rr if es_step(r)]; e['range_es'] = float(np.median(rv)) if rv else None
    e['range_curve'] = {str(s): float(np.median([float(r[col(s, 'act_range_max_median')]) for r in rr if has(r, s)])) for s in CK if any(has(r, s) for r in rr)}
    desc[arm] = e
new_rows = list(NEW.values())
out = dict(n_rows=len(new_rows), failed=sum(r['status'] != 'ok' for r in new_rows), primary=prim, success=success, secondary=sec,
           secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro_ok=repro_ok, descriptive=desc)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b10b_result.json', 'w'), indent=1)
print(f"Primaer kL2B_tr - TAT(1e-4, WD 0,3) w8a4a (fruehes Stoppen): med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Kontrolle B10:', repro_ok, '| Ausfaelle:', out['failed'], '| Zeilen:', out['n_rows'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
for arm, e in desc.items(): print(arm, 'Stopp', e['es_steps'], '| am Stopp:', {k: round(x, 2) for k, x in e['es'].items()}, '| Bereich', e['range_es'] and round(e['range_es'], 2))
