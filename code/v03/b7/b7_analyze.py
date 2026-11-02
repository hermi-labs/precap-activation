# B7 Auswertung nach Praeregistrierung (B7_Praeregistrierung.md)
# Aufruf: python b7_analyze.py [B7-Ordner] [B3-Ordner]   (Pfade werden absolut aufgeloest)
# CIFAR-10, Tiefe 100, kL2B (BPreCap), LB, tat, 6880 Schritte (4-faches Budget von B3), Zwischenmessungen nach 1720 und 3440 Schritten.
# Kontrolle: kL2B und LB Seed 42084, Zwischenmessung nach 1720 Schritten gegen die Endwerte von B3 (bitgleich).
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B3D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b3').resolve()
NEWSEEDS = set(range(42132, 42144)); DEPTH = 100; ACTS = ('kL2B', 'LB', 'tat'); CK = (1720, 3440, 6880)
CKCOLS = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(B3D)} if (B3D / 'workers').exists() else {}
new = read(ROOT); REP = {}; D = {}
for r in new:
    if r['dataset'] == 'cifar10' and int(r['seed']) in NEWSEEDS: D[K(r)] = r
    else: REP[K(r)] = r
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def v(r, step, name):   # Fehler in %, Ausfall (Lauf nicht ok bzw. Zwischenwert fehlt) = 90 % (wie B1 bis B6)
    c = col(step, name)
    if step == 6880 and r['status'] != 'ok': return 90.0
    return float(r[c]) * 100 if r.get(c) not in (None, '') else 90.0
F32 = lambda st: (lambda r: v(r, st, 'test_err_fp32')); W4 = lambda st: (lambda r: v(r, st, 'test_err_w8a4_best')); W4A = lambda st: (lambda r: v(r, st, 'test_err_w8a4a_best'))
def diffs(f, cand, ref=None):
    x = []
    for s in sorted(NEWSEEDS):
        a = D.get(('cifar10', DEPTH, cand, s)); b = D.get(('cifar10', DEPTH, ref, s)) if ref else None
        if a is not None and (ref is None or b is not None): x.append(f(a) - (f(b) if ref else 0.0))
    return np.array(x)
def wtest(x, **info):
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(**info, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'), wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def test(cand, ref, f): return wtest(diffs(f, cand, ref), cand=cand, ref=ref)
def change(cand, ref, f_late, f_early):   # Veraenderung einer Differenz (bzw. eines Werts) zwischen zwei Messzeitpunkten, je Seed
    return wtest(diffs(f_late, cand, ref) - diffs(f_early, cand, ref), cand=cand, ref=ref)
def verdict(r, pcol):
    return 'unklar' if r[pcol] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
# Primaer: kL2B - LB fp32 nach 6880 Schritten, eine Zelle, alpha 0,05
prim = test('kL2B', 'LB', F32(6880)); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
cost = lambda st: (lambda r: W4A(st)(r) - F32(st)(r))   # 4-Bit-Preis (asymmetrisch) eines Laufs
sec = holm({'S1_kL2B_vs_LB_w4': test('kL2B', 'LB', W4(6880)),
            'S2_kL2B_vs_LB_w4a': test('kL2B', 'LB', W4A(6880)),
            'S3_gap_kL2B_LB_fp32_6880_minus_1720': change('kL2B', 'LB', F32(6880), F32(1720)),   # > 0: Vorsprung von kL2B schrumpft
            'S4_kL2B_vs_tat_fp32': test('kL2B', 'tat', F32(6880)),
            'S5_kL2B_vs_tat_w4a': test('kL2B', 'tat', W4A(6880)),
            'S6_LB_vs_tat_w4a': test('LB', 'tat', W4A(6880)),
            'S7_cost4a_kL2B_6880_minus_1720': change('kL2B', None, cost(6880), cost(1720)),       # > 0: 4-Bit-Preis von kL2B waechst
            'S8_gap_kL2B_tat_fp32_6880_minus_1720': change('kL2B', 'tat', F32(6880), F32(1720))})  # > 0: Abstand zu TAT waechst
# Kontrolle: Zwischenmessung 1720 von kL2B/LB Seed 42084 gegen die Endwerte von B3 (gleicher Seed, gleiche LR, 1720 Schritte)
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and r['status'] == 'ok' and all(r[f'c1720_{c}'] == a[c] for c in CKCOLS)))
repro_ok = len(repro) == 2 and all(x['equal'] for x in repro)
# Beschreibend: Mediane je Form und Messzeitpunkt, Trainingsgewinn, 4-Bit-Preis
desc = []
for act in ACTS:
    allr = [D[k] for k in D if k[2] == act]
    for st in CK:
        e = dict(act=act, step=st, n_all=len(allr), n_ok=sum(r['status'] == 'ok' for r in allr), lr=float(allr[0]['lr']) if allr else None)
        if allr:
            e.update({n: float(np.median([v(r, st, n) for r in allr])) for n in ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best')})
            okv = [float(r[col(st, n_)]) for r in allr for n_ in ['train_loss_last50'] if r.get(col(st, n_)) not in (None, '')]
            e['train_loss'] = float(np.median(okv)) if okv else None
            rv = [float(r[col(st, 'act_range_max_median')]) for r in allr if r.get(col(st, 'act_range_max_median')) not in (None, '')]
            e['range_max'] = float(np.median(rv)) if rv else None
            e['cost4a_pp'] = float(np.median([cost(st)(r) for r in allr]))
        desc.append(e)
gaps = {f'{a}_vs_{b}_fp32_{st}': float(np.median(diffs(F32(st), a, b))) for a, b in (('kL2B', 'LB'), ('kL2B', 'tat'), ('LB', 'tat')) for st in CK}
gain = {f'{a}_fp32_6880_minus_1720': float(np.median(diffs(F32(6880), a) - diffs(F32(1720), a))) for a in ACTS}
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, success=success, secondary=sec,
           secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro=repro, repro_ok=repro_ok, descriptive=desc, gaps=gaps, gain=gain)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b7_result.json', 'w'), indent=1)
print(f"Primaer kL2B - LB fp32 (6880): med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Kontrolle B3 (1720):', repro_ok, repro, '| Ausfaelle:', out['failed'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
print('Abstaende:', {k: round(x, 2) for k, x in gaps.items()}); print('Gewinn durch Training:', {k: round(x, 2) for k, x in gain.items()})
for e in desc: print(e['act'], e['step'], {k: (round(x, 2) if isinstance(x, float) else x) for k, x in e.items() if k not in ('act', 'step')})
