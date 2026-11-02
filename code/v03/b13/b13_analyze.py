# B12/B13 Auswertung nach Praeregistrierung, Funktionen wie b10_analyze.py. Diese Datei ist fuer B12 und B13 bis auf den Block STUDIE gleich.
# Aufruf: python b1x_analyze.py [Studienordner] [B10-Ordner]   (Pfade werden absolut aufgeloest)
# Gepaarter Vergleich je Seed (42168-42179): Formen dieser Studie (eigene Laeufe) gegen kL2B_tr bzw. LB_tr aus B10.
# Fruehes Stoppen wie B10: je Lauf der Messpunkt mit dem kleinsten Val-Fehler fp32 (Gleichstand -> frueherer); ohne Messpunkt 90 %.
# Kontrolle: kL2B_tr Seed 42168 (B10-Wahl) in allen Mess- und Endwerten bitgleich zu B10.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
# --- STUDIE ---
NAME = 'b13'
OWN = ('kL2Bh_tr', 'LBh_tr')
PRIMARY = [('kL2B_tr', 'kL2Bh_tr', 'test_err_w8a4a_best')]   # eine Zelle: weich gegen hart beim BPreCap-Kern, faire 4 Bit
SECONDARY = {'S1_LB_tr_vs_LBh_tr_w4a': ('LB_tr', 'LBh_tr', 'test_err_w8a4a_best'), 'S2_kL2B_tr_vs_kL2Bh_tr_w4': ('kL2B_tr', 'kL2Bh_tr', 'test_err_w8a4_best'),
             'S3_LB_tr_vs_LBh_tr_w4': ('LB_tr', 'LBh_tr', 'test_err_w8a4_best'), 'S4_kL2B_tr_vs_kL2Bh_tr_fp32': ('kL2B_tr', 'kL2Bh_tr', 'test_err_fp32'),
             'S5_LB_tr_vs_LBh_tr_fp32': ('LB_tr', 'LBh_tr', 'test_err_fp32'), 'S6_kL2B_tr_vs_kL2Bh_tr_w3': ('kL2B_tr', 'kL2Bh_tr', 'test_err_w8a3_best'),
             'S7_LB_tr_vs_LBh_tr_w3': ('LB_tr', 'LBh_tr', 'test_err_w8a3_best')}   # Holm ueber 7
# --- Ende STUDIE ---
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B10D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b10').resolve()
SEEDS = list(range(42168, 42180)); CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
CKCOLS = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
NEWR = read(ROOT); OLDR = read(B10D) if (B10D / 'workers').exists() else []
NEW = {(r['act'], int(r['seed'])): r for r in NEWR if r['dataset'] == 'cifar10' and r['act'] in OWN}
OLD = {(r['act'], int(r['seed'])): r for r in OLDR if r['dataset'] == 'cifar10' and int(r['seed']) in SEEDS}
def get(act, s): return NEW.get((act, s)) if act in OWN else OLD.get((act, s))
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def has(r, step):
    if step == 6880: return r['status'] == 'ok' and r.get('val_err_fp32') not in (None, '')
    return r.get(col(step, 'val_err_fp32')) not in (None, '')
def v(r, step, name): return float(r[col(step, name)]) * 100 if has(r, step) else 90.0
def es_step(r):
    av = [s for s in CK if has(r, s)]
    return min(av, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s)) if av else None
def es(r, name): return v(r, es_step(r), name) if r is not None and es_step(r) else 90.0
def test(cand, ref, m):
    x = np.array([es(get(cand, s), m) - es(get(ref, s), m) for s in SEEDS if get(cand, s) is not None and get(ref, s) is not None])
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand=cand, ref=ref, metric=m, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: r = res[k]; r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
prim = holm({f'P{i + 1}_{c}_vs_{r}_{m}': test(c, r, m) for i, (c, r, m) in enumerate(PRIMARY)})
sec = holm({k: test(*a) for k, a in SECONDARY.items()})
ctrl = [r for r in NEWR if r['act'] == 'kL2B_tr' and int(r['seed']) == 42168]; ref = OLD.get(('kL2B_tr', 42168))
repro_ok = len(ctrl) == 1 and ref is not None and all(ctrl[0].get(col(s, c)) == ref.get(col(s, c)) for s in CK for c in CKCOLS)
desc = {}
for act in sorted(set(OWN) | {a for a, _, _ in PRIMARY} | {r for _, r, _ in PRIMARY}):
    rr = [get(act, s) for s in SEEDS if get(act, s) is not None]
    if not rr: continue
    e = dict(n=len(rr), n_ok=sum(r['status'] == 'ok' for r in rr), lr=float(rr[0]['lr']), wd=float(rr[0]['wd']), es_steps=sorted(es_step(r) or 0 for r in rr))
    e['es'] = {n: float(np.median([es(r, n) for r in rr])) for n in ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best')}
    e['es']['cost4a_pp'] = float(np.median([es(r, 'test_err_w8a4a_best') - es(r, 'test_err_fp32') for r in rr]))
    rv = [float(r[col(es_step(r), 'act_range_max_median')]) for r in rr if es_step(r)]; e['range_es'] = float(np.median(rv)) if rv else None
    desc[act] = e
out = dict(n_rows=len(NEWR), failed=sum(r['status'] != 'ok' for r in NEWR), primary=prim, primary_verdicts={k: prim[k]['verdict'] for k in prim},
           success=all(prim[k]['verdict'] == 'besser' for k in prim), secondary=sec, secondary_verdicts={k: sec[k]['verdict'] for k in sec},
           repro_ok=repro_ok, descriptive=desc)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / f'{NAME}_result.json', 'w'), indent=1)
print('Primaer (Holm):', {k: (round(prim[k]['med_diff_pp'], 2), f"{prim[k]['wins']}/{prim[k]['losses']}", round(prim[k]['holm'], 4), prim[k]['verdict']) for k in prim}, '| alle besser:', out['success'])
print('Kontrolle B10:', repro_ok, '| Ausfaelle:', out['failed'], '| Zeilen:', out['n_rows'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
for a, e in desc.items(): print(a, e['lr'], e['wd'], 'Stopp', e['es_steps'], '| am Stopp:', {k: round(x, 2) for k, x in e['es'].items()}, '| Bereich', e['range_es'] and round(e['range_es'], 2))
