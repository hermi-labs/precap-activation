# Studie A2 Auswertung nach Praeregistrierung (A2_Praeregistrierung.md)
# Aufruf: python a2_analyze.py [A2-Ordner] [A-Ordner]
# kL2 und SELU im kurzen Budget (1720) stammen aus Studie A (gleiche Seeds 42010-42024, gleiche LR, gleicher Code-Pfad);
# die vier Reproduktionslaeufe in A2 pruefen, dass diese Uebernahme zulaessig ist.
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
AROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'a'
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
new = read(ROOT); old = [r for r in read(AROOT) if r['act'] in ('kL2', 'selu') and int(r['depth']) in (100, 200)]
SHORT, LONG = 1720, 6880; DEPTHS = (100, 200); DS = ('mnist', 'fashion'); CELLS = [(ds, d) for d in DEPTHS for ds in DS]; BUD = (SHORT, LONG)
D = {}
for r in old: D[(r['dataset'], int(r['depth']), r['act'], SHORT, int(r['seed']))] = r
REP = {}
for r in new:
    k = (r['dataset'], int(r['depth']), r['act'], int(r['steps']), int(r['seed']))
    if r['act'] in ('kL2', 'selu') and int(r['steps']) == SHORT: REP[k] = r     # Reproduktion, ersetzt A-Daten nicht
    else: D[k] = r
def val(r): return float(r['test_err_fp32']) * 100 if r['status'] == 'ok' else 90.0   # Ausfall = Zufallsniveau (vorab festgelegt)
def test(ds, d, T, cand, ref):
    x = []
    for s in sorted({k[4] for k in D}):
        a, b = D.get((ds, d, cand, T, s)), D.get((ds, d, ref, T, s))
        if a is not None and b is not None: x.append(val(a) - val(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='auto').pvalue)
    return dict(dataset=ds, depth=d, steps=T, cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p)
def holm(res):
    o = sorted(range(len(res)), key=lambda i: res[i]['p']); m = len(res); run = 0
    for k, i in enumerate(o): run = max(run, min(1.0, (m - k) * res[i]['p'])); res[i]['holm'] = run
    for r in res: r['verdict'] = 'unklar' if r['holm'] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
    return res
def overall(res, ref):
    nb = sum(r['verdict'] == 'besser' for r in res); ns = sum(r['verdict'] == 'schlechter' for r in res)
    return f'besser als {ref}' if nb >= 2 and ns == 0 else (f'schlechter als {ref}' if ns >= 2 and nb == 0 else ('kein Effekt' if nb == 0 and ns == 0 else 'gemischt'))
prim = holm([test(ds, d, T, 'kL2', 'tat') for T in BUD for ds, d in CELLS])
verdict = overall(prim, 'TAT'); verdict_by_budget = {T: overall([r for r in prim if r['steps'] == T], 'TAT') for T in BUD}
sec = {'S1_kL2_vs_selu_6880': holm([test(ds, d, LONG, 'kL2', 'selu') for ds, d in CELLS]),
       'S2_tat_vs_selu': holm([test(ds, d, T, 'tat', 'selu') for T in BUD for ds, d in CELLS])}
# Reproduktion: A2-Lauf gegen A-Lauf mit gleichem Seed
repro = []
for k, r in sorted(REP.items()):
    a = D.get(k)
    repro.append(dict(key=list(k), a2=val(r), a=val(a) if a else None, diff_pp=(val(r) - val(a)) if a else None))
repro_ok = bool(repro) and all(x['diff_pp'] is not None and abs(x['diff_pp']) <= 0.10 for x in repro)
desc = []
for ds, d in CELLS:
    for T in BUD:
        for act in ('kL2', 'tat', 'selu'):
            allr = [D[k] for k in D if k[:4] == (ds, d, act, T)]; ok = [r for r in allr if r['status'] == 'ok']
            e = dict(dataset=ds, depth=d, steps=T, act=act, n_ok=len(ok), n_all=len(allr))
            if allr: e['med_grad_log10_first_last'] = float(np.nanmedian([float(r['grad_log10_first_last']) for r in allr]))
            if ok:
                e.update(lr=float(ok[0]['lr']), med_test_err=float(np.median([val(r) for r in allr])),
                         med_train_loss=float(np.median([float(r['train_loss_last50']) for r in ok])),
                         med_ms_per_step=float(np.median([float(r['ms_per_step']) for r in ok])),
                         med_range_max=float(np.median([float(r['act_range_max_median']) for r in ok])))
                for s in ('1720', '3440'):
                    v = [float(r[f'val_err_at_{s}']) * 100 for r in ok if r.get(f'val_err_at_{s}', '') not in ('', None)]
                    if v: e[f'med_val_err_at_{s}'] = float(np.median(v))
                e['med_val_err_end'] = float(np.median([float(r['val_err_fp32']) * 100 for r in ok]))
            desc.append(e)
lrt = list(csv.DictReader(open(ROOT / 'lr_tuned.csv'))) if (ROOT / 'lr_tuned.csv').exists() else []
out = dict(n_rows_a2=len(new), n_rows_from_a=len(old), failed_a2=sum(r['status'] != 'ok' for r in new), primary=prim, verdict=verdict,
           verdict_by_budget=verdict_by_budget, secondary=sec, secondary_verdicts={'S1': overall(sec['S1_kL2_vs_selu_6880'], 'SELU'),
           'S2_1720': overall([r for r in sec['S2_tat_vs_selu'] if r['steps'] == SHORT], 'SELU'), 'S2_6880': overall([r for r in sec['S2_tat_vs_selu'] if r['steps'] == LONG], 'SELU')},
           repro=repro, repro_ok=repro_ok, descriptive=desc, lr_tuned=lrt, lr_at_edge=[r for r in lrt if r.get('at_edge') == 'True'])
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'a2_result.json', 'w'), indent=1)
print('Urteil kL2 vs TAT:', verdict, verdict_by_budget, '| Sekundaer:', out['secondary_verdicts'], '| Repro ok:', repro_ok, '| Ausfaelle:', out['failed_a2'], '| LR am Rand:', len(out['lr_at_edge']))
for r in prim: print(f"T{r['steps']} {r['dataset']:7s} d{r['depth']:3d} n={r['n']:2d} med={r['med_diff_pp']:+.2f}pp W/L={r['wins']}/{r['losses']} holm={r['holm']:.4f} {r['verdict']}")
