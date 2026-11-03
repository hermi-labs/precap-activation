# V1c Auswertung nach Praeregistrierung (V1c_Praeregistrierung.md), Code aus v1_analyze.py, mit Claude
# Aufruf: python v1c_analyze.py [V1c-Ordner] [B15-Ordner]   (Pfade werden absolut aufgeloest)
# V1c = B15 mit linearem LR-Anlauf (5 %) fuer beide Formen, LR-Raster 5e-5/1e-4/2e-4/3e-4. Gepaart mit B15 ueber die Seeds 42192-42203.
# CIFAR-10, Tiefe 200, kL2B_tr, selu_tr, AdamW mit Kosinus-LR und WD (je Form getunt), 6880 Schritte, Messpunkte alle 860 Schritte.
# Fruehes Stoppen: je Lauf der Messpunkt mit dem kleinsten Val-Fehler fp32 (Gleichstand -> frueherer); fehlende Messpunkte (Ausfall) zaehlen nicht,
# ohne jeden Messpunkt zaehlt der Lauf als 90 %. Kontrolle: kL2B_tr (5e-5/0,3) und selu_tr (2e-4/0,3) Seed 42192 ohne Anlauf in allen Mess- und Endwerten bitgleich zu B15.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B9D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b15').resolve()   # hier: B15-Ordner
NEWSEEDS = set(range(42192, 42204)); DEPTH = 200; ACTS = ('kL2B_tr', 'selu_tr')
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
CKCOLS = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
MET = ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best')
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(B9D) if int(r['seed']) in NEWSEEDS and int(r['depth']) == DEPTH} if (B9D / 'workers').exists() else {}   # B15-Bestaetigung (ohne Anlauf)
new = read(ROOT); REP = {}; D = {}
for r in new:
    if r['dataset'] == 'cifar10' and int(r['seed']) in NEWSEEDS and r['sched'] == 'wcos': D[K(r)] = r
    elif r['sched'] == 'cos': REP[K(r)] = r
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def has(r, step):   # Messpunkt vorhanden (Endmessung nur bei status ok)
    if step == 6880: return r['status'] == 'ok' and r.get('val_err_fp32') not in (None, '')
    return r.get(col(step, 'val_err_fp32')) not in (None, '')
def v(r, step, name): return float(r[col(step, name)]) * 100 if has(r, step) else 90.0   # Fehler in %, fehlend = 90 %
def es_step(r):
    av = [s for s in CK if has(r, s)]
    return min(av, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s)) if av else None
def es(name): return lambda r: (v(r, es_step(r), name) if es_step(r) else 90.0)
def at(step, name): return lambda r: v(r, step, name)
def test(cand, ref, f):
    x = []
    for s in sorted(NEWSEEDS):
        a, b = D.get(('cifar10', DEPTH, cand, s)), D.get(('cifar10', DEPTH, ref, s))
        if a is not None and b is not None: x.append(f(a) - f(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def test_x(act, f):   # V1c (mit Anlauf) - B15 (ohne Anlauf), gleiche Form, gepaart ueber den Seed
    x = []
    for s in sorted(NEWSEEDS):
        a, b = D.get(('cifar10', DEPTH, act, s)), OLD.get(('cifar10', DEPTH, act, s))
        if a is not None and b is not None: x.append(f(a) - f(b))
    x = np.array(x)
    p = 1.0 if len(x) == 0 or np.all(x == 0) else float(wilcoxon(x, zero_method='wilcox', alternative='two-sided', method='exact').pvalue)
    return dict(cand=act + '_V1c', ref=act + '_B15', n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def verdict(r, pcol):
    return 'unklar' if r[pcol] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
# Primaer: kL2B_tr - selu_tr bei w8a4a_best am Messpunkt des fruehen Stoppens, eine Zelle, alpha 0,05
prim = test('kL2B_tr', 'selu_tr', es('test_err_w8a4a_best')); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
sec = holm({'S1_kL2Btr_vs_selu_fp32_es': test('kL2B_tr', 'selu_tr', es('test_err_fp32')),
            'S2_kL2Btr_V1c_vs_B15_fp32_es': test_x('kL2B_tr', es('test_err_fp32')),
            'S3_kL2Btr_V1c_vs_B15_w4a_es': test_x('kL2B_tr', es('test_err_w8a4a_best')),
            'S4_selu_V1c_vs_B15_fp32_es': test_x('selu_tr', es('test_err_fp32')),
            'S5_selu_V1c_vs_B15_w4a_es': test_x('selu_tr', es('test_err_w8a4a_best')),
            'S6_kL2Btr_vs_selu_w4a_6880': test('kL2B_tr', 'selu_tr', at(6880, 'test_err_w8a4a_best'))})
# Kontrolle: B15-Wahl ohne Anlauf (cos), Seed 42192, gegen die B15-Bestaetigung: alle Kennzahlen an allen Messpunkten und am Ende
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r.get(col(s, c)) == a.get(col(s, c)) for s in CK for c in CKCOLS)))
repro_ok = len(repro) == 2 and all(x['equal'] for x in repro)
# Beschreibend: Verteilung der Stopp-Punkte, Mediane je Form am Stopp-Punkt und an jedem Messpunkt, 4-Bit-Preis, Bereich
desc = {}
for act in ACTS:
    allr = [D[k] for k in D if k[2] == act]
    if not allr: continue
    e = dict(n_all=len(allr), n_ok=sum(r['status'] == 'ok' for r in allr), lr=float(allr[0]['lr']), wd=float(allr[0]['wd']), es_steps=sorted(es_step(r) or 0 for r in allr))
    e['n_barely_train'] = sum(r['status'] != 'ok' or float(r['train_loss_last50']) > 2.0 for r in allr)   # V1c: wie B15 (SELU zweigeteilt), Trainingsverlust am Ende > 2,0
    e['n_barely_train_B15'] = sum(float(r['train_loss_last50']) > 2.0 for k, r in OLD.items() if k[2] == act and r['sched'] == 'cos')
    e['es'] = {n: float(np.median([es(n)(r) for r in allr])) for n in MET}
    e['es']['cost4a_pp'] = float(np.median([es('test_err_w8a4a_best')(r) - es('test_err_fp32')(r) for r in allr]))
    e['curve'] = {}
    for st in CK:
        c = {n: float(np.median([v(r, st, n) for r in allr])) for n in MET}
        rv = [float(r[col(st, 'act_range_max_median')]) for r in allr if has(r, st)]; c['range_max'] = float(np.median(rv)) if rv else None
        tl = [float(r[col(st, 'train_loss_last50')]) for r in allr if has(r, st)]; c['train_loss'] = float(np.median(tl)) if tl else None
        e['curve'][str(st)] = c
    desc[act] = e
# Beschreibend: Tuning-Tabelle (Val-Fehler am Ende je LR x WD, beide Seeds) und Einbruch (Val-Fehler > 80 %) je Zelle
tune = {}
for f in sorted((ROOT / 'tune' / 'workers').glob('w*.csv')):
    for r in csv.DictReader(open(f)):
        ve = float(r['val_err_fp32']) * 100 if r['status'] == 'ok' and r['val_err_fp32'] not in ('', None) else 90.0
        tune.setdefault(f"{r['act']}_lr{float(r['lr']):g}_wd{float(r['wd']):g}", []).append(round(ve, 2))
tune_collapse = {k: all(x > 80 for x in v) for k, v in tune.items()}
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, success=success, secondary=sec, tune=tune, tune_collapse=tune_collapse,
           secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro=repro, repro_ok=repro_ok, descriptive=desc)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'v1c_result.json', 'w'), indent=1)
print(f"Primaer kL2B_tr - selu_tr w8a4a (fruehes Stoppen): med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Kontrolle B15:', repro_ok, repro, '| Ausfaelle:', out['failed'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
for a, e in desc.items():
    print(a, 'kaum trainiert', e['n_barely_train'], '(B15:', e['n_barely_train_B15'], ') Stopp-Punkte', e['es_steps'], '| am Stopp:', {k: round(x, 2) for k, x in e['es'].items()})
    print('   Verlauf fp32/w4a/Bereich:', {st: (round(c['test_err_fp32'], 1), round(c['test_err_w8a4a_best'], 1), c['range_max'] and round(c['range_max'], 2)) for st, c in e['curve'].items()})
print('Tuning (Val-Fehler %):', tune)
