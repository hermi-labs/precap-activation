# B15 Auswertung nach Praeregistrierung (B15_Praeregistrierung.md), Code aus b11_analyze.py
# Aufruf: python b15_analyze.py [B15-Ordner] [B10-Ordner]   (Pfade werden absolut aufgeloest)
# CIFAR-10, Tiefe 200, kL2B_tr gegen selu_tr, AdamW mit Kosinus-LR und WD (je Form getunt), 6880 Schritte, Messpunkte alle 860 Schritte.
# Fruehes Stoppen: je Lauf der Messpunkt mit dem kleinsten Val-Fehler fp32 (Gleichstand -> frueherer); fehlende Messpunkte (Ausfall) zaehlen nicht,
# ohne jeden Messpunkt zaehlt der Lauf als 90 % (10 Klassen, Zufall). Kontrolle: kL2B_tr CIFAR-10 d100 Seed 42168 in allen Mess- und Endwerten bitgleich zu B10.
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent).resolve()
B9D = (Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / 'b10').resolve()   # Referenz fuer die Kontrolle: B10
NEWSEEDS = set(range(42192, 42204)); DEPTH = 200; ACTS = ('kL2B_tr', 'selu_tr')
CK = (860, 1720, 2580, 3440, 4300, 5160, 6020, 6880)
CKCOLS = ['train_loss_last50', 'val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best', 'act_range_max_median']
MET = ('val_err_fp32', 'test_err_fp32', 'test_err_w8a4_best', 'test_err_w8a4a_best', 'test_err_w8a3_best')
def read(d):
    r = []
    for f in sorted((d / 'workers').glob('w*.csv')): r += list(csv.DictReader(open(f)))
    return r
K = lambda r: (r['dataset'], int(r['depth']), r['act'], int(r['seed']))
OLD = {K(r): r for r in read(B9D)} if (B9D / 'workers').exists() else {}
new = read(ROOT); REP = {}; D = {}
for r in new:
    if r['dataset'] == 'cifar10' and int(r['depth']) == DEPTH and int(r['seed']) in NEWSEEDS: D[K(r)] = r
    else: REP[K(r)] = r
def col(step, name): return name if step == 6880 else f'c{step}_{name}'
def has(r, step):   # Messpunkt vorhanden (Endmessung nur bei status ok)
    if step == 6880: return r['status'] == 'ok' and r.get('val_err_fp32') not in (None, '')
    return r.get(col(step, 'val_err_fp32')) not in (None, '')
FAIL = 90.0   # Ausfall = Zufallsniveau bei 10 Klassen
def v(r, step, name): return float(r[col(step, name)]) * 100 if has(r, step) else FAIL   # Fehler in %, fehlend = 99 %
def es_step(r):
    av = [s for s in CK if has(r, s)]
    return min(av, key=lambda s: (float(r[col(s, 'val_err_fp32')]), s)) if av else None
def es(name): return lambda r: (v(r, es_step(r), name) if es_step(r) else FAIL)
def at(step, name): return lambda r: v(r, step, name)
def exact_p(x):   # zweiseitiger exakter Wilcoxon-Vorzeichen-Rang-Test ueber alle 2^n Vorzeichenmuster (wie analysis_check/nachrechnen.py)
    import itertools
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
def test(cand, ref, f):
    x = []
    for s in sorted(NEWSEEDS):
        a, b = D.get(('cifar10', DEPTH, cand, s)), D.get(('cifar10', DEPTH, ref, s))
        if a is not None and b is not None: x.append(f(a) - f(b))
    x = np.array(x)
    p = exact_p(x)   # B15: eigener exakter Test (Nullen verworfen, Mittelraenge), weil scipy bei Nullen still zur Normal-Naeherung wechselt
    return dict(cand=cand, ref=ref, n=len(x), med_diff_pp=float(np.median(x)) if len(x) else float('nan'),
                wins=int((x < 0).sum()), losses=int((x > 0).sum()), p=p, diffs=x.round(3).tolist())
def verdict(r, pcol):
    return 'unklar' if r[pcol] >= 0.05 else ('besser' if r['med_diff_pp'] < 0 else ('schlechter' if r['med_diff_pp'] > 0 else 'unklar'))
def holm(res):
    ks = sorted(res, key=lambda k: res[k]['p']); m = len(ks); run = 0
    for i, k in enumerate(ks): run = max(run, min(1.0, (m - i) * res[k]['p'])); res[k]['holm'] = run
    for k in res: res[k]['verdict'] = verdict(res[k], 'holm')
    return res
# Primaer: kL2B_tr - selu_tr bei w8a4a_best am Messpunkt des fruehen Stoppens (d200), eine Zelle, alpha 0,05
prim = test('kL2B_tr', 'selu_tr', es('test_err_w8a4a_best')); prim['verdict'] = verdict(prim, 'p'); success = prim['verdict'] == 'besser'
sec = holm({'S1_kL2Btr_vs_selutr_fp32_es': test('kL2B_tr', 'selu_tr', es('test_err_fp32')),
            'S2_kL2Btr_vs_selutr_w4_es': test('kL2B_tr', 'selu_tr', es('test_err_w8a4_best')),
            'S3_kL2Btr_vs_selutr_w3_es': test('kL2B_tr', 'selu_tr', es('test_err_w8a3_best'))})
# Kontrolle: CIFAR-10 d100 Seed 42168 mit den Hyperparametern aus B10 gegen B10 (gleicher Lauf): alle Kennzahlen an allen Messpunkten und am Ende
repro = []
for k, r in sorted(REP.items()):
    a = OLD.get(k)
    repro.append(dict(key=list(k), equal=a is not None and all(r.get(col(s, c)) == a.get(col(s, c)) for s in CK for c in CKCOLS)))
repro_ok = len(repro) == 1 and all(x['equal'] for x in repro)
# Beschreibend: Verteilung der Stopp-Punkte, Mediane je Form am Stopp-Punkt und an jedem Messpunkt, 4-Bit-Preis, Bereich
desc = {}
for act in ACTS:
    allr = [D[k] for k in D if k[2] == act]
    if not allr: continue
    e = dict(n_all=len(allr), n_ok=sum(r['status'] == 'ok' for r in allr), lr=float(allr[0]['lr']), wd=float(allr[0]['wd']), es_steps=sorted(es_step(r) or 0 for r in allr))
    e['es'] = {n: float(np.median([es(n)(r) for r in allr])) for n in MET}
    e['es']['cost4a_pp'] = float(np.median([es('test_err_w8a4a_best')(r) - es('test_err_fp32')(r) for r in allr]))
    e['curve'] = {}
    for st in CK:
        c = {n: float(np.median([v(r, st, n) for r in allr])) for n in MET}
        rv = [float(r[col(st, 'act_range_max_median')]) for r in allr if has(r, st)]; c['range_max'] = float(np.median(rv)) if rv else None
        tl = [float(r[col(st, 'train_loss_last50')]) for r in allr if has(r, st)]; c['train_loss'] = float(np.median(tl)) if tl else None
        e['curve'][str(st)] = c
    desc[act] = e
out = dict(n_rows=len(new), failed=sum(r['status'] != 'ok' for r in new), primary=prim, success=success, secondary=sec,
           secondary_verdicts={k: sec[k]['verdict'] for k in sec}, repro=repro, repro_ok=repro_ok, descriptive=desc)
(ROOT / 'analysis').mkdir(exist_ok=True); json.dump(out, open(ROOT / 'analysis' / 'b15_result.json', 'w'), indent=1)
print(f"Primaer kL2B_tr - selu_tr w8a4a d200 (fruehes Stoppen): med={prim['med_diff_pp']:+.2f}pp W/L={prim['wins']}/{prim['losses']} p={prim['p']:.4f} {prim['verdict']} | Erfolg: {success}")
print('Kontrolle B10:', repro_ok, repro, '| Ausfaelle:', out['failed'])
print('Sekundaer:', {k: (round(sec[k]['med_diff_pp'], 2), f"{sec[k]['wins']}/{sec[k]['losses']}", round(sec[k]['holm'], 4), sec[k]['verdict']) for k in sec})
for a, e in desc.items():
    print(a, 'Stopp-Punkte', e['es_steps'], '| am Stopp:', {k: round(x, 2) for k, x in e['es'].items()})
    print('   Verlauf fp32/w4a/Bereich:', {st: (round(c['test_err_fp32'], 1), round(c['test_err_w8a4a_best'], 1), c['range_max'] and round(c['range_max'], 2)) for st, c in e['curve'].items()})
